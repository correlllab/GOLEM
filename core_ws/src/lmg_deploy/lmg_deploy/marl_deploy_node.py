"""Run marl_deploy_node: the Locomanipulation_game two-agent MARL policy (legs + IK arms) on the H1-2.

Flow
----
1. Idle (state ``waiting_heartbeat``, nothing published) until golem_safety's
   first true /safety/heartbeat, then until the first /lowstate.
2. Pre-pose: ramp all 27 joint targets (cosine blend) from the measured pose
   to the policy's default pose at the training gains, then wait for the joints
   to settle there. The ramp lasts at least ``prepose_ramp_s`` and is stretched
   so that no target moves faster than ``prepose_max_vel``: a target that runs
   ahead of the joint makes the PD torque spike and trips the safety layer's
   torque e-stop.
3. With ``engage_wait_for_confirm`` (default true; lmg_real keeps it on,
   lmg_sim turns it off and engages automatically) hold the pre-pose until an
   operator calls ~/confirm_engage.
4. Engage: the controller starts an episode at the measured state with a zero
   velocity command (any /cmd_vel received before engaging is dropped) and runs
   at the policy's training rate (export.yaml's control_dt: 50 Hz, or 100 Hz
   for the sport-mode runs; ``control_hz`` only confirms it). With
   ``disable_elastic_band`` (the sim) the elastic band is released
   ``band_release_delay_s`` later and the episode restarts then, so the
   policy's first free step is its episode's first.

While engaged the robot navigates on /cmd_vel, the arms at the rest pose.
~/state (latched) names the current state.

The arm goals stay fixed in the world by the SLAM pipeline's odometry
(/Odometry, FAST-LIO's camera_init -> body), so golem_slam's navigation launch
must run beside this node; without it the goals stay fixed to the pelvis.

This node owns all 27 joints: legs on /safety/lowcmd_lower_in, torso and arms
on /safety/lowcmd_upper_in. frame_task_server and lowerbody_controller_node
must not run beside it (the lmg bringups start neither).

Safety
------
SAFETY: the node publishes nothing, and the pre-pose does not start, before
golem_safety's first true /safety/heartbeat (golem_safety beats once it
relays). After that, a false beat (golem_safety's estop latched) or
``heartbeat_timeout`` of silence latches this node silent for good.

SAFETY: a /lowstate older than ``lowstate_timeout`` (wall clock) skips that
tick's policy step and publish, since stepping on a frozen state would
integrate garbage.

SAFETY: a non-finite state, observation or target latches the node silent
instead of sending it (golem_safety would latch its estop, every motor limp, on
it). Silent means golem_safety keeps relaying the last targets until it is
stopped: its stale times (stale.*_seconds) only report a quiet input.

IMU
---
The policy reads the torso IMU as /lowstate reports it (export.yaml imu:
{source: lowstate, frame: imu_link, ...}; an export with another imu.source is
refused at start): LowState.imu_state's quaternion (wxyz) and its gyroscope
through the GyroLowpass the run trained with (``imu.gyro_lowpass_hz``, sampled
at every /lowstate by its tick).

IMU calibration: export.yaml's ``imu.orientation`` (imu_calibration.py) says
whether the run trained on the torso IMU's quaternion as reported (``raw``, or
absent: the runs with ``imu.quat_bias_deg``, the robot's bias drawn in
training) or with the robot's static offset removed (``calibrated``, only a
residual of ``imu.residual_deg`` drawn). The robot's calibration is its entry
in ``imu_calibration_file`` (default config/imu_calibration.yaml beside the
installed policies), under ``robot`` (default: the policy folder's name,
policies/<robot>/). The node hands it to MarlController, which applies it to a
calibrated policy only (corrected = reported * R_bias^-1), refuses to start a
calibrated policy without one, and ignores it, logging so, for a raw policy.
Once engaged, after about 5 s standing still on both loaded feet at zero
command, the node compares the policy's IMU tilt with the tilt forward
kinematics gives from the soles (calibrate_imu's) and warns once if they differ
by more than 1 deg (calibrated policies; raw ones log it).
"""

import os
import time

import numpy as np
import rclpy
import torch
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

from lmg_deploy.controller import (
    ARM_START,
    NUM_JOINTS,
    NUM_LEGS,
    JointCommand,
    MarlController,
    RobotState,
)
from lmg_deploy.imu_calibration import (
    FEET,
    IMU_FRAME,
    StandingTiltCheck,
    StillThresholds,
    fk_imu_gravity,
    load_calibration,
    still_and_loaded,
)
from lmg_deploy.kinematics import ArmKinematics
from lmg_deploy.gyro_lowpass import GyroLowpass
from lmg_deploy.networks import load_network
from lmg_deploy.rotations import projected_gravity, xyzw_to_wxyz

MOTOR_MODE_PR = 1          # LowCmd motor mode: enabled
SAFETY_HEARTBEAT_TOPIC = "/safety/heartbeat"
PREPOSE_TOL = 0.15          # rad; max |q - default| over all 27 joints to count as posed
PREPOSE_SETTLE_S = 0.2      # s of consecutive in-tolerance ticks
PREPOSE_MAX_S = 8.0         # s after the ramp; posed or not, stop waiting
# A tick this soon after the last command (a fraction of the control period) sends nothing: after an overrun rclpy
# runs the missed tick at once, which sent the next targets 2-5 ms after the late ones (the 11-23 bags' engage).
MIN_COMMAND_GAP = 0.5
TILT_CHECK_HOLD_S = 5.0     # s standing still on both feet before the IMU tilt is compared with the kinematic one
TILT_CHECK_TOL_DEG = 1.0    # deg; a larger difference warns once
SLAM_STALE_S = 0.5          # s; /Odometry older than this while reaching: warn that the goals ride with the pelvis


class MarlDeployNode(Node):
    """Gate, pre-pose and engage the MARL policy, and relay its joint targets to golem_safety's inputs."""

    def __init__(self):
        """Load the policy config and networks, then create the node's topics, services and control timer.

        Raises ValueError on a missing or inconsistent config or a bad parameter.
        """
        super().__init__("marl_deploy_node")
        self.declare_parameter("config", "")   # the policy's yaml, policies/<robot>/<robot>.yaml
        self.declare_parameter("urdf_path", "/home/code/CL_Assets/ros_assets/h1_2_magpie.urdf")
        self.declare_parameter("control_hz", 0.0)         # Hz; 0: the policy's, 1 / control_dt, else it must match
        self.declare_parameter("prepose_ramp_s", 6.0)     # s, minimum ramp duration
        self.declare_parameter("prepose_max_vel", 0.25)   # rad/s, peak target speed of any joint
        self.declare_parameter("engage_wait_for_confirm", True)
        self.declare_parameter("disable_elastic_band", True)   # call /elastic_band/toggle after engaging (sim)
        self.declare_parameter("band_release_delay_s", 1.0)   # s after engaging
        self.declare_parameter("heartbeat_timeout", 0.5)   # s, wall clock
        self.declare_parameter("lowstate_timeout", 0.1)    # s, wall clock
        # The robot's torso-IMU calibration (config/imu_calibration.yaml); "": defaults from the policy folder.
        # robot h12_sim is the H1-2's simulators' entry (no offset), which lmg_sim passes.
        self.declare_parameter("robot", "")                  # its entry; "": the policy folder's name (h12)
        self.declare_parameter("imu_calibration_file", "")   # "": config/imu_calibration.yaml beside policies/

        cfg_path = self.get_parameter("config").value
        if not cfg_path:
            raise ValueError("config:= the policy's deploy yaml, policies/<robot>/<robot>.yaml (lmg_deploy ships "
                             "policies/h12/h12.yaml; README 'Exporting a checkpoint' replaces it)")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
        # export_marl.py's per-checkpoint values (config/example.export.yaml) override the policy's yaml.
        export_path = os.path.join(cfg_dir, "export.yaml")
        if os.path.isfile(export_path):
            with open(export_path) as f:
                cfg.update(yaml.safe_load(f) or {})
            self.get_logger().info(f"export: {cfg.get('source')}")
        # arm_ik's goal table (arm_seed_table.npz), named relative to the export's folder
        if cfg.get("arm_seed_table"):
            cfg["arm_seed_table"] = os.path.join(cfg_dir, cfg["arm_seed_table"])
        control_hz = float(self.get_parameter("control_hz").value) or 1.0 / float(cfg["control_dt"])
        if abs(1.0 / control_hz - float(cfg["control_dt"])) > 1e-6:
            raise ValueError(f"control_hz {control_hz} doesn't match the policy's control_dt {cfg['control_dt']}")

        imu = cfg.get("imu") or {}
        # SAFETY: the torso IMU is the only IMU this node reads; an export trained on another would be misread
        if imu.get("source") != "lowstate" or imu.get("frame") != IMU_FRAME:
            raise ValueError(f"export.yaml imu {imu}: this lmg_deploy runs policies trained on the torso IMU as "
                             f"/lowstate reports it (imu.source lowstate, imu.frame {IMU_FRAME}); the /livox/imu "
                             "attitude path of older exports is no longer supported")
        torch.set_num_threads(1)
        kin = ArmKinematics(
            self.get_parameter("urdf_path").value,
            joint_names=cfg["joint_names"],
            ee_frames=cfg["arms"]["ee_frames"],
            arm_joint_names=[cfg["joint_names"][ARM_START:ARM_START + 7], cfg["joint_names"][ARM_START + 7:]],
            extra_frames=list(dict.fromkeys([cfg["slam_frame"], IMU_FRAME, *FEET])),
        )
        kin.update(np.zeros(NUM_JOINTS))
        # The gyro low-pass the run trained with, per /lowstate sample; absent: none.
        gyro_lowpass_hz = float(imu.get("gyro_lowpass_hz", 0.0))
        self._gyro_lowpass_hz = gyro_lowpass_hz
        self._lowstate_lowpass = GyroLowpass(gyro_lowpass_hz) if gyro_lowpass_hz > 0 else None
        self._lowstate_gyro: np.ndarray | None = None
        self._lowstate_tick: int | None = None
        # The exported networks sit next to the config.
        legs = load_network(os.path.join(cfg_dir, cfg["legs"]["policy"]),
                            cfg["legs"]["num_obs"], cfg["legs"]["num_actions"])   # MLP or LSTM
        arms = load_network(os.path.join(cfg_dir, cfg["arms"]["policy"]),
                            cfg["arms"]["num_obs"], cfg["arms"]["num_actions"])
        calibration = self._load_imu_calibration(cfg_dir)
        self.controller = MarlController(cfg, kin, legs, arms, imu_calibration=calibration)
        ignored = calibration is not None and self.controller.imu_calibration is None
        (self.get_logger().warn if ignored else self.get_logger().info)(self.controller.imu_calibration_note)
        self._kin = kin
        self._still = StillThresholds()
        self._tilt_check = StandingTiltCheck(float(cfg["control_dt"]), TILT_CHECK_HOLD_S, TILT_CHECK_TOL_DEG)

        self._lowstate: LowState | None = None
        self._lowstate_time: float | None = None   # monotonic receipt time
        self._slam_time: float | None = None       # monotonic receipt time of the last /Odometry
        self._confirm_wait = bool(self.get_parameter("engage_wait_for_confirm").value)
        self._confirmed = False
        self._control_hz = control_hz
        self._prepose_settle_ticks = max(1, int(round(PREPOSE_SETTLE_S * control_hz)))
        self._prepose_max_ticks = int(round(PREPOSE_MAX_S * control_hz))
        self._ramp_min_s = float(self.get_parameter("prepose_ramp_s").value)
        self._ramp_max_vel = float(self.get_parameter("prepose_max_vel").value)
        if self._ramp_max_vel <= 0.0:
            raise ValueError(f"prepose_max_vel must be positive, not {self._ramp_max_vel}")
        self._ramp_ticks = 1   # set from the measured start pose on the first pre-pose tick
        self._prepose_ticks = 0
        self._prepose_pass = 0
        self._prepose_start: np.ndarray | None = None
        self._last_sent: np.ndarray | None = None   # rad, the last 27 targets published (the engage blend starts there)
        self._last_publish: float | None = None     # monotonic time they went out
        self._engaged = False
        self._band_released = not bool(self.get_parameter("disable_elastic_band").value)
        self._band_release_at: float | None = None
        self._band_inflight = False
        self._lowstate_timeout = float(self.get_parameter("lowstate_timeout").value)
        self._hb_timeout = float(self.get_parameter("heartbeat_timeout").value)
        self._hb_last: float | None = None
        self._dead = False                  # latched: never publishes again
        self._state_name = ""

        lowstate_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                  history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, lowstate_qos)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Bool, SAFETY_HEARTBEAT_TOPIC, self._on_heartbeat, 10)
        self.create_subscription(Odometry, "/Odometry", self._on_odometry, 10)
        self._lower_pub = self.create_publisher(LowCmd, "/safety/lowcmd_lower_in", 10)
        self._upper_pub = self.create_publisher(LowCmd, "/safety/lowcmd_upper_in", 10)
        self._state_pub = self.create_publisher(String, "~/state", latched)
        self.create_service(Trigger, "~/confirm_engage", self._on_confirm)
        self._band_cli = self.create_client(Trigger, "/elastic_band/toggle")
        self.create_timer(1.0 / control_hz, self._tick)
        self._set_state("idle")
        self.get_logger().info(
            f"marl_deploy_node ready: {cfg_path}, {control_hz:.0f} Hz, "
            f"confirm={'required' if self._confirm_wait else 'auto'}, "
            f"IMU /lowstate imu_state {'calibrated' if self.controller.imu_calibration else 'as reported'}, "
            f"gyro low-pass {f'{self._gyro_lowpass_hz:g} Hz' if self._gyro_lowpass_hz > 0 else 'off'}")

    def _load_imu_calibration(self, cfg_dir: str) -> dict | None:
        """Return the robot's torso-IMU calibration from imu_calibration_file (None: unset, or no file)."""
        robot = self.get_parameter("robot").value or os.path.basename(cfg_dir)
        path = self.get_parameter("imu_calibration_file").value or os.path.join(
            cfg_dir, "..", "..", "config", "imu_calibration.yaml")
        if not os.path.isfile(path):
            self.get_logger().warn(f"no IMU calibration file {path}: robot {robot!r} has no IMU calibration")
            return None
        calibration = load_calibration(path, robot)
        self.get_logger().info(f"IMU calibration of {robot!r} ({os.path.normpath(path)}): "
                               + ("unset" if calibration is None else
                                  "roll %+.3f pitch %+.3f yaw %+.3f deg, rotate_gyro %s" % (
                                      calibration["roll_deg"], calibration["pitch_deg"], calibration["yaw_deg"],
                                      calibration["rotate_gyro"])))
        return calibration

    def _on_lowstate(self, msg: LowState) -> None:
        """Keep the latest /lowstate and its receipt time, and filter its gyroscope."""
        self._lowstate = msg
        self._lowstate_time = time.monotonic()
        gyro = np.array(msg.imu_state.gyroscope, dtype=np.float64)
        if self._lowstate_lowpass is not None:
            # the sample's dt from LowState.tick (ms); a first sample or a gap restarts the filter
            dt = 0.0 if self._lowstate_tick is None else 1e-3 * ((msg.tick - self._lowstate_tick) & 0xFFFFFFFF)
            gyro = self._lowstate_lowpass(gyro, dt)
        self._lowstate_tick = msg.tick
        self._lowstate_gyro = gyro

    def _on_odometry(self, msg: Odometry) -> None:
        """Feed a FAST-LIO pose to the controller's odometry."""
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        quat = xyzw_to_wxyz(np.array([o.x, o.y, o.z, o.w]))
        self.controller.set_slam_pose(np.array([p.x, p.y, p.z]), quat)
        self._slam_time = time.monotonic()

    def _on_cmd_vel(self, msg: Twist) -> None:
        """Set the velocity command from /cmd_vel."""
        self.controller.set_velocity_command(np.array([msg.linear.x, msg.linear.y, msg.angular.z]))

    def _on_heartbeat(self, msg: Bool) -> None:
        """Latch silent on a false beat; otherwise note the beat's time."""
        if self._dead:
            return
        if not msg.data:
            self._latch("safety layer reports estop latched (heartbeat False)")
            return
        self._hb_last = time.monotonic()

    def _on_confirm(self, _req, resp):
        """Serve ~/confirm_engage: allow the engage once the pre-pose settles."""
        if not self._engaged:
            # A /cmd_vel that arrived before the go-ahead is not a command to walk on.
            self.controller.set_velocity_command(np.zeros(3))
        self._confirmed = True
        resp.success = True
        resp.message = "already engaged" if self._engaged else "confirmed: engaging once the pre-pose settles"
        self.get_logger().info(f"operator confirm: {resp.message}")
        return resp

    def _latch(self, reason: str) -> None:
        """Stop publishing for good and report reason on ~/state and the log."""
        # SAFETY: one way until the node restarts; golem_safety keeps relaying the last targets.
        self._dead = True
        self._set_state(f"fault: {reason}")
        self.get_logger().error(
            f"LATCHED SILENT: {reason}. No more commands on /safety/lowcmd_*_in; "
            "restart this node once the cause is cleared.")

    def _set_state(self, name: str) -> None:
        """Publish name on ~/state when it changes."""
        if name != self._state_name:
            self._state_name = name
            self._state_pub.publish(String(data=name))

    def _robot_state(self, msg: LowState) -> RobotState:
        """Return the controller's state from /lowstate: the torso IMU as reported (the controller calibrates it), its
        gyroscope low-passed."""
        motors = msg.motor_state
        q = np.array([motors[i].q for i in range(NUM_JOINTS)], dtype=np.float64)
        dq = np.array([motors[i].dq for i in range(NUM_JOINTS)], dtype=np.float64)
        imu = msg.imu_state
        gyro = self._lowstate_gyro if self._lowstate_gyro is not None else np.array(imu.gyroscope, dtype=np.float64)
        return RobotState(q=q, dq=dq, quat=np.array(imu.quaternion, dtype=np.float64), gyro=gyro.copy(),
                          acc=np.array(imu.accelerometer, dtype=np.float64))

    def _publish(self, cmd: JointCommand, mode_machine: int) -> None:
        """Publish the legs' rows and the torso and arms' rows on golem_safety's two inputs.

        SAFETY: a non-finite command latches the node silent instead of being sent.
        """
        if not (np.isfinite(cmd.q).all() and np.isfinite(cmd.kp).all() and np.isfinite(cmd.kd).all()):
            self._latch(f"non-finite joint command {cmd.q.round(3).tolist()}")
            return
        for pub, rows in ((self._lower_pub, range(0, NUM_LEGS)), (self._upper_pub, range(NUM_LEGS, NUM_JOINTS))):
            msg = LowCmd()
            msg.mode_pr = 0
            msg.mode_machine = mode_machine   # echoes /lowstate's robot identity
            for i in rows:
                m = msg.motor_cmd[i]
                m.mode = MOTOR_MODE_PR
                m.q = float(cmd.q[i])
                m.dq = 0.0
                m.tau = 0.0
                m.kp = float(cmd.kp[i])
                m.kd = float(cmd.kd[i])
            pub.publish(msg)
        self._last_sent = cmd.q.copy()
        self._last_publish = time.monotonic()

    def _release_band(self) -> None:
        """Call /elastic_band/toggle once it is available, one call at a time."""
        if self._band_released or self._band_inflight:
            return
        if not self._band_cli.service_is_ready():
            self.get_logger().warn("waiting for /elastic_band/toggle", throttle_duration_sec=5.0)
            return
        self._band_inflight = True
        self._band_cli.call_async(Trigger.Request()).add_done_callback(self._on_band_toggled)

    def _on_band_toggled(self, fut) -> None:
        """Mark the band released and restart the episode; retry on a failed call."""
        self._band_inflight = False
        result = fut.result()
        if result is None:
            self.get_logger().warn("elastic band toggle failed; retrying")
            return
        self._band_released = True
        self.get_logger().info(f"elastic band released ({result.message}); episode restarts")
        if self._lowstate is not None:
            self.controller.reset(self._robot_state(self._lowstate))

    def _tick(self) -> None:
        """Run one control tick: the heartbeat and freshness gates, then the pre-pose or a policy step."""
        if self._dead:
            return
        now = time.monotonic()
        if self._hb_last is not None and now - self._hb_last > self._hb_timeout:
            self._latch(f"no {SAFETY_HEARTBEAT_TOPIC} for {self._hb_timeout:.1f} s (safety node down?)")
            return
        if self._hb_last is None:
            self._set_state("waiting_heartbeat")
            self.get_logger().info(
                f"no {SAFETY_HEARTBEAT_TOPIC} yet (golem_safety beats once it can relay); sending nothing",
                throttle_duration_sec=5.0)
            return
        if self._lowstate is None:
            return
        if now - self._lowstate_time > self._lowstate_timeout:
            self.get_logger().warn(
                f"/lowstate is {now - self._lowstate_time:.2f} s old; not stepping", throttle_duration_sec=1.0)
            return
        msg = self._lowstate
        state = self._robot_state(msg)
        if not (np.isfinite(state.q).all() and np.isfinite(state.quat).all()):
            self._latch("non-finite /lowstate")
            return
        if self._last_publish is not None and now - self._last_publish < MIN_COMMAND_GAP / self._control_hz:
            # rclpy's catch-up for a tick an overrun missed: the next tick on the timer's grid steps instead
            self.get_logger().warn(f"tick {1e3 * (now - self._last_publish):.1f} ms after the last command "
                                   "(the previous tick overran); skipping it", throttle_duration_sec=1.0)
            return

        if not self._engaged:
            self._prepose(state, msg.mode_machine)
            return

        if self._band_release_at is not None and not self._band_released and now >= self._band_release_at:
            self._release_band()

        if self.controller.mode.value == "arm_goal" and (self._slam_time is None or now - self._slam_time > SLAM_STALE_S):
            self.get_logger().warn("no fresh /Odometry: the arm goals move with the pelvis (is golem_slam running?)",
                                   throttle_duration_sec=5.0)
        cmd = self.controller.step(state)
        for name, obs in self.controller.last_obs.items():
            if not np.isfinite(obs).all():
                self._latch(f"non-finite {name} observation")
                return
        self._publish(cmd, msg.mode_machine)
        self._set_state(self.controller.mode.value)
        self._check_tilt(msg, state)

    def _check_tilt(self, msg: LowState, state: RobotState) -> None:
        """Once, after TILT_CHECK_HOLD_S standing still on both loaded feet at zero command, compare the policy's IMU
        tilt with the kinematic one from the soles; warn if a calibrated policy's differ by over TILT_CHECK_TOL_DEG."""
        if self._tilt_check is None or self._tilt_check.done:
            return
        ctrl = self.controller
        motors = msg.motor_state
        tau = np.array([motors[i].tau_est for i in range(NUM_JOINTS)], dtype=np.float64)
        standing = (ctrl.mode.value == "nav" and not np.any(ctrl.velocity_command())
                    and float(np.linalg.norm(ctrl.observed_command())) < ctrl.zero_cmd_eps
                    and bool(still_and_loaded(state.dq, np.array(msg.imu_state.gyroscope, dtype=np.float64), tau,
                                              self._still)))
        # the kinematics is at this state's q: step() updated it
        g_fk = fk_imu_gravity(self._kin).mean(0) if standing else np.zeros(3)
        report = self._tilt_check.update(standing, projected_gravity(ctrl.corrected_state(state).quat), g_fk)
        if report is None:
            return
        line = ("IMU tilt as the policy sees it roll %+.2f pitch %+.2f deg, by kinematics from the soles roll %+.2f "
                "pitch %+.2f (difference %+.2f / %+.2f) after %.0f s standing"
                % (*report["imu_tilt"], *report["fk_tilt"], *report["diff"], TILT_CHECK_HOLD_S))
        if ctrl.imu_calibration is None:
            self.get_logger().info(line + " (raw policy: the robot's IMU offset is expected here)")
        elif report["mismatch"]:
            self.get_logger().warn(line + f": more than {TILT_CHECK_TOL_DEG:g} deg. Is config/imu_calibration.yaml "
                                   "this robot's, and are both feet flat on level ground? (README 'IMU calibration')")
        else:
            self.get_logger().info(line)

    def _prepose(self, state: RobotState, mode_machine: int) -> None:
        """Ramp to and hold the default pose, then engage once settled and confirmed."""
        ctrl = self.controller
        if self._prepose_start is None:
            self._prepose_start = state.q.copy()
            # The cosine blend's peak speed is (pi / 2) * distance / duration.
            dist = float(np.max(np.abs(ctrl.default - self._prepose_start)))
            ramp_s = max(self._ramp_min_s, 0.5 * np.pi * dist / self._ramp_max_vel)
            self._ramp_ticks = max(1, int(round(ramp_s * self._control_hz)))
            self.get_logger().info(
                f"pre-posing: max joint distance {dist:.2f} rad over {ramp_s:.1f} s "
                f"(peak {0.5 * np.pi * dist / ramp_s:.2f} rad/s)")
            self._set_state("preposing")
        target = ctrl.default.copy()
        if self._prepose_ticks < self._ramp_ticks:
            frac = self._prepose_ticks / self._ramp_ticks
            alpha = 0.5 * (1.0 - np.cos(np.pi * frac))
            target = (1.0 - alpha) * self._prepose_start + alpha * target
        self._publish(JointCommand(q=target, kp=ctrl.kp.copy(), kd=ctrl.kd.copy()), mode_machine)
        self._prepose_ticks += 1
        err = float(np.max(np.abs(state.q - ctrl.default)))
        self._prepose_pass = self._prepose_pass + 1 if err < PREPOSE_TOL else 0
        settled = self._prepose_pass >= self._prepose_settle_ticks
        timed_out = self._prepose_ticks >= self._ramp_ticks + self._prepose_max_ticks
        if not settled and not timed_out:
            return
        if self._confirm_wait and not self._confirmed:
            self._set_state("waiting_confirm")
            self.get_logger().info(
                f"pre-posed ({'settled' if settled else 'NOT SETTLED'}, max err {err:.3f} rad); holding. "
                "Engage with: ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger",
                throttle_duration_sec=5.0)
            return
        self.get_logger().info(f"pre-posed ({'settled' if settled else 'timeout'}, max err {err:.3f} rad): engaging")
        ctrl.reset(state, held_target=self._last_sent)
        # A /cmd_vel that arrived before engaging is not a command to walk now.
        ctrl.set_velocity_command(np.zeros(3))
        self._engaged = True
        if not self._band_released:
            self._band_release_at = time.monotonic() + float(self.get_parameter("band_release_delay_s").value)


def main():
    """Spin marl_deploy_node until interrupted."""
    rclpy.init()
    node = MarlDeployNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
