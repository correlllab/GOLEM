"""LocoMotionGameDeploy node: the two-agent MARL policy (legs + IK arms) on the H1-2.

Flow
----
1. Idle until the first /lowstate.
2. Pre-pose: ramp all 27 joint targets (cosine blend) from the measured pose
   to the policy's default pose at the training gains, then wait for the joints
   to settle there. The ramp lasts at least ``prepose_ramp_s`` and is stretched
   so that no target moves faster than ``prepose_max_vel``: a target that runs
   ahead of the joint makes the PD torque spike and trips the safety layer's
   torque e-stop.
3. With ``engage_wait_for_confirm`` (default true; lmg_real_bringup keeps it
   on, lmg_sim_bringup turns it off and engages automatically) hold the
   pre-pose until an operator calls ~/confirm_engage.
4. Engage: the controller starts an episode at the measured state with a zero
   velocity command (any /cmd_vel received before engaging is dropped) and
   runs at ``control_hz`` (50 Hz, the training policy rate). In the sim the elastic
   band is released ``band_release_delay_s`` later and the episode restarts
   then, so the policy's first free step is its episode's first.

While engaged the robot starts in walking mode and navigates on /cmd_vel. The
mode is switched by hand with ~/set_mode (custom_ros_messages/LmgSetMode: "walk" or
"arm"); ~/set_arm_goals (custom_ros_messages/LmgSetArmGoals) sets the left and right
wrist_yaw_link goals, pelvis frame: in arm mode at once, while walking for the
next switch to arm mode (until some are set, arm mode holds the rest pose).
A PoseArray on ~/arm_goal (left then right) sets the goals and switches to arm
mode in one message, ~/end_arm_goal back to walking. /cmd_vel is ignored in arm
mode: the policy only ever saw zero there.

The arm goals stay fixed in the world by the SLAM pipeline's odometry
(/Odometry, FAST-LIO's camera_init -> body), so h12_slam's navigation launch
must run beside this node; without it the goals stay fixed to the pelvis.

This node owns all 27 joints: legs on /safety/lowcmd_lower_in, torso and arms
on /safety/lowcmd_upper_in. frame_task_server and lowerbody_controller_node
must not run beside it (the lmg bringups start neither).

Safety
------
* /safety/heartbeat watchdog, as in h12_lowerbody_rl: armed on the first beat;
  a False beat (the safety layer's estop latched) or ``heartbeat_timeout`` of
  silence latches this node silent.
* /lowstate older than ``lowstate_timeout`` (wall clock): no policy step and no publish
  that tick (stepping on a frozen state would integrate garbage). The same holds
  for /livox/imu (``livox_imu_timeout``), and the policy doesn't engage until its
  attitude estimate is up.
* A non-finite observation or target latches the node silent instead of
  sending it: the safety layer would latch its estop (all motors limp) on it.
Silent means the safety layer keeps holding the last targets until it is
stopped; it has no staleness watchdog on these channels.

IMU
---
The policy's IMU observations (angular velocity, specific force, projected gravity)
come from the MID360's /livox/imu, never LowState.imu_state (its attitude carries a
~2.5 deg calibration error). They are carried from the chip's frame (URDF
livox_imu_link) into the frame the policy trained with, export.yaml's ``imu.frame``:
imu_link on torso_link for runs up to Explicit500 (the default when the export has no
``imu`` block), livox_link or livox_imu_link from then on. The rotation is the URDF
mount; attitude is livox_imu.LivoxImu's filter. The angular velocity passes the
livox_imu.GyroLowpass the run trained with (``imu.gyro_lowpass_hz``; absent = none),
which suppresses the head's ~20 Hz pitch resonance while walking.

``imu_offset_{roll,pitch,yaw}_deg`` (default 0) trim that IMU, live (``ros2 param set``
or rqt_reconfigure): the attitude it reads with the torso level, as in h12_lowerbody_rl.
The trim rotates the attitude, the angular velocity and the specific force alike, so
projected gravity and the accelerometer agree on the tilt.

Interfaces
----------
sub  /lowstate                       (unitree_hg/LowState)
sub  /livox/imu                      (sensor_msgs/Imu)  livox_imu_link frame (livox_imu_upright)
sub  /cmd_vel                        (geometry_msgs/Twist)
sub  /safety/heartbeat               (std_msgs/Bool)
sub  /Odometry                       (nav_msgs/Odometry)  FAST-LIO body pose
sub  ~/arm_goal                      (geometry_msgs/PoseArray)  2 poses, pelvis frame
pub  /safety/lowcmd_lower_in         (unitree_hg/LowCmd)  rows 0-11
pub  /safety/lowcmd_upper_in         (unitree_hg/LowCmd)  rows 12-26
pub  ~/state                         (std_msgs/String, latched)
pub  ~/wrist_error                   (std_msgs/Float32MultiArray)  m, per arm, while engaged
srv  ~/confirm_engage                (std_srvs/Trigger)
srv  ~/set_mode                      (custom_ros_messages/LmgSetMode)      "walk" | "arm"
srv  ~/set_arm_goals                 (custom_ros_messages/LmgSetArmGoals)  left, right Pose, pelvis frame
srv  ~/end_arm_goal                  (std_srvs/Trigger)            = set_mode walk
"""

import os
import time

import numpy as np
import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32MultiArray, String
from std_srvs.srv import Trigger
from custom_ros_messages.srv import LmgSetArmGoals as SetArmGoals, LmgSetMode as SetMode
from unitree_hg.msg import LowCmd, LowState

from lmg_deploy.controller import (
    ARM_START,
    NUM_JOINTS,
    NUM_LEGS,
    JointCommand,
    MarlController,
    RobotState,
)
from lmg_deploy.kinematics import ArmKinematics
from lmg_deploy.livox_imu import LIVOX_IMU_FRAME, LivoxImu, livox_to_imu_rotation
from lmg_deploy.networks import load_network
from lmg_deploy.rotations import imu_mount_quat, quat_apply, quat_conjugate, quat_mul, xyzw_to_wxyz

MOTOR_MODE_PR = 1
SAFETY_HEARTBEAT_TOPIC = "/safety/heartbeat"
PREPOSE_TOL = 0.15          # rad; max |q - default| over all 27 joints to count as posed
PREPOSE_SETTLE_TICKS = 10   # consecutive in-tolerance ticks (0.2 s at 50 Hz)
PREPOSE_MAX_TICKS = 400     # after the ramp; posed or not, stop waiting (8 s at 50 Hz)
DEFAULT_IMU_FRAME = "imu_link"   # exports without an imu block (Explicit500) trained on the torso IMU
IMU_OFFSET_PARAMS = ("imu_offset_roll_deg", "imu_offset_pitch_deg", "imu_offset_yaw_deg")
IMU_OFFSET_RANGE_DEG = 15.0
SLAM_STALE_S = 0.5          # /Odometry older than this while reaching: warn that the goals ride with the pelvis


def _share(*parts: str) -> str:
    return os.path.join(get_package_share_directory("lmg_deploy"), *parts)


class MarlDeployNode(Node):
    def __init__(self):
        super().__init__("marl_deploy_node")
        self.declare_parameter("config", _share("policies", "explicit500", "explicit500.yaml"))
        self.declare_parameter("urdf_path", "/home/code/CL_Assets/ros_assets/h1_2_magpie.urdf")
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("prepose_ramp_s", 6.0)     # s, minimum ramp duration
        self.declare_parameter("prepose_max_vel", 0.25)   # rad/s, peak target speed of any joint
        self.declare_parameter("engage_wait_for_confirm", True)
        self.declare_parameter("disable_elastic_band", True)
        self.declare_parameter("band_release_delay_s", 1.0)
        self.declare_parameter("heartbeat_timeout", 0.5)
        self.declare_parameter("lowstate_timeout", 0.1)
        self.declare_parameter("livox_imu_timeout", 0.1)   # s, wall clock
        self.declare_parameter("livox_gravity_tau_s", 0.5)  # s, attitude filter's pull toward the accelerometer
        for name in IMU_OFFSET_PARAMS:
            self.declare_parameter(name, 0.0, ParameterDescriptor(
                description="live IMU trim, deg: the IMU's reading with the torso level",
                floating_point_range=[FloatingPointRange(
                    from_value=-IMU_OFFSET_RANGE_DEG, to_value=IMU_OFFSET_RANGE_DEG, step=0.0)]))

        cfg_path = self.get_parameter("config").value
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
        # export_marl.py's per-checkpoint values (rest pose, source files) override the template
        export_path = os.path.join(cfg_dir, "export.yaml")
        if os.path.isfile(export_path):
            with open(export_path) as f:
                cfg.update(yaml.safe_load(f) or {})
            self.get_logger().info(f"export: {cfg.get('source')}")
        control_hz = float(self.get_parameter("control_hz").value)
        if abs(1.0 / control_hz - float(cfg["control_dt"])) > 1e-6:
            raise ValueError(f"control_hz {control_hz} doesn't match the policy's control_dt {cfg['control_dt']}")

        imu_frame = (cfg.get("imu") or {}).get("frame", DEFAULT_IMU_FRAME)
        torch.set_num_threads(1)
        kin = ArmKinematics(
            self.get_parameter("urdf_path").value,
            joint_names=cfg["joint_names"],
            ee_frames=cfg["arms"]["ee_frames"],
            arm_joint_names=[cfg["joint_names"][ARM_START:ARM_START + 7], cfg["joint_names"][ARM_START + 7:]],
            extra_frames=list(dict.fromkeys([cfg["slam_frame"], LIVOX_IMU_FRAME, imu_frame])),
        )
        kin.update(np.zeros(NUM_JOINTS))
        r_imu_livox = livox_to_imu_rotation(kin, imu_frame)
        self._imu_frame = imu_frame
        # the gyro low-pass the run trained with; exports without one (Explicit500) saw the raw rate
        gyro_lowpass_hz = float((cfg.get("imu") or {}).get("gyro_lowpass_hz", 0.0))
        self._livox = LivoxImu(r_imu_livox, tau_s=float(self.get_parameter("livox_gravity_tau_s").value),
                               gyro_lowpass_hz=gyro_lowpass_hz)
        self._gyro_lowpass_hz = gyro_lowpass_hz
        self._livox_time: float | None = None   # monotonic receipt time of the last /livox/imu
        # the exported networks, next to the config
        legs = load_network(os.path.join(cfg_dir, cfg["legs"]["policy"]),
                            cfg["legs"]["num_obs"], cfg["legs"]["num_actions"])   # MLP or LSTM
        arms = load_network(os.path.join(cfg_dir, cfg["arms"]["policy"]),
                            cfg["arms"]["num_obs"], cfg["arms"]["num_actions"])
        self.controller = MarlController(cfg, kin, legs, arms)

        self._lowstate: LowState | None = None
        self._lowstate_time: float | None = None   # monotonic receipt time
        self._slam_time: float | None = None       # monotonic receipt time of the last /Odometry
        self._confirm_wait = bool(self.get_parameter("engage_wait_for_confirm").value)
        self._confirmed = False
        self._control_hz = control_hz
        self._ramp_min_s = float(self.get_parameter("prepose_ramp_s").value)
        self._ramp_max_vel = float(self.get_parameter("prepose_max_vel").value)
        if self._ramp_max_vel <= 0.0:
            raise ValueError(f"prepose_max_vel must be positive, not {self._ramp_max_vel}")
        self._ramp_ticks = 1   # set from the measured start pose on the first pre-pose tick
        self._prepose_ticks = 0
        self._prepose_pass = 0
        self._prepose_start: np.ndarray | None = None
        self._engaged = False
        self._band_released = not bool(self.get_parameter("disable_elastic_band").value)
        self._band_release_at: float | None = None
        self._band_inflight = False
        self._lowstate_timeout = float(self.get_parameter("lowstate_timeout").value)
        self._hb_timeout = float(self.get_parameter("heartbeat_timeout").value)
        self._hb_last: float | None = None
        self._dead = False                  # latched: never publishes again
        self._state_name = ""
        self._imu_mount = self._build_imu_mount({n: float(self.get_parameter(n).value) for n in IMU_OFFSET_PARAMS})
        self.add_on_set_parameters_callback(self._on_set_parameters)

        lowstate_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                  history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, lowstate_qos)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Imu, "/livox/imu", self._on_livox_imu, qos_profile_sensor_data)
        self.create_subscription(Bool, SAFETY_HEARTBEAT_TOPIC, self._on_heartbeat, 10)
        self.create_subscription(Odometry, "/Odometry", self._on_odometry, 10)
        self.create_subscription(PoseArray, "~/arm_goal", self._on_arm_goal, 10)
        self._lower_pub = self.create_publisher(LowCmd, "/safety/lowcmd_lower_in", 10)
        self._upper_pub = self.create_publisher(LowCmd, "/safety/lowcmd_upper_in", 10)
        self._state_pub = self.create_publisher(String, "~/state", latched)
        self._err_pub = self.create_publisher(Float32MultiArray, "~/wrist_error", 10)
        self.create_service(Trigger, "~/confirm_engage", self._on_confirm)
        self.create_service(Trigger, "~/end_arm_goal", self._on_end_arm_goal)
        self.create_service(SetMode, "~/set_mode", self._on_set_mode)
        self.create_service(SetArmGoals, "~/set_arm_goals", self._on_set_arm_goals)
        self._band_cli = self.create_client(Trigger, "/elastic_band/toggle")
        self.create_timer(1.0 / control_hz, self._tick)
        self._set_state("idle")
        self.get_logger().info(
            f"marl_deploy_node ready: {cfg_path}, {control_hz:.0f} Hz, "
            f"confirm={'required' if self._confirm_wait else 'auto'}, IMU /livox/imu in {self._imu_frame}, "
            f"gyro low-pass {f'{self._gyro_lowpass_hz:g} Hz' if self._gyro_lowpass_hz > 0 else 'off'}")

    # -- callbacks -------------------------------------------------------------------
    def _build_imu_mount(self, deg: dict) -> np.ndarray | None:
        self.get_logger().info("IMU trim: roll %+.3f pitch %+.3f yaw %+.3f deg" % tuple(deg[n] for n in IMU_OFFSET_PARAMS))
        if not any(deg.values()):
            return None
        return imu_mount_quat(*(np.radians(deg[n]) for n in IMU_OFFSET_PARAMS))

    def _on_set_parameters(self, params) -> SetParametersResult:
        # same single-threaded executor as _tick: swapping the trim here can't race a step
        touched = [p for p in params if p.name in IMU_OFFSET_PARAMS]
        if touched:
            if any(not np.isfinite(p.value) for p in touched):
                return SetParametersResult(successful=False, reason="IMU trim must be finite")
            # the new values aren't stored until this returns
            deg = {n: float(self.get_parameter(n).value) for n in IMU_OFFSET_PARAMS}
            deg.update({p.name: float(p.value) for p in touched})
            self._imu_mount = self._build_imu_mount(deg)
        return SetParametersResult(successful=True)

    def _on_lowstate(self, msg: LowState) -> None:
        self._lowstate = msg
        self._lowstate_time = time.monotonic()

    def _on_livox_imu(self, msg: Imu) -> None:
        w, a = msg.angular_velocity, msg.linear_acceleration
        ready = self._livox.ready
        self._livox.update(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec,
                           np.array([w.x, w.y, w.z]), np.array([a.x, a.y, a.z]))
        self._livox_time = time.monotonic()
        if self._livox.ready and not ready:
            self.get_logger().info(
                f"/livox/imu attitude up ({'g' if self._livox.scale != 1.0 else 'm/s^2'} input), "
                f"projected gravity {(-self._livox.up).round(3).tolist()}")

    def _on_odometry(self, msg: Odometry) -> None:
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        self.controller.set_slam_pose(np.array([p.x, p.y, p.z]), xyzw_to_wxyz(np.array([o.x, o.y, o.z, o.w])))
        self._slam_time = time.monotonic()

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.controller.set_velocity_command(np.array([msg.linear.x, msg.linear.y, msg.angular.z]))

    def _poses(self, poses_msg) -> np.ndarray | str:
        """(arms, 7) wxyz poses from geometry_msgs Poses (left, right), or why they can't be a goal."""
        n = self.controller.num_arms
        if len(poses_msg) != n:
            return f"arm goal needs {n} poses (left, right), got {len(poses_msg)}"
        poses = np.array([[p.position.x, p.position.y, p.position.z,
                           *xyzw_to_wxyz(np.array([p.orientation.x, p.orientation.y,
                                                   p.orientation.z, p.orientation.w]))]
                          for p in poses_msg])
        if not np.isfinite(poses).all() or np.any(np.linalg.norm(poses[:, 3:], axis=1) < 1e-6):
            return "arm goal has a non-finite position or a zero quaternion"
        return poses

    def _on_arm_goal(self, msg: PoseArray) -> None:
        poses = self._poses(msg.poses)
        if isinstance(poses, str):
            self.get_logger().error(f"{poses}; ignored")
            return
        if not self._engaged:
            self.get_logger().warn("arm goal before the policy is engaged; ignored")
            return
        self.controller.request_arm_goal(poses)
        self.get_logger().info(f"arm goal: left {poses[0, :3].round(3)}, right {poses[1, :3].round(3)} (pelvis)")

    def _on_heartbeat(self, msg: Bool) -> None:
        if self._dead:
            return
        if not msg.data:
            self._latch("safety layer reports estop latched (heartbeat False)")
            return
        self._hb_last = time.monotonic()

    def _on_confirm(self, _req, resp):
        if not self._engaged:
            # a /cmd_vel that arrived before the go-ahead is not a command to walk
            # on (arm goals are already ignored until engaged)
            self.controller.set_velocity_command(np.zeros(3))
        self._confirmed = True
        resp.success = True
        resp.message = "already engaged" if self._engaged else "confirmed: engaging once the pre-pose settles"
        self.get_logger().info(f"operator confirm: {resp.message}")
        return resp

    def _on_end_arm_goal(self, _req, resp):
        self.controller.end_arm_goal()
        resp.success = True
        resp.message = f"mode {self.controller.mode.value}"
        return resp

    def _on_set_mode(self, req, resp):
        mode = req.mode.strip().lower()
        if mode not in (SetMode.Request.WALK, SetMode.Request.ARM):
            resp.success, resp.message = False, f"mode must be '{SetMode.Request.WALK}' or '{SetMode.Request.ARM}', not {req.mode!r}"
        elif not self._engaged:
            resp.success, resp.message = False, "not engaged yet: the policy starts in walking mode once it engages"
        else:
            self.controller.set_mode(mode)
            resp.success, resp.message = True, f"mode {self.controller.mode.value}"
            self.get_logger().info(f"set_mode {mode}: {resp.message}")
        return resp

    def _on_set_arm_goals(self, req, resp):
        poses = self._poses([req.left, req.right])
        if isinstance(poses, str):
            resp.success, resp.message = False, poses
            return resp
        self.controller.set_arm_goals(poses)
        applied = self.controller.mode.value == "arm_goal"
        resp.success = True
        resp.message = (f"left {poses[0, :3].round(3).tolist()}, right {poses[1, :3].round(3).tolist()} (pelvis): "
                        + ("applied" if applied else "kept for the next set_mode arm"))
        self.get_logger().info(f"set_arm_goals: {resp.message}")
        return resp

    # -- helpers -------------------------------------------------------------------------
    def _latch(self, reason: str) -> None:
        self._dead = True
        self._set_state(f"fault: {reason}")
        self.get_logger().error(
            f"LATCHED SILENT: {reason}. No more commands on /safety/lowcmd_*_in; "
            "restart this node once the cause is cleared.")

    def _set_state(self, name: str) -> None:
        if name != self._state_name:
            self._state_name = name
            self._state_pub.publish(String(data=name))

    def _robot_state(self, msg: LowState) -> RobotState:
        motors = msg.motor_state
        if self._livox.ready:
            quat, gyro, acc = self._livox.quat(), self._livox.gyro.copy(), self._livox.acc.copy()
        else:
            # only pre-pose runs before the attitude is up, and it reads q alone
            quat, gyro, acc = np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)
        if self._imu_mount is not None:
            quat = quat_mul(quat, quat_conjugate(self._imu_mount))
            gyro, acc = quat_apply(self._imu_mount, gyro), quat_apply(self._imu_mount, acc)
        return RobotState(
            q=np.array([motors[i].q for i in range(NUM_JOINTS)], dtype=np.float64),
            dq=np.array([motors[i].dq for i in range(NUM_JOINTS)], dtype=np.float64),
            tau=np.array([motors[i].tau_est for i in range(NUM_JOINTS)], dtype=np.float64),
            quat=quat, gyro=gyro, acc=acc,
        )

    def _livox_fresh(self, now: float) -> bool:
        return (self._livox.ready and self._livox_time is not None
                and now - self._livox_time <= float(self.get_parameter("livox_imu_timeout").value))

    def _publish(self, cmd: JointCommand, mode_machine: int) -> None:
        if not (np.isfinite(cmd.q).all() and np.isfinite(cmd.kp).all() and np.isfinite(cmd.kd).all()):
            self._latch(f"non-finite joint command {cmd.q.round(3).tolist()}")
            return
        for pub, rows in ((self._lower_pub, range(0, NUM_LEGS)), (self._upper_pub, range(NUM_LEGS, NUM_JOINTS))):
            msg = LowCmd()
            msg.mode_pr = 0
            msg.mode_machine = mode_machine   # the safety layer copies the header of whichever channel spoke last
            for i in rows:
                m = msg.motor_cmd[i]
                m.mode = MOTOR_MODE_PR
                m.q = float(cmd.q[i])
                m.dq = 0.0
                m.tau = 0.0
                m.kp = float(cmd.kp[i])
                m.kd = float(cmd.kd[i])
            pub.publish(msg)

    def _release_band(self) -> None:
        if self._band_released or self._band_inflight:
            return
        if not self._band_cli.service_is_ready():
            self.get_logger().warn("waiting for /elastic_band/toggle", throttle_duration_sec=5.0)
            return
        self._band_inflight = True
        self._band_cli.call_async(Trigger.Request()).add_done_callback(self._on_band_toggled)

    def _on_band_toggled(self, fut) -> None:
        self._band_inflight = False
        result = fut.result()
        if result is None:
            self.get_logger().warn("elastic band toggle failed; retrying")
            return
        self._band_released = True
        self.get_logger().info(f"elastic band released ({result.message}); episode restarts")
        if self._lowstate is not None:
            self.controller.reset(self._robot_state(self._lowstate))

    # -- main loop ---------------------------------------------------------------------------
    def _tick(self) -> None:
        if self._dead:
            return
        now = time.monotonic()
        if self._hb_last is not None and now - self._hb_last > self._hb_timeout:
            self._latch(f"no {SAFETY_HEARTBEAT_TOPIC} for {self._hb_timeout:.1f} s (safety node down?)")
            return
        if self._lowstate is None:
            return
        if now - self._lowstate_time > self._lowstate_timeout:
            self.get_logger().warn(
                f"/lowstate is {now - self._lowstate_time:.2f} s old; not stepping", throttle_duration_sec=1.0)
            return
        msg = self._lowstate
        if self._engaged and not self._livox_fresh(now):
            self.get_logger().warn("/livox/imu stale; not stepping", throttle_duration_sec=1.0)
            return
        state = self._robot_state(msg)
        if not (np.isfinite(state.q).all() and np.isfinite(state.quat).all()):
            self._latch("non-finite /lowstate")
            return

        if not self._engaged:
            self._prepose(state, msg.mode_machine)
            return

        if self._band_release_at is not None and not self._band_released and now >= self._band_release_at:
            self._release_band()

        if self.controller.mode.value in ("arm_goal", "settle_to_arm") and (self._slam_time is None or now - self._slam_time > SLAM_STALE_S):
            self.get_logger().warn("no fresh /Odometry: the arm goals move with the pelvis (is h12_slam running?)",
                                   throttle_duration_sec=5.0)
        cmd = self.controller.step(state)
        for name, obs in self.controller.last_obs.items():
            if not np.isfinite(obs).all():
                self._latch(f"non-finite {name} observation")
                return
        self._publish(cmd, msg.mode_machine)
        self._set_state(self.controller.mode.value)
        self._err_pub.publish(Float32MultiArray(data=self.controller.wrist_errors().astype(np.float32).tolist()))

    def _prepose(self, state: RobotState, mode_machine: int) -> None:
        ctrl = self.controller
        if self._prepose_start is None:
            self._prepose_start = state.q.copy()
            # the cosine blend's peak speed is (pi / 2) * distance / duration
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
        settled = self._prepose_pass >= PREPOSE_SETTLE_TICKS
        timed_out = self._prepose_ticks >= self._ramp_ticks + PREPOSE_MAX_TICKS
        if not settled and not timed_out:
            return
        if not self._livox_fresh(time.monotonic()):
            self._set_state("waiting_imu")
            self.get_logger().info("pre-posed; waiting for /livox/imu (h12_real_drivers use_lidar?)",
                                   throttle_duration_sec=5.0)
            return
        if self._confirm_wait and not self._confirmed:
            self._set_state("waiting_confirm")
            self.get_logger().info(
                f"pre-posed ({'settled' if settled else 'NOT SETTLED'}, max err {err:.3f} rad); holding. "
                "Engage with: ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger",
                throttle_duration_sec=5.0)
            return
        self.get_logger().info(f"pre-posed ({'settled' if settled else 'timeout'}, max err {err:.3f} rad): engaging")
        ctrl.reset(state)
        # a /cmd_vel that arrived before engaging is not a command to walk now
        ctrl.set_velocity_command(np.zeros(3))
        self._engaged = True
        if not self._band_released:
            self._band_release_at = time.monotonic() + float(self.get_parameter("band_release_delay_s").value)


def main():
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
