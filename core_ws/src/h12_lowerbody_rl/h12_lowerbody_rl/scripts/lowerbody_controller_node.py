"""Lower-body controller node: band-held idle -> pre-pose -> one pinned policy.

Flow
----
1. The robot starts held by the elastic band, **idle** (no policy drives the legs).
2. At startup the ``active_policy`` (default ``fame``) is auto-activated: once the
   safety layer's /safety/heartbeat arrives, the node pre-poses the legs (ramping
   the target from the measured pose to the policy's startup pose over
   ``prepose_ramp_s`` (FAME finetune uses its upstream 2s linear ramp)). Band
   release requires policy commitment and frame_task readiness (/left_ee_pose).
   FAME finetune sends its first policy output before requesting release.
   With ``engage_wait_for_confirm``
   (the default) the node HOLDS at the pre-pose crouch and only engages the policy
   after the operator calls the /lowerbody/confirm_engage Trigger service — the
   real-robot arming step. The sim bringup disables it for unattended runs.
3. From then on ``active_policy`` drives the legs for the whole run; there is no
   switching between policies. It receives /cmd_vel directly (ALMI and walk
   locomote on it; FAME stands). The opt-in ``fame_finetune`` and
   ``almi_manip_stand`` variants force zero velocity and fixed height.

Interfaces
----------
sub  /lowstate                 (unitree_hg/LowState)   robot state
sub  /cmd_vel                  (geometry_msgs/Twist)   velocity command -> policy
sub  /lowerbody/squat_cmd      (std_msgs/Float32)      base-height / squat (FAME)
sub  /left_ee_pose             (geometry_msgs/PoseStamped)  frame_task-ready signal
sub  /safety/heartbeat         (std_msgs/Bool)         safety-layer liveness; a False
                                                       beat OR silence for
                                                       heartbeat_timeout latches this
                                                       node silent (see watchdog)
pub  /safety/lowcmd_lower_in   (unitree_hg/LowCmd)     12-joint leg setpoints
pub  /lowerbody/active_policy  (std_msgs/String, latched)  active policy ("idle" when none)
srv  /lowerbody/confirm_engage (std_srvs/Trigger)      operator go-ahead: engage the
                                                       pre-posed policy (see Flow #2);
                                                       clears /cmd_vel and squat_cmd
"""

import os
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

from h12_lowerbody_rl.policy import (
    NUM_LEG_JOINTS,
    NUM_POLICY_JOINTS,
    AlmiPolicy,
    FamePolicy,
    LegCommand,
    RobotState,
    WalkPolicy,
    apply_imu_offset,
    imu_offset_quat,
    quat_rotate_inverse,
)
from h12_lowerbody_rl.manipulation_policy import AlmiManipStandPolicy, FameFinetunePolicy

# Live-tunable IMU trim parameters (degrees). Declared with a range so
# rqt_reconfigure renders them as sliders; updates apply on the next control
# tick. Only orientation axes exist — no policy observes base position, so
# there is nothing an x/y/z offset could feed into.
IMU_OFFSET_PARAMS = ("imu_offset_roll_deg", "imu_offset_pitch_deg", "imu_offset_yaw_deg")
IMU_OFFSET_RANGE_DEG = 15.0
from h12_lowerbody_rl.policy_manager import GateConfig, PolicyManager

MOTOR_MODE_PR = 1
# LowCmd header: ankle pitch/roll mode and the H1-2 machine id.
MODE_PR = 0
MODE_MACHINE = 6

# Safety-layer liveness flag (published by safety_node at 10 Hz: True while
# commands are relayed, False once its estop latches). The watchdog below arms
# on the FIRST beat received — so a bringup where the safety node comes up
# later (or a sim without one) is not tripped by startup silence — and then
# latches this node silent on a False beat or on heartbeat_timeout of silence.
SAFETY_HEARTBEAT_TOPIC = "/safety/heartbeat"

# Policy registry keys (see LowerBodyControllerNode.__init__).
FAME = "fame"   # RMA balance/stand policy (does not locomote)
WALK = "walk"   # TorchScript walk policy (follows /cmd_vel)
ALMI = "almi"   # ALMI LSTM policy: stands (incl. under arm motion) and walks on
                # /cmd_vel; AlmiPolicy resets its LSTM at gait onset

# Pre-pose: before engaging the policy, PD-drive the legs to its startup target
# (band-held). Legacy policies use their nominal crouch; FAME finetune uses the
# upstream lower_start_pos and 2s linear ramp. The legacy RMA policy requires its
# trained crouch before the band drops, rather than the straight-leg spawn pose.
# Commit follows measured settling or the existing timeout, plus operator confirm.
#
# Legacy targets use a cosine blend over ``prepose_ramp_s`` from the first
# measured leg pose. The ramp begins only once the safety relay is live to avoid
# a full-gain position step when commands first reach the motors.
PREPOSE_TOL = 0.15          # rad; max |q - startup target| across all 12 leg joints
PREPOSE_SETTLE_TICKS = 10   # consecutive in-tolerance ticks required (~0.2s @ 50Hz)
PREPOSE_MAX_TICKS = 400     # ~8s @ 50Hz after the ramp; commit anyway if never settled (safety)


def _share(*parts: str) -> str:
    return os.path.join(get_package_share_directory("h12_lowerbody_rl"), *parts)


class LowerBodyControllerNode(Node):
    def __init__(self):
        super().__init__("lowerbody_controller_node")

        self.declare_parameter("control_hz", 50.0)
        # Policy to auto-activate at startup (releases the band when ready).
        # Default "fame" (stand). "none" leaves the robot band-held idle forever,
        # so set a real policy here.
        self.declare_parameter("active_policy", "fame")
        self.declare_parameter("walk_config", _share("policies", "walk", "walk.yaml"))
        self.declare_parameter("fame_config", _share("policies", "fame", "fame.yaml"))
        self.declare_parameter("almi_config", _share("policies", "almi", "almi.yaml"))
        for name in ("fame_finetune", "almi_manip_stand"):
            self.declare_parameter(name + "_config", _share("policies", name, name + ".yaml"))
        self.declare_parameter("standing_lowstate_timeout", 0.1)
        # IMU mounting/calibration trim (deg): what the raw quaternion reads
        # when the torso is known to be level (e.g. measured hanging plumb).
        # Positive pitch = reads tilted forward, positive roll = tilted right.
        # Applied once in _state_from_lowstate so every policy's gravity obs
        # sees the corrected orientation. DYNAMIC — adjust live during a run
        # via rqt_reconfigure sliders or `ros2 param set` (see the parameter
        # callback below); launch files seed the startup values.
        for pname in IMU_OFFSET_PARAMS:
            self.declare_parameter(pname, 0.0, ParameterDescriptor(
                description="live IMU orientation trim, degrees (raw reading at true level)",
                floating_point_range=[FloatingPointRange(
                    from_value=-IMU_OFFSET_RANGE_DEG, to_value=IMU_OFFSET_RANGE_DEG,
                    step=0.0)]))
        self.declare_parameter("default_height_cmd", 1.0)
        # NOTE: /cmd_vel is latched, so if a publisher stops mid-command the last
        # (nonzero) value holds and the robot keeps walking -- publish an
        # explicit zero to stand.
        # Seconds to ramp the pre-pose target from the measured leg pose to the
        # policy's nominal (cosine blend; see the PREPOSE block above). Starting
        # at the measured pose means zero initial PD error, so full gains apply
        # no step torque. 0 disables the ramp (step straight to the nominal).
        self.declare_parameter("prepose_ramp_s", 2.0)
        # Hold at the pre-pose crouch until an operator confirms engagement via
        # the /lowerbody/confirm_engage Trigger service. Default TRUE — on the
        # real robot the engage is the moment the policy takes authority, so it
        # must be an explicit operator action. The sim bringup passes false to
        # keep unattended runs auto-engaging. While waiting,
        # the pre-pose timeout never auto-commits.
        self.declare_parameter("engage_wait_for_confirm", True)
        self.declare_parameter("disable_elastic_band", True)
        # Release the band only after frame_task_server has finished its open-loop
        # startup (it publishes /left_ee_pose only then); earlier release crashes
        # frame_task and free-falls the robot.
        self.declare_parameter("band_wait_for_frame_task", True)
        self.declare_parameter("band_release_topic", "/left_ee_pose")
        self.declare_parameter("band_max_wait", 30.0)
        # Seconds of /safety/heartbeat silence (after the first beat) before
        # this node latches silent. The heartbeat runs at 10 Hz, so 0.5 s =
        # five consecutive missed beats — enough slack for scheduler jitter,
        # short enough that a dead safety node stops leg commands within half
        # a second.
        self.declare_parameter("heartbeat_timeout", 0.5)

        control_hz = float(self.get_parameter("control_hz").value)
        startup_policy = str(self.get_parameter("active_policy").value).strip().lower()
        walk_cfg = self.get_parameter("walk_config").value
        fame_cfg = self.get_parameter("fame_config").value
        almi_cfg = self.get_parameter("almi_config").value
        self._imu_corr = self._build_imu_corr()
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._height_cmd = float(self.get_parameter("default_height_cmd").value)
        self._confirm_wait = bool(self.get_parameter("engage_wait_for_confirm").value)
        self._engage_confirmed = False
        self._prepose_ramp_ticks = max(1, int(round(
            float(self.get_parameter("prepose_ramp_s").value) * control_hz)))

        self.get_logger().info("loading lower-body policies...")
        standing_classes = {"fame_finetune": FameFinetunePolicy,
                            "almi_manip_stand": AlmiManipStandPolicy}
        self._standing_policy = None
        self._standing_state_timeout = float(
            self.get_parameter("standing_lowstate_timeout").value)
        if startup_policy in standing_classes:
            if control_hz != 50.0:
                raise ValueError("Manipulation standing policies require control_hz=50")
            if (not np.isfinite(self._standing_state_timeout)
                    or self._standing_state_timeout <= 0.):
                raise ValueError("standing_lowstate_timeout must be finite and positive")
            self._standing_policy = standing_classes[startup_policy](
                self.get_parameter(startup_policy + "_config").value)
            policies = {startup_policy: self._standing_policy}
        else:
            policies = {
                "walk": WalkPolicy(walk_cfg),
                "fame": FamePolicy(fame_cfg),
                "almi": AlmiPolicy(almi_cfg),
            }
        if "fame" in policies and not policies["fame"].has_encoder:
            self.get_logger().warn(
                "FAME encoder not loaded — z_t will be zeros (out-of-distribution). "
                "Check policies/fame/encoder_3800.pt."
            )
        self._manager = PolicyManager(
            policies, gate=GateConfig(), log=lambda m: self.get_logger().info(m)
        )

        self._lowstate: LowState | None = None
        self._lowstate_received_at: float | None = None
        self._cmd = np.zeros(3, dtype=np.float32)

        # band / activation state
        self._band_released = not bool(self.get_parameter("disable_elastic_band").value)
        self._frame_task_ready = not bool(self.get_parameter("band_wait_for_frame_task").value)
        self._band_cli = self.create_client(Trigger, "/elastic_band/toggle")
        self._awaiting_band_release = False  # policy committed, band not yet released
        self._band_release_inflight = False  # a toggle call is dispatched, awaiting result
        self._band_retry_at = 0.0  # monotonic retry deadline after unavailable/failed calls
        self._band_wait_warned = False       # rate-limit the "service not ready" warning
        self._request_time: float | None = None  # when the pending activation was asked
        self._prepose_pass = 0   # consecutive ticks the legs have held at nominal (pre-engage)
        self._prepose_ticks = 0  # total pre-pose ticks so far (ramp phase + settle timeout)
        self._prepose_start_q: np.ndarray | None = None  # measured legs at first pre-pose tick

        # safety-heartbeat watchdog state (see SAFETY_HEARTBEAT_TOPIC)
        self._hb_timeout = float(self.get_parameter("heartbeat_timeout").value)
        self._hb_last: float | None = None  # monotonic stamp of the last healthy beat
        self._hb_dead = False               # latched: never publishes again once True

        lowstate_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                  history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, lowstate_qos)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Float32, "/lowerbody/squat_cmd", self._on_squat_cmd, 10)
        self.create_subscription(
            PoseStamped, self.get_parameter("band_release_topic").value,
            self._on_frame_task_ready, 10)
        self.create_subscription(
            Bool, SAFETY_HEARTBEAT_TOPIC, self._on_heartbeat, 10)
        self._cmd_pub = self.create_publisher(LowCmd, "/safety/lowcmd_lower_in", 10)
        self._active_pub = self.create_publisher(String, "/lowerbody/active_policy", latched)
        self._confirm_srv = self.create_service(
            Trigger, "/lowerbody/confirm_engage", self._on_confirm_engage)
        self._publish_active()

        self.create_timer(1.0 / control_hz, self._tick)

        command_mode = ("Standing-only: velocity and squat commands ignored."
                        if self._standing_policy is not None
                        else "Policy follows /cmd_vel; no start services.")
        self.get_logger().info(
            f"lowerbody_controller ready: policies={self._manager.names()}, "
            f"control_hz={control_hz}. {command_mode}"
        )

        if startup_policy and startup_policy != "none":
            self.get_logger().info(f"active_policy={startup_policy!r}: auto-activating at startup")
            self._request_policy(startup_policy)
        else:
            self.get_logger().warn(
                "active_policy='none' with start services removed -> robot stays "
                "band-held idle forever. Set active_policy to 'fame' or 'walk'."
            )

    # -- request entry points ------------------------------------------------
    def _request_policy(self, name: str) -> tuple[bool, str]:
        ok, msg = self._manager.request(name)
        if ok and self._manager.is_pending() and self._manager.is_idle():
            self._request_time = time.monotonic()  # start band-release timeout
        return ok, msg

    # -- callbacks -----------------------------------------------------------
    def _on_lowstate(self, msg: LowState) -> None:
        self._lowstate = msg
        self._lowstate_received_at = time.monotonic()

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._cmd[0] = msg.linear.x
        self._cmd[1] = msg.linear.y
        self._cmd[2] = msg.angular.z
        np.clip(self._cmd, -1.0, 1.0, out=self._cmd)

    def _on_squat_cmd(self, msg: Float32) -> None:
        self._height_cmd = float(msg.data)

    def _on_frame_task_ready(self, _msg: PoseStamped) -> None:
        self._frame_task_ready = True

    def _on_heartbeat(self, msg: Bool) -> None:
        # First beat ARMS the watchdog; a False beat means the safety layer's
        # estop latched (it keeps publishing False after the latch — see
        # safety_node._publish_heartbeat), so leg commands are pointless at
        # best and a fight against the latch at worst.
        if self._hb_dead:
            return
        if not msg.data:
            self._trip_heartbeat("safety layer reports estop latched (heartbeat False)")
            return
        self._hb_last = time.monotonic()

    def _trip_heartbeat(self, reason: str) -> None:
        """Latch the node silent: log once, and _tick never publishes again.
        Deliberately NOT recoverable at runtime — the safety layer's own latch
        needs a safety_node restart, so this node restarts with it."""
        self._hb_dead = True
        self.get_logger().error(
            f"SAFETY HEARTBEAT LOST: {reason} — stopping all "
            "/safety/lowcmd_lower_in output (latched; restart this node once "
            "the safety layer is back)")

    def _on_confirm_engage(self, _req, resp):
        pending = self._manager.desired_name if self._manager.is_pending() else None
        if self._engage_confirmed:
            resp.success = True
            resp.message = "already confirmed"
            return resp
        self._engage_confirmed = True
        self._clear_commands()
        resp.success = True
        resp.message = (f"confirmed — engaging {pending!r} once the pre-pose settles"
                        if pending else "confirmed (no activation pending)")
        self.get_logger().info(f"operator confirm: {resp.message}")
        return resp

    def _clear_commands(self) -> None:
        # Commands that arrived before the operator's go-ahead are not commands
        # to act on now: zero the velocity and restore the default squat height.
        self._cmd[:] = 0.0
        self._height_cmd = float(self.get_parameter("default_height_cmd").value)

    # -- IMU trim ------------------------------------------------------------
    def _build_imu_corr(self):
        r, p, y = (float(self.get_parameter(n).value) for n in IMU_OFFSET_PARAMS)
        return imu_offset_quat(np.radians(r), np.radians(p), np.radians(y))

    def _on_set_parameters(self, params) -> SetParametersResult:
        # Runs on the same single-threaded executor as _tick, so swapping the
        # cached correction here is race-free. Values outside the declared
        # range are already rejected by rclpy before this callback fires.
        touched = [p for p in params if p.name in IMU_OFFSET_PARAMS]
        if touched:
            if any(not np.isfinite(p.value) for p in touched):
                return SetParametersResult(
                    successful=False, reason="IMU trim must be finite")
            # Parameters in `params` are not stored yet — merge them with the
            # current values instead of reading back via get_parameter.
            vals = {n: float(self.get_parameter(n).value) for n in IMU_OFFSET_PARAMS}
            vals.update({p.name: float(p.value) for p in touched})
            self._imu_corr = imu_offset_quat(*(np.radians(vals[n]) for n in IMU_OFFSET_PARAMS))
            self.get_logger().info(
                "IMU trim -> roll %+.3f pitch %+.3f yaw %+.3f deg"
                % tuple(vals[n] for n in IMU_OFFSET_PARAMS))
        return SetParametersResult(successful=True)

    # -- helpers -------------------------------------------------------------
    def _publish_active(self) -> None:
        self._active_pub.publish(String(data=self._manager.active_name or "idle"))

    def _publish_leg(self, leg: LegCommand) -> None:
        cmd_msg = LowCmd()
        # Header matches frame_task_server's LowCmdPublisher
        # (h12_ros2_controller/core/channel_interface.py). The safety layer
        # forwards the header of whichever channel spoke last, so both channels
        # must send the same values, and the robot drops any LowCmd whose
        # mode_machine differs from its own.
        cmd_msg.mode_pr = MODE_PR
        cmd_msg.mode_machine = MODE_MACHINE
        for i in range(NUM_LEG_JOINTS):
            m = cmd_msg.motor_cmd[i]
            m.mode = MOTOR_MODE_PR
            m.q = float(leg.target_q[i])
            m.dq = 0.0
            m.tau = 0.0
            m.kp = float(leg.kp[i])
            m.kd = float(leg.kd[i])
        self._cmd_pub.publish(cmd_msg)

    def _release_band(self, reason: str) -> None:
        # A toggle is not an idempotent disable: allow one outstanding request
        # and require the simulator's explicit disabled acknowledgment.
        if (self._band_released or self._band_release_inflight
                or time.monotonic() < self._band_retry_at):
            return
        self._band_retry_at = time.monotonic() + 0.5
        if not self._band_cli.service_is_ready():
            if not self._band_wait_warned:
                self.get_logger().warn(
                    "elastic band toggle service not ready yet — retrying until it appears")
                self._band_wait_warned = True
            return
        self._band_release_inflight = True
        self.get_logger().info(f"releasing elastic band ({reason})")
        try:
            self._band_cli.call_async(Trigger.Request()).add_done_callback(self._on_band_toggled)
        except Exception as error:
            self._band_release_inflight = False
            self.get_logger().warn(f"elastic band toggle dispatch failed: {error}; will retry")

    def _on_band_toggled(self, fut) -> None:
        self._band_release_inflight = False
        self._band_retry_at = time.monotonic() + 0.5
        try:
            result = fut.result()
        except Exception as error:
            self.get_logger().warn(f"elastic band toggle call failed: {error}; will retry")
            return
        if result is None or not result.success or result.message != "disabled":
            self.get_logger().warn(
                f"elastic band disable not confirmed: {result}; will retry")
            return
        self._band_released = True
        self.get_logger().info(f"elastic band toggle: {result.message}")

    def _state_from_lowstate(self, msg: LowState) -> RobotState:
        q = np.array([msg.motor_state[i].q for i in range(NUM_POLICY_JOINTS)], dtype=np.float32)
        dq = np.array([msg.motor_state[i].dq for i in range(NUM_POLICY_JOINTS)], dtype=np.float32)
        quat = apply_imu_offset(
            np.asarray(msg.imu_state.quaternion, dtype=np.float32), self._imu_corr)
        gyro = np.asarray(msg.imu_state.gyroscope, dtype=np.float32)
        if self._standing_policy is not None and self._imu_corr is not None:
            gyro = quat_rotate_inverse(self._imu_corr, gyro)
        t = self.get_clock().now().nanoseconds * 1e-9
        return RobotState(q=q, dq=dq, quat=quat, gyro=gyro,
                          cmd=self._cmd.copy(), height_cmd=self._height_cmd, t=t)

    # -- main loop -----------------------------------------------------------
    def _tick(self) -> None:
        # Safety-heartbeat watchdog: once tripped (False beat, or silence past
        # heartbeat_timeout after the first beat), publish NOTHING — no policy
        # step, no pre-pose, no band release. Checked before anything else so
        # every publish path in this tick is behind the gate.
        if self._hb_dead:
            return
        if (self._hb_last is not None
                and time.monotonic() - self._hb_last > self._hb_timeout):
            self._trip_heartbeat(
                f"no /safety/heartbeat for {self._hb_timeout:.1f}s "
                "(safety node died or stopped publishing)")
            return
        if self._lowstate is None:
            return
        state = self._state_from_lowstate(self._lowstate)
        if self._standing_policy is not None:
            try:
                if (self._lowstate_received_at is None
                        or time.monotonic() - self._lowstate_received_at
                        > self._standing_state_timeout):
                    raise ValueError("stale LowState for manipulation standing")
                self._standing_policy.validate_state(state)
            except ValueError as error:
                self._trip_heartbeat(str(error))
                return

        committed = False
        if self._manager.is_pending() and self._manager.is_idle():
            # Hold the policy's startup pose before engagement. FAME finetune's
            # deploy lower_start_pos differs from its observation/action center.
            # The band stays held until the policy is committed.
            pending = self._manager.desired_policy()
            if pending is None:
                return
            # Leg commands only reach the motors once the safety layer is
            # relaying. Ramping before then runs the ramp out against legs
            # that never move, and the first relayed command steps them to
            # the full crouch at full gains.
            if self._hb_last is None:
                self.get_logger().info(
                    "waiting for /safety/heartbeat before pre-posing the legs",
                    throttle_duration_sec=5.0)
                return
            fame_startup = pending.name == "fame_finetune"
            cmd = pending.prepose_command() if fame_startup else pending.nominal_command()
            prepose_target = cmd.target_q.copy()
            ramp_ticks = (round(pending.startup_seconds * pending.control_hz)
                          if fame_startup else self._prepose_ramp_ticks)
            if self._prepose_start_q is None:
                self._prepose_start_q = state.q[:NUM_LEG_JOINTS].copy()
                if fame_startup and (
                        np.any(self._prepose_start_q < pending._lower_limit)
                        or np.any(self._prepose_start_q > pending._upper_limit)):
                    # deploy_real.py:_move_to_default_pose rejects out-of-limit
                    # measured initial joints before sending any startup command.
                    self._trip_heartbeat("FAME startup joints outside configured limits")
                    return
            if self._prepose_ticks < ramp_ticks:
                frac = self._prepose_ticks / ramp_ticks
                alpha = ((self._prepose_ticks + 1) / ramp_ticks if fame_startup
                         else 0.5 * (1.0 - np.cos(np.pi * frac)))
                cmd.target_q = (1.0 - alpha) * self._prepose_start_q + alpha * cmd.target_q
                if fame_startup:
                    cmd.target_q = np.clip(cmd.target_q, pending._lower_limit, pending._upper_limit)
            self._publish_leg(cmd)
            err = float(np.max(np.abs(state.q[:NUM_LEG_JOINTS] - prepose_target)))
            self._prepose_ticks += 1
            ramp_done = not fame_startup or self._prepose_ticks >= ramp_ticks
            self._prepose_pass = (self._prepose_pass + 1
                                 if ramp_done and err < PREPOSE_TOL else 0)
            settled = self._prepose_pass >= PREPOSE_SETTLE_TICKS
            timed_out = self._prepose_ticks >= ramp_ticks + PREPOSE_MAX_TICKS
            if fame_startup and self._prepose_ticks < ramp_ticks:
                return  # Complete the upstream 2s ramp even if already near the target.
            if not settled and not timed_out:
                return  # keep posing; don't engage the policy or drop the band yet
            if self._confirm_wait and not self._engage_confirmed:
                # ARMING GATE: hold the legs at the pre-pose crouch until the
                # operator explicitly confirms. Applies to the timeout path
                # too — an unsettled pre-pose must never auto-engage on real.
                self.get_logger().info(
                    f"legs pre-posed ({'settled' if settled else 'NOT SETTLED (timeout)'}, "
                    f"err={err:.3f} rad) — holding; to engage "
                    f"{self._manager.desired_name!r} run: ros2 service call "
                    "/lowerbody/confirm_engage std_srvs/srv/Trigger",
                    throttle_duration_sec=5.0)
                return
            self.get_logger().info(
                f"legs pre-posed to startup target ({'settled' if settled else 'timeout'}, "
                f"err={err:.3f} rad) — engaging {self._manager.desired_name!r}")
            # Now engage the policy; it drives the legs while the band still holds.
            # Engage from zero commands (the policy reset below seeds from them).
            self._clear_commands()
            state = self._state_from_lowstate(self._lowstate)
            self._manager.commit(state)   # resets policy -> clean warm-up
            committed = True
            self._publish_active()
            self._awaiting_band_release = not self._band_released

        # Release the band only after a policy is committed and driving the legs
        # (gated on frame_task being ready, or the max-wait timeout).
        fame_active = (self._standing_policy is not None
                       and self._standing_policy.name == "fame_finetune")
        if self._awaiting_band_release and not (fame_active and committed):
            if not self._band_released and self._frame_task_ready:
                self._release_band("policy active + frame_task ready")
            elif not self._band_released and self._request_time is not None and \
                    (time.monotonic() - self._request_time) > self.get_parameter("band_max_wait").value:
                self._release_band("policy active + band_max_wait timeout")
            if self._band_released:
                # FAME's upstream history starts at confirm and runs continuously;
                # a GOLEM tether acknowledgment must not replay its first action.
                if not fame_active:
                    self._manager.reset_active(state)
                self._awaiting_band_release = False

        if self._manager.is_idle():
            return  # nothing requested yet -> band-held, legs uncommanded

        try:
            command = self._manager.run(state)
        except (ValueError, RuntimeError) as error:
            if self._standing_policy is None:
                raise
            self._trip_heartbeat(f"standing policy inference failed: {error}")
            return
        self._publish_leg(command)


def main():
    rclpy.init()
    node = LowerBodyControllerNode()
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
