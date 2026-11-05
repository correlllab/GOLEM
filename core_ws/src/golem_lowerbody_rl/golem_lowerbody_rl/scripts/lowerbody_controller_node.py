"""
Lower-body controller node: one pinned policy on the H1-2's legs.

Start-up
--------
The phases and every check are ``LegSupervisor``'s (``supervisor.py``):

1. **waiting**: silent until golem_safety's first true /safety/heartbeat and
   a finite, upright /lowstate. ``active_policy`` (``fame``, ``walk`` or
   ``almi``; the bringups set it from ``lowerbody:=``) is the policy for the
   whole run; there is no switching between policies.
2. **ramping**: the node pre-poses the legs, ramping the target from the
   measured pose to the policy's nominal crouch over ``prepose_ramp_s``
   (never a step at full gains), until they settle there or
   PREPOSE_MAX_TICKS pass.
3. **holding** (``engage_wait_for_confirm``, the default, the real robot):
   the node holds the crouch until the operator calls the
   /lowerbody/confirm_engage Trigger service, accepted only now. The sim
   bringup disables it for unattended runs.
4. **running**: the policy engages at zero velocity and the default squat
   height and drives the legs for the whole run. It receives /cmd_vel (ALMI
   and walk locomote on it; FAME stands); a command older than
   ``cmd_vel_timeout`` is zero velocity. With ``disable_elastic_band`` (the
   sim) the node then calls /elastic_band/toggle once, when the service is
   there, once /left_ee_pose has come (frame_task_server publishes it after
   its start-up moves) or ``band_max_wait`` after the start. The reply is
   final: a success takes the band off and resets the policy's history on
   the next tick; a failure is logged and leaves the band on.
5. **faulted**: a fault (``supervisor.py``: a false or silent heartbeat, a
   stale, non-finite or fallen /lowstate, a policy step that fails or gives
   non-finite targets, an unexpected error in a tick) is logged once, and
   the node publishes nothing from then on and stays up. golem_safety keeps
   relaying the last leg command.

Interfaces
----------
sub  /lowstate                 (unitree_hg/LowState)   robot state
sub  /cmd_vel                  (geometry_msgs/Twist)   velocity command -> policy
sub  /lowerbody/squat_cmd      (std_msgs/Float32)      base-height / squat (FAME)
sub  /left_ee_pose             (geometry_msgs/PoseStamped)  frame_task-ready signal
sub  /safety/heartbeat         (std_msgs/Bool)         golem_safety's: true while it relays
pub  /safety/lowcmd_lower_in   (unitree_hg/LowCmd)     12-joint leg setpoints
pub  /lowerbody/active_policy  (std_msgs/String, latched)  active policy ("idle" when none)
srv  /lowerbody/confirm_engage (std_srvs/Trigger)      operator go-ahead while holding
cli  /elastic_band/toggle      (std_srvs/Trigger)      the sim's band, with disable_elastic_band
"""

import dataclasses
import os
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from golem_lowerbody_rl.policy import (
    AlmiPolicy,
    FamePolicy,
    LegCommand,
    NUM_LEG_JOINTS,
    NUM_POLICY_JOINTS,
    RobotState,
    WalkPolicy,
    apply_imu_offset,
    imu_offset_quat,
)
from golem_lowerbody_rl.policy_manager import PolicyManager
from golem_lowerbody_rl.supervisor import FAULTED, HOLDING, RAMPING, RUNNING, LegSupervisor
import numpy as np
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

# Live-tunable IMU trim parameters (degrees). Declared with a range so
# rqt_reconfigure renders them as sliders; updates apply on the next /lowstate
# sample. Only orientation axes exist — no policy observes base position, so
# there is nothing an x/y/z offset could feed into.
IMU_OFFSET_PARAMS = ("imu_offset_roll_deg", "imu_offset_pitch_deg", "imu_offset_yaw_deg")
IMU_OFFSET_RANGE_DEG = 15.0

MOTOR_MODE_PR = 1
# LowCmd header: ankle pitch/roll mode and the H1-2 machine id.
MODE_PR = 0
MODE_MACHINE = 6

SAFETY_HEARTBEAT_TOPIC = "/safety/heartbeat"
# /cmd_vel components are clipped to this (m/s, m/s, rad/s)
CMD_VEL_LIMIT = 1.0
# /elastic_band/toggle's message when the call left the band on (golem_robocasa's
# MeasurementBridge answers "enabled" or "disabled")
BAND_ON = "enabled"

# Pre-pose: before engaging the policy, PD-drive the legs to its nominal crouch
# (band-held in the sim) and wait until they settle there. The RMA policy warms
# up from its trained default pose; released from the straight-leg spawn it can't
# recover the transition before the band drops.
#
# The pre-pose target is RAMPED (cosine blend over ``prepose_ramp_s``) from the
# measured leg pose captured at the first pre-pose tick to nominal. Stepping
# straight to nominal at full kp jerked the real robot's hanging legs hard enough
# (0.24 rad hip-yaw error x kp 200 -> 4.7 rad/s) to trip the safety layer's dq
# estop (hip yaw limit 0.2 x 23 = 4.6 rad/s) the moment the relay opened.
PREPOSE_TOL = 0.15          # rad; max |q - nominal| across the 12 legs to count as posed
PREPOSE_SETTLE_TICKS = 10   # consecutive in-tolerance ticks required (~0.2s @ 50Hz)
PREPOSE_MAX_TICKS = 400     # ~8s @ 50Hz after the ramp; posed anyway if never settled
# how often the holding node reminds the operator to confirm (s)
CONFIRM_REMINDER_SECONDS = 5.0


def _share(*parts: str) -> str:
    return os.path.join(get_package_share_directory("golem_lowerbody_rl"), *parts)


class LowerBodyControllerNode(Node):
    """The leg node (module docstring)."""

    def __init__(self):
        super().__init__("lowerbody_controller_node")

        self.declare_parameter("control_hz", 50.0)
        # The policy for the whole run: fame (stands), walk or almi (both follow
        # /cmd_vel). "none" leaves the legs uncommanded (band-held in the sim).
        self.declare_parameter("active_policy", "fame")
        self.declare_parameter("walk_config", _share("policies", "walk", "walk.yaml"))
        self.declare_parameter("fame_config", _share("policies", "fame", "fame.yaml"))
        self.declare_parameter("almi_config", _share("policies", "almi", "almi.yaml"))
        # IMU mounting/calibration trim (deg): what the raw quaternion reads
        # when the torso is known to be level (e.g. measured hanging plumb).
        # Positive pitch = reads tilted forward, positive roll = tilted right.
        # Applied to every /lowstate sample, so every policy's gravity obs and
        # the upright and fall checks see the corrected orientation. DYNAMIC —
        # adjust live during a run via rqt_reconfigure sliders or
        # `ros2 param set` (see _on_set_parameters); launch files seed the
        # startup values.
        for pname in IMU_OFFSET_PARAMS:
            self.declare_parameter(pname, 0.0, ParameterDescriptor(
                description="live IMU orientation trim, degrees (raw reading at true level)",
                floating_point_range=[FloatingPointRange(
                    from_value=-IMU_OFFSET_RANGE_DEG, to_value=IMU_OFFSET_RANGE_DEG,
                    step=0.0)]))
        self.declare_parameter("default_height_cmd", 1.0)
        # Seconds to ramp the pre-pose target from the measured leg pose to the
        # policy's nominal (cosine blend; see the PREPOSE block above). Starting
        # at the measured pose means zero initial PD error, so full gains apply
        # no step torque. 0 disables the ramp (step straight to the nominal).
        self.declare_parameter("prepose_ramp_s", 2.0)
        # Hold at the pre-pose crouch until an operator confirms engagement via
        # the /lowerbody/confirm_engage Trigger service. Default TRUE — on the
        # real robot the engage is the moment the policy takes authority, so it
        # must be an explicit operator action. The sim bringup passes false to
        # keep unattended runs auto-engaging. With this true, a pre-pose that
        # never settles holds for the confirm too.
        self.declare_parameter("engage_wait_for_confirm", True)
        # true: the sim's elastic band holds the robot and the node releases it
        self.declare_parameter("disable_elastic_band", True)
        # Release the band only after frame_task_server has finished its open-loop
        # startup (it publishes /left_ee_pose only then); earlier release crashes
        # frame_task and free-falls the robot.
        self.declare_parameter("band_wait_for_frame_task", True)
        self.declare_parameter("band_release_topic", "/left_ee_pose")
        # s from the node's start to releasing the band without /left_ee_pose
        self.declare_parameter("band_max_wait", 30.0)
        # Seconds of /safety/heartbeat silence (after the first beat) before
        # this node faults. The heartbeat runs at 10 Hz, so 0.5 s = five
        # consecutive missed beats — enough slack for scheduler jitter, short
        # enough that a dead safety node stops leg commands within half a
        # second.
        self.declare_parameter("heartbeat_timeout", 0.5)
        # Longest wall-clock gap (s) between /lowstate samples once the legs
        # are commanded; the sim configs relax it, since the sim can run
        # slower than real time.
        self.declare_parameter("max_state_gap", 0.25)
        # A /cmd_vel older than this (s) is zero velocity: publishers send it
        # continuously (Nav2, teleop at 20 Hz).
        self.declare_parameter("cmd_vel_timeout", 0.5)

        control_hz = float(self.get_parameter("control_hz").value)
        startup_policy = str(self.get_parameter("active_policy").value).strip().lower()
        walk_cfg = self.get_parameter("walk_config").value
        fame_cfg = self.get_parameter("fame_config").value
        almi_cfg = self.get_parameter("almi_config").value
        self._imu_corr = self._build_imu_corr()
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._prepose_ramp_ticks = max(1, int(round(
            float(self.get_parameter("prepose_ramp_s").value) * control_hz)))

        self._sup = LegSupervisor(
            heartbeat_timeout=float(self.get_parameter("heartbeat_timeout").value),
            max_state_gap=float(self.get_parameter("max_state_gap").value),
            cmd_vel_timeout=float(self.get_parameter("cmd_vel_timeout").value),
            wait_for_confirm=bool(self.get_parameter("engage_wait_for_confirm").value),
            default_height=float(self.get_parameter("default_height_cmd").value),
            log=lambda level, message: self._log(level, message))

        self.get_logger().info("loading lower-body policies...")
        policies = {
            "walk": WalkPolicy(walk_cfg),
            "fame": FamePolicy(fame_cfg),
            "almi": AlmiPolicy(almi_cfg),
        }
        if not policies["fame"].has_encoder:
            self.get_logger().warn(
                "FAME encoder not loaded — z_t will be zeros (out-of-distribution). "
                "Check fame.yaml's encoder_path."
            )
        self._manager = PolicyManager(
            policies, log=lambda m: self.get_logger().info(m)
        )

        # the newest /lowstate sample (cmd, height_cmd and t are filled per tick)
        self._state: RobotState | None = None
        self._gains: LegCommand | None = None   # the engaged policy's PD gains
        self._prepose_pass = 0   # consecutive ticks the legs have held at nominal
        self._prepose_ticks = 0  # total pre-pose ticks so far (ramp phase + settle timeout)
        self._prepose_start_q: np.ndarray | None = None  # measured legs at first pre-pose tick
        self._posed_note = ""    # how the pre-pose ended, for the confirm reminder
        self._last_reminder: float | None = None

        # the sim's elastic band: one call to /elastic_band/toggle, its reply final
        self._band_enabled = bool(self.get_parameter("disable_elastic_band").value)
        self._frame_task_ready = not bool(self.get_parameter("band_wait_for_frame_task").value)
        self._band_max_wait = float(self.get_parameter("band_max_wait").value)
        self._band_cli = self.create_client(Trigger, "/elastic_band/toggle")
        self._band_due = False        # the policy is engaged and the band still on
        self._band_called = False     # the one call has been sent
        self._band_released = False   # the call took the band off
        self._band_wait_warned = False
        self._reset_due = False       # reset the policy's history on the next tick
        self._start_time = time.monotonic()

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

        self.get_logger().info(
            f"lowerbody_controller ready: policies={self._manager.names()}, "
            f"control_hz={control_hz}."
        )

        if startup_policy and startup_policy != "none":
            ok, msg = self._manager.request(startup_policy)
            if not ok:
                self.get_logger().error(f"active_policy: {msg} -> the legs stay uncommanded")
        else:
            self.get_logger().warn(
                "active_policy='none' -> the legs stay uncommanded. "
                f"Set active_policy to one of {self._manager.names()}."
            )

    def _log(self, level: str, message: str) -> None:
        # one call per severity: rclpy refuses a change of severity at one call site
        if level == "error":
            self.get_logger().error(message)
        elif level == "warn":
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)

    # -- callbacks -----------------------------------------------------------
    def _on_lowstate(self, msg: LowState) -> None:
        self._state = self._read_state(msg)
        self._sup.on_lowstate(self._state, time.monotonic())

    def _on_cmd_vel(self, msg: Twist) -> None:
        cmd = np.clip([msg.linear.x, msg.linear.y, msg.angular.z], -CMD_VEL_LIMIT, CMD_VEL_LIMIT)
        self._sup.set_cmd_vel(*cmd, time.monotonic())

    def _on_squat_cmd(self, msg: Float32) -> None:
        self._sup.set_height_cmd(float(msg.data))

    def _on_frame_task_ready(self, _msg: PoseStamped) -> None:
        self._frame_task_ready = True

    def _on_heartbeat(self, msg: Bool) -> None:
        self._sup.on_heartbeat(bool(msg.data), time.monotonic())

    def _on_confirm_engage(self, _req, resp):
        resp.success, resp.message = self._sup.confirm()
        self._log("info" if resp.success else "warn",
                  f"/lowerbody/confirm_engage: {resp.message}")
        return resp

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
        # The H1-2's header, the one golem_safety stamps on every command it writes. The
        # robot drops any LowCmd whose mode_machine differs from its own.
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

    def _read_state(self, msg: LowState) -> RobotState:
        """Return a /lowstate sample, IMU trimmed; cmd, height_cmd and t are placeholders."""
        q = np.array([msg.motor_state[i].q for i in range(NUM_POLICY_JOINTS)], dtype=np.float32)
        dq = np.array([msg.motor_state[i].dq for i in range(NUM_POLICY_JOINTS)], dtype=np.float32)
        quat = apply_imu_offset(
            np.asarray(msg.imu_state.quaternion, dtype=np.float32), self._imu_corr)
        gyro = np.asarray(msg.imu_state.gyroscope, dtype=np.float32)
        return RobotState(q=q, dq=dq, quat=quat, gyro=gyro,
                          cmd=np.zeros(3, dtype=np.float32), height_cmd=0.0, t=0.0)

    def _policy_state(self, now: float) -> RobotState:
        """Return the newest sample with the commands at now and the node clock's time."""
        assert self._state is not None
        return dataclasses.replace(
            self._state, cmd=self._sup.cmd_vel(now), height_cmd=self._sup.height_cmd,
            t=self.get_clock().now().nanoseconds * 1e-9)

    # -- elastic band (sim) --------------------------------------------------
    def _update_band(self, now: float) -> None:
        """Send the one /elastic_band/toggle call once it is due and the service is there."""
        if not self._band_due or self._band_called:
            return
        if self._frame_task_ready:
            reason = "the policy is engaged and frame_task_server is up (/left_ee_pose)"
        elif now - self._start_time > self._band_max_wait:
            reason = (f"the policy is engaged and band_max_wait ({self._band_max_wait:.0f} s) "
                      "has passed with no /left_ee_pose")
        else:
            return
        if not self._band_cli.service_is_ready():
            if not self._band_wait_warned:
                self._log("warn", "the elastic band is due for release, but "
                                  "/elastic_band/toggle is not there yet: waiting for it")
                self._band_wait_warned = True
            return
        self._band_called = True
        self._log("info", f"releasing the elastic band: {reason}")
        self._band_cli.call_async(Trigger.Request()).add_done_callback(self._on_band_toggled)

    def _on_band_toggled(self, future) -> None:
        try:
            response = future.result()
            failure = None if response is not None and response.success else (
                "no response" if response is None
                else f'the service answered "{response.message}"')
        except Exception as e:  # noqa: BLE001 - reported once; the call is never sent again
            response, failure = None, repr(e)
        if failure is not None:
            self._log("error", f"elastic band release failed: {failure}. The band stays on; "
                               "the node does not call /elastic_band/toggle again")
            return
        if response.message == BAND_ON:
            # the band was already off (SPACE in the viewer), and the toggle hung the
            # robot from it again. Another call would take it off again, under a
            # robot someone may have hung up on purpose, so none is sent
            self._log("warn", '/elastic_band/toggle answered "enabled": the elastic band was '
                              "already off and is now ON, and the robot hangs from it again. "
                              "The node does not call the toggle again")
            return
        self._band_released = True
        # the policy's history filled with band-held states: it restarts on the
        # free robot, or FAME's first free actions come from stale obs and it topples
        self._reset_due = True
        self._log("info", f"elastic band released (/elastic_band/toggle: {response.message})")

    # -- main loop -----------------------------------------------------------
    def _tick(self) -> None:
        try:
            self._step(time.monotonic())
        except Exception as e:  # noqa: BLE001 - never leave the legs on a broken node
            self._sup.latch(f"unexpected error: {e!r}")

    def _step(self, now: float) -> None:
        self._sup.check_stale(now)
        phase = self._sup.phase
        if phase == FAULTED or self._state is None:
            return
        policy = self._manager.desired_policy()
        if policy is None or phase not in (RAMPING, HOLDING, RUNNING):
            return
        if phase == RAMPING:
            self._prepose(policy)
            return
        if phase == HOLDING:
            self._publish_leg(policy.nominal_command())
            if (self._last_reminder is None
                    or now - self._last_reminder >= CONFIRM_REMINDER_SECONDS):
                self._last_reminder = now
                self._log("info", f"legs pre-posed ({self._posed_note}) — holding; to engage "
                                  f"{self._manager.desired_name!r} run: ros2 service call "
                                  "/lowerbody/confirm_engage std_srvs/srv/Trigger")
            return
        state = self._policy_state(now)
        if self._manager.is_idle():
            self._manager.commit(state)   # resets the policy -> clean warm-up
            self._gains = policy.nominal_command()
            self._publish_active()
            self._band_due = self._band_enabled
        self._update_band(now)
        if self._reset_due:
            self._reset_due = False
            self._manager.reset_active(state)
        targets = self._sup.step_policy(lambda: self._manager.run(state).target_q)
        if targets is None:
            return
        assert self._gains is not None
        self._publish_leg(LegCommand(target_q=targets, kp=self._gains.kp, kd=self._gains.kd))

    def _prepose(self, policy) -> None:
        """PD-drive the legs along the ramp to the policy's nominal crouch, then call posed()."""
        assert self._state is not None
        q = self._state.q[:NUM_LEG_JOINTS]
        cmd = policy.nominal_command()
        if self._prepose_start_q is None:
            self._prepose_start_q = q.copy()
        if self._prepose_ticks < self._prepose_ramp_ticks:
            frac = self._prepose_ticks / self._prepose_ramp_ticks
            alpha = 0.5 * (1.0 - np.cos(np.pi * frac))
            cmd.target_q = (1.0 - alpha) * self._prepose_start_q + alpha * cmd.target_q
        self._publish_leg(cmd)
        err = float(np.max(np.abs(q - policy.nominal_lower)))
        self._prepose_ticks += 1
        self._prepose_pass = self._prepose_pass + 1 if err < PREPOSE_TOL else 0
        settled = self._prepose_pass >= PREPOSE_SETTLE_TICKS
        timed_out = self._prepose_ticks >= self._prepose_ramp_ticks + PREPOSE_MAX_TICKS
        if not settled and not timed_out:
            return
        # an unsettled pre-pose (timeout) is posed all the same; with
        # engage_wait_for_confirm it then holds for the confirm
        self._posed_note = (f"{'settled' if settled else 'NOT SETTLED (timeout)'}, "
                            f"err={err:.3f} rad")
        self._log("info", f"legs pre-posed to nominal ({self._posed_note})")
        self._sup.posed()


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
