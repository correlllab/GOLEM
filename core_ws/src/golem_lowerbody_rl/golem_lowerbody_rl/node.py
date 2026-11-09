"""
Run one leg policy on a robot's legs: the leg node, lowerbody_controller_node.

    ros2 launch golem_lowerbody_rl lowerbody.launch.py robot:=h12 config:=sim sim:=true

Parameters, read once at start except the IMU trim:

- robot (required): a robot with leg data here and a joint table in golem_safety.
- config (required): the run config, sim or real (config/<robot>/<name>.yaml), or a path.
- policy: one of the robot's policies (config/<robot>/policies/<name>.yaml), or a path;
  empty for the robot's default_policy.
- imu_trim_{roll,pitch,yaw}_deg: live, deg within +-IMU_TRIM_RANGE_DEG, what the IMU reads
  when the body is level; seeded from the run config's imu_trim_deg and applied to every
  /lowstate quaternion before anything reads it.

The node refuses at start what launch/lowerbody.launch.py refuses (launch_check.py). The
phases, checks and log lines are LegSupervisor's (supervisor.py), the observation and
action the policy spec's (policy/); this module moves messages to and from them.

Interfaces:

- sub /lowstate (unitree_hg/LowState, best effort, newest only): robot state.
- sub /cmd_vel (geometry_msgs/Twist): velocity command, raw; the policy shapes it.
- sub /lowerbody/height (std_msgs/Float32): the policy's height command, in its units.
- sub /safety/heartbeat (std_msgs/Bool): golem_safety's, true while it relays.
- sub /safety/lowcmd_upper_in (unitree_hg/LowCmd): the arm controller's commands; only
  whether they are fresh is read.
- sub /left_ee_pose (geometry_msgs/PoseStamped): with band.release only, dropped after the
  first: frame_task_server has finished its start-up moves.
- pub /safety/lowcmd_lower_in (unitree_hg/LowCmd): the leg targets, in the leg slots.
- pub /lowerbody/active_policy (std_msgs/String, latched): the policy's name while it
  drives the legs, otherwise "idle".
- srv /lowerbody/confirm_engage (std_srvs/Trigger): the operator's go-ahead.
- cli /elastic_band/toggle (std_srvs/Trigger): the sim's band, with band.release.

Clocks: robot time times the state gaps, the commands, the readiness waits and the band:
the node's clock (the sim's /clock) under use_sim_time, the steady clock otherwise. Under
use_sim_time the node takes no /lowstate before the first /clock, logging that it waits.
The heartbeat watchdog is on the steady clock always (LegSupervisor's module docstring).

Threads: /lowstate has a callback group of its own and everything else shares another,
under a two-thread executor (make_executor), so a state sample never queues behind the
other callbacks. One lock serialises every call into the supervisor and the policy, so a
band reply's policy reset never runs inside a policy step; a callback holding the lock
delays a state sample by as long as it holds it.
"""

import sys
import threading
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.exceptions import ParameterException
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

from golem_lowerbody_rl.imu import apply_trim, trim_quaternion
from golem_lowerbody_rl.launch_check import lowerbody_refusal
from golem_lowerbody_rl.policy import LegTargets, load_policy_spec, PolicyRunner
from golem_lowerbody_rl.profile import load_profile
from golem_lowerbody_rl.run_config import load_run_config, run_config_path
from golem_lowerbody_rl.supervisor import LegSupervisor, Phase, StateSample

__all__ = ['LowerBodyControllerNode', 'main', 'make_executor']

NODE_NAME = 'lowerbody_controller_node'
IMU_TRIM_PARAMS = ('imu_trim_roll_deg', 'imu_trim_pitch_deg', 'imu_trim_yaw_deg')
# deg: a trim beyond this is a mounting fault, not a calibration
IMU_TRIM_RANGE_DEG = 15.0
# s of steady time: the steady timer runs the heartbeat and /lowstate watchdogs and the band
# at 100 Hz
CHECK_PERIOD = 0.01
# LowCmd motor mode: the joint is driven
MOTOR_ON = 1
IDLE = 'idle'

# a reader that falls behind gets the newest sample, never a queue of old ones
NEWEST_ONLY = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
# golem_safety reads its inputs newest-only; a reliable writer matches that reader
COMMAND = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.RELIABLE)
LATCHED = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class LowerBodyControllerNode(Node):
    """The leg node (the module docstring)."""

    def __init__(self, **kwargs):
        """
        Load the robot, run config and policy, then create the node's interfaces.

        kwargs go to rclpy's Node (context, parameter_overrides). Raises ValueError for a
        missing parameter or a refused robot, run config or policy, and rclpy's
        ParameterException for an IMU trim override outside its range, before any interface
        exists.
        """
        super().__init__(NODE_NAME, **kwargs)
        robot = self._text_parameter('robot', 'the robot (required)')
        config = self._text_parameter('config', 'the run config: sim, real or a path '
                                                '(required)')
        policy = self._text_parameter('policy', "the policy, by name or path; empty: the "
                                                "robot's default_policy")
        for name, value in (('robot', robot), ('config', config)):
            if not value:
                raise ValueError(f'parameter {name} is required')
        refusal = lowerbody_refusal(robot, config, policy, sim=None)
        if refusal is not None:
            raise ValueError(refusal)
        self._profile = load_profile(robot)
        self._run = load_run_config(robot, config)
        spec = load_policy_spec(robot, policy or self._profile.default_policy)
        self._runner = PolicyRunner(self._profile, self._run, spec,
                                    get_package_share_directory('golem_lowerbody_rl'))
        self._lock = threading.Lock()
        self._supervisor = LegSupervisor(
            self._run, self._profile, self._runner.prepose, self._runner,
            log=lambda level, message: self._log(level, message))
        self._body = len(self._profile.body_joints)
        self._leg_slots = [int(slot) for slot in self._profile.slots(self._profile.legs)]
        # robot time is the node's clock (the sim's /clock) under use_sim_time
        self._sim_time = bool(self.get_parameter('use_sim_time').value)
        self._clock_wait_logged = False

        for name, value in zip(IMU_TRIM_PARAMS, self._run.imu_trim_deg):
            if abs(value) > IMU_TRIM_RANGE_DEG:
                raise ValueError(f"the run config's imu_trim_deg {list(self._run.imu_trim_deg)} "
                                 f'is not within +-{IMU_TRIM_RANGE_DEG:.0f} deg')
            self.declare_parameter(name, float(value), ParameterDescriptor(
                description='live IMU trim, deg: what the IMU reads when the body is level',
                floating_point_range=[FloatingPointRange(
                    from_value=-IMU_TRIM_RANGE_DEG, to_value=IMU_TRIM_RANGE_DEG, step=0.0)]))
        # the declared values: an override given at start-up wins over the run config's
        self._trim = trim_quaternion(*(float(self.get_parameter(name).value)
                                       for name in IMU_TRIM_PARAMS))
        self.add_on_set_parameters_callback(self._on_set_parameters)

        state_group = MutuallyExclusiveCallbackGroup()
        group = MutuallyExclusiveCallbackGroup()
        self._command_pub = self.create_publisher(LowCmd, '/safety/lowcmd_lower_in', COMMAND)
        self._active_pub = self.create_publisher(String, '/lowerbody/active_policy', LATCHED)
        self._active: str | None = None
        self._publish_active()
        self.create_subscription(LowState, '/lowstate', self._on_lowstate, NEWEST_ONLY,
                                 callback_group=state_group)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10,
                                 callback_group=group)
        self.create_subscription(Float32, '/lowerbody/height', self._on_height, 10,
                                 callback_group=group)
        self.create_subscription(Bool, '/safety/heartbeat', self._on_heartbeat, 10,
                                 callback_group=group)
        self.create_subscription(LowCmd, '/safety/lowcmd_upper_in', self._on_upper,
                                 NEWEST_ONLY, callback_group=group)
        self._ee_sub = None
        if self._run.band.release:
            self._ee_sub = self.create_subscription(
                PoseStamped, '/left_ee_pose', self._on_ee_pose, 10, callback_group=group)
        self._band_client = self.create_client(Trigger, '/elastic_band/toggle',
                                               callback_group=group)
        self.create_service(Trigger, '/lowerbody/confirm_engage', self._on_confirm,
                            callback_group=group)
        # SAFETY: a steady timer, whatever use_sim_time says: the heartbeat watchdog it runs
        # must fire while a sim's /clock stands still. The /lowstate watchdog and the band it
        # runs too count robot time
        self.create_timer(CHECK_PERIOD, self._on_check, callback_group=group,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))
        self._log('info', f'loaded run config {run_config_path(robot, config)} and policy '
                          f'{spec.path}')

    @property
    def supervisor(self) -> LegSupervisor:
        """Return the leg state machine; read only its phase without the lock."""
        return self._supervisor

    @property
    def runner(self) -> PolicyRunner:
        """Return the node's policy runner."""
        return self._runner

    def _text_parameter(self, name: str, description: str) -> str:
        """Declare a read-only string parameter (default '') and return its value."""
        self.declare_parameter(name, '', ParameterDescriptor(description=description,
                                                             read_only=True))
        return str(self.get_parameter(name).value).strip()

    def _log(self, level: str, message: str) -> None:
        """Log message at level 'info', 'warn' or 'error'."""
        # one call per severity: rclpy refuses a change of severity at one call site
        if level == 'error':
            self.get_logger().error(message)
        elif level == 'warn':
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)

    def _publish_active(self) -> None:
        """Publish the active policy when it changed: its name while it runs, else idle."""
        name = self._runner.name if self._supervisor.phase is Phase.RUNNING else IDLE
        if name != self._active:
            self._active = name
            self._active_pub.publish(String(data=name))

    def _node_t(self) -> float:
        """Return the node clock's time, s (a simulator's in the sims)."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _times(self) -> tuple[float, float]:
        """
        Return (robot time, steady time), s: the node's clock under use_sim_time, else steady.

        On the steady clock both are the one reading, as the supervisor takes them on the
        robots.
        """
        steady = time.monotonic()
        return (self._node_t() if self._sim_time else steady), steady

    def _clock_started(self) -> bool:
        """
        Return whether robot time has started (under use_sim_time, a first /clock).

        Logs once while it has not.
        """
        if not self._sim_time or self.get_clock().now().nanoseconds > 0:
            return True
        if not self._clock_wait_logged:
            self._clock_wait_logged = True
            self._log('info', 'waiting for the first /clock (use_sim_time): no /lowstate is '
                              'taken before it')
        return False

    # -- callbacks ---------------------------------------------------------------------

    def _on_lowstate(self, msg: LowState) -> None:
        """Hand a /lowstate sample to the supervisor and publish the targets it returns."""
        if not self._clock_started():
            return
        now, steady = self._times()
        try:
            sample = self._sample(msg)
        except Exception as e:  # noqa: BLE001 - a sample that cannot be read stops the legs
            with self._lock:
                self._supervisor.latch(f'unexpected error reading /lowstate: {e!r}')
                self._publish_active()
            return
        node_t = self._node_t()
        with self._lock:
            targets = self._supervisor.on_state(sample, now, node_t, steady)
            self._publish_active()
            if targets is not None:
                self._publish_legs(targets)

    def _sample(self, msg: LowState) -> StateSample:
        """Return msg as a StateSample, its IMU quaternion trimmed."""
        motors = msg.motor_state[:self._body]
        return StateSample(
            tick=int(msg.tick),
            q=np.array([motor.q for motor in motors], dtype=np.float32),
            dq=np.array([motor.dq for motor in motors], dtype=np.float32),
            quat_wxyz=apply_trim(np.asarray(msg.imu_state.quaternion, dtype=np.float32),
                                 self._trim),
            gyro=np.asarray(msg.imu_state.gyroscope, dtype=np.float32))

    def _publish_legs(self, targets: LegTargets) -> None:
        """Publish targets to golem_safety's lower-body input."""
        msg = LowCmd()
        # golem_safety stamps its own header on what it writes; the profile's matches it
        msg.mode_pr = self._profile.mode_pr
        msg.mode_machine = self._profile.mode_machine
        for i, slot in enumerate(self._leg_slots):
            motor = msg.motor_cmd[slot]
            motor.mode = MOTOR_ON
            motor.q = float(targets.q[i])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(targets.kp[i])
            motor.kd = float(targets.kd[i])
        self._command_pub.publish(msg)

    def _on_cmd_vel(self, msg: Twist) -> None:
        """Take a /cmd_vel."""
        with self._lock:
            self._supervisor.on_cmd(msg.linear.x, msg.linear.y, msg.angular.z,
                                    self._times()[0])

    def _on_height(self, msg: Float32) -> None:
        """Take a /lowerbody/height."""
        with self._lock:
            self._supervisor.on_height(float(msg.data), self._times()[0])

    def _on_heartbeat(self, msg: Bool) -> None:
        """Take golem_safety's heartbeat, on the steady clock (LegSupervisor.on_heartbeat)."""
        with self._lock:
            self._supervisor.on_heartbeat(bool(msg.data), time.monotonic())
            self._publish_active()

    def _on_upper(self, _msg: LowCmd) -> None:
        """Note that the arm controller commands."""
        with self._lock:
            self._supervisor.on_upper_cmd(self._times()[0])

    def _on_ee_pose(self, _msg: PoseStamped) -> None:
        """Take the first /left_ee_pose and drop the subscription."""
        with self._lock:
            self._supervisor.on_ee_pose(self._times()[0])
        if self._ee_sub is not None:
            self.destroy_subscription(self._ee_sub)
            self._ee_sub = None

    def _on_confirm(self, _request, response):
        """Answer /lowerbody/confirm_engage with the supervisor's verdict."""
        with self._lock:
            response.success, response.message = self._supervisor.confirm(self._times()[0])
            self._publish_active()
        self._log('info' if response.success else 'warn',
                  f'/lowerbody/confirm_engage: {response.message}')
        return response

    def _on_check(self) -> None:
        """Run the watchdogs; call the band toggle once it is due."""
        now, steady = self._times()
        ready = self._band_client.service_is_ready() if self._run.band.release else False
        with self._lock:
            self._supervisor.check(now, steady)
            self._publish_active()
            due = (self._supervisor.band_due(now, service_ready=ready)
                   if self._run.band.release else None)
        if due is not None:
            # SAFETY: the one call; its reply is final (BandRelease)
            future = self._band_client.call_async(Trigger.Request())
            future.add_done_callback(self._on_band_reply)

    def _on_band_reply(self, future) -> None:
        """Hand the band toggle's reply to the supervisor."""
        try:
            response = future.result()
        except Exception as e:  # noqa: BLE001 - reported once; the call is never sent again
            success, message = False, repr(e)
        else:
            if response is None:
                success, message = False, 'no response'
            elif not response.success:
                success, message = False, f'the service answered "{response.message}"'
            else:
                success, message = True, response.message
        with self._lock:
            self._supervisor.on_band_reply(success, message)
            self._publish_active()

    def _on_set_parameters(self, params) -> SetParametersResult:
        """Apply a new IMU trim from the next /lowstate sample on."""
        touched = [p for p in params if p.name in IMU_TRIM_PARAMS]
        if not touched:
            return SetParametersResult(successful=True)
        if any(not np.isfinite(p.value) for p in touched):
            return SetParametersResult(successful=False, reason='the IMU trim must be finite')
        # the new values are not stored yet: merge them with the current ones
        values = {name: float(self.get_parameter(name).value) for name in IMU_TRIM_PARAMS}
        values.update({p.name: float(p.value) for p in touched})
        # one reference swap: the /lowstate thread reads either the old or the new trim
        self._trim = trim_quaternion(*(values[name] for name in IMU_TRIM_PARAMS))
        self._log('info', 'IMU trim -> roll %+.3f pitch %+.3f yaw %+.3f deg'
                          % tuple(values[name] for name in IMU_TRIM_PARAMS))
        return SetParametersResult(successful=True)


def make_executor(node: Node, context=None) -> MultiThreadedExecutor:
    """Return a two-thread executor with node added: /lowstate gets a thread of its own."""
    executor = MultiThreadedExecutor(num_threads=2, context=context)
    executor.add_node(node)
    return executor


def main(args=None) -> int:
    """Run the leg node until shutdown; return 1 when it refuses to start."""
    rclpy.init(args=args)
    try:
        node = LowerBodyControllerNode()
    except (ValueError, ParameterException) as error:
        rclpy.logging.get_logger(NODE_NAME).error(f'not starting: {error}')
        rclpy.try_shutdown()
        return 1
    executor = make_executor(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
