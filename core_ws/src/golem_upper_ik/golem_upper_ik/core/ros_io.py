'''
The controller's ROS I/O, on rclpy:

- StateSubscriber: the newest /lowstate, its tick lag and its age. It is RobotModel's state
  source (golem_upper_ik.core.robot_model.StateSource).
- CommandPublisher: a thread that publishes the newest command to a golem_safety input
  (/safety/...) at pub_hz.
- SafetyOutputSubscriber: the newest command golem_safety relays (/lowcmd or /arm_sdk), which
  the follow gate reads and nothing here ever writes.

The subscribers keep one sample (keep-last 1, best effort): a reader that falls behind gets the
newest sample, never a queue of old ones. Each takes its samples in a mutually exclusive
callback group, its own unless the caller passes one, so a long callback elsewhere in the node
does not hold them up under a multi-threaded executor.

A subscriber's shutdown() only marks it closed: it takes no more samples, and its next
callback destroys the subscription. rclpy destroys a subscription safely there: its mutually
exclusive group is busy with that callback, so no wait set holds it. Destroyed from another
thread while an executor spins, a subscription the executor has already found ready raises
InvalidHandle in the executor and ends its spin. shutdown() therefore does not unsubscribe:
if no further sample arrives (or no executor spins), the subscription stays until the node is
destroyed, which is harmless.
'''
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_hg.msg import LowCmd, LowState

from golem_upper_ik.config import SAFETY_INPUT_PREFIX, SAFETY_OUTPUT_TOPICS
from golem_upper_ik.profile import RobotProfile

__all__ = [
    'CommandPublisher', 'SafetyOutputSubscriber', 'StaleStateError', 'StateSubscriber',
    'TickLagTracker', 'NEWEST_ONLY_QOS', 'COMMAND_QOS',
]

NEWEST_ONLY_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
# The Unitree SDK's writer QoS: it matches best-effort and reliable readers alike
COMMAND_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE)
# LowState.tick is a uint32 millisecond counter
TICK_WRAP = 1 << 32
# How fast [ms per s] the tick-lag baseline may rise to follow clock drift between the
# robot's tick and this host (500 ppm, well above crystal drift). A transport delay that
# builds faster than this registers as lag.
LAG_DRIFT_MS_PER_S = 0.5
# Largest magnitude rosidl accepts in a float32 message field
FLOAT32_MAX = 3.402823466e+38
# The publisher keeps to absolute deadlines and makes up a short stall with a burst. A stall
# longer than this restarts the deadlines from now, so it is not followed by a long burst of
# the same command.
MAX_CATCH_UP_S = 0.02


class StaleStateError(RuntimeError):
    '''/lowstate is too stale to control on'''


class TickLagTracker:
    '''
    How late [ms] samples stamped with the robot's millisecond tick arrive. The baseline is
    the earliest arrival seen; it may rise by LAG_DRIFT_MS_PER_S to follow clock drift, so
    only a delay that builds faster than drift registers.
    '''

    def __init__(self):
        self._last_tick = None
        self._wraps = 0
        self._base_ms = None
        self._base_time = 0.0

    def update(self, tick: int, now: float) -> float:
        '''The lag [ms] of a sample with tick that arrived at now [s, any steady clock]'''
        tick = int(tick)
        if self._last_tick is not None and tick < self._last_tick - TICK_WRAP // 2:
            self._wraps += 1
        self._last_tick = tick
        offset_ms = now * 1000.0 - (tick + self._wraps * TICK_WRAP)
        if self._base_ms is None:
            self._base_ms = offset_ms
        else:
            drift_ms = LAG_DRIFT_MS_PER_S * (now - self._base_time)
            self._base_ms = min(offset_ms, self._base_ms + drift_ms)
        self._base_time = now
        return offset_ms - self._base_ms


def _own_group(callback_group):
    if callback_group is None:
        return MutuallyExclusiveCallbackGroup()
    if not isinstance(callback_group, MutuallyExclusiveCallbackGroup):
        raise ValueError('a subscriber needs a mutually exclusive callback group, not '
                         f'{type(callback_group).__name__}')
    return callback_group


class _ClosableSubscription:
    '''A subscription whose close() is safe from any thread (see the module docstring)'''

    def __init__(self, node, msg_type, topic, callback, callback_group):
        self._node = node
        self._callback = callback
        self._closed = False
        self._subscription = node.create_subscription(
            msg_type, topic, self._receive, NEWEST_ONLY_QOS,
            callback_group=_own_group(callback_group))

    def _receive(self, msg):
        if not self._closed:
            self._callback(msg)
            return
        subscription, self._subscription = self._subscription, None
        if subscription is not None:
            self._node.destroy_subscription(subscription)

    def close(self):
        '''
        Take no more samples. The next callback destroys the subscription; without one it
        stays until the node is destroyed
        '''
        self._closed = True


class StateSubscriber:
    '''
    The newest /lowstate of one robot.

    latest(): (message, receive time on time.monotonic()), or None before the first sample.
    state: the newest sample as arrays, each read the reader's own copy: 'q' (rad) and 'dq'
        (rad/s), float32, one value per body joint in slot order, and 'imu_state', whose
        quaternion (w, x, y, z), gyroscope (rad/s), accelerometer (m/s^2) and rpy (rad) are
        float32 arrays. Before the first sample every array is 0 (the IMU quaternion too,
        which RobotModel reads as level).
    lag_ms: how late the newest sample arrived against the robot's tick (TickLagTracker), or
        None before the first sample.
    staleness_ms: the larger of lag_ms and the newest sample's age, ms; inf before the first.
    check_fresh(max_age_s): raises StaleStateError unless a sample arrived within max_age_s.
    '''

    def __init__(self, node, profile: RobotProfile, topic: str = '/lowstate',
                 callback_group=None):
        self._topic = topic
        self._count = len(profile.body_joints)
        self._lock = threading.Lock()
        self._latest = None
        self._sequence = 0
        self._lag = TickLagTracker()
        self._lag_ms = None
        self._converted_sequence = 0
        self._arrays = self._at_rest()
        self._subscription = _ClosableSubscription(
            node, LowState, topic, self._receive, callback_group)

    @property
    def topic(self) -> str:
        return self._topic

    def _receive(self, msg: LowState) -> None:
        now = time.monotonic()
        with self._lock:
            self._lag_ms = self._lag.update(msg.tick, now)
            self._latest = (msg, now)
            self._sequence += 1

    def latest(self):
        '''(the newest LowState, its receive time on time.monotonic()), or None'''
        with self._lock:
            return self._latest

    def _at_rest(self) -> dict:
        zeros = np.zeros(self._count, dtype=np.float32)
        return {
            'q': zeros, 'dq': zeros,
            'quaternion': np.zeros(4, dtype=np.float32),
            'gyroscope': np.zeros(3, dtype=np.float32),
            'accelerometer': np.zeros(3, dtype=np.float32),
            'rpy': np.zeros(3, dtype=np.float32),
            'temperature': 0,
        }

    def _convert(self, msg: LowState) -> dict:
        motors = msg.motor_state
        count = self._count
        imu = msg.imu_state
        return {
            'q': np.fromiter((motors[i].q for i in range(count)), np.float32, count),
            'dq': np.fromiter((motors[i].dq for i in range(count)), np.float32, count),
            'quaternion': np.array(imu.quaternion, dtype=np.float32),
            'gyroscope': np.array(imu.gyroscope, dtype=np.float32),
            'accelerometer': np.array(imu.accelerometer, dtype=np.float32),
            'rpy': np.array(imu.rpy, dtype=np.float32),
            'temperature': int(imu.temperature),
        }

    @property
    def state(self) -> dict:
        with self._lock:
            # each message is converted once, on the first read after it arrived
            if self._latest is not None and self._converted_sequence != self._sequence:
                self._arrays = self._convert(self._latest[0])
                self._converted_sequence = self._sequence
            arrays = self._arrays
        return {
            'q': arrays['q'].copy(),
            'dq': arrays['dq'].copy(),
            'imu_state': SimpleNamespace(
                quaternion=arrays['quaternion'].copy(),
                gyroscope=arrays['gyroscope'].copy(),
                accelerometer=arrays['accelerometer'].copy(),
                rpy=arrays['rpy'].copy(),
                temperature=arrays['temperature'],
            ),
        }

    @property
    def lag_ms(self) -> float | None:
        with self._lock:
            return self._lag_ms

    def age_s(self) -> float | None:
        '''Seconds since the newest sample arrived, or None before the first'''
        with self._lock:
            latest = self._latest
        return None if latest is None else time.monotonic() - latest[1]

    @property
    def staleness_ms(self) -> float:
        with self._lock:
            latest, lag_ms = self._latest, self._lag_ms
        if latest is None:
            return float('inf')
        return max(lag_ms, (time.monotonic() - latest[1]) * 1000.0)

    def check_fresh(self, max_age_s: float) -> None:
        '''Raise StaleStateError unless the newest sample arrived within max_age_s'''
        age = self.age_s()
        if age is None:
            raise StaleStateError(f'no {self._topic} received yet')
        if age > max_age_s:
            raise StaleStateError(f'{self._topic} is {age:.3f} s old (> {max_age_s} s)')

    def shutdown(self) -> None:
        '''Take no more samples; latest() and state keep the last one'''
        self._subscription.close()


class CommandPublisher:
    '''
    Publishes the newest command to a golem_safety input at pub_hz, from its own thread.

    set(q, dq, tau, kp, kd) gives the command of the profile's upper slots, one value each, in
    slot order (rad, rad/s, N*m and the gains); enabled=False sends mode 0 (motor off) on
    them, and one bool per upper slot sets each slot's mode. Nothing is published before the
    first set(). Every message carries the profile's mode_pr and mode_machine; its lower
    slots and the slots past the body are mode 0 with every value 0, and its CRC is 0
    (golem_safety computes the CRC of what it writes).

    The thread keeps to absolute deadlines on time.monotonic(). If it fails, set() raises.
    stop() makes sure the newest command has been published once, then stops.
    '''

    def __init__(self, node, topic: str, pub_hz: float, profile: RobotProfile):
        if not str(topic).startswith(SAFETY_INPUT_PREFIX):
            raise ValueError(f"refusing to publish commands on '{topic}': the controller "
                             f'writes only golem_safety inputs ({SAFETY_INPUT_PREFIX}...)')
        pub_hz = float(pub_hz)
        if not math.isfinite(pub_hz) or pub_hz <= 0.0:
            raise ValueError(f'pub_hz must be a positive rate, not {pub_hz}')
        self._node = node
        self._topic = topic
        self._period = 1.0 / pub_hz
        self._slots = tuple(profile.upper_slots)
        self._msg = LowCmd()
        self._msg.mode_pr = int(profile.mode_pr)
        self._msg.mode_machine = int(profile.mode_machine)
        self._publisher = node.create_publisher(LowCmd, topic, COMMAND_QOS)

        self._lock = threading.Lock()
        self._staged = None  # (mode, q, dq, tau, kp, kd) of the upper slots, as lists
        self._version = 0
        self._written_version = 0  # the staged command the message holds
        self._published_version = 0
        self._error = None
        self._stopped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name='command_publisher', daemon=True)
        self._thread.start()

    @property
    def topic(self) -> str:
        return self._topic

    def set(self, q, dq, tau, kp, kd, enabled=True) -> None:
        '''
        The command of the upper slots from now on; enabled is a bool for every upper slot or
        one per upper slot (mode 1 or 0). ValueError on a bad value
        '''
        count = len(self._slots)
        modes = np.asarray(enabled)
        if modes.dtype != np.bool_ or modes.shape not in ((), (count,)):
            raise ValueError(f'enabled must be a bool or {count} bools (the upper slots), '
                             f'not {enabled!r}')
        modes = np.broadcast_to(modes, (count,)).astype(np.int64).tolist()
        fields = []
        for name, values in (('q', q), ('dq', dq), ('tau', tau), ('kp', kp), ('kd', kd)):
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (count,):
                raise ValueError(f'{name} must hold {count} values (the upper slots), '
                                 f'not shape {values.shape}')
            if not np.all(np.abs(values) <= FLOAT32_MAX):
                raise ValueError(f'{name} must be finite float32 values, not {values}')
            fields.append(values.tolist())
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f'the command publisher on {self._topic} failed: '
                                   f'{self._error!r}')
            if self._stopped:
                raise RuntimeError(f'the command publisher on {self._topic} is stopped')
            self._staged = (modes, *fields)
            self._version += 1

    def _write(self, staged) -> None:
        mode, q, dq, tau, kp, kd = staged
        motors = self._msg.motor_cmd
        for k, slot in enumerate(self._slots):
            motor = motors[slot]
            motor.mode = mode[k]
            motor.q = q[k]
            motor.dq = dq[k]
            motor.tau = tau[k]
            motor.kp = kp[k]
            motor.kd = kd[k]

    def _publish_newest(self) -> None:
        with self._lock:
            staged, version = self._staged, self._version
        if staged is None:
            return
        if version != self._written_version:
            self._write(staged)
            self._written_version = version
        self._publisher.publish(self._msg)
        self._published_version = version

    def _run(self) -> None:
        period = self._period
        next_at = time.monotonic()
        try:
            while not self._stop.is_set():
                self._publish_newest()
                next_at += period
                now = time.monotonic()
                if now - next_at > MAX_CATCH_UP_S:
                    next_at = now
                time.sleep(max(0.0, next_at - now))
        except Exception as error:  # reported by set() and logged; publishing has stopped
            with self._lock:
                self._error = error
            self._node.get_logger().error(
                f'the command publisher on {self._topic} stopped: {error!r}')

    def stop(self) -> None:
        '''Publish the newest command if it has not gone out yet, then stop publishing'''
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        self._stop.set()
        self._thread.join()
        try:
            if self._error is None and self._published_version != self._version:
                self._publish_newest()
        finally:
            self._node.destroy_publisher(self._publisher)


class SafetyOutputSubscriber:
    '''
    The newest command golem_safety relays on its output topic (/lowcmd or /arm_sdk): latest()
    is (the LowCmd, its receive time on time.monotonic()), or None before the first. Only
    read: the controller never writes golem_safety's output.
    '''

    def __init__(self, node, topic: str, callback_group=None):
        outputs = sorted(set(SAFETY_OUTPUT_TOPICS.values()))
        if topic not in outputs:
            raise ValueError(f"'{topic}' is not a golem_safety output ({', '.join(outputs)})")
        self._topic = topic
        self._lock = threading.Lock()
        self._latest = None
        self._subscription = _ClosableSubscription(
            node, LowCmd, topic, self._receive, callback_group)

    @property
    def topic(self) -> str:
        return self._topic

    def _receive(self, msg: LowCmd) -> None:
        now = time.monotonic()
        with self._lock:
            self._latest = (msg, now)

    def latest(self):
        with self._lock:
            return self._latest

    def shutdown(self) -> None:
        '''Take no more samples'''
        self._subscription.close()
