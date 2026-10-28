'''
The ROS side of the tests, on domain 92: one rclpy context per test with an executor
spinning its nodes in a thread (Ros), a fake robot publishing /lowstate (FakeRobot), a
recorder of a command topic (Recorder) and a stand-in for golem_safety that relays the
controller's commands on its output topic (Echo).

Every node lives in the test's context, so they share one DDS participant and discover each
other at once.
'''
import itertools
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_hg.msg import LowCmd, LowState

TEST_DOMAIN = 92
# keeps every sample: the recorder must see each command the controller sends
RECORD_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST, depth=10000)
# a writer as the Unitree SDK's and golem_safety's: reliable, keep-last 1
WRITER_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST, depth=1)
_names = itertools.count()


def wait_for(condition, timeout, period=0.005):
    '''True as soon as condition() is, False once timeout s have passed without it'''
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(period)
    return bool(condition())


class Ros:
    def __init__(self):
        self.context = rclpy.Context()
        rclpy.init(context=self.context, domain_id=TEST_DOMAIN)
        self.executor = MultiThreadedExecutor(num_threads=4, context=self.context)
        self._nodes = []
        self._closers = []
        self.spin_error = None
        self._spinner = threading.Thread(target=self._spin, daemon=True)
        self._spinner.start()

    def _spin(self):
        try:
            self.executor.spin()
        except Exception as error:  # close() fails the test with it
            self.spin_error = error

    def node(self, name):
        node = rclpy.create_node(f'{name}_{next(_names)}', context=self.context)
        self._nodes.append(node)
        self.executor.add_node(node)
        return node

    def on_close(self, close):
        '''Run close() when the test ends, before the nodes go; the last registered runs first'''
        self._closers.append(close)

    def close(self):
        errors = []
        for close in reversed(self._closers):
            try:
                close()
            except Exception as error:  # every closer runs; the first error is raised
                errors.append(error)
        self.executor.shutdown(timeout_sec=2.0)
        self._spinner.join(timeout=2.0)
        for node in self._nodes:
            node.destroy_node()
        rclpy.shutdown(context=self.context)
        if self.spin_error is not None:
            raise AssertionError(f'the executor stopped spinning: {self.spin_error!r}')
        if errors:
            raise errors[0]


class FakeRobot:
    '''
    Publishes /lowstate for a profile: q, dq and tick on demand (publish) or at a fixed rate
    from a thread (start), with the profile's mode_machine and a level IMU
    '''

    def __init__(self, ros, profile):
        self.profile = profile
        self.node = ros.node('fake_robot')
        self._publisher = self.node.create_publisher(
            LowState, '/lowstate',
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=100))
        count = len(profile.body_joints)
        self.q = np.zeros(count)
        self.dq = np.zeros(count)
        self._start = time.monotonic()
        self._stop = threading.Event()
        self._thread = None
        ros.on_close(self.stop)

    def tick_now(self):
        '''The robot's millisecond tick: ms since this fake started, from 1000'''
        return 1000 + int((time.monotonic() - self._start) * 1000.0)

    def publish(self, q=None, dq=None, tick=None):
        q = self.q if q is None else np.asarray(q, dtype=float)
        dq = self.dq if dq is None else np.asarray(dq, dtype=float)
        msg = LowState()
        msg.mode_pr = self.profile.mode_pr
        msg.mode_machine = self.profile.mode_machine
        msg.tick = self.tick_now() if tick is None else int(tick)
        msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        for slot in range(len(q)):
            motor = msg.motor_state[slot]
            motor.mode = 1
            motor.q = float(q[slot])
            motor.dq = float(dq[slot])
            motor.tau_est = 0.5 * float(slot)
        self._publisher.publish(msg)

    def start(self, rate_hz=200.0):
        '''Publish the current q and dq at rate_hz until stop()'''
        period = 1.0 / rate_hz

        def run():
            next_at = time.monotonic()
            while not self._stop.is_set():
                self.publish()
                next_at += period
                time.sleep(max(0.0, next_at - time.monotonic()))
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class Recorder:
    '''Every LowCmd on a topic, with its receive time on time.monotonic()'''

    def __init__(self, ros, topic):
        self.node = ros.node('recorder')
        self._lock = threading.Lock()
        self._samples = []
        self.node.create_subscription(LowCmd, topic, self._on_cmd, RECORD_QOS)

    def _on_cmd(self, msg):
        with self._lock:
            self._samples.append((time.monotonic(), msg))

    @property
    def samples(self):
        with self._lock:
            return list(self._samples)

    @property
    def count(self):
        with self._lock:
            return len(self._samples)

    @property
    def last(self):
        with self._lock:
            return self._samples[-1][1] if self._samples else None

    def since(self, start):
        '''The messages received at or after monotonic time start'''
        return [msg for t, msg in self.samples if t >= start]


class Echo:
    '''
    Plays golem_safety relaying the controller: every command on in_topic is published on
    out_topic, with weight in weight_slot when it is given (the /arm_sdk blend weight)
    '''

    def __init__(self, ros, in_topic, out_topic, weight_slot=None, weight=1.0):
        self.node = ros.node('echo')
        self.weight_slot = weight_slot
        self.weight = weight
        self._publisher = self.node.create_publisher(LowCmd, out_topic, WRITER_QOS)
        self.node.create_subscription(LowCmd, in_topic, self._relay, RECORD_QOS)

    def _relay(self, msg):
        if self.weight_slot is not None:
            msg.motor_cmd[self.weight_slot].q = float(self.weight)
        self._publisher.publish(msg)


def upper_q(msg, profile):
    '''The upper-slot q of a LowCmd, as float64'''
    return np.array([msg.motor_cmd[slot].q for slot in profile.upper_slots], dtype=np.float64)
