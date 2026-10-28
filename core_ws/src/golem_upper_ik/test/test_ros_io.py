'''
The controller's ROS I/O on domain 92 (golem_upper_ik.core.ros_io): StateSubscriber against a
fake /lowstate, including its tick-lag measurement and the RobotModel state-source protocol,
and CommandPublisher's rate and header.
'''
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
from rclpy.qos import HistoryPolicy, ReliabilityPolicy

from ros_harness import FakeRobot, Recorder, TEST_DOMAIN, wait_for

RATE_TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'command_publisher_rate.py'
COMMAND_TOPIC = '/safety/lowcmd_upper_in'


def profile_of(robot):
    if robot == 'g1':
        pytest.importorskip('g1_description')
    from golem_upper_ik.profile import load_profile
    return load_profile(robot)


def subscribed(ros, profile, robot):
    '''A StateSubscriber on a controller node that has received robot's first sample'''
    from golem_upper_ik.core.ros_io import StateSubscriber
    subscriber = StateSubscriber(ros.node('controller'), profile)
    ros.on_close(subscriber.shutdown)
    assert wait_for(lambda: robot.publish() or subscriber.latest() is not None, 2.0)
    return subscriber


def test_state_subscriber_keeps_newest(ros):
    profile = profile_of('h12')
    robot = FakeRobot(ros, profile)
    subscriber = subscribed(ros, profile, robot)
    # keep-last 1, best effort: a late reader gets the newest sample, never a queue
    [endpoint] = subscriber_endpoints(ros, '/lowstate')
    assert endpoint.qos_profile.history == HistoryPolicy.KEEP_LAST
    assert endpoint.qos_profile.depth == 1
    assert endpoint.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT

    for k in range(100):
        robot.publish(q=np.full(len(profile.body_joints), 0.01 * k), tick=5000 + k)
    assert wait_for(lambda: subscriber.latest()[0].tick == 5099, 2.0)
    msg, received = subscriber.latest()
    assert received <= time.monotonic()
    np.testing.assert_array_equal(subscriber.state['q'], np.float32(0.99))


def subscriber_endpoints(ros, topic):
    '''The subscriptions to topic in the test's context'''
    node = ros.node('graph')
    assert wait_for(lambda: len(node.get_subscriptions_info_by_topic(topic)) > 0, 2.0)
    return node.get_subscriptions_info_by_topic(topic)


def test_check_fresh_raises_when_stale(ros):
    from golem_upper_ik.core.ros_io import StaleStateError, StateSubscriber
    profile = profile_of('h12')
    subscriber = StateSubscriber(ros.node('controller'), profile)
    ros.on_close(subscriber.shutdown)
    with pytest.raises(StaleStateError, match='no /lowstate received'):
        subscriber.check_fresh(0.1)
    assert subscriber.staleness_ms == float('inf')

    robot = FakeRobot(ros, profile)
    assert wait_for(lambda: robot.publish() or subscriber.latest() is not None, 2.0)
    # the discovery loop above may still have a sample on its way; one writer's samples
    # arrive in order, so once this one is the newest none is left to come
    robot.publish(tick=9000)
    assert wait_for(lambda: subscriber.latest()[0].tick == 9000, 2.0)
    subscriber.check_fresh(0.1)
    time.sleep(0.35)
    with pytest.raises(StaleStateError, match=r'/lowstate is 0\.(3[5-9]|[4-9])[0-9]* s old'):
        subscriber.check_fresh(0.1)
    assert subscriber.staleness_ms >= 350.0
    assert issubclass(StaleStateError, RuntimeError)


def test_tick_lag_tracker():
    from golem_upper_ik.core.ros_io import TICK_WRAP, TickLagTracker
    tracker = TickLagTracker()
    # a steady stream (any constant offset between the clocks) shows no lag
    for k in range(100):
        assert tracker.update(1000 + 2 * k, 50.0 + 0.002 * k) == pytest.approx(0.0, abs=1e-6)
    # a sample that arrives 80 ms late; the baseline has risen by the 0.5 ms/s drift
    # allowance over the 82 ms since the last sample
    assert tracker.update(1200, 50.2 + 0.080) == pytest.approx(80.0 - 0.5 * 0.082, abs=1e-6)
    # back on time: no lag
    assert tracker.update(1202, 50.202) == pytest.approx(0.0, abs=1e-6)
    # the baseline follows clock drift of up to 0.5 ms per s: a host clock running 0.2 ms/s
    # fast never registers
    tracker = TickLagTracker()
    lags = [tracker.update(1000 * k, 1.0 * k * 1.0002) for k in range(1, 100)]
    assert max(lags) == pytest.approx(0.0, abs=1e-6)
    # the uint32 tick wraps without a jump in lag
    tracker = TickLagTracker()
    tracker.update(TICK_WRAP - 2, 10.0)
    assert tracker.update(0, 10.002) == pytest.approx(0.0, abs=1e-6)


def test_state_subscriber_measures_tick_lag(ros):
    profile = profile_of('h12')
    robot = FakeRobot(ros, profile)
    subscriber = subscribed(ros, profile, robot)
    for _ in range(50):
        robot.publish()
        time.sleep(0.002)
    assert wait_for(lambda: subscriber.lag_ms is not None and subscriber.lag_ms < 20.0, 1.0)
    # a sample stamped now, delivered 100 ms later
    tick = robot.tick_now()
    time.sleep(0.1)
    robot.publish(tick=tick)
    assert wait_for(lambda: subscriber.latest()[0].tick == tick, 1.0)
    assert 70.0 < subscriber.lag_ms < 200.0, subscriber.lag_ms
    # staleness is the larger of the lag and the time since the newest sample
    assert subscriber.staleness_ms >= subscriber.lag_ms


def test_state_subscriber_is_a_state_source(ros, models):
    from golem_upper_ik.core.robot_model import StateSource
    profile = profile_of('h12')
    model = models('h12')
    robot = FakeRobot(ros, profile)
    subscriber = subscribed(ros, profile, robot)
    assert isinstance(subscriber, StateSource)

    count = len(profile.body_joints)
    q = np.clip(0.02 * np.arange(count) - 0.2, profile.q_low, profile.q_high)
    dq = -0.5 * q
    robot.publish(q=q, dq=dq, tick=7000)
    assert wait_for(lambda: subscriber.latest()[0].tick == 7000, 2.0)

    state = subscriber.state
    assert set(state) == {'q', 'dq', 'imu_state'}
    for key in ('q', 'dq'):
        assert state[key].shape == (count,) and state[key].dtype == np.float32, key
    np.testing.assert_array_equal(state['imu_state'].quaternion, [1.0, 0.0, 0.0, 0.0])
    # a read is the reader's own: changing it changes no later read
    state['q'][:] = 9.0
    state['imu_state'].quaternion[:] = 0.0
    np.testing.assert_array_equal(subscriber.state['q'], q.astype(np.float32))
    np.testing.assert_array_equal(subscriber.state['imu_state'].quaternion, [1, 0, 0, 0])

    model.state_subscriber = subscriber
    try:
        np.testing.assert_array_equal(model.state['q'], q.astype(np.float32))
        np.testing.assert_array_equal(model.state_reduced['q'],
                                      q.astype(np.float32)[model.reduced_mask])
        np.testing.assert_array_equal(model.state_reduced['dq'],
                                      dq.astype(np.float32)[model.reduced_mask])
        model.update_kinematics()
        frame = profile.ee_frames[0]
        placed = model.data_body_reduced.oMf[model.model_body_reduced.getFrameId(frame)]
        expected = model.get_frame_transformation_reduced(
            frame, q.astype(np.float32).astype(np.float64)[model.reduced_mask])
        np.testing.assert_allclose(placed.homogeneous, expected, atol=1e-12)
        assert 0.0 <= model.state_staleness_ms < 1000.0
    finally:
        model.state_subscriber = None


def test_state_before_the_first_sample_is_at_rest(ros):
    from golem_upper_ik.core.ros_io import StateSubscriber
    profile = profile_of('h12')
    subscriber = StateSubscriber(ros.node('controller'), profile)
    ros.on_close(subscriber.shutdown)
    assert subscriber.latest() is None
    assert subscriber.lag_ms is None
    state = subscriber.state
    np.testing.assert_array_equal(state['q'], np.zeros(len(profile.body_joints)))
    np.testing.assert_array_equal(state['imu_state'].quaternion, np.zeros(4))


def test_state_subscriber_shutdown_stops_taking_samples(ros):
    from rclpy.callback_groups import ReentrantCallbackGroup
    from golem_upper_ik.core.ros_io import StateSubscriber
    profile = profile_of('h12')
    node = ros.node('controller')
    # the subscription is destroyed inside its own callback, which needs its group busy
    with pytest.raises(ValueError, match='mutually exclusive'):
        StateSubscriber(node, profile, callback_group=ReentrantCallbackGroup())
    robot = FakeRobot(ros, profile)
    subscriber = subscribed(ros, profile, robot)
    robot.publish(tick=8000)
    assert wait_for(lambda: subscriber.latest()[0].tick == 8000, 2.0)
    subscriber.shutdown()
    robot.start()  # samples keep coming while the executor spins (Ros.close checks it still does)
    time.sleep(0.2)
    assert subscriber.latest()[0].tick == 8000
    # and the subscription is gone from the graph
    assert wait_for(lambda: not node.get_subscriptions_info_by_topic('/lowstate'), 2.0)


def measure_rate(*flags):
    '''The receive-side JSON of tools/command_publisher_rate.py at 500 Hz for 3 s'''
    env = dict(os.environ, ROS_DOMAIN_ID=str(TEST_DOMAIN))
    done = subprocess.run(
        [sys.executable, str(RATE_TOOL), '--rate', '500', '--seconds', '3', *flags],
        env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    stats = json.loads(done.stdout.strip().splitlines()[-1])
    print(json.dumps(stats))
    print(done.stderr.strip().splitlines()[-1])
    return stats


def test_command_publisher_rate():
    stats = measure_rate()
    assert stats['mean_hz'] >= 495, stats
    assert stats['p99_period_ms'] <= 4.0, stats
    assert stats['max_gap_ms'] <= 20.0, stats


def test_command_publisher_rate_with_the_gate_thread():
    '''LowCmdHandler with its follow gate shut: the 1 kHz gate check runs beside the publisher'''
    stats = measure_rate('--gate')
    assert stats['gate_shut'] is True, stats
    assert stats['mean_hz'] >= 495, stats
    assert stats['p99_period_ms'] <= 4.0, stats
    assert stats['max_gap_ms'] <= 20.0, stats


@pytest.mark.parametrize('robot, mode_machine', [('h12', 6), ('g1', 5)])
def test_header_is_profile(ros, robot, mode_machine):
    from golem_upper_ik.core.ros_io import CommandPublisher
    profile = profile_of(robot)
    assert profile.mode_machine == mode_machine
    recorder = Recorder(ros, COMMAND_TOPIC)
    publisher = CommandPublisher(ros.node('controller'), COMMAND_TOPIC, 500.0, profile)
    ros.on_close(publisher.stop)
    # nothing goes out before the first command
    time.sleep(0.3)
    assert recorder.count == 0

    upper = list(profile.upper_slots)
    n = len(upper)
    q = np.linspace(-0.5, 0.5, n)
    dq = np.linspace(0.0, 0.1, n)
    tau = np.linspace(-2.0, 2.0, n)
    kp = np.full(n, 60.0)
    kd = np.linspace(1.0, 2.0, n)
    publisher.set(q, dq, tau, kp, kd)
    assert wait_for(lambda: recorder.count > 10, 2.0)
    msg = recorder.last
    assert msg.mode_machine == mode_machine
    assert msg.mode_pr == profile.mode_pr == 0
    assert msg.crc == 0  # golem_safety stamps the CRC
    for k, slot in enumerate(upper):
        motor = msg.motor_cmd[slot]
        assert motor.mode == 1
        np.testing.assert_array_equal(
            [motor.q, motor.dq, motor.tau, motor.kp, motor.kd],
            np.float32([q[k], dq[k], tau[k], kp[k], kd[k]]))
    for slot in set(range(len(msg.motor_cmd))) - set(upper):
        motor = msg.motor_cmd[slot]
        assert (motor.mode, motor.q, motor.dq, motor.tau, motor.kp, motor.kd) == (0, 0, 0, 0, 0, 0)

    # disabled: mode 0 on the upper slots
    publisher.set(q, dq, np.zeros(n), np.zeros(n), np.zeros(n), enabled=False)
    assert wait_for(lambda: recorder.last.motor_cmd[upper[0]].mode == 0, 2.0)
    # a mode per upper slot
    enabled = np.arange(n) % 2 == 0
    publisher.set(q, dq, tau, kp, kd, enabled=enabled)
    assert wait_for(lambda: recorder.last.motor_cmd[upper[0]].mode == 1, 2.0)
    assert [recorder.last.motor_cmd[slot].mode for slot in upper] == enabled.astype(int).tolist()


def test_command_publisher_refuses_bad_commands(ros):
    from golem_upper_ik.core.ros_io import CommandPublisher
    profile = profile_of('h12')
    node = ros.node('controller')
    # the controller writes only golem_safety's inputs
    for topic in ('/lowcmd', '/arm_sdk', 'safety/lowcmd_upper_in'):
        with pytest.raises(ValueError, match='/safety/'):
            CommandPublisher(node, topic, 500.0, profile)
    with pytest.raises(ValueError, match='pub_hz'):
        CommandPublisher(node, COMMAND_TOPIC, 0.0, profile)
    publisher = CommandPublisher(node, COMMAND_TOPIC, 500.0, profile)
    ros.on_close(publisher.stop)
    n = len(profile.upper_slots)
    good = [np.zeros(n)] * 5
    for field in range(5):
        for bad in (np.nan, np.inf, 1e39):
            values = [array.copy() for array in good]
            values[field][3] = bad
            with pytest.raises(ValueError):
                publisher.set(*values)
    with pytest.raises(ValueError, match=str(n)):
        publisher.set(np.zeros(n + 1), *good[1:])
    for enabled in ([True] * (n - 1), 1, [1] * n):
        with pytest.raises(ValueError, match='enabled'):
            publisher.set(*good, enabled=enabled)
    publisher.stop()
    with pytest.raises(RuntimeError, match='stopped'):
        publisher.set(*good)
