"""Test Dex3Hand against a fake hand over DDS inside one process (no robot, no network).

By default the driver keeps the steady clock; given a robot clock (a sim's, under
use_sim_time), its moves count that clock's time.
"""

import gc
import time
from types import SimpleNamespace

import numpy as np
import pytest

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_ as HandStateDefault
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from dex_3 import driver
from dex_3.driver import RELEASE_MESSAGES, Dex3Hand
from dex_3.dex3 import DEFAULT_KD, DEFAULT_KP, DEFAULT_MAX_SPEED, UPPER, motor_mode, named_pose

TEST_DOMAIN = 97  # SAFETY: never 0 (the real robot), nor 1 (the sim)

# The current test's fake-hand state publishers and command recorders.
_ENTITIES = []


@pytest.fixture(scope='module', autouse=True)
def dds():
    """Join Unitree DDS on the test domain once for the module."""
    ChannelFactoryInitialize(TEST_DOMAIN)


@pytest.fixture(autouse=True)
def close_entities():
    """Close each test's DDS entities in the main thread.

    Left to the garbage collector (a pytest.raises keeps them in a reference
    cycle), they can be deleted from inside a DDS listener thread, which
    crashes the process.
    """
    yield
    while _ENTITIES:
        _ENTITIES.pop().Close()
    gc.collect()


def _state_publisher(side, q):
    """Return a fake hand's state publisher and a HandState_ at joint positions q."""
    pub = ChannelPublisher(f'rt/dex3/{side}/state', HandState_)
    pub.Init()
    _ENTITIES.append(pub)
    msg = HandStateDefault()
    for i, value in enumerate(q):
        msg.motor_state[i].q = float(value)
    return pub, msg


class _Recorder:
    """Record rt/dex3/<side>/cmd messages with their arrival times (time.monotonic())."""

    def __init__(self, side):
        """Subscribe to the side's command topic."""
        self.messages = []
        self.times = []
        sub = ChannelSubscriber(f'rt/dex3/{side}/cmd', HandCmd_)
        sub.Init(self._on_cmd, 50)
        _ENTITIES.append(sub)

    def _on_cmd(self, msg):
        """Store a command with its arrival time."""
        self.times.append(time.monotonic())
        self.messages.append(msg)

    def __len__(self):
        """Return the number of commands received."""
        return len(self.messages)

    def since(self, n):
        """Return the positions (k x 7) and arrival times of messages n onward."""
        k = len(self.messages)
        return _positions(self.messages[n:k]), np.array(self.times[n:k])


def _publish_state(pub, msg, n=20):
    """Publish the hand state n times, 10 ms apart."""
    for _ in range(n):
        pub.Write(msg)
        time.sleep(0.01)


def _positions(messages):
    """Return the commanded positions of messages, one row each."""
    return np.array([[m.motor_cmd[i].q for i in range(7)] for m in messages])


def _is_stop(message):
    """Return whether a command is Unitree's stop: timeout bit set, zero gains."""
    return all(m.mode == motor_mode(i, timeout=True) and m.kp == 0.0 and m.kd == 0.0
               for i, m in enumerate(message.motor_cmd[:7]))


def _stops_after(messages):
    """Return the stop commands in messages, asserting nothing else follows the first.

    A hold command already on its way can still arrive just before them.
    """
    first = next((i for i, m in enumerate(messages) if _is_stop(m)), len(messages))
    assert all(_is_stop(m) for m in messages[first:]), 'a command after the stop'
    return messages[first:]


def _wait_until(condition, timeout=2.0):
    """Poll condition every 10 ms until it holds or timeout (s) passes."""
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)


def _assert_within_speed_cap(qs, times, q_from, t_from, max_speed=DEFAULT_MAX_SPEED):
    """Assert no command is farther from q_from than max_speed allows since t_from.

    t_from is taken just before the set_pose call. A message arrives after it
    was computed, so a loaded CPU cannot fail this; a jump fails it at once.
    """
    ahead = np.abs(qs - q_from).max(axis=1) - max_speed * (times - t_from)
    assert ahead.max() < 1e-3, f'{ahead.max():.3f} rad ahead of the {max_speed} rad/s cap'


def test_needs_state_before_the_first_command():
    """A first command without hand state is refused."""
    hand = Dex3Hand('right')
    try:
        with pytest.raises(RuntimeError, match='no recent right hand state'):
            hand.set_pose(named_pose('right', 'close'))
    finally:
        hand.shutdown()


def test_silent_then_smooth_move_hold_and_release():
    """The driver is silent until commanded, moves within the cap, clips, holds, then releases."""
    start = np.array([0.1, 0.0, 0.2, -0.3, 0.0, 0.0, 0.0])
    state_pub, state = _state_publisher('left', start)
    received = _Recorder('left')
    hand = Dex3Hand('left', rate_hz=200.0)
    try:
        _publish_state(state_pub, state)
        assert hand.state is not None and np.allclose(hand.state['q'], start, atol=1e-6)
        assert len(received) == 0, 'must stay silent until the first command'
        assert hand.release() is False
        time.sleep(0.1)
        assert len(received) == 0

        goal = named_pose('left', 'close')
        t_cmd = time.monotonic()
        hand.set_pose(goal, duration=0.5)
        time.sleep(0.8)
        qs, times = received.since(0)
        # The move starts from the measured pose.
        _assert_within_speed_cap(qs, times, start, t_cmd)
        assert np.allclose(qs[-1], goal, atol=1e-6), 'ends at the goal'
        last = received.messages[-1]
        assert [m.mode for m in last.motor_cmd] == [motor_mode(i) for i in range(7)]
        assert all(abs(m.kp - DEFAULT_KP) < 1e-6 and abs(m.kd - DEFAULT_KD) < 1e-6
                   for m in last.motor_cmd)

        # Out-of-range targets are clipped to the URDF limits, and a 0 s move
        # still goes no faster than max_speed (1.22 rad here: about 0.61 s).
        n = len(received)
        t_cmd = time.monotonic()
        hand.set_pose([9.0] * 7, duration=0.0)
        time.sleep(0.8)
        qs, times = received.since(n)
        _assert_within_speed_cap(qs, times, goal, t_cmd)
        assert np.allclose(qs[-1], UPPER['left'])

        n = len(received)
        assert hand.release() is True
        _wait_until(lambda: len(_stops_after(received.messages[n:])) >= RELEASE_MESSAGES)
        count = len(received)
        assert len(_stops_after(received.messages[n:])) == RELEASE_MESSAGES
        time.sleep(0.1)
        assert len(received) == count, 'silent after the stop'
    finally:
        hand.shutdown()


def test_zero_duration_first_command_is_capped_by_max_speed():
    """A 0 s first command far from the goal moves no faster than max_speed."""
    start = np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
    goal = np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0])
    expected = 1.3 / DEFAULT_MAX_SPEED  # s, middle_0's 1.3 rad at the cap
    state_pub, state = _state_publisher('right', start)
    received = _Recorder('right')
    hand = Dex3Hand('right', rate_hz=200.0)
    try:
        _publish_state(state_pub, state)
        t_cmd = time.monotonic()
        hand.set_pose(goal, duration=0.0)
        time.sleep(expected + 0.3)
        qs, times = received.since(0)
        _assert_within_speed_cap(qs, times, start, t_cmd)
        assert np.allclose(qs[-1], goal, atol=1e-6)
        first_at_goal = times[np.argmax(np.abs(qs[:, 3] - 1.5) < 1e-6)]
        assert first_at_goal - t_cmd > expected, 'reached the goal no sooner than the cap allows'
    finally:
        hand.shutdown()


def test_refuses_non_finite_state_and_bad_durations():
    """A non-finite hand state or a bad duration is refused and publishes nothing."""
    state_pub, state = _state_publisher('left', [0.0, np.nan, 0, 0, 0, 0, 0])
    received = _Recorder('left')
    hand = Dex3Hand('left', rate_hz=200.0)
    try:
        _publish_state(state_pub, state)
        with pytest.raises(RuntimeError, match='non-finite'):
            hand.set_pose(named_pose('left', 'open'), duration=1.0)

        state.motor_state[1].q = 0.0
        _publish_state(state_pub, state)
        for bad in (float('nan'), float('inf'), -1.0):
            with pytest.raises(ValueError, match='duration'):
                hand.set_pose(named_pose('left', 'close'), duration=bad)
        time.sleep(0.1)
        assert len(received) == 0, 'refused commands publish nothing'
    finally:
        hand.shutdown()


@pytest.mark.parametrize('kwargs', [
    {'kp': -1.0}, {'kd': float('nan')}, {'max_speed': 0.0}, {'max_speed': float('inf')},
    {'rate_hz': 0.0},
])
def test_rejects_bad_settings(kwargs):
    """Negative, non-finite or zero settings are refused."""
    with pytest.raises(ValueError, match='must be finite'):
        Dex3Hand('left', **kwargs)


def test_a_wall_clock_step_does_not_skip_a_move(monkeypatch):
    """A wall-clock step during a move, as from NTP, does not shorten it."""
    # 1 s for 1.22 rad is below the speed cap, so only a skipped move can fail this.
    offset = [0.0]
    monkeypatch.setattr(driver, 'time', SimpleNamespace(
        time=lambda: time.time() + offset[0], monotonic=time.monotonic, sleep=time.sleep))
    start = np.zeros(7)
    state_pub, state = _state_publisher('left', start)
    received = _Recorder('left')
    hand = Dex3Hand('left', rate_hz=200.0)
    try:
        _publish_state(state_pub, state)
        goal = named_pose('left', 'close')
        t_cmd = time.monotonic()
        hand.set_pose(goal, duration=1.0)
        time.sleep(0.1)
        offset[0] = 100.0
        time.sleep(1.2)
        qs, times = received.since(0)
        _assert_within_speed_cap(qs, times, start, t_cmd, max_speed=1.23)
        assert np.allclose(qs[-1], goal, atol=1e-6)
    finally:
        hand.shutdown()


@pytest.mark.parametrize('release', [True, False])
def test_shutdown_releases_or_just_goes_silent(release):
    """Shutdown sends Unitree's stop with release, and only goes silent without."""
    state_pub, state = _state_publisher('right', np.zeros(7))
    received = _Recorder('right')
    hand = Dex3Hand('right', rate_hz=200.0)
    try:
        _publish_state(state_pub, state)
        hand.set_pose(named_pose('right', 'close'), duration=0.2)  # 0.61 s at max_speed
        time.sleep(0.9)
        n = len(received)
        assert n > 0 and not any(_is_stop(m) for m in received.messages)
    except BaseException:
        hand.shutdown()
        raise
    hand.shutdown(release=release)
    # The recorder receives messages on its own thread, so give it a moment.
    _wait_until(lambda: len(_stops_after(received.messages[n:])) >= RELEASE_MESSAGES,
                timeout=1.0 if release else 0.2)
    after = received.messages[n:]
    if release:
        assert len(_stops_after(after)) == RELEASE_MESSAGES
    else:
        # The last command the hand got is the held close pose.
        assert not any(_is_stop(m) for m in after)
        assert np.allclose(_positions(received.messages[-1:])[0], named_pose('right', 'close'),
                           atol=1e-6)
    count = len(received)
    time.sleep(0.1)
    assert len(received) == count, 'silent after shutdown'


class ManualClock:
    """Stand in for a sim's robot clock: the test sets its time; a sleep waits for it."""

    def __init__(self):
        """Start at 100 s."""
        self.t = 100.0

    def now(self):
        """Return the clock's time, s."""
        return self.t

    def sleep(self, seconds, stop):
        """Wait until the clock has moved seconds on, or stop() is true."""
        end = self.t + seconds
        while self.t < end and not stop():
            time.sleep(0.001)


def test_a_move_counts_the_robot_clock():
    """With a robot clock, a move's duration is that clock's time, not the wall clock's."""
    clock = ManualClock()
    start = np.zeros(7)
    goal = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])  # middle_0, inside the right limits
    state_pub, state = _state_publisher('right', start)
    received = _Recorder('right')
    hand = Dex3Hand('right', rate_hz=200.0, clock=clock)
    try:
        _publish_state(state_pub, state)
        hand.set_pose(goal, duration=1.0)
        # 0.5 s of wall time with the robot clock stood still: the hand stays at the start
        time.sleep(0.5)
        qs, _ = received.since(0)
        assert len(qs) >= 1 and np.allclose(qs[-1], start, atol=1e-6)
        clock.t += 0.5
        _wait_until(lambda: np.allclose(received.since(0)[0][-1], goal / 2, atol=1e-6))
        assert np.allclose(received.since(0)[0][-1], goal / 2, atol=1e-6), 'halfway'
        clock.t += 0.6
        _wait_until(lambda: np.allclose(received.since(0)[0][-1], goal, atol=1e-6))
        assert np.allclose(received.since(0)[0][-1], goal, atol=1e-6)
    finally:
        hand.shutdown()


def test_a_stopped_robot_clock_never_holds_up_the_release():
    """With the robot clock stood still mid-move, the release still sends Unitree's stop."""
    clock = ManualClock()
    state_pub, state = _state_publisher('right', np.zeros(7))
    received = _Recorder('right')
    hand = Dex3Hand('right', rate_hz=200.0, clock=clock)
    try:
        _publish_state(state_pub, state)
        hand.set_pose(named_pose('right', 'close'), duration=1.0)
        time.sleep(0.2)
        n = len(received)
        assert hand.release() is True
        _wait_until(lambda: len(_stops_after(received.messages[n:])) >= RELEASE_MESSAGES)
        assert len(_stops_after(received.messages[n:])) == RELEASE_MESSAGES
    finally:
        start = time.monotonic()
        hand.shutdown()
        assert time.monotonic() - start < 0.5, 'shutdown waited on the stopped clock'
