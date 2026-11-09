"""Test Dex3HandNode's DDS domain, shutdown choice and signals, with the hands and DDS faked.

Needs rclpy: run with the sourced workspace on PYTHONPATH (see README).
"""

import os
import signal
import threading
import time

import pytest

rclpy = pytest.importorskip('rclpy')

from dex_3 import hand_node  # noqa: E402
from dex_3.dex3 import DEFAULT_MAX_SPEED  # noqa: E402
from dex_3.hand_node import Dex3HandNode  # noqa: E402

ROS_TEST_DOMAIN = 97  # SAFETY: never 0 (the real robot), nor 1 (the sim)


class FakeHand:
    """Stand in for Dex3Hand, recording how it was made and shut down."""

    instances = []
    state = None  # as before a hand's first rt/dex3 state

    def __init__(self, side, **kwargs):
        """Record the side and settings."""
        self.side = side
        self.kwargs = kwargs
        self.shutdowns = []
        FakeHand.instances.append(self)

    def shutdown(self, release=True):
        """Record the release choice."""
        self.shutdowns.append(release)


@pytest.fixture
def channel_inits(monkeypatch):
    """Fake the hands and DDS, and yield the ChannelFactoryInitialize calls."""
    calls = []
    monkeypatch.setattr(hand_node, 'ChannelFactoryInitialize', lambda *args: calls.append(args))
    monkeypatch.setattr(hand_node, 'Dex3Hand', FakeHand)
    FakeHand.instances = []
    monkeypatch.delenv('ROS_DOMAIN_ID', raising=False)
    monkeypatch.delenv('G1_DDS_INTERFACE', raising=False)
    yield calls
    if rclpy.ok():
        rclpy.shutdown()


def _make_node(*parameters):
    """Make a Dex3HandNode with the given `name:=value` parameters, ROS on the test domain."""
    args = ['--ros-args'] + [arg for p in parameters for arg in ('-p', p)]
    rclpy.init(args=args, domain_id=ROS_TEST_DOMAIN)
    return Dex3HandNode()


def test_refuses_to_start_without_a_domain(channel_inits):
    """With no ROS_DOMAIN_ID and no domain_id the node refuses before any DDS or hand."""
    with pytest.raises(ValueError, match='no DDS domain'):
        _make_node()
    assert channel_inits == [] and FakeHand.instances == []


def test_domain_from_the_parameter(channel_inits):
    """domain_id sets the DDS domain, and both hands start with the default speed cap."""
    _make_node(f'domain_id:={ROS_TEST_DOMAIN}')
    assert channel_inits == [(ROS_TEST_DOMAIN,)]
    assert [hand.side for hand in FakeHand.instances] == ['left', 'right']
    assert all(hand.kwargs['max_speed'] == DEFAULT_MAX_SPEED for hand in FakeHand.instances)


def test_domain_and_interface_from_the_environment(channel_inits, monkeypatch):
    """ROS_DOMAIN_ID and G1_DDS_INTERFACE set the DDS domain and interface."""
    monkeypatch.setenv('ROS_DOMAIN_ID', str(ROS_TEST_DOMAIN))
    monkeypatch.setenv('G1_DDS_INTERFACE', 'lo')
    _make_node()
    assert channel_inits == [(ROS_TEST_DOMAIN, 'lo')]


def test_refuses_a_parameter_that_disagrees_with_ros_domain_id(channel_inits, monkeypatch):
    """A domain_id that disagrees with ROS_DOMAIN_ID is refused."""
    monkeypatch.setenv('ROS_DOMAIN_ID', str(ROS_TEST_DOMAIN))
    with pytest.raises(ValueError, match='disagree'):
        _make_node('domain_id:=1')
    assert channel_inits == []


def test_refuses_a_parameter_that_is_not_the_nodes_ros_domain(channel_inits):
    """A domain_id other than the node's ROS domain is refused, so ROS and DDS cannot split."""
    with pytest.raises(ValueError, match=f'runs ROS on domain {ROS_TEST_DOMAIN}'):
        _make_node(f'domain_id:={ROS_TEST_DOMAIN + 1}')
    assert channel_inits == [] and FakeHand.instances == []


def test_refuses_an_octal_ros_domain_id(channel_inits, monkeypatch):
    """A leading-0 ROS_DOMAIN_ID, which rcl reads as octal (0141 is 97), is refused."""
    monkeypatch.setenv('ROS_DOMAIN_ID', '0141')
    rclpy.init(args=[])
    with pytest.raises(ValueError, match=f'runs ROS on domain {ROS_TEST_DOMAIN} but would '
                                         'run Unitree DDS on domain 141'):
        Dex3HandNode()
    assert channel_inits == [] and FakeHand.instances == []


@pytest.mark.parametrize('parameters, release', [
    ((), True),
    (('on_shutdown:=release',), True),
    (('on_shutdown:=silent',), False),
])
def test_on_shutdown(channel_inits, parameters, release):
    """on_shutdown chooses release (the default) or silent for every hand."""
    node = _make_node(f'domain_id:={ROS_TEST_DOMAIN}', *parameters)
    node.destroy_node()
    assert [hand.shutdowns for hand in FakeHand.instances] == [[release], [release]]


def test_rejects_an_unknown_on_shutdown(channel_inits):
    """An unknown on_shutdown is refused before any DDS or hand."""
    with pytest.raises(ValueError, match='on_shutdown'):
        _make_node(f'domain_id:={ROS_TEST_DOMAIN}', 'on_shutdown:=hold')
    assert channel_inits == [] and FakeHand.instances == []


@pytest.mark.parametrize('first', hand_node.EXIT_SIGNALS, ids=lambda s: signal.Signals(s).name)
def test_main_releases_once_through_repeated_exit_signals(channel_inits, monkeypatch, first):
    """No exit signal, first or repeated, cuts the release short."""
    # Ctrl-C in a terminal reaches the node twice: from the terminal, and from
    # ros2 launch passing it on, which may follow with SIGTERM.
    monkeypatch.setenv('ROS_DOMAIN_ID', str(ROS_TEST_DOMAIN))
    handlers = {signum: signal.getsignal(signum) for signum in hand_node.EXIT_SIGNALS}
    unhandled = []

    def send(signum):
        """Send signum to this process, recording it instead if main() left it unhandled."""
        if signal.getsignal(signum) is signal.SIG_DFL:
            # Unhandled, it would end the test process.
            unhandled.append(signal.Signals(signum).name)
            signum = signal.SIGINT
        os.kill(os.getpid(), signum)

    def slow_shutdown(self, release=True):
        """Send every exit signal mid-release, then record the release."""
        for signum in hand_node.EXIT_SIGNALS:
            send(signum)
        time.sleep(0.3)
        self.shutdowns.append(release)

    def first_signal():
        """Send the first exit signal once the node has made its hands."""
        while not FakeHand.instances:
            time.sleep(0.01)
        time.sleep(0.3)
        send(first)

    monkeypatch.setattr(FakeHand, 'shutdown', slow_shutdown)
    threading.Thread(target=first_signal, daemon=True).start()
    try:
        hand_node.main(['--ros-args', '-p', 'sides:=[left]'])
    except KeyboardInterrupt:
        pytest.fail('a second signal interrupted the shutdown')
    finally:
        for signum, handler in handlers.items():
            if handler is not None:
                signal.signal(signum, handler)
    assert unhandled == []
    assert [hand.shutdowns for hand in FakeHand.instances] == [[True]]


@pytest.mark.parametrize('sim_time', [False, True])
def test_the_hands_keep_the_robot_clock(channel_inits, sim_time):
    """Under use_sim_time the hands get the node's clock; otherwise none (the steady clock)."""
    node = _make_node(f'domain_id:={ROS_TEST_DOMAIN}',
                      f'use_sim_time:={"true" if sim_time else "false"}')
    try:
        clocks = [hand.kwargs['clock'] for hand in FakeHand.instances]
        assert len(clocks) == 2
        if sim_time:
            assert all(clock is not None for clock in clocks)
        else:
            assert clocks == [None, None]
    finally:
        node.destroy_node()


def test_no_pose_before_the_first_clock(channel_inits):
    """Under use_sim_time a pose is refused before the first /clock: robot time has not begun."""
    node = _make_node(f'domain_id:={ROS_TEST_DOMAIN}', 'use_sim_time:=true')
    try:
        response = node._named_pose('left', 'close', None, hand_node.Trigger.Response())
        assert not response.success and 'first /clock' in response.message
    finally:
        node.destroy_node()


def test_sim_time_sleep_wakes_on_clock_and_on_stop(channel_inits):
    """A sim-time sleep ends when /clock reaches its end, or once stop() is true."""
    from rclpy.clock import ClockType
    from rclpy.time import Time
    node = _make_node(f'domain_id:={ROS_TEST_DOMAIN}', 'use_sim_time:=true')
    try:
        clock = node.get_clock()

        def set_time(seconds):
            """Set sim time, as a /clock message does."""
            clock.set_ros_time_override(Time(nanoseconds=int(seconds * 1e9),
                                             clock_type=ClockType.ROS_TIME))
        set_time(100.0)
        sim = hand_node.SimTime(node)
        done = threading.Event()
        threading.Thread(target=lambda: (sim.sleep(0.1, lambda: False), done.set()),
                         daemon=True).start()
        time.sleep(0.2)
        assert not done.is_set(), 'the sleep ended with /clock stood still'
        set_time(100.1)
        assert done.wait(0.05), 'the sleep did not wake on /clock'
        stop = threading.Event()
        done.clear()
        threading.Thread(target=lambda: (sim.sleep(10.0, stop.is_set), done.set()),
                         daemon=True).start()
        time.sleep(0.05)
        stop.set()
        assert done.wait(0.05), 'the sleep did not end on stop'
    finally:
        node.destroy_node()
