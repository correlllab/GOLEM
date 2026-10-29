'''
Dex3HandNode's DDS domain and shutdown choice, with the hands and DDS faked.
Needs rclpy: run with the sourced workspace on PYTHONPATH (see README)
'''

import os
import signal
import threading
import time

import pytest

rclpy = pytest.importorskip('rclpy')

from dex_3 import hand_node  # noqa: E402
from dex_3.dex3 import DEFAULT_MAX_SPEED  # noqa: E402
from dex_3.hand_node import Dex3HandNode  # noqa: E402

ROS_TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)


class FakeHand:
    '''Stands in for Dex3Hand: records how it was made and shut down'''
    instances = []
    state = None  # no hand state yet, as before a hand's first rt/dex3 state

    def __init__(self, side, **kwargs):
        self.side = side
        self.kwargs = kwargs
        self.shutdowns = []
        FakeHand.instances.append(self)

    def shutdown(self, release=True):
        self.shutdowns.append(release)


@pytest.fixture
def channel_inits(monkeypatch):
    '''ChannelFactoryInitialize calls; no DDS is started'''
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
    args = ['--ros-args'] + [arg for p in parameters for arg in ('-p', p)]
    # the node's own ROS side always runs on the test domain
    rclpy.init(args=args, domain_id=ROS_TEST_DOMAIN)
    return Dex3HandNode()


def test_refuses_to_start_without_a_domain(channel_inits):
    with pytest.raises(ValueError, match='no DDS domain'):
        _make_node()
    assert channel_inits == [] and FakeHand.instances == []


def test_domain_from_the_parameter(channel_inits):
    _make_node(f'domain_id:={ROS_TEST_DOMAIN}')
    assert channel_inits == [(ROS_TEST_DOMAIN,)]
    assert [hand.side for hand in FakeHand.instances] == ['left', 'right']
    assert all(hand.kwargs['max_speed'] == DEFAULT_MAX_SPEED for hand in FakeHand.instances)


def test_domain_and_interface_from_the_environment(channel_inits, monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', str(ROS_TEST_DOMAIN))
    monkeypatch.setenv('G1_DDS_INTERFACE', 'lo')
    _make_node()
    assert channel_inits == [(ROS_TEST_DOMAIN, 'lo')]


def test_refuses_a_parameter_that_disagrees_with_ros_domain_id(channel_inits, monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', str(ROS_TEST_DOMAIN))
    with pytest.raises(ValueError, match='disagree'):
        _make_node('domain_id:=1')
    assert channel_inits == []


def test_refuses_a_parameter_that_is_not_the_nodes_ros_domain(channel_inits):
    # as under ros2 launch with ROS_DOMAIN_ID unset and domain_id:=1, where
    # ROS itself runs on domain 0: ROS and Unitree DDS would split
    with pytest.raises(ValueError, match=f'runs ROS on domain {ROS_TEST_DOMAIN}'):
        _make_node(f'domain_id:={ROS_TEST_DOMAIN + 1}')
    assert channel_inits == [] and FakeHand.instances == []


def test_refuses_an_octal_ros_domain_id(channel_inits, monkeypatch):
    # rcl reads ROS_DOMAIN_ID with a leading 0 as octal: 0141 is domain 97
    monkeypatch.setenv('ROS_DOMAIN_ID', '0141')
    rclpy.init(args=[])
    with pytest.raises(ValueError, match=f'runs ROS on domain {ROS_TEST_DOMAIN} but would '
                                         'run Unitree DDS on domain 141'):
        Dex3HandNode()
    assert channel_inits == [] and FakeHand.instances == []


@pytest.mark.parametrize('parameters, release', [
    ((), True),  # default: release, the fingers go limp
    (('on_shutdown:=release',), True),
    (('on_shutdown:=silent',), False),
])
def test_on_shutdown(channel_inits, parameters, release):
    node = _make_node(f'domain_id:={ROS_TEST_DOMAIN}', *parameters)
    node.destroy_node()
    assert [hand.shutdowns for hand in FakeHand.instances] == [[release], [release]]


def test_rejects_an_unknown_on_shutdown(channel_inits):
    with pytest.raises(ValueError, match='on_shutdown'):
        _make_node(f'domain_id:={ROS_TEST_DOMAIN}', 'on_shutdown:=hold')
    assert channel_inits == [] and FakeHand.instances == []


@pytest.mark.parametrize('first', hand_node.EXIT_SIGNALS, ids=lambda s: signal.Signals(s).name)
def test_main_releases_once_through_repeated_exit_signals(channel_inits, monkeypatch, first):
    # Ctrl-C in a terminal reaches the node twice: from the terminal, and from
    # ros2 launch passing it on, which may follow with SIGTERM. No exit signal,
    # first or repeated, may cut the release short
    monkeypatch.setenv('ROS_DOMAIN_ID', str(ROS_TEST_DOMAIN))
    handlers = {signum: signal.getsignal(signum) for signum in hand_node.EXIT_SIGNALS}
    unhandled = []

    def send(signum):
        if signal.getsignal(signum) is signal.SIG_DFL:
            # main() does not handle it: sent, it would end the test process
            unhandled.append(signal.Signals(signum).name)
            signum = signal.SIGINT
        os.kill(os.getpid(), signum)

    def slow_shutdown(self, release=True):
        for signum in hand_node.EXIT_SIGNALS:  # more signals, mid-release
            send(signum)
        time.sleep(0.3)
        self.shutdowns.append(release)

    def first_signal():
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
