"""
Test that skill deadlines and arm-move waits count robot time, and the heartbeat steady time.

Robot time is the node's clock under use_sim_time, here moved by hand as /clock moves it,
and the steady clock (time.monotonic) otherwise, faked here.
"""

import os
import threading
import time

from golem_skills import base
from golem_skills.base import SkillsBase, _Run
from custom_ros_messages.action import SkillGrasp
import pytest
import rclpy
from rclpy.clock import ClockType
from rclpy.task import Future
from rclpy.time import Time
from std_msgs.msg import Bool

from test_heartbeat_halt import DDS_CONFIG, SILENT, FakeClock, FakeNamedConfigClient

TEST_DOMAIN = 95


class FakeSkillGoal:
    """Stand in for a skill's goal handle: its request carries the timeout, never canceled."""

    def __init__(self, timeout_sec):
        """Hold a SkillGrasp request with timeout_sec."""
        self.request = SkillGrasp.Goal()
        self.request.timeout.sec = int(timeout_sec)
        self.request.timeout.nanosec = int((timeout_sec - int(timeout_sec)) * 1e9)
        self.is_cancel_requested = False
        self.aborted = False

    def abort(self):
        """Record the abort."""
        self.aborted = True

    def publish_feedback(self, _feedback):
        """Drop the feedback."""


def _node(tmp_path, sim_time):
    """Return a SkillsBase on the test domain, with use_sim_time as sim_time."""
    config = tmp_path / 'cyclonedds.xml'
    config.write_text(DDS_CONFIG)
    os.environ['CYCLONEDDS_URI'] = 'file://' + str(config)
    os.environ['ROS_DOMAIN_ID'] = str(TEST_DOMAIN)
    os.environ['ROS_LOCALHOST_ONLY'] = '0'
    rclpy.init(args=['--ros-args', '-p', 'wait_for_services:=false',
                     '-p', 'record_runs:=false',
                     '-p', f'use_sim_time:={"true" if sim_time else "false"}'])
    return SkillsBase('robot_time_test')


class SimClock:
    """Move a use_sim_time node's clock by hand, as /clock moves it."""

    def __init__(self, node):
        """Start the node's clock at 100 s."""
        self._clock = node.get_clock()
        self.now = 100.0
        self.advance(0.0)

    def advance(self, seconds):
        """Move the clock forward by seconds."""
        self.now += seconds
        self._clock.set_ros_time_override(
            Time(nanoseconds=int(self.now * 1e9), clock_type=ClockType.ROS_TIME))


@pytest.fixture
def sim(tmp_path):
    """Yield (a use_sim_time SkillsBase, its SimClock, its fake steady clock)."""
    node = _node(tmp_path, True)
    steady = FakeClock()
    node._monotonic = steady
    yield node, SimClock(node), steady
    node.destroy_node()
    rclpy.shutdown()


@pytest.fixture
def real(tmp_path, monkeypatch):
    """Yield (a SkillsBase without use_sim_time, the fake time.monotonic it reads)."""
    node = _node(tmp_path, False)
    steady = FakeClock()
    monkeypatch.setattr(base.time, 'monotonic', steady)
    yield node, steady
    node.destroy_node()
    rclpy.shutdown()


def test_a_skill_deadline_counts_sim_time(sim):
    """Under use_sim_time a skill's timeout is s of sim time."""
    node, clock, steady = sim
    goal = FakeSkillGoal(1.0)
    run = _Run(node, goal, SkillGrasp, 'grasp')
    steady.advance(100.0)  # the steady clock plays no part
    clock.advance(0.9)
    assert run.phase('detect', 0.1) and run.remaining() == pytest.approx(0.1)
    clock.advance(0.2)
    assert not run.phase('detect', 0.2)
    assert goal.aborted and run.result.message == 'skill timeout'


def test_a_skill_deadline_counts_the_steady_clock_without_sim_time(real):
    """Without use_sim_time a skill's timeout is s of the steady clock, as on the robots."""
    node, steady = real
    goal = FakeSkillGoal(1.0)
    run = _Run(node, goal, SkillGrasp, 'grasp')
    steady.advance(0.9)
    assert run.phase('detect', 0.1)
    steady.advance(0.2)
    assert not run.phase('detect', 0.2) and goal.aborted


def test_an_arm_move_wait_counts_sim_time(sim):
    """Under use_sim_time an arm move's result wait ends after its timeout of sim time."""
    node, clock, _ = sim
    client = FakeNamedConfigClient()
    node.named_config_cli = client
    outcome = []
    caller = threading.Thread(target=lambda: outcome.append(
        node._send_action(client, 'goal', result_timeout=1.0)))
    caller.start()
    time.sleep(1.5)
    # 1.5 s of wall time with /clock stood still: still waiting
    assert caller.is_alive() and not client.handle.canceled.is_set()
    clock.advance(0.9)
    time.sleep(0.3)
    assert caller.is_alive()
    clock.advance(0.2)
    caller.join(timeout=2.0)
    assert not caller.is_alive() and outcome == [None]
    assert client.handle.canceled.is_set()


def test_a_service_wait_counts_the_steady_clock_under_sim_time(sim):
    """Under use_sim_time a service call's wait still counts steady time: host latency."""
    node, _, _ = sim
    start = time.monotonic()
    assert node._wait_future(Future(), 0.3) is None
    assert 0.3 <= time.monotonic() - start < 0.6


def test_the_heartbeat_counts_steady_time_under_sim_time(sim):
    """SAFETY: under use_sim_time, with /clock stood still, a silent heartbeat halts the arms."""
    node, clock, steady = sim
    node._on_safety_heartbeat(Bool(data=True))
    clock.advance(5.0)  # sim time plays no part
    assert node.arm_halt_reason() is None
    steady.advance(0.6)
    assert node.arm_halt_reason() == SILENT + '0.6 s'


def test_the_grip_hold_period_counts_sim_time(sim):
    """Under use_sim_time the grip-hold loop's period is s of sim time."""
    node, clock, _ = sim
    stop = threading.Event()
    done = []
    waiter = threading.Thread(target=lambda: done.append(node._sleep_robot(0.2, stop)))
    waiter.start()
    time.sleep(0.5)
    assert waiter.is_alive(), 'the period ended with /clock stood still'
    clock.advance(0.25)
    waiter.join(timeout=1.0)
    assert done == [False]
    waiter = threading.Thread(target=lambda: done.append(node._sleep_robot(10.0, stop)))
    waiter.start()
    stop.set()
    waiter.join(timeout=1.0)
    assert done == [False, True]
