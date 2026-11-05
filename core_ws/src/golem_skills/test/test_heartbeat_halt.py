"""The arms halt on a false or silent /safety/heartbeat."""

import os
import threading
import time

from custom_ros_messages.action import NamedConfig
from golem_skills.base import SkillsBase
import pytest
import rclpy
from rclpy.task import Future
from std_msgs.msg import Bool

TEST_DOMAIN = 96
DDS_CONFIG = """<CycloneDDS><Domain Id="any">
  <General>
    <Interfaces><NetworkInterface name="lo"/></Interfaces>
    <AllowMulticast>false</AllowMulticast>
  </General>
  <Discovery>
    <ParticipantIndex>auto</ParticipantIndex>
    <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
    <Peers><Peer address="127.0.0.1"/></Peers>
  </Discovery>
</Domain></CycloneDDS>
"""
SILENT = 'safety heartbeat silent for '
NOT_YET = 'safety heartbeat not received yet'
LATCHED = 'golem_safety estop latched'


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeGoalHandle:
    def __init__(self, result_future):
        self.accepted = True
        self.canceled = threading.Event()
        self._result = result_future

    def get_result_async(self):
        return self._result

    def cancel_goal_async(self):
        self.canceled.set()
        future = Future()
        future.set_result(None)
        return future


class FakeNamedConfigClient:
    """An action client whose goal never finishes by itself."""

    def __init__(self):
        self.goals = []
        self.handle = FakeGoalHandle(Future())

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        future = Future()
        future.set_result(self.handle)
        return future


@pytest.fixture
def skills(tmp_path):
    config = tmp_path / 'cyclonedds.xml'
    config.write_text(DDS_CONFIG)
    os.environ['CYCLONEDDS_URI'] = 'file://' + str(config)
    os.environ['ROS_DOMAIN_ID'] = str(TEST_DOMAIN)
    os.environ['ROS_LOCALHOST_ONLY'] = '0'
    rclpy.init(args=['--ros-args', '-p', 'wait_for_services:=false',
                     '-p', 'record_runs:=false'])
    node = SkillsBase('heartbeat_halt_test')
    node._monotonic = FakeClock()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def _beat(node, value):
    node._on_safety_heartbeat(Bool(data=value))


def test_halted_until_the_first_true_beat(skills):
    assert skills.arm_halt_reason() == NOT_YET
    _beat(skills, True)
    assert skills.arm_halt_reason() is None


def test_halted_after_a_silent_heartbeat(skills):
    _beat(skills, True)
    skills._monotonic.advance(0.4)
    assert skills.arm_halt_reason() is None
    skills._monotonic.advance(0.2)
    assert skills.arm_halt_reason() == SILENT + '0.6 s'
    _beat(skills, True)
    assert skills.arm_halt_reason() is None


def test_the_silence_limit_is_a_parameter(skills):
    skills.set_parameters([rclpy.parameter.Parameter(
        'heartbeat_timeout', rclpy.parameter.Parameter.Type.DOUBLE, 2.0)])
    _beat(skills, True)
    skills._monotonic.advance(1.9)
    assert skills.arm_halt_reason() is None
    skills._monotonic.advance(0.2)
    assert skills.arm_halt_reason() == SILENT + '2.1 s'


def test_halted_for_good_after_a_false_beat(skills):
    _beat(skills, True)
    _beat(skills, False)
    assert skills.arm_halt_reason() == LATCHED
    _beat(skills, True)
    assert skills.arm_halt_reason() == LATCHED


def test_a_false_first_beat_latches(skills):
    _beat(skills, False)
    assert skills.arm_halt_reason() == LATCHED


def test_a_latched_estop_wins_over_a_silent_heartbeat(skills):
    _beat(skills, False)
    skills._monotonic.advance(5.0)
    assert skills.arm_halt_reason() == LATCHED


def test_no_arm_goal_is_sent_while_halted(skills):
    client = FakeNamedConfigClient()
    skills.named_config_cli = client
    assert skills.goto_named_config('home') is False
    assert client.goals == []


def test_a_silent_heartbeat_cancels_the_arm_goal_in_flight(skills):
    client = FakeNamedConfigClient()
    skills.named_config_cli = client
    _beat(skills, True)
    outcome = []
    caller = threading.Thread(
        target=lambda: outcome.append(skills.goto_named_config('home')))
    caller.start()
    deadline = time.monotonic() + 5.0
    while not client.goals and time.monotonic() < deadline:
        time.sleep(0.01)
    assert isinstance(client.goals[0], NamedConfig.Goal)
    assert caller.is_alive() and not client.handle.canceled.is_set()

    skills._monotonic.advance(1.0)

    caller.join(timeout=5.0)
    assert not caller.is_alive()
    assert outcome == [False]
    assert client.handle.canceled.is_set()
    _beat(skills, True)
    assert skills.arm_halt_reason().startswith(SILENT)


def test_a_false_beat_cancels_the_arm_goal_in_flight(skills):
    client = FakeNamedConfigClient()
    skills.named_config_cli = client
    _beat(skills, True)
    outcome = []
    caller = threading.Thread(
        target=lambda: outcome.append(skills.goto_named_config('home')))
    caller.start()
    deadline = time.monotonic() + 5.0
    while not client.goals and time.monotonic() < deadline:
        time.sleep(0.01)

    _beat(skills, False)

    caller.join(timeout=5.0)
    assert outcome == [False]
    assert client.handle.canceled.is_set()
    assert skills.arm_halt_reason() == LATCHED


def test_waits_that_are_not_arm_goals_ignore_the_heartbeat(skills):
    future = Future()
    threading.Timer(0.3, lambda: future.set_result('done')).start()
    assert skills._wait_future(future, 5.0) == 'done'
