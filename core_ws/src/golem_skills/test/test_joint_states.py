"""Test that SkillsBase keeps /joint_states per joint, merging its publishers."""

import pytest
import rclpy
from sensor_msgs.msg import JointState

from golem_skills import base
from test_heartbeat_halt import FakeClock
from test_robot_time import _node


def _js(names, positions):
    """Return a JointState of `names` at `positions`."""
    msg = JointState()
    msg.name = list(names)
    msg.position = [float(p) for p in positions]
    return msg


@pytest.fixture
def node(tmp_path, monkeypatch):
    """Yield (a SkillsBase without use_sim_time, the fake time.monotonic it reads)."""
    n = _node(tmp_path, False)
    steady = FakeClock()
    monkeypatch.setattr(base.time, 'monotonic', steady)
    yield n, steady
    n.destroy_node()
    rclpy.shutdown()


def test_two_publishers_merge(node):
    """The body's and the hand driver's messages both stay in the joint state."""
    n, _ = node
    assert n.latest_joint_state() is None
    n._on_joint_state(_js(['left_elbow_joint', 'right_elbow_joint'], [0.5, -0.5]))
    n._on_joint_state(_js(['left_hand_index_0_joint'], [-0.3]))
    js = n.latest_joint_state()
    assert dict(zip(js.name, js.position)) == {
        'left_elbow_joint': 0.5, 'right_elbow_joint': -0.5, 'left_hand_index_0_joint': -0.3}
    # a later message updates its own joints only
    n._on_joint_state(_js(['left_elbow_joint'], [0.7]))
    js = n.latest_joint_state()
    assert dict(zip(js.name, js.position))['left_elbow_joint'] == 0.7
    assert dict(zip(js.name, js.position))['right_elbow_joint'] == -0.5


def test_joint_positions_since_wants_every_joint_newer(node):
    """Only positions published after `since` count, and every named joint must have one."""
    n, steady = node
    n._on_joint_state(_js(['a', 'b'], [1.0, 2.0]))
    since = n.robot_now()
    steady.advance(0.1)
    n._on_joint_state(_js(['a'], [1.5]))
    assert n.joint_positions_since(['a', 'b'], since, 0.0) is None
    assert n.joint_positions_since(['a'], since, 0.0) == {'a': 1.5}
    assert n.joint_positions_since(['c'], since, 0.0) is None
    steady.advance(0.1)
    n._on_joint_state(_js(['b'], [2.5]))
    assert n.joint_positions_since(['a', 'b'], since, 0.0) == {'a': 1.5, 'b': 2.5}
