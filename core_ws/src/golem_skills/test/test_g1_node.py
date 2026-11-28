"""Test a SkillsBase started with robot:=g1: its hand clients, grasp frames and arm goals."""

import os

import numpy as np
import pytest
import rclpy
from rclpy.time import Time

from golem_skills.base import SkillsBase
from golem_skills.perception_utils import (matrix_to_pose, pose_to_matrix,
                                           transform_to_matrix)
from test_heartbeat_halt import DDS_CONFIG

TEST_DOMAIN = 94


@pytest.fixture
def g1(tmp_path):
    """Yield a SkillsBase with robot:=g1 on a loopback test domain."""
    config = tmp_path / 'cyclonedds.xml'
    config.write_text(DDS_CONFIG)
    os.environ['CYCLONEDDS_URI'] = 'file://' + str(config)
    os.environ['ROS_DOMAIN_ID'] = str(TEST_DOMAIN)
    os.environ['ROS_LOCALHOST_ONLY'] = '0'
    rclpy.init(args=['--ros-args', '-p', 'wait_for_services:=false',
                     '-p', 'record_runs:=false', '-p', 'robot:=g1'])
    node = SkillsBase('g1_skills_test')
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_g1_node_uses_the_dex3_driver(g1):
    """No Magpie clients; open and close clients on dex_3's /<arm>/hand/ services."""
    assert g1.robot.name == 'g1'
    assert g1.gripper_clis == {} and g1.gripper_close_clis == {}
    assert {a: c.srv_name for a, c in g1.hand_close_clis.items()} == {
        'left': '/left/hand/close', 'right': '/right/hand/close'}
    assert {a: c.srv_name for a, c in g1.hand_open_clis.items()} == {
        'left': '/left/hand/open', 'right': '/right/hand/open'}


def _lookup(node, target, source):
    """Spin `node` until its TF buffer has `source` in `target`; return the 4x4 transform."""
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.tf_buffer.can_transform(target, source, Time()):
            break
    return transform_to_matrix(node.tf_buffer.lookup_transform(target, source, Time()).transform)


def test_g1_grasp_frames_are_on_tf_static(g1):
    """golem_upper_ik's grasp frame on the palm and the convention frame on it are on TF."""
    from golem_upper_ik.profile import load_profile
    left = load_profile('g1').grasp_frames[0]
    np.testing.assert_allclose(_lookup(g1, 'left_hand_palm_link', 'left_grasp_frame'),
                               left.offset, atol=1e-6)
    np.testing.assert_allclose(
        _lookup(g1, 'left_grasp_frame', 'left_dex3_grasp_frame'),
        g1.robot.tool_frames['left_dex3_grasp_frame'].matrix(), atol=1e-6)


def test_g1_grasp_frame_moves_target_the_pinch_frame(g1):
    """A move of the convention frame to T sends the left grasp frame to T inv(offset)."""
    sent = []

    def fake_send(_client, goal, **_kw):
        """Record the goal and report no result."""
        sent.append(goal)
        return None

    _lookup(g1, 'left_dex3_grasp_frame', 'left_grasp_frame')
    g1._send_action = fake_send
    g1._last_beat = g1._monotonic()
    T = np.eye(4)
    T[:3, 3] = (0.35, 0.15, 0.05)
    g1.move_frame_to('left_dex3_grasp_frame', matrix_to_pose(T), duration_sec=1.0)
    assert len(sent) == 1
    goal = sent[0]
    assert goal.left.header.frame_id == 'pelvis' and goal.right.header.frame_id == ''
    offset = g1.robot.tool_frames['left_dex3_grasp_frame'].matrix()
    np.testing.assert_allclose(pose_to_matrix(goal.left.pose) @ offset, T, atol=1e-9)
