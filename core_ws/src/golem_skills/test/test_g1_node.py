"""Test a SkillsBase started with robot:=g1: its hand clients, tool frames and frame_task goals."""

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


def test_g1_tool_frames_are_on_tf_static(g1):
    """The node's own TF listener resolves the Dex3-1 grasp frame from its palm link."""
    for _ in range(50):
        rclpy.spin_once(g1, timeout_sec=0.1)
        if g1.tf_buffer.can_transform('left_hand_palm_link', 'left_dex3_grasp_frame',
                                      Time()):
            break
    tf = g1.tf_buffer.lookup_transform('left_hand_palm_link', 'left_dex3_grasp_frame',
                                       Time())
    np.testing.assert_allclose(
        transform_to_matrix(tf.transform),
        g1.robot.tool_frames['left_dex3_grasp_frame'].matrix(), atol=1e-6)


def test_g1_grasp_frame_moves_go_to_the_palm(g1):
    """A frame_task goal for the grasp frame drives the palm link to T inv(offset)."""
    sent = []

    def fake_send(_client, goal, **_kw):
        """Record the goal and report no result."""
        sent.append(goal)
        return None

    g1._send_action = fake_send
    g1._last_beat = g1._monotonic()
    T = np.eye(4)
    T[:3, 3] = (0.35, 0.15, 0.05)
    g1.move_frame_to('left_dex3_grasp_frame', matrix_to_pose(T), duration_sec=1.0)
    assert len(sent) == 1
    assert list(sent[0].frame_names) == ['left_hand_palm_link']
    offset = g1.robot.tool_frames['left_dex3_grasp_frame'].matrix()
    np.testing.assert_allclose(pose_to_matrix(sent[0].frame_targets[0]) @ offset, T,
                               atol=1e-9)
