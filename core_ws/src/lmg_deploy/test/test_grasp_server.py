"""Test /grasp_frame_target's server and the grasp-frame pose topics against a stand-in controller, over ROS.

    ROS_DOMAIN_ID=<not 0> python -m pytest test/test_grasp_server.py

GraspGoalServer runs on a node of its own as marl_deploy_node runs it; the test plays the control tick (step() with
its own robot time) on the H1-2 kinematics, and a client sends goals. The test publishes on the DDS domain, so it is
skipped on domain 0 (the real robots' bus).
"""

import os
import threading
import time

import numpy as np
import pytest
import rclpy
from custom_ros_messages.action import GraspFrameTarget
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import StaticTransformBroadcaster

from lmg_deploy import rotations as R
from lmg_deploy.grasp_goal import GRASP_FRAMES
from lmg_deploy.grasp_server import GraspGoalServer
from lmg_deploy.kinematics import ArmKinematics
from shipped_policy import URDF, shipped_cfg

Result = GraspFrameTarget.Result  # the outcome constants
TOLERANCE = {"tolerance_m": 0.01, "tolerance_rad": 0.1, "settle_s": 0.2, "default_timeout_s": 1.0}
DT = 0.01   # s of the test's robot time per tick


class StandInController:
    """Record the arm goals and holds the server asks for, as MarlController's interface takes them."""

    def __init__(self):
        """Start with no command."""
        self.believed = np.zeros((2, 7))
        self.goals, self.holds = [], []

    def request_arm_goal(self, poses, arms=None):
        """Record a goal: its wrist poses and its arms."""
        self.goals.append((np.array(poses), list(arms)))

    def hold_arms(self, q, arms):
        """Record a hold of the arms."""
        self.holds.append(list(arms))


class Rig:
    """Hold a control node with the server, a client node, the kinematics and the test's robot time."""

    def __init__(self, ready=lambda: None):
        """Start both nodes; the client node spins on its own thread, the server on its."""
        rclpy.init()
        cfg = shipped_cfg()
        names = cfg["joint_names"]
        self.kin = ArmKinematics(URDF, names, cfg["arms"]["ee_frames"], [names[13:20], names[20:]],
                                 extra_frames=list(GRASP_FRAMES))
        self.q = np.array(cfg["default_joint_pos"], dtype=np.float64)
        self.kin.update(self.q)
        self.node = rclpy.create_node("grasp_server_test_control")
        self.server = GraspGoalServer(self.node, self.kin, ready=ready, **TOLERANCE)
        self.ctrl = StandInController()
        self.client_node = rclpy.create_node("grasp_server_test_client")
        self.client = ActionClient(self.client_node, GraspFrameTarget, "/grasp_frame_target")
        self.poses = {}
        for arm, topic in enumerate(("/left_grasp_frame_pose", "/right_grasp_frame_pose")):
            self.client_node.create_subscription(PoseStamped, topic,
                                                 lambda msg, arm=arm: self.poses.__setitem__(arm, msg), 10)
        self.tf = StaticTransformBroadcaster(self.client_node)
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.client_node)
        self.executor.add_node(self.node)
        self.spinner = threading.Thread(target=self.executor.spin, daemon=True)
        self.spinner.start()
        self.now = 100.0
        assert self.client.wait_for_server(timeout_sec=10.0)

    def grasp_pose(self, arm):
        """Return the arm's grasp-frame pose (position, quaternion wxyz) at the rig's joints."""
        return self.kin.frame_poses(self.q, list(GRASP_FRAMES))[arm]

    def send(self, targets, frame_id="pelvis", timeout=0.0, feedback=None):
        """Send a goal {arm: (position, quaternion wxyz)}; return its goal handle."""
        goal = GraspFrameTarget.Goal()
        for arm, (pos, quat) in targets.items():
            stamped = goal.left if arm == 0 else goal.right
            stamped.header.frame_id = frame_id
            stamped.pose.position.x, stamped.pose.position.y, stamped.pose.position.z = (float(v) for v in pos)
            o = stamped.pose.orientation
            o.x, o.y, o.z, o.w = (float(v) for v in R.wxyz_to_xyzw(quat))
        goal.timeout.sec, goal.timeout.nanosec = int(timeout), int(round((timeout % 1.0) * 1e9))
        future = self.client.send_goal_async(goal, feedback_callback=feedback)
        assert self._wait(future.done, 10.0)
        handle = future.result()
        assert handle.accepted
        return handle

    def tick(self, seconds, until=lambda: False):
        """Tick the server for seconds of robot time (real-time paced), stopping early once until()."""
        for _ in range(int(round(seconds / DT))):
            if until():
                return
            self.server.step(self.now, self.ctrl, self.q)
            self.now += DT
            time.sleep(DT)

    def run(self, handle, seconds):
        """Tick the server for up to seconds of robot time, until the goal ends; return its result."""
        result = handle.get_result_async()
        self.tick(seconds, until=result.done)
        assert self._wait(result.done, 5.0)
        return result.result()

    @staticmethod
    def _wait(cond, seconds):
        """Wait up to seconds (wall) for cond()."""
        end = time.monotonic() + seconds
        while time.monotonic() < end and not cond():
            time.sleep(0.01)
        return cond()

    def close(self):
        """Stop the server and both nodes and shut rclpy down."""
        self.server.shutdown()
        self.executor.shutdown()
        self.client_node.destroy_node()
        self.node.destroy_node()
        rclpy.shutdown()


@pytest.fixture
def rig():
    """Yield a Rig, skipped on domain 0 (the real robots' bus)."""
    if os.environ.get("ROS_DOMAIN_ID", "0") in ("", "0"):
        pytest.skip("publishes on ROS: set ROS_DOMAIN_ID to a non-zero domain")
    r = Rig()
    try:
        yield r
    finally:
        r.close()


def test_a_goal_on_the_grasp_frame_succeeds_after_the_settle_and_sends_its_wrist_goal(rig):
    """A one-arm goal sends the wrist goal that puts the grasp frame on the target for that arm alone, publishes its
    errors (0 for the other arm) and succeeds once within the tolerance for settle_s."""
    pos, quat = rig.grasp_pose(0)
    feedback = []
    handle = rig.send({0: (pos, quat)}, feedback=lambda m: feedback.append(m.feedback))
    wrapped = rig.run(handle, 2.0)
    assert wrapped.status == 4 and wrapped.result.success, wrapped.result.message   # SUCCEEDED
    assert wrapped.result.outcome == Result.REACHED
    assert wrapped.result.message.startswith("reached")
    poses, arms = rig.ctrl.goals[0]
    assert arms == [0]
    np.testing.assert_allclose(rig.server.frames.grasp_pose(0, poses[0, :3], poses[0, 3:])[0], pos, atol=1e-9)
    assert feedback and feedback[-1].left_error_m < 1e-6 and feedback[-1].right_error_m == 0.0
    assert rig.ctrl.holds == []


def test_a_goal_off_reach_times_out_and_holds_its_arm(rig):
    """A target the arm does not reach aborts at the timeout (robot time) and stops the targeted arm where it is."""
    pos, quat = rig.grasp_pose(1)
    start = rig.now
    wrapped = rig.run(rig.send({1: (pos + [0.1, 0.0, 0.0], quat)}, timeout=0.5), 2.0)
    assert wrapped.status == 6 and "timed out after 0.5 s" in wrapped.result.message   # ABORTED
    assert wrapped.result.outcome == Result.NOT_REACHED and not wrapped.result.success
    assert rig.now - start == pytest.approx(0.5, abs=2 * DT)
    assert rig.ctrl.holds == [[1]]


def test_a_cancel_holds_the_targeted_arms(rig):
    """A cancel mid-goal ends it canceled and stops its arms where they are."""
    pos, quat = rig.grasp_pose(0)
    handle = rig.send({0: (pos + [0.1, 0.0, 0.0], quat), 1: rig.grasp_pose(1)}, timeout=5.0)
    rig.tick(0.2)
    handle.cancel_goal_async()
    wrapped = rig.run(handle, 1.0)
    assert wrapped.status == 5 and "canceled" in wrapped.result.message   # CANCELED
    assert wrapped.result.outcome == Result.CANCELED and not wrapped.result.success
    assert rig.ctrl.holds == [[0, 1]]


def test_a_world_frame_target_is_resolved_through_tf_and_a_missing_frame_aborts(rig):
    """A target in another frame is taken into the pelvis frame through TF when the goal starts; a frame TF does not
    have aborts the goal, naming it, and moves nothing."""
    pos, quat = rig.grasp_pose(0)
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = "pelvis", "test_world"
    t.transform.translation.x, t.transform.rotation.z, t.transform.rotation.w = 0.5, np.sin(0.2), np.cos(0.2)
    rig.tf.sendTransform(t)
    q_pw = np.array([np.cos(0.2), 0.0, 0.0, np.sin(0.2)])
    world_pos = R.quat_apply(R.quat_conjugate(q_pw), pos - np.array([0.5, 0.0, 0.0]))
    world_quat = R.quat_mul(R.quat_conjugate(q_pw), quat)
    wrapped = rig.run(rig.send({0: (world_pos, world_quat)}, frame_id="test_world"), 2.0)
    assert wrapped.status == 4, wrapped.result.message
    wrapped = rig.run(rig.send({0: (pos, quat)}, frame_id="nowhere"), 2.0)
    assert wrapped.status == 6 and "nowhere" in wrapped.result.message
    assert wrapped.result.outcome == Result.FAILED
    assert len(rig.ctrl.goals) == 1


def test_a_queued_goal_canceled_before_it_starts_ends_canceled(rig):
    """A goal canceled while another runs ends CANCELED without moving its arm."""
    pos, quat = rig.grasp_pose(0)
    first = rig.send({0: (pos + [0.1, 0.0, 0.0], quat)}, timeout=5.0)
    second = rig.send({1: rig.grasp_pose(1)}, timeout=5.0)
    rig.tick(0.1)
    second.cancel_goal_async()
    wrapped = rig.run(second, 1.0)
    assert wrapped.status == 5 and wrapped.result.outcome == Result.CANCELED   # CANCELED
    assert wrapped.result.message.startswith("canceled before it started")
    assert [arms for _, arms in rig.ctrl.goals] == [[0]]
    rig.server.abort_all("test over")
    rig.run(first, 0.1)


def test_goals_run_one_at_a_time_in_order(rig):
    """A goal sent while another runs waits for it; abort_all ends both."""
    pos, quat = rig.grasp_pose(0)
    first = rig.send({0: (pos + [0.1, 0.0, 0.0], quat)}, timeout=5.0)
    second = rig.send({1: rig.grasp_pose(1)}, timeout=5.0)
    rig.tick(0.1)
    assert [arms for _, arms in rig.ctrl.goals] == [[0]]
    rig.server.abort_all("test fault")
    for handle in (first, second):
        wrapped = rig.run(handle, 0.1)
        assert wrapped.status == 6 and wrapped.result.message == "test fault"
        assert wrapped.result.outcome == Result.FAILED


def test_the_grasp_frame_poses_are_published_in_the_pelvis_frame(rig):
    """publish_poses() sends each grasp frame's pose at the joints, frame pelvis."""
    rig.server.publish_poses(rig.q, rig.node.get_clock().now().to_msg())
    assert rig._wait(lambda: len(rig.poses) == 2, 5.0)
    for arm in range(2):
        msg = rig.poses[arm]
        assert msg.header.frame_id == "pelvis"
        pos, quat = rig.grasp_pose(arm)
        p, o = msg.pose.position, msg.pose.orientation
        np.testing.assert_allclose([p.x, p.y, p.z], pos, atol=1e-12)
        np.testing.assert_allclose([o.w, o.x, o.y, o.z], quat, atol=1e-12)


def test_a_goal_the_node_refuses_aborts_with_its_reason():
    """While ready() gives a reason (the node not engaged), a goal aborts with it and moves nothing."""
    if os.environ.get("ROS_DOMAIN_ID", "0") in ("", "0"):
        pytest.skip("publishes on ROS: set ROS_DOMAIN_ID to a non-zero domain")
    rig = Rig(ready=lambda: "the policy is not engaged yet")
    try:
        wrapped = rig.run(rig.send({0: rig.grasp_pose(0)}), 1.0)
        assert wrapped.status == 6 and wrapped.result.message == "the policy is not engaged yet"
        assert wrapped.result.outcome == Result.FAILED
        assert rig.ctrl.goals == []
    finally:
        rig.close()
