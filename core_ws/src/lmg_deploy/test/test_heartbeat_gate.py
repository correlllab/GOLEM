"""marl_deploy_node commands nothing until golem_safety's first true /safety/heartbeat.

    ROS_DOMAIN_ID=<not 0> python -m pytest test/test_heartbeat_gate.py

A real node runs with the shipped mid360v2 policy and the H1-2 URDF beside a test node that
plays /lowstate and the heartbeat and records what reaches the safety layer's inputs.
The test publishes on the DDS domain, so it is skipped on domain 0 (the real robots' bus).
"""

import os
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from unitree_hg.msg import LowCmd, LowState

from lmg_deploy.marl_deploy_node import MarlDeployNode

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY = os.path.join(HERE, "..", "policies", "mid360v2", "mid360v2.yaml")
URDF = os.environ.get(
    "H12_URDF", os.path.join(HERE, "..", "..", "..", "..", "CL_Assets", "ros_assets", "h1_2_magpie.urdf"))
H12_MODE_MACHINE = 6
LOWSTATE_HZ = 100.0
BEAT_HZ = 10.0


class Rig:
    """The node under test, a plant publishing /lowstate and the heartbeat, and recorders."""

    def __init__(self):
        rclpy.init(args=["--ros-args", "-p", f"config:={POLICY}", "-p", f"urdf_path:={URDF}",
                         "-p", "disable_elastic_band:=false"])
        self.node = MarlDeployNode()
        self.plant = rclpy.create_node("heartbeat_gate_plant")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.plant)
        self.lower: list[LowCmd] = []
        self.upper: list[LowCmd] = []
        self.states: list[str] = []
        best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.plant.create_subscription(LowCmd, "/safety/lowcmd_lower_in", self.lower.append, 10)
        self.plant.create_subscription(LowCmd, "/safety/lowcmd_upper_in", self.upper.append, 10)
        self.plant.create_subscription(String, "/marl_deploy_node/state", lambda m: self.states.append(m.data), latched)
        self._lowstate_pub = self.plant.create_publisher(LowState, "/lowstate", best_effort)
        self._beat_pub = self.plant.create_publisher(Bool, "/safety/heartbeat", 10)
        self.q0 = 0.0
        self.beating = False
        self.plant.create_timer(1.0 / LOWSTATE_HZ, self._publish_lowstate)
        self.plant.create_timer(1.0 / BEAT_HZ, self._publish_beat)

    def _publish_lowstate(self):
        msg = LowState()
        msg.mode_machine = H12_MODE_MACHINE
        msg.motor_state[0].q = float(self.q0)
        self._lowstate_pub.publish(msg)

    def _publish_beat(self):
        if self.beating:
            self._beat_pub.publish(Bool(data=True))

    def spin(self, seconds, until=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not (until is not None and until()):
            self.executor.spin_once(timeout_sec=0.005)

    def close(self):
        self.executor.shutdown()
        self.plant.destroy_node()
        self.node.destroy_node()
        rclpy.shutdown()


@pytest.fixture
def rig():
    if os.environ.get("ROS_DOMAIN_ID", "0") in ("", "0"):
        pytest.skip("publishes /lowstate and /safety/heartbeat: set ROS_DOMAIN_ID to a non-zero domain")
    r = Rig()
    try:
        yield r
    finally:
        r.close()


def test_silent_until_the_first_heartbeat_then_pre_poses_from_the_pose_at_the_beat(rig):
    # a fresh /lowstate and no heartbeat: nothing on either input for a second, in a named state
    rig.spin(1.0)
    assert rig.lower == [] and rig.upper == []
    assert rig.states[-1] == "waiting_heartbeat"

    # the pose moves while waiting; the ramp starts from the pose at the first beat, not from boot
    rig.q0 = 0.3
    rig.spin(0.3)
    assert rig.lower == [] and rig.upper == []
    rig.beating = True
    rig.spin(5.0, until=lambda: rig.lower and rig.upper and rig.states[-1] == "preposing")

    assert rig.lower and rig.upper
    assert rig.states[-1] == "preposing"
    assert rig.lower[0].motor_cmd[0].q == pytest.approx(0.3, abs=1e-6)
    assert rig.lower[0].mode_machine == H12_MODE_MACHINE
