"""The mujoco_sensors ROS 2 node, which hosts a robot's sensor suite.

The node itself publishes:
  /clock                       rosgraph_msgs/Clock (sim time), on every tick
  /odom, odom -> pelvis on /tf ground-truth base pose, only with base_body_id >= 0
                               (opt-in, GOLEM_SIM_ODOM)
Everything else on the node comes from the suite, which reproduces the robot's
own sensor drivers (h12_sensors.H12Sensors for the H1-2). The suite is built in
the constructor, so a missing GL fails at startup and a camera's EGL context
exists before the passive viewer's GLFW window.

tick() is called once per sim step from the physics thread: /clock, odom when
due, then the suite's tick(sim_t, stamp). With sim_lock, the host holds it only
while it reads MjData; call tick() without holding it. Without sim_lock, the
caller keeps physics from advancing during tick(), e.g. by holding its own lock
around it.
"""
import os
from contextlib import nullcontext
from typing import Callable, Protocol

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster


def _sim_time_to_msg(sim_time: float) -> TimeMsg:
    """Convert MuJoCo sim time (float seconds) to a ROS Time message."""
    sec = int(sim_time)
    nanosec = int((sim_time - sec) * 1e9)
    msg = TimeMsg()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


class SensorSuite(Protocol):
    """A robot's sensor models, publishing on the host's node."""

    def tick(self, sim_t: float, stamp: TimeMsg) -> None:
        """Publish what is due at sim_t (s); stamp is sim_t as a ROS time.
        Called after /clock and odom, on every tick."""

    def shutdown(self) -> None:
        """Stop the suite's threads and free its GL context on the thread that owns it."""

    def stats(self) -> str:
        """A summary for the sim's log; "" when there is nothing to report."""


class RosSensorBridge(Node):
    def __init__(
        self,
        model,
        data,
        make_sensors: Callable[[Node], SensorSuite],
        *,
        base_body_id: int = -1,
        odom_frame: str = "odom",
        base_frame: str = "pelvis",
        odom_rate_hz: float = 50.0,
        sim_lock=None,
    ) -> None:
        super().__init__("mujoco_sensors")
        self.model = model
        self.data = data
        self.sim_lock = sim_lock

        # /clock stays on default reliable QoS so subscribers don't drop ticks.
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)

        # The suite creates its publishers on this node.
        self.suite = make_sensors(self)

        # Ground-truth base odometry (odom -> base_frame TF + /odom). The sim knows
        # the pelvis world pose exactly, so we publish it directly instead of
        # running FAST-LIO. This is the odom source SLAM (slam_toolbox) consumes.
        self.base_body_id = int(base_body_id)
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.odom_period = 1.0 / odom_rate_hz
        self._last_odom_sim_t = 0.0
        if self.base_body_id >= 0:
            self._tf_broadcaster = TransformBroadcaster(self)
            self.pub_odom = self.create_publisher(Odometry, "/odom", 10)

    def shutdown(self) -> None:
        self.suite.shutdown()

    def stats(self) -> str:
        return self.suite.stats()

    def tick(self) -> None:
        """Call once per sim step from the physics thread.

        /clock is published every call, odom at its rate keyed off sim time,
        then the suite ticks with the same sim time and stamp. sim_lock, when
        given, is held only to read MjData, never while publishing.
        """
        with self.sim_lock or nullcontext():
            sim_t = float(self.data.time)
        stamp = _sim_time_to_msg(sim_t)

        # Always publish /clock so subscriber nodes can track sim time.
        clock_msg = Clock()
        clock_msg.clock = stamp
        self.pub_clock.publish(clock_msg)

        if self.base_body_id >= 0 and sim_t - self._last_odom_sim_t >= self.odom_period:
            self._last_odom_sim_t = sim_t
            try:
                self._publish_odom(stamp)
            except Exception as e:
                self.get_logger().warn(f"odom publish failed: {e}")

        self.suite.tick(sim_t, stamp)

    def _publish_odom(self, stamp: TimeMsg) -> None:
        # Ground-truth pelvis pose in the sim world (used as the odom frame).
        # MuJoCo quaternions are (w, x, y, z); ROS wants (x, y, z, w).
        with self.sim_lock or nullcontext():
            pos = self.data.xpos[self.base_body_id]
            quat = self.data.xquat[self.base_body_id]
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            qw, qx, qy, qz = (float(quat[0]), float(quat[1]),
                              float(quat[2]), float(quat[3]))

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = px
        tf.transform.translation.y = py
        tf.transform.translation.z = pz
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = pz
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self.pub_odom.publish(odom)


def init_ros() -> bool:
    raw = os.getenv("ROS_DOMAIN_ID")
    if raw is None or raw == "":
        raise RuntimeError(
            "ROS_DOMAIN_ID is not set. Export a positive integer "
            "(domain 0 is reserved for the real robot), e.g. "
            "`export ROS_DOMAIN_ID=1`."
        )
    try:
        domain = int(raw)
    except ValueError as e:
        raise RuntimeError(
            f"ROS_DOMAIN_ID must be an integer, got {raw!r}."
        ) from e
    if domain <= 0:
        raise RuntimeError(
            f"ROS_DOMAIN_ID must be a positive integer (>0), got {domain}. "
            "Domain 0 is reserved for the real robot."
        )
    if not rclpy.ok():
        rclpy.init()
    return True


def shutdown_ros() -> None:
    if rclpy.ok():
        rclpy.shutdown()
