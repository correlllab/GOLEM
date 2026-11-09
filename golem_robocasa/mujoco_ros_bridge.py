"""Run the mujoco_sensors ROS 2 node: /clock, opt-in ground-truth odom and a robot's sensors.

The node itself publishes:
  /clock                       rosgraph_msgs/Clock (sim time), on every tick
  /odom, odom -> pelvis on /tf ground-truth base pose, only with base_body_id >= 0
                               (opt-in, GOLEM_SIM_ODOM)
Everything else on the node comes from the sensor suite, which reproduces the
robot's own sensor drivers (h12_sensors.H12Sensors, g1.g1_sensors.G1Sensors).
The suite is built in the constructor, so a missing GL fails at startup and a
camera's EGL context exists before the passive viewer's GLFW window.

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

# The ground-truth odom's frames and rate (Hz of sim time).
ODOM_FRAME = "odom"
BASE_FRAME = "pelvis"
ODOM_RATE_HZ = 50.0


def _sim_time_to_msg(sim_time: float) -> TimeMsg:
    """Convert MuJoCo sim time (float seconds) to a ROS Time message."""
    sec = int(sim_time)
    nanosec = int((sim_time - sec) * 1e9)
    msg = TimeMsg()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


class SensorSuite(Protocol):
    """Define the interface of a robot's sensor models, which publish on the host's node."""

    def tick(self, sim_t: float, stamp: TimeMsg) -> None:
        """Publish what is due at sim_t (s); stamp is sim_t as a ROS time.

        Called after /clock and odom, on every tick.
        """

    def shutdown(self) -> None:
        """Stop the suite's threads and free its GL context on the thread that owns it."""

    def stats(self) -> str:
        """Return a summary for the sim's log, or "" when there is nothing to report."""


class RosSensorBridge(Node):
    """Host a sensor suite on the mujoco_sensors node and publish /clock and optional odom."""

    def __init__(
        self,
        model,
        data,
        make_sensors: Callable[[Node], SensorSuite],
        *,
        base_body_id: int = -1,
        sim_lock=None,
    ) -> None:
        """Build the suite with make_sensors(node).

        base_body_id >= 0 publishes that body's world pose as ODOM_FRAME ->
        BASE_FRAME on /tf and /odom at ODOM_RATE_HZ of sim time. sim_lock, when
        given, is held only while reading MjData.
        """
        super().__init__("mujoco_sensors")
        self.model = model
        self.data = data
        self.sim_lock = sim_lock

        # Reliable QoS, so subscribers do not drop ticks.
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)

        self.suite = make_sensors(self)

        # Ground truth, which the real robot does not have: GOLEM_SIM_ODOM opts in.
        self.base_body_id = int(base_body_id)
        self.odom_period = 1.0 / ODOM_RATE_HZ
        self._last_odom_sim_t = 0.0
        if self.base_body_id >= 0:
            self._tf_broadcaster = TransformBroadcaster(self)
            self.pub_odom = self.create_publisher(Odometry, "/odom", 10)

    def shutdown(self) -> None:
        """Stop the suite."""
        self.suite.shutdown()

    def stats(self) -> str:
        """Return the suite's summary for the sim's log."""
        return self.suite.stats()

    def tick(self) -> None:
        """Publish /clock, odom when due, then tick the suite; once per sim step, on the physics thread.

        With sim_lock, call it without holding the lock; without, keep physics
        from advancing during the call.
        """
        with self.sim_lock or nullcontext():
            sim_t = float(self.data.time)
        stamp = _sim_time_to_msg(sim_t)

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
        """Publish the base body's world pose as ODOM_FRAME -> BASE_FRAME on /tf and /odom."""
        # MuJoCo quaternions are (w, x, y, z); ROS's are (x, y, z, w).
        with self.sim_lock or nullcontext():
            pos = self.data.xpos[self.base_body_id]
            quat = self.data.xquat[self.base_body_id]
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            qw, qx, qy, qz = (float(quat[0]), float(quat[1]),
                              float(quat[2]), float(quat[3]))

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = ODOM_FRAME
        tf.child_frame_id = BASE_FRAME
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
        odom.header.frame_id = ODOM_FRAME
        odom.child_frame_id = BASE_FRAME
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = pz
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self.pub_odom.publish(odom)


def init_ros() -> None:
    """Initialise rclpy, raising RuntimeError unless ROS_DOMAIN_ID is a positive integer.

    SAFETY: domain 0 is the real robots' command bus; a sim never joins it.
    """
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


def shutdown_ros() -> None:
    """Shut rclpy down if it is running."""
    if rclpy.ok():
        rclpy.shutdown()
