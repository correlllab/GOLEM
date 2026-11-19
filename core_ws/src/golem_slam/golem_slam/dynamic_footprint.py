"""
Publish the local costmap's footprint from the arms' measured pose.

The footprint is the convex hull, on the floor (base_footprint's x-y plane), of
the fixed body polygon (the local costmap's ``footprint`` from nav2.yaml) and
the corners of the arm links' padded collision boxes (robot_geometry.link_boxes,
posed by TF). It never shrinks inside the body polygon, so an arm held ahead
of the robot (carrying, or between moves) widens it, and the arms at the sides
change nothing.

While an arm goal is in flight on any of ``arm_action_status`` (accepted,
executing or cancelling), the footprint is the body polygon alone: a hand
reaching for a handle or a counter must not read as the robot colliding with
it, or Nav2 would stop the base during a grasp. The same body polygon goes out
until the first message on ``arms_ready_topic``, which the arm controller
publishes once its start-up moves are done (frame_task_server's
``/left_ee_pose``): those moves swing the arms behind the pelvis, past the
local costmap's inflation radius, while the robot cannot drive yet. It also goes
out while any arm link has no TF within ``max_tf_age``.

Nav2's costmap reads a polygon on its ``footprint`` topic in place of its
``footprint`` parameter (nav2_costmap_2d's Costmap2DROS); the launch remaps
``footprint`` to the local costmap's. The global costmap keeps the fixed
polygon, so the planner's paths do not change with the arms.

Subscribes ``/robot_description`` (std_msgs/String, transient local), each
``arm_action_status`` topic (action_msgs/GoalStatusArray) and
``arms_ready_topic`` (geometry_msgs/PoseStamped); publishes
``footprint`` (geometry_msgs/Polygon, in ``base_frame``: Humble's costmap takes a
Polygon with no frame and reads it in its robot_base_frame).

ROS parameters:
  base_frame          the costmap's robot frame
  body_footprint      the body polygon, Nav2's string form "[[x, y], ...]" (m)
  arm_links           URDF links whose boxes widen the footprint
  padding             m added on every side of each link box
  rate                Hz, how often the footprint is recomputed
  min_change          m; a new polygon is published when a vertex of the hull
                      moves by more than this, and at least once a second
  max_tf_age          s; an arm link whose latest TF is older drops the arms
  arm_action_status   action status topics whose live goals drop the arms
  arms_ready_topic    the arms count from its first message on (empty: from the
                      start)
"""

import yaml

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Point32, Polygon, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from golem_slam.robot_geometry import link_boxes, transform_to_rt

# Goal states in which an arm may still move.
LIVE_GOAL_STATES = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING,
                    GoalStatus.STATUS_CANCELING)
# s, the longest time between two publishes, for a costmap that starts late.
REPUBLISH_PERIOD = 1.0


def parse_polygon(text):
    """Return Nav2's footprint string "[[x, y], ...]" as an (n, 2) array; raise on a bad one."""
    points = np.asarray(yaml.safe_load(text), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f'not a polygon of at least 3 [x, y] points: {text!r}')
    return points


def convex_hull(points):
    """Return the convex hull of 2D points, counter-clockwise, without repeats."""
    pts = sorted(set(map(tuple, np.round(np.asarray(points, dtype=float), 6))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        """Return the z of (a - o) x (b - o)."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def box_corners(lo, hi):
    """Return the 8 corners of the axis-aligned box [lo, hi], as an (8, 3) array."""
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def footprint(body, arm_points):
    """Return the hull of the body polygon and the arm points' x-y (m, base frame)."""
    if len(arm_points) == 0:
        return convex_hull(body)
    return convex_hull(np.vstack([body, np.asarray(arm_points)[:, :2]]))


def moved(old, new, tolerance):
    """Return whether polygon new differs from old by more than tolerance (m)."""
    if old is None or len(old) != len(new):
        return True
    # Each vertex's distance to the nearest vertex of the other polygon.
    d = np.linalg.norm(old[:, None, :] - new[None, :, :], axis=2)
    return max(d.min(axis=0).max(), d.min(axis=1).max()) > tolerance


def arm_goal_live(status_arrays):
    """Return whether any goal in the GoalStatusArrays may still move an arm."""
    return any(s.status in LIVE_GOAL_STATES
               for array in status_arrays for s in array.status_list)


def body_only_reason(arms_ready, status_arrays, ready_topic):
    """
    Return why the footprint is the body alone, or None when the arms count.

    arms_ready is whether ready_topic has had a message; status_arrays are the arm
    actions' latest GoalStatusArrays.
    """
    if not arms_ready:
        return f'the arm controller is starting up (no {ready_topic} yet)'
    if arm_goal_live(status_arrays):
        return 'an arm goal is in flight'
    return None


class DynamicFootprint(Node):
    """Publish the local costmap's footprint widened by the arms."""

    def __init__(self):
        """Declare the parameters, subscribe to the description and arm goals, start the timer."""
        super().__init__('dynamic_footprint')
        self.base_frame = self.declare_parameter('base_frame', 'base_footprint').value
        self.body = parse_polygon(self.declare_parameter(
            'body_footprint', '[[0.2, 0.2], [0.2, -0.2], [-0.2, -0.2], [-0.2, 0.2]]').value)
        self.arm_links = [n for n in self.declare_parameter('arm_links', ['']).value if n]
        self.padding = self.declare_parameter('padding', 0.05).value
        rate = self.declare_parameter('rate', 5.0).value
        self.min_change = self.declare_parameter('min_change', 0.03).value
        self.max_tf_age = Duration(seconds=self.declare_parameter('max_tf_age', 0.5).value)
        topics = [t for t in self.declare_parameter(
            'arm_action_status',
            ['/frame_task/_action/status', '/named_config/_action/status']).value if t]
        self.ready_topic = self.declare_parameter('arms_ready_topic', '/left_ee_pose').value

        self.boxes = None
        self.status = {}
        self.published = None
        self.last_publish = None
        self.reason = None
        self.arms_ready = not self.ready_topic
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            String, '/robot_description', self._on_description,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # Action status is latched (transient local) by rclcpp/rclpy action servers.
        status_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for topic in topics:
            self.create_subscription(
                GoalStatusArray, topic,
                lambda msg, topic=topic: self.status.__setitem__(topic, msg), status_qos)
        if self.ready_topic:
            self.create_subscription(PoseStamped, self.ready_topic, self._on_ready, 1)
        self.pub = self.create_publisher(Polygon, 'footprint', 10)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'footprint from {len(self.arm_links)} arm links, body '
            f'{self.body.tolist()}, dropping the arms during goals on {topics}')

    def _on_ready(self, _msg):
        """Count the arms from now on: the arm controller's start-up moves are done."""
        self.arms_ready = True

    def _on_description(self, msg):
        """Build the arm links' boxes from the URDF."""
        try:
            boxes = link_boxes(msg.data, self.padding)
        except (OSError, ValueError, KeyError) as e:
            self.get_logger().error(f'cannot build link boxes from robot_description: {e}')
            return
        missing = [n for n in self.arm_links if n not in boxes]
        if missing:
            self.get_logger().error(
                f'arm links with no collision geometry, left out: {", ".join(missing)}')
        self.boxes = {n: boxes[n] for n in self.arm_links if n in boxes}

    def _arm_points(self):
        """Return the arm boxes' corners in base_frame, or (None, why) when any is missing."""
        if self.boxes is None:
            return None, 'no robot_description yet'
        now = self.get_clock().now()
        points = []
        for link, (lo, hi) in self.boxes.items():
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, link, Time())
            except TransformException:
                return None, f'no TF {self.base_frame} -> {link}'
            if now - Time.from_msg(tf.header.stamp) > self.max_tf_age:
                return None, f'TF {self.base_frame} -> {link} older than max_tf_age'
            rot, trans = transform_to_rt(tf.transform)
            points.append(box_corners(lo, hi) @ rot.T + trans)
        return np.vstack(points) if points else np.empty((0, 3)), None

    def _tick(self):
        """Recompute the footprint and publish it when it moved, or once a second."""
        reason = body_only_reason(self.arms_ready, self.status.values(), self.ready_topic)
        if reason:
            arms = np.empty((0, 3))
        else:
            arms, reason = self._arm_points()
            if arms is None:
                arms = np.empty((0, 3))
        if reason != self.reason:
            self.reason = reason
            self.get_logger().info(
                f'body footprint only: {reason}' if reason else 'footprint follows the arms')
        polygon = footprint(self.body, arms)
        now = self.get_clock().now()
        due = self.last_publish is None or (
            now - self.last_publish > Duration(seconds=REPUBLISH_PERIOD))
        if not (due or moved(self.published, polygon, self.min_change)):
            return
        msg = Polygon()
        msg.points = [Point32(x=float(x), y=float(y), z=0.0) for x, y in polygon]
        self.pub.publish(msg)
        self.published = polygon
        self.last_publish = now


def main(args=None):
    """Run the dynamic_footprint node."""
    rclpy.init(args=args)
    node = DynamicFootprint()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
