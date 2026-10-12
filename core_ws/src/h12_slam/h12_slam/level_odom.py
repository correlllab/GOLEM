"""
Publish level, floor-referenced ``odom`` and ``base_footprint`` frames around FAST-LIO.

FAST-LIO's world frame ``camera_init`` is the IMU frame at startup, so it
carries the MID360's mounting pitch (~14 deg on the H1-2) and sits at lidar
height. Anything that thresholds on height (octomap projection, costmap
obstacle bands) needs a frame with +z against gravity and z = 0 on the floor.

Once at startup this node:
  1. averages the IMU specific force (``imu``; FAST-LIO's body frame, the same
     frame the identity extrinsic_R in mid360.yaml assumes) rotated into
     camera_init through TF, giving the up direction in camera_init. The robot
     must hold still for this window (spread above ``max_accel_spread`` of
     the mean restarts it);
  2. accumulates ``floor_scans`` self-filtered scans (``cloud``) in
     camera_init and finds the floor plane in them (find_floor_height);
  3. publishes the static transform odom -> camera_init: the smallest rotation
     taking camera_init's up direction to +z (camera_init's heading is kept),
     and a z offset putting the floor at z = 0.

The floor comes from the lidar, not the leg kinematics, so it is right even
when the feet are off the ground (gantry) or the legs are bent.

From then on, for every FAST-LIO ``odometry`` message it publishes
odom -> base_footprint: the ``base_frame`` (pelvis) position projected onto the
floor (z = 0) with its yaw only. slam_toolbox and Nav2 must use this frame as
their base: slam_toolbox composes map -> odom through its base frame, so a
base that rolls, pitches and sits above the floor makes the map frame bob and
tilt with the robot's sway.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, PointCloud2
from tf2_ros import (
    Buffer, StaticTransformBroadcaster, TransformBroadcaster, TransformException,
    TransformListener)

from h12_slam.robot_geometry import (
    matrix_to_quat, quat_to_matrix, rotation_aligning, transform_to_rt)
from h12_slam.self_filter import cloud_points


def find_floor_height(points, up, sensor_height, min_depth, min_points,
                      bin_size=0.02, band=0.04, max_tilt=np.radians(5.0)):
    """
    Height along ``up`` of the floor plane in ``points``, or None.

    Only points at least ``min_depth`` below the sensor (``sensor_height``
    along up) are candidates, which leaves out walls and furniture tops near
    lidar height. Their heights are histogrammed and the floor is the lowest
    bin holding ``min_points`` or more, so a table top with more returns than
    the floor still loses, while a few stray points below the floor (glass and
    mirror reflections) don't qualify. The points within ``band`` of that bin
    are fit with a plane; it is rejected if its normal is more than
    ``max_tilt`` off ``up`` (a ramp, or a wall base).
    """
    h = points @ up
    cand = h[h < sensor_height - min_depth]
    if cand.size < min_points:
        return None
    edges = np.arange(cand.min(), cand.max() + 2 * bin_size, bin_size)
    counts, edges = np.histogram(cand, bins=edges)
    dense = np.flatnonzero(counts >= min_points)
    if dense.size == 0:
        return None
    centre = 0.5 * (edges[dense[0]] + edges[dense[0] + 1])

    inliers = points[np.abs(h - centre) < band]
    mean = inliers.mean(axis=0)
    normal = np.linalg.svd(inliers - mean, full_matrices=False)[2][-1]
    if np.arccos(min(1.0, abs(float(normal @ up)))) > max_tilt:
        return None
    return float(np.median(inliers @ up))


class LevelOdom(Node):

    def __init__(self):
        super().__init__('level_odom')
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.world_frame = self.declare_parameter('world_frame', 'camera_init').value
        self.body_frame = self.declare_parameter('body_frame', 'body').value
        self.base_frame = self.declare_parameter('base_frame', 'pelvis').value
        self.footprint_frame = self.declare_parameter(
            'footprint_frame', 'base_footprint').value
        self.samples = self.declare_parameter('samples', 200).value  # ~1 s of MID360 IMU
        self.max_spread = self.declare_parameter('max_accel_spread', 0.1).value
        self.floor_scans = self.declare_parameter('floor_scans', 10).value  # ~1 s at 10 Hz
        # m below the lidar; the lidar is >= ~1.4 m above the floor in every
        # stance, and table tops (~0.75 m) sit closer to it than this.
        self.min_depth = self.declare_parameter('min_floor_depth', 1.0).value
        self.min_points = self.declare_parameter('min_floor_points', 300).value

        self.up = None
        self.odom_from_world = None  # (rot, trans), set once odom is published
        self.body_from_base = None
        self.ups = []
        self.scans = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.broadcaster = StaticTransformBroadcaster(self)
        self.footprint_broadcaster = TransformBroadcaster(self)

        self.imu_sub = self.create_subscription(Imu, 'imu', self._on_imu,
                                                qos_profile_sensor_data)
        self.cloud_sub = self.create_subscription(PointCloud2, 'cloud', self._on_cloud,
                                                  qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odometry', self._on_odometry, 10)

    def _world_from_body(self, stamp):
        try:
            return transform_to_rt(self.tf_buffer.lookup_transform(
                self.world_frame, self.body_frame, stamp).transform)
        except TransformException:
            pass
        try:
            return transform_to_rt(self.tf_buffer.lookup_transform(
                self.world_frame, self.body_frame, Time()).transform)
        except TransformException:
            self.get_logger().info(
                f'waiting for {self.world_frame}->{self.body_frame} (FAST-LIO)',
                throttle_duration_sec=5.0)
            return None

    def _on_imu(self, msg):
        if self.up is not None:
            return
        rt = self._world_from_body(Time())
        if rt is None:
            return
        a = msg.linear_acceleration
        # Specific force at rest points up; units (g or m/s^2) cancel below.
        self.ups.append(rt[0] @ np.array([a.x, a.y, a.z]))
        if len(self.ups) < self.samples:
            return

        ups = np.array(self.ups)
        self.ups = []
        mean = ups.mean(axis=0)
        spread = np.linalg.norm(ups - mean, axis=1).max() / np.linalg.norm(mean)
        if spread > self.max_spread:
            self.get_logger().warn(
                f'robot moving (accel spread {spread:.2f}); waiting for it to stand still',
                throttle_duration_sec=5.0)
            return
        self.up = mean / np.linalg.norm(mean)
        self.destroy_subscription(self.imu_sub)

    def _on_cloud(self, msg):
        if self.up is None:
            return
        if msg.header.frame_id != self.body_frame:
            self.get_logger().error(
                f'cloud frame {msg.header.frame_id!r} is not {self.body_frame!r}',
                throttle_duration_sec=5.0)
            return
        rt = self._world_from_body(msg.header.stamp)
        if rt is None:
            return
        rot, trans = rt
        _, xyz = cloud_points(msg)
        self.scans.append(xyz @ rot.T + trans)
        if len(self.scans) < self.floor_scans:
            return

        points = np.vstack(self.scans)
        self.scans = []
        floor = find_floor_height(points, self.up, float(trans @ self.up),
                                  self.min_depth, self.min_points)
        if floor is None:
            self.get_logger().warn(
                'no level floor plane in view yet; retrying', throttle_duration_sec=5.0)
            return
        self._publish(floor, float(trans @ self.up))

    def _publish(self, floor, sensor_height):
        rot = rotation_aligning(self.up, np.array([0.0, 0.0, 1.0]))
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.world_frame
        t.transform.translation.z = -floor
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = (float(v) for v in matrix_to_quat(rot))
        self.broadcaster.sendTransform(t)
        self.odom_from_world = (rot, np.array([0.0, 0.0, -floor]))
        self.destroy_subscription(self.cloud_sub)
        tilt = np.degrees(np.arccos(np.clip(self.up[2], -1.0, 1.0)))
        self.get_logger().info(
            f'{self.odom_frame}->{self.world_frame}: tilt {tilt:.1f} deg, '
            f'lidar {sensor_height - floor:.3f} m above the floor')

    def _on_odometry(self, msg):
        if self.odom_from_world is None:
            return
        if self.body_from_base is None:
            try:
                self.body_from_base = transform_to_rt(self.tf_buffer.lookup_transform(
                    self.body_frame, self.base_frame, Time()).transform)
            except TransformException:
                self.get_logger().info(
                    f'waiting for {self.body_frame}->{self.base_frame}',
                    throttle_duration_sec=5.0)
                return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        r_wb, t_wb = quat_to_matrix(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z])
        r_ow, t_ow = self.odom_from_world
        r_bp, t_bp = self.body_from_base
        r_ob, t_ob = r_ow @ r_wb, r_ow @ t_wb + t_ow
        r_op, t_op = r_ob @ r_bp, r_ob @ t_bp + t_ob
        yaw = np.arctan2(r_op[1, 0], r_op[0, 0])

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.footprint_frame
        t.transform.translation.x = float(t_op[0])
        t.transform.translation.y = float(t_op[1])
        t.transform.rotation.z = float(np.sin(yaw / 2.0))
        t.transform.rotation.w = float(np.cos(yaw / 2.0))
        self.footprint_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = LevelOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
