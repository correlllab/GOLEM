"""
Remove the robot's own body from a point cloud.

Every URDF link with collision geometry is approximated by its padded
axis-aligned bounding box in the link frame (robot_geometry.link_boxes). Each
cloud is tested against the boxes posed by TF at the cloud's stamp, and points
inside any box are dropped. All point fields are passed through unchanged.

Subscribes ``cloud_in`` (sensor_msgs/PointCloud2) and ``/robot_description``
(std_msgs/String, transient local, from robot_state_publisher); publishes
``cloud_out`` in the input cloud's frame.

ROS parameters:
  padding         m added on every side of each link box; covers TF latency
                  while the arms move and lidar range noise
  exclude_links   links never filtered
  max_tf_age      s; if TF at the cloud stamp is unavailable, the latest
                  transform is used as long as it is at most this old
"""

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from golem_slam.robot_geometry import link_boxes, points_in_box, transform_to_rt

# sensor_msgs/PointField datatype -> numpy type
_FIELD_TYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def cloud_dtype(msg):
    endian = '>' if msg.is_bigendian else '<'
    fields = [f for f in msg.fields if f.count == 1]
    return np.dtype({
        'names': [f.name for f in fields],
        'formats': [endian + _FIELD_TYPES[f.datatype] for f in fields],
        'offsets': [f.offset for f in fields],
        'itemsize': msg.point_step,
    })


def cloud_points(msg):
    """(structured array of all points, (N, 3) float64 xyz)."""
    arr = np.frombuffer(msg.data, dtype=cloud_dtype(msg),
                        count=msg.width * msg.height)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)
    return arr, xyz


class SelfFilter(Node):

    def __init__(self):
        super().__init__('self_filter')
        self.padding = self.declare_parameter('padding', 0.05).value
        self.exclude = set(self.declare_parameter('exclude_links', ['']).value) - {''}
        self.max_tf_age = Duration(
            seconds=self.declare_parameter('max_tf_age', 0.2).value)

        self.boxes = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            String, '/robot_description', self._on_description,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pub = self.create_publisher(PointCloud2, 'cloud_out', qos_profile_sensor_data)
        self.create_subscription(PointCloud2, 'cloud_in', self._on_cloud,
                                 qos_profile_sensor_data)

    def _on_description(self, msg):
        try:
            boxes = link_boxes(msg.data, self.padding)
        except (OSError, ValueError, KeyError) as e:
            self.get_logger().error(f'cannot build link boxes from robot_description: {e}')
            return
        self.boxes = {k: v for k, v in boxes.items() if k not in self.exclude}
        self.get_logger().info(f'filtering {len(self.boxes)} links, padding {self.padding} m')

    def _lookup(self, frame, link, stamp):
        try:
            return self.tf_buffer.lookup_transform(frame, link, stamp)
        except TransformException:
            pass
        tf = self.tf_buffer.lookup_transform(frame, link, Time())
        if Time.from_msg(stamp) - Time.from_msg(tf.header.stamp) > self.max_tf_age:
            raise TransformException(f'{frame}->{link} older than max_tf_age')
        return tf

    def _on_cloud(self, msg):
        if self.boxes is None or msg.width * msg.height == 0:
            return
        arr, xyz = cloud_points(msg)
        keep = np.isfinite(xyz).all(axis=1)
        missing = []
        for link, (lo, hi) in self.boxes.items():
            try:
                tf = self._lookup(msg.header.frame_id, link, msg.header.stamp)
            except TransformException:
                missing.append(link)
                continue
            rot, trans = transform_to_rt(tf.transform)
            # Cheap reject: only test points within the box's bounding sphere.
            centre = rot @ (0.5 * (lo + hi)) + trans
            radius = 0.5 * np.linalg.norm(hi - lo)
            near = keep & (np.sum((xyz - centre) ** 2, axis=1) <= radius * radius)
            if near.any():
                idx = np.flatnonzero(near)
                keep[idx[points_in_box(xyz[idx], rot, trans, lo, hi)]] = False
        if missing:
            # Publishing with unfiltered links would put the body back into the
            # maps, so drop the cloud instead.
            self.get_logger().warn(
                f'no TF for {len(missing)} links (e.g. {missing[0]}); dropping cloud',
                throttle_duration_sec=5.0)
            return

        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = int(keep.sum())
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = msg.point_step * out.width
        out.is_dense = True
        out.data = arr[keep].tobytes()
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SelfFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
