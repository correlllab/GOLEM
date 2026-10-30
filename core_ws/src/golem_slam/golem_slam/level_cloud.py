"""
Transform the filtered cloud before scan conversion, without a TF message filter.

The G1 can stall Humble's pointcloud_to_laserscan TF filter while walking.
This adapter uses nonblocking, stamp-exact lookups and a bounded queue instead.
It preserves FAST-LIO deskewing, point fields and timestamps. OctoMap and floor
alignment still consume the original body-frame cloud.
"""
from collections import deque
from copy import deepcopy

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from golem_slam.robot_geometry import transform_to_rt
from golem_slam.self_filter import cloud_points


class LevelCloud(Node):

    def __init__(self):
        super().__init__('level_cloud')
        self.target_frame = self.declare_parameter('target_frame', 'base_footprint').value
        self.pending = deque(maxlen=20)  # at most two seconds at the sensor's 10 Hz
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.publisher = self.create_publisher(PointCloud2, 'cloud_out', qos_profile_sensor_data)
        self.create_subscription(PointCloud2, 'cloud_in', self._on_cloud, qos_profile_sensor_data)
        self.create_timer(0.02, self._drain)

    def _on_cloud(self, msg):
        if len(self.pending) == self.pending.maxlen:
            self.get_logger().warn('dropping oldest cloud waiting for scan-frame TF',
                                   throttle_duration_sec=5.0)
        self.pending.append(msg)
        self._drain()

    def _drain(self):
        # A missing old transform must not prevent a newer transformable scan.
        waiting = []
        while self.pending:
            msg = self.pending.popleft()
            try:
                tf = self.buffer.lookup_transform(
                    self.target_frame, msg.header.frame_id, msg.header.stamp)
            except TransformException:
                waiting.append(msg)
                continue
            fields, xyz = cloud_points(msg)
            rot, trans = transform_to_rt(tf.transform)
            xyz = xyz @ rot.T + trans
            fields = fields.copy()
            for i, name in enumerate(('x', 'y', 'z')):
                fields[name] = xyz[:, i]
            out = deepcopy(msg)
            out.header.frame_id = self.target_frame
            out.height = 1
            out.width = len(fields)
            out.row_step = out.width * out.point_step
            out.data = fields.tobytes()
            self.publisher.publish(out)
            # Older untransformable clouds are obsolete once a newer one is sent.
            waiting.clear()
        self.pending.extend(waiting)


def main(args=None):
    rclpy.init(args=args)
    node = LevelCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
