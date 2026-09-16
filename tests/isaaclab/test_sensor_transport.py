"""Actual rclpy transport; run via run_sensor_transport.sh without a GPU app."""
import os
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

import numpy as np

from src.python.sensors.ros_sensor_bridge import RosSensorBridge, POINT_DTYPE


@unittest.skipUnless(os.environ.get('GOLEM_TEST_ROS_TRANSPORT') == '1', 'requires bundled ROS library paths')
class SensorTransportTests(unittest.TestCase):
    def test_failed_node_creation_releases_context(self):
        from src.python.sensors.ros_sensor_bridge import bootstrap_ros
        rclpy = bootstrap_ros()
        bridge = RosSensorBridge.__new__(RosSensorBridge)
        with patch.object(rclpy, 'create_node', side_effect=RuntimeError('injected node creation failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected node creation failure'):
                bridge.__init__(NS(sensors={}))
        try:
            self.assertFalse(bridge.context.ok())
        finally:
            if bridge.context.ok():
                bridge.context.shutdown()

    def test_real_ros_subscriber_receives_clock_rgb_depth_intrinsics_cloud_and_imu(self):
        camera = NS(cfg=NS(update_period=.02, spawn=NS(clipping_range=(.1, 100.))), data=NS(
            output={'rgb': np.full((1, 2, 3, 4), 77, dtype=np.uint8),
                    'distance_to_image_plane': np.array([[[[1.], [np.inf], [2.]], [[.5], [0.], [100.]]]])},
            intrinsic_matrices=np.array([[[100., 0., 1.5], [0., 100., 1.], [0., 0., 1.]]])))
        lidar = NS(cfg=NS(update_period=.1, pattern_cfg=NS(horizontal_fov_range=(0., 360.), horizontal_res=180.)),
                   data=NS(ray_hits_w=np.array([[[1., 2., 3.], [np.inf, 0, 0]]]), pos_w=np.zeros((1, 3)), quat_w=np.array([[1., 0., 0., 0.]])))
        imu = NS(cfg=NS(update_period=.02), data=NS(ang_vel_b=np.array([[.1, .2, .3]]), lin_acc_b=np.array([[0., 0., 9.81]])))
        bridge = RosSensorBridge(NS(sensors={'front_camera': camera, 'lidar': lidar, 'imu': imu}))
        from src.python.dds.common.transport import initialize, ChannelPublisher
        # RMW must create Cyclone's explicit domain before the SDK participant.
        initialize(int(os.environ['ROS_DOMAIN_ID']))
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        sdk_publisher = ChannelPublisher('rt/test_isaac_sensor_transport', String_)
        sdk_publisher.Init()
        sdk_publisher.Write(String_('coexisting DDS participant'))
        node = bridge.rclpy.create_node('sensor_transport_check', context=bridge.context)
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu, PointCloud2
        from rosgraph_msgs.msg import Clock
        from rclpy.executors import SingleThreadedExecutor
        executor = SingleThreadedExecutor(context=bridge.context)
        executor.add_node(node)
        received = {}
        def receive(key):
            return lambda msg: received.__setitem__(key, msg)
        subscriptions = []
        for key, msg_type, topic in (
            ('clock', Clock, '/clock'), ('rgb', Image, '/realsense/head/color/image_raw'),
            ('jpeg', CompressedImage, '/realsense/head/color/image_raw/compressed'),
            ('depth', Image, '/realsense/head/aligned_depth_to_color/image_raw'),
            ('png', CompressedImage, '/realsense/head/aligned_depth_to_color/image_raw/compressedDepth'),
            ('info', CameraInfo, '/realsense/head/color/camera_info'),
            ('lidar', PointCloud2, '/livox/pointcloud'), ('imu', Imu, '/livox/imu'),
        ):
            subscriptions.append(node.create_subscription(msg_type, topic, receive(key), qos_profile_sensor_data))
        try:
            deadline = time.monotonic() + 10
            tick = 0
            while len(received) < 8 and time.monotonic() < deadline:
                bridge.publish(2. + .1 * tick)
                tick += 1
                executor.spin_once(timeout_sec=.05)
            self.assertEqual(len(received), 8, f'missing topics: {received.keys()}')
            self.assertEqual(received['rgb'].encoding, 'rgb8')
            self.assertEqual(received['rgb'].step, 9)
            self.assertEqual(len(received['rgb'].data), 18)
            np.testing.assert_array_equal(np.frombuffer(received['depth'].data, dtype='<u2'), [1000, 0, 2000, 500, 0, 0])
            self.assertEqual(received['info'].k[0], 100.)
            self.assertEqual(received['info'].header.frame_id, 'camera_color_optical_frame')
            cloud = np.frombuffer(received['lidar'].data, dtype=POINT_DTYPE)
            self.assertEqual(cloud['x'].tolist(), [1.])
            self.assertEqual(received['lidar'].header.frame_id, 'lidar_link')
            self.assertAlmostEqual(received['imu'].linear_acceleration.z, 9.81)
            self.assertEqual(received['imu'].orientation_covariance[0], -1.)
        finally:
            sdk_publisher.Close()
            executor.shutdown()
            node.destroy_node()
            bridge.close()


if __name__ == '__main__':
    unittest.main()
