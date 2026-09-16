"""Real MuJoCo rendering and ROS messages; no ROS node or DDS traffic."""
import io
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
from PIL import Image as PILImage
from rclpy.serialization import serialize_message
from sensor_msgs.msg import PointCloud2

from mujoco_ros_bridge import (
    RosSensorBridge, _CameraPub, _build_camera_info, _sim_time_to_msg, PC2_DTYPE,
)

SCENE = '''<mujoco>
  <visual><global offwidth="32" offheight="32"/></visual>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="4 4 .1" group="1" rgba=".4 .5 .6 1"/>
    <body name="robot" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size=".2" group="1" rgba="1 0 0 1"/>
    </body>
    <camera name="head" pos="0 -3 2" xyaxes="1 0 0 0 .316 .949"/>
  </worldbody>
</mujoco>'''


class Recorder:
    def __init__(self, lock):
        self.lock = lock
        self.messages = []

    def publish(self, msg):
        if self.lock.locked():
            raise AssertionError('ROS publication must not hold the simulation lock')
        self.messages.append(msg)


class SensorTests(unittest.TestCase):
    def setUp(self):
        self.bridge = object.__new__(RosSensorBridge)
        b = self.bridge
        b.model = mujoco.MjModel.from_xml_string(SCENE)
        b.data = mujoco.MjData(b.model)
        mujoco.mj_forward(b.model, b.data)
        b.data.time = 1.25
        b.sim_lock = threading.Lock()
        b.cam_width = b.cam_height = 32
        b.zfar = float(b.model.vis.map.zfar * b.model.stat.extent)
        b._renderer = None
        b.cam_period = 1 / 15
        b.lidar_period = .1
        b.imu_period = .005
        b.odom_period = .02
        b.base_body_id = -1
        b._last_cam_sim_t = b._last_lidar_sim_t = b._last_imu_sim_t = b._last_odom_sim_t = 0.
        b._camera_specs = [('head', 'head', 'optical')]
        pubs = [Recorder(b.sim_lock) for _ in range(5)]
        b.cameras = [_CameraPub('head', 'optical', 0,
                               _build_camera_info(32, 32, 45, 'optical'), *pubs)]
        b.pub_clock = Recorder(b.sim_lock)
        b.get_logger = lambda: SimpleNamespace(warn=lambda text: self.fail(text))
        # Timing is optional for existing callers of the bridge.
        try:
            from sim_timing import LoopTimings
            b.timings = LoopTimings()
        except ImportError:
            pass
        self.addCleanup(b.shutdown)

    def test_one_scene_update_per_rgb_depth_pair_and_unlocked_render(self):
        b = self.bridge
        original_update = mujoco.Renderer.update_scene
        original_render = mujoco.Renderer.render
        updates = []

        def update(renderer, *args, **kwargs):
            self.assertTrue(b.sim_lock.locked())
            updates.append(kwargs['camera'])
            return original_update(renderer, *args, **kwargs)

        def render(renderer, *args, **kwargs):
            self.assertFalse(b.sim_lock.locked())
            return original_render(renderer, *args, **kwargs)

        with patch.object(mujoco.Renderer, 'update_scene', update), patch.object(mujoco.Renderer, 'render', render):
            b._publish_camera_frame(_sim_time_to_msg(b.data.time))
        self.assertEqual(updates, [0])

    def test_images_match_two_scene_update_reference(self):
        b = self.bridge
        stamp = _sim_time_to_msg(b.data.time)
        b._publish_camera_frame(stamp)
        cam = b.cameras[0]
        rgb = cam.pub_rgb_raw.messages[0]
        depth = cam.pub_depth_raw.messages[0]
        # Reference path rebuilds the same scene separately for RGB and depth.
        r = b._renderer
        r.disable_depth_rendering()
        r.update_scene(b.data, camera=cam.name, scene_option=b._scene_opt)
        reference_rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(b.data, camera=cam.name, scene_option=b._scene_opt)
        reference_depth = r.render()
        invalid = ~np.isfinite(reference_depth) | (reference_depth <= 0) | (reference_depth >= .999 * b.zfar)
        reference_mm = np.clip(reference_depth * 1000, 0, 65535).astype(np.uint16)
        reference_mm[invalid] = 0
        self.assertEqual(bytes(rgb.data), reference_rgb.tobytes())
        self.assertEqual(bytes(depth.data), reference_mm.tobytes())
        compressed = cam.pub_depth.messages[0]
        decoded = np.asarray(PILImage.open(io.BytesIO(bytes(compressed.data)[12:])))
        np.testing.assert_array_equal(decoded, reference_mm)
        expected_jpeg = io.BytesIO()
        PILImage.fromarray(reference_rgb, mode='RGB').save(expected_jpeg, format='JPEG', quality=80)
        self.assertEqual(bytes(cam.pub_rgb.messages[0].data), expected_jpeg.getvalue())
        for pub in (cam.pub_rgb, cam.pub_depth, cam.pub_rgb_raw, cam.pub_depth_raw, cam.pub_info):
            self.assertEqual(pub.messages[0].header.stamp, stamp)
            self.assertEqual(pub.messages[0].header.frame_id, 'optical')
        self.assertEqual(rgb.encoding, 'rgb8')
        self.assertEqual(depth.encoding, '16UC1')

    def test_tick_preserves_sim_time_schedule_and_publication_order(self):
        b = self.bridge
        events = []
        b.pub_clock.publish = lambda msg: events.append(('clock', msg.clock.sec, msg.clock.nanosec))
        b._publish_imu = lambda stamp: events.append(('imu', stamp.sec, stamp.nanosec))
        b._publish_camera_frame = lambda stamp: events.append(('camera', stamp.sec, stamp.nanosec))
        b._publish_lidar_scan = lambda stamp: events.append(('lidar', stamp.sec, stamp.nanosec))
        b.tick()
        self.assertEqual([e[0] for e in events], ['clock', 'imu', 'camera', 'lidar'])
        self.assertTrue(all(e[1:] == (1, 250000000) for e in events))
        events.clear()
        b.data.time = 1.251
        b.tick()
        self.assertEqual([e[0] for e in events], ['clock'])
        events.clear()
        b.data.time = 1.256
        b.tick()
        self.assertEqual([e[0] for e in events], ['clock', 'imu'])

    def test_lidar_capture_locked_but_both_messages_unlocked(self):
        b = self.bridge
        b.lidar_body_id = 1
        b.lidar_exclude_body_id = 1
        b.lidar_frame = 'lidar'
        b.lidar_geomgroup = np.ones((6, 1), dtype=np.uint8)
        b.lidar_min_range, b.lidar_max_range = .1, 40.
        b.lidar_rays = 2
        b.lidar_local_dirs = np.array([[0., 0., -1.], [0., 0., 1.]])
        b.lidar_lines = np.array([0, 1], dtype=np.uint8)
        b.lidar_offset_time_ns = np.array([0, 1000], dtype=np.uint32)
        b._lidar_dists = np.empty((2, 1))
        b._lidar_geomids = np.empty((2, 1), dtype=np.int32)
        b.geom_is_robot = np.zeros(b.model.ngeom, dtype=bool)
        b.pub_lidar = Recorder(b.sim_lock)
        b.pub_pc2 = Recorder(b.sim_lock)
        original = mujoco.mj_multiRay

        def cast(*args):
            self.assertTrue(b.sim_lock.locked())
            return original(*args)

        with patch.object(mujoco, 'mj_multiRay', cast):
            b._publish_lidar_scan(_sim_time_to_msg(b.data.time))
        msg = b.pub_lidar.messages[0]
        pc2 = b.pub_pc2.messages[0]
        self.assertEqual(msg.timebase, 1250000000)
        self.assertEqual(msg.point_num, 1)
        self.assertAlmostEqual(msg.points[0].z, -1.)
        points = np.frombuffer(bytes(pc2.data), dtype=PC2_DTYPE)
        self.assertEqual(points.size, 1)
        self.assertAlmostEqual(float(points['z'][0]), -1.)
        self.assertEqual(pc2.header, msg.header)
        reference = PointCloud2()
        for field in pc2.get_fields_and_field_types():
            setattr(reference, field, bytes(pc2.data) if field == 'data' else getattr(pc2, field))
        self.assertEqual(serialize_message(pc2), serialize_message(reference))


if __name__ == '__main__':
    unittest.main()
