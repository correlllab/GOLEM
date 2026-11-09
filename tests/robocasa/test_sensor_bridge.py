"""Check the H1-2 sensor suite (h12_sensors.py) with real MuJoCo rendering and ROS messages.

The suite is built without its constructor, on a small scene, with recording publishers: no ROS
node starts and no DDS traffic flows.
"""
import copy
import io
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
from PIL import Image as PILImage
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from rclpy.serialization import deserialize_message, serialize_message

import mid360
import realsense_profile as rp
from h12_sensors import (
    H12Sensors, _CameraPub, _LidarScans, _camera_info, _serialize_livox_scan,
    self_filter_geoms, PC2_DTYPE, _COMPRESSED_DEPTH_HEADER, _JPEG_QUALITY,
)
from mujoco_ros_bridge import RosSensorBridge, _sim_time_to_msg
from sim_timing import LoopTimings

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
    """Stand in for a ROS publisher: record messages, and fail one published under the sim lock."""

    def __init__(self, lock, subscribers=1):
        """Watch `lock` and report `subscribers` subscriptions."""
        self.lock = lock
        self.messages = []
        self.subscribers = subscribers

    def get_subscription_count(self):
        """Return the configured subscriber count."""
        return self.subscribers

    def publish(self, msg):
        """Record a message; publishing under the sim lock raises AssertionError."""
        if self.lock.locked():
            raise AssertionError('ROS publication must not hold the simulation lock')
        self.messages.append(msg)


# A scene with what the scene update reads of MjData besides plain geoms: a spot light on a
# target, a light and a camera that ride a moving body, tracking and fixed cameras, a tendon,
# and a transparent geom.
MOVING_SCENE = '''<mujoco>
  <visual><global offwidth="48" offheight="48"/></visual>
  <worldbody>
    <light name="sun" pos="0 0 3" dir="0 0 -1"/>
    <light name="spot" mode="targetbody" target="base" pos="2 -2 3" cutoff="30"/>
    <geom type="plane" size="4 4 .1" group="1" rgba=".4 .5 .6 1"/>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".2 .1 .3" group="1" rgba="1 0 0 1"/>
      <body name="arm" pos=".3 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".05 .2" group="1" rgba="0 1 0 1"/>
        <camera name="on_arm" pos="0 0 .3" xyaxes="1 0 0 0 1 0"/>
        <light name="arm_light" pos="0 0 .3" dir="0 0 1"/>
        <site name="tip" pos="0 0 .2" size=".03"/>
      </body>
    </body>
    <body name="slider" pos="1 1 .5">
      <joint type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".1" group="1" rgba="0 0 1 .5"/>
      <site name="top" size=".05"/>
    </body>
    <camera name="track" mode="targetbody" target="base" pos="2 -3 2"/>
    <camera name="fixed" pos="0 -3 2" xyaxes="1 0 0 0 .316 .949"/>
    <camera name="com" mode="trackcom" pos="0 -3 2" xyaxes="1 0 0 0 .316 .949"/>
  </worldbody>
  <tendon><spatial name="cord" width=".02"><site site="tip"/><site site="top"/></spatial></tendon>
</mujoco>'''


class CameraSnapshotTests(unittest.TestCase):
    """Check the camera snapshot against a full MjData copy."""

    def setUp(self):
        """Build an H12Sensors holding only the model, data and sim lock the snapshot reads."""
        self.bridge = object.__new__(H12Sensors)
        b = self.bridge
        b.model = mujoco.MjModel.from_xml_string(MOVING_SCENE)
        b.data = mujoco.MjData(b.model)
        b.sim_lock = threading.Lock()
        self.renderer = mujoco.Renderer(b.model, height=48, width=48)
        self.addCleanup(self.renderer.close)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[1] = 1
        self.option.sitegroup[:] = 0

    def render(self, data, camera):
        """Return (rgb, depth) of camera rendered from data."""
        r = self.renderer
        r.disable_depth_rendering()
        r.update_scene(data, camera=camera, scene_option=self.option)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=camera, scene_option=self.option)
        depth = r.render().copy()
        r.disable_depth_rendering()
        return rgb, depth

    def test_camera_snapshot_renders_the_same_image(self):
        """Every camera renders the same pixels and depths from the partial snapshot as from a full copy."""
        b = self.bridge
        rng = np.random.default_rng(0)
        cameras = range(b.model.ncam)
        for _ in range(6):
            b.data.qpos[:] += rng.normal(0, .3, b.model.nq)
            b.data.qpos[3:7] /= np.linalg.norm(b.data.qpos[3:7])
            for _ in range(3):
                mujoco.mj_step(b.model, b.data)
            snapshot = b._capture_camera_snapshot()
            full = copy.copy(b.data)
            for camera in cameras:
                for got, want in zip(self.render(snapshot, camera), self.render(full, camera)):
                    np.testing.assert_array_equal(got, want)

    def test_camera_snapshot_is_preallocated_and_alternates_two_slots(self):
        """Frames are captured into two preallocated MjData in turn, never the live one, with the kinematics current."""
        b = self.bridge
        slots = []
        for step in range(4):
            b.data.qpos[2] = 1.0 + step
            mujoco.mj_forward(b.model, b.data)
            snapshot = b._capture_camera_snapshot()
            self.assertIsNot(snapshot, b.data)
            np.testing.assert_array_equal(snapshot.geom_xpos, b.data.geom_xpos)
            slots.append(snapshot)
        self.assertIs(slots[0], slots[2])
        self.assertIs(slots[1], slots[3])
        self.assertIsNot(slots[0], slots[1])

    def test_camera_snapshot_is_taken_under_the_sim_lock(self):
        """The snapshot is filled while the sim lock is held, and the lock is free afterwards."""
        b = self.bridge
        held = []

        class Spy:
            """Stand in for the data to record the lock's state when its fields are read."""

            def __getattr__(inner, name):
                held.append(b.sim_lock.locked())
                return getattr(real, name)

        real = b.data
        b.data = Spy()
        b._capture_camera_snapshot()
        self.assertTrue(held and all(held))
        self.assertFalse(b.sim_lock.locked())


class SensorTests(unittest.TestCase):
    """Run H12Sensors' camera and lidar paths on a small scene."""

    def setUp(self):
        """Build an H12Sensors with one 32x32 head camera (depth from the same camera) and recorders."""
        self.bridge = object.__new__(H12Sensors)
        b = self.bridge
        b.model = mujoco.MjModel.from_xml_string(SCENE)
        b.data = mujoco.MjData(b.model)
        mujoco.mj_forward(b.model, b.data)
        b.data.time = 1.25
        b.sim_lock = threading.Lock()
        b.zfar = float(b.model.vis.map.zfar * b.model.stat.extent)
        b._renderer = None
        b.cam_period = 1 / 15
        b.imu_period = .005
        b._last_cam_sim_t = b._last_imu_sim_t = 0.
        b.depth_stream = b.pointcloud = True
        b._camera_tf = []
        color = rp.fovy_intrinsics(32, 32, 45)
        spec = rp.CameraSpec(
            ns='head', color_camera='head', depth_camera='head', color=color, depth_info=color,
            depth_image=color.scaled(16, 16), color_frame='optical', depth_frame='depth_optical',
            T_color_depth=np.eye(4))
        b._camera_specs = [spec]
        keys = ('color', 'color_jpeg', 'color_info', 'aligned', 'aligned_png', 'aligned_info',
                'depth', 'depth_png', 'depth_info', 'points')
        b.cameras = [_CameraPub(spec, 0, 0, _camera_info(spec.color, 'optical'),
                                _camera_info(spec.depth_info, 'depth_optical'),
                                {k: Recorder(b.sim_lock) for k in keys})]
        b.get_logger = lambda: SimpleNamespace(warn=lambda text: self.fail(text))
        b.timings = LoopTimings()
        self.addCleanup(b.shutdown)

    def test_one_scene_update_per_rgb_depth_pair_and_unlocked_render(self):
        """Colour and aligned depth share one scene update, and capture and rendering run outside the sim lock."""
        b = self.bridge
        original_update = mujoco.Renderer.update_scene
        original_render = mujoco.Renderer.render
        updates = []

        def update(renderer, data, *args, **kwargs):
            """Check the scene is captured from a snapshot, outside the sim lock, and record the camera."""
            self.assertFalse(b.sim_lock.locked())
            self.assertIsNot(data, b.data)
            updates.append(kwargs['camera'])
            return original_update(renderer, data, *args, **kwargs)

        def render(renderer, *args, **kwargs):
            """Check rendering runs outside the sim lock."""
            self.assertFalse(b.sim_lock.locked())
            return original_render(renderer, *args, **kwargs)

        with patch.object(mujoco.Renderer, 'update_scene', update), patch.object(mujoco.Renderer, 'render', render):
            b._publish_camera_frame(_sim_time_to_msg(b.data.time))
        # Colour and aligned depth share one scene update; the decimated depth stream is the
        # second (here from the same camera, as on a D405).
        self.assertEqual(updates, [0, 0])

    def test_camera_renders_hold_the_gl_lock(self):
        """Every camera render holds mid360.GL_LOCK, taking turns with the lidar's GPU caster."""
        b = self.bridge
        original = mujoco.Renderer.render
        held = []

        def render(renderer, *args, **kwargs):
            """Record whether the GL lock is held."""
            held.append(mid360.GL_LOCK.locked())
            return original(renderer, *args, **kwargs)

        with patch.object(mujoco.Renderer, 'render', render):
            b._publish_camera_frame(_sim_time_to_msg(b.data.time))
        # Colour, aligned depth and the decimated depth stream.
        self.assertEqual(held, [True, True, True])
        self.assertFalse(mid360.GL_LOCK.locked())

    def test_images_match_two_scene_update_reference(self):
        """The published images, compressed images, frames and stamps match separate reference renders."""
        b = self.bridge
        stamp = _sim_time_to_msg(b.data.time)
        b._publish_camera_frame(stamp)
        cam = b.cameras[0]
        rgb = cam.pubs['color'].messages[0]
        depth = cam.pubs['aligned'].messages[0]
        # The reference rebuilds the same scene separately for RGB and for depth.
        r = b._renderer[(32, 32)]
        r.disable_depth_rendering()
        r.update_scene(b.data, camera=0, scene_option=b._scene_opt)
        reference_rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(b.data, camera=0, scene_option=b._scene_opt)
        reference_depth = r.render()
        invalid = ~np.isfinite(reference_depth) | (reference_depth <= 0) | (reference_depth >= .999 * b.zfar)
        reference_mm = np.clip(reference_depth * 1000, 0, 65535).astype(np.uint16)
        reference_mm[invalid] = 0
        self.assertEqual(bytes(rgb.data), reference_rgb.tobytes())
        self.assertEqual(bytes(depth.data), reference_mm.tobytes())
        compressed = cam.pubs['aligned_png'].messages[0]
        self.assertEqual(bytes(compressed.data)[:12], _COMPRESSED_DEPTH_HEADER)
        self.assertEqual(compressed.format, '16UC1; compressedDepth')
        decoded = np.asarray(PILImage.open(io.BytesIO(bytes(compressed.data)[12:])))
        np.testing.assert_array_equal(decoded, reference_mm)
        expected_jpeg = io.BytesIO()
        PILImage.fromarray(reference_rgb, mode='RGB').save(expected_jpeg, format='JPEG', quality=_JPEG_QUALITY)
        self.assertEqual(bytes(cam.pubs['color_jpeg'].messages[0].data), expected_jpeg.getvalue())
        self.assertEqual(cam.pubs['color_jpeg'].messages[0].format, 'rgb8; jpeg compressed bgr8')
        for key in ('color', 'color_jpeg', 'color_info', 'aligned', 'aligned_png', 'aligned_info'):
            self.assertEqual(cam.pubs[key].messages[0].header.stamp, stamp, key)
            self.assertEqual(cam.pubs[key].messages[0].header.frame_id, 'optical', key)
        for key in ('depth', 'depth_png', 'depth_info', 'points'):
            self.assertEqual(cam.pubs[key].messages[0].header.stamp, stamp, key)
            self.assertEqual(cam.pubs[key].messages[0].header.frame_id, 'depth_optical', key)
        self.assertEqual(rgb.encoding, 'rgb8')
        self.assertEqual(depth.encoding, '16UC1')
        # The decimated depth stream is a separate render at the depth image size.
        rd = b._renderer[(16, 16)]
        rd.enable_depth_rendering()
        rd.update_scene(b.data, camera=0, scene_option=b._scene_opt)
        np.testing.assert_array_equal(
            np.frombuffer(bytes(cam.pubs['depth'].messages[0].data), dtype=np.uint16).reshape(16, 16),
            b._depth_mm(rd.render()))
        self.assertEqual((cam.pubs['depth_info'].messages[0].width, cam.pubs['depth'].messages[0].width), (32, 16))

    def test_compressed_topics_encode_only_for_subscribers(self):
        """Compressed topics are encoded only when they have subscribers; raw topics always publish."""
        b = self.bridge
        cam = b.cameras[0]
        for key in ('color_jpeg', 'aligned_png', 'depth_png'):
            cam.pubs[key].subscribers = 0
        b._publish_camera_frame(_sim_time_to_msg(b.data.time))
        for key in ('color_jpeg', 'aligned_png', 'depth_png'):
            self.assertEqual(cam.pubs[key].messages, [], key)
        for key in ('color', 'aligned', 'depth', 'points'):
            self.assertEqual(len(cam.pubs[key].messages), 1, key)

    def test_point_cloud_matches_the_real_driver_layout(self):
        """The camera point cloud has the real driver's field layout and one point per valid depth pixel."""
        b = self.bridge
        b._publish_camera_frame(_sim_time_to_msg(b.data.time))
        msg = b.cameras[0].pubs['points'].messages[0]
        self.assertEqual([(f.name, f.offset, f.datatype, f.count) for f in msg.fields],
                         [('x', 0, 7, 1), ('y', 4, 7, 1), ('z', 8, 7, 1), ('rgb', 16, 7, 1)])
        self.assertEqual((msg.height, msg.point_step, msg.is_dense), (1, 20, True))
        depth = np.frombuffer(bytes(b.cameras[0].pubs['depth'].messages[0].data), dtype=np.uint16)
        self.assertEqual(msg.width, int(np.count_nonzero(depth)))
        points = np.frombuffer(bytes(msg.data), dtype=rp.CLOUD_DTYPE)
        np.testing.assert_allclose(np.sort(points['z']), np.sort(depth[depth > 0] / 1000.0), rtol=1e-6)

    def test_camera_worker_matches_inline_frames_and_owns_renderer(self):
        """Frames from the camera worker thread match inline frames, and shutdown frees its renderer."""
        from concurrent.futures import ThreadPoolExecutor
        b = self.bridge
        stamp = _sim_time_to_msg(b.data.time)
        b._publish_camera_frame(stamp)
        inline = [bytes(b.cameras[0].pubs[k].messages[0].data) for k in ('color', 'aligned')]
        b._close_renderer()
        for pub in b.cameras[0].pubs.values():
            pub.messages.clear()
        b._camera_worker, b._camera_pending = ThreadPoolExecutor(max_workers=1), None
        threads = []
        original = mujoco.Renderer.render

        def render(renderer, *args, **kwargs):
            """Record the rendering thread."""
            threads.append(threading.current_thread())
            return original(renderer, *args, **kwargs)

        with patch.object(mujoco.Renderer, 'render', render):
            for _ in range(2):
                b._publish_camera_frame(stamp)
            b.shutdown()
        self.assertTrue(threads and all(t is not threading.main_thread() for t in threads))
        self.assertIsNone(b._renderer)
        cam = b.cameras[0]
        self.assertEqual(len(cam.pubs['color'].messages), 2)
        self.assertEqual([bytes(cam.pubs[k].messages[0].data) for k in ('color', 'aligned')], inline)

    def test_tick_preserves_sim_time_schedule_and_publication_order(self):
        """A tick publishes /clock, IMU and camera on their sim-time periods, then ticks the lidar, in that order."""
        b = self.bridge
        # The mujoco_sensors host publishes /clock, then ticks the suite; the suite's publishers
        # are replaced by event recorders.
        host = object.__new__(RosSensorBridge)
        host.suite, host.data, host.sim_lock, host.base_body_id = b, b.data, b.sim_lock, -1
        host.pub_clock = Recorder(b.sim_lock)
        events = []
        host.pub_clock.publish = lambda msg: events.append(('clock', msg.clock.sec, msg.clock.nanosec))
        b._publish_imu = lambda stamp: events.append(('imu', stamp.sec, stamp.nanosec))
        b._publish_camera_frame = lambda stamp: events.append(('camera', stamp.sec, stamp.nanosec))
        # The lidar is ticked with the state time of the data, one physics step before sim_t.
        b._lidar = SimpleNamespace(tick=lambda data, t: events.append(('lidar', round(t * 1e9))),
                                   close=lambda: None, stats=lambda: '')
        host.tick()
        self.assertEqual([e[0] for e in events], ['clock', 'imu', 'camera', 'lidar'])
        self.assertTrue(all(e[1:] == (1, 250000000) for e in events[:3]))
        self.assertEqual(events[3][1], round((1.25 - b.model.opt.timestep) * 1e9))
        events.clear()
        b.data.time = 1.251
        host.tick()
        self.assertEqual([e[0] for e in events], ['clock', 'lidar'])
        events.clear()
        b.data.time = 1.256
        host.tick()
        self.assertEqual([e[0] for e in events], ['clock', 'imu', 'lidar'])

    def test_packed_livox_scan_matches_rclpy_serialization(self):
        """The hand-packed CustomMsg is byte-identical to rclpy's serialisation, for 0 to 20161 points."""
        rng = np.random.default_rng(7)
        stamp = _sim_time_to_msg(3.217)
        for count in (0, 1, 2, 20161):
            points = rng.normal(size=(count, 3)).astype(np.float32)
            offsets = rng.integers(0, 100_000_000, count, dtype=np.uint32)
            lines = rng.integers(0, 4, count, dtype=np.uint8)
            reference = CustomMsg()
            reference.header.stamp = stamp
            reference.header.frame_id = 'livox_frame'
            reference.timebase = 3217000000
            reference.point_num = count
            reference.rsvd = [0, 0, 0]
            reference.points = [
                CustomPoint(offset_time=int(offsets[i]), x=float(points[i, 0]),
                            y=float(points[i, 1]), z=float(points[i, 2]), line=int(lines[i]))
                for i in range(count)
            ]
            packed = _serialize_livox_scan(stamp, 'livox_frame', 3217000000, offsets, points, lines)
            self.assertEqual(packed, serialize_message(reference), count)

# A closed room around a fixed torso that carries the lidar upside down, as the H1-2's does, and
# an arm on a slide joint whose hand sits in the lidar's field of view at qpos 0 and below the
# floor, out of every ray's way, at qpos -2.
LIDAR_ROOM = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .1" group="1"/>
    <geom type="box" pos="3.05 0 1.5" size=".05 3.1 1.5" group="1"/>
    <geom type="box" pos="-3.05 0 1.5" size=".05 3.1 1.5" group="1"/>
    <geom type="box" pos="0 3.05 1.5" size="3.1 .05 1.5" group="1"/>
    <geom type="box" pos="0 -3.05 1.5" size="3.1 .05 1.5" group="1"/>
    <body name="torso_link" pos="0 0 1">
      <geom name="shell" type="sphere" size=".08" group="1"/>
      <body name="livox_link" euler="180 0 0"/>
      <body name="arm_link" pos=".5 0 -.4">
        <joint name="arm" type="slide" axis="0 0 1"/>
        <geom name="hand" type="box" size=".1 .2 .2" group="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""
HAND_DOWN = -2.0   # m, the arm's qpos that puts the hand under the floor


def _lidar_backends():
    """Return the lidar backends to test: cpu, and gpu when a GPU renders (else say why not)."""
    try:
        mid360.GpuCaster(mujoco.MjModel.from_xml_string(LIDAR_ROOM), -1, (1,) * 6).close()
    except mid360.GpuUnavailable as e:
        print(f'  (gpu skipped: {e})')
        return ('cpu',)
    return ('cpu', 'gpu')


def _points(msg):
    """Return a deserialised CustomMsg's points (n, 3), lines and offset_time."""
    p = np.array([[q.x, q.y, q.z] for q in msg.points], np.float64).reshape(-1, 3)
    return (p, np.array([q.line for q in msg.points], np.uint8),
            np.array([q.offset_time for q in msg.points], np.uint32))


class LidarTests(unittest.TestCase):
    """Run the H1-2's MID-360 scan thread (_LidarScans) in LIDAR_ROOM, with recording publishers."""

    def setUp(self):
        """Build the room with the hand out of view, a sim lock and the body ids the lidar uses."""
        self.model = mujoco.MjModel.from_xml_string(LIDAR_ROOM)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0] = HAND_DOWN
        mujoco.mj_forward(self.model, self.data)
        self.lock = threading.Lock()
        body = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.lidar, self.torso = body('livox_link'), body('torso_link')
        self.robot = self_filter_geoms(self.model, ('torso_link', 'livox_link', 'arm_link'))

    def scans(self, backend='cpu', robot_geoms=None):
        """Return a started _LidarScans on the room, publishing to recorders; closed at cleanup."""
        self.cloud, self.custom = Recorder(self.lock), Recorder(self.lock)
        scans = _LidarScans(self.model, self.lidar, self.torso,
                            self.robot if robot_geoms is None else robot_geoms, backend, 'livox_frame',
                            self.custom.publish, self.cloud.publish, LoopTimings(), self.lock)
        self.addCleanup(scans.close)
        return scans

    def run_ticks(self, scans, seconds, start=0.0, step=None):
        """Tick scans every physics step from start to start + seconds (s of sim time), never skipping a cast.

        The loop outruns real time, so it waits for a free ring slot before each tick; step(k)
        may move the scene first. Waits for and returns the frame count the ticks complete.
        """
        dt = self.model.opt.timestep
        for k in range(round(seconds / dt) + 1):
            if step is not None:
                step(k)
            while scans._free.empty():
                time.sleep(0.0005)
            scans.tick(self.data, start + k * dt)
        frames = int(round(seconds * mid360.FRAME_RATE_HZ - 1e-9))
        end = time.time() + 30.0
        while len(self.custom.messages) < frames and time.time() < end:
            time.sleep(0.01)
        return frames

    def test_frames_are_the_mid360_pattern_on_both_topics(self):
        """Each 10 Hz frame fires 20,000 pattern rays on 4 lines; both topics carry it, stamped at its start.

        Every ray hits the room, so the published points are the frame's rays in order: point k
        lies along pattern row 20,000 f + k, on line k % 4. offset_time rises within the frame
        period (no cast waits here), and /livox/pointcloud holds the same points, lines and
        offsets (its timestamp field) under the same header.
        """
        pattern = mid360.load_pattern()
        for backend in _lidar_backends():
            with self.subTest(backend=backend):
                scans = self.scans(backend)
                self.assertEqual(scans.interval, scans.caster.batch_s)
                frames = self.run_ticks(scans, 0.3, start=2.0)
                self.assertEqual(len(self.custom.messages), frames)
                self.assertEqual(len(self.cloud.messages), frames)
                for f, (raw, pc2) in enumerate(zip(self.custom.messages, self.cloud.messages)):
                    msg = deserialize_message(raw, CustomMsg)
                    points, lines, offsets = _points(msg)
                    self.assertEqual(msg.point_num, 20_000)
                    stamp = msg.header.stamp
                    self.assertEqual(stamp.sec * 10**9 + stamp.nanosec, msg.timebase)
                    self.assertEqual(msg.timebase, round((2.0 + 0.1 * f) * 1e9))
                    self.assertEqual(msg.header.frame_id, 'livox_frame')
                    want = mid360.directions(pattern, 20_000 * f, 20_000)
                    got = points / np.linalg.norm(points, axis=1)[:, None]
                    np.testing.assert_allclose(got, want, atol=1e-5)
                    np.testing.assert_array_equal(lines, np.arange(20_000) % 4)
                    self.assertTrue(np.all(np.diff(offsets.astype(np.int64)) >= 0))
                    self.assertLessEqual(int(offsets[-1]), 100_000_000)
                    cloud = np.frombuffer(bytes(pc2.data), dtype=PC2_DTYPE)
                    self.assertEqual(pc2.header, msg.header)
                    self.assertEqual((pc2.width, pc2.height, pc2.row_step), (20_000, 1, 20_000 * 26))
                    np.testing.assert_array_equal(np.stack([cloud['x'], cloud['y'], cloud['z']], 1),
                                                  points.astype(np.float32))
                    np.testing.assert_array_equal(cloud['line'], lines)
                    np.testing.assert_array_equal(cloud['timestamp'], offsets.astype(np.float64))

    def test_scan_thread_matches_live_casts(self):
        """The thread's messages are byte-identical to a scanner casting on the live data at the same times.

        The hand moves through the view while the thread works from its ring of pose snapshots.
        """
        for backend in _lidar_backends():
            with self.subTest(backend=backend):
                scans = self.scans(backend)
                live = mid360.Mid360Scanner(mid360.make_caster(self.model, self.torso, (1,) * 6, backend,
                                                               robot_geoms=self.robot))
                self.addCleanup(live.caster.close)
                want = []
                dt = self.model.opt.timestep
                due = []

                def step(k):
                    """Move the hand and cast the live reference when the thread's cast is due."""
                    self.data.qpos[0] = HAND_DOWN + 2.0 * min(1.0, k * dt / 0.3)
                    mujoco.mj_forward(self.model, self.data)
                    t = k * dt
                    if not due or t >= due[-1] + scans.interval - 1e-9:
                        due.append(t)
                        pose = (self.data.xpos[self.lidar].copy(), self.data.xmat[self.lidar].reshape(3, 3).copy())
                        frame = live.update(mid360.Snapshot(*pose, self.data, self.model.geom_rgba[:, 3].copy(),
                                                            self.model.mat_rgba[:, 3].copy()), t)
                        if frame is not None:
                            want.append(frame)

                self.run_ticks(scans, 0.5, step=step)
                self.assertEqual(scans.waits, 0)
                self.assertEqual(len(want), 5)
                self.assertEqual(len(self.custom.messages), len(want))
                for raw, frame in zip(self.custom.messages, want):
                    msg = deserialize_message(raw, CustomMsg)
                    points, lines, offsets = _points(msg)
                    np.testing.assert_array_equal(points.astype(np.float32), frame.points)
                    np.testing.assert_array_equal(lines, frame.line)
                    np.testing.assert_array_equal(offsets, frame.offset_time)
                    self.assertEqual(msg.timebase, round(frame.timebase * 1e9))

    def test_self_filter_drops_the_robots_returns_but_keeps_its_shadow(self):
        """A ray whose first hit is a self-filtered geom gives no point; unfiltered, the hand gives points.

        The hand in view hides the floor behind it either way (GOLEM_LIDAR_SELF_FILTER=0 keeps
        the robot's returns).
        """
        self.data.qpos[0] = 0.0
        mujoco.mj_forward(self.model, self.data)
        hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'hand')
        none = np.zeros(self.model.ngeom, dtype=bool)
        for backend in _lidar_backends():
            for robot_geoms, filtered in ((self.robot, True), (none, False)):
                with self.subTest(backend=backend, filtered=filtered):
                    scans = self.scans(backend, robot_geoms)
                    self.run_ticks(scans, 0.1)
                    msg = deserialize_message(self.custom.messages[0], CustomMsg)
                    points, _, _ = _points(msg)
                    world = self.data.xpos[self.lidar] + points @ self.data.xmat[self.lidar].reshape(3, 3).T
                    local = (world - self.data.geom_xpos[hand]) @ self.data.geom_xmat[hand].reshape(3, 3)
                    on_hand = np.all(np.abs(local) < self.model.geom_size[hand] + 0.08, axis=1)
                    # The floor under the hand, seen past it from the sensor at (0, 0, 1).
                    shadow = (np.abs(world[:, 2]) < 0.1) & (np.abs(world[:, 1]) < 0.1) & \
                        (world[:, 0] > 1.3) & (world[:, 0] < 1.5)
                    self.assertEqual(int(shadow.sum()), 0)
                    if filtered:
                        self.assertEqual(int(on_hand.sum()), 0)
                        self.assertLess(msg.point_num, 20_000 - 100)
                    else:
                        self.assertGreater(int(on_hand.sum()), 100)

    def test_self_filter_geoms_by_body_prefix(self):
        """self_filter_geoms marks the geoms of bodies whose names start with a prefix; none without one."""
        mask = self_filter_geoms(self.model, ('arm_', 'torso'))
        names = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in np.flatnonzero(mask)}
        self.assertEqual(names, {'hand', 'shell'})
        self.assertFalse(self_filter_geoms(self.model, ()).any())

    def test_capture_holds_the_sim_lock_and_cast_and_publish_do_not(self):
        """tick() copies what a cast reads under the sim lock; the scan thread casts and publishes without it."""
        scans = self.scans()
        casts = []
        original = scans.caster.cast

        def cast(*args, **kwargs):
            """Record the casting thread and whether the sim lock is held."""
            casts.append((threading.current_thread(), self.lock.locked()))
            return original(*args, **kwargs)

        scans.caster.cast = cast
        held = []
        real = self.data

        class Spy:
            """Stand in for the data to record the lock's state when its fields are read."""

            def __getattr__(inner, name):
                held.append(self.lock.locked())
                return getattr(real, name)

        dt = self.model.opt.timestep
        for k in range(60):
            scans.tick(Spy(), k * dt)
        scans.close()
        self.assertTrue(held and all(held))
        self.assertFalse(self.lock.locked())
        self.assertTrue(casts and all(t is not threading.current_thread() and not locked for t, locked in casts))
        self.assertEqual(len(self.custom.messages), 1)   # Recorder fails a publish under the lock

    def test_suite_ticks_the_lidar_at_the_state_time_and_shuts_it_down(self):
        """H12Sensors.tick(sim_t) casts at sim_t - timestep, the state mj_step started from; shutdown closes the caster.

        So the first frame of ticks from sim_t = dt is stamped 0, and shutdown logs the counts.
        """
        b = object.__new__(H12Sensors)
        b.model, b.data, b.sim_lock = self.model, self.data, self.lock
        b.imu_period = b.cam_period = float('inf')
        b._last_imu_sim_t = b._last_cam_sim_t = 0.0
        b._camera_tf, b._camera_specs = [], []
        b._camera_worker, b._renderer = None, None
        b.get_logger = lambda: SimpleNamespace(warn=lambda text: self.fail(text))
        b._lidar = scans = self.scans()
        closed = []
        close = scans.caster.close
        scans.caster.close = lambda: (closed.append(True), close())
        dt = self.model.opt.timestep
        for step in range(1, 52):   # the last tick's state time is 0.1 s, which ends the first frame
            while scans._free.empty():
                time.sleep(0.0005)
            b.tick(step * dt, _sim_time_to_msg(step * dt))
        with patch('builtins.print') as printed:
            b.shutdown()
        self.assertEqual(closed, [True])
        msg = deserialize_message(self.custom.messages[0], CustomMsg)
        self.assertEqual((msg.header.stamp.sec, msg.header.stamp.nanosec, msg.timebase), (0, 0, 0))
        self.assertIn('1 lidar frames (0 of', b.stats())
        self.assertTrue(any('1 lidar frames' in str(c) for c in printed.call_args_list))


if __name__ == '__main__':
    unittest.main()
