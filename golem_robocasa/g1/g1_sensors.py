"""The G1's sensors as their drivers publish them: the Livox MID-360 lidar + IMU
and the head RealSense D435.

  /livox/lidar                               livox_ros_driver2/CustomMsg, 10 Hz, livox_frame
  /livox/imu                                 sensor_msgs/Imu, 200 Hz, livox_frame
  /realsense/head/color/image_raw            sensor_msgs/Image rgb8, 6 Hz, head_color_optical_frame
  /realsense/head/color/image_raw/compressed sensor_msgs/CompressedImage (JPEG)
  /realsense/head/aligned_depth_to_color/image_raw
                                             sensor_msgs/Image 16UC1 (mm)
  /realsense/head/aligned_depth_to_color/image_raw/compressedDepth
                                             sensor_msgs/CompressedImage (16UC1 PNG)
  /realsense/head/color/camera_info          sensor_msgs/CameraInfo
  /tf_static                                 d435_link -> head_color_optical_frame, with the camera

G1Sensors is the sensor suite of the mujoco_sensors node
(mujoco_ros_bridge.RosSensorBridge), which publishes /clock and the optional
ground-truth odom and calls tick() once per sim step after them.

Both sensors are on by default, as on the robot; SensorConfig lists the env
knobs that turn each off or lower its rate or resolution. g1_lidar.py and
g1_head_camera.py model them and g1_sensor_msgs.py packs their messages (see
those for the frames, units and formats). Like the Livox driver in CustomMsg
mode, the sim does not publish /livox/pointcloud. robot_state_publisher gives
torso_link -> mid360_link and -> d435_link, and golem_slam mid360_link ->
livox_frame; head_color_optical_frame stands in for the RealSense node's own
frames.

Stamps are sim time, of the state a sample measures. tick() runs after mj_step,
which leaves the kinematics, velocities and accelerations of the state it
started from, so the sensors are stamped one physics step (2 ms) before /clock.
The IMU's 200 Hz falls between the 2 ms physics steps: its samples alternate
4 and 6 ms apart, 200 per sim second.

Threads. tick() runs on the physics thread between steps, with the sim lock
held. The sensors' heavy work runs on two threads of their own, which never
touch the sim's MjData:
  - mid360_scan casts the lidar's rays (an exact ray costs 4 to 6 us, 50,000
    per sim second by default). Every LIDAR_CAST_EVERY physics steps (10 ms)
    the physics thread copies what a cast reads, every geom's pose and the
    sensor's, into a slot of a ring; the thread copies the slot into its own
    MjData, on its own copy of the model (cast() edits the model it is given,
    and mj_step reads the same field), and casts the rays due by then, labelled
    with that time. mj_multiRay releases the GIL, so the casts run beside the
    physics; casting every step instead would wake the thread 500 times a sim
    second, and its Python between casts would hold up the physics thread.
    The slot also carries the geoms' and materials' alpha, which RoboCasa tasks
    change at runtime (liquids, fruit, hidden walls) and the rays skip at 0.
    Other runtime edits of the model do not reach the thread's copy; RoboCasa's
    others (body poses, which reach the casts through the geom poses, sites and
    contact flags) do not change what a ray hits.
  - head_camera renders, converts, compresses and publishes each frame. At the
    camera's rate the physics thread copies what the camera sees into the
    renderer's scene (mjv_updateScene, which reads MjData: about 0.1 ms in the
    G1-only model, CPU only); the thread owns the renderer's GL context (it is
    thread-affine) and renders that scene, with the GIL released, so the render
    (15-20 ms a frame at 1280x720) does not hold up the physics or rt/lowstate
    (and g1_mujoco's 0.5 ms GIL switch interval keeps the thread's Python work
    from doing so). The scene is handed over, not copied: a frame that comes due
    while the thread still renders the last is dropped (and counted).
The IMU is read and published on the physics thread (tens of us a sample).
"""
import copy
import math
import os
import queue
import threading
import time
from typing import NamedTuple

import mujoco
import numpy as np
from geometry_msgs.msg import TransformStamped
from livox_ros_driver2.msg import CustomMsg
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu
from tf2_ros import StaticTransformBroadcaster

from . import g1_head_camera
from . import g1_lidar
from . import g1_sensor_msgs as sm
from mujoco_ros_bridge import _sim_time_to_msg

HEAD_CAMERA_NS = "/realsense/head"
HEAD_CAMERA_LINK = "d435_link"
# the real driver's QoS on both MID-360 topics: default (reliable), queue 256
LIVOX_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=256)
# physics steps between lidar casts: the points are cast from 100 poses per sim
# second (at the 2 ms step), each labelled with its pose's time
LIDAR_CAST_EVERY = 5
# casts the lidar's ring holds before the physics thread skips one
LIDAR_SLOTS = 20
MAX_CAMERA_HZ = 30.0   # the D435's top rate at 1280x720
CAMERA_START_S = 60.0   # wall s for the head camera's thread to make and try its renderer
WARN_PERIOD_S = 5.0   # wall s between repeats of one warning
_EPS_T = 1e-9   # s, tolerance on accumulated sim time


def _log(text):
    print(f"[g1_sensors] {text}", flush=True)


def _env_flag(env, name, default):
    raw = env.get(name, "").strip().lower()
    return default if raw == "" else raw not in ("0", "off", "false", "no")


def _env_number(env, name, default, kind):
    raw = env.get(name, "").strip()
    try:
        value = kind(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be a {kind.__name__}, got {raw!r}") from None
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive, got {raw!r}")
    return value


# default head-camera scale of the real 1280x720, the H1-2 sim's: a quarter of the
# pixels keeps the render inside the sim's frame time
CAMERA_SCALE = 0.5


class SensorConfig(NamedTuple):
    """The simulated sensors. from_env() reads them from these env knobs:
      GOLEM_LIDAR=0         no MID-360 (/livox/lidar, /livox/imu)
      GOLEM_LIDAR_POINTS    points per lidar frame (default 5000; the real sensor's
                            ~20,000 cost 4x as much, see g1_lidar.py)
      GOLEM_LIDAR_HZ        lidar frames per sim second (default 10, the driver's;
                            at most 100, the rate the rays are cast at)
      GOLEM_CAMERAS=0       no head camera (/realsense/head/*, its /tf_static)
      GOLEM_CAMERA_HZ       head camera frames per sim second (default 6; at most
                            30, the D435's top rate at 1280x720)
      GOLEM_CAMERA_SCALE    head camera size as a fraction of the real 1280x720
                            (default 0.5, as the H1-2 sim's): round(1280 s) x
                            round(720 s) with the same fovy, and camera_info to
                            match; 6 Hz at 1280x720 is the head camera's profile in
                            cl_realsense's launch file
    """
    lidar: bool = True
    lidar_points: int = g1_lidar.POINTS_PER_FRAME
    lidar_hz: float = g1_lidar.FRAME_RATE_HZ
    imu_hz: float = g1_lidar.IMU_RATE_HZ
    camera: bool = True
    camera_hz: float = 6.0
    camera_width: int = round(g1_head_camera.WIDTH * CAMERA_SCALE)
    camera_height: int = round(g1_head_camera.HEIGHT * CAMERA_SCALE)

    @classmethod
    def from_env(cls, env=os.environ):
        scale = _env_number(env, "GOLEM_CAMERA_SCALE", CAMERA_SCALE, float)
        width, height = round(g1_head_camera.WIDTH * scale), round(g1_head_camera.HEIGHT * scale)
        if width < 1 or height < 1:
            raise ValueError(f"GOLEM_CAMERA_SCALE must leave the head camera at least 1x1 pixels, "
                             f"got {scale:g} ({width}x{height})")
        lidar_hz = _env_number(env, "GOLEM_LIDAR_HZ", cls._field_defaults["lidar_hz"], float)
        if lidar_hz > 100.0:
            raise ValueError(f"GOLEM_LIDAR_HZ must be at most 100, got {lidar_hz:g}")
        camera_hz = _env_number(env, "GOLEM_CAMERA_HZ", cls._field_defaults["camera_hz"], float)
        if camera_hz > MAX_CAMERA_HZ:
            raise ValueError(f"GOLEM_CAMERA_HZ must be at most {MAX_CAMERA_HZ:g}, got {camera_hz:g}")
        return cls(lidar=_env_flag(env, "GOLEM_LIDAR", True),
                   lidar_points=_env_number(env, "GOLEM_LIDAR_POINTS", cls._field_defaults["lidar_points"], int),
                   lidar_hz=lidar_hz,
                   camera=_env_flag(env, "GOLEM_CAMERAS", True),
                   camera_hz=camera_hz,
                   camera_width=width, camera_height=height)

    def describe(self):
        lidar = (f"MID-360 {self.lidar_points} points x {self.lidar_hz:g} Hz + IMU {self.imu_hz:g} Hz"
                 if self.lidar else "no lidar (GOLEM_LIDAR=0)")
        camera = (f"head D435 {self.camera_width}x{self.camera_height} at {self.camera_hz:g} Hz"
                  if self.camera else "no head camera (GOLEM_CAMERAS=0)")
        return f"{lidar}; {camera}"


class _Schedule:
    """Sample times at rate_hz of sim time from the first tick, starting again
    when the sim time goes back (MuJoCo's reset)"""

    def __init__(self, rate_hz):
        self.period = 1.0 / rate_hz
        self._next = self._last = None

    def due(self, t):
        if self._next is None or t < self._last:
            self._next = t
        self._last = t
        if t < self._next - _EPS_T:
            return False
        self._next += self.period
        return True


class _LidarScans:
    """The MID-360's scans on the mid360_scan thread (see the module docstring);
    submit() on the physics thread, between steps"""

    def __init__(self, model, data, prefix, config, publish):
        self._live = model   # read on the physics thread only, for the alpha
        self.model = copy.copy(model)
        self.data = mujoco.MjData(self.model)
        self.mid360 = g1_lidar.Mid360(self.model, prefix, points_per_frame=config.lidar_points,
                                      rate_hz=config.lidar_hz)
        self._site = self.mid360.site_id
        self._publish = publish
        self._geom_xpos = np.empty((LIDAR_SLOTS,) + data.geom_xpos.shape)
        self._geom_xmat = np.empty((LIDAR_SLOTS,) + data.geom_xmat.shape)
        self._site_xpos = np.empty((LIDAR_SLOTS, 3))
        self._site_xmat = np.empty((LIDAR_SLOTS, 9))
        self._geom_alpha = np.empty((LIDAR_SLOTS, model.ngeom), model.geom_rgba.dtype)
        self._mat_alpha = np.empty((LIDAR_SLOTS, model.nmat), model.mat_rgba.dtype)
        self._free = queue.SimpleQueue()
        for slot in range(LIDAR_SLOTS):
            self._free.put(slot)
        self._work = queue.SimpleQueue()
        self.frames = self.skipped = 0
        self._errors = set()
        self._warned = -WARN_PERIOD_S
        self._thread = threading.Thread(target=self._run, name="mid360_scan", daemon=True)
        self._thread.start()

    def submit(self, data, t):
        """Hand the poses of data's kinematics, at sim time t, and the model's
        alpha to the thread; if it is LIDAR_SLOTS casts behind, skip this one:
        the next casts the rays due by then"""
        try:
            slot = self._free.get_nowait()
        except queue.Empty:
            self.skipped += 1
            if time.monotonic() - self._warned >= WARN_PERIOD_S:
                self._warned = time.monotonic()
                _log(f"WARNING: the lidar thread is {LIDAR_SLOTS} casts behind at sim t {t:.2f} s (short of "
                     f"CPU); skipping casts, the next one casts their rays ({self.skipped} skipped so far)")
            return
        np.copyto(self._geom_xpos[slot], data.geom_xpos)
        np.copyto(self._geom_xmat[slot], data.geom_xmat)
        self._site_xpos[slot] = data.site_xpos[self._site]
        self._site_xmat[slot] = data.site_xmat[self._site]
        self._geom_alpha[slot] = self._live.geom_rgba[:, 3]
        self._mat_alpha[slot] = self._live.mat_rgba[:, 3]
        self._work.put((slot, t))

    def _run(self):
        d = self.data
        while (item := self._work.get()) is not None:
            slot, t = item
            np.copyto(d.geom_xpos, self._geom_xpos[slot])
            np.copyto(d.geom_xmat, self._geom_xmat[slot])
            d.site_xpos[self._site] = self._site_xpos[slot]
            d.site_xmat[self._site] = self._site_xmat[slot]
            self.model.geom_rgba[:, 3] = self._geom_alpha[slot]
            self.model.mat_rgba[:, 3] = self._mat_alpha[slot]
            self._free.put(slot)
            try:
                frame = self.mid360.update(d, t)
                if frame is not None:
                    self._publish(sm.livox_custom_msg_cdr(frame))   # rclpy publishes CDR bytes as is
                    self.frames += 1
            except Exception as e:   # keep the sim running; say so once per kind
                if type(e) not in self._errors:
                    self._errors.add(type(e))
                    _log(f"WARNING: /livox/lidar scan failed at sim t {t:.3f}: {e!r}")

    def close(self):
        self._work.put(None)
        self._thread.join(timeout=5.0)


class _Mid360:
    """/livox/lidar (scans on their own thread) and /livox/imu (on the physics thread)"""

    def __init__(self, node, model, data, prefix, config):
        self.pub_lidar = node.create_publisher(CustomMsg, "/livox/lidar", LIVOX_QOS)
        self.pub_imu = node.create_publisher(Imu, "/livox/imu", LIVOX_QOS)
        self._imu = g1_lidar.Mid360(model, prefix)   # imu() only: it reads the live MjData
        self._imu_schedule = _Schedule(config.imu_hz)
        self.imu_samples = 0
        self._steps = 0
        self.scans = _LidarScans(model, data, prefix, config, self.pub_lidar.publish)

    def tick(self, data, t):
        if self._steps % LIDAR_CAST_EVERY == 0:
            self.scans.submit(data, t)
        self._steps += 1
        if self._imu_schedule.due(t):
            angular_velocity, specific_force = self._imu.imu(data)
            self.pub_imu.publish(sm.livox_imu_msg(t, angular_velocity, specific_force))
            self.imu_samples += 1

    def close(self):
        self.scans.close()

    def stats(self):
        return (f"{self.scans.frames} lidar frames ({self.scans.skipped} casts skipped), "
                f"{self.imu_samples} IMU samples")


class _HeadCamera:
    """/realsense/head/*: the physics thread fills the camera's scene at its rate;
    the head_camera thread renders, converts, compresses and publishes it"""

    def __init__(self, node, model, data, prefix, config):
        self.camera = g1_head_camera.HeadCamera(model, prefix, config.camera_width, config.camera_height)
        pub = {}
        for name, kind, topic in (("color", Image, "/color/image_raw"),
                                  ("depth", Image, "/aligned_depth_to_color/image_raw"),
                                  ("info", CameraInfo, "/color/camera_info"),
                                  ("color_jpeg", CompressedImage, "/color/image_raw/compressed"),
                                  ("depth_png", CompressedImage,
                                   "/aligned_depth_to_color/image_raw/compressedDepth")):
            pub[name] = node.create_publisher(kind, HEAD_CAMERA_NS + topic, qos_profile_sensor_data)
        self._pub = pub
        self._schedule = _Schedule(config.camera_hz)
        # set while the thread is not rendering the scene: only then does the
        # physics thread fill it; a frame due while it is clear is dropped
        self._idle = threading.Event()
        self._frames = queue.SimpleQueue()
        self.rendered = self.dropped = self.published = 0
        self._errors = set()
        # the thread makes and tries its renderer now, so a missing GL fails at startup
        started = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, args=(started,), name="head_camera", daemon=True)
        self._thread.start()
        try:
            error = started.get(timeout=CAMERA_START_S)
        except queue.Empty:
            error = TimeoutError(f"the head camera's renderer was not ready after {CAMERA_START_S:g} s")
        if error is not None:
            raise error
        # head_color_optical_frame on the URDF's camera link, as head_cam sits there
        tf = TransformStamped()
        tf.header.stamp = _sim_time_to_msg(float(data.time))
        tf.header.frame_id = HEAD_CAMERA_LINK
        tf.child_frame_id = sm.HEAD_OPTICAL_FRAME
        r = tf.transform.rotation
        r.x, r.y, r.z, r.w = g1_head_camera.OPTICAL_IN_D435_XYZW
        self._static_tf = StaticTransformBroadcaster(node)
        self._static_tf.sendTransform(tf)

    def tick(self, data, t):
        """On the physics thread: fill the scene from data at the camera's rate"""
        if not self._schedule.due(t):
            return
        if not self._idle.is_set():
            self.dropped += 1
            return
        self._idle.clear()
        try:
            self.camera.update_scene(data)
        except Exception:
            self._idle.set()
            raise
        self.rendered += 1
        self._frames.put(t)

    def _run(self, started):
        cam = self.camera
        try:
            cam.make_renderer()
            cam.render_scene()   # an empty scene: the GL context works
        except Exception as e:
            started.put(e)
            cam.close()
            return
        started.put(None)
        self._idle.set()
        while (t := self._frames.get()) is not None:
            try:
                rgb, depth = cam.render_scene()
            except Exception as e:   # keep the sim running; say so once per kind
                self._warn_once(e, f"WARNING: /realsense/head render failed at sim t {t:.3f}: {e!r}")
                continue
            finally:
                self._idle.set()
            try:
                depth = cam.depth_mm(depth)
                pub = self._pub
                pub["color"].publish(sm.color_image_msg(t, rgb))
                pub["depth"].publish(sm.depth_image_msg(t, depth))
                pub["info"].publish(sm.camera_info_msg(t, cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy))
                pub["color_jpeg"].publish(sm.color_compressed_msg(t, rgb))
                pub["depth_png"].publish(sm.depth_compressed_msg(t, depth))
                self.published += 1
            except Exception as e:
                self._warn_once(e, f"WARNING: /realsense/head frame failed at sim t {t:.3f}: {e!r}")
        cam.close()   # on this thread, which owns the GL context

    def _warn_once(self, error, text):
        if type(error) not in self._errors:
            self._errors.add(type(error))
            _log(text)

    def close(self):
        """After the last tick: the thread renders what is queued, then closes the renderer"""
        self._frames.put(None)
        self._thread.join(timeout=5.0)

    def stats(self):
        return (f"{self.published} head camera frames ({self.dropped} of {self.rendered + self.dropped} "
                f"due dropped)")


class G1Sensors:
    """The G1's sensor suite for the mujoco_sensors host: the MID-360 and the head
    camera, as config says (SensorConfig.from_env() by default). The camera's
    thread makes and tries its renderer here, so a missing GL fails at startup."""

    def __init__(self, node, model, data, prefix="robot0_", config=None):
        self.node = node
        self.model = model
        self.data = data
        config = SensorConfig.from_env() if config is None else config
        self.mid360 = _Mid360(node, model, data, prefix, config) if config.lidar else None
        self.head_camera = _HeadCamera(node, model, data, prefix, config) if config.camera else None

    def tick(self, sim_t, stamp):
        """Call once per sim step, on the physics thread after mj_step: the IMU
        and the lidar's and camera's scenes at their rates, keyed off sim time.
        The samples are stamped one physics step before sim_t, not with stamp."""
        # the state the sensors measure: the one mj_step started from
        t = sim_t - self.model.opt.timestep
        if self.mid360 is not None:
            self.mid360.tick(self.data, t)
        if self.head_camera is not None:
            try:
                self.head_camera.tick(self.data, t)
            except Exception as e:
                self.node.get_logger().warn(f"head camera scene update failed: {e}", throttle_duration_sec=5.0)

    def shutdown(self) -> None:
        """Stop the sensors' threads; on the physics thread, after the last tick()"""
        for sensor in (self.mid360, self.head_camera):
            if sensor is not None:
                sensor.close()
        if self.mid360 is not None or self.head_camera is not None:
            _log("sensors published " + ", ".join(s.stats() for s in (self.mid360, self.head_camera) if s))

    def stats(self):
        """The sensors' counts so far, for the sim's log"""
        return ", ".join(s.stats() for s in (self.mid360, self.head_camera) if s is not None)
