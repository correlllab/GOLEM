"""Simulate the G1's MID-360 lidar + IMU and head D435 as their drivers publish them.

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

As with image_transport on the robot, each compressed image is encoded only
while something subscribes to its topic.

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
  - mid360_scan casts the lidar's rays (20,000 per frame by default) through
    the caster mid360.make_caster() picks by GOLEM_LIDAR_BACKEND: depth renders
    on the GPU, else exact rays on the CPU. Every caster's batch_s of sim time
    (20 ms on the GPU, 10 ms on the CPU) the physics thread copies what a cast
    reads, every geom's pose and the sensor's, into a slot of a ring; the thread
    copies the slot into its own MjData and casts the rays due by then, labelled
    with that time. The caster casts on its own copy of the model, and both
    casters release the GIL while they cast, so the casts run beside the
    physics; casting every step instead would wake the thread 500 times a sim
    second, and its Python between casts would hold up the physics thread.
    The slot also carries the geoms' and materials' alpha, which RoboCasa tasks
    change at runtime (liquids, fruit, hidden walls) and the rays skip at 0.
    Other runtime edits of the model do not reach the caster's copy; RoboCasa's
    others (body poses, which reach the casts through the geom poses, sites and
    contact flags) do not change what a ray hits. The ring holds one frame of
    casts: when the thread falls that far behind, the physics thread waits for
    it, so a frame trails the sim by at most one frame period and the real-time
    factor drops instead (mid360.ScanRing).
  - head_camera renders, converts, compresses and publishes each frame. At the
    camera's rate the physics thread copies what the camera sees into the
    renderer's scene (mjv_updateScene, which reads MjData: about 0.1 ms in the
    G1-only model, CPU only); the thread owns the renderer's GL context (it is
    thread-affine) and renders that scene, with the GIL released, so the render
    (15-20 ms a frame at 1280x720) does not hold up the physics or rt/lowstate
    (and g1_mujoco's 0.5 ms GIL switch interval keeps the thread's Python work
    from doing so). It renders under mid360.GL_LOCK, taking turns with the GPU
    caster (g1_head_camera.py). The scene is handed over, not copied: a frame
    that comes due while the thread still renders the last is dropped (and
    counted).
The IMU is read and published on the physics thread (tens of us a sample).
"""
import math
import os
import queue
import threading
from typing import NamedTuple

from geometry_msgs.msg import TransformStamped
from livox_ros_driver2.msg import CustomMsg
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu
from tf2_ros import StaticTransformBroadcaster

import mid360
from . import g1_head_camera
from . import g1_lidar
from . import g1_sensor_msgs as sm
from mujoco_ros_bridge import _sim_time_to_msg

HEAD_CAMERA_NS = "/realsense/head"
HEAD_CAMERA_LINK = "d435_link"
# The real driver's QoS on both MID-360 topics: reliable, queue 256.
LIVOX_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=256)
MAX_CAMERA_HZ = 30.0   # the D435's top rate at 1280x720
CAMERA_START_S = 60.0   # wall s for the head camera's thread to make and try its renderer
_EPS_T = 1e-9   # s, tolerance on accumulated sim time


def _log(text):
    """Print a [g1_sensors] line."""
    print(f"[g1_sensors] {text}", flush=True)


def _env_flag(env, name, default):
    """Return the env flag name: default when unset or empty, off for 0/off/false/no."""
    raw = env.get(name, "").strip().lower()
    return default if raw == "" else raw not in ("0", "off", "false", "no")


def _env_number(env, name, default, kind):
    """Return the env knob name as kind, or default when unset or empty.

    Raises ValueError, naming the knob, for a value that is not a finite positive kind.
    """
    raw = env.get(name, "").strip()
    try:
        value = kind(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be a {kind.__name__}, got {raw!r}") from None
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive, got {raw!r}")
    return value


# The default head-camera scale of the real 1280x720, as in the H1-2 sim: a
# quarter of the pixels keeps the render inside the sim's frame time.
CAMERA_SCALE = 0.5


class SensorConfig(NamedTuple):
    """Configure the simulated sensors; from_env() reads these env knobs:
      GOLEM_LIDAR=0         no MID-360 (/livox/lidar, /livox/imu)
      GOLEM_LIDAR_POINTS    points per lidar frame (default 20,000, the real
                            sensor's), so the sensor fires GOLEM_LIDAR_POINTS x
                            GOLEM_LIDAR_HZ points per sim second (the real 200,000
                            at the defaults); fewer points cover the pattern
                            more sparsely
      GOLEM_LIDAR_HZ        lidar frames per sim second (default 10, the driver's;
                            at most 100)
      GOLEM_LIDAR_BACKEND   how the lidar's rays are cast (mid360.make_caster):
                            auto (default) renders them on the GPU where MuJoCo
                            renders through EGL on one, else casts them exactly on
                            the CPU, logging why; gpu stops the sim without a GPU;
                            cpu always casts on the CPU
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
    lidar_backend: str = "auto"
    imu_hz: float = g1_lidar.IMU_RATE_HZ
    camera: bool = True
    camera_hz: float = 6.0
    camera_width: int = round(g1_head_camera.WIDTH * CAMERA_SCALE)
    camera_height: int = round(g1_head_camera.HEIGHT * CAMERA_SCALE)

    @classmethod
    def from_env(cls, env=os.environ):
        """Return the config the env knobs give, raising ValueError for a bad value."""
        scale = _env_number(env, "GOLEM_CAMERA_SCALE", CAMERA_SCALE, float)
        width, height = round(g1_head_camera.WIDTH * scale), round(g1_head_camera.HEIGHT * scale)
        if width < 1 or height < 1:
            raise ValueError(f"GOLEM_CAMERA_SCALE must leave the head camera at least 1x1 pixels, "
                             f"got {scale:g} ({width}x{height})")
        lidar_hz = _env_number(env, "GOLEM_LIDAR_HZ", cls._field_defaults["lidar_hz"], float)
        if lidar_hz > 100.0:
            raise ValueError(f"GOLEM_LIDAR_HZ must be at most 100, got {lidar_hz:g}")
        lidar_backend = mid360.backend_from_env(env)
        camera_hz = _env_number(env, "GOLEM_CAMERA_HZ", cls._field_defaults["camera_hz"], float)
        if camera_hz > MAX_CAMERA_HZ:
            raise ValueError(f"GOLEM_CAMERA_HZ must be at most {MAX_CAMERA_HZ:g}, got {camera_hz:g}")
        return cls(lidar=_env_flag(env, "GOLEM_LIDAR", True),
                   lidar_points=_env_number(env, "GOLEM_LIDAR_POINTS", cls._field_defaults["lidar_points"], int),
                   lidar_hz=lidar_hz,
                   lidar_backend=lidar_backend,
                   camera=_env_flag(env, "GOLEM_CAMERAS", True),
                   camera_hz=camera_hz,
                   camera_width=width, camera_height=height)

    def describe(self):
        """Return a one-line summary for the sim's log."""
        lidar = (f"MID-360 {self.lidar_points} points x {self.lidar_hz:g} Hz (backend "
                 f"{self.lidar_backend}) + IMU {self.imu_hz:g} Hz"
                 if self.lidar else "no lidar (GOLEM_LIDAR=0)")
        camera = (f"head D435 {self.camera_width}x{self.camera_height} at {self.camera_hz:g} Hz"
                  if self.camera else "no head camera (GOLEM_CAMERAS=0)")
        return f"{lidar}; {camera}"


class _Schedule:
    """Schedule samples at rate_hz of sim time from the first tick.

    It starts again when the sim time goes back (MuJoCo's reset).
    """

    def __init__(self, rate_hz):
        """Schedule rate_hz samples per sim second."""
        self.period = 1.0 / rate_hz
        self._next = self._last = None

    def due(self, t):
        """Return whether a sample is due at sim time t (s), and if so schedule the next."""
        if self._next is None or t < self._last:
            self._next = t
        self._last = t
        if t < self._next - _EPS_T:
            return False
        self._next += self.period
        return True


class _LidarScans(mid360.ScanRing):
    """Cast the MID-360's scans on the mid360_scan thread (a mid360.ScanRing; see the module docstring)."""

    def __init__(self, model, data, prefix, config, publish):
        """Build the caster config.lidar_backend names and start the thread, which calls publish(cdr bytes).

        Raises RuntimeError when the backend is gpu and no GPU caster can be built.
        """
        self.mid360 = g1_lidar.Mid360(model, prefix)
        caster = self.mid360.make_caster(config.lidar_backend)
        # s of sim time between casts: the caster's batch, or each frame's period when that
        # is shorter, so that every frame is cast from a pose of its own.
        interval = min(caster.batch_s, 1.0 / config.lidar_hz)
        _log(f"lidar rays cast on the {'GPU' if isinstance(caster, mid360.GpuCaster) else 'CPU'} "
             f"every {interval * 1e3:g} ms of sim time")
        super().__init__(g1_lidar.make_scanner(caster, config.lidar_points, config.lidar_hz), model,
                         self.mid360.pose, lambda frame: publish(sm.livox_custom_msg_cdr(frame)),
                         interval, log=_log)


class _Mid360:
    """Publish /livox/lidar (scans on their own thread) and /livox/imu (on the physics thread)."""

    def __init__(self, node, model, data, prefix, config):
        """Create the publishers and start the scan thread."""
        self.pub_lidar = node.create_publisher(CustomMsg, "/livox/lidar", LIVOX_QOS)
        self.pub_imu = node.create_publisher(Imu, "/livox/imu", LIVOX_QOS)
        self._imu = g1_lidar.Mid360(model, prefix)   # for imu() on the live MjData only
        self._imu_schedule = _Schedule(config.imu_hz)
        self.imu_samples = 0
        self.scans = _LidarScans(model, data, prefix, config, self.pub_lidar.publish)

    def tick(self, data, t):
        """Submit a cast every scans.interval and publish the IMU when due, at sim time t."""
        self.scans.tick(data, t)
        if self._imu_schedule.due(t):
            angular_velocity, specific_force = self._imu.imu(data)
            self.pub_imu.publish(sm.livox_imu_msg(t, angular_velocity, specific_force))
            self.imu_samples += 1

    def close(self):
        """Stop the scan thread."""
        self.scans.close()

    def stats(self):
        """Return the lidar's counts (mid360.ScanRing.stats) and the IMU sample count."""
        return f"{self.scans.stats()}, {self.imu_samples} IMU samples"


class _HeadCamera:
    """Publish /realsense/head/*: the physics thread fills the scene, the head_camera thread renders it."""

    def __init__(self, node, model, data, prefix, config):
        """Start the head_camera thread and wait for its renderer, raising if it fails.

        Also latches d435_link -> head_color_optical_frame on /tf_static.
        """
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
        # Set while the thread is not rendering the scene: only then does the
        # physics thread fill it, and a frame due while it is clear is dropped.
        self._idle = threading.Event()
        self._frames = queue.SimpleQueue()
        self.rendered = self.dropped = self.published = 0
        self._errors = set()
        # The thread makes and tries its renderer now, so a missing GL fails at startup.
        started = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, args=(started,), name="head_camera", daemon=True)
        self._thread.start()
        try:
            error = started.get(timeout=CAMERA_START_S)
        except queue.Empty:
            error = TimeoutError(f"the head camera's renderer was not ready after {CAMERA_START_S:g} s")
        if error is not None:
            raise error
        # d435_link -> head_color_optical_frame: head_cam sits at d435_link's origin.
        tf = TransformStamped()
        tf.header.stamp = _sim_time_to_msg(float(data.time))
        tf.header.frame_id = HEAD_CAMERA_LINK
        tf.child_frame_id = sm.HEAD_OPTICAL_FRAME
        r = tf.transform.rotation
        r.x, r.y, r.z, r.w = g1_head_camera.OPTICAL_IN_D435_XYZW
        self._static_tf = StaticTransformBroadcaster(node)
        self._static_tf.sendTransform(tf)

    def tick(self, data, t):
        """Fill the scene from data when a frame is due at sim time t; on the physics thread."""
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
        """Make the renderer (reporting on started), then render and publish each queued frame."""
        cam = self.camera
        try:
            cam.make_renderer()
            cam.render_scene()   # an empty scene proves the GL context works
        except Exception as e:
            started.put(e)
            cam.close()
            return
        started.put(None)
        self._idle.set()
        while (t := self._frames.get()) is not None:
            try:
                rgb, depth = cam.render_scene()
            except Exception as e:   # keeps the sim running; logged once per kind
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
                # As the driver's image_transport plugins do, encode a compressed
                # image only while something subscribes to its topic.
                if pub["color_jpeg"].get_subscription_count() > 0:
                    pub["color_jpeg"].publish(sm.color_compressed_msg(t, rgb))
                if pub["depth_png"].get_subscription_count() > 0:
                    pub["depth_png"].publish(sm.depth_compressed_msg(t, depth))
                self.published += 1
            except Exception as e:
                self._warn_once(e, f"WARNING: /realsense/head frame failed at sim t {t:.3f}: {e!r}")
        cam.close()   # on this thread, which owns the GL context

    def _warn_once(self, error, text):
        """Log text the first time an error of this type occurs."""
        if type(error) not in self._errors:
            self._errors.add(type(error))
            _log(text)

    def close(self):
        """Stop the thread after it renders what is queued; call after the last tick."""
        self._frames.put(None)
        self._thread.join(timeout=5.0)

    def stats(self):
        """Return the published and dropped frame counts."""
        return (f"{self.published} head camera frames ({self.dropped} of {self.rendered + self.dropped} "
                f"due dropped)")


class G1Sensors:
    """Publish the G1's MID-360 and head camera on the mujoco_sensors node."""

    def __init__(self, node, model, data, prefix="robot0_", config=None):
        """Start the sensors config switches on (SensorConfig.from_env() by default).

        The camera's thread makes and tries its renderer here, so a missing GL
        fails at startup.
        """
        self.node = node
        self.model = model
        self.data = data
        config = SensorConfig.from_env() if config is None else config
        self.mid360 = _Mid360(node, model, data, prefix, config) if config.lidar else None
        self.head_camera = _HeadCamera(node, model, data, prefix, config) if config.camera else None

    def tick(self, sim_t, stamp):
        """Sample the IMU, lidar and camera when due; once per sim step, on the physics thread.

        The samples are stamped one physics step before sim_t (the state mj_step
        started from), not with stamp.
        """
        t = sim_t - self.model.opt.timestep
        if self.mid360 is not None:
            self.mid360.tick(self.data, t)
        if self.head_camera is not None:
            try:
                self.head_camera.tick(self.data, t)
            except Exception as e:
                self.node.get_logger().warn(f"head camera scene update failed: {e}", throttle_duration_sec=5.0)

    def shutdown(self) -> None:
        """Stop the sensors' threads and log their counts; on the physics thread, after the last tick()."""
        for sensor in (self.mid360, self.head_camera):
            if sensor is not None:
                sensor.close()
        if self.mid360 is not None or self.head_camera is not None:
            _log("sensors published " + ", ".join(s.stats() for s in (self.mid360, self.head_camera) if s))

    def stats(self):
        """Return the sensors' counts so far, for the sim's log."""
        return ", ".join(s.stats() for s in (self.mid360, self.head_camera) if s is not None)
