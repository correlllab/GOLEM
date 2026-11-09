"""Simulate the H1-2's RealSense cameras and MID-360 lidar + IMU as their drivers publish them.

Each camera publishes the topics, frames and TF of the real robot's RealSense
driver (realsense_profile.yaml / realsense_profile.py), under
/realsense/<namespace>/* (namespaces head, left_hand, right_hand). Shown for
`head`:
  /realsense/head/color/image_raw                               sensor_msgs/Image (rgb8)
  /realsense/head/color/image_raw/compressed                    sensor_msgs/CompressedImage (jpeg)
  /realsense/head/color/camera_info                             sensor_msgs/CameraInfo
  /realsense/head/aligned_depth_to_color/image_raw              sensor_msgs/Image (16UC1 mm)
  /realsense/head/aligned_depth_to_color/image_raw/compressedDepth
                                                                sensor_msgs/CompressedImage (16UC1 png)
  /realsense/head/aligned_depth_to_color/camera_info            sensor_msgs/CameraInfo
  /realsense/head/depth/image_rect_raw[/compressedDepth]        depth camera, decimated (depth_stream)
  /realsense/head/depth/camera_info                             sensor_msgs/CameraInfo
  /realsense/head/depth/color/points                            sensor_msgs/PointCloud2 (pointcloud)
  /tf_static, /tf                                               mount + driver camera frames
As with image_transport on the robot, the compressed topics are encoded only
while something subscribes to them. The MID-360, in its raw (upside-down)
livox_frame:
  /livox/lidar                                                  livox_ros_driver2/CustomMsg
  /livox/pointcloud                                             sensor_msgs/PointCloud2 (same scan, driver-native layout)
  /livox/imu                                                    sensor_msgs/Imu (the MJCF's livox_imu_site)

H12Sensors is the sensor suite of the mujoco_sensors node
(mujoco_ros_bridge.RosSensorBridge), which publishes /clock and the optional
ground-truth odom and calls tick() once per sim step after them. The IMU and
camera TF are published at their rates keyed off sim time; camera frames and
lidar scans are rate-throttled the same way and produced from MjData snapshots
on worker threads. MuJoCo's renderer GL context is thread-affine: it is
created, used and freed on the thread that renders (see _camera_worker).
"""
import copy
import io
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from array import array as _array
from collections import namedtuple
from contextlib import nullcontext

from sim_timing import LoopTimings

import mujoco
import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from PIL import Image as PILImage
from geometry_msgs.msg import TransformStamped
from livox_ros_driver2.msg import CustomMsg
from rclpy.serialization import serialize_message
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

import realsense_profile
from mujoco_ros_bridge import _sim_time_to_msg


# /livox/pointcloud's point, as livox_ros_driver2 lays it out with
# xfer_format 0 (LivoxPointXyzrtlt in its lddc.cpp): 26 bytes, unaligned.
PC2_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
        ("tag", np.uint8),
        ("line", np.uint8),
        ("timestamp", np.float64),
    ],
    align=False,
)
assert PC2_DTYPE.itemsize == 26
PC2_POINT_STEP = PC2_DTYPE.itemsize
PC2_FIELDS = [
    PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="tag",       offset=16, datatype=PointField.UINT8,   count=1),
    PointField(name="line",      offset=17, datatype=PointField.UINT8,   count=1),
    PointField(name="timestamp", offset=18, datatype=PointField.FLOAT64, count=1),
]


# livox_ros_driver2/CustomPoint in CDR: uint32 offset_time, float32 x/y/z, then
# uint8 reflectivity/tag/line. The next element's uint32 is 4-byte aligned, so
# consecutive points are 20 bytes apart and only the last point omits the pad.
LIVOX_POINT_DTYPE = np.dtype([
    ("offset_time", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("reflectivity", "u1"), ("tag", "u1"), ("line", "u1"), ("pad", "u1"),
])

# The MID-360's elevation span (deg) in its raw frame, the MJCF's livox_link. The
# sensor hangs upside down, so in the world the fan covers about -52 to +7 deg.
LIDAR_ELEVATION_DEG = (-7.0, 52.0)


def lidar_fan(az_rays: int, el_rays: int) -> np.ndarray:
    """Return unit ray directions (az_rays * el_rays, 3) in the lidar's raw frame.

    A uniform azimuth x elevation grid, elevation-major and azimuth-fast.
    """
    az = np.linspace(0.0, 2.0 * np.pi, az_rays, endpoint=False, dtype=np.float64)
    low, high = np.deg2rad(LIDAR_ELEVATION_DEG)
    el = np.linspace(low, high, el_rays, endpoint=False, dtype=np.float64)
    az_grid, el_grid = np.meshgrid(az, el, indexing="xy")
    cos_el = np.cos(el_grid)
    return np.stack(
        [cos_el * np.cos(az_grid), cos_el * np.sin(az_grid), np.sin(el_grid)],
        axis=-1,
    ).reshape(-1, 3)


def _serialize_livox_scan(stamp, frame_id, timebase, offset_time, points, lines) -> bytes:
    """Return a /livox/lidar CustomMsg serialized, without one Python object per point.

    rclpy serializes the header and fixed fields of a point-free message; the
    empty points sequence is its trailing uint32 length, replaced by the count
    and followed by the packed points. The bytes are those of the full message.
    stamp is a ROS time, timebase ns, offset_time ns per point.
    """
    msg = CustomMsg()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.timebase = int(timebase)
    msg.point_num = int(points.shape[0])
    msg.lidar_id = 0
    msg.rsvd = [0, 0, 0]
    prefix = serialize_message(msg)
    packed = np.zeros(msg.point_num, dtype=LIVOX_POINT_DTYPE)
    packed["offset_time"] = offset_time
    packed["x"], packed["y"], packed["z"] = points[:, 0], points[:, 1], points[:, 2]
    packed["line"] = lines
    return prefix[:-4] + struct.pack("<I", msg.point_num) + packed.tobytes()[:-1]


# One RealSense camera: its realsense_profile.CameraSpec, MuJoCo camera ids,
# camera_info messages and publishers by stream key (a stream switched off has
# none).
_CameraPub = namedtuple("_CameraPub", "spec color_id depth_id color_info depth_info pubs")

# The real driver's compressedDepth ConfigHeader for 16UC1 PNG, byte for byte
# as captured from the robot: compression format 0 (PNG), then two 32-bit
# depth-quantization words, the first 9 and the second 0.
_COMPRESSED_DEPTH_HEADER = struct.pack("<III", 0, 9, 0)
_JPEG_QUALITY = 95   # image_transport's compressed default, as on the robot


# Preallocated MjData the camera frames are captured into: one being rendered
# while the next is filled.
_CAMERA_SNAPSHOTS = 2


def _camera_snapshot_fields(model):
    """Return the MjData fields mujoco.Renderer.update_scene reads for the geoms, cameras and lights shown.

    Geoms, fixed, tracking and body cameras, and lights come from the world
    poses of kinematics; tendons (drawn by default) also from their wrap points.
    """
    fields = ("geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat", "light_xpos", "light_xdir")
    if model.ntendon:
        fields += ("ten_wrapadr", "ten_wrapnum", "wrap_xpos", "wrap_obj", "ten_length")
    return fields


# depth/color/points as the real driver lays it out (x y z, pad, rgb).
CLOUD_FIELDS = [
    PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1),
]


def _camera_info(intr, frame_id: str) -> CameraInfo:
    """Return the distortion-free camera_info of a realsense_profile.Intrinsics."""
    info = CameraInfo()
    info.width = intr.width
    info.height = intr.height
    info.distortion_model = "plumb_bob"
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.k = [intr.fx, 0.0, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [intr.fx, 0.0, intr.cx, 0.0, 0.0, intr.fy, intr.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    info.header.frame_id = frame_id
    return info


class H12Sensors:
    """Publish the H1-2's cameras, lidar and IMU on the mujoco_sensors node.

    It holds sim_lock only while it reads MuJoCo state, never while it
    renders, encodes or publishes.
    """

    def __init__(
        self,
        node,
        model,
        data,
        cameras=(),
        camera_tf=(),
        camera_tf_rate_hz: float = 10.0,
        depth_stream: bool = True,
        pointcloud: bool = True,
        lidar_body: str = "livox_link",
        lidar_frame: str = "livox_frame",
        lidar_exclude_body: str = "torso_link",
        cam_rate_hz: float = 15.0,
        lidar_az_rays: int = 180,
        lidar_el_rays: int = 28,
        lidar_rate_hz: float = 5.0,
        lidar_max_range: float = 30.0,
        lidar_min_range: float = 0.1,
        imu_quat_sensor: str = "livox_imu_quat",
        imu_gyro_sensor: str = "livox_imu_gyro",
        imu_acc_sensor: str = "livox_imu_acc",
        imu_frame: str = "livox_frame",
        imu_rate_hz: float = 100.0,
        lidar_self_prefixes: tuple = (),
        sim_lock=None,
        timings=None,
    ):
        """Build the sensors; raise RuntimeError for a body, camera or sensor missing from model.

        cameras are realsense_profile.CameraSpec and camera_tf the profile's TF
        edges (realsense_profile.tf_edges), the driver's re-published at
        camera_tf_rate_hz. depth_stream and pointcloud switch the depth stream
        and depth/color/points on. lidar_exclude_body is the body the lidar
        rides in, whose geoms the rays pass through; geoms of bodies named with
        a lidar_self_prefixes prefix give no return. Rates are Hz of sim time,
        ranges m.
        """
        self.node = node
        self.model = model
        self.data = data
        self.sim_lock = sim_lock
        self.timings = timings if timings is not None else LoopTimings()

        self.lidar_frame = lidar_frame

        self._camera_specs = list(cameras)
        self.depth_stream = bool(depth_stream)
        self.pointcloud = bool(pointcloud)

        self.lidar_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lidar_body)
        if self.lidar_body_id < 0:
            raise RuntimeError(f"body '{lidar_body}' not found in MJCF")
        # The rays pass through the torso's own geoms (mj_multiRay's
        # bodyexclude): the lidar sits inside its shell. Limbs still occlude.
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lidar_exclude_body)
        if torso_id < 0:
            raise RuntimeError(f"body '{lidar_exclude_body}' not found in MJCF")
        self.lidar_exclude_body_id = int(torso_id)
        # Rays whose first hit is one of these geoms give no point: the limbs
        # still occlude what is behind them, but map no obstacle under the robot.
        self.geom_is_robot = np.zeros(model.ngeom, dtype=bool)
        if lidar_self_prefixes:
            _pfx = tuple(lidar_self_prefixes)
            for g in range(model.ngeom):
                bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                       int(model.geom_bodyid[g])) or ""
                if bn.startswith(_pfx):
                    self.geom_is_robot[g] = True
        # mj_multiRay takes a (6, 1) column vector here.
        self.lidar_geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8).reshape(6, 1)

        self.cam_period = 1.0 / cam_rate_hz
        # m, MuJoCo's far plane: depth there means no surface.
        self.zfar = float(model.vis.map.zfar * model.stat.extent)

        self.lidar_period = 1.0 / lidar_rate_hz
        self.lidar_max_range = lidar_max_range
        self.lidar_min_range = lidar_min_range
        # Rays and points are in the sensor's raw frame. The uniform grid
        # stands in for the MID-360's non-repetitive scan pattern.
        self.lidar_local_dirs = lidar_fan(lidar_az_rays, lidar_el_rays)
        self.lidar_rays = self.lidar_local_dirs.shape[0]
        # Line number = elevation index mod 6, spreading the points evenly over
        # the lines FAST-LIO's N_SCANS = 6 accepts.
        self.lidar_lines = ((np.arange(self.lidar_rays) // lidar_az_rays) % 6).astype(np.uint8)
        # ns, each point's offset_time in the scan, for FAST-LIO's deskew: the
        # azimuth sweeps once per scan period at a constant rate, and every
        # elevation at one azimuth fires together.
        az_idx = np.arange(self.lidar_rays) % lidar_az_rays
        self.lidar_offset_time_ns = (
            (az_idx / lidar_az_rays) * self.lidar_period * 1e9).astype(np.uint32)
        # mj_multiRay's output buffers; mujoco 3.3.1's binding takes (n, 1)
        # column vectors for them.
        self._lidar_dists = np.full((self.lidar_rays, 1), -1.0, dtype=np.float64)
        self._lidar_geomids = np.full((self.lidar_rays, 1), -1, dtype=np.int32)

        self.cameras = [self._make_camera_pub(spec) for spec in self._camera_specs]

        # Every edge is latched on /tf_static once, as the robot's mounts and
        # the driver do; the driver's own edges also go on /tf at
        # camera_tf_rate_hz, as it re-publishes them.
        self._camera_tf = list(camera_tf)
        self._camera_tf_period = 1.0 / camera_tf_rate_hz
        self._last_camera_tf_sim_t = -float("inf")
        if self._camera_tf:
            self._static_tf = StaticTransformBroadcaster(self.node)
            self._static_tf.sendTransform(
                [self._tf_msg(e, _sim_time_to_msg(float(data.time))) for e in self._camera_tf])
            self._camera_tf_broadcaster = TransformBroadcaster(self.node)
        # Reliable, as FAST-LIO subscribes (laserMapping.cpp); a best-effort
        # publisher would not match it.
        self.pub_lidar = self.node.create_publisher(
            CustomMsg, "/livox/lidar",
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=20))
        self.pub_pc2 = self.node.create_publisher(
            PointCloud2, "/livox/pointcloud",
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=20))
        self.pub_imu = self.node.create_publisher(
            Imu, "/livox/imu",
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=10))

        self.imu_frame = imu_frame
        self.imu_period = 1.0 / imu_rate_hz
        self.imu_quat_adr, self.imu_quat_dim = self._sensor_adr(imu_quat_sensor)
        self.imu_gyro_adr, self.imu_gyro_dim = self._sensor_adr(imu_gyro_sensor)
        self.imu_acc_adr, self.imu_acc_dim = self._sensor_adr(imu_acc_sensor)

        # One renderer per render size, made on the rendering thread.
        self._renderer = None

        # s of sim time, so the rates do not depend on the real-time factor.
        self._last_cam_sim_t = 0.0
        self._last_lidar_sim_t = 0.0
        self._last_imu_sim_t = 0.0
        self._lidar_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lidar")
        self._lidar_pending = None
        # A GLFW context lives on the main thread; EGL and OSMesa contexts can
        # live on a worker thread, which renders while physics runs.
        self._camera_worker = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera")
            if os.environ.get("MUJOCO_GL", "").strip().lower() in ("egl", "osmesa") else None)
        self._camera_pending = None
        if self._camera_worker is not None and self.cameras:
            # Make the EGL context now: once the passive viewer's GLFW window
            # exists, a new EGL context can no longer be made current.
            self._camera_worker.submit(self._ensure_renderer).result()

    def _make_camera_pub(self, spec) -> _CameraPub:
        """Return a camera's MuJoCo ids, camera_info and publishers for the streams switched on."""
        ids = []
        for name in (spec.color_camera, spec.depth_camera):
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cam_id < 0:
                raise RuntimeError(f"camera '{name}' not found in MJCF")
            ids.append(cam_id)
        base = f"/realsense/{spec.ns}"
        topics = {
            "color": (Image, "color/image_raw"),
            "color_jpeg": (CompressedImage, "color/image_raw/compressed"),
            "color_info": (CameraInfo, "color/camera_info"),
            "aligned": (Image, "aligned_depth_to_color/image_raw"),
            "aligned_png": (CompressedImage, "aligned_depth_to_color/image_raw/compressedDepth"),
            "aligned_info": (CameraInfo, "aligned_depth_to_color/camera_info"),
        }
        if self.depth_stream:
            topics.update({
                "depth": (Image, "depth/image_rect_raw"),
                "depth_png": (CompressedImage, "depth/image_rect_raw/compressedDepth"),
                "depth_info": (CameraInfo, "depth/camera_info"),
            })
        if self.pointcloud:
            topics["points"] = (PointCloud2, "depth/color/points")
        pubs = {key: self.node.create_publisher(typ, f"{base}/{suffix}", qos_profile_sensor_data)
                for key, (typ, suffix) in topics.items()}
        return _CameraPub(spec, ids[0], ids[1], _camera_info(spec.color, spec.color_frame),
                          _camera_info(spec.depth_info, spec.depth_frame), pubs)

    @staticmethod
    def _tf_msg(edge, stamp) -> TransformStamped:
        """Return a profile TF edge as a TransformStamped at stamp."""
        parent, child, xyz, q, _ = edge
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = q
        return t

    def _sensor_adr(self, name: str):
        """Return (sensordata address, dimension) of the named sensor."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f"sensor '{name}' not found in MJCF")
        return int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])

    def get_logger(self):
        """Return the host node's logger."""
        return self.node.get_logger()

    def stats(self) -> str:
        """Return "": every scheduled frame and scan is published, so there is nothing to report."""
        return ""

    def shutdown(self) -> None:
        """Stop the worker threads and free the renderers on the thread that owns them."""
        if getattr(self, "_lidar_worker", None) is not None:
            self._lidar_worker.shutdown(wait=True)
            self._lidar_worker = None
        camera_worker = getattr(self, "_camera_worker", None)
        if camera_worker is not None:
            camera_worker.submit(self._close_renderer).result()
            camera_worker.shutdown(wait=True)
            self._camera_worker = None
        else:
            self._close_renderer()

    def _close_renderer(self) -> None:
        """Close every renderer, on the thread that made them."""
        for r in (self._renderer or {}).values():
            try:
                r.close()
            except Exception:
                pass
        self._renderer = None

    def tick(self, sim_t: float, stamp: TimeMsg) -> None:
        """Publish the IMU, camera TF, camera frames and lidar scan due at sim_t (s).

        The host calls it once per sim step, after /clock and odom; stamp is
        sim_t as a ROS time. Camera frames and lidar scans come from MjData
        snapshots on worker threads, one job in flight per sensor (the cameras
        stay on this thread under GLFW). Call it without holding sim_lock, on
        the thread that steps the physics.
        """
        if sim_t - self._last_imu_sim_t >= self.imu_period:
            self._last_imu_sim_t = sim_t
            try:
                self._publish_imu(stamp)
            except Exception as e:
                self.get_logger().warn(f"imu publish failed: {e}")

        if self._camera_tf and sim_t - self._last_camera_tf_sim_t >= self._camera_tf_period:
            self._last_camera_tf_sim_t = sim_t
            self._camera_tf_broadcaster.sendTransform(
                [self._tf_msg(e, stamp) for e in self._camera_tf if e[4]])

        if self._camera_specs and sim_t - self._last_cam_sim_t >= self.cam_period:
            self._last_cam_sim_t = sim_t
            try:
                self._publish_camera_frame(stamp)
            except Exception as e:
                self.get_logger().warn(f"camera publish failed: {e}")

        if sim_t - self._last_lidar_sim_t >= self.lidar_period:
            self._last_lidar_sim_t = sim_t
            try:
                self._publish_lidar_scan(stamp)
            except Exception as e:
                self.get_logger().warn(f"lidar publish failed: {e}")

    def _publish_imu(self, stamp: TimeMsg) -> None:
        """Publish /livox/imu from the IMU site's sensors."""
        with self.sim_lock or nullcontext():
            sd = self.data.sensordata
            # framequat is (w, x, y, z) in the world; gyro and accelerometer are
            # in the site's frame (rad/s, m/s^2).
            q = sd[self.imu_quat_adr : self.imu_quat_adr + self.imu_quat_dim].copy()
            w = sd[self.imu_gyro_adr : self.imu_gyro_adr + self.imu_gyro_dim].copy()
            a = sd[self.imu_acc_adr : self.imu_acc_adr + self.imu_acc_dim].copy()

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.imu_frame
        msg.orientation.w = float(q[0])
        msg.orientation.x = float(q[1])
        msg.orientation.y = float(q[2])
        msg.orientation.z = float(q[3])
        msg.angular_velocity.x = float(w[0])
        msg.angular_velocity.y = float(w[1])
        msg.angular_velocity.z = float(w[2])
        msg.linear_acceleration.x = float(a[0])
        msg.linear_acceleration.y = float(a[1])
        msg.linear_acceleration.z = float(a[2])
        self.pub_imu.publish(msg)

    def _publish_camera_frame(self, stamp: TimeMsg) -> None:
        """Snapshot MjData and render and publish every camera from it, on the worker when there is one."""
        capture_start = self.timings.start()
        snapshot = self._capture_camera_snapshot()
        self.timings.stop("camera_capture", capture_start)
        worker = getattr(self, "_camera_worker", None)
        if worker is None:
            self._render_and_publish_cameras(stamp, snapshot)
            return
        # One frame set in flight at most: wait rather than drop, so every
        # scheduled frame is published. result() re-raises the last one's failure.
        wait_start = self.timings.start()
        pending, self._camera_pending = self._camera_pending, None
        if pending is not None:
            pending.result()
        self.timings.stop("camera_wait", wait_start)
        self._camera_pending = worker.submit(self._render_and_publish_cameras, stamp, snapshot)

    def _capture_camera_snapshot(self):
        """Return an MjData holding what the cameras' scenes read of self.data, taken under the sim lock.

        It is one of _CAMERA_SNAPSHOTS preallocated MjData, filled with
        _camera_snapshot_fields() only, in turn. This runs while the previous
        frame's job may still render, but the one before that finished before the
        previous was submitted, so the slot reused here is idle. A model with
        flexes or skins, whose scenes read more than those fields, takes a full
        copy of self.data instead.
        """
        slots = getattr(self, "_camera_slots", None)
        if slots is None:
            full_copy = self.model.nflex > 0 or self.model.nskin > 0
            slots = self._camera_slots = [] if full_copy else [
                mujoco.MjData(self.model) for _ in range(_CAMERA_SNAPSHOTS)]
            self._camera_slot_count = 0
        with self.sim_lock or nullcontext():
            if not slots:
                return copy.copy(self.data)
            snapshot = slots[self._camera_slot_count % len(slots)]
            self._camera_slot_count += 1
            for field in _camera_snapshot_fields(self.model):
                getattr(snapshot, field)[:] = getattr(self.data, field)
        return snapshot

    def _render_sizes(self):
        """Return the (width, height) of every render the cameras need."""
        sizes = {(c.spec.color.width, c.spec.color.height) for c in self.cameras}
        if self.depth_stream or self.pointcloud:
            sizes |= {(c.spec.depth_image.width, c.spec.depth_image.height) for c in self.cameras}
        return sizes

    def _ensure_renderer(self) -> None:
        """Make the renderers and scene options on the calling thread, once."""
        if self._renderer is not None:
            return
        sizes = self._render_sizes()
        if sizes:
            with self.sim_lock or nullcontext():
                vis = self.model.vis.global_
                vis.offwidth = max(vis.offwidth, max(w for w, _ in sizes))
                vis.offheight = max(vis.offheight, max(h for _, h in sizes))
        self._renderer = {(w, h): mujoco.Renderer(self.model, height=h, width=w) for w, h in sizes}
        self._buffers = {(w, h): (np.empty((h, w, 3), dtype=np.uint8), np.empty((h, w), dtype=np.float32))
                         for w, h in sizes}
        # Shown: geom group 1 (the H1-2's visual meshes) and 2 (the MAGPIE
        # visuals and wrist mounts). Hidden: 0 (collision primitives) and 3
        # (MAGPIE collision meshes and finger pads).
        self._scene_opt = mujoco.MjvOption()
        self._scene_opt.geomgroup[:] = 0
        self._scene_opt.geomgroup[1] = 1
        self._scene_opt.geomgroup[2] = 1
        # Sites are hidden: a hand camera looks straight at robosuite's
        # grip_site, which is faint in colour but solid in depth.
        self._scene_opt.sitegroup[:] = 0

    def _depth_mm(self, depth):
        """Return depth (m) as 16UC1 mm, with no surface (far plane, non-finite, <= 0) as 0.

        0 is the RealSense's "no depth", which compressed_depth honours.
        """
        invalid = ~np.isfinite(depth) | (depth <= 0) | (depth >= 0.999 * self.zfar)
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        depth_mm[invalid] = 0
        return depth_mm

    @staticmethod
    def _image_msg(stamp, frame, pixels, encoding, bytes_per_pixel) -> Image:
        """Return a sensor_msgs/Image of pixels."""
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.height, msg.width = pixels.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = msg.width * bytes_per_pixel
        # rclpy checks bytes one element at a time (~17 ms a 320x240 colour and
        # depth frame); an array.array('B') is copied as is.
        msg.data = _array("B", np.ascontiguousarray(pixels).tobytes())
        return msg

    @staticmethod
    def _compressed_depth_msg(stamp, frame, depth_mm) -> CompressedImage:
        """Return depth_mm (16UC1) as compressedDepth: the 12-byte ConfigHeader, then a PNG."""
        png = io.BytesIO()
        PILImage.fromarray(depth_mm, mode="I;16").save(png, format="PNG")
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.format = "16UC1; compressedDepth"
        msg.data = _array("B", _COMPRESSED_DEPTH_HEADER + png.getvalue())
        return msg

    def _publish_info(self, pub, info, stamp) -> None:
        """Publish camera_info info stamped at stamp."""
        info.header.stamp = stamp
        pub.publish(info)

    def _render_and_publish_cameras(self, stamp: TimeMsg, data) -> None:
        """Render and publish every camera from the MjData snapshot data.

        The renderers' GL contexts are made, used and freed on the calling
        thread only (the camera worker when there is one). Per camera, the
        colour camera renders colour and aligned depth from one scene update,
        and the depth camera (the colour camera itself on a D405) the decimated
        depth stream and the point cloud.
        """
        self._ensure_renderer()
        for cam in self.cameras:
            spec, pubs = cam.spec, cam.pubs
            render_start = self.timings.start()
            size = (spec.color.width, spec.color.height)
            r, (rgb_buf, depth_buf) = self._renderer[size], self._buffers[size]
            r.disable_depth_rendering()
            r.update_scene(data, camera=cam.color_id, scene_option=self._scene_opt)
            rgb_u8 = r.render(out=rgb_buf)
            r.enable_depth_rendering()
            aligned_mm = self._depth_mm(r.render(out=depth_buf))
            self.timings.stop("camera_render", render_start)

            depth_mm = None
            if self.depth_stream or self.pointcloud:
                depth_start = self.timings.start()
                size = (spec.depth_image.width, spec.depth_image.height)
                rd, (_, ddepth_buf) = self._renderer[size], self._buffers[size]
                rd.enable_depth_rendering()
                rd.update_scene(data, camera=cam.depth_id, scene_option=self._scene_opt)
                depth_mm = self._depth_mm(rd.render(out=ddepth_buf))
                self.timings.stop("camera_depth_render", depth_start)

            publish_start = self.timings.start()
            pubs["color"].publish(self._image_msg(stamp, spec.color_frame, rgb_u8, "rgb8", 3))
            if pubs["color_jpeg"].get_subscription_count() > 0:
                jpeg = io.BytesIO()
                PILImage.fromarray(rgb_u8, mode="RGB").save(jpeg, format="JPEG", quality=_JPEG_QUALITY)
                msg = CompressedImage()
                msg.header.stamp = stamp
                msg.header.frame_id = spec.color_frame
                msg.format = "rgb8; jpeg compressed bgr8"
                msg.data = _array("B", jpeg.getvalue())
                pubs["color_jpeg"].publish(msg)
            self._publish_info(pubs["color_info"], cam.color_info, stamp)

            pubs["aligned"].publish(self._image_msg(stamp, spec.color_frame, aligned_mm, "16UC1", 2))
            if pubs["aligned_png"].get_subscription_count() > 0:
                pubs["aligned_png"].publish(self._compressed_depth_msg(stamp, spec.color_frame, aligned_mm))
            self._publish_info(pubs["aligned_info"], cam.color_info, stamp)

            if self.depth_stream:
                pubs["depth"].publish(self._image_msg(stamp, spec.depth_frame, depth_mm, "16UC1", 2))
                if pubs["depth_png"].get_subscription_count() > 0:
                    pubs["depth_png"].publish(self._compressed_depth_msg(stamp, spec.depth_frame, depth_mm))
                self._publish_info(pubs["depth_info"], cam.depth_info, stamp)
            self.timings.stop("camera_publish", publish_start)

            if self.pointcloud:
                cloud_start = self.timings.start()
                points = realsense_profile.colored_cloud(
                    depth_mm.astype(np.float32) / 1000.0, spec.depth_image,
                    rgb_u8, spec.color, spec.T_color_depth)
                msg = PointCloud2()
                msg.header.stamp = stamp
                msg.header.frame_id = spec.depth_frame
                msg.height, msg.width = 1, int(points.size)
                msg.fields = CLOUD_FIELDS
                msg.is_bigendian = False
                msg.point_step = realsense_profile.CLOUD_DTYPE.itemsize
                msg.row_step = msg.point_step * msg.width
                msg.is_dense = True
                msg.data = _array("B", points.tobytes())
                pubs["points"].publish(msg)
                self.timings.stop("camera_cloud", cloud_start)

    def _publish_lidar_scan(self, stamp: TimeMsg) -> None:
        """Snapshot MjData and cast and publish a scan from it on the lidar worker.

        The snapshot takes ~1 ms in a kitchen; the ~75 ms cast runs beside the
        physics, as mj_multiRay releases the GIL.
        """
        capture_start = self.timings.start()
        with self.sim_lock or nullcontext():
            snapshot = copy.copy(self.data)
        self.timings.stop("lidar_capture", capture_start)
        worker = getattr(self, "_lidar_worker", None)
        if worker is None:
            self._cast_and_publish_lidar(stamp, snapshot)
            return
        # One scan in flight at most: wait rather than drop, so every scheduled
        # scan is published. result() re-raises the last one's failure.
        wait_start = self.timings.start()
        pending, self._lidar_pending = self._lidar_pending, None
        if pending is not None:
            pending.result()
        self.timings.stop("lidar_wait", wait_start)
        self._lidar_pending = worker.submit(self._cast_and_publish_lidar, stamp, snapshot)

    def _cast_and_publish_lidar(self, stamp: TimeMsg, data) -> None:
        """Cast and publish one scan from the MjData snapshot data; never touches self.data."""
        cast_start = self.timings.start()
        timebase = int(data.time * 1e9)
        origin = np.ascontiguousarray(data.xpos[self.lidar_body_id], dtype=np.float64).reshape(3, 1)
        rot = np.array(data.xmat[self.lidar_body_id], dtype=np.float64).reshape(3, 3)

        # rot maps lidar to world for column vectors, so rows go through rot.T.
        world_dirs = np.ascontiguousarray(self.lidar_local_dirs @ rot.T, dtype=np.float64)

        # bodyexclude skips the torso the lidar sits in; otherwise every ray
        # would hit its shell at ~2 cm.
        dists = self._lidar_dists
        geomids = self._lidar_geomids
        dists.fill(-1.0)
        geomids.fill(-1)
        mujoco.mj_multiRay(
            self.model, data,
            origin, world_dirs.reshape(-1, 1),
            self.lidar_geomgroup,
            1,                              # include static geoms
            self.lidar_exclude_body_id,
            geomids, dists,
            self.lidar_rays,
            self.lidar_max_range,
        )
        self.timings.stop("lidar_cast", cast_start)
        publish_start = self.timings.start()
        hit_dists = dists.ravel()

        if self.geom_is_robot.any():
            gid = geomids.ravel()
            self_hit = (gid >= 0) & self.geom_is_robot[np.where(gid >= 0, gid, 0)]
            hit_dists = np.where(self_hit, -1.0, hit_dists)

        valid = (hit_dists >= self.lidar_min_range) & (hit_dists <= self.lidar_max_range)
        pts = (self.lidar_local_dirs[valid] * hit_dists[valid, np.newaxis]).astype(np.float32)
        lines = self.lidar_lines[valid]
        offs = self.lidar_offset_time_ns[valid]

        self.pub_lidar.publish(
            _serialize_livox_scan(stamp, self.lidar_frame, timebase, offs, pts, lines))

        # The same scan on /livox/pointcloud: timestamp is offset_time (ns, as
        # float64); intensity and tag stay 0, as the sim has no reflectivity.
        arr = np.zeros(pts.shape[0], dtype=PC2_DTYPE)
        arr["x"] = pts[:, 0]
        arr["y"] = pts[:, 1]
        arr["z"] = pts[:, 2]
        arr["line"] = lines
        arr["timestamp"] = offs.astype(np.float64)
        pc2 = PointCloud2()
        pc2.header.stamp = stamp
        pc2.header.frame_id = self.lidar_frame
        pc2.height = 1
        pc2.width = int(pts.shape[0])
        pc2.fields = PC2_FIELDS
        pc2.is_bigendian = False
        pc2.point_step = PC2_POINT_STEP
        pc2.row_step = PC2_POINT_STEP * pc2.width
        pc2.is_dense = True
        pc2.data = _array("B", arr.tobytes())
        self.pub_pc2.publish(pc2)
        self.timings.stop("lidar_publish", publish_start)
