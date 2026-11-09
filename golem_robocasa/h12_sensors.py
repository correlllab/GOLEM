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
camera TF are published at their rates keyed off sim time; camera frames are
rate-throttled the same way and rendered from MjData snapshots on the camera
worker thread, through EGL. MuJoCo's renderer GL context is thread-affine: it is
created, used and freed on that thread (see _camera_worker). The renders hold
mid360.GL_LOCK, taking turns with the lidar's GPU caster.

The MID-360 is the model both robots share (golem_robocasa/mid360.py): the
real non-repetitive scan pattern, 20,000 points a 10 Hz frame (200,000
points/s), lines 0-3, the blind zone, maximum range and range noise, with each
point's offset_time the time of the pose it was cast from. The rays leave
livox_link and pass through torso_link's geoms, in which the sensor sits; a ray
whose first hit is a self-filtered geom (self_filter_geoms) gives no return, so
the limbs still occlude what is behind them. /livox/lidar and /livox/pointcloud
carry the same frame, stamped at its first point, as the driver stamps them.

Threads. The lidar's rays are cast on the mid360_scan thread (_LidarScans, a
mid360.ScanRing, as on the G1) through mid360.GpuCaster, from depth renders on
the GPU. Every caster's batch_s of sim time, tick() copies under the sim lock
what a cast reads (every geom's pose, the sensor's, and the geoms' and
materials' live alpha, which RoboCasa tasks change and the rays skip at 0) into
a slot of a ring; the thread casts the rays due by then, labelled with that
time, and packs and publishes each frame it completes. The caster casts on its
own copy of the model and releases the GIL while it renders, so the casts run
beside the physics. The ring holds one frame of casts: when the thread falls
that far behind, tick() waits for it (the lidar_wait stage), so a frame trails
the sim by at most one frame period and the real-time factor drops instead.

State time. tick() runs after mj_step, which leaves the kinematics of the
state it started from, so the lidar's casts are labelled one physics step
before sim_t, as the G1's are.
"""
import copy
import io
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

import mid360
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

# Geom groups the lidar's rays hit: all six, the robot's and the scene's collision geoms included.
LIDAR_GEOM_GROUPS = (1, 1, 1, 1, 1, 1)


def _log(text):
    """Print an [h12_sensors] line."""
    print(f"[h12_sensors] {text}", flush=True)


def self_filter_geoms(model, prefixes) -> np.ndarray:
    """Return the bool mask (ngeom,) of geoms on bodies whose names start with one of prefixes.

    A lidar ray whose first hit is one of them gives no return. No prefixes mark none.
    """
    mask = np.zeros(model.ngeom, dtype=bool)
    prefixes = tuple(prefixes)
    if prefixes:
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
            mask[g] = name.startswith(prefixes)
    return mask


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


def _livox_pointcloud(stamp, frame_id, offset_time, points, lines) -> PointCloud2:
    """Return a /livox/pointcloud PointCloud2 of a scan, in the driver's xfer_format 0 layout.

    timestamp is offset_time (ns, as float64); intensity and tag stay 0, as the sim has no
    reflectivity.
    """
    arr = np.zeros(points.shape[0], dtype=PC2_DTYPE)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["line"] = lines
    arr["timestamp"] = offset_time.astype(np.float64)
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(points.shape[0])
    msg.fields = PC2_FIELDS
    msg.is_bigendian = False
    msg.point_step = PC2_POINT_STEP
    msg.row_step = PC2_POINT_STEP * msg.width
    msg.is_dense = True
    msg.data = _array("B", arr.tobytes())
    return msg


class _LidarScans(mid360.ScanRing):
    """Cast the MID-360's scans on the mid360_scan thread (a mid360.ScanRing) and publish both lidar topics."""

    def __init__(self, model, body_id, exclude_body_id, robot_geoms, frame_id,
                 publish_lidar, publish_cloud, timings=None, sim_lock=None):
        """Build the GPU caster and start the thread.

        The sensor's pose is body body_id's; the rays pass through body exclude_body_id's
        geoms, and a ray whose first hit is in the bool mask robot_geoms (ngeom,) gives no
        return (a zero point). Each frame goes to publish_lidar(CustomMsg CDR bytes) and
        publish_cloud(PointCloud2) in frame_id. sim_lock is held for the snapshot copies only.
        Raises RuntimeError when EGL fails to render.
        """
        body = int(body_id)
        self._frame_id = frame_id
        self._publish_lidar = publish_lidar
        self._publish_cloud = publish_cloud
        caster = mid360.GpuCaster(model, exclude_body_id, LIDAR_GEOM_GROUPS, robot_geoms=robot_geoms)
        _log(f"lidar rays cast on the GPU every {caster.batch_s * 1e3:g} ms of sim time")
        super().__init__(mid360.Mid360Scanner(caster), model,
                         lambda data: (data.xpos[body], data.xmat[body].reshape(3, 3)),
                         self._publish_frame, caster.batch_s, sim_lock, timings, _log)

    def _publish_frame(self, frame):
        """Publish a mid360.LidarFrame on /livox/lidar and /livox/pointcloud, stamped at its first point."""
        timebase = int(round(frame.timebase * 1e9))   # ns
        stamp = TimeMsg(sec=timebase // 1_000_000_000, nanosec=timebase % 1_000_000_000)
        self._publish_lidar(_serialize_livox_scan(stamp, self._frame_id, timebase, frame.offset_time,
                                                  frame.points, frame.line))
        self._publish_cloud(_livox_pointcloud(stamp, self._frame_id, frame.offset_time, frame.points,
                                              frame.line))


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

    # The camera snapshots' MjData (_capture_camera_snapshot), made on the first frame; an
    # empty list for a model with flexes or skins, which takes full copies.
    _camera_slots = None
    _camera_slot_count = 0

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
        and depth/color/points on. The lidar's pose is lidar_body's;
        lidar_exclude_body is the body it rides in, whose geoms the rays pass
        through; geoms of bodies named with a lidar_self_prefixes prefix give no
        return. The lidar's GPU caster raises RuntimeError when EGL fails to
        render. Rates are Hz of sim time.
        """
        self.node = node
        self.model = model
        self.data = data
        self.sim_lock = sim_lock
        self.timings = timings if timings is not None else LoopTimings()

        self._camera_specs = list(cameras)
        self.depth_stream = bool(depth_stream)
        self.pointcloud = bool(pointcloud)

        self.lidar_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lidar_body)
        if self.lidar_body_id < 0:
            raise RuntimeError(f"body '{lidar_body}' not found in MJCF")
        # The rays pass through the torso's own geoms (the casters'
        # exclude_body): the lidar sits inside its shell. Limbs still occlude.
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lidar_exclude_body)
        if torso_id < 0:
            raise RuntimeError(f"body '{lidar_exclude_body}' not found in MJCF")
        self.lidar_exclude_body_id = int(torso_id)
        # Rays whose first hit is one of these geoms give no return: the limbs
        # still occlude what is behind them, but map no obstacle under the robot.
        self.geom_is_robot = self_filter_geoms(model, lidar_self_prefixes)

        self.cam_period = 1.0 / cam_rate_hz
        # m, MuJoCo's far plane: depth there means no surface.
        self.zfar = float(model.vis.map.zfar * model.stat.extent)

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
        self._last_imu_sim_t = 0.0
        # The scan thread publishes both lidar topics; its GPU caster makes its
        # EGL context here, before the passive viewer's window exists.
        self._lidar = _LidarScans(model, self.lidar_body_id, self.lidar_exclude_body_id,
                                  self.geom_is_robot, lidar_frame,
                                  self.pub_lidar.publish, self.pub_pc2.publish,
                                  self.timings, sim_lock)
        # The cameras render on a worker thread, through EGL, while physics runs.
        self._camera_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera")
        self._camera_pending = None
        if self.cameras:
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
        """Return the lidar's counts (mid360.ScanRing.stats); every scheduled camera frame is published."""
        lidar = getattr(self, "_lidar", None)
        return lidar.stats() if lidar is not None else ""

    def shutdown(self) -> None:
        """Stop the worker threads, log the lidar's counts and free the renderers on the thread that owns them.

        A second call, or one on a suite whose constructor failed before the threads started,
        stops what is left.
        """
        lidar = getattr(self, "_lidar", None)
        if lidar is not None:
            lidar.close()
            _log(f"sensors published {lidar.stats()}")
        camera_worker = getattr(self, "_camera_worker", None)
        if camera_worker is not None:
            camera_worker.submit(self._close_renderer).result()
            camera_worker.shutdown(wait=True)
            self._camera_worker = None

    def _close_renderer(self) -> None:
        """Close every renderer, on the thread that made them."""
        for r in (self._renderer or {}).values():
            try:
                r.close()
            except Exception:
                pass
        self._renderer = None

    def tick(self, sim_t: float, stamp: TimeMsg) -> None:
        """Publish the IMU, camera TF and camera frames due at sim_t (s), and submit the lidar's casts.

        The host calls it once per sim step, after /clock and odom; stamp is
        sim_t as a ROS time. Camera frames come from MjData snapshots on the
        camera worker thread, one job in flight; the lidar's casts are labelled
        with the state time, one physics step before sim_t (module docstring).
        Call it without holding sim_lock, on the thread that steps the physics.
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

        try:
            self._lidar.tick(self.data, sim_t - self.model.opt.timestep)
        except Exception as e:
            self.get_logger().warn(f"lidar capture failed: {e}", throttle_duration_sec=5.0)

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
        """Snapshot MjData and submit rendering and publishing every camera from it to the camera worker."""
        capture_start = self.timings.start()
        snapshot = self._capture_camera_snapshot()
        self.timings.stop("camera_capture", capture_start)
        # One frame set in flight at most: wait rather than drop, so every
        # scheduled frame is published. result() re-raises the last one's failure.
        wait_start = self.timings.start()
        pending, self._camera_pending = self._camera_pending, None
        if pending is not None:
            pending.result()
        self.timings.stop("camera_wait", wait_start)
        self._camera_pending = self._camera_worker.submit(self._render_and_publish_cameras, stamp, snapshot)

    def _capture_camera_snapshot(self):
        """Return an MjData holding what the cameras' scenes read of self.data, taken under the sim lock.

        It is one of _CAMERA_SNAPSHOTS preallocated MjData, filled with
        _camera_snapshot_fields() only, in turn. This runs while the previous
        frame's job may still render, but the one before that finished before the
        previous was submitted, so the slot reused here is idle. A model with
        flexes or skins, whose scenes read more than those fields, takes a full
        copy of self.data instead.
        """
        slots = self._camera_slots
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
        # Rangefinder rays are not drawn: no camera sees them, and the partial snapshot does
        # not carry the sensor data they are drawn from.
        self._scene_opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

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
            with mid360.GL_LOCK:
                rgb_u8 = r.render(out=rgb_buf)
                r.enable_depth_rendering()
                aligned = r.render(out=depth_buf)
            aligned_mm = self._depth_mm(aligned)
            self.timings.stop("camera_render", render_start)

            depth_mm = None
            if self.depth_stream or self.pointcloud:
                depth_start = self.timings.start()
                size = (spec.depth_image.width, spec.depth_image.height)
                rd, (_, ddepth_buf) = self._renderer[size], self._buffers[size]
                rd.enable_depth_rendering()
                rd.update_scene(data, camera=cam.depth_id, scene_option=self._scene_opt)
                with mid360.GL_LOCK:
                    depth = rd.render(out=ddepth_buf)
                depth_mm = self._depth_mm(depth)
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
