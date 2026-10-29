'''Livox CustomMsg to PointCloud2, deskewed for the sensor's rotation, in target_frame.

    ros2 run g1_navigation livox_to_pointcloud    # livox/lidar + livox/imu -> lidar_points
    ros2 run g1_navigation livox_to_pointcloud --ros-args -p target_frame:=pelvis_level

FAST_LIO needs the MID-360 driver in its CustomMsg mode, and the G1 sim
publishes the same, so no PointCloud2 of the raw scan exists for
pointcloud_to_laserscan (the driver's /livox/pointcloud is not published in
that mode). The cloud has float32 x, y, z and intensity (the Livox
reflectivity), with the CustomMsg's header stamp. Every point is passed
through; the scan's height band and minimum range drop the floor and the
robot's own body.

Deskew. A frame's points are measured over 0.1 s (offset_time), but the cloud
places them all with the sensor's pose at the stamp. The walking G1's torso
turns fast: in the G1 sim's kitchen test the MID-360's tilt changed by a median
4.6 deg and up to 8.9 deg within one frame while navigating, and its heading by
up to 4.7 deg. Cut with the stamp's pose, the band, 0.15 m above the floor,
then took floor points into the scan from 0.9 m out. So each point is turned
back to the sensor's orientation at the stamp, using the rotation the sensor's
own gyro (livox/imu, in the same axes) measured between the stamp and the
point's time. The translation within a frame (at most 0.05 m at the 0.5 m/s
cap) is not corrected. Without IMU samples around a frame, the frame is passed
through as it is (logged).

target_frame. Empty: the cloud is in the CustomMsg's frame (livox_frame). Else
each frame waits, in arrival order, until TF has target_frame <- livox_frame at
its stamp (plain lookups, at most max_tf_wait s of stamps behind the newest
frame, then it is dropped and logged), and the cloud comes out in target_frame.
g1_navigation.launch.py sets pelvis_level, so that pointcloud_to_laserscan needs
no TF: with its own TF message filter it stopped for good in 7 of 27 runs of the
G1 sim's nav test (its executor thread blocked, and once its TF thread too).
'''

import array
from collections import deque

import numpy as np
import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

FIELDS = [PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
          for i, name in enumerate(('x', 'y', 'z', 'intensity'))]
POINT_STEP = 4 * len(FIELDS)
DESKEW_STEP = 0.001  # s: the gyro's rotation is integrated on this grid, then on to each point
IMU_GAP = 0.05       # s: largest gap the gyro may leave at either end of a frame; more, no deskew
MAX_PENDING = 20     # frames waiting for TF (2 s at 10 Hz); more, the oldest is dropped
IMU_KEEP = 400       # gyro samples kept (2 s at the MID-360's 200 Hz)


def _stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _rodrigues(rotvec):
    '''3x3 rotation matrices (n, 3, 3) of rotation vectors (n, 3)'''
    angle = np.linalg.norm(rotvec, axis=-1)
    axis = rotvec / np.where(angle > 0, angle, 1.0)[..., None]
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    k = np.stack([np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1),
                  np.stack([-y, x, zero], -1)], -2)
    s, c = np.sin(angle)[..., None, None], np.cos(angle)[..., None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def deskew(xyz, times, t_ref, imu_times, imu_rates, step=DESKEW_STEP):
    '''xyz (n, 3), measured at times (n), in the sensor's axes at t_ref.

    The sensor's angular velocity (rad/s, in its own axes) is taken between the
    gyro samples (imu_times, imu_rates (m, 3)) linearly, and held past the last.
    A point measured at t, with the sensor turned by R(t_ref -> t), lies at
    R p in the sensor's axes at t_ref. Returns None, changing nothing, without
    gyro samples within IMU_GAP of the frame's span.
    '''
    if not len(xyz):
        return xyz
    lo, hi = min(t_ref, times.min()), max(t_ref, times.max())
    if (not len(imu_times) or imu_times[0] > lo + IMU_GAP or imu_times[-1] < hi - IMU_GAP):
        return None
    # the rotation from t_ref along a grid over [lo, hi]
    n_back = int(np.ceil((t_ref - lo) / step))
    grid = t_ref + step * np.arange(-n_back, int(np.ceil((hi - t_ref) / step)) + 1)
    rates = np.stack([np.interp(grid, imu_times, imu_rates[:, i]) for i in range(3)], -1)
    steps = _rodrigues(0.5 * (rates[1:] + rates[:-1]) * step)
    turned = np.empty((len(grid), 3, 3))
    turned[n_back] = np.eye(3)
    for k in range(n_back + 1, len(grid)):
        turned[k] = turned[k - 1] @ steps[k - 1]
    for k in range(n_back - 1, -1, -1):
        turned[k] = turned[k + 1] @ steps[k].T
    # from the grid point before each point's time on to it
    before = np.clip(np.floor((times - grid[0]) / step).astype(int), 0, len(grid) - 1)
    rest = np.stack([np.interp(times, imu_times, imu_rates[:, i]) for i in range(3)], -1)
    rest *= (times - grid[before])[:, None]
    return np.einsum('nij,njk,nk->ni', turned[before], _rodrigues(rest), xyz)


def to_pointcloud2(msg, gyro=None, frame=None):
    '''The PointCloud2 of a livox_ros_driver2/CustomMsg; with gyro, (times, rates (m, 3)) of
    the sensor's IMU, deskewed to the stamp (deskew()); with frame, (frame_id, 4x4 transform
    frame_id <- the CustomMsg's frame at the stamp), in that frame. Returns the cloud, and
    False if it needed a deskew but did not get one (no gyro samples around it).'''
    points = np.array([(p.x, p.y, p.z, p.reflectivity, p.offset_time) for p in msg.points],
                      dtype=np.float64).reshape(-1, 5)
    deskewed = True
    times = msg.timebase * 1e-9 + points[:, 4] * 1e-9
    # points all taken at the stamp need none
    if gyro is not None and len(points) and np.ptp(np.append(times, _stamp(msg.header))) > 0:
        turned = deskew(points[:, :3], times, _stamp(msg.header), *gyro)
        if turned is not None:
            points[:, :3] = turned
        deskewed = turned is not None
    if frame is not None:
        points[:, :3] = points[:, :3] @ frame[1][:3, :3].T + frame[1][:3, 3]
    points = points[:, :4].astype(np.float32)
    cloud = PointCloud2()
    cloud.header.stamp = msg.header.stamp
    cloud.header.frame_id = msg.header.frame_id if frame is None else frame[0]
    cloud.height = 1
    cloud.width = len(points)
    cloud.fields = FIELDS
    cloud.is_bigendian = False
    cloud.point_step = POINT_STEP
    cloud.row_step = POINT_STEP * len(points)
    cloud.is_dense = bool(np.isfinite(points).all())
    # an array.array is taken as is; bytes would be checked one by one (~15 ms per scan)
    cloud.data = array.array('B', points.tobytes())
    return cloud, deskewed


def to_matrix(transform):
    '''4x4 matrix of a geometry_msgs/Transform'''
    t, q = transform.translation, transform.rotation
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    m[:3, 3] = (t.x, t.y, t.z)
    return m


class LivoxToPointCloud(Node):

    def __init__(self, **kwargs):
        super().__init__('livox_to_pointcloud', **kwargs)
        self.use_deskew = self.declare_parameter('deskew', True).value
        self.target_frame = self.declare_parameter('target_frame', '').value
        self.max_tf_wait = self.declare_parameter('max_tf_wait', 0.5).value
        self.pub = self.create_publisher(PointCloud2, 'lidar_points', QoSProfile(depth=5))
        self.gyro = deque(maxlen=IMU_KEEP)
        if self.use_deskew:
            # 0.25 s of the 200 Hz gyro, so that converting a frame does not cost samples
            self.create_subscription(Imu, 'livox/imu', self._on_imu, QoSProfile(
                depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.pending = deque()   # frames waiting for TF, oldest first
        if self.target_frame:
            self.tf = Buffer()
            self.tf_listener = TransformListener(self.tf, self)
            self.create_timer(0.01, self._flush)
        # the driver and the sim publish reliable; a best-effort reader matches either
        self.create_subscription(CustomMsg, 'livox/lidar', self._on_scan, qos_profile_sensor_data)

    def _on_imu(self, msg):
        w = msg.angular_velocity
        self.gyro.append((_stamp(msg.header), w.x, w.y, w.z))

    def _on_scan(self, msg):
        self.pending.append(msg)
        if len(self.pending) > MAX_PENDING:
            dropped = self.pending.popleft()
            self.get_logger().warn('dropped the scan at %.3f s: %d frames waiting for TF'
                                   % (_stamp(dropped.header), MAX_PENDING), throttle_duration_sec=5.0)
        self._flush()

    def _flush(self):
        '''Publish the waiting frames in order, each once TF places it in target_frame'''
        while self.pending:
            msg = self.pending[0]
            frame = None
            if self.target_frame:
                try:
                    tf = self.tf.lookup_transform(self.target_frame, msg.header.frame_id,
                                                  Time.from_msg(msg.header.stamp))
                    frame = (self.target_frame, to_matrix(tf.transform))
                except TransformException as e:
                    behind = _stamp(self.pending[-1].header) - _stamp(msg.header)
                    # (behind < 0: the clock went back, the sim restarted; TF at the old stamp won't come)
                    if 0.0 <= behind <= self.max_tf_wait:
                        return   # TF may still come
                    self.pending.popleft()
                    self.get_logger().warn(
                        'dropped the scan at %.3f s: no %s <- %s after %.2f s of scans (%s)'
                        % (_stamp(msg.header), self.target_frame, msg.header.frame_id, behind, e),
                        throttle_duration_sec=5.0)
                    continue
            self.pending.popleft()
            self._publish(msg, frame)

    def _publish(self, msg, frame):
        gyro = None
        if self.use_deskew:
            samples = np.array(sorted(self.gyro), dtype=np.float64).reshape(-1, 4)
            _, first = np.unique(samples[:, 0], return_index=True)   # one per stamp
            gyro = (samples[first, 0], samples[first, 1:])
        cloud, deskewed = to_pointcloud2(msg, gyro, frame)
        if not deskewed:
            self.get_logger().warn('no livox/imu samples around the scan at %.3f s: not deskewed'
                                   % _stamp(msg.header), throttle_duration_sec=5.0)
        self.pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = LivoxToPointCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
