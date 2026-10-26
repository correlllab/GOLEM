'''FAST_LIO from g1_navigation.launch.py (odom_source:=fastlio sim:=true) on a fake G1

It has to subscribe to the MID-360's topics and publish its odometry, and
odom -> camera_init (static) and body -> pelvis (body_to_pelvis) have to hang it
off odom and the pelvis as frames.py computes them; in TF, body -> livox_frame
has to be FAST_LIO's extrinsic whatever the waist does. fake_g1.py then feeds
it an upside-down MID-360 and its IMU, standing the way HOMIE does (the waist
pitched forward, the pelvis back by as much, so the torso is level), walking
forward, turning about 180 deg and walking back. FAST_LIO's odom -> pelvis has to follow the fake's
true pose: at the standing height, leaning back, and turning the right way
(camera_init is upside down, so a sign error would mirror it). So does
odom -> mid360_link, the lidar the scan is cut from: a fixed body -> pelvis
with the waist at 0 would tilt it by the waist angle. The scan and the SLAM
map, which hang off FAST_LIO's odometry in this mode, have to come out too,
with nothing in the map that is not in the room.
'''

import bisect
import math
import os
import threading
import time

import numpy as np
import pytest
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

import fake_g1
import stack
from g1_description.paths import urdf_path
from g1_navigation import frames

SIM_CONFIG = os.path.join(get_package_share_directory('g1_navigation'), 'config',
                          'g1_fastlio_sim.yaml')
PELVIS_HEIGHT = 0.744  # the launch default, and the fake's standing height
# rad, waist pitch; the pelvis leans back by as much. More than HOMIE's 0.0256,
# so that a waist counted twice (3.4 deg) stands clear of FAST_LIO's own tilt error.
WAIST = 0.06
STARTUP_TIMEOUT = 60.0
# (seconds, vx, wz) sent to the fake; FAST_LIO initialises while it stands
MOTION = ((4.0, 0.0, 0.0), (3.0, 0.3, 0.0), (1.0, 0.0, 0.0), (8.0, 0.0, 0.4), (1.0, 0.0, 0.0),
          (3.0, 0.3, 0.0), (2.0, 0.0, 0.0))
POSITION_TOLERANCE = 0.05  # m
# FAST_LIO's own tilt error on the fake's scans reaches about 1 deg once it turns
TILT_TOLERANCE = math.radians(1.5)
YAW_TOLERANCE = math.radians(1.0)
MAP_TOLERANCE = 0.15  # m, occupied map cells to the nearest box of the room


class Probe(Node):

    def __init__(self):
        super().__init__('fastlio_probe')
        self.lock = threading.Lock()
        self.truth = []  # (stamp s, T_odom_pelvis)
        self.create_subscription(PoseStamped, '/fake_g1/pose', self._on_truth, 100)
        self.scans = 0
        self.create_subscription(LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)
        self.map = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, latched)
        self.cmd = self.create_publisher(Twist, '/fake_g1/cmd_vel', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _on_truth(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        t = frames.transform((p.x, p.y, p.z), Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix())
        with self.lock:
            self.truth.append((Time.from_msg(msg.header.stamp).nanoseconds * 1e-9, t))

    def _on_scan(self, msg):
        with self.lock:
            self.scans += 1

    def _on_map(self, msg):
        with self.lock:
            self.map = msg

    def lookup(self, parent, child, stamp=None):
        tf = self.tf_buffer.lookup_transform(parent, child, stamp or Time())
        tr, q = tf.transform.translation, tf.transform.rotation
        return (Time.from_msg(tf.header.stamp).nanoseconds * 1e-9,
                frames.transform((tr.x, tr.y, tr.z), Rotation.from_quat([q.x, q.y, q.z, q.w])
                                 .as_matrix()))

    def truth_at(self, stamp):
        '''The fake's T_odom_pelvis at stamp, interpolated; None outside the samples'''
        with self.lock:
            i = bisect.bisect_left(self.truth, stamp, key=lambda s: s[0])
            if i == 0 or i == len(self.truth):
                return None
            (t0, a), (t1, b) = self.truth[i - 1], self.truth[i]
        if t1 - t0 > 0.05:
            return None
        f = (stamp - t0) / (t1 - t0)
        rotation = Slerp([0.0, 1.0], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))(f)
        return frames.transform((1.0 - f) * a[:3, 3] + f * b[:3, 3], rotation.as_matrix())


@pytest.fixture
def stack_up(tmp_path):
    stack.use_loopback_dds(str(tmp_path))
    procs = stack.Processes(str(tmp_path))
    rclpy.init()
    probe = Probe()
    executor = MultiThreadedExecutor()
    executor.add_node(probe)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        procs.robot_state_publisher()
        procs.fake_g1('--imu', '--cmd-topic', '/fake_g1/cmd_vel', '--waist', str(WAIST),
                      '--pitch', str(-WAIST))
        procs.navigation('odom_source:=fastlio', 'sim:=true')
        yield procs, probe
    finally:
        executor.shutdown()
        probe.destroy_node()
        rclpy.shutdown()
        procs.stop_all()


def _wait_for_fast_lio(procs, probe):
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        procs.check_alive()
        if 'fast_lio' in probe.get_node_names():
            subs = dict(probe.get_subscriber_names_and_types_by_node('fast_lio', '/'))
            pubs = dict(probe.get_publisher_names_and_types_by_node('fast_lio', '/'))
            if '/livox/lidar' in subs and '/Odometry' in pubs:
                return subs, pubs
        time.sleep(1.0)
    pytest.fail(f'fast_lio did not come up:\n{procs.tail("navigation")}')


def _yaw(t):
    return math.atan2(t[1, 0], t[0, 0])


def _yaw_error(a, b):
    return abs((_yaw(a) - _yaw(b) + math.pi) % (2 * math.pi) - math.pi)


def _tilt_error(a, b):
    '''Angle between the z axes of two transforms'''
    return math.acos(min(1.0, float(a[:3, 2] @ b[:3, 2])))


def test_fastlio_follows_the_fake_g1(stack_up):
    procs, probe = stack_up
    subs, pubs = _wait_for_fast_lio(procs, probe)
    assert subs['/livox/lidar'] == ['livox_ros_driver2/msg/CustomMsg']
    assert subs['/livox/imu'] == ['sensor_msgs/msg/Imu']
    for topic, kind in (('/Odometry', 'nav_msgs/msg/Odometry'),
                        ('/cloud_registered_body', 'sensor_msgs/msg/PointCloud2'),
                        ('/cloud_registered', 'sensor_msgs/msg/PointCloud2')):
        assert pubs[topic] == [kind]

    # odom -> camera_init as frames.py computes it from the URDF and the config,
    # and body -> pelvis through the fake's waist
    urdf = urdf_path('with_hand')
    odom_camera_init, _ = frames.fastlio_transforms(urdf, SIM_CONFIG, PELVIS_HEIGHT)
    body_pelvis = frames.invert(frames.pelvis_to_body(
        urdf, frames.fastlio_extrinsic(SIM_CONFIG), positions={fake_g1.WAIST_PITCH: WAIST}))
    pelvis_lidar = frames.link_pose(urdf, frames.BASE_LINK, frames.LIDAR_LINK,
                                    {fake_g1.WAIST_PITCH: WAIST})
    deadline = time.time() + 20.0
    while True:
        try:
            _, got_odom_camera_init = probe.lookup('odom', 'camera_init')
            _, got_body_pelvis = probe.lookup('body', 'pelvis')
            break
        except TransformException:
            assert time.time() < deadline, 'no static transforms'
            time.sleep(0.5)
    np.testing.assert_allclose(got_odom_camera_init, odom_camera_init, atol=1e-6)
    np.testing.assert_allclose(got_body_pelvis, body_pelvis, atol=1e-6)
    # the IMU FAST_LIO tracks and the lidar the scan is cut from are one rigid body
    _, body_lidar = probe.lookup('body', frames.LIVOX_FRAME)
    np.testing.assert_allclose(body_lidar, frames.fastlio_extrinsic(SIM_CONFIG), atol=1e-6)

    # stand, walk, turn left, walk back: FAST_LIO's odom -> pelvis and odom ->
    # mid360_link against the truth
    errors = {'pelvis': [], 'mid360_link': []}
    for seconds, vx, wz in MOTION:
        end = time.time() + seconds
        while time.time() < end:
            cmd = Twist()
            cmd.linear.x, cmd.angular.z = vx, wz
            probe.cmd.publish(cmd)
            time.sleep(0.05)
            for frame, offset in (('pelvis', np.eye(4)), ('mid360_link', pelvis_lidar)):
                try:
                    stamp, estimate = probe.lookup('odom', frame)
                except TransformException:
                    continue  # FAST_LIO still initialising
                truth = probe.truth_at(stamp)
                if truth is not None:
                    errors[frame].append((stamp, estimate, truth @ offset))
    procs.check_alive()
    for frame, poses in errors.items():
        assert len(poses) > 50, f'{len(poses)} FAST_LIO poses of {frame}:\n' + procs.tail(
            'navigation')
        position = max(np.linalg.norm(e[:3, 3] - t[:3, 3]) for _, e, t in poses)
        tilt = max(_tilt_error(e, t) for _, e, t in poses)
        yaw = max(_yaw_error(e, t) for _, e, t in poses)
        _, estimate, truth = poses[-1]
        print(f'{frame}: {len(poses)} poses, largest errors {position:.3f} m, tilt '
              f'{math.degrees(tilt):.2f} deg, yaw {math.degrees(yaw):.2f} deg; end: truth '
              f'x {truth[0, 3]:.3f} y {truth[1, 3]:.3f} yaw {math.degrees(_yaw(truth)):.1f}, '
              f'FAST_LIO x {estimate[0, 3]:.3f} y {estimate[1, 3]:.3f} '
              f'yaw {math.degrees(_yaw(estimate)):.1f}')

    poses = errors['pelvis']
    # at startup odom -> pelvis is (0, 0, pelvis_height), leaning back by the waist angle
    np.testing.assert_allclose(poses[0][1], poses[0][2], atol=0.02)
    # the fake walked ~0.85 m out and back and turned about 180 deg left in between
    turned = np.unwrap([_yaw(t) for _, _, t in poses])
    assert turned[-1] - turned[0] > 2.9, turned[-1] - turned[0]
    path = sum(math.dist(a[2][:2, 3], b[2][:2, 3]) for a, b in zip(poses, poses[1:]))
    assert path > 1.4, path
    for frame, poses in errors.items():
        for stamp, estimate, truth in poses:
            assert np.linalg.norm(estimate[:3, 3] - truth[:3, 3]) < POSITION_TOLERANCE, (
                frame, stamp, estimate, truth)
            assert _tilt_error(estimate, truth) < TILT_TOLERANCE, (
                frame, stamp, math.degrees(_tilt_error(estimate, truth)))
            assert _yaw_error(estimate, truth) < YAW_TOLERANCE, (
                frame, stamp, math.degrees(_yaw_error(estimate, truth)))

    # the scan and the map, through FAST_LIO's odometry (map and odom both start
    # at the fake's start pose)
    time.sleep(2.0)  # slam_toolbox publishes the map every second
    with probe.lock:
        scans, grid = probe.scans, probe.map
    assert scans > 50, f'{scans} scans'
    assert grid is not None, 'no /map'
    occupied = fake_g1.occupied_cells(grid)
    far = fake_g1.room_distance(occupied) > MAP_TOLERANCE
    assert len(occupied) > 100, f'only {len(occupied)} occupied cells'
    assert not far.any(), (f'{far.sum()} of {len(occupied)} occupied map cells are more than '
                           f'{MAP_TOLERANCE} m from the room, e.g. {occupied[far][:5].round(2)}')
