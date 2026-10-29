'''The helper nodes (livox_to_pointcloud, level_frame, body_to_pelvis): their conversions,
and each node over DDS in one process'''

import os
import threading
import time

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

import stack
from g1_navigation import frames
from g1_navigation.body_to_pelvis import BodyToPelvis, to_matrix
from g1_navigation.level_frame import LevelFrame, level_rotation
from g1_navigation.livox_to_pointcloud import IMU_GAP, LivoxToPointCloud, deskew, to_pointcloud2
from golem_ros2_models.paths import urdf_path

POINTS = [(1.0, -2.0, 0.5, 7), (0.1, 0.2, -3.0, 255), (-4.5, 0.0, 1.25, 0)]
SIM_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config',
                          'g1_fastlio_sim.yaml')


def _scan(points, sec=12, nanosec=340):
    msg = CustomMsg()
    msg.header.frame_id = 'livox_frame'
    msg.header.stamp.sec, msg.header.stamp.nanosec = sec, nanosec
    msg.timebase = sec * 1_000_000_000 + nanosec
    msg.point_num = len(points)
    msg.points = [CustomPoint(offset_time=i, x=x, y=y, z=z, reflectivity=r, tag=0, line=i % 4)
                  for i, (x, y, z, r) in enumerate(points)]
    return msg


def test_to_pointcloud2_keeps_frame_stamp_and_points():
    scan = _scan(POINTS)
    cloud, deskewed = to_pointcloud2(scan)
    assert deskewed   # nothing to do without a gyro
    assert cloud.header == scan.header
    assert (cloud.height, cloud.width) == (1, len(POINTS))
    names = ('x', 'y', 'z', 'intensity')
    assert [(f.name, f.offset, f.datatype, f.count) for f in cloud.fields] == [
        (name, 4 * i, PointField.FLOAT32, 1) for i, name in enumerate(names)]
    assert (cloud.point_step, cloud.row_step) == (16, 16 * len(POINTS))
    assert cloud.is_dense and not cloud.is_bigendian
    read = point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z', 'intensity'))
    np.testing.assert_allclose(read, np.array(POINTS, dtype=np.float32))


def test_to_pointcloud2_of_an_empty_scan():
    cloud, _ = to_pointcloud2(_scan([]))
    assert (cloud.width, cloud.row_step, len(cloud.data)) == (0, 0, 0)


# a frame of the MID-360 while its torso turns: 100 ms, the sim's alternating 4/6 ms gyro samples
T_REF = 12.0
GYRO_TIMES = T_REF - 0.012 + np.cumsum(np.tile([0.004, 0.006], 13))
RATE = np.array([0.4, -0.3, 1.5])   # rad/s in the sensor's axes, about the walking G1's peak


def _turn(rate_at, t, substeps=200):
    '''R(T_REF -> t), integrating the angular velocity rate_at(time) finely'''
    r = Rotation.identity()
    for u in np.linspace(T_REF, t, substeps + 1)[:-1]:
        dt = (t - T_REF) / substeps
        r = r * Rotation.from_rotvec(rate_at(u + 0.5 * dt) * dt)
    return r


def _measured(rate_at, n=40, seed=0):
    '''Points fixed in the sensor's axes at T_REF, measured over the frame by the turning sensor'''
    rng = np.random.default_rng(seed)
    fixed = rng.uniform(-6.0, 6.0, (n, 3))
    times = T_REF + np.linspace(0.0, 0.1, n)
    measured = np.array([_turn(rate_at, t).inv().apply(p) for p, t in zip(fixed, times)])
    return fixed, measured, times


def test_deskew_undoes_a_constant_turn():
    fixed, measured, times = _measured(lambda t: RATE)
    gyro = (GYRO_TIMES, np.tile(RATE, (len(GYRO_TIMES), 1)))
    # the sensor turns 9.3 deg in the frame: up to 0.97 m at the points' 6 m
    assert np.abs(measured - fixed).max() > 0.5
    np.testing.assert_allclose(deskew(measured, times, T_REF, *gyro), fixed, atol=0.002)


def test_deskew_follows_a_changing_rate():
    def rate_at(t):
        return RATE * np.cos(2 * np.pi * 1.75 * (t - T_REF))   # HOMIE's sway frequency
    fixed, measured, times = _measured(rate_at)
    gyro = (GYRO_TIMES, np.array([rate_at(t) for t in GYRO_TIMES]))
    np.testing.assert_allclose(deskew(measured, times, T_REF, *gyro), fixed, atol=0.002)


def test_deskew_needs_gyro_samples_around_the_frame():
    _, measured, times = _measured(lambda t: RATE, n=5)
    rates = np.tile(RATE, (len(GYRO_TIMES), 1))
    assert deskew(measured, times, T_REF, np.array([]), np.zeros((0, 3))) is None
    early = GYRO_TIMES < T_REF + 0.1 - IMU_GAP - 0.005   # the gyro stops too early
    assert deskew(measured, times, T_REF, GYRO_TIMES[early], rates[early]) is None
    late = GYRO_TIMES > T_REF + IMU_GAP + 0.005          # and starts too late
    assert deskew(measured, times, T_REF, GYRO_TIMES[late], rates[late]) is None
    # a gyro that stops a little early is held at its last rate
    held = GYRO_TIMES < T_REF + 0.09
    assert deskew(measured, times, T_REF, GYRO_TIMES[held], rates[held]) is not None


def test_to_pointcloud2_deskews_with_the_gyro():
    fixed, measured, times = _measured(lambda t: RATE, n=8)
    sec, nanosec = int(T_REF), 0
    scan = _scan([(x, y, z, 1) for x, y, z in measured], sec, nanosec)
    for point, t in zip(scan.points, times):
        point.offset_time = int(round((t - T_REF) * 1e9))
    gyro = (GYRO_TIMES, np.tile(RATE, (len(GYRO_TIMES), 1)))
    cloud, deskewed = to_pointcloud2(scan, gyro)
    assert deskewed and cloud.header == scan.header
    np.testing.assert_allclose(point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z')),
                               fixed, atol=0.005)
    # without the gyro the points stay as measured, and the converter says so
    cloud, deskewed = to_pointcloud2(scan, (np.array([]), np.zeros((0, 3))))
    assert not deskewed
    np.testing.assert_allclose(point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z')),
                               measured, atol=1e-5)


@pytest.fixture
def ros(tmp_path):
    stack.use_loopback_dds(str(tmp_path))
    rclpy.init()
    executor = SingleThreadedExecutor()
    spinner = threading.Thread(target=executor.spin, daemon=True)
    nodes = []

    def spin(*new_nodes):
        for node in new_nodes:
            executor.add_node(node)
            nodes.append(node)
        if not spinner.is_alive():
            spinner.start()

    yield spin
    executor.shutdown()
    for node in nodes:
        node.destroy_node()
    rclpy.shutdown()


def _until(condition, timeout=10.0, every=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if every is not None:
            every()
        if condition():
            return True
        time.sleep(0.1)
    return False


def test_converter_node_over_dds(ros):
    peer = Node('peer')
    clouds = []
    peer.create_subscription(PointCloud2, 'lidar_points', clouds.append, 10)
    scans = peer.create_publisher(CustomMsg, 'livox/lidar', 10)  # reliable, like the driver
    ros(LivoxToPointCloud(), peer)
    assert _until(lambda: clouds, every=lambda: scans.publish(_scan(POINTS))), 'no cloud'
    cloud = clouds[0]
    assert cloud.header.frame_id == 'livox_frame'
    assert (cloud.header.stamp.sec, cloud.header.stamp.nanosec) == (12, 340)
    np.testing.assert_allclose(
        point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z', 'intensity')),
        np.array(POINTS, dtype=np.float32))


def test_converter_node_places_the_cloud_in_target_frame(ros):
    # pelvis_level <- livox_frame: upside down, 0.47 m up, turned 30 deg
    turn = Rotation.from_euler('xyz', [np.pi, 0.0, np.pi / 6])
    lift = np.array([0.05, -0.01, 0.47])
    peer = Node('peer')
    clouds = []
    peer.create_subscription(PointCloud2, 'lidar_points', clouds.append, 10)
    scans = peer.create_publisher(CustomMsg, 'livox/lidar', 10)
    static = StaticTransformBroadcaster(peer)
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = 'pelvis_level', 'livox_frame'
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = lift
    q = turn.as_quat()
    tf.transform.rotation.x, tf.transform.rotation.y = q[0], q[1]
    tf.transform.rotation.z, tf.transform.rotation.w = q[2], q[3]
    static.sendTransform(tf)
    node = LivoxToPointCloud(parameter_overrides=[Parameter('target_frame', value='pelvis_level')])
    ros(node, peer)
    assert _until(lambda: clouds, every=lambda: scans.publish(_scan(POINTS))), 'no cloud'
    cloud = clouds[0]
    assert cloud.header.frame_id == 'pelvis_level'
    assert (cloud.header.stamp.sec, cloud.header.stamp.nanosec) == (12, 340)
    xyz = np.array(POINTS, dtype=float)[:, :3]
    np.testing.assert_allclose(point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z')),
                               turn.apply(xyz) + lift, atol=1e-5)
    np.testing.assert_allclose(point_cloud2.read_points_numpy(cloud, field_names=('intensity',)),
                               np.array(POINTS, dtype=np.float32)[:, 3:])


def test_converter_node_drops_scans_it_cannot_place(ros):
    # no TF to target_frame: each scan waits until it is max_tf_wait behind the newest
    peer = Node('peer')
    clouds = []
    peer.create_subscription(PointCloud2, 'lidar_points', clouds.append, 10)
    scans = peer.create_publisher(CustomMsg, 'livox/lidar', 10)
    node = LivoxToPointCloud(parameter_overrides=[Parameter('target_frame', value='nowhere'),
                                                  Parameter('max_tf_wait', value=0.25)])
    ros(node, peer)
    sent = []

    def publish():
        stamp = 12.0 + 0.1 * len(sent)
        scans.publish(_scan(POINTS, int(stamp), int(round((stamp - int(stamp)) * 1e9))))
        sent.append(stamp)
    assert _until(lambda: len(sent) >= 10, every=publish)
    time.sleep(0.5)
    assert not clouds
    # at most the scans within max_tf_wait of the newest still wait
    assert 1 <= len(node.pending) <= 4, len(node.pending)


def test_converter_node_deskews_with_livox_imu(ros):
    fixed, measured, times = _measured(lambda t: RATE, n=8)
    scan = _scan([(x, y, z, 1) for x, y, z in measured], int(T_REF), 0)
    for point, t in zip(scan.points, times):
        point.offset_time = int(round((t - T_REF) * 1e9))
    peer = Node('peer')
    clouds = []
    peer.create_subscription(PointCloud2, 'lidar_points', clouds.append, 10)
    imu = peer.create_publisher(Imu, 'livox/imu', 50)
    scans = peer.create_publisher(CustomMsg, 'livox/lidar', 10)
    ros(LivoxToPointCloud(), peer)

    def publish():
        for t in GYRO_TIMES:
            msg = Imu()
            msg.header.frame_id = 'livox_frame'
            msg.header.stamp.sec, msg.header.stamp.nanosec = int(t), int(round((t - int(t)) * 1e9))
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = RATE
            imu.publish(msg)
        time.sleep(0.2)   # the node takes the gyro before the frame
        scans.publish(scan)
    assert _until(lambda: clouds, every=publish), 'no cloud'
    deskewed = point_cloud2.read_points_numpy(clouds[-1], field_names=('x', 'y', 'z'))
    np.testing.assert_allclose(deskewed, fixed, atol=0.005)


def test_level_frame_node_over_dds(ros):
    peer = Node('peer')
    received, floor = [], []
    peer.create_subscription(TFMessage, '/tf', lambda m: (received.extend(
        t for t in m.transforms if t.child_frame_id == 'pelvis_level'), floor.extend(
        t for t in m.transforms if t.child_frame_id == 'base_footprint')), 100)
    broadcaster = TransformBroadcaster(peer)
    tilt = Rotation.from_euler('xyz', (0.07, -0.09, 1.2))  # rolled, leaning back, turned
    odom_pelvis = TransformStamped()
    odom_pelvis.header.frame_id, odom_pelvis.child_frame_id = 'odom', 'pelvis'
    odom_pelvis.transform.translation.x, odom_pelvis.transform.translation.z = 1.0, 0.72
    q = odom_pelvis.transform.rotation
    q.x, q.y, q.z, q.w = tilt.as_quat()

    def publish():
        odom_pelvis.header.stamp = peer.get_clock().now().to_msg()
        broadcaster.sendTransform(odom_pelvis)

    ros(LevelFrame(), peer)
    assert _until(lambda: received, every=publish), 'no pelvis -> pelvis_level'
    level = received[-1]
    assert level.header.frame_id == 'pelvis'
    stamps = level.header.stamp
    assert (stamps.sec, stamps.nanosec) != (0, 0)
    t = level.transform
    assert (t.translation.x, t.translation.y, t.translation.z) == (0.0, 0.0, 0.0)
    got = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
    np.testing.assert_allclose(np.abs(np.dot(got, level_rotation(tilt.as_quat()))), 1.0,
                               atol=1e-9)
    # pelvis_level in odom: level, with the pelvis's heading
    in_odom = tilt * Rotation.from_quat(got)
    np.testing.assert_allclose(in_odom.as_matrix()[:, 2], [0.0, 0.0, 1.0], atol=1e-9)
    # base_footprint (slam_toolbox's base_frame): pelvis_level on odom's floor, at the same stamp
    assert _until(lambda: floor), 'no pelvis_level -> base_footprint'
    down = next(t for t in reversed(floor) if t.header.stamp == level.header.stamp)
    assert down.header.frame_id == 'pelvis_level'
    t = down.transform
    assert (t.translation.x, t.translation.y, t.translation.z) == (0.0, 0.0, -0.72)
    assert (t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w) == (0.0, 0.0, 0.0, 1.0)
    # so base_footprint in odom: on the floor under the pelvis, level, with its heading
    in_odom_xyz = np.array([1.0, 0.0, 0.72]) + in_odom.apply([0.0, 0.0, -0.72])
    np.testing.assert_allclose(in_odom_xyz, [1.0, 0.0, 0.0], atol=1e-9)


def test_body_to_pelvis_node_over_dds(ros):
    peer = Node('peer')
    received = []
    peer.create_subscription(TFMessage, '/tf', lambda m: received.extend(
        t for t in m.transforms if t.child_frame_id == 'pelvis'), 100)
    broadcaster = TransformBroadcaster(peer)
    # pelvis -> livox_frame as robot_state_publisher gives it with the waist off 0
    waist = {'waist_yaw_joint': 0.3, 'waist_roll_joint': -0.1, 'waist_pitch_joint': 0.2}
    pelvis_lidar = frames.link_pose(urdf_path('g1'), 'pelvis', 'mid360_link', waist)
    lidar = TransformStamped()
    lidar.header.frame_id, lidar.child_frame_id = 'pelvis', 'livox_frame'
    tr, q = lidar.transform.translation, lidar.transform.rotation
    tr.x, tr.y, tr.z = pelvis_lidar[:3, 3]
    q.x, q.y, q.z, q.w = Rotation.from_matrix(pelvis_lidar[:3, :3]).as_quat()
    sent = []

    def publish():
        lidar.header.stamp = peer.get_clock().now().to_msg()
        sent.append((lidar.header.stamp.sec, lidar.header.stamp.nanosec))
        broadcaster.sendTransform(lidar)

    ros(BodyToPelvis(parameter_overrides=[Parameter('fastlio_config', value=SIM_CONFIG)]), peer)
    assert _until(lambda: received, every=publish), 'no body -> pelvis'
    got = received[-1]
    assert got.header.frame_id == 'body'
    assert (got.header.stamp.sec, got.header.stamp.nanosec) in sent  # the lidar's stamp
    expected = frames.body_to_pelvis(pelvis_lidar, frames.fastlio_extrinsic(SIM_CONFIG))
    np.testing.assert_allclose(to_matrix(got.transform), expected, atol=1e-9)
