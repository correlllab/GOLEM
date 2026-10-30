import numpy as np

from golem_slam.level_odom import find_floor_height, motion_spread
from golem_slam.robot_geometry import rpy_to_matrix

RNG = np.random.default_rng(0)


def plane(n, z, half=4.0, noise=0.01):
    xy = RNG.uniform(-half, half, (n, 2))
    return np.column_stack([xy, z + RNG.normal(0.0, noise, n)])


def test_floor_below_denser_table_and_sparse_reflections():
    # Lidar at the origin, 1.7 m above the floor.
    floor = plane(3000, -1.7)
    seat = plane(8000, -1.25, half=0.5)        # denser than the floor, still lower than lidar - 1
    reflections = plane(100, -2.4)             # stray points below the floor
    wall = np.column_stack([np.full(2000, 3.0), RNG.uniform(-3, 3, 2000),
                            RNG.uniform(-1.7, 1.0, 2000)])
    pts = np.vstack([floor, seat, reflections, wall])
    h = find_floor_height(pts, np.array([0.0, 0.0, 1.0]), 0.0, 1.0, 300)
    assert abs(h - (-1.7)) < 0.01


def test_floor_in_tilted_frame():
    # Points expressed in a frame pitched 0.24 rad, like camera_init.
    rot = rpy_to_matrix(0.0, 0.24, 0.0)
    pts = plane(5000, -1.6) @ rot            # world -> tilted: p_t = R^T p_w
    up = rot.T @ np.array([0.0, 0.0, 1.0])
    h = find_floor_height(pts, up, 0.0, 1.0, 300)
    assert abs(h - (-1.6)) < 0.01


def test_ramp_rejected_and_no_floor():
    ramp = plane(5000, 0.0)
    ramp[:, 2] += 0.3 * ramp[:, 0] - 1.7     # ~17 deg slope
    assert find_floor_height(ramp, np.array([0.0, 0.0, 1.0]), 0.0, 0.5, 300) is None
    assert find_floor_height(plane(100, -1.7), np.array([0.0, 0.0, 1.0]),
                             0.0, 1.0, 300) is None


# level_odom's defaults: a 2 s window at the MID360's 200 Hz in 4 slices, 0.02 of the mean
WINDOW, CHUNKS, MAX_SPREAD = 400, 4, 0.02


def _specific_force(tilt_rad, rate=200.0):
    t = np.arange(WINDOW) / rate
    tilt = np.asarray(tilt_rad(t))
    return np.column_stack([9.81 * np.sin(tilt), np.zeros(WINDOW), 9.81 * np.cos(tilt)]), t


def test_motion_spread_ignores_standing_buzz():
    # a standing policy's leg buzz: ~2 m/s^2 lateral at 12-30 Hz (the H1-2 sim's: 22 Hz)
    for freq in (12.0, 22.0, 30.0):
        ups, t = _specific_force(lambda t: 0.0 * t)
        ups[:, 1] += 2.7 * np.sin(2 * np.pi * freq * t + 0.7) + RNG.normal(0.0, 0.3, WINDOW)
        assert motion_spread(ups, CHUNKS) < MAX_SPREAD / 2


def test_motion_spread_sees_swaying():
    # a +-2 deg sway moves the up direction itself
    for freq in (0.3, 0.5, 1.0):
        ups, _ = _specific_force(lambda t: np.radians(2.0) * np.sin(2 * np.pi * freq * t))
        assert motion_spread(ups, CHUNKS) > MAX_SPREAD


def test_motion_spread_still():
    ups, _ = _specific_force(lambda t: 0.0 * t)
    assert motion_spread(ups, CHUNKS) < 1e-9


def test_g1_floor_with_upside_down_sensor():
    points = plane(5000, -0.9)
    rotation = rpy_to_matrix(np.pi, 0.0, 0.0)
    up = rotation.T @ np.array([0.0, 0.0, 1.0])
    floor = find_floor_height(points @ rotation, up, 0.0, 0.5, 300)
    assert abs(floor + 0.9) < 0.01


def test_moving_waist_updates_footprint_and_missing_tf_skips():
    from types import SimpleNamespace
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from tf2_ros import Buffer
    from golem_slam.level_odom import LevelOdom

    buffer = Buffer()
    published = []
    node = SimpleNamespace(
        odom_from_world=(np.eye(3), np.zeros(3)), body_from_base=None,
        dynamic_base_transform=True, tf_buffer=buffer, body_frame='body',
        base_frame='pelvis', odom_frame='odom', footprint_frame='base_footprint',
        footprint_broadcaster=SimpleNamespace(sendTransform=published.append),
        get_logger=lambda: SimpleNamespace(info=lambda *a, **kw: None))
    for second, offset in [(1, 0.1), (2, 0.3)]:
        transform = TransformStamped()
        transform.header.frame_id = 'body'
        transform.child_frame_id = 'pelvis'
        transform.header.stamp.sec = second
        transform.transform.rotation.w = 1.0
        transform.transform.translation.x = offset
        buffer.set_transform(transform, 'test')
        msg = Odometry()
        msg.header.stamp.sec = second
        msg.pose.pose.position.x = 2.0
        msg.pose.pose.orientation.w = 1.0
        LevelOdom._on_odometry(node, msg)
    assert np.allclose([t.transform.translation.x for t in published], [2.1, 2.3])
    msg.header.stamp.sec = 3  # no transform for this odometry sample
    LevelOdom._on_odometry(node, msg)
    assert len(published) == 2
    # The missing TF arrives after the odometry callback: retry without needing
    # a new odometry message (which can always run a few milliseconds ahead).
    transform.header.stamp.sec = 3
    transform.transform.translation.x = 0.4
    buffer.set_transform(transform, 'test')
    node._on_odometry = lambda sample: LevelOdom._on_odometry(node, sample)
    assert hasattr(LevelOdom, '_retry_odometry'), 'delayed TF must trigger a retry'
    LevelOdom._retry_odometry(node)
    assert len(published) == 3
    assert published[-1].transform.translation.x == 2.4
    assert published[-1].header.stamp.sec == 3

    # Fixed mode retains the existing H1-2 behavior, including the cached TF.
    node.dynamic_base_transform = False
    LevelOdom._on_odometry(node, msg)
    assert len(published) == 4
    assert published[-1].transform.translation.x == 2.4


def test_motion_spread_accepts_sample_lists():
    assert motion_spread([[0, 0, 9.81]] * 8, 4) == 0.0
