"""Test level_odom's floor search, stillness check and odom -> base_footprint."""
import numpy as np

from golem_slam.level_odom import find_floor_height, motion_spread
from golem_slam.robot_geometry import rpy_to_matrix

RNG = np.random.default_rng(0)


def plane(n, z, half=4.0, noise=0.01):
    """Return n points of a horizontal square (half-width half, m) at height z with noise."""
    xy = RNG.uniform(-half, half, (n, 2))
    return np.column_stack([xy, z + RNG.normal(0.0, noise, n)])


def test_floor_below_denser_table_and_sparse_reflections():
    """The floor wins over a denser seat above it and sparse reflections below it."""
    # Lidar at the origin, 1.7 m above the floor.
    floor = plane(3000, -1.7)
    seat = plane(8000, -1.25, half=0.5)        # denser than the floor, below min_depth
    reflections = plane(100, -2.4)             # stray points below the floor
    wall = np.column_stack([np.full(2000, 3.0), RNG.uniform(-3, 3, 2000),
                            RNG.uniform(-1.7, 1.0, 2000)])
    pts = np.vstack([floor, seat, reflections, wall])
    h = find_floor_height(pts, np.array([0.0, 0.0, 1.0]), 0.0, 1.0, 300)
    assert abs(h - (-1.7)) < 0.01


def test_floor_in_tilted_frame():
    """The floor is found along up in a pitched frame like camera_init."""
    # Points expressed in a frame pitched 0.24 rad, like camera_init.
    rot = rpy_to_matrix(0.0, 0.24, 0.0)
    pts = plane(5000, -1.6) @ rot            # world -> tilted: p_t = R^T p_w
    up = rot.T @ np.array([0.0, 0.0, 1.0])
    h = find_floor_height(pts, up, 0.0, 1.0, 300)
    assert abs(h - (-1.6)) < 0.01


def test_ramp_rejected_and_no_floor():
    """A sloped plane and too few points both give no floor."""
    ramp = plane(5000, 0.0)
    ramp[:, 2] += 0.3 * ramp[:, 0] - 1.7     # ~17 deg slope
    assert find_floor_height(ramp, np.array([0.0, 0.0, 1.0]), 0.0, 0.5, 300) is None
    assert find_floor_height(plane(100, -1.7), np.array([0.0, 0.0, 1.0]),
                             0.0, 1.0, 300) is None


# level_odom's defaults: a 2 s window at the MID-360's 200 Hz in 4 slices, 0.02 of the mean.
WINDOW, CHUNKS, MAX_SPREAD = 400, 4, 0.02


def _specific_force(tilt_rad, rate=200.0):
    """Return (specific force window, times) for a body tilting by tilt_rad(t) (rad)."""
    t = np.arange(WINDOW) / rate
    tilt = np.asarray(tilt_rad(t))
    return np.column_stack([9.81 * np.sin(tilt), np.zeros(WINDOW), 9.81 * np.cos(tilt)]), t


def test_motion_spread_ignores_standing_buzz():
    """A standing policy's leg buzz passes the stillness check."""
    # About 2 m/s^2 lateral at 12-30 Hz (the H1-2 sim's: 22 Hz).
    for freq in (12.0, 22.0, 30.0):
        ups, t = _specific_force(lambda t: 0.0 * t)
        ups[:, 1] += 2.7 * np.sin(2 * np.pi * freq * t + 0.7) + RNG.normal(0.0, 0.3, WINDOW)
        assert motion_spread(ups, CHUNKS) < MAX_SPREAD / 2


def test_motion_spread_sees_swaying():
    """A +-2 deg sway moves the up direction itself and fails the stillness check."""
    for freq in (0.3, 0.5, 1.0):
        ups, _ = _specific_force(lambda t: np.radians(2.0) * np.sin(2 * np.pi * freq * t))
        assert motion_spread(ups, CHUNKS) > MAX_SPREAD


def test_motion_spread_still():
    """A still window has no spread."""
    ups, _ = _specific_force(lambda t: 0.0 * t)
    assert motion_spread(ups, CHUNKS) < 1e-9


def test_g1_floor_with_upside_down_sensor():
    """The floor is found in an upside-down sensor frame, as on the G1."""
    points = plane(5000, -0.9)
    rotation = rpy_to_matrix(np.pi, 0.0, 0.0)
    up = rotation.T @ np.array([0.0, 0.0, 1.0])
    floor = find_floor_height(points @ rotation, up, 0.0, 0.5, 300)
    assert abs(floor + 0.9) < 0.01


def test_level_odom_follows_base():
    """Each odometry sample uses body -> pelvis at its own stamp, retried until TF has it."""
    from types import SimpleNamespace
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from tf2_ros import Buffer
    from golem_slam.level_odom import LevelOdom

    buffer = Buffer()
    published = []
    node = SimpleNamespace(
        odom_from_world=(np.eye(3), np.zeros(3)), pending_odometry=None,
        tf_buffer=buffer, body_frame='body',
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
    msg.header.stamp.sec = 3  # TF has no body -> pelvis at this stamp yet
    LevelOdom._on_odometry(node, msg)
    assert len(published) == 2
    # The transform arrives after the odometry callback; the retry publishes without
    # a new odometry message.
    transform.header.stamp.sec = 3
    transform.transform.translation.x = 0.4
    buffer.set_transform(transform, 'test')
    node._on_odometry = lambda sample: LevelOdom._on_odometry(node, sample)
    LevelOdom._retry_odometry(node)
    assert len(published) == 3
    assert published[-1].transform.translation.x == 2.4
    assert published[-1].header.stamp.sec == 3
    LevelOdom._retry_odometry(node)  # nothing pending once published
    assert len(published) == 3


def test_motion_spread_accepts_sample_lists():
    """motion_spread takes a plain list of samples."""
    assert motion_spread([[0, 0, 9.81]] * 8, 4) == 0.0
