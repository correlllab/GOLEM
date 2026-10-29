'''frames.py against an independent forward kinematics (Pinocchio) of the G1 URDF

The standing robot is built with Pinocchio: the pelvis level at PELVIS_HEIGHT,
the legs bent (feet flat) until the soles touch the floor at z = 0. Lidar rays
from the upside-down MID-360 then hit that floor and a wall, and the points,
taken through FAST_LIO's frames the way the launch file sets them up, have to
land where they are in the world.
'''

import os

import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation

from g1_navigation import frames
from g1_navigation.level_frame import level_rotation
from golem_ros2_models.paths import urdf_path

URDF = urdf_path('g1')
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
SIM_CONFIG = os.path.join(CONFIG_DIR, 'g1_fastlio_sim.yaml')
PELVIS_HEIGHT = 0.744  # HOMIE standing
LEGS = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')
# the foot's collision spheres: radius 0.005 at z = -0.03 in ankle_roll_link
SOLES = [(-0.05, 0.025), (-0.05, -0.025), (0.12, 0.03), (0.12, -0.03)]
SOLE_Z = -0.035


@pytest.fixture(scope='module')
def model():
    m = pin.buildModelFromUrdf(URDF)
    return m, m.createData()


def _pin_pose(model, positions, frame):
    m, data = model
    q = pin.neutral(m)
    for name, value in positions.items():
        q[m.joints[m.getJointId(name)].idx_q] = value
    pin.framesForwardKinematics(m, data, q)
    placement = data.oMf[m.getFrameId(frame)]
    return frames.transform(placement.translation, placement.rotation)


def _standing_legs(model, height):
    '''Leg angles with the feet flat and the soles height below the pelvis (Pinocchio FK)'''
    def legs(bend):  # HOMIE's default pose scaled: hip -0.1, knee 0.3, ankle -0.2 per unit
        pose = dict(zip(LEGS, (-0.1 * bend, 0.0, 0.0, 0.3 * bend, -0.2 * bend, 0.0)))
        return {f'{side}_{j}_joint': q for side in ('left', 'right') for j, q in pose.items()}

    def sole_height(bend):
        foot = _pin_pose(model, legs(bend), 'left_ankle_roll_link')
        return (foot @ [0.0, 0.0, SOLE_Z, 1.0])[2]

    low, high = 0.0, 5.0  # bending the knees raises the soles towards the pelvis
    for _ in range(60):
        mid = 0.5 * (low + high)
        low, high = (low, mid) if sole_height(mid) > -height else (mid, high)
    return legs(0.5 * (low + high))


def _floor_and_wall(world_lidar):
    '''Upside-down MID-360 rays that hit the floor (z = 0) and a wall (x = 2): world points'''
    el, az = np.meshgrid(np.radians(np.linspace(-7, 52, 12)), np.radians(np.arange(0, 360, 15)))
    rays = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
                    axis=-1).reshape(-1, 3)
    origin, world_rays = world_lidar[:3, 3], rays @ world_lidar[:3, :3].T
    with np.errstate(divide='ignore'):
        to_floor = np.where(world_rays[:, 2] < 0, -origin[2] / world_rays[:, 2], np.inf)
        to_wall = np.where(world_rays[:, 0] > 0, (2.0 - origin[0]) / world_rays[:, 0], np.inf)
    t = np.minimum(to_floor, to_wall)
    hit = np.isfinite(t)
    return origin + world_rays[hit] * t[hit, None], to_floor[hit] < to_wall[hit]


def test_link_pose_matches_pinocchio(model):
    rng = np.random.default_rng(0)
    joints = frames.load_joints(URDF)
    movable = [j.name for j in joints.values() if j.type == 'revolute']
    for _ in range(5):
        positions = {name: rng.uniform(-0.5, 0.5) for name in movable}
        for link in ('mid360_link', 'd435_link', 'left_ankle_roll_link',
                     'right_hand_palm_link', 'imu_in_pelvis'):
            np.testing.assert_allclose(frames.link_pose(URDF, 'pelvis', link, positions),
                                       _pin_pose(model, positions, link), atol=1e-9)


def test_lidar_is_upside_down_on_the_torso():
    pelvis_lidar = frames.link_pose(URDF, 'pelvis', 'mid360_link')
    np.testing.assert_allclose(pelvis_lidar[:3, 3], [-0.00368, 0.00003, 0.472434], atol=1e-6)
    roll, pitch, yaw = Rotation.from_matrix(pelvis_lidar[:3, :3]).as_euler('xyz')
    assert abs(abs(roll) - np.pi) < 1e-9 and abs(pitch - 0.0511207) < 1e-6 and abs(yaw) < 1e-9


def test_imu_is_above_the_optical_centre_on_the_g1():
    # FAST_LIO: p_imu = R p_lidar + T, so the IMU is at -T in the lidar frame:
    # 4.4 cm towards the sensor's base, which is up on the G1
    lidar_body = frames.invert(frames.fastlio_extrinsic(SIM_CONFIG))
    np.testing.assert_allclose(lidar_body[:3, 3], [0.011, 0.02329, -0.04412])
    pelvis_body = frames.pelvis_to_body(URDF, frames.fastlio_extrinsic(SIM_CONFIG))
    pelvis_lidar = frames.link_pose(URDF, 'pelvis', 'mid360_link')
    assert 0.04 < pelvis_body[2, 3] - pelvis_lidar[2, 3] < 0.05


def test_standing_pose_puts_the_soles_on_the_floor(model):
    legs = _standing_legs(model, PELVIS_HEIGHT)
    for side in ('left', 'right'):
        foot = frames.transform((0.0, 0.0, PELVIS_HEIGHT)) @ _pin_pose(
            model, legs, f'{side}_ankle_roll_link')
        np.testing.assert_allclose(foot[:3, :3], np.eye(3), atol=1e-9)  # flat
        soles = [(foot @ [x, y, SOLE_Z, 1.0])[2] for x, y in SOLES]
        np.testing.assert_allclose(soles, 0.0, atol=1e-6)


def test_floor_is_z0_in_odom_and_nothing_is_mirrored(model):
    '''World points seen by the lidar, through odom -> camera_init -> body, land where they are'''
    legs = _standing_legs(model, PELVIS_HEIGHT)
    world_pelvis = frames.transform((0.0, 0.0, PELVIS_HEIGHT))  # level, at odom's origin
    world_lidar = world_pelvis @ _pin_pose(model, legs, 'mid360_link')
    points, on_floor = _floor_and_wall(world_lidar)
    assert on_floor.sum() > 50 and (~on_floor).sum() > 50

    # what the sensor reports, in livox_frame
    in_lidar = (frames.invert(world_lidar) @ np.c_[points, np.ones(len(points))].T).T
    # FAST_LIO at startup: camera_init = body, points taken into body by the extrinsic
    odom_camera_init, body_pelvis = frames.fastlio_transforms(URDF, SIM_CONFIG, PELVIS_HEIGHT)
    in_odom = (odom_camera_init @ frames.fastlio_extrinsic(SIM_CONFIG) @ in_lidar.T).T[:, :3]
    np.testing.assert_allclose(in_odom, points, atol=1e-9)
    np.testing.assert_allclose(in_odom[on_floor, 2], 0.0, atol=1e-9)
    np.testing.assert_allclose(in_odom[~on_floor, 0], 2.0, atol=1e-9)  # the wall ahead, at +x

    # the pelvis in odom at startup: at pelvis_height, level, facing +x
    odom_pelvis = odom_camera_init @ body_pelvis
    np.testing.assert_allclose(odom_pelvis, world_pelvis, atol=1e-12)


def test_body_stays_on_the_torso_at_any_waist_angle(model):
    '''pelvis -> body through the waist against Pinocchio, and body_to_pelvis's inverse of it'''
    extrinsic = frames.fastlio_extrinsic(SIM_CONFIG)
    rng = np.random.default_rng(2)
    torso_body = None
    for _ in range(5):
        waist = {f'waist_{j}_joint': rng.uniform(-0.3, 0.3) for j in ('yaw', 'roll', 'pitch')}
        pelvis_lidar = _pin_pose(model, waist, 'mid360_link')
        pelvis_body = frames.pelvis_to_body(URDF, extrinsic, positions=waist)
        np.testing.assert_allclose(pelvis_body, pelvis_lidar @ frames.invert(extrinsic),
                                   atol=1e-9)
        np.testing.assert_allclose(frames.body_to_pelvis(pelvis_lidar, extrinsic),
                                   frames.invert(pelvis_body), atol=1e-9)
        # the IMU is fixed on the torso, whatever the waist does
        on_torso = frames.invert(_pin_pose(model, waist, 'torso_link')) @ pelvis_body
        torso_body = on_torso if torso_body is None else torso_body
        np.testing.assert_allclose(on_torso, torso_body, atol=1e-9)


def test_a_fixed_body_to_pelvis_would_count_the_waist_twice(model):
    '''The lidar in TF: FAST_LIO's body, body -> pelvis, then robot_state_publisher's waist'''
    extrinsic = frames.fastlio_extrinsic(SIM_CONFIG)
    w = 0.0256  # HOMIE's standing waist pitch, with the pelvis leaning back by as much
    waist = {'waist_pitch_joint': w}
    world_pelvis = frames.transform((0.0, 0.0, PELVIS_HEIGHT),
                                    Rotation.from_euler('y', -w).as_matrix())
    world_lidar = world_pelvis @ _pin_pose(model, waist, 'mid360_link')
    world_body = world_lidar @ frames.invert(extrinsic)  # what FAST_LIO tracks
    pelvis_lidar = frames.link_pose(URDF, 'pelvis', 'mid360_link', waist)
    fixed = frames.invert(frames.pelvis_to_body(URDF, extrinsic))  # the waist at 0
    following = frames.body_to_pelvis(pelvis_lidar, extrinsic)  # what body_to_pelvis publishes
    for body_pelvis, error in ((fixed, w), (following, 0.0)):
        tf_lidar = world_body @ body_pelvis @ pelvis_lidar
        angle = Rotation.from_matrix(tf_lidar[:3, :3].T @ world_lidar[:3, :3]).magnitude()
        assert abs(angle - error) < 1e-8, (angle, error)
    np.testing.assert_allclose(world_body @ following, world_pelvis, atol=1e-9)


def test_camera_init_is_upside_down_so_it_cannot_be_odom():
    # the H1 used camera_init as its odom frame; on the G1 its z axis points at the floor
    odom_camera_init, _ = frames.fastlio_transforms(URDF, SIM_CONFIG, PELVIS_HEIGHT)
    assert odom_camera_init[2, 2] < -0.99
    assert odom_camera_init[1, 1] < -0.99  # and its y axis points right: a map in it is mirrored
    np.testing.assert_allclose(odom_camera_init[2, 3], PELVIS_HEIGHT + 0.472434 + 0.0441,
                               atol=2e-3)


def test_pelvis_height_only_moves_odom_up_and_down():
    low, _ = frames.fastlio_transforms(URDF, SIM_CONFIG, 0.70)
    high, body_pelvis = frames.fastlio_transforms(URDF, SIM_CONFIG, 0.75)
    np.testing.assert_allclose(high - low, frames.transform((0.0, 0.0, 0.05)) - np.eye(4),
                               atol=1e-12)
    _, body_pelvis_low = frames.fastlio_transforms(URDF, SIM_CONFIG, 0.70)
    np.testing.assert_allclose(body_pelvis, body_pelvis_low)


def test_a_rotated_driver_frame_leaves_the_imu_in_place():
    # config/g1_fastlio.yaml's TODO: if the driver rolls the points by pi
    # (MID360_config.json roll 180) but not the IMU, extrinsic_R becomes that
    # roll and livox_frame is mid360_link rolled by pi; body must not move
    roll = frames.transform(rotation=Rotation.from_euler('x', np.pi).as_matrix())
    raw = frames.fastlio_extrinsic(SIM_CONFIG)
    rolled = raw.copy()
    rolled[:3, :3] = roll[:3, :3]
    np.testing.assert_allclose(frames.pelvis_to_body(URDF, rolled, lidar_frame=roll),
                               frames.pelvis_to_body(URDF, raw), atol=1e-12)


@pytest.mark.parametrize('config', ['g1_fastlio_sim.yaml', 'g1_fastlio.yaml'])
def test_configs_have_a_rigid_extrinsic(config):
    t = frames.fastlio_extrinsic(os.path.join(CONFIG_DIR, config))
    np.testing.assert_allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(t[:3, :3]) - 1.0) < 1e-12


def test_static_transform_arguments_round_trip():
    t = frames.fastlio_transforms(URDF, SIM_CONFIG, PELVIS_HEIGHT)[0]
    args = frames.static_transform_arguments(t, 'odom', 'camera_init')
    values = dict(zip(args[0::2], args[1::2]))
    assert values['--frame-id'] == 'odom' and values['--child-frame-id'] == 'camera_init'
    xyz = [float(values[f'--{k}']) for k in 'xyz']
    quat = [float(values[f'--{k}']) for k in ('qx', 'qy', 'qz', 'qw')]
    np.testing.assert_allclose(frames.transform(xyz, Rotation.from_quat(quat).as_matrix()), t,
                               atol=1e-10)


def test_level_rotation_keeps_the_heading_and_levels_the_frame():
    rng = np.random.default_rng(1)
    for _ in range(50):
        roll, pitch = rng.uniform(-0.3, 0.3, 2)
        yaw = rng.uniform(-np.pi, np.pi)
        base = Rotation.from_euler('xyz', (roll, pitch, yaw))
        level = base * Rotation.from_quat(level_rotation(base.as_quat()))
        np.testing.assert_allclose(level.as_matrix()[:, 2], [0.0, 0.0, 1.0], atol=1e-12)
        x_axis = base.as_matrix()[:, 0]
        heading = np.arctan2(x_axis[1], x_axis[0])
        np.testing.assert_allclose(level.as_euler('xyz'), [0.0, 0.0, heading], atol=1e-12)
