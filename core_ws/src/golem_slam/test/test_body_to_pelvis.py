"""body_to_pelvis's transform helpers, against each robot's model and FAST-LIO configs."""
import glob
import os

from ament_index_python.packages import get_package_share_directory
from golem_ros2_models.paths import urdf_path
from golem_slam.body_to_pelvis import (
    body_to_pelvis, fastlio_extrinsic, invert, link_pose, static_transform_arguments, transform,
)
from golem_slam.robot_geometry import link_boxes, quat_to_matrix, rpy_to_matrix
import numpy as np
import pytest
import yaml

CONFIG = os.path.join(get_package_share_directory('golem_slam'), 'config')
FASTLIO_CONFIGS = sorted(glob.glob(os.path.join(CONFIG, '*', 'fastlio*.yaml')))


def _lidar_link(robot):
    with open(os.path.join(CONFIG, robot, 'pipeline.yaml')) as f:
        return yaml.safe_load(f)['lidar_link']


def test_g1_collision_model_loads_for_self_filter():
    with open(urdf_path('g1', ros=True)) as source:
        boxes = link_boxes(source.read(), 0.05)
    assert 'pelvis_contour_link' in boxes and 'torso_link' in boxes
    assert any('hand' in name for name in boxes)
    for low, high in boxes.values():
        assert np.isfinite(low).all() and np.isfinite(high).all()
        assert (high > low).all()


def test_g1_sensor_mount_and_extrinsic():
    config = os.path.join(CONFIG, 'g1', 'fastlio_sim.yaml')
    ext = fastlio_extrinsic(config)
    np.testing.assert_allclose(ext[:3, :3], np.eye(3))
    np.testing.assert_allclose(ext[:3, 3], [-0.011, -0.02329, 0.04412])
    mount = link_pose(urdf_path('g1'), 'pelvis', _lidar_link('g1'))
    assert mount[2, 2] < -0.99  # sensor up axis points down
    with open(config) as source:
        params = yaml.safe_load(source)['/**']['ros__parameters']
    assert params['publish']['scan_bodyframe_pub_en'] is True
    assert params['mapping']['extrinsic_est_en'] is False


def test_every_robot_has_its_fastlio_configs():
    names = [os.path.relpath(path, CONFIG) for path in FASTLIO_CONFIGS]
    assert names == ['g1/fastlio.yaml', 'g1/fastlio_sim.yaml', 'h12/fastlio.yaml']


@pytest.mark.parametrize('config', FASTLIO_CONFIGS,
                         ids=[os.path.relpath(p, CONFIG) for p in FASTLIO_CONFIGS])
def test_fastlio_configs_have_rigid_extrinsic(config):
    t = fastlio_extrinsic(config)
    np.testing.assert_allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-11)
    assert abs(np.linalg.det(t[:3, :3]) - 1.0) < 1e-11


@pytest.mark.parametrize('robot, joint, angle', [
    ('h12', 'torso_joint', 0.3),
    ('g1', 'waist_pitch_joint', 0.0256),  # HOMIE's standing waist pitch
], ids=['h12-torso-0.3', 'g1-waist-0.0256'])
def test_body_to_pelvis(robot, joint, angle):
    # FAST-LIO tracks its IMU, body, rigid with the lidar on the torso; the pelvis
    # hangs off it through the torso or waist joint
    extrinsic = fastlio_extrinsic(os.path.join(CONFIG, robot, 'fastlio.yaml'))
    world_pelvis = transform((0.3, -0.2, 0.744), rpy_to_matrix(0.02, -0.05, 0.4))
    # pelvis -> livox_frame, which is the lidar link itself (navigation.launch.py's static)
    pelvis_lidar = link_pose(urdf_path(robot), 'pelvis', _lidar_link(robot), {joint: angle})
    assert not np.allclose(pelvis_lidar, link_pose(urdf_path(robot), 'pelvis',
                                                   _lidar_link(robot)))
    world_body = world_pelvis @ pelvis_lidar @ invert(extrinsic)
    np.testing.assert_allclose(world_body @ body_to_pelvis(pelvis_lidar, extrinsic),
                               world_pelvis, atol=1e-9)


def test_h12_body_is_the_urdf_imu():
    # FAST-LIO's body on the H1-2 is the URDF's livox_imu_link at any torso angle
    extrinsic = fastlio_extrinsic(os.path.join(CONFIG, 'h12', 'fastlio.yaml'))
    torso = {'torso_joint': 0.3}
    pelvis_lidar = link_pose(urdf_path('h12'), 'pelvis', 'livox_link', torso)
    pelvis_imu = link_pose(urdf_path('h12'), 'pelvis', 'livox_imu_link', torso)
    np.testing.assert_allclose(invert(body_to_pelvis(pelvis_lidar, extrinsic)), pelvis_imu,
                               atol=1e-9)


def test_static_transform_arguments_round_trip():
    t = transform((0.1, -0.02, 0.7), rpy_to_matrix(np.pi, 0.15, 0.3))
    args = static_transform_arguments(t, 'pelvis', 'body')
    values = dict(zip(args[0::2], args[1::2]))
    assert values['--frame-id'] == 'pelvis' and values['--child-frame-id'] == 'body'
    xyz = [float(values[f'--{k}']) for k in 'xyz']
    quat = [float(values[f'--{k}']) for k in ('qx', 'qy', 'qz', 'qw')]
    np.testing.assert_allclose(transform(xyz, quat_to_matrix(*quat)), t, atol=1e-10)
