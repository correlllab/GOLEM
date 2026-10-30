"""Installed G1 model, waist transform helpers and FAST-LIO profile agree with golem_slam."""
import os

from ament_index_python.packages import get_package_share_directory
from golem_ros2_models.paths import urdf_path
from golem_slam.g1_body_to_pelvis import (
    body_to_pelvis, fastlio_extrinsic, fastlio_transforms, invert, link_pose,
    pelvis_to_body, static_transform_arguments, transform,
)
from golem_slam.robot_geometry import link_boxes, quat_to_matrix, rpy_to_matrix
import numpy as np
import pytest
import yaml


def test_g1_collision_model_loads_for_self_filter():
    with open(urdf_path('g1', ros=True)) as source:
        boxes = link_boxes(source.read(), 0.05)
    assert 'pelvis_contour_link' in boxes and 'torso_link' in boxes
    assert any('hand' in name for name in boxes)
    for low, high in boxes.values():
        assert np.isfinite(low).all() and np.isfinite(high).all()
        assert (high > low).all()


def test_g1_sensor_mount_and_extrinsic():
    config = os.path.join(
        get_package_share_directory('golem_slam'), 'config', 'g1_fastlio_sim.yaml')
    ext = fastlio_extrinsic(config)
    np.testing.assert_allclose(ext[:3, :3], np.eye(3))
    np.testing.assert_allclose(ext[:3, 3], [-0.011, -0.02329, 0.04412])
    mount = link_pose(urdf_path('g1'), 'pelvis', 'mid360_link')
    assert mount[2, 2] < -0.99  # sensor up axis points down
    with open(config) as source:
        params = yaml.safe_load(source)['/**']['ros__parameters']
    assert params['publish']['scan_bodyframe_pub_en'] is True
    assert params['mapping']['extrinsic_est_en'] is False


@pytest.mark.parametrize('config_name', ['g1_fastlio_sim.yaml', 'g1_fastlio.yaml'])
def test_g1_fastlio_configs_have_rigid_extrinsic(config_name):
    config = os.path.join(
        get_package_share_directory('golem_slam'), 'config', config_name)
    t = fastlio_extrinsic(config)
    np.testing.assert_allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(t[:3, :3]) - 1.0) < 1e-12


def test_g1_body_to_pelvis_tracks_waist_without_double_counting():
    urdf = urdf_path('g1')
    config = os.path.join(
        get_package_share_directory('golem_slam'), 'config', 'g1_fastlio_sim.yaml')
    extrinsic = fastlio_extrinsic(config)
    w = 0.0256  # HOMIE standing waist pitch
    waist = {'waist_pitch_joint': w}
    world_pelvis = transform((0.0, 0.0, 0.744), rpy_to_matrix(0.0, -w, 0.0))
    pelvis_lidar = link_pose(urdf, 'pelvis', 'mid360_link', waist)
    world_lidar = world_pelvis @ pelvis_lidar
    world_body = world_lidar @ invert(extrinsic)
    following = body_to_pelvis(pelvis_lidar, extrinsic)
    np.testing.assert_allclose(world_body @ following, world_pelvis, atol=1e-9)
    np.testing.assert_allclose(
        pelvis_to_body(urdf, extrinsic, positions=waist),
        invert(following),
        atol=1e-9,
    )


def test_static_transform_arguments_round_trip():
    urdf = urdf_path('g1')
    config = os.path.join(
        get_package_share_directory('golem_slam'), 'config', 'g1_fastlio_sim.yaml')
    t = fastlio_transforms(urdf, config, 0.744)[0]
    args = static_transform_arguments(t, 'odom', 'camera_init')
    values = dict(zip(args[0::2], args[1::2]))
    assert values['--frame-id'] == 'odom' and values['--child-frame-id'] == 'camera_init'
    xyz = [float(values[f'--{k}']) for k in 'xyz']
    quat = [float(values[f'--{k}']) for k in ('qx', 'qy', 'qz', 'qw')]
    np.testing.assert_allclose(transform(xyz, quat_to_matrix(*quat)), t, atol=1e-10)
