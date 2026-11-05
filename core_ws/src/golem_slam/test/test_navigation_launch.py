"""
navigation.launch.py's graph, built without starting processes.

One graph on both robots: the same nodes, executables and remaps, with each
robot's differences in its config directory, config/<robot>/.
"""
import importlib
import os

import numpy as np
import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from golem_ros2_models.paths import urdf_path
from launch import LaunchContext
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node, SetRemap
from launch_ros.utilities import evaluate_parameters

from golem_slam.body_to_pelvis import fastlio_extrinsic
from golem_slam.robot_geometry import fixed_transform

ROBOTS = ('h12', 'g1')
SHARE = get_package_share_directory('golem_slam')
CONFIG = os.path.join(SHARE, 'config')
LAUNCH = os.path.join(SHARE, 'launch', 'navigation.launch.py')
# the pipeline profiles' values (config/<robot>/pipeline.yaml)
PROFILES = {
    'h12': {'lidar_link': 'livox_link', 'scan': (0.25, 1.7), 'min_floor_depth': 1.0},
    'g1': {'lidar_link': 'mid360_link', 'scan': (0.15, 1.3), 'min_floor_depth': 0.5},
}


def _description():
    spec = importlib.util.spec_from_file_location('navigation_launch_file', LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def pipeline(robot, **overrides):
    """Return (context, actions) of navigation.launch.py with robot and overrides."""
    description = _description()
    context = LaunchContext()
    context.launch_configurations.update(robot=robot, **overrides)
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument) and \
                action.name not in context.launch_configurations:
            assert action.default_value is not None, f'{action.name} has no default'
            context.launch_configurations[action.name] = perform_substitutions(
                context, action.default_value)
    setup, = [a for a in description.entities if isinstance(a, OpaqueFunction)]
    return context, setup.execute(context)


def nodes(context, actions):
    """Return {node name: (Node, {parameter: value}, [parameter file])}."""
    result = {}
    for action in actions:
        if isinstance(action, Node):
            values, files = {}, []
            for entry in evaluate_parameters(context, action._Node__parameters):
                if isinstance(entry, dict):
                    values.update(entry)
                else:
                    files.append(str(entry))
            result[action._Node__node_name] = (action, values, files)
    return result


def _text(context, value):
    return perform_substitutions(context, normalize_to_list_of_substitutions(value))


def remaps(context, node):
    """Return a node's remappings as (from, to) text pairs."""
    return tuple((_text(context, a), _text(context, b)) for a, b in node._Node__remappings)


def _include_name(context, include):
    include.launch_description_source.get_launch_description(context)  # resolves its location
    return os.path.basename(include.launch_description_source.location)


def graph(context, actions):
    """Return the graph's shape: nodes with their remaps, scoped remaps and includes."""
    shape = set()
    for action in actions:
        if isinstance(action, Node):
            shape.add(('node', action._Node__package, action._Node__node_executable,
                       action._Node__node_name, remaps(context, action)))
        elif isinstance(action, GroupAction):
            for entity in action.get_sub_entities():
                if isinstance(entity, SetRemap):
                    shape.add(('remap', _text(context, entity._SetRemap__src),
                               _text(context, entity._SetRemap__dst)))
                elif isinstance(entity, IncludeLaunchDescription):
                    shape.add(('include', _include_name(context, entity)))
        elif isinstance(action, IncludeLaunchDescription):
            shape.add(('include', _include_name(context, action)))
    return shape


def test_robot_is_required_and_the_files_default_per_robot():
    declared = {a.name: a for a in _description().entities
                if isinstance(a, DeclareLaunchArgument)}
    assert set(declared) == {'robot', 'use_sim_time', 'fast_lio_config', 'nav2_params_file',
                             'octomap_params_file', 'slam_params_file'}
    assert declared['robot'].default_value is None


def test_one_graph():
    h12 = graph(*pipeline('h12'))
    g1 = graph(*pipeline('g1'))
    assert h12 == g1
    names = {entry[3] for entry in h12 if entry[0] == 'node'}
    assert names == {'livox_frame_tf', 'fast_lio', 'body_to_pelvis', 'level_odom',
                     'self_filter', 'level_cloud', 'octomap_server',
                     'cloud_registered_to_laserscan', 'slam_toolbox'}
    # Humble's navigation_launch remaps the controller but not the recoveries
    assert ('remap', 'behavior_server:cmd_vel', 'cmd_vel_nav') in h12
    assert ('include', 'navigation_launch.py') in h12


def test_the_scan_reads_the_level_cloud():
    for robot in ROBOTS:
        context, actions = pipeline(robot)
        ns = nodes(context, actions)
        scan, values, _ = ns['cloud_registered_to_laserscan']
        assert values['target_frame'] == ''
        assert ('cloud_in', '/cloud_base_filtered') in remaps(context, scan)
        level_cloud = remaps(context, ns['level_cloud'][0])
        assert ('cloud_in', '/cloud_body_filtered') in level_cloud
        assert ('cloud_out', '/cloud_base_filtered') in level_cloud
        assert (values['range_min'], values['range_max']) == (0.3, 3.0)


@pytest.mark.parametrize('robot', ROBOTS)
def test_profiles_drive_the_differences(robot):
    profile = PROFILES[robot]
    with open(os.path.join(CONFIG, robot, 'pipeline.yaml')) as f:
        assert yaml.safe_load(f) == {
            'lidar_link': profile['lidar_link'],
            'scan': dict(zip(('min_height', 'max_height'), profile['scan'])),
            'level_odom': {'min_floor_depth': profile['min_floor_depth']},
        }
    context, actions = pipeline(robot)
    ns = nodes(context, actions)
    scan = ns['cloud_registered_to_laserscan'][1]
    assert (scan['min_height'], scan['max_height']) == profile['scan']
    assert ns['level_odom'][1]['min_floor_depth'] == profile['min_floor_depth']
    assert 'dynamic_base_transform' not in ns['level_odom'][1]
    # the driver and both sims stamp the lidar and IMU livox_frame, the raw frame of
    # the URDF's lidar link
    frame_tf = ns['livox_frame_tf'][0]
    assert list(frame_tf._Node__arguments[-4:]) == [
        '--frame-id', profile['lidar_link'], '--child-frame-id', 'livox_frame']
    fast_lio = os.path.join(CONFIG, robot, 'fastlio.yaml')
    assert ns['fast_lio'][2] == [fast_lio]
    assert ns['body_to_pelvis'][1]['fastlio_config'] == fast_lio
    assert ns['body_to_pelvis'][1]['lidar_frame'] == 'livox_frame'
    assert ns['octomap_server'][2] == [os.path.join(CONFIG, robot, 'octomap.yaml')]
    assert ns['slam_toolbox'][2] == [os.path.join(CONFIG, 'slam_toolbox.yaml')]
    assert context.launch_configurations['nav2_params_file'] == os.path.join(
        CONFIG, robot, 'nav2.yaml')


def test_the_arguments_override_the_profile_files():
    sim = os.path.join(CONFIG, 'g1', 'fastlio_sim.yaml')
    ns = nodes(*pipeline('g1', fast_lio_config=sim, use_sim_time='true'))
    assert ns['fast_lio'][2] == [sim]
    assert ns['body_to_pelvis'][1]['fastlio_config'] == sim
    for name, (_, values, _) in ns.items():
        if name != 'livox_frame_tf':   # static_transform_publisher: arguments only
            assert values['use_sim_time'] is True, name


def test_a_robot_without_a_config_directory_is_refused():
    with pytest.raises(RuntimeError, match="no config directory for robot 'h1'"):
        pipeline('h1')


def test_a_profile_missing_a_key_is_refused(tmp_path):
    from golem_slam.navigation_launch import load_pipeline
    path = tmp_path / 'pipeline.yaml'
    path.write_text('lidar_link: livox_link\nscan: {min_height: 0.25, max_height: 1.7}\n')
    with pytest.raises(RuntimeError, match='level_odom.min_floor_depth'):
        load_pipeline(str(path))


def test_h12_extrinsic_matches_urdf():
    with open(urdf_path('h12', ros=True)) as f:
        rot, trans = fixed_transform(f.read(), 'livox_imu_link', 'livox_link')
    path = os.path.join(CONFIG, 'h12', 'fastlio.yaml')
    extrinsic = fastlio_extrinsic(path)
    np.testing.assert_allclose(extrinsic[:3, :3], rot, atol=1e-6)
    np.testing.assert_allclose(extrinsic[:3, 3], trans, atol=1e-6)
    with open(path) as f:
        assert yaml.safe_load(f)['/**']['ros__parameters']['mapping']['extrinsic_est_en'] is True


def test_g1_motion_limits_and_static_map_source_agree():
    with open(os.path.join(CONFIG, 'g1', 'nav2.yaml')) as source:
        config = yaml.safe_load(source)
    controller = config['controller_server']['ros__parameters']['FollowPath']
    smoother = config['velocity_smoother']['ros__parameters']
    assert controller['motion_model'] == 'DiffDrive'
    assert controller['vy_max'] == 0.0
    assert smoother['max_velocity'] == [controller['vx_max'], 0.0, controller['wz_max']]
    assert smoother['min_velocity'] == [controller['vx_min'], 0.0, -controller['wz_max']]
    for name in ('global_costmap', 'local_costmap'):
        params = config[name][name]['ros__parameters']
        assert params['static_layer']['map_topic'] == '/projected_map'
