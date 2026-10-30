"""Exercise launch construction without starting processes or sending commands."""
import importlib
import os

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters


def pipeline(robot, **overrides):
    module_name = 'golem_slam.navigation_launch'
    assert importlib.util.find_spec(module_name), 'shared robot launch module is missing'
    module = importlib.import_module(module_name)
    description = module.generate_navigation_launch_description(robot)
    context = LaunchContext()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            context.launch_configurations[action.name] = perform_substitutions(
                context, action.default_value)
    context.launch_configurations.update(overrides)
    setup, = [a for a in description.entities if isinstance(a, OpaqueFunction)]
    return context, setup.execute(context)


def nodes(context, actions):
    result = {}
    for action in actions:
        if isinstance(action, Node):
            values = {}
            for entry in evaluate_parameters(context, action._Node__parameters):
                if isinstance(entry, dict):
                    values.update(entry)
            result[action._Node__node_name] = (action, values)
    return result


def test_h12_preserves_floor_and_scan_defaults():
    context, actions = pipeline('h12')
    ns = nodes(context, actions)
    assert ns['level_odom'][1].get('dynamic_base_transform', False) is False
    scan = ns['cloud_registered_to_laserscan'][1]
    assert (scan['min_height'], scan['max_height'], scan['range_min'], scan['range_max']) == (
        0.25, 1.7, 0.3, 3.0)
    assert 'body_to_pelvis_broadcaster' in ns
    assert any(isinstance(a, IncludeLaunchDescription) for a in actions)


def test_g1_uses_shared_cloud_pipeline_and_dynamic_waist():
    context, actions = pipeline('g1', sim='true', use_sim_time='true')
    ns = nodes(context, actions)
    assert {'self_filter', 'level_odom', 'octomap_server', 'slam_toolbox',
            'cloud_registered_to_laserscan', 'body_to_pelvis', 'fast_lio'} <= set(ns)
    assert not {'level_frame', 'livox_to_pointcloud', 'odom_camera_init_tf',
                'body_to_pelvis_broadcaster'} & set(ns)
    assert 'level_cloud' in ns
    assert ns['cloud_registered_to_laserscan'][1]['target_frame'] == ''
    assert ns['level_odom'][1]['dynamic_base_transform'] is True
    assert ns['level_odom'][1]['min_floor_depth'] == 0.5
    assert ns['cloud_registered_to_laserscan'][1]['use_sim_time'] is True
    assert ns['body_to_pelvis'][0]._Node__package == 'golem_slam'
    assert ns['body_to_pelvis'][0]._Node__node_executable == 'g1_body_to_pelvis'
    assert 'frontier_explore' not in ns


def test_g1_real_mode_builds_the_pipeline():
    context, actions = pipeline('g1', fast_lio_config_file='g1_fastlio.yaml')
    ns = nodes(context, actions)
    assert {'fast_lio', 'body_to_pelvis', 'slam_toolbox', 'level_odom'} <= set(ns)
    assert ns['level_odom'][1]['use_sim_time'] is False


@pytest.mark.parametrize('overrides,match', [
    ({'sim': 'true', 'odom_source': 'sim_gt'}, 'FAST-LIO'),
    ({'sim': 'true', 'odom_source': 'wheel'}, 'FAST-LIO'),
])
def test_g1_invalid_modes_refused_before_actions(overrides, match):
    with pytest.raises(RuntimeError, match=match):
        pipeline('g1', **overrides)


def test_g1_motion_limits_and_static_map_source_agree():
    path = os.path.join(get_package_share_directory('golem_slam'), 'config', 'g1_nav2.yaml')
    assert os.path.exists(path), 'G1 Nav2 profile is missing'
    with open(path) as source:
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


def test_g1_recovery_commands_use_velocity_smoother():
    from launch.actions import GroupAction
    from launch_ros.actions import SetRemap
    context, actions = pipeline('g1', sim='true')
    groups = [a for a in actions if isinstance(a, GroupAction)]
    assert len(groups) == 1, 'G1 Nav2 needs a scoped recovery remap'
    remaps = [a for a in groups[0].get_sub_entities() if isinstance(a, SetRemap)]
    assert len(remaps) == 1
    remaps[0].execute(context)
    remappings = context.launch_configurations['ros_remaps']
    assert ('behavior_server:cmd_vel', 'cmd_vel_nav') in remappings
    _, h12 = pipeline('h12')
    assert not any(isinstance(a, GroupAction) for a in h12)
