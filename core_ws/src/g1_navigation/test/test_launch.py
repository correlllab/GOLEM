'''g1_navigation.launch.py resolves for both odom sources (needs the workspace built, sourced)'''

import importlib.util
import os
import subprocess

import numpy as np
import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
from scipy.spatial.transform import Rotation

from g1_navigation import frames
from golem_ros2_models.paths import urdf_path

SHARE = get_package_share_directory('g1_navigation')
LAUNCH = os.path.join(SHARE, 'launch', 'g1_navigation.launch.py')
NAV2 = {'controller_server', 'planner_server', 'behavior_server', 'bt_navigator',
        'velocity_smoother', 'lifecycle_manager_navigation'}
COMMON = {'livox_frame_tf', 'livox_to_pointcloud', 'level_frame', 'pointcloud_to_laserscan',
          'slam_toolbox'} | NAV2
FASTLIO = {'fast_lio', 'odom_camera_init_tf', 'body_to_pelvis'}


def _launch_module():
    spec = importlib.util.spec_from_file_location('g1_navigation_launch', LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve(**arguments):
    '''{node name: Node} that the launch file starts with these arguments, and the context'''
    description = _launch_module().generate_launch_description()
    context = LaunchContext()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            value = arguments.pop(entity.name, None)
            if value is None:
                value = perform_substitutions(context, entity.default_value)
            context.launch_configurations[entity.name] = value
    assert not arguments, f'not launch arguments: {arguments}'
    [setup] = [e for e in description.entities if isinstance(e, OpaqueFunction)]
    nodes = {}
    for action in setup.execute(context):
        assert isinstance(action, Node)
        nodes[action._Node__node_name] = action
    return nodes, context


def _remappings(node, context):
    return {perform_substitutions(context, list(a)): perform_substitutions(context, list(b))
            for a, b in node._Node__remappings}


def _parameters(node, context):
    '''{name: value} of a Node's parameters given as dicts (not files)'''
    values = {}
    for params in evaluate_parameters(context, node._Node__parameters):
        if isinstance(params, dict):
            values.update(params)
    return values


def _static_transform(node):
    values = dict(zip(node._Node__arguments[0::2], node._Node__arguments[1::2]))
    xyz = [float(values[f'--{k}']) for k in 'xyz']
    quat = [float(values[f'--{k}']) for k in ('qx', 'qy', 'qz', 'qw')]
    t = frames.transform(xyz, Rotation.from_quat(quat).as_matrix())
    return values['--frame-id'], values['--child-frame-id'], t


def test_show_args():
    out = subprocess.run(['ros2', 'launch', 'g1_navigation', 'g1_navigation.launch.py',
                          '--show-args'], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    for name in ('odom_source', 'sim', 'use_sim_time', 'fastlio_config', 'pelvis_height',
                 'nav2_params', 'slam_params', 'frontier_explore'):
        assert f"'{name}'" in out.stdout


@pytest.mark.parametrize('sim', ['false', 'true'])
def test_fastlio_mode(sim):
    nodes, context = _resolve(odom_source='fastlio', sim=sim, pelvis_height='0.73')
    assert set(nodes) == COMMON | FASTLIO
    config = os.path.join(SHARE, 'config', 'g1_fastlio_sim.yaml' if sim == 'true'
                          else 'g1_fastlio.yaml')
    fast_lio = nodes['fast_lio']
    assert (fast_lio.node_package, fast_lio.node_executable) == ('fast_lio', 'fastlio_mapping')
    params_file = fast_lio._Node__parameters[0]  # a launch_ros ParameterFile
    assert perform_substitutions(context, params_file.param_file) == config

    odom_camera_init, _ = frames.fastlio_transforms(urdf_path('g1'), config, 0.73)
    for name, parent, child, expected in (
            ('odom_camera_init_tf', 'odom', 'camera_init', odom_camera_init),
            ('livox_frame_tf', 'mid360_link', 'livox_frame', np.eye(4))):
        got_parent, got_child, t = _static_transform(nodes[name])
        assert (got_parent, got_child) == (parent, child)
        np.testing.assert_allclose(t, expected, atol=1e-9)
    # body -> pelvis follows the waist, from the same FAST_LIO config
    body_to_pelvis = nodes['body_to_pelvis']
    assert (body_to_pelvis.node_package, body_to_pelvis.node_executable) == (
        'g1_navigation', 'body_to_pelvis')
    assert _parameters(body_to_pelvis, context)['fastlio_config'] == config


def test_sim_gt_mode_has_no_fastlio():
    nodes, _ = _resolve(odom_source='sim_gt', use_sim_time='true')
    assert set(nodes) == COMMON


def test_scan_and_cmd_vel_wiring():
    nodes, context = _resolve(odom_source='sim_gt')
    scan = nodes['pointcloud_to_laserscan']
    assert _remappings(scan, context) == {'cloud_in': '/lidar_points', 'scan': '/scan'}
    # the converter places the cloud in pelvis_level; pointcloud_to_laserscan needs no TF
    assert _parameters(nodes['livox_to_pointcloud'], context)['target_frame'] == 'pelvis_level'
    assert _launch_module().SCAN['target_frame'] == ''
    # pointcloud_to_laserscan on the wall clock even with sim time, the rest on sim time
    nodes, context = _resolve(odom_source='sim_gt', use_sim_time='true')
    assert _parameters(nodes['pointcloud_to_laserscan'], context)['use_sim_time'] is False
    for name in ('livox_to_pointcloud', 'level_frame', 'slam_toolbox', 'controller_server'):
        assert _parameters(nodes[name], context)['use_sim_time'] is True, name
    # the controller and the behaviors feed the smoother; only the smoother writes /cmd_vel
    for name in ('controller_server', 'behavior_server'):
        assert _remappings(nodes[name], context) == {'cmd_vel': 'cmd_vel_nav'}
    assert _remappings(nodes['velocity_smoother'], context) == {
        'cmd_vel': 'cmd_vel_nav', 'cmd_vel_smoothed': 'cmd_vel'}
    for name in NAV2 - {'controller_server', 'behavior_server', 'velocity_smoother'}:
        assert 'cmd_vel' not in _remappings(nodes[name], context).values()


@pytest.mark.parametrize('odom_source', ['fastlio', 'sim_gt'])
def test_slam_toolbox_does_not_localise(odom_source):
    '''The odometry localises; slam_toolbox only draws the map at its poses, with no scan
    matching and no loop closing, and its base frame on the floor (README "slam_toolbox")'''
    nodes, context = _resolve(odom_source=odom_source)
    slam = nodes['slam_toolbox']
    params_file = slam._Node__parameters[0]  # a launch_ros ParameterFile
    path = perform_substitutions(context, params_file.param_file)
    assert path == os.path.join(SHARE, 'config', 'g1_slam_toolbox.yaml')
    with open(path) as f:
        params = yaml.safe_load(f)['slam_toolbox']['ros__parameters']
    params.update(_parameters(slam, context))  # what the node gets
    assert params['use_scan_matching'] is False
    assert params['do_loop_closing'] is False
    # level and on the floor in odom (level_frame's default floor_frame): map -> odom is the
    # identity
    assert params['base_frame'] == 'base_footprint'
    assert 'floor_frame' not in _parameters(nodes['level_frame'], context)


@pytest.mark.parametrize('odom_source', ['fastlio', 'sim_gt'])
def test_frontier_explorer_only_on_request(odom_source):
    nodes, _ = _resolve(odom_source=odom_source)
    assert 'frontier_explore' not in nodes
    nodes, context = _resolve(odom_source=odom_source, frontier_explore='true',
                              use_sim_time='true')
    assert set(nodes) == COMMON | (FASTLIO if odom_source == 'fastlio' else set()) | {
        'frontier_explore'}
    explorer = nodes['frontier_explore']
    assert (explorer.node_package, explorer.node_executable) == ('g1_navigation',
                                                                  'frontier_explore')
    params = _parameters(explorer, context)
    assert params['use_sim_time'] is True
    # the footprint of the Nav2 params in use
    with open(os.path.join(SHARE, 'config', 'g1_nav2.yaml')) as f:
        nav2 = yaml.safe_load(f)
    assert params['footprint'] == nav2['global_costmap']['global_costmap']['ros__parameters'][
        'footprint']


def test_unknown_odom_source_is_refused():
    with pytest.raises(RuntimeError, match='odom_source'):
        _resolve(odom_source='wheel')
