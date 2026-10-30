"""
use_nav and use_skills in the G1 bringup.

g1_core includes golem_slam's G1 launch, on by default, with the sim's or the
real robot's arguments, beside the walking node the sim always starts; on the
real robot also the MID-360 driver. The real and sim launches pass use_nav on.
use_skills starts golem_skills on the G1 bringup. Needs the workspace built and
sourced (golem_bringup, golem_ros2_models, golem_slam and golem_upper_ik installed).
"""

import os
import pathlib
import re

import pytest
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

pytest.importorskip('golem_slam')

from test_g1_launch import (  # noqa: E402
    _context, _core_nodes, _load, _passed_to_core, core)

NAV_LAUNCH = os.path.join(get_package_share_directory('golem_slam'), 'launch',
                          'g1_navigation.launch.py')
SIM = dict(sim='true', use_sim_time='true')
REAL_VERIFIED = dict(safety_config='split', start_position_verified='true')


def _includes(**args):
    """Return the IncludeLaunchDescription actions g1_core starts with args."""
    actions = core._setup(_context(core.generate_launch_description(), **args))
    return [a for a in actions if isinstance(a, IncludeLaunchDescription)]


def _location(include):
    include.launch_description_source.get_launch_description(LaunchContext())
    return include.launch_description_source.location


def _nav_include(**args):
    """Return (path, {argument: value}) of golem_slam's launch file g1_core includes with args."""
    (include,) = [i for i in _includes(**args) if _location(i) == NAV_LAUNCH]
    context = LaunchContext()
    include.launch_description_source.get_launch_description(context)   # resolves its location
    path = include.launch_description_source.location

    def text(value):
        return perform_substitutions(context, normalize_to_list_of_substitutions(value))
    return path, {text(key): text(value) for key, value in include.launch_arguments}


def _nav_nodes(path, arguments):
    """
    Return the nodes the included launch file starts with arguments.

    The result is {node name: (Node, {parameter: value} of its dict parameters, its
    parameter files)}. The launch file must declare every argument.
    """
    description = PythonLaunchDescriptionSource(path).get_launch_description(LaunchContext())
    declared = {e.name for e in description.entities if isinstance(e, DeclareLaunchArgument)}
    assert set(arguments) <= declared, f'not declared by {path}: {set(arguments) - declared}'
    context = _context(description, **arguments)
    (setup,) = [e for e in description.entities if isinstance(e, OpaqueFunction)]
    nodes = {}
    for action in setup.execute(context):
        if isinstance(action, (IncludeLaunchDescription, GroupAction)):
            continue  # the shared pipeline includes Nav2 navigation_launch.py
        assert isinstance(action, Node)
        values, files = {}, []
        for params in evaluate_parameters(context, action._Node__parameters):
            if isinstance(params, dict):
                values.update(params)
            elif isinstance(params, pathlib.Path):
                files.append(str(params))
        nodes[action._Node__node_name] = (action, values, files)
    return nodes


def test_nav_is_on_by_default():
    # the core, the sim bringup and the real bringup start navigation unless told not to
    assert [_location(i) for i in _includes(**SIM)] == [NAV_LAUNCH]
    for launch_file in ('g1_sim.launch.py', 'g1_real_robot.launch.py'):
        passed = _passed_to_core(launch_file)
        assert passed['use_nav'] == 'true', launch_file
        assert _includes(**dict(passed, use_nav='false')) == []
    assert _passed_to_core('g1_sim.launch.py')['odom_source'] == 'fastlio'


def test_real_bringup_starts_the_lidar_driver_and_the_real_fastlio_config():
    passed = _passed_to_core('g1_real_robot.launch.py', **REAL_VERIFIED)
    includes = _includes(**passed)
    drivers = os.path.join(get_package_share_directory('golem_bringup'), 'launch', 'g1',
                           'g1_real_drivers.launch.py')
    assert sorted(_location(i) for i in includes) == sorted([drivers, NAV_LAUNCH])
    path, arguments = _nav_include(**passed)
    assert arguments == {'odom_source': 'fastlio', 'sim': 'false', 'use_sim_time': 'false',
                         'fast_lio_config_file': 'g1_fastlio.yaml'}
    nodes = _nav_nodes(path, arguments)
    _, values, files = nodes['fast_lio']
    assert files == [os.path.join(os.path.dirname(os.path.dirname(NAV_LAUNCH)), 'config',
                                  'g1_fastlio.yaml')]
    assert values.get('use_sim_time') is False
    # the driver: CustomMsg at FAST_LIO's 10 Hz, on the G1's config
    description = PythonLaunchDescriptionSource(drivers).get_launch_description(LaunchContext())
    (driver,) = [e for e in description.entities if isinstance(e, Node)]
    context = _context(description)
    (params,) = evaluate_parameters(context, driver._Node__parameters)
    assert (params['xfer_format'], params['publish_freq'], params['frame_id']) == (
        1, 10.0, 'livox_frame')
    assert params['user_config_path'].endswith('golem_slam/config/g1_MID360_config.json')
    # the sim publishes its own lidar
    sim = _passed_to_core('g1_sim.launch.py')
    assert [_location(i) for i in _includes(**sim)] == [NAV_LAUNCH]


@pytest.mark.parametrize('odom_source', ['fastlio'])
def test_sim_bringup_includes_navigation(odom_source):
    args = dict(use_nav='true')
    if odom_source != 'fastlio':
        args['odom_source'] = odom_source
    passed = _passed_to_core('g1_sim.launch.py', **args)
    assert (passed['use_nav'], passed['odom_source']) == ('true', odom_source)
    # the walking node starts; navigation comes with the core, on sim time
    assert 'locomotion_node' in _core_nodes(**passed)
    path, arguments = _nav_include(**passed)
    assert path == NAV_LAUNCH
    assert arguments == {'odom_source': odom_source, 'sim': 'true', 'use_sim_time': 'true',
                         'fast_lio_config_file': 'g1_fastlio_sim.yaml'}

    nodes = _nav_nodes(path, arguments)
    assert ('fast_lio' in nodes) == (odom_source == 'fastlio')
    if odom_source == 'fastlio':
        # the sim's FAST_LIO config: the raw sensor frame, every point kept
        _, _, files = nodes['fast_lio']
        assert files == [os.path.join(os.path.dirname(os.path.dirname(NAV_LAUNCH)), 'config',
                                      'g1_fastlio_sim.yaml')]
    for name, (_, values, _) in nodes.items():
        if name.endswith('_tf'):
            continue   # static_transform_publisher: arguments only
        # The shared pipeline consistently uses simulation time.
        assert values.get('use_sim_time') is True, name


def test_use_skills_is_declared_and_starts_golem_skills():
    # an argument of the sim bringup and of g1_core, off by default
    for launch_file in ('g1_sim.launch.py', 'g1_core.launch.py'):
        description = _load(launch_file).generate_launch_description()
        declared = {e.name: e for e in description.entities
                    if isinstance(e, DeclareLaunchArgument)}
        assert 'use_skills' in declared, f'{launch_file} does not declare it'
        default = perform_substitutions(LaunchContext(),
                                        declared['use_skills'].default_value)
        assert default == 'false', launch_file
    assert _passed_to_core('g1_sim.launch.py')['use_skills'] == 'false'

    # the sim bringup hands it to g1_core, which starts golem_skills only when asked
    for value in ('true', 'false'):
        passed = _passed_to_core('g1_sim.launch.py', use_nav='true', use_skills=value)
        assert passed['use_skills'] == value
        assert ('skills' in _core_nodes(**passed)) == (value == 'true')


def test_core_passes_use_sim_time_on():
    # g1_core's use_sim_time reaches golem_slam as it reaches the core's nodes
    _, arguments = _nav_include(**dict(SIM, use_nav='true', use_sim_time='false'))
    assert arguments['use_sim_time'] == 'false'


def test_sim_nav_without_use_locomotion():
    # the sim always starts the walking node, which Nav2 drives through /cmd_vel
    passed = _passed_to_core('g1_sim.launch.py', use_nav='true')
    assert 'use_locomotion' not in passed
    nodes = _core_nodes(**passed)
    assert nodes['locomotion_node'].arguments == ['--config', 'sim.yaml']
    path, arguments = _nav_include(**passed)
    assert path == NAV_LAUNCH
    assert arguments['sim'] == 'true'


@pytest.mark.parametrize('args, message', [
    (dict(SIM, use_nav='true', odom_source='sim_gt'), 'golem_slam requires FAST-LIO'),
    (dict(SIM, use_nav='yes', odom_source='wheel'),
     "golem_slam requires FAST-LIO"),
    (dict(REAL_VERIFIED, use_nav='true', odom_source='sim_gt'), 'golem_slam requires FAST-LIO'),
])
def test_core_refuses_nav(args, message):
    with pytest.raises(RuntimeError, match='g1_core: not starting: ' + re.escape(message)):
        core._setup(_context(core.generate_launch_description(), **args))


def test_real_nav_with_the_legs():
    passed = _passed_to_core('g1_real_robot.launch.py', **REAL_VERIFIED)
    assert (passed['sim'], passed['use_nav']) == ('false', 'true')
    assert 'locomotion_node' in _core_nodes(**passed)
