"""
Test use_nav and use_skills in the G1 bringup, and the H1-2 bringups' navigation includes.

g1_core includes golem_slam's navigation.launch.py, on by default, with robot g1
and the sim's or the real robot's arguments, beside the leg node the sim
always starts; on the real robot also the MID-360 driver. The real and sim
launches pass use_nav on. use_skills starts golem_skills on the G1 bringup. The
H1-2's includers pass robot h12. Needs the workspace built and sourced
(golem_bringup, golem_ros2_models, golem_slam and golem_upper_ik installed).
"""

import importlib.util
import os
import pathlib

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
    _context, _core, _core_nodes, _load, _passed_to_core)

SLAM_SHARE = get_package_share_directory('golem_slam')
NAV_LAUNCH = os.path.join(SLAM_SHARE, 'launch', 'navigation.launch.py')
G1_CONFIG = os.path.join(SLAM_SHARE, 'config', 'g1')
SIM = dict(sim='true', use_sim_time='true')
REAL_VERIFIED = dict(safety_config='split', start_position_verified='true')


def _includes(**args):
    """
    Return the IncludeLaunchDescription actions g1_core starts with args.

    Not the golem_safety and golem_upper_ik launch files, which start the core's own nodes.
    """
    return _core(**args)[1]


def _location(include):
    """Return the path of the launch file include includes."""
    include.launch_description_source.get_launch_description(LaunchContext())
    return include.launch_description_source.location


def _nav_include(**args):
    """Return (path, {argument: value}) of golem_slam's launch file g1_core includes with args."""
    (include,) = [i for i in _includes(**args) if _location(i) == NAV_LAUNCH]
    context = LaunchContext()
    include.launch_description_source.get_launch_description(context)  # resolves its location
    path = include.launch_description_source.location

    def text(value):
        """Return a substitution's text in context."""
        return perform_substitutions(context, normalize_to_list_of_substitutions(value))
    return path, {text(key): text(value) for key, value in include.launch_arguments}


def _nav_nodes(path, arguments):
    """
    Return the nodes the included launch file starts with arguments.

    The result is {node name: (Node, {parameter: value} of its dict parameters, its
    parameter files)}. The launch file must declare every argument.
    """
    description = PythonLaunchDescriptionSource(path).get_launch_description(LaunchContext())
    declared = {e.name: e for e in description.entities if isinstance(e, DeclareLaunchArgument)}
    assert set(arguments) <= set(declared), \
        f'not declared by {path}: {set(arguments) - set(declared)}'
    # The arguments first: navigation.launch.py's file defaults name the robot.
    context = LaunchContext()
    context.launch_configurations.update(arguments)
    for name, declaration in declared.items():
        if name not in arguments:
            context.launch_configurations[name] = perform_substitutions(
                context, declaration.default_value)
    (setup,) = [e for e in description.entities if isinstance(e, OpaqueFunction)]
    nodes = {}
    for action in setup.execute(context):
        if isinstance(action, (IncludeLaunchDescription, GroupAction)):
            continue  # Nav2's navigation_launch.py
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
    """The core, the sim bringup and the real bringup start navigation unless told not to."""
    assert [_location(i) for i in _includes(**SIM)] == [NAV_LAUNCH]
    for launch_file in ('g1_sim.launch.py', 'g1_real_robot.launch.py'):
        passed = _passed_to_core(launch_file)
        assert passed['use_nav'] == 'true', launch_file
        assert _includes(**dict(passed, use_nav='false')) == []


def test_real_bringup_starts_the_lidar_driver_and_the_real_fastlio_config():
    """The real G1 starts the MID-360 driver and golem_slam on the real FAST-LIO config."""
    passed = _passed_to_core('g1_real_robot.launch.py', **REAL_VERIFIED)
    includes = _includes(**passed)
    drivers = os.path.join(get_package_share_directory('golem_bringup'), 'launch', 'g1',
                           'g1_real_drivers.launch.py')
    assert sorted(_location(i) for i in includes) == sorted([drivers, NAV_LAUNCH])
    path, arguments = _nav_include(**passed)
    assert arguments == {'robot': 'g1', 'use_sim_time': 'false'}
    nodes = _nav_nodes(path, arguments)
    _, values, files = nodes['fast_lio']
    assert files == [os.path.join(G1_CONFIG, 'fastlio.yaml')]
    assert values.get('use_sim_time') is False
    # The driver: CustomMsg at FAST-LIO's 10 Hz, on the G1's config.
    description = PythonLaunchDescriptionSource(drivers).get_launch_description(LaunchContext())
    (driver,) = [e for e in description.entities if isinstance(e, Node)]
    context = _context(description)
    (params,) = evaluate_parameters(context, driver._Node__parameters)
    assert (params['xfer_format'], params['publish_freq'], params['frame_id']) == (
        1, 10.0, 'livox_frame')
    assert params['user_config_path'] == os.path.join(G1_CONFIG, 'MID360_config.json')
    # The sim publishes its own lidar.
    sim = _passed_to_core('g1_sim.launch.py')
    assert [_location(i) for i in _includes(**sim)] == [NAV_LAUNCH]


def test_sim_bringup_includes_navigation():
    """The G1 sim includes golem_slam on sim time and the sim's FAST-LIO config."""
    passed = _passed_to_core('g1_sim.launch.py', use_nav='true')
    assert passed['use_nav'] == 'true'
    # The leg node starts; navigation comes with the core, on sim time.
    assert 'lowerbody_controller_node' in _core_nodes(**passed)
    path, arguments = _nav_include(**passed)
    assert path == NAV_LAUNCH
    assert arguments == {'robot': 'g1', 'use_sim_time': 'true',
                         'fast_lio_config': os.path.join(G1_CONFIG, 'fastlio_sim.yaml')}
    assert arguments['fast_lio_config'].endswith('config/g1/fastlio_sim.yaml')

    nodes = _nav_nodes(path, arguments)
    # The sim's FAST-LIO config.
    _, _, files = nodes['fast_lio']
    assert files == [os.path.join(G1_CONFIG, 'fastlio_sim.yaml')]
    for name, (_, values, _) in nodes.items():
        if name.endswith('_tf'):
            continue  # static_transform_publisher: arguments only
        assert values.get('use_sim_time') is True, name


def test_use_skills_is_declared_and_starts_golem_skills():
    """use_skills, off by default in g1_sim and g1_core, starts golem_skills when on."""
    for launch_file in ('g1_sim.launch.py', 'g1_core.launch.py'):
        description = _load(launch_file).generate_launch_description()
        declared = {e.name: e for e in description.entities
                    if isinstance(e, DeclareLaunchArgument)}
        assert 'use_skills' in declared, f'{launch_file} does not declare it'
        default = perform_substitutions(LaunchContext(),
                                        declared['use_skills'].default_value)
        assert default == 'false', launch_file
    assert _passed_to_core('g1_sim.launch.py')['use_skills'] == 'false'

    # The sim bringup hands it to g1_core, which starts golem_skills only when asked.
    for value in ('true', 'false'):
        passed = _passed_to_core('g1_sim.launch.py', use_nav='true', use_skills=value)
        assert passed['use_skills'] == value
        assert ('skills' in _core_nodes(**passed)) == (value == 'true')


def test_core_passes_use_sim_time_on():
    """g1_core's use_sim_time reaches golem_slam as it reaches the core's nodes."""
    _, arguments = _nav_include(**dict(SIM, use_nav='true', use_sim_time='false'))
    assert arguments['use_sim_time'] == 'false'


def test_sim_nav_without_use_locomotion():
    """The sim always starts the leg node, which Nav2 drives through /cmd_vel."""
    passed = _passed_to_core('g1_sim.launch.py', use_nav='true')
    assert 'use_locomotion' not in passed
    nodes = _core_nodes(**passed)
    assert nodes['lowerbody_controller_node'].parameters['config'] == 'sim'
    path, arguments = _nav_include(**passed)
    assert path == NAV_LAUNCH
    assert arguments['fast_lio_config'] == os.path.join(G1_CONFIG, 'fastlio_sim.yaml')


def test_real_nav_with_the_legs():
    """The real G1 with verified legs starts both navigation and the leg node."""
    passed = _passed_to_core('g1_real_robot.launch.py', **REAL_VERIFIED)
    assert (passed['sim'], passed['use_nav']) == ('false', 'true')
    assert 'lowerbody_controller_node' in _core_nodes(**passed)


def _launch_module(path):
    """Import a launch file at path as a module."""
    spec = importlib.util.spec_from_file_location(os.path.basename(path).split('.')[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('package, launch_file', [
    ('golem_bringup', 'h12/h12_sim.launch.py'),
    ('golem_bringup', 'h12/h12_real_robot.launch.py'),
    ('golem_bringup', 'h12/lmg_sim.launch.py'),
    ('golem_bringup', 'h12/lmg_real.launch.py'),
    ('golem_bringup', 'h12/h12_real_slam.launch.py'),
])
def test_every_h12_includer_passes_robot_h12(package, launch_file, monkeypatch):
    """Every H1-2 file that includes navigation passes robot h12 and its clock."""
    monkeypatch.setenv('ROS_DOMAIN_ID', '87')  # lmg_sim refuses an unset or 0 domain
    path = os.path.join(get_package_share_directory(package), 'launch', launch_file)
    description = _launch_module(path).generate_launch_description()
    context = _context(description)
    includes = [e for e in description.entities if isinstance(e, IncludeLaunchDescription)
                and _location(e) == NAV_LAUNCH]
    assert len(includes) == 1, launch_file

    def text(value):
        """Return a substitution's text in context."""
        return perform_substitutions(context, normalize_to_list_of_substitutions(value))
    arguments = {text(key): text(value) for key, value in includes[0].launch_arguments}
    assert arguments['robot'] == 'h12'
    assert arguments['use_sim_time'] == ('true' if 'sim' in launch_file else 'false')
