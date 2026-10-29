"""
use_nav in the bringup.

g1_core includes g1_navigation's launch with the sim's arguments, beside the
walking node the sim always starts, and refuses navigation on the real robot
(its lidar is not connected yet); the real and sim launches pass use_nav on.
frontier_explore reaches g1_navigation as an explicit launch argument of that
include. Needs the workspace built and sourced (golem_bringup, golem_ros2_models,
g1_navigation and golem_upper_ik installed); skipped without g1_navigation, which an
H1-2 workspace does not build.
"""

import os
import pathlib
import re

import pytest
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

# the navigation launch is g1_navigation's
pytest.importorskip('g1_navigation')

from test_g1_launch import (  # noqa: E402
    _context, _core_nodes, _load, _passed_to_core, core)

NAV_LAUNCH = os.path.join(get_package_share_directory('g1_navigation'), 'launch',
                          'g1_navigation.launch.py')
SIM = dict(sim='true', use_sim_time='true')
REAL_VERIFIED = dict(safety_config='split', start_position_verified='true')


def _includes(**args):
    """Return the IncludeLaunchDescription actions g1_core starts with args."""
    actions = core._setup(_context(core.generate_launch_description(), **args))
    return [a for a in actions if isinstance(a, IncludeLaunchDescription)]


def _nav_include(**args):
    """Return (path, {argument: value}) of the one launch file g1_core includes with args."""
    (include,) = _includes(**args)
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
        assert isinstance(action, Node)
        values, files = {}, []
        for params in evaluate_parameters(context, action._Node__parameters):
            if isinstance(params, dict):
                values.update(params)
            elif isinstance(params, pathlib.Path):
                files.append(str(params))
        nodes[action._Node__node_name] = (action, values, files)
    return nodes


def test_nav_is_off_by_default():
    # the core, the sim bringup and the real bringup start no navigation unless asked
    assert _includes(**SIM) == []
    assert _includes() == []
    for launch_file in ('g1_sim.launch.py', 'g1_real_robot.launch.py'):
        passed = _passed_to_core(launch_file)
        assert passed['use_nav'] == 'false', launch_file
        assert _includes(**passed) == []
    assert _passed_to_core('g1_sim.launch.py')['odom_source'] == 'fastlio'


@pytest.mark.parametrize('odom_source', ['fastlio', 'sim_gt'])
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
                         'frontier_explore': 'false'}

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
        # pointcloud_to_laserscan runs on the wall clock (g1_navigation.launch.py)
        assert values.get('use_sim_time') is (name != 'pointcloud_to_laserscan'), name


def test_frontier_explore_is_declared_and_forwarded():
    # an argument of the sim bringup and of g1_core, off by default
    for launch_file in ('g1_sim.launch.py', 'g1_core.launch.py'):
        description = _load(launch_file).generate_launch_description()
        declared = {e.name: e for e in description.entities
                    if isinstance(e, DeclareLaunchArgument)}
        assert 'frontier_explore' in declared, f'{launch_file} does not declare it'
        default = perform_substitutions(LaunchContext(),
                                        declared['frontier_explore'].default_value)
        assert default == 'false', launch_file
    assert _passed_to_core('g1_sim.launch.py')['frontier_explore'] == 'false'

    # the sim bringup hands it to g1_core, and g1_core to g1_navigation, as a launch
    # argument of the include: not left to the launch context they share
    for value in ('true', 'false'):
        passed = _passed_to_core('g1_sim.launch.py', use_nav='true', frontier_explore=value)
        assert passed['frontier_explore'] == value
        path, arguments = _nav_include(**passed)
        assert path == NAV_LAUNCH
        assert arguments['frontier_explore'] == value
        # and g1_navigation starts the explorer only when asked
        assert ('frontier_explore' in _nav_nodes(path, arguments)) == (value == 'true')


def test_core_passes_use_sim_time_on():
    # g1_core's use_sim_time reaches g1_navigation as it reaches the core's nodes
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
    (dict(SIM, use_nav='yes', odom_source='wheel'),
     "odom_source must be one of fastlio, sim_gt, not 'wheel'"),
    # the real robot refuses it, whatever else is set: its lidar is a deferred TODO
    (dict(use_nav='true'), 'use_nav:=true is sim only for now: the real MID-360 is not connected'),
    (dict(REAL_VERIFIED, use_nav='true'), 'use_nav:=true is sim only for now'),
    (dict(REAL_VERIFIED, use_nav='true', odom_source='sim_gt'),
     'use_nav:=true is sim only for now'),
    (dict(use_nav='true', odom_source='wheel'), 'use_nav:=true is sim only for now'),
])
def test_core_refuses_nav(args, message):
    with pytest.raises(RuntimeError, match='g1_core: not starting: ' + re.escape(message)):
        core._setup(_context(core.generate_launch_description(), **args))


def test_real_refuses_nav():
    for args in (dict(use_nav='true'), dict(REAL_VERIFIED, use_nav='true')):
        passed = _passed_to_core('g1_real_robot.launch.py', **args)
        assert (passed['sim'], passed['use_nav']) == ('false', 'true')
        with pytest.raises(RuntimeError, match='use_nav:=true is sim only for now'):
            _core_nodes(**passed)
    # without it the legs start once verified
    assert 'locomotion_node' in _core_nodes(**_passed_to_core('g1_real_robot.launch.py',
                                                              **REAL_VERIFIED))
