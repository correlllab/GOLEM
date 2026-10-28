"""
use_nav in the bringup.

g1_core includes g1_navigation's launch with the sim's arguments, refuses
navigation on the real robot (its lidar is not connected yet) and without the
walking node, and the real and sim launches pass use_nav on. frontier_explore
reaches g1_navigation as an explicit launch argument of that include. Needs the
workspace built and sourced (g1_bringup, g1_description, g1_navigation and
golem_upper_ik installed).
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

from test_locomotion_launch import _context, _core_nodes, _load, _passed_to_core, core

NAV_LAUNCH = os.path.join(get_package_share_directory('g1_navigation'), 'launch',
                          'g1_navigation.launch.py')
SIM_WALKING = dict(sim='true', use_sim_time='true', use_locomotion='true')
REAL_WALKING = dict(use_locomotion='true', safety_config='split', start_position_verified='true')


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
    assert _includes(**SIM_WALKING) == []
    assert _includes() == []
    for launch_file in ('g1_sim_bringup.launch.py', 'g1_bringup.launch.py'):
        passed = _passed_to_core(launch_file)
        assert passed['use_nav'] == 'false', launch_file
        assert _includes(**passed) == []
    assert _passed_to_core('g1_sim_bringup.launch.py')['odom_source'] == 'fastlio'


@pytest.mark.parametrize('odom_source', ['fastlio', 'sim_gt'])
def test_sim_bringup_includes_navigation(odom_source):
    args = dict(use_locomotion='true', use_nav='true')
    if odom_source != 'fastlio':
        args['odom_source'] = odom_source
    passed = _passed_to_core('g1_sim_bringup.launch.py', **args)
    assert (passed['use_nav'], passed['odom_source']) == ('true', odom_source)
    # the walking node still starts; navigation comes with the core, on sim time
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
    for launch_file in ('g1_sim_bringup.launch.py', 'g1_core.launch.py'):
        description = _load(launch_file).generate_launch_description()
        declared = {e.name: e for e in description.entities
                    if isinstance(e, DeclareLaunchArgument)}
        assert 'frontier_explore' in declared, f'{launch_file} does not declare it'
        default = perform_substitutions(LaunchContext(),
                                        declared['frontier_explore'].default_value)
        assert default == 'false', launch_file
    assert _passed_to_core('g1_sim_bringup.launch.py')['frontier_explore'] == 'false'

    # the sim bringup hands it to g1_core, and g1_core to g1_navigation, as a launch
    # argument of the include: not left to the launch context they share
    for value in ('true', 'false'):
        passed = _passed_to_core('g1_sim_bringup.launch.py', use_locomotion='true',
                                 use_nav='true', frontier_explore=value)
        assert passed['frontier_explore'] == value
        path, arguments = _nav_include(**passed)
        assert path == NAV_LAUNCH
        assert arguments['frontier_explore'] == value
        # and g1_navigation starts the explorer only when asked
        assert ('frontier_explore' in _nav_nodes(path, arguments)) == (value == 'true')


def test_core_passes_use_sim_time_on():
    # g1_core's use_sim_time reaches g1_navigation as it reaches the core's nodes
    _, arguments = _nav_include(**dict(SIM_WALKING, use_nav='true', use_sim_time='false'))
    assert arguments['use_sim_time'] == 'false'


@pytest.mark.parametrize('args, message', [
    # navigation drives the walking node
    (dict(sim='true', use_nav='true'),
     "use_nav:=true needs use_locomotion:=true: Nav2 drives the walking node's /cmd_vel"),
    (dict(SIM_WALKING, use_nav='yes', odom_source='wheel'),
     "odom_source must be one of fastlio, sim_gt, not 'wheel'"),
    # the real robot refuses it, whatever else is set: its lidar is a deferred TODO
    (dict(use_nav='true'), 'use_nav:=true is sim only for now: the real MID-360 is not connected'),
    (dict(REAL_WALKING, use_nav='true'), 'use_nav:=true is sim only for now'),
    (dict(REAL_WALKING, use_nav='true', odom_source='sim_gt'),
     'use_nav:=true is sim only for now'),
    (dict(use_locomotion='true', use_nav='true'), 'use_nav:=true is sim only for now'),
])
def test_core_refuses_nav(args, message):
    with pytest.raises(RuntimeError, match='g1_core: not starting: ' + re.escape(message)):
        core._setup(_context(core.generate_launch_description(), **args))


def test_real_bringup_refuses_nav():
    for args in (dict(use_nav='true'), dict(REAL_WALKING, use_nav='true')):
        passed = _passed_to_core('g1_bringup.launch.py', **args)
        assert (passed['sim'], passed['use_nav']) == ('false', 'true')
        with pytest.raises(RuntimeError, match='use_nav:=true is sim only for now'):
            _core_nodes(**passed)
    # and still walks without it
    assert 'locomotion_node' in _core_nodes(**_passed_to_core('g1_bringup.launch.py',
                                                              **REAL_WALKING))


def test_sim_bringup_refuses_nav_without_walking():
    passed = _passed_to_core('g1_sim_bringup.launch.py', use_nav='true')
    with pytest.raises(RuntimeError, match='needs use_locomotion:=true'):
        _core_nodes(**passed)
