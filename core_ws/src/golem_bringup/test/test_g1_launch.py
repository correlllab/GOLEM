"""
The G1's bringups: HOMIE's walking node, and the configs they refuse.

g1_core always starts the walking node (locomotion_node) in the sim, and on the
real robot only with start_position_verified:=true, as the H1-2's leg node is
started. It runs the split-mode configs and refuses sim/real mixes, and the
real and sim launches pass their settings on. Needs the workspace built and
sourced (golem_bringup, g1_description and golem_upper_ik installed); skipped
without g1_description, which an H1-2 workspace does not build.
"""

import importlib.util
from pathlib import Path
import re
from typing import NamedTuple

import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

# g1_core.launch.py imports g1_description
pytest.importorskip('g1_description')

LAUNCH_DIR = Path(__file__).resolve().parents[1] / 'launch' / 'g1'


def _load(name):
    spec = importlib.util.spec_from_file_location(name.split('.')[0], LAUNCH_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = _load('g1_core.launch.py')


def _context(description, **args):
    """Return a LaunchContext with the description's declared defaults, overridden by args."""
    context = LaunchContext()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            context.launch_configurations[entity.name] = perform_substitutions(
                context, entity.default_value)
    context.launch_configurations.update(args)
    return context


class Started(NamedTuple):
    """A node g1_core starts: its command-line arguments, start delay and ROS parameters."""

    arguments: list
    delay: float
    parameters: dict


def _core_nodes(**args):
    """Return {executable: Started} of the nodes g1_core starts with args."""
    nodes = {}

    def add(action, delay):
        if isinstance(action, TimerAction):
            for inner in action.actions:
                add(inner, delay + float(action.period))
        elif isinstance(action, Node):
            context = LaunchContext()
            arguments = []
            for part in action.cmd[1:]:
                text = perform_substitutions(context, part)
                if text == '--ros-args':
                    break
                arguments.append(text)
            parameters = {}
            for params in evaluate_parameters(context, action._Node__parameters):
                parameters.update(params)
            nodes[action.node_executable] = Started(arguments, delay, parameters)

    for action in core._setup(_context(core.generate_launch_description(), **args)):
        add(action, 0.0)
    return nodes


def _passed_to_core(launch_file, **args):
    """Return the launch arguments launch_file passes to g1_core, given its own args."""
    description = _load(launch_file).generate_launch_description()
    context = _context(description, **args)
    (include,) = [e for e in description.entities if isinstance(e, IncludeLaunchDescription)]

    def text(value):
        return perform_substitutions(context, normalize_to_list_of_substitutions(value))
    return {text(key): text(value) for key, value in include.launch_arguments}


def _configs(nodes):
    return {name: nodes[name].parameters['config']
            for name in ('safety_node', 'frame_task_server') if name in nodes}


def _declared(launch_file):
    description = _load(launch_file).generate_launch_description()
    return {e.name for e in description.entities if isinstance(e, DeclareLaunchArgument)}


def _refused(message, **args):
    """Assert that g1_core refuses to start with args, with message."""
    with pytest.raises(RuntimeError, match='g1_core: not starting: ' + re.escape(message)):
        core._setup(_context(core.generate_launch_description(), **args))


def test_use_locomotion_not_declared():
    for launch_file in ('g1_sim.launch.py', 'g1_real_robot.launch.py', 'g1_core.launch.py'):
        assert 'use_locomotion' not in _declared(launch_file), launch_file
    for launch_file in ('g1_sim.launch.py', 'g1_real_robot.launch.py'):
        assert 'use_locomotion' not in _passed_to_core(launch_file), launch_file


def test_sim_always_starts_locomotion():
    # the sim needs no interlock: start_position_verified changes nothing there
    for args in (dict(), dict(start_position_verified='false'),
                 dict(start_position_verified='true')):
        nodes = _core_nodes(sim='true', **args)
        assert _configs(nodes) == {'safety_node': 'sim_split', 'frame_task_server': 'sim_split'}
        assert nodes['locomotion_node'].arguments == ['--config', 'sim.yaml']

    passed = _passed_to_core('g1_sim.launch.py')
    assert passed['sim'] == 'true'
    nodes = _core_nodes(**passed)
    assert _configs(nodes) == {'safety_node': 'sim_split', 'frame_task_server': 'sim_split'}
    assert nodes['locomotion_node'].arguments == ['--config', 'sim.yaml']


def test_real_starts_locomotion_only_when_verified():
    # without the operator's word the legs stay on golem_safety's zero-gain stream,
    # and the arms and the safety layer start
    for args in (dict(), dict(start_position_verified='false'),
                 dict(start_position_verified='no')):
        nodes = _core_nodes(**args)
        assert 'locomotion_node' not in nodes
        assert {'robot_state_publisher', 'joint_state_publisher'} <= set(nodes)
        assert _configs(nodes) == {'safety_node': 'split', 'frame_task_server': 'g1_split'}
    nodes = _core_nodes(start_position_verified='true')
    assert _configs(nodes) == {'safety_node': 'split', 'frame_task_server': 'g1_split'}
    assert nodes['locomotion_node'].arguments == ['--config', 'g1.yaml']

    # the real bringup passes the interlock on, false by default
    passed = _passed_to_core('g1_real_robot.launch.py')
    assert (passed['sim'], passed['start_position_verified']) == ('false', 'false')
    nodes = _core_nodes(**passed)
    assert 'locomotion_node' not in nodes
    assert _configs(nodes) == {'safety_node': 'split', 'frame_task_server': 'g1_split'}
    passed = _passed_to_core('g1_real_robot.launch.py', start_position_verified='true')
    assert passed['start_position_verified'] == 'true'
    assert _core_nodes(**passed)['locomotion_node'].arguments == ['--config', 'g1.yaml']


def test_locomotion_starts_after_the_controller():
    nodes = _core_nodes(start_position_verified='true')
    assert nodes['frame_task_server'].delay == 2.0
    assert nodes['locomotion_node'].delay == nodes['frame_task_server'].delay + 2.0
    nodes = _core_nodes(sim='true', controller_delay='1.0', locomotion_delay='1.5')
    assert nodes['locomotion_node'].delay == 2.5


def test_controller_config_follows_the_safety_config():
    for args, safety, controller in [
        (dict(), 'split', 'g1_split'),
        (dict(safety_config='split'), 'split', 'g1_split'),
        (dict(sim='true'), 'sim_split', 'sim_split'),
        (dict(sim='true', safety_config='sim_split'), 'sim_split', 'sim_split'),
    ]:
        assert _configs(_core_nodes(**args)) == {'safety_node': safety,
                                                 'frame_task_server': controller}


def test_custom_controller_config_is_not_refused():
    # a controller config outside the table is not matched to the safety config
    # (g1_core's "Not checked"), and the walking node starts beside it
    nodes = _core_nodes(controller_config='custom', start_position_verified='true')
    assert _configs(nodes) == {'safety_node': 'split', 'frame_task_server': 'custom'}
    assert 'locomotion_node' in nodes
    nodes = _core_nodes(sim='true', controller_config='sim_custom')
    assert _configs(nodes) == {'safety_node': 'sim_split', 'frame_task_server': 'sim_custom'}
    assert 'locomotion_node' in nodes


@pytest.mark.parametrize('args, message', [
    (dict(safety_config='sim_split'), 'safety config "sim_split" is a sim config'),
    (dict(controller_config='sim_split'), 'controller config "sim_split" is a sim config'),
    (dict(controller_config='sim_fast'), 'controller config "sim_fast" is a sim config'),
    # the walking node's sim config too, whether or not the legs are verified
    (dict(locomotion_config='sim'), 'locomotion_config:=sim is for the sim; use g1'),
    (dict(locomotion_config='sim', start_position_verified='true'),
     'locomotion_config:=sim is for the sim; use g1'),
])
def test_real_refuses_sim_configs(args, message):
    _refused(message, **args)
    # through the real bringup, for the arguments it declares
    declared = _declared('g1_real_robot.launch.py')
    if set(args) <= declared:
        passed = _passed_to_core('g1_real_robot.launch.py', **args)
        assert passed['sim'] == 'false'
        with pytest.raises(RuntimeError, match=re.escape(message)):
            _core_nodes(**passed)


@pytest.mark.parametrize('args, message', [
    (dict(safety_config='split'), 'safety config "split" is for the real robot'),
    (dict(controller_config='g1_split'), 'controller config "g1_split" is for the real robot'),
    (dict(locomotion_config='g1'), 'locomotion_config:=g1 is for the real robot; use sim'),
])
def test_sim_refuses_real_configs(args, message):
    _refused(message, sim='true', **args)
    declared = _declared('g1_sim.launch.py')
    if set(args) <= declared:
        passed = _passed_to_core('g1_sim.launch.py', **args)
        assert passed['sim'] == 'true'
        with pytest.raises(RuntimeError, match=re.escape(message)):
            _core_nodes(**passed)


def test_core_refuses_a_safety_config_without_the_upper_body_channel():
    for sim in ('false', 'true'):
        _refused('safety config "full" must relay the upper-body channel', sim=sim,
                 safety_config='full', start_position_verified='true')
