"""
The G1's bringups: HOMIE's walking node, and the configs they refuse.

g1_core always starts the walking node (locomotion_node) in the sim, and on the
real robot only with start_position_verified:=true, as the H1-2's leg node is
started. It runs the split-mode configs and refuses sim/real mixes, and the
real and sim launches pass their settings on. g1_core starts golem_safety and
golem_upper_ik through their launch files, so the nodes are read by walking
those includes, as ros2 launch runs them. Needs the workspace built and
sourced (golem_bringup, golem_ros2_models, golem_safety and golem_upper_ik
installed).
"""

import importlib.util
import os
from pathlib import Path
import re
from typing import NamedTuple

from ament_index_python.packages import get_package_share_directory
import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

LAUNCH_DIR = Path(__file__).resolve().parents[1] / 'launch' / 'g1'


def _package_launch(package, launch_file):
    return os.path.join(get_package_share_directory(package), 'launch', launch_file)


SAFETY_LAUNCH = _package_launch('golem_safety', 'safety.launch.py')
UPPER_IK_LAUNCH = _package_launch('golem_upper_ik', 'upper_ik.launch.py')
ROBOT_STATE_LAUNCH = _package_launch('golem_upper_ik', 'robot_state.launch.py')
# the shared packages' launch files, which the bringups include for their nodes
PACKAGE_LAUNCHES = {SAFETY_LAUNCH, UPPER_IK_LAUNCH, ROBOT_STATE_LAUNCH}


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
    """A node a launch starts: its package, arguments, start delay and ROS parameters."""

    package: str
    arguments: list
    delay: float
    parameters: dict


def _location(include):
    """Return the path of the launch file include includes."""
    include.launch_description_source.get_launch_description(LaunchContext())
    return include.launch_description_source.location


def _started(entities, context, descend=PACKAGE_LAUNCHES):
    """
    Return ({executable: Started}, [include]) of what entities start in context.

    The entities run in order, as ros2 launch runs them: conditions apply, a
    refusal raises before any later entity, and an include sets its arguments in
    context (Humble includes are not scoped). Includes of the files in descend
    are walked into; the others are returned unwalked.
    """
    nodes, includes = {}, []

    def walk(entities, delay):
        for entity in entities:
            if getattr(entity, 'condition', None) is not None and not \
                    entity.condition.evaluate(context):
                continue
            if isinstance(entity, Node):
                arguments = []
                for part in entity.cmd[1:]:
                    text = perform_substitutions(context, part)
                    if text == '--ros-args':
                        break
                    arguments.append(text)
                parameters = {}
                for params in evaluate_parameters(context, entity._Node__parameters):
                    assert isinstance(params, dict), params
                    parameters.update(params)
                nodes[entity.node_executable] = Started(entity.node_package, arguments, delay,
                                                        parameters)
            elif isinstance(entity, TimerAction):
                walk(entity.actions, delay + float(entity.period))
            elif isinstance(entity, IncludeLaunchDescription) and \
                    _location(entity) not in descend:
                includes.append(entity)
            else:
                walk(entity.visit(context) or [], delay)

    walk(entities, 0.0)
    return nodes, includes


def _core(**args):
    """Return ({executable: Started}, [include]) of what g1_core starts with args."""
    context = _context(core.generate_launch_description(), **args)
    return _started(core._setup(context), context)


def _core_nodes(**args):
    """Return {executable: Started} of the nodes g1_core starts with args."""
    return _core(**args)[0]


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
    """
    Assert that g1_core refuses to start with args, with message.

    The pairing refusals are golem_upper_ik's (upper_ik: not starting: ...), the
    walking node's g1_core's own (g1_core: not starting: ...).
    """
    with pytest.raises(RuntimeError, match='^' + re.escape(message)):
        _core_nodes(**args)


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
    assert nodes['safety_node'].delay == 0.0
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


def test_custom_controller_config_is_refused():
    # the controller config must be the one paired with the safety config
    _refused('upper_ik: not starting: safety config "split" runs with controller config '
             '"g1_split", not "custom"',
             controller_config='custom', start_position_verified='true')
    _refused('upper_ik: not starting: safety config "sim_split" runs with controller '
             'config "sim_split", not "sim_custom"', sim='true', controller_config='sim_custom')


@pytest.mark.parametrize('args, message', [
    (dict(safety_config='sim_split'),
     'upper_ik: not starting: safety config "sim_split" is a sim config'),
    (dict(controller_config='sim_split'),
     'upper_ik: not starting: safety config "split" runs with controller config "g1_split", '
     'not "sim_split"'),
    (dict(controller_config='sim_fast'),
     'upper_ik: not starting: safety config "split" runs with controller config "g1_split", '
     'not "sim_fast"'),
    # the walking node's sim config too, whether or not the legs are verified
    (dict(locomotion_config='sim'),
     'g1_core: not starting: locomotion_config:=sim is for the sim; use g1'),
    (dict(locomotion_config='sim', start_position_verified='true'),
     'g1_core: not starting: locomotion_config:=sim is for the sim; use g1'),
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
    (dict(safety_config='split'),
     'upper_ik: not starting: safety config "split" is for the real robot'),
    (dict(controller_config='g1_split'),
     'upper_ik: not starting: safety config "sim_split" runs with controller config '
     '"sim_split", not "g1_split"'),
    (dict(locomotion_config='g1'),
     'g1_core: not starting: locomotion_config:=g1 is for the real robot; use sim'),
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
        _refused('upper_ik: not starting: safety config "full" is not one of the G1 pairs',
                 sim=sim, safety_config='full', start_position_verified='true')


def test_core_checks_the_pairing_before_any_node():
    # golem_upper_ik's launch file refuses a pairing before it starts the controller, and
    # g1_core includes it first, so a refusal stops the launch before any node starts
    for sim in ('false', 'true'):
        actions = core._setup(_context(core.generate_launch_description(), sim=sim))
        assert isinstance(actions[0], IncludeLaunchDescription)
        assert _location(actions[0]) == UPPER_IK_LAUNCH


def test_core_starts_the_shared_packages_through_their_launch_files():
    for sim in ('false', 'true'):
        nodes, _ = _core(sim=sim, use_sim_time=sim)
        sim_time = {'use_sim_time': sim == 'true'}
        assert nodes['robot_state_publisher'].package == 'robot_state_publisher'
        assert nodes['robot_state_publisher'].delay == 0.0
        assert nodes['joint_state_publisher'] == Started(
            'golem_upper_ik', [], 0.0, {'robot': 'g1', **sim_time})
        assert nodes['safety_node'] == Started('golem_safety', [], 0.0, {
            'robot': 'g1', 'config': 'sim_split' if sim == 'true' else 'split',
            'sim': sim == 'true', 'use_estop': False, **sim_time})
