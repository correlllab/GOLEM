"""
use_locomotion and config pairing in the bringup.

g1_core refuses unsafe settings and sim/real mixes, picks the split-mode
configs, and the real and sim launches pass the interlock on. Needs the
workspace built and sourced (g1_bringup, g1_description and golem_upper_ik
installed).
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

LAUNCH_DIR = Path(__file__).resolve().parents[1] / 'launch'


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


@pytest.mark.parametrize('args, message', [
    # the legs need split mode, and the real robot never switches to it by itself
    (dict(use_locomotion='true', start_position_verified='true'),
     'drives the legs through golem_safety split mode, '
     'so it needs safety_config:=split, not arm_sdk'),
    (dict(use_locomotion='true', safety_config='sim_split', start_position_verified='true'),
     'needs safety_config:=split, not sim_split'),
    (dict(sim='true', use_locomotion='true', safety_config='sim_arm_sdk'),
     'needs safety_config:=sim_split, not sim_arm_sdk'),
    (dict(sim='true', use_locomotion='true', safety_config='split'),
     'needs safety_config:=sim_split, not split'),
    # the real robot's interlock
    (dict(use_locomotion='true', safety_config='split'),
     'use_locomotion:=true on the real robot needs start_position_verified:=true'),
    (dict(use_locomotion='true', safety_config='split', start_position_verified='no'),
     'needs start_position_verified:=true'),
    # the walking node's sim and real configs do not mix
    (dict(use_locomotion='true', safety_config='split', start_position_verified='true',
          locomotion_config='sim'), 'locomotion_config:=sim is for the sim; use g1'),
    (dict(sim='true', use_locomotion='true', locomotion_config='g1'),
     'locomotion_config:=g1 is for the real robot; use sim'),
    # walking needs the split controller config's stiff waist
    (dict(use_locomotion='true', safety_config='split', start_position_verified='true',
          controller_config='g1'),
     'needs controller_config:=g1_split, whose stiff waist the walking policy expects, not g1'),
    (dict(use_locomotion='true', safety_config='split', start_position_verified='true',
          controller_config='sim_split'),
     'needs controller_config:=g1_split, whose stiff waist the walking policy expects, '
     'not sim_split'),
    (dict(sim='true', use_locomotion='true', controller_config='sim'),
     'needs controller_config:=sim_split'),
])
def test_core_refuses_to_start_walking_unsafely(args, message):
    with pytest.raises(RuntimeError, match='g1_core: not starting: .*' + re.escape(message)):
        core._setup(_context(core.generate_launch_description(), **args))


def test_core_defaults_are_unchanged():
    # arm_sdk mode and no walking node, on the real robot and in the sim; the
    # interlock alone starts nothing. The controller config follows the safety
    # config: split mode holds the waist stiff, walking or not
    for args, safety, controller in [
        (dict(), 'arm_sdk', 'g1'),
        (dict(start_position_verified='true'), 'arm_sdk', 'g1'),
        (dict(sim='true'), 'sim_arm_sdk', 'sim'),
        (dict(safety_config='split'), 'split', 'g1_split'),
        (dict(sim='true', safety_config='sim_split'), 'sim_split', 'sim_split'),
    ]:
        nodes = _core_nodes(**args)
        assert _configs(nodes) == {'safety_node': safety, 'frame_task_server': controller}
        assert 'locomotion_node' not in nodes


def test_core_starts_walking_in_split_mode_after_the_controller():
    nodes = _core_nodes(use_locomotion='true', safety_config='split',
                        start_position_verified='true')
    assert _configs(nodes) == {'safety_node': 'split', 'frame_task_server': 'g1_split'}
    arguments, delay, _ = nodes['locomotion_node']
    assert arguments == ['--config', 'g1.yaml']
    assert delay == nodes['frame_task_server'].delay + 2.0 == 4.0

    # the sim switches to sim_split by itself and needs no interlock
    nodes = _core_nodes(sim='true', use_locomotion='true', locomotion_delay='1.5')
    assert _configs(nodes) == {'safety_node': 'sim_split', 'frame_task_server': 'sim_split'}
    assert nodes['locomotion_node'][:2] == (['--config', 'sim.yaml'], 3.5)


def test_real_bringup_passes_the_interlock_on():
    passed = _passed_to_core('g1_bringup.launch.py')
    assert (passed['sim'], passed['safety_config'], passed['use_locomotion'],
            passed['start_position_verified']) == ('false', 'arm_sdk', 'false', 'false')
    assert _configs(_core_nodes(**passed))['frame_task_server'] == 'g1'

    # use_locomotion alone is refused (arm_sdk mode), and split mode without
    # start_position_verified too
    for args, message in [
        (dict(use_locomotion='true'), 'not arm_sdk'),
        (dict(use_locomotion='true', safety_config='split'), 'start_position_verified'),
    ]:
        with pytest.raises(RuntimeError, match=message):
            _core_nodes(**_passed_to_core('g1_bringup.launch.py', **args))
    passed = _passed_to_core('g1_bringup.launch.py', use_locomotion='true', safety_config='split',
                             start_position_verified='true')
    assert passed['start_position_verified'] == 'true'
    assert 'locomotion_node' in _core_nodes(**passed)


def test_sim_bringup_switches_to_sim_split_for_walking():
    passed = _passed_to_core('g1_sim_bringup.launch.py')
    assert (passed['sim'], passed['use_locomotion']) == ('true', 'false')
    nodes = _core_nodes(**passed)
    assert _configs(nodes) == {'safety_node': 'sim_arm_sdk', 'frame_task_server': 'sim'}
    assert 'locomotion_node' not in nodes

    nodes = _core_nodes(**_passed_to_core('g1_sim_bringup.launch.py', use_locomotion='true'))
    assert _configs(nodes) == {'safety_node': 'sim_split', 'frame_task_server': 'sim_split'}
    assert nodes['locomotion_node'].arguments == ['--config', 'sim.yaml']
    with pytest.raises(RuntimeError, match='not sim_arm_sdk'):
        _core_nodes(**_passed_to_core('g1_sim_bringup.launch.py', use_locomotion='true',
                                      safety_config='sim_arm_sdk'))


@pytest.mark.parametrize('args, message', [
    # the real robot never runs a sim config, walking or not
    (dict(safety_config='sim_arm_sdk'), 'safety config "sim_arm_sdk" is a sim config'),
    (dict(safety_config='sim_split'), 'safety config "sim_split" is a sim config'),
    (dict(controller_config='sim'), 'controller config "sim" is a sim config'),
    (dict(safety_config='split', controller_config='sim_split'),
     'controller config "sim_split" is a sim config'),
    (dict(controller_config='sim_fast'), 'controller config "sim_fast" is a sim config'),
    # nor the sim a real one
    (dict(sim='true', safety_config='arm_sdk'), 'safety config "arm_sdk" is for the real robot'),
    (dict(sim='true', controller_config='g1'), 'controller config "g1" is for the real robot'),
    # golem_safety must relay the controller's upper-body channel
    (dict(safety_config='full'), 'safety config "full" must relay the upper-body channel'),
    # each safety config runs with one controller config, walking or not
    (dict(safety_config='split', controller_config='g1'),
     'safety config "split" runs with controller config "g1_split", not "g1"'),
    (dict(controller_config='g1_split'),
     'safety config "arm_sdk" runs with controller config "g1", not "g1_split"'),
    (dict(sim='true', safety_config='sim_split', controller_config='sim'),
     'safety config "sim_split" runs with controller config "sim_split", not "sim"'),
    (dict(sim='true', controller_config='sim_split'),
     'safety config "sim_arm_sdk" runs with controller config "sim", not "sim_split"'),
])
def test_core_refuses_config_mixes(args, message):
    with pytest.raises(RuntimeError, match='g1_core: not starting: ' + re.escape(message)):
        core._setup(_context(core.generate_launch_description(), **args))


def test_real_bringup_refuses_sim_configs():
    for safety in ('sim_arm_sdk', 'sim_split'):
        passed = _passed_to_core('g1_bringup.launch.py', safety_config=safety)
        assert passed['sim'] == 'false'
        with pytest.raises(RuntimeError, match=f'"{safety}" is a sim config'):
            _core_nodes(**passed)
    # the real configs still start; split mode runs the stiff split waist even
    # without the walking node
    nodes = _core_nodes(**_passed_to_core('g1_bringup.launch.py', safety_config='split'))
    assert _configs(nodes) == {'safety_node': 'split', 'frame_task_server': 'g1_split'}
