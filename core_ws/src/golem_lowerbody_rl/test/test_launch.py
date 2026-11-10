"""
Check launch/lowerbody.launch.py: its arguments, its refusals and the node it starts.

The launch file's OpaqueFunctions run here in order, as ros2 launch runs them, so a refusal
raises before the node is returned. Needs the package and golem_safety installed.
"""

import importlib.util
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
import pytest
import yaml

LAUNCH_FILE = Path(__file__).resolve().parents[1] / 'launch' / 'lowerbody.launch.py'
ARGUMENTS = {'robot', 'config', 'policy', 'sim', 'use_sim_time', 'start_delay'}
REFUSED = '^lowerbody: not starting: '


def _description():
    """Return the launch file's description."""
    spec = importlib.util.spec_from_file_location('lowerbody_launch', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def _declared(description):
    """Return {name: DeclareLaunchArgument} of description."""
    return {entity.name: entity for entity in description.entities
            if isinstance(entity, DeclareLaunchArgument)}


def _text(context, value):
    """Return a substitution's text in context."""
    return perform_substitutions(context, normalize_to_list_of_substitutions(value))


def _launch(**args):
    """
    Run the launch file's actions with args over its defaults; return what they start.

    Raises what an action raises, as ros2 launch does before any later action.
    """
    description = _description()
    context = LaunchContext()
    for name, argument in _declared(description).items():
        if argument.default_value is not None:
            context.launch_configurations[name] = _text(context, argument.default_value)
    context.launch_configurations.update(args)
    started = []
    for entity in description.entities:
        if isinstance(entity, OpaqueFunction):
            started += entity.execute(context) or []
    return started, context


def _real_config(tmp_path, name, **changes):
    """Write the H1-2's real.yaml with changes ({section: {key: value}}) as name; return it."""
    source = os.path.join(get_package_share_directory('golem_lowerbody_rl'), 'config', 'h12',
                          'real.yaml')
    with open(source, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    for section, values in changes.items():
        raw[section].update(values)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def test_every_argument_has_a_description():
    """The launch file declares its six arguments, each with a description."""
    declared = _declared(_description())
    assert set(declared) == ARGUMENTS
    for name, argument in declared.items():
        assert argument.description and argument.description != 'no description given', name


def test_required_arguments_have_no_default():
    """robot, config and sim must be given: a forgotten one is no silent default."""
    declared = _declared(_description())
    for name in ('robot', 'config', 'sim'):
        assert declared[name].default_value is None, name


@pytest.mark.parametrize('robot', ['h12', 'g1'])
@pytest.mark.parametrize('config, sim', [('real', 'true'), ('sim', 'false')])
def test_launch_refuses_real_config_in_sim(robot, config, sim):
    """A run config whose sim differs from sim:= is refused before any node starts."""
    with pytest.raises(RuntimeError, match=REFUSED + f"run config '{config}' is for "):
        _launch(robot=robot, config=config, sim=sim)


def test_launch_refuses_unlisted_policy(tmp_path):
    """A policy the robot does not list is refused, by name or by path."""
    with pytest.raises(RuntimeError, match=REFUSED + "unknown policy 'homie' of robot h12"):
        _launch(robot='h12', config='sim', sim='true', policy='homie')
    share = get_package_share_directory('golem_lowerbody_rl')
    spec = tmp_path / 'mine.yaml'
    spec.write_text(Path(share, 'config', 'h12', 'policies', 'almi.yaml').read_text())
    with pytest.raises(RuntimeError,
                       match=REFUSED + "policy 'mine' .* is not one of robot h12's policies"):
        _launch(robot='h12', config='sim', sim='true', policy=str(spec))


def test_launch_refuses_unknown_robot():
    """An unknown robot is refused, naming the known ones."""
    with pytest.raises(RuntimeError, match=REFUSED + "unknown robot 'h1': one of g1, h12"):
        _launch(robot='h1', config='sim', sim='true')


def test_launch_refuses_real_config_without_confirm_or_with_band(tmp_path):
    """SAFETY: a real-robot run config needs the operator's confirm and releases no band."""
    path = _real_config(tmp_path, 'no_confirm.yaml', startup={'wait_for_confirm': False})
    with pytest.raises(RuntimeError, match=REFUSED + '.*startup.wait_for_confirm false'):
        _launch(robot='h12', config=path, sim='false')
    path = _real_config(tmp_path, 'band.yaml', band={'release': True})
    with pytest.raises(RuntimeError, match=REFUSED + '.*band.release true'):
        _launch(robot='h12', config=path, sim='false')
    # the same file with neither starts
    path = _real_config(tmp_path, 'custom.yaml')
    started, _ = _launch(robot='h12', config=path, sim='false')
    assert len(started) == 1


@pytest.mark.parametrize('robot, config, sim, policy', [
    ('h12', 'sim', 'true', 'almi'),
    ('h12', 'real', 'false', 'fame'),
    ('g1', 'sim', 'true', ''),
])
def test_launch_starts_the_node_after_start_delay(robot, config, sim, policy):
    """An accepted launch starts lowerbody_controller_node start_delay s later, with its args."""
    started, context = _launch(robot=robot, config=config, sim=sim, policy=policy,
                               use_sim_time=sim, start_delay='2.5')
    (timer,) = started
    assert isinstance(timer, TimerAction)
    assert timer.period == 2.5
    (node,) = timer.actions
    assert isinstance(node, Node)
    assert (node.node_package, node.node_executable) == (
        'golem_lowerbody_rl', 'lowerbody_controller_node')
    parameters = {}
    for params in evaluate_parameters(context, node._Node__parameters):
        parameters.update(params)
    assert parameters == {'robot': robot, 'config': config, 'policy': policy,
                          'use_sim_time': sim == 'true'}
