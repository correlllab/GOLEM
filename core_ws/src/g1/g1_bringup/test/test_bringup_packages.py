"""
The G1 bringups start golem_safety and golem_upper_ik.

No launch file of g1_bringup names h12_safety_layer, h12_ros2_controller,
g1_safety_layer or g1_control, and g1_bringup depends on none of them.
g1_core.launch.py starts golem_safety's safety_node with robot g1, the safety
config, sim as the top-level file says and no hardware estop, and
golem_upper_ik's frame_task_server and joint_state_publisher with robot g1.
The launch tests need the workspace built and sourced.
"""

import ast
import importlib.util
from pathlib import Path
import re
import xml.etree.ElementTree as ElementTree

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
OLD_PACKAGES = ('h12_safety_layer', 'h12_ros2_controller', 'g1_safety_layer', 'g1_control')
OLD_NAME = re.compile(r'\b(?:' + '|'.join(OLD_PACKAGES) + r')\b')


def _old_names(path):
    """Return the OLD_PACKAGES names in path's source, which must parse."""
    source = path.read_text(encoding='utf-8')
    ast.parse(source, filename=str(path))
    return sorted(set(OLD_NAME.findall(source)))


def test_no_launch_names_old_packages():
    files = sorted((PACKAGE / 'launch').glob('*.py'))
    assert files, 'no launch file in g1_bringup/launch'
    named = {path.name: _old_names(path) for path in files}
    assert not {name: old for name, old in named.items() if old}


def test_package_depends_on_golem_safety_and_golem_upper_ik():
    root = ElementTree.parse(PACKAGE / 'package.xml').getroot()
    depends = {element.text for element in root
               if element.tag in ('depend', 'exec_depend', 'build_depend', 'test_depend')}
    assert {'golem_safety', 'golem_upper_ik'} <= depends
    assert not depends & set(OLD_PACKAGES)


def _load(launch_file):
    spec = importlib.util.spec_from_file_location(
        launch_file.split('.')[0], PACKAGE / 'launch' / launch_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _text(context, value):
    return perform_substitutions(context, normalize_to_list_of_substitutions(value))


def _context(description, **args):
    """Return a LaunchContext with description's defaults, overridden by args."""
    context = LaunchContext()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument) and entity.default_value is not None:
            context.launch_configurations[entity.name] = _text(context, entity.default_value)
    context.launch_configurations.update(args)
    return context


def _passed_to_core(launch_file):
    """Return the launch arguments launch_file passes to g1_core, with its defaults."""
    description = _load(launch_file).generate_launch_description()
    context = _context(description)
    (include,) = [e for e in description.entities if isinstance(e, IncludeLaunchDescription)]
    return {_text(context, key): _text(context, value) for key, value in include.launch_arguments}


def _core_nodes(**args):
    """Return {node name: (package, executable, parameters)} that g1_core starts with args."""
    core = _load('g1_core.launch.py')
    context = _context(core.generate_launch_description(), **args)
    nodes = {}

    def add(action):
        if isinstance(action, TimerAction):
            for inner in action.actions:
                add(inner)
        elif isinstance(action, Node):
            parameters = {}
            for params in evaluate_parameters(context, action._Node__parameters):
                assert isinstance(params, dict), params
                parameters.update(params)
            nodes[action._Node__node_name] = (action.node_package, action.node_executable,
                                              parameters)

    for action in core._setup(context):
        add(action)
    return nodes


@pytest.mark.parametrize('launch_file, sim, safety_config, controller_config', [
    ('g1_bringup.launch.py', False, 'arm_sdk', 'g1'),
    ('g1_sim_bringup.launch.py', True, 'sim_arm_sdk', 'sim'),
])
def test_core_runs_golem_safety_and_golem_upper_ik(launch_file, sim, safety_config,
                                                   controller_config):
    nodes = _core_nodes(**_passed_to_core(launch_file))
    sim_time = {'use_sim_time': sim}
    assert nodes['safety_node'] == ('golem_safety', 'safety_node', {
        'robot': 'g1', 'config': safety_config, 'sim': sim, 'use_estop': False, **sim_time})
    assert nodes['frame_task_server'] == ('golem_upper_ik', 'frame_task_server', {
        'robot': 'g1', 'config': controller_config, **sim_time})
    assert nodes['joint_state_publisher'] == ('golem_upper_ik', 'joint_state_publisher', {
        'robot': 'g1', **sim_time})
