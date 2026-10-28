"""
The bringups of both robots start golem_safety and golem_upper_ik.

golem_bringup depends on both packages. For the H1-2, h12_control.launch.py needs sim
from every top-level file, the real-robot ones pass use_estop, and it refuses a
frame_task_config that does not run with its safety_config. For the G1, g1_core.launch.py
starts golem_safety's safety_node with robot g1, the safety config, sim as the top-level
file says and no hardware estop, and golem_upper_ik's frame_task_server and
joint_state_publisher with robot g1. The launch tests need the workspace built and
sourced; the G1 one is skipped without g1_description, which an H1-2 workspace does
not build.
"""

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ElementTree

from launch import LaunchContext
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            TimerAction)
import launch.launch_description_sources
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
import pytest

PACKAGE = Path(__file__).resolve().parents[1]


def test_package_depends_on_golem_safety_and_golem_upper_ik():
    root = ElementTree.parse(PACKAGE / 'package.xml').getroot()
    depends = {element.text for element in root
               if element.tag in ('depend', 'exec_depend', 'build_depend', 'test_depend')}
    assert {'golem_safety', 'golem_upper_ik'} <= depends


def _load(launch_file):
    (path,) = (PACKAGE / 'launch').rglob(launch_file)
    spec = importlib.util.spec_from_file_location(launch_file.split('.')[0], path)
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


def test_control_sim_has_no_default():
    description = _load('h12_control.launch.py').generate_launch_description()
    (sim,) = [entity for entity in description.entities
              if isinstance(entity, DeclareLaunchArgument) and entity.name == 'sim']
    assert sim.default_value is None


@pytest.mark.parametrize('launch_file, sim, use_estop', [
    ('h12_sim.launch.py', 'true', 'false'),
    ('lmg_sim.launch.py', 'true', 'false'),
    ('h12_real_robot.launch.py', 'false', 'true'),
    ('lmg_real.launch.py', 'false', 'true'),
    ('h12_safety_grippers.launch.py', 'false', 'true'),
])
def test_top_level_files_pass_sim_and_use_estop(launch_file, sim, use_estop, monkeypatch):
    class Source(PythonLaunchDescriptionSource):
        """A PythonLaunchDescriptionSource that keeps the path it was given."""

        def __init__(self, launch_file_path):
            super().__init__(launch_file_path)
            self.path = launch_file_path

    monkeypatch.setattr(launch.launch_description_sources, 'PythonLaunchDescriptionSource',
                        Source)
    # lmg_sim refuses an unset or 0 ROS_DOMAIN_ID
    monkeypatch.setenv('ROS_DOMAIN_ID', '193')
    description = _load(launch_file).generate_launch_description()
    context = _context(description)
    (control,) = [entity for entity in description.entities
                  if isinstance(entity, IncludeLaunchDescription)
                  and Path(entity.launch_description_source.path).name == 'h12_control.launch.py']
    arguments = {_text(context, name): _text(context, value)
                 for name, value in control.launch_arguments}
    assert arguments['sim'] == sim
    # absent: h12_control.launch.py's default, false
    assert arguments.get('use_estop', 'false') == use_estop


def test_control_checks_the_pair_before_any_node(monkeypatch):
    control = _load('h12_control.launch.py')
    calls = []
    monkeypatch.setattr(control, 'check_pairing', lambda context: calls.append(context) or [])
    entities = [entity for entity in control.generate_launch_description().entities
                if not isinstance(entity, DeclareLaunchArgument)]
    assert isinstance(entities[0], OpaqueFunction)
    context = LaunchContext()
    assert entities[0].execute(context) == []
    assert calls == [context]


@pytest.mark.parametrize('safety_config, frame_task_config, use_frame_task, refused', [
    ('relax_safety_split.yaml', 'safety_split.yaml', 'true', False),
    ('sim_safety_split.yaml', 'sim_safety_split.yaml', 'true', False),
    ('tight_safety_full', 'safety_full', 'true', False),
    ('sim_safety_split.yaml', 'safety_split.yaml', 'true', True),
    ('tight_safety_full.yaml', 'safety_split.yaml', 'true', True),
    ('tight_safety_full.yaml', 'safety_split.yaml', 'false', False),
])
def test_control_refuses_a_mismatched_pair(safety_config, frame_task_config, use_frame_task,
                                           refused):
    control = _load('h12_control.launch.py')
    context = _context(control.generate_launch_description(), sim='true',
                       safety_config=safety_config, frame_task_config=frame_task_config,
                       use_frame_task=use_frame_task)
    if refused:
        with pytest.raises(RuntimeError, match='^h12_control: not starting: safety config '):
            control.check_pairing(context)
    else:
        assert control.check_pairing(context) == []


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
    ('g1_real_robot.launch.py', False, 'split', 'g1_split'),
    ('g1_sim.launch.py', True, 'sim_split', 'sim_split'),
])
def test_core_runs_golem_safety_and_golem_upper_ik(launch_file, sim, safety_config,
                                                   controller_config):
    pytest.importorskip('g1_description')   # g1_core.launch.py imports it
    nodes = _core_nodes(**_passed_to_core(launch_file))
    sim_time = {'use_sim_time': sim}
    assert nodes['safety_node'] == ('golem_safety', 'safety_node', {
        'robot': 'g1', 'config': safety_config, 'sim': sim, 'use_estop': False, **sim_time})
    assert nodes['frame_task_server'] == ('golem_upper_ik', 'frame_task_server', {
        'robot': 'g1', 'config': controller_config, **sim_time})
    assert nodes['joint_state_publisher'] == ('golem_upper_ik', 'joint_state_publisher', {
        'robot': 'g1', **sim_time})
