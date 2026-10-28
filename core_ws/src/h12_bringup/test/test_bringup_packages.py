"""
The H1-2 bringups and the stack test start golem_safety and golem_upper_ik.

No launch file of either, and no dependency of h12_bringup, names
h12_safety_layer, h12_ros2_controller, g1_safety_layer or g1_control.
h12_control.launch.py needs its sim argument from every top-level file, and
refuses a frame_task_config that does not run with its safety_config. The
launch tests need the workspace built and sourced.
"""

import ast
import importlib.util
from pathlib import Path
import re
import xml.etree.ElementTree as ElementTree

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
import launch.launch_description_sources
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
# The repository's tests/stack, beside core_ws. The golem_ros container mounts
# core_ws only; the ROS container of tests/stack/run.sh mounts tests/stack here too.
STACK = Path(__file__).resolve().parents[4] / 'tests' / 'stack'
OLD_PACKAGES = ('h12_safety_layer', 'h12_ros2_controller', 'g1_safety_layer', 'g1_control')
OLD_NAME = re.compile(r'\b(?:' + '|'.join(OLD_PACKAGES) + r')\b')


def _old_names(path):
    """Return the OLD_PACKAGES names in path's source, which must parse."""
    source = path.read_text(encoding='utf-8')
    ast.parse(source, filename=str(path))
    return sorted(set(OLD_NAME.findall(source)))


@pytest.mark.parametrize('directory, pattern', [
    (PACKAGE / 'launch', '*.py'),
    (STACK, '*.launch.py'),
], ids=['h12_bringup/launch', 'tests/stack'])
def test_no_launch_names_old_packages(directory, pattern):
    if directory == STACK and not STACK.is_dir():
        pytest.skip(f'{STACK} is not mounted in this container')
    files = sorted(directory.glob(pattern))
    assert files, f'no {pattern} in {directory}'
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


def test_control_sim_has_no_default():
    description = _load('h12_control.launch.py').generate_launch_description()
    (sim,) = [entity for entity in description.entities
              if isinstance(entity, DeclareLaunchArgument) and entity.name == 'sim']
    assert sim.default_value is None


@pytest.mark.parametrize('launch_file, sim', [
    ('h12_sim_bringup.launch.py', 'true'),
    ('lmg_sim_bringup.launch.py', 'true'),
    ('h12_real_robot_bringup.launch.py', 'false'),
    ('lmg_real_bringup.launch.py', 'false'),
    ('h12_safety_grippers.launch.py', 'false'),
])
def test_top_level_files_pass_sim(launch_file, sim, monkeypatch):
    class Source(PythonLaunchDescriptionSource):
        """A PythonLaunchDescriptionSource that keeps the path it was given."""

        def __init__(self, launch_file_path):
            super().__init__(launch_file_path)
            self.path = launch_file_path

    monkeypatch.setattr(launch.launch_description_sources, 'PythonLaunchDescriptionSource',
                        Source)
    # lmg_sim_bringup refuses an unset or 0 ROS_DOMAIN_ID
    monkeypatch.setenv('ROS_DOMAIN_ID', '193')
    description = _load(launch_file).generate_launch_description()
    context = _context(description)
    (control,) = [entity for entity in description.entities
                  if isinstance(entity, IncludeLaunchDescription)
                  and Path(entity.launch_description_source.path).name == 'h12_control.launch.py']
    arguments = {_text(context, name): _text(context, value)
                 for name, value in control.launch_arguments}
    assert arguments['sim'] == sim


def test_control_checks_the_pair_before_any_node():
    entities = [entity for entity in
                _load('h12_control.launch.py').generate_launch_description().entities
                if not isinstance(entity, DeclareLaunchArgument)]
    assert isinstance(entities[0], OpaqueFunction)


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
