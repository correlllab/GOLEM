"""
Test that every node a top-level launch starts runs on the right clock: use_sim_time.

Each sim bringup (the H1-2's, the loco-manipulation game's and the G1's) starts every
node with use_sim_time true; each real bringup starts every node with it false, given or
by ROS's default (a node that does not set it). The launch is walked as ros2 launch runs
it, into every include (golem_slam's navigation, Nav2's navigation_launch.py, the
drivers'), and a node's use_sim_time is read from its parameters: a dict, or a parameter
file's `/**` or node-name section (Nav2's params file, which RewrittenYaml rewrites with
the launch's use_sim_time). Needs the workspace built and sourced.
"""

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
import pytest
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
# A sim bringup's ROS domain: the loco-manipulation game's refuses 0 and an unset one.
SIM_DOMAIN = '221'

# (launch file, its arguments, use_sim_time of every node): every top-level file, with the
# arguments that start the most nodes.
TOP_LEVEL = [
    ('h12_sim.launch.py', {'use_skills': 'true', 'use_nav': 'true', 'use_rviz': 'true'}, True),
    ('lmg_sim.launch.py', {'use_slam': 'true', 'use_skills': 'true'}, True),
    ('g1_sim.launch.py', {'use_skills': 'true', 'use_nav': 'true', 'use_rviz': 'true'}, True),
    ('h12_real_robot.launch.py', {'start_position_verified': 'true'}, False),
    ('h12_real_desktop.launch.py', {'use_hand_cameras': 'true'}, False),
    ('h12_safety_grippers.launch.py', {}, False),
    ('h12_real_slam.launch.py', {}, False),
    ('g1_real_robot.launch.py', {'start_position_verified': 'true', 'use_rviz': 'true'},
     False),
]


def _load(launch_file):
    """Import a golem_bringup launch file, found by name, as a module."""
    (path,) = (PACKAGE / 'launch').rglob(launch_file)
    spec = importlib.util.spec_from_file_location(launch_file.split('.')[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context(description, args):
    """Return a LaunchContext with description's defaults, overridden by args."""
    context = LaunchContext()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument) and entity.default_value is not None:
            context.launch_configurations[entity.name] = perform_substitutions(
                context, entity.default_value)
    context.launch_configurations.update(args)
    return context


def _from_file(path, name):
    """Return use_sim_time from a parameter file's `/**` or `name` section, or None."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    value = None
    for key in ('/**', name, '/' + name):
        section = (data.get(key) or {}).get('ros__parameters') or {}
        if 'use_sim_time' in section:
            value = section['use_sim_time']
    return value


def _use_sim_time(context, node):
    """Return the use_sim_time node starts with in context, or None when nothing sets it."""
    node._perform_substitutions(context)
    name = node.node_name.lstrip('/').split('/')[-1]
    value = None
    for name_, given in context.launch_configurations.get('global_params', []):
        if name_ == 'use_sim_time':
            value = given
    for params in evaluate_parameters(context, node._Node__parameters):
        if isinstance(params, dict):
            if 'use_sim_time' in params:
                value = params['use_sim_time']
        else:
            from_file = _from_file(params, name)
            if from_file is not None:
                value = from_file
    if isinstance(value, str):
        value = {'true': True, 'false': False}.get(value.lower(), value)
    return value


def _nodes(launch_file, args):
    """Return [(package, executable, name, use_sim_time)] of the nodes launch_file starts."""
    description = _load(launch_file).generate_launch_description()
    context = _context(description, args)
    found = []

    def walk(entities):
        """Record the nodes entities start, in order, as ros2 launch runs them."""
        for entity in entities:
            condition = getattr(entity, 'condition', None)
            if condition is not None and not condition.evaluate(context):
                continue
            if isinstance(entity, Node):
                value = _use_sim_time(context, entity)
                found.append((entity.node_package, entity.node_executable, entity.node_name,
                              value))
            elif isinstance(entity, TimerAction):
                walk(entity.actions)
            elif isinstance(entity, IncludeLaunchDescription):
                walk(entity.visit(context) or [])
            else:
                walk(entity.visit(context) or [])

    walk(description.entities)
    return found


@pytest.mark.parametrize('launch_file, args, sim_time', TOP_LEVEL,
                         ids=[launch_file for launch_file, _, _ in TOP_LEVEL])
def test_every_node_runs_on_its_bringups_clock(launch_file, args, sim_time, monkeypatch):
    """Every node of a sim bringup runs with use_sim_time true, of a real one false or unset."""
    monkeypatch.setenv('ROS_DOMAIN_ID', SIM_DOMAIN if sim_time else '0')
    nodes = _nodes(launch_file, args)
    assert len(nodes) > 3, nodes
    allowed = (True,) if sim_time else (False, None)
    wrong = [(package, executable, name, value) for package, executable, name, value in nodes
             if value not in allowed]
    assert wrong == [], f'{launch_file}: nodes not on use_sim_time {sim_time}: {wrong}'


def test_lmg_real_refuses_before_any_node():
    """lmg_real.launch.py refuses to start, so it starts no node on any clock."""
    with pytest.raises(RuntimeError):
        _nodes('lmg_real.launch.py', {'start_position_verified': 'true'})
