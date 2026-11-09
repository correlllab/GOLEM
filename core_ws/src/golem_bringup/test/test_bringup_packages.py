"""
Test that both robots' bringups start the shared packages' nodes through their launch files.

golem_bringup depends on golem_safety, golem_upper_ik and golem_lowerbody_rl. For the
H1-2, h12_state.launch.py includes golem_upper_ik's robot_state.launch.py, and
h12_control.launch.py, which needs sim from
every top-level file (the real-robot ones pass use_estop), includes golem_upper_ik's
upper_ik.launch.py first, whose pairing check refuses a frame_task_config that does not
run with its safety_config, then golem_safety's safety.launch.py. For the G1,
g1_core.launch.py includes the same three files: safety_node with robot g1, the safety
config, sim as the top-level file says and no hardware estop, and frame_task_server and
joint_state_publisher with robot g1. The H1-2's h12_lowerbody.launch.py includes
golem_lowerbody_rl's lowerbody.launch.py with robot h12 and every argument it takes, behind
start_position_verified and not with policy:=mjpc (g1_core's include, robot g1, is in
test_g1_launch.py). The nodes are read by walking those
includes, as ros2 launch runs them. The launch tests need the workspace built and sourced.
"""

import importlib.util
import os
from pathlib import Path
import xml.etree.ElementTree as ElementTree

from ament_index_python.packages import get_package_share_directory
from golem_lowerbody_rl import launch_check
from golem_ros2_models.paths import urdf_path
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
import pytest

from test_g1_launch import (
    _core_nodes, _location, _started, Started, UPPER_IK_LAUNCH)

PACKAGE = Path(__file__).resolve().parents[1]
LOWERBODY_LAUNCH = os.path.join(get_package_share_directory('golem_lowerbody_rl'), 'launch',
                                'lowerbody.launch.py')


def test_package_depends_on_the_shared_packages():
    """package.xml depends on golem_safety, golem_upper_ik and golem_lowerbody_rl."""
    root = ElementTree.parse(PACKAGE / 'package.xml').getroot()
    depends = {element.text for element in root
               if element.tag in ('depend', 'exec_depend', 'build_depend', 'test_depend')}
    assert {'golem_safety', 'golem_upper_ik', 'golem_lowerbody_rl'} <= depends


def _load(launch_file):
    """Import a golem_bringup launch file, found by name, as a module."""
    (path,) = (PACKAGE / 'launch').rglob(launch_file)
    spec = importlib.util.spec_from_file_location(launch_file.split('.')[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _text(context, value):
    """Return a substitution's text in context."""
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
    """h12_control's sim argument has no default."""
    description = _load('h12_control.launch.py').generate_launch_description()
    (sim,) = [entity for entity in description.entities
              if isinstance(entity, DeclareLaunchArgument) and entity.name == 'sim']
    assert sim.default_value is None


def _passed_to(launch_file, included):
    """Return the launch arguments launch_file passes to its include of included."""
    description = _load(launch_file).generate_launch_description()
    context = _context(description)
    (include,) = [entity for entity in description.entities
                  if isinstance(entity, IncludeLaunchDescription)
                  and Path(_location(entity)).name == included]
    return {_text(context, name): _text(context, value)
            for name, value in include.launch_arguments}


def _walk(launch_file, **args):
    """Return {executable: Started} of the nodes launch_file starts with args."""
    description = _load(launch_file).generate_launch_description()
    context = _context(description, **args)
    return _started(description.entities, context)[0]


@pytest.mark.parametrize('launch_file, sim, use_estop', [
    ('h12_sim.launch.py', 'true', 'false'),
    ('lmg_sim.launch.py', 'true', 'false'),
    ('h12_real_robot.launch.py', 'false', 'true'),
    ('lmg_real.launch.py', 'false', 'true'),
    ('h12_safety_grippers.launch.py', 'false', 'true'),
])
def test_top_level_files_pass_sim_and_use_estop(launch_file, sim, use_estop, monkeypatch):
    """Each top-level H1-2 file passes h12_control its sim and use_estop, which reach the nodes."""
    # lmg_sim refuses an unset or 0 ROS_DOMAIN_ID.
    monkeypatch.setenv('ROS_DOMAIN_ID', '193')
    arguments = _passed_to(launch_file, 'h12_control.launch.py')
    assert arguments['sim'] == sim
    # Absent: h12_control.launch.py's default, false.
    assert arguments.get('use_estop', 'false') == use_estop
    nodes = _walk('h12_control.launch.py', **arguments)
    assert nodes['safety_node'].parameters['sim'] == (sim == 'true')
    assert nodes['safety_node'].parameters['use_estop'] == (use_estop == 'true')
    assert ('estop_node' in nodes) == (use_estop == 'true')


def test_lmg_real_refuses_to_start_before_any_node():
    """lmg_real refuses first: lmg_deploy reads /livox/imu as upright, the H1-2's is not."""
    description = _load('lmg_real.launch.py').generate_launch_description()
    assert isinstance(description.entities[0], OpaqueFunction)
    with pytest.raises(RuntimeError, match='^lmg_real: not starting: lmg_deploy reads /livox/imu'):
        description.entities[0].execute(LaunchContext())


def test_lmg_sim_is_not_guarded(monkeypatch):
    """lmg_sim has no refusal of its own on a sim domain."""
    monkeypatch.setenv('ROS_DOMAIN_ID', '193')
    description = _load('lmg_sim.launch.py').generate_launch_description()
    assert not any(isinstance(entity, OpaqueFunction) for entity in description.entities)


def test_control_checks_the_pair_before_any_node():
    """h12_control includes upper_ik.launch.py, which checks the pairing, before anything else."""
    entities = [entity for entity in _load('h12_control.launch.py')
                .generate_launch_description().entities
                if not isinstance(entity, DeclareLaunchArgument)]
    assert isinstance(entities[0], IncludeLaunchDescription)
    assert _location(entities[0]) == UPPER_IK_LAUNCH


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
    """With frame_task_server, a safety and controller config that do not pair are refused."""
    args = dict(sim=str(safety_config.startswith('sim')).lower(),
                safety_config=safety_config, frame_task_config=frame_task_config,
                use_frame_task=use_frame_task)
    if refused:
        with pytest.raises(RuntimeError, match='^upper_ik: not starting: safety config '):
            _walk('h12_control.launch.py', **args)
        return
    nodes = _walk('h12_control.launch.py', **args)
    assert nodes['safety_node'].parameters['config'] == safety_config
    if use_frame_task == 'true':
        assert nodes['frame_task_server'].parameters['config'] == frame_task_config
    else:
        assert 'frame_task_server' not in nodes


def test_control_starts_safety_and_the_controller_after_their_delays():
    """The real robot staggers estop_node, safety_node and frame_task_server; the sim does not."""
    nodes = _walk('h12_control.launch.py', sim='false', use_estop='true')
    assert nodes['estop_node'] == Started('estop', [], 0.0, {'use_sim_time': False})
    assert nodes['safety_node'] == Started('golem_safety', [], 3.0, {
        'robot': 'h12', 'config': 'relax_safety_split.yaml', 'sim': False,
        'use_estop': True, 'use_sim_time': False})
    assert nodes['frame_task_server'] == Started('golem_upper_ik', [], 5.0, {
        'robot': 'h12', 'config': 'safety_split.yaml', 'use_sim_time': False})
    # The sim bringup waits for no estop.
    sim = _passed_to('h12_sim.launch.py', 'h12_control.launch.py')
    nodes = _walk('h12_control.launch.py', **sim)
    assert set(nodes) == {'safety_node', 'frame_task_server'}
    assert nodes['safety_node'] == Started('golem_safety', [], 0.0, {
        'robot': 'h12', 'config': 'sim_safety_split.yaml', 'sim': True,
        'use_estop': False, 'use_sim_time': True})
    assert nodes['frame_task_server'] == Started('golem_upper_ik', [], 0.0, {
        'robot': 'h12', 'config': 'sim_safety_split.yaml', 'use_sim_time': True})


def test_state_starts_golem_upper_ik_robot_state():
    """h12_state starts golem_upper_ik's two state nodes on urdf_file."""
    sim = _passed_to('h12_sim.launch.py', 'h12_state.launch.py')
    nodes = _walk('h12_state.launch.py', **sim)
    assert set(nodes) == {'robot_state_publisher', 'joint_state_publisher'}
    assert nodes['joint_state_publisher'] == Started(
        'golem_upper_ik', [], 0.0, {'robot': 'h12', 'use_sim_time': True})
    assert nodes['robot_state_publisher'] == Started('robot_state_publisher', [], 0.0, {
        'robot_description': Path(urdf_path('h12', ros=True)).read_text(),
        'use_sim_time': True})
    # urdf_file passes through.
    nodes = _walk('h12_state.launch.py', urdf_file=str(PACKAGE / 'package.xml'))
    assert nodes['robot_state_publisher'].parameters['robot_description'] == (
        PACKAGE / 'package.xml').read_text()


def _passed_to_core(launch_file):
    """Return the launch arguments launch_file passes to g1_core, with its defaults."""
    description = _load(launch_file).generate_launch_description()
    context = _context(description)
    (include,) = [e for e in description.entities if isinstance(e, IncludeLaunchDescription)]
    return {_text(context, key): _text(context, value) for key, value in include.launch_arguments}


@pytest.mark.parametrize('launch_file, sim, safety_config, controller_config', [
    ('g1_real_robot.launch.py', False, 'split', 'g1_split'),
    ('g1_sim.launch.py', True, 'sim_split', 'sim_split'),
])
def test_core_runs_golem_safety_and_golem_upper_ik(launch_file, sim, safety_config,
                                                   controller_config):
    """Both G1 bringups start golem_safety's and golem_upper_ik's nodes with robot g1."""
    nodes = _core_nodes(**_passed_to_core(launch_file))
    sim_time = {'use_sim_time': sim}
    assert nodes['safety_node'] == Started('golem_safety', [], 0.0, {
        'robot': 'g1', 'config': safety_config, 'sim': sim, 'use_estop': False, **sim_time})
    assert nodes['frame_task_server'] == Started('golem_upper_ik', [], 2.0, {
        'robot': 'g1', 'config': controller_config, **sim_time})
    assert nodes['joint_state_publisher'] == Started('golem_upper_ik', [], 0.0, {
        'robot': 'g1', **sim_time})


def _legs(**args):
    """Return {executable: Started} of what h12_lowerbody starts with args."""
    description = _load('h12_lowerbody.launch.py').generate_launch_description()
    context = _context(description, **args)
    return _started(description.entities, context, descend={LOWERBODY_LAUNCH})[0]


@pytest.mark.parametrize('launch_file, sim, config', [
    ('h12_sim.launch.py', 'true', 'sim'),
    ('h12_real_robot.launch.py', 'false', 'real'),
])
def test_top_level_files_run_the_leg_node(launch_file, sim, config):
    """The sim and the real robot run ALMI on their run config, through lowerbody.launch.py."""
    arguments = _passed_to(launch_file, 'h12_lowerbody.launch.py')
    assert (arguments['policy'], arguments['sim'], arguments['lowerbody_config']) == (
        'almi', sim, config)
    # SAFETY: the real robot passes the operator's start_position_verified, false by default
    assert arguments['start_position_verified'] == ('true' if sim == 'true' else 'false')
    nodes = _legs(**dict(arguments, start_position_verified='true'))
    assert nodes['lowerbody_controller_node'] == Started('golem_lowerbody_rl', [], 0.0, {
        'robot': 'h12', 'config': config, 'policy': 'almi', 'use_sim_time': sim == 'true'})
    assert 'estimator_node' not in nodes


def test_leg_block_passes_every_argument():
    """An earlier include's config and start_delay (Humble leaks them) do not reach the node."""
    nodes = _legs(sim='true', lowerbody_config='sim', start_position_verified='true',
                  policy='fame', config='sim_safety_split.yaml', start_delay='5.0')
    assert nodes['lowerbody_controller_node'] == Started('golem_lowerbody_rl', [], 0.0, {
        'robot': 'h12', 'config': 'sim', 'policy': 'fame', 'use_sim_time': False})


@pytest.mark.parametrize('args', [
    dict(start_position_verified='false'),
    dict(start_position_verified='true', policy='mjpc'),
])
def test_leg_block_interlock_and_mjpc_start_no_leg_node(args):
    """SAFETY: no leg node without start_position_verified, and none with policy:=mjpc."""
    nodes = _legs(sim='false', lowerbody_config='real', **args)
    assert 'lowerbody_controller_node' not in nodes


@pytest.mark.parametrize('args, reason', [
    (dict(sim='true', lowerbody_config='real'), "run config 'real' is for the real robot"),
    (dict(sim='false', lowerbody_config='sim'), "run config 'sim' is for a simulator"),
    (dict(sim='true', lowerbody_config='sim', policy='homie'), "unknown policy 'homie'"),
])
def test_leg_block_refuses_before_the_leg_node(args, reason):
    """lowerbody.launch.py refuses a run config for the other platform and an unlisted policy."""
    with pytest.raises(RuntimeError, match=f'^lowerbody: not starting: {reason}'):
        _legs(start_position_verified='true', **args)


def _after_the_arguments(description):
    """Return description's entities after its DeclareLaunchArguments."""
    return [entity for entity in description.entities
            if not isinstance(entity, DeclareLaunchArgument)]


# (launch file, arguments, run config, sim) of each top-level H1-2 file that includes the
# leg block, set so that the leg node starts
LEG_BRINGUPS = [
    ('h12_sim.launch.py', {}, 'sim', True),
    ('h12_real_robot.launch.py', {'start_position_verified': 'true'}, 'real', False),
]


@pytest.mark.parametrize('launch_file, args, config, sim', LEG_BRINGUPS,
                         ids=[bringup[0] for bringup in LEG_BRINGUPS])
def test_h12_bringups_check_the_legs_before_any_node(launch_file, args, config, sim,
                                                     monkeypatch):
    """
    SAFETY: each H1-2 bringup with legs runs the leg node's check first.

    The leg block is included last, so its own check would refuse only after the other
    blocks' nodes started; the bringup's copy, first, refuses before any node, checking
    what the leg block is passed.
    """
    description = _load(launch_file).generate_launch_description()
    assert isinstance(_after_the_arguments(description)[0], OpaqueFunction)
    calls = []

    def refuse(*arguments, **keywords):
        """Record the check's arguments and refuse."""
        calls.append((arguments, keywords))
        return 'refused for the test'
    monkeypatch.setattr(launch_check, 'lowerbody_refusal', refuse)
    with pytest.raises(RuntimeError, match='^lowerbody: not starting: refused for the test$'):
        _started(description.entities, _context(description, **args))
    assert calls == [(('h12', config, 'almi'), {'sim': sim})]


@pytest.mark.parametrize('launch_file, args', [
    ('h12_sim.launch.py', {'policy': 'homie'}),
    ('h12_real_robot.launch.py', {'policy': 'homie', 'start_position_verified': 'true'}),
])
def test_h12_bringups_refuse_an_unlisted_policy_first(launch_file, args):
    """A policy the H1-2 does not list is refused by the bringup's first action."""
    description = _load(launch_file).generate_launch_description()
    context = _context(description, **args)
    with pytest.raises(RuntimeError, match="^lowerbody: not starting: unknown policy 'homie'"):
        _after_the_arguments(description)[0].execute(context)


@pytest.mark.parametrize('launch_file, args', [
    ('h12_real_robot.launch.py', {'policy': 'homie'}),
    ('h12_real_robot.launch.py', {'policy': 'mjpc', 'start_position_verified': 'true'}),
    ('h12_sim.launch.py', {'policy': 'mjpc'}),
], ids=['not verified', 'real mjpc', 'sim mjpc'])
def test_h12_bringups_check_no_leg_node_they_do_not_start(launch_file, args, monkeypatch):
    """Check no leg node where the leg block starts none: no verified start pose, or mjpc."""
    monkeypatch.setattr(launch_check, 'lowerbody_refusal', lambda *a, **k: 'never asked')
    description = _load(launch_file).generate_launch_description()
    context = _context(description, **args)
    check = _after_the_arguments(description)[0]
    assert isinstance(check, OpaqueFunction)
    assert not check.condition.evaluate(context)
