"""Real ROS message types, mocked skill endpoints; never initialize ROS/DDS."""

import ast
import copy
import json
from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Duration, Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from custom_ros_messages.action import SkillGrasp, SkillPickPlace
from h12_skills.base import SkillsBase, _Run
from h12_skills.marker_targets import BoardObservation, MarkerCatalog
from h12_skills.node import SkillsNode
from h12_skills.perception_utils import matrix_to_pose, pose_to_matrix
from h12_skills.skills.grasp import GraspSkill
from h12_skills.skills.pick_place import PickPlaceSkill


ROOT = Path(__file__).resolve().parents[4]
CATALOG = ROOT / 'CL_Assets/mujoco_assets/manipulation_net/manipulation_catalog.json'


@pytest.fixture(autouse=True)
def no_live_ros_endpoints(monkeypatch):
    import rclpy

    def forbidden(*args, **kwargs):
        pytest.fail('offline tests must not initialize ROS or construct graph endpoints')

    monkeypatch.setattr(rclpy, 'init', forbidden)
    monkeypatch.setattr(Node, '__init__', forbidden)
    monkeypatch.setattr(ActionClient, '__init__', forbidden)


class Log:
    def info(self, *args, **kwargs):
        pass

    warn = error = info


class Goal:
    def __init__(self, request):
        self.request = request
        self.is_cancel_requested = False
        self.terminal = None
        self.feedback = []

    def publish_feedback(self, feedback):
        self.feedback.append(copy.deepcopy(feedback))

    def succeed(self):
        self.terminal = 'succeeded'

    def abort(self):
        self.terminal = 'aborted'

    def canceled(self):
        self.terminal = 'canceled'


def observation(stamp=10, board=None, valid=True):
    pose = PoseStamped()
    pose.header.stamp = Time(sec=stamp)
    pose.header.frame_id = 'pelvis'
    pose.pose = matrix_to_pose(np.eye(4) if board is None else board)
    diagnostic = DiagnosticArray()
    diagnostic.header.stamp = Time(sec=stamp)
    status = DiagnosticStatus(name='pegboard_localization',
                              level=DiagnosticStatus.OK if valid else DiagnosticStatus.WARN)
    status.values = [KeyValue(key='valid', value=str(valid).lower()),
                     KeyValue(key='observation_stamp_ns', value=str(stamp * 10**9))]
    diagnostic.status = [status]
    return pose, diagnostic


def test_catalog_geometry_tip_contact_and_destination():
    catalog = MarkerCatalog(CATALOG)
    g = catalog.geometry('cylinder_1', 'cylinder_3', 'left')
    assert g.source_tip[2, 3] == pytest.approx(.003)
    assert g.length_m == pytest.approx(.05)
    assert g.aperture_m == pytest.approx(.020, abs=1e-7)
    contact = g.grasp @ [0, 0, .1146, 1]
    np.testing.assert_allclose(contact[:3], g.source_tip[:3, 3] + [0, 0, .04])
    np.testing.assert_allclose(g.grasp[:3, 2], [0, 0, -1], atol=1e-10)
    np.testing.assert_allclose(g.extracted[:3, 3] - g.grasp[:3, 3], [0, 0, .06])
    carried_tip = g.preinsert @ np.linalg.inv(g.grasp) @ g.source_tip
    target = catalog.catalog['holes']['cylinder_3']
    np.testing.assert_allclose(carried_tip[:3, 3], np.array(target['entry']) + [0, 0, .04])
    # Global board rotation/translation affects BOTH contact and approach axes.
    board = np.eye(4)
    board[:3, :3] = Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix()
    board[:3, 3] = [.5, .2, -.05]
    candidate = g.candidates(board).grasps[0]
    np.testing.assert_allclose(pose_to_matrix(candidate.pose), board @ g.grasp, atol=1e-7)


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_top_down_matches_workspace_analysis_axes_and_contact(arm):
    catalog = MarkerCatalog(CATALOG)
    report = ROOT / 'core_ws/src/h12_ros2_controller/docs/workspace_analysis/head_camera.json'
    sample = json.loads(report.read_text())['boxes'][f'pitch90_{arm}']
    grasp_rotation = Rotation.from_euler('xyz', sample['rpy_deg'], degrees=True).as_matrix()
    # G axes in grasp_frame: closing=+Y, lateral=+Z, approach=+X.
    grasp_to_g = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    g = catalog.geometry('cylinder_1', 'cylinder_3', arm, top_down=True)
    np.testing.assert_allclose(g.grasp[:3, :3], grasp_rotation @ grasp_to_g, atol=1e-14)
    np.testing.assert_allclose(g.grasp[:3, 2], [0, 0, -1], atol=1e-14)
    contact = g.grasp @ [0, 0, .1146, 1]
    np.testing.assert_allclose(contact[:3], g.source_tip[:3, 3] + [0, 0, .04])
    np.testing.assert_allclose(g.approach[:3, 3] - g.grasp[:3, 3], [0, 0, .05], atol=1e-14)
    # Preserve the non-top-down calibrated default, for both arms.
    legacy = catalog.geometry('cylinder_1', 'cylinder_3', arm)
    np.testing.assert_allclose(legacy.grasp[:3, :3], Rotation.from_euler(
        'xyz', [180, 0, -60], degrees=True).as_matrix())


def test_all_catalog_shapes_use_actual_polygon_not_row_scale():
    catalog = MarkerCatalog(CATALOG)
    for shape in catalog.catalog['variants']:
        for source in range(4):
            for destination in range(4):
                peg = catalog.compatible(f'{shape}_{source}', f'{shape}_{destination}')
                assert peg['length'] > .022
    for name, hole in [('cylinder_1', 'hex_3'), ('missing', 'cylinder_3')]:
        with pytest.raises(ValueError):
            catalog.compatible(name, hole)
    modified = copy.deepcopy(catalog.catalog['variants']['cylinder']['1'])
    modified['profile_xy'] = (np.asarray(modified['profile_xy']) * 2).tolist()
    catalog.catalog['variants']['cylinder']['1'] = modified
    with pytest.raises(ValueError, match='dimensions'):
        catalog.compatible('cylinder_1', 'cylinder_3')


def test_present_peg_source_alias_and_unsupported_grasp():
    catalog = MarkerCatalog(CATALOG)
    with pytest.raises(ValueError, match='not present'):
        catalog.geometry('cylinder_2', 'cylinder_3', 'left')
    catalog.profile['active_pegs']['hex_1'] = {'source_hole': 'hex_1'}
    with pytest.raises(ValueError, match='grasp profile'):
        catalog.geometry('hex_1', 'hex_3', 'left')
    catalog.profile['active_pegs']['cylinder_1']['source_hole'] = 'cylinder_2'
    catalog.profile['aliases']['my_peg'] = 'cylinder_1'
    g = catalog.geometry('my_peg', 'cylinder_3', 'right')
    assert g.source_hole == 'cylinder_2'
    assert g.source_tip[1, 3] == catalog.catalog['holes']['cylinder_2']['target'][1]
    with pytest.raises(ValueError, match='explicit'):
        catalog.geometry('cylinder_1', 'cylinder_3', None)
    catalog.profile['grasp_rpy_board_deg'] = [0, 90, -60]
    with pytest.raises(ValueError, match='GRASPGENX'):
        catalog.geometry('cylinder_1', 'cylinder_3', 'left')


@pytest.mark.parametrize('failure', ['missing', 'invalid', 'stale', 'future', 'frame',
                                      'quaternion', 'nan', 'stamp_mismatch', 'clock_reset'])
@pytest.mark.parametrize('use_sim_time', [False, True])
def test_observation_fail_closed(failure, use_sim_time):
    cache = BoardObservation(.5, use_sim_time=use_sim_time)
    pose, diag = observation()
    now = 10 * 10**9
    if failure == 'invalid':
        diag.status[0].level = DiagnosticStatus.WARN
    elif failure == 'stale':
        now += 10**9
    elif failure in ('future', 'clock_reset'):
        now -= 10**9
    elif failure == 'frame':
        pose.header.frame_id = 'head_color_optical_frame'
    elif failure == 'quaternion':
        pose.pose.orientation.w = 0.0
    elif failure == 'nan':
        pose.pose.position.x = float('nan')
    elif failure == 'stamp_mismatch':
        diag.status[0].values[1].value = str(9 * 10**9)
    cache.on_pose(pose)
    if failure != 'missing':
        cache.on_diagnostics(diag)
    with pytest.raises(ValueError):
        cache.matrix(now)


@pytest.mark.parametrize('use_sim_time', [False, True])
def test_explicit_invalidation_overrides_still_fresh_pose(use_sim_time):
    cache = BoardObservation(.5, use_sim_time=use_sim_time)
    pose, diag = observation()
    cache.on_diagnostics(diag)
    cache.on_pose(pose)
    np.testing.assert_allclose(cache.matrix(10 * 10**9), np.eye(4))
    _, invalid = observation(valid=False)
    cache.on_diagnostics(invalid)
    with pytest.raises(ValueError, match='invalid'):
        cache.matrix(10 * 10**9)


@pytest.mark.parametrize('failure,reason', [
    ('invalid', 'observation invalidated by invalid pose or diagnostics'),
    ('observation_age', 'stale observation ROS age'),
    ('diagnostic_age', 'stale diagnostic ROS age'),
    ('wall_age', 'stale monotonic wall receipt age'),
    ('future_observation', 'observation timestamp is in the future'),
    ('future_diagnostic', 'diagnostic timestamp is in the future'),
    ('pending_pose', 'pending future observation timestamp'),
    ('pending_diagnostic', 'pending future diagnostic timestamp'),
])
def test_observation_rejection_explains_existing_condition(failure, reason, monkeypatch):
    wall = [100.0]
    monkeypatch.setattr('h12_skills.marker_targets.time.monotonic', lambda: wall[0])
    cache = BoardObservation(.5)
    accept(cache, 10 * 10**9)
    now = 10 * 10**9
    if failure == 'invalid':
        _, diag = observation(valid=False)
        cache.on_diagnostics(diag)
    elif failure == 'observation_age':
        now += 501_000_000
    elif failure == 'diagnostic_age':
        # Isolate the existing independent diagnostic-age check.
        stamp, matrix, _, received = cache._accepted
        cache._accepted = (stamp, matrix, stamp - 600_000_000, received)
    elif failure == 'wall_age':
        wall[0] += .501
    elif failure == 'future_observation':
        now -= 1
    elif failure == 'future_diagnostic':
        stamp, matrix, _, received = cache._accepted
        cache._accepted = (stamp, matrix, stamp + 1, received)
    elif failure == 'pending_pose':
        pose, _ = observation_ns(now + 1)
        cache.on_pose(pose)
    else:
        _, diag = observation_ns(now + 1)
        cache.on_diagnostics(diag)
    with pytest.raises(ValueError) as exc:
        cache.matrix(now)
    assert str(exc.value) == 'board localization invalid, stale, future, or clock reset: ' + reason


class Harness(GraspSkill, PickPlaceSkill):
    _validated_arm = SkillsBase._validated_arm
    _timeout_sec = SkillsBase._timeout_sec

    def __init__(self):
        self._target_backend = 'marker'
        self._marker_catalog = MarkerCatalog(CATALOG)
        self._saved_marker_approach = None
        self.approach_mode = 'plan'
        self.preps, self.plans = [], []
        self.plan_frame_task_cli = 'plan'
        self.replay_frame_task_cli = 'replay'
        self._board_observation = BoardObservation(.5)
        self.now_ns = 10 * 10**9
        self.board = np.eye(4)
        self.board[:3, 3] = [.5, .2, -.05]
        self._arm_halt_reason = None
        self._last_grasp_outcome = None
        self.moves, self.commands = [], []
        self.gripper_clis = {'left': 'position'}
        self.gripper_force_clis = {'left': 'force'}
        self.gripper_close_clis = {'left': 'close'}
        self.current = matrix_to_pose(self.board @ self._marker_catalog.geometry(
            'cylinder_1', 'cylinder_3', 'left').approach)
        self.refresh()
        self._marker_reference = self.board.copy()

    def get_parameter(self, name):
        assert name == 'approach_mode'
        return SimpleNamespace(value=self.approach_mode)

    def goto_named_config(self, name, **kwargs):
        assert name == 'prep' and kwargs['plan'] and kwargs['slow_mode']
        assert not kwargs['outer_gh'].is_cancel_requested
        self.preps.append(name)
        return True

    def _send_action(self, client, goal, **kwargs):
        assert client == self.plan_frame_task_cli
        assert goal.plan and goal.slow_mode and not kwargs['outer_gh'].is_cancel_requested
        self.plans.append(copy.deepcopy(goal))
        return SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED,
                               result=SimpleNamespace(success=True))

    _note_arm_move_status = SkillsBase._note_arm_move_status

    def refresh(self):
        if self._board_observation._accepted is not None:
            self.now_ns += 1_000_000
        pose, diag = observation_ns(self.now_ns, board=self.board)
        self._board_observation.on_pose(pose)
        self._board_observation.on_diagnostics(diag)

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=self.now_ns))

    def get_logger(self):
        return Log()

    def arm_halt_reason(self):
        return self._arm_halt_reason

    def _frame_pose_in_pelvis(self, frame):
        return self.current

    def move_frame_to(self, frame, pose, **kwargs):
        self.moves.append((frame, copy.deepcopy(pose), kwargs))
        assert kwargs['duration_sec'] <= (30 if kwargs['do_plan'] or kwargs.get('client') else 8)
        self.current = copy.deepcopy(pose)
        return True

    def _stop_grip_hold(self, arm):
        pass

    def _call_service(self, client, request, name, **kwargs):
        assert not kwargs['outer_gh'].is_cancel_requested
        self.commands.append(client)
        return SimpleNamespace(success=True)

    def _marker_settle(self, run, gh):
        # No sleeps or nodes: the real settle helper is tested separately.
        if gh.is_cancel_requested or run.remaining() <= 0:
            raise ValueError('canceled or timed out')

    def query_gemini(self, *args, **kwargs):
        pytest.fail('marker flow used AI')

    segment = plan_grasp = detect_object_cloud = query_gemini


def request(pick=True):
    goal = SkillPickPlace.Goal() if pick else SkillGrasp.Goal()
    goal.target_object = 'cylinder_1'
    goal.arm = 'left'
    goal.timeout = Duration(sec=30)
    if pick:
        goal.place_target = 'cylinder_3'
    return Goal(goal)


def test_marker_pickplace_holds_preinsert_never_releases():
    node, gh = Harness(), request()
    result = node._exec_pick_place(gh)
    assert result.success and gh.terminal == 'succeeded'
    assert 'no insertion or release' in result.message
    assert node.commands == ['position', 'force', 'close']
    assert node._last_grasp_outcome.method == 'marker_catalog'
    g = node._marker_catalog.geometry('cylinder_1', 'cylinder_3', 'left')
    np.testing.assert_allclose(pose_to_matrix(node.current), node.board @ g.preinsert, atol=1e-7)
    assert all(not k['do_plan'] for _, _, k in node.moves if 'extraction' in k['label'])
    # Each explicit call assumes the operator restored the physical seated source.
    node.refresh()
    assert node._exec_pick_place(request()).success
    assert node.commands == ['position', 'force', 'close'] * 2
    assert not hasattr(node, '_marker_displaced')


def test_marker_grasp_uses_existing_schema_without_extracting():
    node, gh = Harness(), request(False)
    assert node._exec_grasp(gh).success
    assert not any('extraction' in k['label'] for _, _, k in node.moves)


def test_plan_only_does_not_measure_unexecuted_endpoint_and_replay_asserts_target():
    node, gh = Harness(), request(False)
    node.current = matrix_to_pose(np.eye(4))
    start = copy.deepcopy(node.current)
    original = node._send_action

    def plan(client, goal, **kwargs):
        assert not node.moves
        np.testing.assert_array_equal(pose_to_matrix(node.current), pose_to_matrix(start))
        assert kwargs['result_timeout'] <= 30
        return original(client, goal, **kwargs)

    node._send_action = plan
    assert node._exec_grasp(gh).success
    assert node.preps == ['prep'] and len(node.plans) == 1
    _, replay_target, replay_options = node.moves[0]
    assert replay_options['client'] == 'replay'
    np.testing.assert_array_equal(pose_to_matrix(node.plans[0].frame_targets[0]),
                                  pose_to_matrix(replay_target))
    assert not node._saved_marker_approach.board.flags.writeable
    assert not node._saved_marker_approach.target.flags.writeable


@pytest.mark.parametrize('mode', ['plan', 'reuse'])
def test_marker_replay_transport_satisfies_controller_saved_plan_contract(mode):
    # Load the actual validator without importing the controller's Pinocchio stack.
    source = (ROOT / 'core_ws/src/h12_ros2_controller/h12_ros2_controller/core/'
              'saved_frame_plan.py')
    tree = ast.parse(source.read_text(), filename=str(source))
    saved_class = next(item for item in tree.body
                       if isinstance(item, ast.ClassDef) and item.name == 'SavedFramePlan')
    validator = next(item for item in saved_class.body
                     if isinstance(item, ast.FunctionDef) and item.name == 'assert_goal')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[validator], type_ignores=[]), str(source), 'exec'), namespace)
    assert_goal = namespace['assert_goal']

    node = Harness()
    node._send_frame_move = SkillsBase._send_frame_move.__get__(node)
    plan = node._send_action
    move = node.move_frame_to
    replays = []

    def action(client, goal, feedback_cb=None, **kwargs):
        if client == node.plan_frame_task_cli:
            return plan(client, goal, **kwargs)
        assert client == node.replay_frame_task_cli
        planned = node.plans[-1]
        assert planned.plan and planned.slow_mode and not planned.relative
        record = SimpleNamespace(frame_names=tuple(planned.frame_names), targets=np.array([
            pose_to_matrix(target) for target in planned.frame_targets]))
        assert_goal(record, goal)
        assert not goal.plan and not goal.relative and not goal.slow_mode
        np.testing.assert_allclose(
            [pose_to_matrix(target) for target in goal.frame_targets], record.targets, atol=1e-14)
        for flag in ('plan', 'relative', 'slow_mode'):
            invalid = copy.deepcopy(goal)
            setattr(invalid, flag, True)
            with pytest.raises(ValueError, match='Replay accepts only'):
                assert_goal(record, invalid)
        replays.append(copy.deepcopy(goal))
        feedback_cb(SimpleNamespace(feedback=SimpleNamespace(
            errors_linear=[0.0], errors_angular=[0.0])))
        return SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED,
                               result=SimpleNamespace(success=True))

    def execute(frame, target, **kwargs):
        if kwargs.get('client') == node.replay_frame_task_cli:
            assert SkillsBase.move_frame_to(node, frame, target, **kwargs)
        else:
            assert kwargs['slow_mode']  # Ordinary contact/carry keep their slow setting.
        return move(frame, target, **kwargs)

    node._send_action = action
    node.move_frame_to = execute
    assert node._exec_grasp(request(False)).success
    if mode == 'reuse':
        node.approach_mode = mode
        node.refresh()
        assert node._exec_grasp(request(False)).success
    assert len(node.plans) == 1 and node.preps == ['prep']
    assert len(replays) == (2 if mode == 'reuse' else 1)


def test_reuse_keeps_snapshot_and_approach_with_new_destination_and_external_mode():
    node = Harness()
    assert node._exec_grasp(request(False)).success
    saved = node._saved_marker_approach
    node.approach_mode = 'reuse'
    # Operator has restored the peg and returned the controller to approach_start.
    # Skills neither sets nor reads the controller's independent control_mode.
    node.control_mode = 'counter'
    node.board[0, 3] += .01  # CV noise below the existing board jump bound, not 5 mm.
    node.refresh()
    node.moves.clear()
    gh = request()
    gh.request.place_target = 'cylinder_2'
    assert node._exec_pick_place(gh).success
    assert node.preps == ['prep'] and len(node.plans) == 1
    assert node._saved_marker_approach is saved
    np.testing.assert_allclose(pose_to_matrix(node.moves[0][1]), saved.target, atol=1e-14)
    geometry = node._marker_catalog.geometry('cylinder_1', 'cylinder_2', 'left')
    np.testing.assert_allclose(pose_to_matrix(node.current),
                               geometry.target_matrix(saved.board, 'preinsert'), atol=1e-14)


@pytest.mark.parametrize('fault', ['none', 'peg', 'source', 'arm', 'top_down', 'profile'])
def test_reuse_requires_matching_saved_source_identity_before_any_command(fault):
    node = Harness()
    if fault != 'none':
        assert node._exec_grasp(request(False)).success
    node.approach_mode = 'reuse'
    node.refresh()
    node.moves.clear()
    node.commands.clear()
    gh = request(False)
    if fault == 'peg':
        node._marker_catalog.profile['active_pegs']['cylinder_2'] = {'source_hole': 'cylinder_2'}
        gh.request.target_object = 'cylinder_2'
    elif fault == 'source':
        node._marker_catalog.profile['active_pegs']['cylinder_1']['source_hole'] = 'cylinder_2'
    elif fault == 'arm':
        gh.request.arm = 'right'
    elif fault == 'top_down':
        gh.request.top_down = True
    elif fault == 'profile':
        node._marker_catalog.profile['step_m'] *= .9
    result = node._exec_grasp(gh)
    assert not result.success and 'matching saved marker approach' in result.message
    assert not node.moves and not node.commands


@pytest.mark.parametrize('fault', ['missing', 'stale', 'invalid', 'translation', 'rotation'])
def test_reuse_requires_fresh_unchanged_workcell_before_motion(fault):
    node = Harness()
    assert node._exec_grasp(request(False)).success
    node.approach_mode = 'reuse'
    node.moves.clear()
    node.commands.clear()
    if fault == 'missing':
        node._board_observation = BoardObservation(.5)
    elif fault == 'stale':
        node.now_ns += 10**9
    elif fault == 'invalid':
        _, diag = observation_ns(node.now_ns, valid=False)
        node._board_observation.on_diagnostics(diag)
    elif fault == 'translation':
        node.board[0, 3] += node._marker_catalog.profile['max_board_jump_m'] + .001
        node.refresh()
    else:
        node.board[:3, :3] = Rotation.from_rotvec(
            [0, 0, node._marker_catalog.profile['max_board_jump_rad'] + .001]).as_matrix()
        node.refresh()
    result = node._exec_grasp(request(False))
    assert not result.success
    assert 'board not visible before replay or workcell changed' in result.message
    assert not node.moves and not node.commands and node.preps == ['prep']


def test_new_plan_without_board_fails_after_named_prep_before_approach_or_close():
    node = Harness()
    node._board_observation = BoardObservation(.5)
    result = node._exec_grasp(request(False))
    assert not result.success and 'no matched observation' in result.message
    assert node.preps == ['prep']
    assert not node.plans and not node.moves and not node.commands


def test_plan_captures_fresh_board_only_after_successful_prep():
    node = Harness()
    original = node.goto_named_config
    node._board_observation = BoardObservation(.5)

    def prep(*args, **kwargs):
        result = original(*args, **kwargs)
        node.board[0, 3] += .01
        node.refresh()
        return result

    node.goto_named_config = prep
    assert node._exec_grasp(request(False)).success
    np.testing.assert_array_equal(node._saved_marker_approach.board, node.board)


@pytest.mark.parametrize('fault', ['none', 'aborted', 'false'])
def test_plan_result_failure_does_not_execute_replay_or_close(fault):
    node = Harness()
    node._send_action = lambda *a, **k: (
        None if fault == 'none' else SimpleNamespace(
            status=(GoalStatus.STATUS_ABORTED if fault == 'aborted'
                    else GoalStatus.STATUS_SUCCEEDED),
            result=SimpleNamespace(success=fault != 'false')))
    result = node._exec_grasp(request(False))
    assert not result.success and 'pregrasp plan failed' in result.message
    assert not node.moves and node.commands == ['position']
    assert node._saved_marker_approach is None


@pytest.mark.parametrize('stage', ['prep', 'plan', 'replay', 'contact'])
@pytest.mark.parametrize('fault', ['failure', 'cancel', 'deadline', 'halt'])
def test_failed_approach_stages_never_close(stage, fault, monkeypatch):
    node, gh = Harness(), request(False)

    def fail():
        if fault == 'cancel':
            gh.is_cancel_requested = True
        elif fault == 'deadline':
            monkeypatch.setattr(_Run, 'remaining', lambda self: 0)
        elif fault == 'halt':
            node._arm_halt_reason = 'hardware stop'

    if stage == 'prep':
        def prep(*args, **kwargs):
            fail()
            return fault != 'failure'
        node.goto_named_config = prep
    elif stage == 'plan':
        def plan(*args, **kwargs):
            fail()
            return SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED,
                                   result=SimpleNamespace(success=fault != 'failure'))
        node._send_action = plan
    else:
        original = node.move_frame_to

        def move(frame, target, **kwargs):
            label = 'pregrasp' if stage == 'replay' else 'contact approach'
            hit = label in kwargs['label']
            if hit:
                fail()
                return False if fault == 'failure' else original(frame, target, **kwargs)
            return original(frame, target, **kwargs)

        node.move_frame_to = move
    assert not node._exec_grasp(gh).success
    assert 'force' not in node.commands and 'close' not in node.commands


def test_replay_controller_cache_target_rejection_propagates_without_replanning():
    node = Harness()
    assert node._exec_grasp(request(False)).success
    saved = node._saved_marker_approach
    node.approach_mode = 'reuse'
    node.refresh()
    node.commands.clear()

    def controller_reject(frame, target, **kwargs):
        assert kwargs['client'] == 'replay'
        np.testing.assert_allclose(pose_to_matrix(target), saved.target, atol=1e-14)
        node._arm_halt_reason = 'replay aborted: cached target mismatch'
        return False

    node.move_frame_to = controller_reject
    result = node._exec_grasp(request(False))
    assert not result.success and 'pregrasp waypoint not reached' in result.message
    assert len(node.plans) == 1 and node.commands == ['position']


def test_measured_contact_error_still_prevents_close_with_frozen_vision():
    node = Harness()
    settles = []

    def settle(run, gh):
        settles.append(True)
        if len(settles) == 2:
            node.current.position.x += .02

    node._marker_settle = settle
    result = node._exec_grasp(request(False))
    assert not result.success and 'measured contact pose outside tolerance' in result.message
    assert node.commands == ['position']


def test_approach_mode_is_idle_only_and_frozen_for_entire_goal():
    node = Harness()
    node._manipulation_lock = threading.Lock()

    def param(value):
        return [SimpleNamespace(name='approach_mode', value=value)]
    check = SkillsBase._check_approach_mode
    assert check(node, param('reuse')).successful
    assert not check(node, param('automatic')).successful
    with node._manipulation_lock:
        assert not check(node, param('reuse')).successful
    original = node.goto_named_config

    def prep(*args, **kwargs):
        result = original(*args, **kwargs)
        node.approach_mode = 'reuse'  # Even an injected change cannot alter this episode.
        return result

    node.goto_named_config = prep
    assert node._exec_grasp(request(False)).success
    assert len(node.plans) == 1 and node.preps == ['prep']


@pytest.mark.parametrize('failure', ['move', 'cancel', 'timeout', 'stale', 'invalid', 'jump'])
@pytest.mark.parametrize('top_down', [False, True])
def test_abort_before_close_and_never_release(failure, top_down, monkeypatch):
    node, gh = Harness(), request()
    gh.request.top_down = top_down
    original = node.move_frame_to

    def fail(frame, pose, **kwargs):
        if 'contact approach' not in kwargs['label']:
            return original(frame, pose, **kwargs)
        if failure == 'cancel':
            gh.is_cancel_requested = True
        elif failure == 'timeout':
            # The real motion guard sees the outer budget expire in-flight.
            monkeypatch.setattr(_Run, 'remaining', lambda self: 0.0)
        elif failure == 'stale':
            node.now_ns += 10**9
        elif failure == 'invalid':
            _, diag = observation(valid=False)
            node._board_observation.on_diagnostics(diag)
        elif failure == 'jump':
            node.board[0, 3] += .2
            node.refresh()
        return False

    node.move_frame_to = fail
    result = node._exec_pick_place(gh)
    assert not result.success
    assert 'close' not in node.commands
    assert node.commands == ['position']
    assert node._last_grasp_outcome is None


@pytest.mark.parametrize('failure,reason', [
    ('cancel', 'user cancellation requested'),
    ('timeout', 'skill deadline expired'),
    ('halt', 'arm halted: hardware stop'),
    ('workspace', 'target outside configured arm workspace; no motion permitted'),
    ('move', None),
])
def test_marker_move_reports_first_guard_cause(failure, reason):
    node, gh = Harness(), request(False)
    run = SimpleNamespace(remaining=lambda: 30.0)
    geometry = node._marker_catalog.geometry('cylinder_1', 'cylinder_3', 'left')

    def fail(frame, pose, **kwargs):
        guard = kwargs['outer_gh']
        if failure == 'move':
            return False  # No guard rejection: retain controller failure wording.
        if failure == 'cancel':
            gh.is_cancel_requested = True
        elif failure == 'timeout':
            run.remaining = lambda: 0.0
        elif failure == 'halt':
            node._arm_halt_reason = 'hardware stop'
        else:
            node.board[0, 3] += .04  # Below jump limit, outside arm workspace.
            node._marker_catalog.profile['arms']['left']['workspace_max_m'][0] = .1
            node.refresh()
        assert guard.is_cancel_requested
        assert guard.reason == reason
        node._arm_halt_reason = 'later stop'
        assert guard.is_cancel_requested
        assert guard.reason == reason
        return False

    node.move_frame_to = fail
    expected = (f'pregrasp canceled: {reason}' if reason is not None
                else 'pregrasp waypoint not reached')
    with pytest.raises(ValueError) as exc:
        node._marker_move('approach', geometry, run, gh, 'pregrasp', planned=True)
    assert str(exc.value) == expected + '; no subsequent close or release'
    assert node.commands == []


def test_pregrasp_guard_failure_never_closes_or_releases():
    node, gh = Harness(), request()

    def fail(frame, pose, **kwargs):
        node._arm_halt_reason = 'hardware stop'
        assert kwargs['outer_gh'].is_cancel_requested
        return False

    node.move_frame_to = fail
    result = node._exec_pick_place(gh)
    assert not result.success
    assert result.message == (
        'pregrasp canceled: arm halted: hardware stop; no subsequent close or release'
        '; no automatic retreat or release')
    assert node.commands == ['position']
    assert node._last_grasp_outcome is None


def test_valid_updated_observation_does_not_cancel_motion():
    node, gh = Harness(), request(False)
    original = node.move_frame_to

    def move(frame, pose, **kwargs):
        guard = kwargs['outer_gh']
        assert not guard.is_cancel_requested
        node.refresh()
        assert not guard.is_cancel_requested
        assert guard.reason is None
        return original(frame, pose, **kwargs)

    node.move_frame_to = move
    assert node._exec_grasp(gh).success
    assert node.commands == ['position', 'force', 'close']


@pytest.mark.parametrize('failure', ['unknown', 'absent', 'shape', 'servo', 'workspace'])
@pytest.mark.parametrize('top_down', [False, True])
def test_rejects_before_motion_or_gripper(failure, top_down):
    node, gh = Harness(), request()
    gh.request.top_down = top_down
    if failure == 'unknown':
        gh.request.place_target = 'unknown'
    elif failure == 'absent':
        gh.request.target_object = 'cylinder_2'
    elif failure == 'shape':
        gh.request.place_target = 'triangle_3'
    elif failure == 'servo':
        gh.request.visual_servo = True
    else:
        node.board[0, 3] += 2
        node.refresh()
    assert not node._exec_pick_place(gh).success
    assert not node.moves and not node.commands


@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('board_rpy', [[0, 0, 0], [2, -3, 3], [-3, 2, -3]])
@pytest.mark.parametrize('pick_place,minimum', [(False, 0.0), (False, 90.0), (True, None)])
def test_explicit_top_down_is_geometric_and_controls_only_requested_arm(
        arm, minimum, board_rpy, pick_place):
    node, gh = Harness(), request(pick_place)
    gh.request.arm, gh.request.top_down = arm, True
    if not pick_place:
        gh.request.min_downward_pitch_deg = minimum
    node.board[:3, 3] = [.58, .225 if arm == 'left' else -.225, -.05]
    node.board[:3, :3] = Rotation.from_euler('xyz', board_rpy, degrees=True).as_matrix()
    node.gripper_clis[arm] = 'position'
    node.gripper_force_clis[arm] = 'force'
    node.gripper_close_clis[arm] = 'close'
    geometry = node._marker_catalog.geometry(
        'cylinder_1', 'cylinder_3' if pick_place else 'cylinder_1', arm, top_down=True)
    node.current = matrix_to_pose(geometry.target_matrix(node.board, 'approach'))
    node.refresh()
    execute = node._exec_pick_place if pick_place else node._exec_grasp
    result = execute(gh)
    assert result.success, result.message
    assert node.commands == ['position', 'force', 'close']
    assert all(frame == f'{arm}_graspgenx_frame' for frame, _, _ in node.moves)
    np.testing.assert_allclose(pose_to_matrix(node.current),
                               geometry.target_matrix(node.board, 'preinsert' if pick_place else 'grasp'),
                               atol=1e-7)
    np.testing.assert_allclose(pose_to_matrix(node.current)[:3, :3], geometry.grasp[:3, :3], atol=1e-14)
    node.refresh()
    assert execute(Goal(copy.deepcopy(gh.request))).success
    assert node.commands == ['position', 'force', 'close'] * 2


@pytest.mark.parametrize('fault', ['tilted_profile', 'invalid_minimum', 'servo'])
def test_top_down_rejects_nonvertical_or_ai_before_any_command(fault):
    node, gh = Harness(), request(False)
    gh.request.top_down = True
    if fault == 'tilted_profile':
        node._marker_catalog.profile['top_down_rpy_pelvis_deg']['left'] = [179.999, 0, 30]
    elif fault == 'invalid_minimum':
        gh.request.min_downward_pitch_deg = float('nan')
    else:
        gh.request.visual_servo = True
    result = node._exec_grasp(gh)
    assert not result.success
    assert not node.commands and not node.moves


def test_known_left_fixture_transform_remains_outside_right_arm_admission():
    catalog = MarkerCatalog(CATALOG)
    board = np.eye(4)
    # Reference fixture world (.5,.185,.95), pelvis (-.134,-.04,.99);
    # runtime admission always uses the actual stamped board observation.
    board[:3, 3] = [.634, .225, -.04]
    geometry = catalog.geometry('cylinder_1', 'cylinder_1', 'right', top_down=True)
    np.testing.assert_allclose(geometry.target_matrix(board, 'grasp')[:3, 3],
                               [.5061622983, .2375282240, .1176], atol=1e-10)
    with pytest.raises(ValueError, match='outside configured arm workspace'):
        catalog.check_workspace(board, geometry)
    board[1, 3] = -.225
    catalog.check_workspace(board, geometry)


def test_top_down_freezes_board_position_without_rotating_pelvis_orientation():
    node, gh = Harness(), request(False)
    gh.request.top_down = True
    frozen = node.board.copy()
    original = node.move_frame_to

    def action(frame, pose, **kwargs):
        result = original(frame, pose, **kwargs)
        if kwargs.get('client') == 'replay':
            node.board[:3, :3] = Rotation.from_euler('xyz', [2, -1, 3], degrees=True).as_matrix()
            node.board[:3, 3] += [.001, -.002, .001]
            node.refresh()
        return result

    node.move_frame_to = action
    result = node._exec_grasp(gh)
    assert result.success, result.message
    assert node.commands == ['position', 'force', 'close']
    g = node._marker_catalog.geometry('cylinder_1', 'cylinder_1', 'left', top_down=True)
    np.testing.assert_allclose(pose_to_matrix(node.current), g.target_matrix(frozen, 'grasp'), atol=1e-14)
    for _, pose, _ in node.moves:
        np.testing.assert_allclose(pose_to_matrix(pose)[:3, :3], g.grasp[:3, :3], atol=1e-14)


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_top_down_tilted_board_resolves_contact_offset_and_all_waypoints(arm):
    catalog = MarkerCatalog(CATALOG)
    offset = np.array([.002, -.003, .1146])
    catalog.profile['arms'][arm]['contact_in_graspgenx_m'] = offset.tolist()
    g = catalog.geometry('cylinder_1', 'cylinder_3', arm, top_down=True)
    board = np.eye(4)
    board[:3, :3] = Rotation.from_euler('xyz', [3, -2, 3], degrees=True).as_matrix()
    board[:3, 3] = [.58, .225 if arm == 'left' else -.225, -.05]
    grasp = g.target_matrix(board, 'grasp')
    expected_contact = board @ g.source_tip @ np.array([0, 0, .04, 1])
    np.testing.assert_allclose(grasp @ np.r_[offset, 1], expected_contact, atol=1e-14)
    np.testing.assert_allclose(g.target_matrix(board, 'approach')[:3, 3] - grasp[:3, 3],
                               [0, 0, .05], atol=1e-14)
    np.testing.assert_allclose(g.target_matrix(board, 'extracted')[:3, 3] - grasp[:3, 3],
                               board[:3, :3] @ [0, 0, .06], atol=1e-14)
    grasp_to_tip = np.linalg.inv(grasp) @ board @ g.source_tip
    carried_tip = g.target_matrix(board, 'preinsert') @ grasp_to_tip
    expected_tip = board @ np.r_[np.array(catalog.catalog['holes']['cylinder_3']['entry'])
                                 + [0, 0, .04], 1]
    np.testing.assert_allclose(carried_tip[:, 3], expected_tip, atol=1e-14)
    for name in ('grasp', 'approach', 'extracted', 'preinsert', 'carry'):
        np.testing.assert_allclose(g.target_matrix(board, name)[:3, :3], g.grasp[:3, :3], atol=1e-14)
    np.testing.assert_allclose(pose_to_matrix(g.candidates(board).grasps[0].pose), grasp, atol=1e-14)
    catalog.check_workspace(board, g)


def test_legacy_waypoint_resolution_remains_board_relative_when_tilted():
    g = MarkerCatalog(CATALOG).geometry('cylinder_1', 'cylinder_3', 'left')
    board = np.eye(4)
    board[:3, :3] = Rotation.from_euler('xyz', [3, -2, 3], degrees=True).as_matrix()
    board[:3, 3] = [.58, .225, -.05]
    for name in ('grasp', 'approach', 'extracted', 'preinsert'):
        np.testing.assert_allclose(g.target_matrix(board, name), board @ getattr(g, name))
    carry = g.preinsert.copy()
    carry[2, 3] = max(g.extracted[2, 3], carry[2, 3]) + .02
    np.testing.assert_allclose(g.target_matrix(board, 'carry'), board @ carry)


def test_no_close_after_cancel_between_force_and_close():
    node, gh = Harness(), request()
    original = node._call_service

    def service(client, *args, **kwargs):
        result = original(client, *args, **kwargs)
        if client == 'force':
            gh.is_cancel_requested = True
        return result

    node._call_service = service
    assert not node._exec_pick_place(gh).success
    assert node.commands == ['position', 'force']
    assert gh.terminal == 'canceled'


def test_legacy_helpers_refuse_marker_backend_without_calls():
    node = object.__new__(SkillsBase)
    node._target_backend = 'marker'
    with pytest.raises(RuntimeError, match='disabled'):
        node.query_gemini('never sent')
    with pytest.raises(RuntimeError, match='disabled'):
        node.segment()
    with pytest.raises(RuntimeError, match='disabled'):
        node.plan_grasp(None, 'pelvis', 'magpie')


def test_perception_clients_and_waits_are_statically_legacy_only():
    tree = ast.parse((Path(__file__).parents[1] / 'h12_skills/base.py').read_text())
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    checked = 0
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        is_client = call.func.attr == 'create_client' and any(
            isinstance(arg, ast.Constant) and arg.value in ('gemini_query', 'sam_segment', 'graspgen')
            for arg in call.args)
        is_wait = (call.func.attr == 'wait_for_service' and isinstance(call.func.value, ast.Attribute)
                   and call.func.value.attr in ('gemini_cli', 'sam_cli', 'graspgen_cli'))
        if not (is_client or is_wait):
            continue
        checked += 1
        ancestor = parents.get(call)
        while ancestor is not None:
            if isinstance(ancestor, ast.If) and "self._target_backend == 'legacy'" == ast.unparse(ancestor.test):
                break
            ancestor = parents.get(ancestor)
        assert ancestor is not None, 'AI endpoint not protected by immutable backend'
    assert checked == 6


def test_overlap_reservation_and_direct_nested_composition():
    node = object.__new__(SkillsNode)
    node._manipulation_lock = threading.Lock()
    record = SimpleNamespace(set=lambda **k: None, finish=lambda **k: None)
    node.logger = SimpleNamespace(start=lambda: record)
    node._log_head_frame = lambda *a: None
    node._start_run_telemetry = lambda *a: None
    node._finish_run_telemetry = lambda *a, **k: None
    entered, leave = threading.Event(), threading.Event()

    def active(gh):
        entered.set()
        assert leave.wait(2)
        return SkillGrasp.Result(success=True)

    wrapper = node._safe_exec('grasp', SkillGrasp, active)
    thread = threading.Thread(target=wrapper, args=(request(False),))
    thread.start()
    try:
        assert entered.wait(2)
        other = request()
        result = node._safe_exec('pick_place', SkillPickPlace, active)(other)
        assert not result.success and 'reservation' in result.message
        assert other.terminal == 'aborted'
    finally:
        leave.set()
        thread.join(2)
    assert not node._manipulation_lock.locked()

    # The direct mixin call does not reacquire the outer lock or touch DDS.
    gh = request()
    run = SimpleNamespace(remaining=lambda: 120.0,
                          feedback=SkillPickPlace.Feedback())
    node._last_grasp_outcome = object()
    outcome = node._last_grasp_outcome

    def nested(inner):
        assert node._manipulation_lock.locked()
        inner.publish_feedback(SkillGrasp.Feedback(progress=.5))
        inner.succeed()
        return SkillGrasp.Result(success=True)

    node._exec_grasp = nested
    with node._manipulation_lock:
        got, err = node._grasp_via_skill(gh.request, 'legacy mug', 'left', run, gh)
    assert got is outcome and err is None and gh.terminal is None


@pytest.mark.parametrize('fault', ['ok', 'false_result', 'aborted', 'no_feedback',
                                   'empty_errors', 'large_error', 'nan'])
@pytest.mark.parametrize('replay', [False, True])
def test_marker_transport_requires_boolean_and_finite_measured_error(fault, replay):
    node = object.__new__(SkillsBase)
    node._target_backend = 'marker'
    node._arm_halt_reason = None
    node._safety_alive = True
    node.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None,
                                             warn=lambda *a, **k: None,
                                             error=lambda *a, **k: None)
    node.frame_task_cli = object()
    selected = object() if replay else node.frame_task_cli

    def action(client, goal, feedback_cb, **kwargs):
        assert client is selected
        if fault != 'no_feedback':
            lin = float('nan') if fault == 'nan' else .03 if fault == 'large_error' else .001
            feedback_cb(SimpleNamespace(feedback=SimpleNamespace(
                errors_linear=[] if fault == 'empty_errors' else [lin], errors_angular=[.001])))
        return SimpleNamespace(status=GoalStatus.STATUS_ABORTED if fault == 'aborted'
                               else GoalStatus.STATUS_SUCCEEDED,
                               result=SimpleNamespace(success=fault != 'false_result'))

    node._send_action = action
    kwargs = {'client': selected} if replay else {}
    result = node.move_frame_to('left_graspgenx_frame', matrix_to_pose(np.eye(4)), **kwargs)
    assert result == (fault == 'ok')


def test_inflight_invalidation_does_not_cancel_frozen_motion():
    node, gh = Harness(), request()
    original = node.move_frame_to
    observe = node._marker_board
    captures = []

    def capture(*args, **kwargs):
        captures.append(True)
        return observe(*args, **kwargs)

    node._marker_board = capture

    def action(frame, target, **kwargs):
        guard = kwargs['outer_gh']
        assert not guard.is_cancel_requested
        _, diag = observation(valid=False)
        node._board_observation.on_diagnostics(diag)
        assert not guard.is_cancel_requested
        return original(frame, target, **kwargs)

    node.move_frame_to = action
    assert node._exec_pick_place(gh).success
    assert node.commands == ['position', 'force', 'close']
    assert captures == [True]


@pytest.mark.parametrize('fault', ['cancel', 'stale', 'timeout', 'move'])
def test_extraction_abort_keeps_grasp_without_reclose_or_release(fault, monkeypatch):
    node, gh = Harness(), request()
    original = node.move_frame_to

    def action(frame, target, **kwargs):
        if 'extraction' not in kwargs['label']:
            return original(frame, target, **kwargs)
        if fault == 'cancel':
            gh.is_cancel_requested = True
        elif fault == 'stale':
            node.now_ns += 10**9
        elif fault == 'timeout':
            monkeypatch.setattr(_Run, 'remaining', lambda self: 0.0)
        return False

    node.move_frame_to = action
    assert not node._exec_pick_place(gh).success
    assert node.commands == ['position', 'force', 'close']
    assert node._last_grasp_outcome is None


def test_settle_checks_real_cancel_deadline_and_halt_not_marker_visibility():
    node, gh = Harness(), request()
    # Harness overrides settle for fast flow tests. Test the actual method here.
    from h12_skills.skills.marker import MarkerManipulationSkill
    actual_settle = MarkerManipulationSkill._marker_settle
    gh.is_cancel_requested = True
    with pytest.raises(ValueError):
        actual_settle(node, SimpleNamespace(remaining=lambda: 10), gh)
    gh.is_cancel_requested = False
    with pytest.raises(ValueError):
        actual_settle(node, SimpleNamespace(remaining=lambda: 0), gh)
    _, diag = observation(valid=False)
    node._board_observation.on_diagnostics(diag)
    actual_settle(node, SimpleNamespace(remaining=lambda: 10), gh)
    node._arm_halt_reason = 'hardware stop'
    with pytest.raises(ValueError):
        actual_settle(node, SimpleNamespace(remaining=lambda: 10), gh)


def test_wall_stale_diagnostics_and_malformed_stamp_fail_closed(monkeypatch):
    cache = BoardObservation(.5)
    pose, diag = observation()
    wall_time = [time.monotonic()]
    monkeypatch.setattr('h12_skills.marker_targets.time.monotonic', lambda: wall_time[0])
    cache.on_pose(pose)
    cache.on_diagnostics(diag)
    wall_time[0] += 1
    with pytest.raises(ValueError):
        cache.matrix(10 * 10**9)
    diag.status[0].values[1].value = 'bad stamp'
    cache.on_diagnostics(diag)
    with pytest.raises(ValueError):
        cache.matrix(10 * 10**9)


def test_default_legacy_branch_is_retained():
    node, gh = Harness(), request(False)
    del node._target_backend
    node.get_parameter = lambda name: SimpleNamespace(value='graspgenx')
    node.goto_named_config = lambda *a, **k: False
    node._exec_marker = lambda *a, **k: pytest.fail('default called marker backend')
    result = node._exec_grasp(gh)
    assert not result.success and "'prep'" in result.message
    assert not node.commands and not node.moves


def test_late_action_acceptance_is_canceled_not_orphaned():
    node = object.__new__(SkillsBase)
    node.get_logger = lambda: Log()
    node._wait_future = lambda *args: None
    future = Future()
    client = SimpleNamespace(send_goal_async=lambda *a, **k: future)
    canceled = []
    assert node._send_action(client, object()) is None
    future.set_result(SimpleNamespace(accepted=True,
                                      cancel_goal_async=lambda: canceled.append(True)))
    assert canceled == [True]


def observation_ns(stamp_ns, board=None, valid=True):
    pose, diag = observation(stamp=stamp_ns // 10**9, board=board, valid=valid)
    pose.header.stamp.nanosec = stamp_ns % 10**9
    diag.header.stamp.nanosec = stamp_ns % 10**9
    diag.status[0].values[1].value = str(stamp_ns)
    return pose, diag


def accept(cache, stamp_ns, board=None):
    pose, diag = observation_ns(stamp_ns, board)
    cache.on_pose(pose)
    cache.on_diagnostics(diag)
    return pose, diag


@pytest.mark.parametrize('use_sim_time', [False, True])
def test_marker_cache_initialization_uses_node_sim_time_parameter(use_sim_time):
    source = ast.parse((Path(__file__).parents[1] / 'h12_skills/base.py').read_text())
    assignment = next(statement for statement in ast.walk(source)
                      if isinstance(statement, ast.Assign)
                      and any(isinstance(target, ast.Attribute) and target.attr == '_board_observation'
                              for target in statement.targets))
    node = SimpleNamespace(
        _marker_catalog=SimpleNamespace(profile={'max_pose_age_sec': .5}),
        get_parameter=lambda name: SimpleNamespace(value=use_sim_time)
        if name == 'use_sim_time' else pytest.fail('unexpected parameter'))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<cache initialization>', 'exec'),
         {'self': node, 'BoardObservation': BoardObservation})
    assert node._board_observation.use_sim_time is use_sim_time
    assert node._board_observation.max_age == .5


@pytest.mark.parametrize('use_sim_time', [False, True])
def test_slow_simulation_freshness_uses_selected_clock(use_sim_time, monkeypatch):
    wall = [100.0]
    monkeypatch.setattr('h12_skills.marker_targets.time.monotonic', lambda: wall[0])
    cache = BoardObservation(.5, use_sim_time=use_sim_time)
    accept(cache, 10 * 10**9)
    wall[0] += .81
    if use_sim_time:
        np.testing.assert_allclose(cache.matrix(10_363_000_000), np.eye(4))
    else:
        with pytest.raises(ValueError, match='stale'):
            cache.matrix(10_363_000_000)
    with pytest.raises(ValueError, match='stale'):
        cache.matrix(10_501_000_000)


def test_paused_simulation_does_not_expire_observation_in_wall_time(monkeypatch):
    wall = [100.0]
    monkeypatch.setattr('h12_skills.marker_targets.time.monotonic', lambda: wall[0])
    cache = BoardObservation(.5, use_sim_time=True)
    accept(cache, 10 * 10**9)
    cache.matrix(10 * 10**9)
    wall[0] += 60
    np.testing.assert_allclose(cache.matrix(10 * 10**9), np.eye(4))


@pytest.mark.parametrize('pose_first', [True, False])
@pytest.mark.parametrize('use_sim_time', [False, True])
def test_unmatched_good_update_preserves_latest_matched_pair(pose_first, use_sim_time):
    cache = BoardObservation(.5, use_sim_time=use_sim_time)
    accept(cache, 10 * 10**9)
    board = np.eye(4)
    board[0, 3] = .01
    pose, diag = observation_ns(10_100_000_000, board)
    first, second = ((lambda: cache.on_pose(pose), lambda: cache.on_diagnostics(diag))
                     if pose_first else
                     (lambda: cache.on_diagnostics(diag), lambda: cache.on_pose(pose)))
    first()
    np.testing.assert_allclose(cache.matrix(10_200_000_000), np.eye(4))
    second()
    np.testing.assert_allclose(cache.matrix(10_200_000_000), board)


def test_old_heartbeat_between_new_pose_and_diagnostics_does_not_abort():
    cache = BoardObservation(.5)
    _, heartbeat = accept(cache, 10 * 10**9)
    pose, diag = observation_ns(10_100_000_000)
    cache.on_pose(pose)
    heartbeat.header.stamp.nanosec = 120_000_000
    cache.on_diagnostics(heartbeat)
    np.testing.assert_allclose(cache.matrix(10_150_000_000), np.eye(4))
    cache.on_diagnostics(diag)
    # Publication time is newer, but this invalid heartbeat is for an OLD observation.
    heartbeat.status[0].level = DiagnosticStatus.WARN
    heartbeat.status[0].values[0].value = 'false'
    cache.on_diagnostics(heartbeat)
    np.testing.assert_allclose(cache.matrix(10_150_000_000), np.eye(4))


@pytest.mark.parametrize('use_sim_time', [False, True])
def test_new_invalid_observation_blocks_late_old_pairs_and_same_stamp_revalidation(use_sim_time):
    cache = BoardObservation(.5, use_sim_time=use_sim_time)
    old_pose, old_diag = accept(cache, 10 * 10**9)
    _, invalid = observation_ns(10_200_000_000, valid=False)
    cache.on_diagnostics(invalid)
    with pytest.raises(ValueError, match='invalid'):
        cache.matrix(10_300_000_000)
    cache.on_pose(old_pose)
    cache.on_diagnostics(old_diag)
    accept(cache, 10_100_000_000)
    accept(cache, 10_200_000_000)
    with pytest.raises(ValueError, match='invalid'):
        cache.matrix(10_300_000_000)
    accept(cache, 10_250_000_000)
    cache.on_diagnostics(invalid)
    np.testing.assert_allclose(cache.matrix(10_300_000_000), np.eye(4))


@pytest.mark.parametrize('pose_first', [True, False])
def test_pending_and_duplicate_packets_cannot_extend_original_pair_age(pose_first, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('h12_skills.marker_targets.time.monotonic', lambda: clock[0])
    cache = BoardObservation(.5)
    old_pose, old_diag = accept(cache, 10 * 10**9)
    clock[0] += .3
    pose, diag = observation_ns(10_100_000_000)
    if pose_first:
        cache.on_pose(pose)
    else:
        cache.on_diagnostics(diag)
    cache.on_pose(old_pose)
    old_diag.header.stamp.nanosec = 300_000_000
    cache.on_diagnostics(old_diag)
    np.testing.assert_allclose(cache.matrix(10_300_000_000), np.eye(4))
    clock[0] += .21
    with pytest.raises(ValueError, match='stale'):
        cache.matrix(10_400_000_000)  # ROS age is fine; wall receipt age is not.


def test_pending_missing_diagnostic_does_not_extend_ros_age():
    cache = BoardObservation(.5)
    accept(cache, 10 * 10**9)
    pose, _ = observation_ns(10_400_000_000)
    cache.on_pose(pose)
    with pytest.raises(ValueError, match='stale'):
        cache.matrix(10_600_000_000)


def test_same_stamp_heartbeat_and_pose_cannot_renew_wall_lease_with_frozen_ros(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('h12_skills.marker_targets.time.monotonic', lambda: clock[0])
    cache = BoardObservation(.5)
    pose, diag = accept(cache, 10 * 10**9)
    np.testing.assert_allclose(cache.matrix(10 * 10**9), np.eye(4))
    clock[0] = 160.0
    cache.on_diagnostics(diag)
    cache.on_pose(pose)
    with pytest.raises(ValueError, match='stale'):
        cache.matrix(10 * 10**9)
    accept(cache, 10_100_000_000)
    np.testing.assert_allclose(cache.matrix(10_100_000_000), np.eye(4))


@pytest.mark.parametrize('kind', ['pose', 'diagnostic'])
@pytest.mark.parametrize('use_sim_time', [False, True])
def test_pending_future_stamp_still_fails_closed(kind, use_sim_time):
    cache = BoardObservation(.5, use_sim_time=use_sim_time)
    accept(cache, 10 * 10**9)
    pose, diag = observation_ns(10_300_000_000)
    if kind == 'pose':
        cache.on_pose(pose)
    else:
        cache.on_diagnostics(diag)
    with pytest.raises(ValueError, match='future'):
        cache.matrix(10_200_000_000)


@pytest.mark.parametrize('bad_field', ['frame', 'quaternion', 'nan'])
def test_new_bad_pose_invalidates_but_old_bad_pose_cannot_poison_fresh_pair(bad_field):
    cache = BoardObservation(.5)
    accept(cache, 10 * 10**9)
    pose, _ = observation_ns(10_100_000_000)
    if bad_field == 'frame':
        pose.header.frame_id = 'camera'
    elif bad_field == 'quaternion':
        pose.pose.orientation.w = 0.0
    else:
        pose.pose.position.x = float('nan')
    cache.on_pose(pose)
    with pytest.raises(ValueError, match='invalid'):
        cache.matrix(10_200_000_000)
    accept(cache, 10_150_000_000)
    cache.on_pose(pose)
    np.testing.assert_allclose(cache.matrix(10_200_000_000), np.eye(4))


def test_pending_caches_bounded_independently_and_do_not_evict_accepted_pair():
    cache = BoardObservation(.5)
    accept(cache, 10 * 10**9)
    for n in range(1, 30):
        pose, diag = observation_ns(10 * 10**9 + n * 10**6)
        cache.on_pose(pose)
        _, unmatched_diag = observation_ns(10_100_000_000 + n * 10**6)
        cache.on_diagnostics(unmatched_diag)
        assert len(cache._poses) <= cache.PENDING_LIMIT
        assert len(cache._diagnostics) <= cache.PENDING_LIMIT
    np.testing.assert_allclose(cache.matrix(10_200_000_000), np.eye(4))
    cache.on_diagnostics(diag)
    assert cache._accepted[0] == 10_029_000_000


@pytest.mark.parametrize('use_sim_time', [False, True])
def test_clock_reset_requires_new_epoch_pair(use_sim_time):
    cache = BoardObservation(.5, use_sim_time=use_sim_time)
    accept(cache, 10 * 10**9)
    cache.matrix(10 * 10**9)
    with pytest.raises(ValueError, match='clock reset'):
        cache.matrix(9 * 10**9)
    with pytest.raises(ValueError):
        cache.matrix(9 * 10**9)
    accept(cache, 9 * 10**9)
    np.testing.assert_allclose(cache.matrix(9 * 10**9), np.eye(4))


def test_callback_message_mutation_and_returned_matrix_do_not_mutate_accepted_pair():
    cache = BoardObservation(.5)
    pose, diag = accept(cache, 10 * 10**9)
    pose.pose.position.x = float('nan')
    diag.header.stamp.sec = 20
    result = cache.matrix(10 * 10**9)
    result[0, 3] = 999
    np.testing.assert_allclose(cache.matrix(10 * 10**9), np.eye(4))


def test_healthy_interleaved_updates_do_not_trigger_motion_guard_cancel():
    node, gh = Harness(), request()
    original = node.move_frame_to

    def action(frame, target, **kwargs):
        guard = kwargs['outer_gh']
        assert not guard.is_cancel_requested
        node.now_ns += 1_000_000
        pose, diag = observation_ns(node.now_ns, node.board)
        node._board_observation.on_pose(pose)
        assert not guard.is_cancel_requested
        node._board_observation.on_diagnostics(diag)
        assert not guard.is_cancel_requested
        return original(frame, target, **kwargs)

    node.move_frame_to = action
    assert node._exec_pick_place(gh).success
    assert node.commands == ['position', 'force', 'close']


def test_callback_and_worker_threads_observe_atomic_matched_snapshots():
    cache = BoardObservation(5.0)
    accept(cache, 10 * 10**9)
    ready, read = threading.Event(), threading.Event()

    def publish():
        for n in range(1, 61):
            board = np.eye(4)
            board[:2, 3] = [n / 1000, n / 500]
            pose, diag = observation_ns(10 * 10**9 + n * 10**6, board)
            callbacks = [(cache.on_pose, pose), (cache.on_diagnostics, diag)]
            if n % 2:
                callbacks.reverse()
            for callback, msg in callbacks:
                callback(msg)
                ready.set()
                assert read.wait(5)
                read.clear()

    def consume():
        for n in range(1, 61):
            for expected in (n - 1, n):
                assert ready.wait(5)
                matrix = cache.matrix(10_100_000_000)
                assert matrix[0, 3] == pytest.approx(expected / 1000)
                assert matrix[1, 3] == pytest.approx(expected / 500)
                matrix[0, 3] = -100  # Returned storage belongs to this worker only.
                ready.clear()
                read.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer, reader = pool.submit(publish), pool.submit(consume)
        writer.result(timeout=10)
        reader.result(timeout=10)
