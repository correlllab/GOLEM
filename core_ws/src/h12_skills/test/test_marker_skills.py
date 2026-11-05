"""Real ROS message types, mocked skill endpoints; never initialize ROS/DDS."""

import ast
import copy
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
    def info(self, *args):
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
def test_observation_fail_closed(failure):
    cache = BoardObservation(.5)
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


def test_explicit_invalidation_overrides_still_fresh_pose():
    cache = BoardObservation(.5)
    pose, diag = observation()
    cache.on_diagnostics(diag)
    cache.on_pose(pose)
    np.testing.assert_allclose(cache.matrix(10 * 10**9), np.eye(4))
    _, invalid = observation(valid=False)
    cache.on_diagnostics(invalid)
    with pytest.raises(ValueError, match='invalid'):
        cache.matrix(10 * 10**9)


class Harness(GraspSkill, PickPlaceSkill):
    _validated_arm = SkillsBase._validated_arm
    _timeout_sec = SkillsBase._timeout_sec

    def __init__(self):
        self._target_backend = 'marker'
        self._marker_catalog = MarkerCatalog(CATALOG)
        self._marker_displaced = set()
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
        assert kwargs['duration_sec'] <= (30 if kwargs['do_plan'] else 8)
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
        self._marker_board()

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
    # A consumed seated-source assumption cannot be silently reused.
    commands = node.commands.copy()
    assert not node._exec_pick_place(request()).success
    assert node.commands == commands


def test_marker_grasp_uses_existing_schema_without_extracting():
    node, gh = Harness(), request(False)
    assert node._exec_grasp(gh).success
    assert not any('extraction' in k['label'] for _, _, k in node.moves)


@pytest.mark.parametrize('failure', ['move', 'cancel', 'timeout', 'stale', 'invalid', 'jump'])
def test_abort_before_close_and_never_release(failure, monkeypatch):
    node, gh = Harness(), request()
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


@pytest.mark.parametrize('failure', ['unknown', 'absent', 'shape', 'servo', 'override', 'workspace'])
def test_rejects_before_motion_or_gripper(failure):
    node, gh = Harness(), request()
    if failure == 'unknown':
        gh.request.place_target = 'unknown'
    elif failure == 'absent':
        gh.request.target_object = 'cylinder_2'
    elif failure == 'shape':
        gh.request.place_target = 'triangle_3'
    elif failure == 'servo':
        gh.request.visual_servo = True
    elif failure == 'override':
        gh.request.top_down = True
    else:
        node.board[0, 3] += 2
        node.refresh()
    assert not node._exec_pick_place(gh).success
    assert not node.moves and not node.commands


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
def test_marker_transport_requires_boolean_and_finite_measured_error(fault):
    node = object.__new__(SkillsBase)
    node._target_backend = 'marker'
    node._arm_halt_reason = None
    node._safety_alive = True
    node.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None,
                                             warn=lambda *a, **k: None,
                                             error=lambda *a, **k: None)
    node.frame_task_cli = object()

    def action(client, goal, feedback_cb, **kwargs):
        if fault != 'no_feedback':
            lin = float('nan') if fault == 'nan' else .03 if fault == 'large_error' else .001
            feedback_cb(SimpleNamespace(feedback=SimpleNamespace(
                errors_linear=[] if fault == 'empty_errors' else [lin], errors_angular=[.001])))
        return SimpleNamespace(status=GoalStatus.STATUS_ABORTED if fault == 'aborted'
                               else GoalStatus.STATUS_SUCCEEDED,
                               result=SimpleNamespace(success=fault != 'false_result'))

    node._send_action = action
    assert node.move_frame_to('left_graspgenx_frame', matrix_to_pose(np.eye(4))) == (fault == 'ok')


def test_inflight_invalidation_requests_inner_motion_cancel():
    node, gh = Harness(), request()

    def action(frame, target, **kwargs):
        guard = kwargs['outer_gh']
        assert not guard.is_cancel_requested
        _, diag = observation(valid=False)
        node._board_observation.on_diagnostics(diag)
        assert guard.is_cancel_requested
        return False

    node.move_frame_to = action
    assert not node._exec_pick_place(gh).success
    assert node.commands == ['position']


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


def test_settle_checks_real_cancel_deadline_and_invalid_observation():
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
    with pytest.raises(ValueError, match='invalid'):
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


@pytest.mark.parametrize('pose_first', [True, False])
def test_unmatched_good_update_preserves_latest_matched_pair(pose_first):
    cache = BoardObservation(.5)
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


def test_new_invalid_observation_blocks_late_old_pairs_and_same_stamp_revalidation():
    cache = BoardObservation(.5)
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
def test_pending_future_stamp_still_fails_closed(kind):
    cache = BoardObservation(.5)
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


def test_clock_reset_requires_new_epoch_pair():
    cache = BoardObservation(.5)
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
