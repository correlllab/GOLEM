"""Check frame_task_server's goal loops against a scripted controller, without ROS.

FakeController answers each IK step, hold step and error read from a script and advances a
fake robot clock, so every path of /grasp_frame_target runs in exact robot time: direct and
planned moves, the IK's catch-up steps, the hold and its settling, cancels, timeouts, a hold
pinned at a limit, plan and execution failures, the stale-state abort and refused goals, and
the path it can stop. Each test reads the trace of what the server did, in order.
"""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

# s of robot time: a control period, the work each step takes inside it and a goal's timeout,
# all exact in binary so the loops' iteration counts are exact
DT = 0.125
WORK = 0.03125
TIMEOUT = 1.0
ITERATIONS = 8  # TIMEOUT / DT
FAR = 1.0  # an error well outside every threshold
NEAR = 0.0  # an error inside every threshold


class FakeClock:
    """Robot time that moves only when the controller works or the loop sleeps."""

    def __init__(self, trace):
        """Start at 0 s and record every sleep in trace."""
        self.t = 0.0
        self._trace = trace

    def now(self):
        """Return robot time, s."""
        return self.t

    def sleep(self, seconds):
        """Advance robot time by seconds, recording the sleep."""
        self._trace.append(('sleep', seconds))
        self.t += seconds


def _scripted(script, index):
    """Return script's value at index, its last value past its end."""
    return script[min(index, len(script) - 1)]


class FakeController:
    """Answer the server's calls from scripts and record each call in the trace.

    ik: the largest |joint velocity| of each IK step, rad/s.
    hold: what each steady_state_step returns.
    limited: the joints pinned at a limit after each hold step (a list of (name, q, limit)).
    errors: the goal's error at each read: each grasp frame's linear and angular error.
    plan_error, execute_error: raised by the plan or by execute_path.
    stale_at: the IK or hold step (1-based, both counted) that raises StaleStateError.
    work: s of robot time each IK or hold step takes.
    path: s of robot time a planned path takes to execute.
    """

    def __init__(self, trace, *, ik=(FAR,), hold=(False,), limited=((),), errors=(FAR,),
                 plan_error=None, execute_error=None, stale_at=None, work=WORK, path=0.5):
        """Keep the scripts; every count starts at 0."""
        self.trace = trace
        self.clock = FakeClock(trace)
        self.dt = DT
        self._work_seconds = work
        self._path_seconds = path
        self._ik, self._hold, self._limited, self._errors = ik, hold, limited, errors
        self._plan_error, self._execute_error, self._stale_at = (
            plan_error, execute_error, stale_at)
        self.ik_steps = self.hold_steps = self.error_reads = 0
        self.steady_state_limited_joints = []
        self._bias_joints = None

    @property
    def bias_joints(self):
        """Return the joints whose torque bias the goal changes, or None for every joint."""
        return self._bias_joints

    @bias_joints.setter
    def bias_joints(self, value):
        """Record the joints whose torque bias the goal changes."""
        self.trace.append(('bias_joints', value))
        self._bias_joints = value

    def arm_joint_ids(self, arms):
        """Return the arms' joint ids: the left arm's are 0 and 1, the right arm's 2 and 3."""
        return sorted(joint for arm in arms for joint in {'left': (0, 1), 'right': (2, 3)}[arm])

    def _work(self, name):
        """Record a step, spend its work of robot time on it, and raise at stale_at."""
        self.trace.append((name, self.ik_steps + self.hold_steps))
        self.clock.t += self._work_seconds
        if self._stale_at is not None and self.ik_steps + self.hold_steps >= self._stale_at:
            from golem_upper_ik.core.ros_io import StaleStateError
            raise StaleStateError('/lowstate is 80 ms stale (> 50 ms); holding the last command')

    def control_step_reduced(self):
        """Step the IK on the frame tasks; return the step's joint velocity."""
        self.ik_steps += 1
        self._work('ik_step')
        return np.array([0.0, -_scripted(self._ik, self.ik_steps - 1)])

    def steady_state_step(self):
        """Step the hold; return whether it converged, pinning the scripted joints."""
        self.hold_steps += 1
        self._work('hold_step')
        self.steady_state_limited_joints = list(
            _scripted(self._limited, self.hold_steps - 1))
        return bool(_scripted(self._hold, self.hold_steps - 1))

    def _error(self):
        """Return the next scripted error."""
        self.error_reads += 1
        return _scripted(self._errors, self.error_reads - 1)

    def frame_pose_error(self, frame_name, target):
        """Return a grasp frame's error: the scripted value, m and rad."""
        error = self._error()
        return error, error

    def _call(self, *entry):
        """Record a call."""
        self.trace.append(entry)

    def init_steady_state(self):
        """Start the hold."""
        self._call('init_steady_state')

    def stop_reference(self, joint_ids):
        """Return a stand-in for the joints' resting offsets, recording the call."""
        self._call('stop_reference', list(joint_ids))
        return ('reference', tuple(joint_ids))

    def stop_joints(self, joint_ids, reference=None):
        """Stop the joints where they are."""
        self._call('stop_joints', list(joint_ids), reference)

    def check_state_fresh(self):
        """Pass the freshness check."""
        self._call('check_state_fresh')

    def clear_frame_tasks(self):
        """Drop the frame tasks."""
        self._call('clear_frame_tasks')

    def add_frame_task(self, task_name, frame_name, target):
        """Add a frame task."""
        self._call('add_frame_task', task_name, frame_name)

    def update_ik_solver(self):
        """Re-seed the IK."""
        self._call('update_ik_solver')

    def plan_to_ik_target(self, ik_timeout):
        """Record a plan to the tasks' IK solution; return 3 waypoints or raise plan_error."""
        self._call('plan_to_ik_target', ik_timeout)
        if self._plan_error is not None:
            raise self._plan_error
        return [np.zeros(2)] * 3

    def execute_path(self, path, stop=None):
        """Run a path for its s of robot time, or raise execute_error.

        With stop, each waypoint takes its share of the time after stop() says go on; a true
        stop() ends the path there, recorded, and returns False.
        """
        self._call('execute_path', len(path))
        if stop is None:
            self.clock.t += self._path_seconds
        else:
            for index in range(len(path)):
                if stop():
                    self._call('path_stopped', index)
                    return False
                self.clock.t += self._path_seconds / len(path)
        if self._execute_error is not None:
            raise self._execute_error
        return True


class FakeLogger:
    """Record every log line in the trace as (level, message)."""

    def __init__(self, trace):
        """Record in trace."""
        self._trace = trace

    def info(self, message):
        """Record an info line."""
        self._trace.append(('info', message))

    def warn(self, message):
        """Record a warning."""
        self._trace.append(('warn', message))

    def error(self, message):
        """Record an error."""
        self._trace.append(('error', message))


class FakeGoalHandle:
    """Hand the server a goal; cancel from the cancel_at-th check on (1-based)."""

    def __init__(self, trace, request, cancel_at=None):
        """Keep the request; no status yet."""
        self.request = request
        self._trace = trace
        self._cancel_at = cancel_at
        self.checks = 0
        self.status = None

    @property
    def is_cancel_requested(self):
        """Return whether a cancel has been requested, recording the check."""
        self.checks += 1
        self._trace.append(('cancel?', self.checks))
        return self._cancel_at is not None and self.checks >= self._cancel_at

    def publish_feedback(self, feedback):
        """Record a feedback message's values."""
        self._trace.append(('feedback', feedback.left_error_m, feedback.left_error_rad,
                            feedback.right_error_m, feedback.right_error_rad))

    def _end(self, status):
        """Record the goal's end."""
        assert self.status is None, f'{status} after {self.status}'
        self.status = status
        self._trace.append(('status', status))

    def succeed(self):
        """End the goal succeeded."""
        self._end('succeeded')

    def abort(self):
        """End the goal aborted."""
        self._end('aborted')

    def canceled(self):
        """End the goal canceled."""
        self._end('canceled')


def server_of(controller, trace):
    """Return a FrameTaskServer on controller, built without a ROS node.

    Its attributes mirror those FrameTaskServer.__init__ and start() set that the goal paths
    read (the profile, the thresholds, the controller, the goal's frames and the lock); a new
    one the goal paths read must be added here too.
    """
    from golem_upper_ik.ros2.frame_task_server import FrameTaskServer
    server = FrameTaskServer.__new__(FrameTaskServer)
    logger = FakeLogger(trace)
    server.get_logger = lambda: logger
    server.robot = 'h12'
    server.profile = SimpleNamespace(
        grasp_frames=(SimpleNamespace(name='left_grasp'), SimpleNamespace(name='right_grasp')))
    server.controller = controller
    server.timeout = TIMEOUT
    server.threshold_ik = 1e-3
    server.threshold_linear = 0.002
    server.threshold_angular = 0.01
    server.frame_names = []
    server.frame_targets = []
    server._controller_lock = threading.Lock()
    return server


def grasp_request(arms=('left',), plan=False, timeout=0.0, frame_id='pelvis'):
    """Return a GraspFrameTarget goal with a target for arms, in frame_id; timeout in s."""
    from custom_ros_messages.action import GraspFrameTarget
    goal = GraspFrameTarget.Goal()
    for arm in arms:
        pose = getattr(goal, arm)
        pose.header.frame_id = frame_id
        pose.pose.position.x = 0.3
        pose.pose.orientation.w = 1.0
    goal.plan = plan
    goal.timeout.sec = int(timeout)
    return goal


def run_goal(plan=False, cancel_at=None, request=None, **script):
    """Run one grasp goal through the server; return (trace, goal handle, result, controller)."""
    trace = []
    controller = FakeController(trace, **script)
    server = server_of(controller, trace)
    if request is None:
        request = grasp_request(plan=plan)
    handle = FakeGoalHandle(trace, request, cancel_at)
    result = server.grasp_frame_target_callback(handle)
    return trace, handle, result, controller


def entries(trace, *kinds):
    """Return the trace's entries whose first field is one of kinds."""
    return [entry for entry in trace if entry[0] in kinds]


def steps(trace):
    """Return the kind of each IK or hold step, in order."""
    return [entry[0] for entry in trace if entry[0] in ('ik_step', 'hold_step')]


def ik_steps_per_iteration(trace):
    """Return how many IK steps each loop iteration ran, an iteration ending at its sleep."""
    counts, count = [], 0
    for entry in trace:
        if entry[0] == 'ik_step':
            count += 1
        elif entry[0] == 'sleep':
            counts.append(count)
            count = 0
    return counts


def outcome(name):
    """Return the GraspFrameTarget result's outcome constant called name."""
    from custom_ros_messages.action import GraspFrameTarget
    return getattr(GraspFrameTarget.Result, name)


def messages(trace, level):
    """Return the log messages of a level."""
    return [entry[1] for entry in trace if entry[0] == level]


# robot-time periods of the settling window: the reached reads a direct goal needs in a row
SETTLE_READS = 3  # reached at t, t + DT and t + 2 * DT >= t + GRASP_SETTLE_S


def test_grasp_settling_time_is_whole_periods():
    """The scripted loops' settling takes SETTLE_READS reached reads at DT."""
    from golem_upper_ik.ros2.frame_task_server import GRASP_SETTLE_S
    assert (SETTLE_READS - 2) * DT < GRASP_SETTLE_S <= (SETTLE_READS - 1) * DT


def test_grasp_goal_succeeds_once_the_targets_have_settled():
    """A grasp goal succeeds only after its targets stay reached for GRASP_SETTLE_S.

    A read outside the thresholds restarts the window; the hold step's own convergence ends
    nothing.
    """
    errors = (FAR, FAR, NEAR, FAR) + (NEAR,) * SETTLE_READS
    trace, handle, result, _ = run_goal(ik=(FAR, NEAR), hold=(True,), errors=errors)
    assert steps(trace) == ['ik_step', 'ik_step'] + ['hold_step'] * 5
    assert handle.status == 'succeeded' and result.success
    assert result.outcome == outcome('REACHED')
    assert result.message == 'reached (left 0.0 mm 0.0000 rad, right 0.0 mm 0.0000 rad)'
    feedback = entries(trace, 'feedback')
    assert len(feedback) == 7
    # the arm without a target reports 0
    assert feedback[0] == ('feedback', FAR, FAR, 0.0, 0.0)


def test_grasp_goal_drives_only_the_targeted_arms():
    """A grasp goal replaces the frame tasks with one on each targeted grasp frame."""
    trace, _, _, _ = run_goal(errors=(NEAR,))
    assert entries(trace, 'add_frame_task') == [
        ('add_frame_task', 'left_grasp_task', 'left_grasp')]
    names = [entry[0] for entry in trace]
    assert names.index('check_state_fresh') < names.index('clear_frame_tasks') < names.index(
        'add_frame_task') < names.index('update_ik_solver')
    trace, _, _, _ = run_goal(request=grasp_request(arms=('left', 'right')),
                              errors=(NEAR,))
    assert [entry[2] for entry in entries(trace, 'add_frame_task')] == [
        'left_grasp', 'right_grasp']


@pytest.mark.parametrize('arms', [('left',), ('right',), ('left', 'right')])
@pytest.mark.parametrize('ending', [
    {}, {'ik': (NEAR,), 'errors': (NEAR,)}, {'cancel_at': 2}, {'stale_at': 2},
    {'plan': True, 'plan_error': RuntimeError('no path found')}])
def test_grasp_goal_changes_only_the_targeted_arms_bias(arms, ending):
    """A grasp goal changes the torque bias of its targeted arms' joints alone, until it ends.

    However it ends (timed out, reached, canceled, a stale state, a failed plan), the bias
    joints are set before the IK is seeded and cleared after its last step.
    """
    trace, handle, _, _ = run_goal(request=grasp_request(
        arms=arms, plan=ending.get('plan', False)),
        **{key: value for key, value in ending.items() if key != 'plan'})
    joints = sorted(joint for arm in arms for joint in {'left': (0, 1), 'right': (2, 3)}[arm])
    assert entries(trace, 'bias_joints') == [('bias_joints', joints), ('bias_joints', None)]
    names = [entry[0] for entry in trace]
    set_at, cleared_at = [i for i, name in enumerate(names) if name == 'bias_joints']
    assert set_at < names.index('update_ik_solver')
    assert all(set_at < i < cleared_at for i, name in enumerate(names)
               if name in ('ik_step', 'hold_step', 'execute_path'))
    assert handle.status is not None


def test_grasp_goal_aborts_at_its_timeout():
    """A grasp goal whose targets are never reached aborts at its timeout, stopping its arms.

    A moving arm lags its command, so the timeout stops it where it is, as a cancel does.
    """
    trace, handle, result, _ = run_goal(ik=(NEAR,), hold=(True,))
    assert handle.status == 'aborted' and not result.success
    assert result.outcome == outcome('NOT_REACHED')
    assert steps(trace) == ['ik_step'] + ['hold_step'] * (ITERATIONS - 1)
    assert result.message.startswith('timed out after 1.0 s; the arms hold where they are '
                                     '(left 1000.0 mm 1.0000 rad')
    names = [entry[0] for entry in trace]
    assert entries(trace, 'stop_joints') == [('stop_joints', [0, 1], ('reference', (0, 1)))]
    assert names.index('hold_step') < names.index('stop_joints') < names.index('status')
    trace, handle, result, _ = run_goal(request=grasp_request(timeout=2.0))
    assert steps(trace) == ['ik_step'] * 2 * ITERATIONS
    assert handle.status == 'aborted' and result.outcome == outcome('NOT_REACHED')
    assert result.message.startswith('timed out after 2.0 s before the IK converged; the arms '
                                     'hold where they are')


def test_grasp_goal_canceled_holds_where_it_is():
    """A canceled grasp goal stops its arms where they are, then ends canceled, saying so."""
    trace, handle, result, _ = run_goal(ik=(FAR, NEAR), cancel_at=3)
    assert handle.status == 'canceled' and not result.success
    assert result.outcome == outcome('CANCELED')
    assert steps(trace) == ['ik_step', 'ik_step', 'hold_step']
    assert result.message.startswith('canceled; the arms hold where they are (left')
    names = [entry[0] for entry in trace]
    # with the resting offsets taken once the IK was seeded, before the first step
    assert entries(trace, 'stop_joints') == [('stop_joints', [0, 1], ('reference', (0, 1)))]
    assert names.index('update_ik_solver') < names.index('stop_reference') < names.index(
        'ik_step')
    assert names.index('hold_step') < names.index('stop_joints') < names.index('status')


@pytest.mark.parametrize('ending', [
    {'ik': (NEAR,), 'errors': (NEAR,)}, {'stale_at': 2},
    {'ik': (NEAR,), 'limited': ((), (('left_elbow_joint', 1.5, 1.5),))}])
def test_grasp_goal_stops_its_arms_only_on_a_cancel_or_timeout(ending):
    """Reached, stale or pinned at a limit, a grasp goal leaves its last command."""
    trace, handle, _, _ = run_goal(request=grasp_request(arms=('left', 'right')),
                                   **ending)
    assert handle.status in ('aborted', 'succeeded')
    assert entries(trace, 'stop_joints') == []


def test_grasp_goal_pinned_at_a_limit_aborts():
    """A grasp goal whose hold is pinned at a joint limit aborts, naming the joint: not reached."""
    limited = ((), (('left_elbow_joint', 1.5, 1.5),))
    trace, handle, result, _ = run_goal(ik=(NEAR,), limited=limited)
    assert handle.status == 'aborted' and not result.success
    assert result.outcome == outcome('NOT_REACHED')
    assert result.message.startswith('the hold is pinned at a joint position limit: '
                                     'left_elbow_joint q=+1.500 (limit +1.500)')


# s of robot time a scripted grasp path takes: 3 waypoints of one DT each
PATH = 3 * DT


def test_planned_grasp_goal_plans_executes_then_holds():
    """A planned grasp goal plans, runs the path with feedback at each waypoint, then holds.

    Its timeout counts from the goal's start, the path included.
    """
    trace, handle, result, _ = run_goal(plan=True, path=PATH)
    names = [entry[0] for entry in trace]
    assert ('plan_to_ik_target', TIMEOUT) in trace
    start = names.index('execute_path')
    # before each of the 3 waypoints a feedback and a cancel check, then the hold
    assert names[start + 1:start + 7] == ['feedback', 'cancel?'] * 3
    assert names[start + 7:start + 9] == ['init_steady_state', 'cancel?']
    # the rest of the 1 s timeout after the path, in hold steps
    assert steps(trace) == ['hold_step'] * (ITERATIONS - 3)
    assert handle.status == 'aborted' and result.outcome == outcome('NOT_REACHED')
    assert result.message.startswith('timed out after 1.0 s; the arms hold where they are (left')


def test_planned_grasp_goal_reached():
    """A planned grasp goal that settles after its path succeeds."""
    trace, handle, result, _ = run_goal(plan=True, path=PATH,
                                        errors=(FAR,) * 3 + (NEAR,) * SETTLE_READS)
    assert handle.status == 'succeeded' and result.success
    assert result.outcome == outcome('REACHED')
    assert steps(trace) == ['hold_step'] * SETTLE_READS


def test_planned_grasp_goal_stops_its_path():
    """A cancel or the timeout during the path stops it at the next waypoint."""
    trace, handle, result, _ = run_goal(plan=True, path=PATH, cancel_at=2)
    assert ('path_stopped', 1) in trace and steps(trace) == []
    assert handle.status == 'canceled' and result.outcome == outcome('CANCELED')
    assert result.message.startswith('canceled during the path; the arms hold where they are')
    # a cancel stops the arm where it is
    names = [entry[0] for entry in trace]
    assert entries(trace, 'stop_joints') == [('stop_joints', [0, 1], ('reference', (0, 1)))]
    assert names.index('stop_reference') < names.index('plan_to_ik_target')
    assert names.index('path_stopped') < names.index('stop_joints') < names.index('status')
    trace, handle, result, _ = run_goal(plan=True, path=3.0)
    assert ('path_stopped', 1) in trace and steps(trace) == []
    # so does the timeout
    assert handle.status == 'aborted' and result.outcome == outcome('NOT_REACHED')
    assert entries(trace, 'stop_joints') == [('stop_joints', [0, 1], ('reference', (0, 1)))]
    assert result.message.startswith(
        'timed out after 1.0 s during the path; the arms hold where they are (left')


@pytest.mark.parametrize('failing, message, stops', [
    ('plan_error', 'planning failed: no path found', False),
    ('execute_error', 'execution failed: no path found; the arms hold where they are', True)])
def test_planned_grasp_goal_failure_aborts(failing, message, stops):
    """A planned grasp goal whose plan or path fails aborts with the reason.

    A path that fails once the arms move stops them where they are; a failed plan moved
    nothing, so it leaves the command.
    """
    trace, handle, result, _ = run_goal(plan=True,
                                        **{failing: RuntimeError('no path found')})
    assert handle.status == 'aborted' and not result.success
    assert result.outcome == outcome('FAILED')
    assert result.message == message
    assert steps(trace) == []
    assert entries(trace, 'stop_joints') == (
        [('stop_joints', [0, 1], ('reference', (0, 1)))] if stops else [])


@pytest.mark.parametrize('plan', [False, True])
def test_grasp_goal_stale_state_aborts(plan):
    """A stale /lowstate mid-goal aborts a grasp goal with the reason."""
    trace, handle, result, _ = run_goal(plan=plan, ik=(NEAR,), stale_at=2)
    assert handle.status == 'aborted' and not result.success
    assert result.outcome == outcome('FAILED')
    assert result.message == ('/lowstate is 80 ms stale (> 50 ms); holding the last command')


@pytest.mark.parametrize('request_, words', [
    (lambda: grasp_request(arms=()), 'neither arm has a target'),
    (lambda: _grasp_with(position_x=float('nan')), 'the left target is not finite'),
    (lambda: _grasp_with(w=0.0), 'the left target has no orientation'),
])
def test_grasp_goal_refused_before_any_step(request_, words):
    """A grasp goal without a target or with a bad pose aborts before the controller moves."""
    trace, handle, result, controller = run_goal(request=request_())
    assert handle.status == 'aborted' and not result.success
    assert result.outcome == outcome('FAILED')
    assert result.message.startswith(words), result.message
    assert steps(trace) == [] and entries(trace, 'check_state_fresh') == []


def _grasp_with(position_x=0.3, w=1.0):
    """Return a left-arm grasp goal with that position x (m) and orientation w."""
    goal = grasp_request()
    goal.left.pose.position.x = position_x
    goal.left.pose.orientation.w = w
    return goal


def test_direct_goal_steps_the_ik_then_holds_until_reached():
    """A direct goal steps the IK until it converges, then holds until the target settles.

    The hold starts after the feedback of the step whose velocity was under threshold_ik, and
    each period sleeps what is left of it after the step's work.
    """
    trace, handle, result, _ = run_goal(ik=(FAR, FAR, NEAR),
                                        errors=(FAR,) * 3 + (NEAR,) * SETTLE_READS)
    assert steps(trace) == ['ik_step'] * 3 + ['hold_step'] * SETTLE_READS
    assert handle.status == 'succeeded' and result.success
    start = trace.index(('cancel?', 3))
    assert [entry[0] for entry in trace[start:start + 5]] == [
        'cancel?', 'feedback', 'init_steady_state', 'info', 'sleep']
    assert ('info', 'IK converged; entering steady-state hold') in trace
    # the last period, which ends the goal, does not sleep
    assert entries(trace, 'sleep') == [('sleep', DT - WORK)] * (3 + SETTLE_READS - 1)


def test_direct_goal_catches_up_on_the_periods_an_iteration_took():
    """An iteration longer than the control period owes the IK one step per period it took.

    Each IK step takes two periods here: the first iteration runs one step, the next the two
    it owes, and every later one CATCH_UP_STEPS, however many periods it took.
    """
    from golem_upper_ik.ros2.frame_task_server import CATCH_UP_STEPS
    assert CATCH_UP_STEPS == 3
    trace, handle, result, _ = run_goal(work=2 * DT)
    assert ik_steps_per_iteration(trace) == [1, 2, 3]
    # an iteration that overran its period sleeps none of it
    assert entries(trace, 'sleep') == [('sleep', 0.0)] * 3
    assert handle.status == 'aborted' and result.outcome == outcome('NOT_REACHED')
    assert 'before the IK converged' in result.message


def test_catch_up_ends_at_the_step_that_converges():
    """The IK steps an iteration owes stop at the one under threshold_ik; the hold follows."""
    trace, handle, result, _ = run_goal(work=2 * DT, ik=(FAR, NEAR),
                                        errors=(FAR, FAR) + (NEAR,) * SETTLE_READS)
    assert steps(trace)[:3] == ['ik_step'] * 2 + ['hold_step']
    assert ik_steps_per_iteration(trace)[:2] == [1, 1]
    assert handle.status == 'succeeded' and result.success


def test_planned_goal_canceled_before_its_hold_step():
    """A planned goal checks for a cancel before each hold step, and ends canceled at once."""
    # 3 checks before the path's waypoints, then one before each hold step
    trace, handle, result, _ = run_goal(plan=True, path=PATH, cancel_at=6)
    assert steps(trace) == ['hold_step'] * 2
    assert handle.status == 'canceled' and result.outcome == outcome('CANCELED')
    names = [entry[0] for entry in trace]
    assert names.index('hold_step') < names.index('stop_joints') < names.index('status')


def test_planned_goal_pinned_at_a_limit_aborts():
    """A planned goal whose hold is pinned at a joint limit ends at once: not reached."""
    limited = ((), (('left_elbow_joint', 1.5, 1.5),))
    trace, handle, result, _ = run_goal(plan=True, path=PATH, limited=limited)
    assert steps(trace) == ['hold_step'] * 2
    assert handle.status == 'aborted' and result.outcome == outcome('NOT_REACHED')
    # the pinned step does not sleep out its period, and the arms keep the last command
    names = [entry[0] for entry in trace]
    assert names[-5:] == ['hold_step', 'feedback', 'warn', 'status', 'bias_joints']
    assert 'stop_joints' not in names
