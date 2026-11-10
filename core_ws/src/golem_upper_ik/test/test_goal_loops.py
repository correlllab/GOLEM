"""Check frame_task_server's goal loops against a scripted controller, without ROS.

FakeController answers each IK step, hold step and error read from a script and advances a
fake robot clock, so every path of /frame_task and /named_config runs in exact robot time:
direct and planned moves, the hold, cancel, timeouts, a hold blocked at a limit, plan and
execution failures, the stale-state abort, slow mode and an unknown named configuration.
Each test reads the trace of what the server did, in order.
"""
from types import SimpleNamespace
import threading

from geometry_msgs.msg import Pose
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
    errors: the goal's error at each read: each frame task's linear and angular error, or the
        named configuration's joint error.
    plan_error, execute_error: raised by the plan or by execute_path.
    stale_at: the IK or hold step (1-based, both counted) that raises StaleStateError.
    """

    def __init__(self, trace, *, ik=(FAR,), hold=(False,), limited=((),), errors=(FAR,),
                 plan_error=None, execute_error=None, stale_at=None):
        """Keep the scripts; every count starts at 0."""
        self.trace = trace
        self.clock = FakeClock(trace)
        self.dt = DT
        self._ik, self._hold, self._limited, self._errors = ik, hold, limited, errors
        self._plan_error, self._execute_error, self._stale_at = (
            plan_error, execute_error, stale_at)
        self.ik_steps = self.hold_steps = self.error_reads = 0
        self.steady_state_limited_joints = []
        self._speed_scale = 1.0

    @property
    def speed_scale(self):
        """Return the speed multiplier."""
        return self._speed_scale

    @speed_scale.setter
    def speed_scale(self, value):
        """Record the speed multiplier."""
        self.trace.append(('speed_scale', value))
        self._speed_scale = value

    def _work(self, name):
        """Record a step, spend WORK of robot time on it, and raise at stale_at."""
        self.trace.append((name, self.ik_steps + self.hold_steps))
        self.clock.t += WORK
        if self._stale_at is not None and self.ik_steps + self.hold_steps >= self._stale_at:
            from golem_upper_ik.core.ros_io import StaleStateError
            raise StaleStateError('/lowstate is 80 ms stale (> 50 ms); holding the last command')

    def _ik_step(self, name):
        """Return the next IK step's joint velocity."""
        self.ik_steps += 1
        self._work(name)
        return np.array([0.0, -_scripted(self._ik, self.ik_steps - 1)])

    def control_step_reduced(self):
        """Step the IK on the frame tasks."""
        return self._ik_step('ik_step')

    def goto_reduced_configuration(self, q_reduced):
        """Step the IK toward q_reduced."""
        return self._ik_step(f'goto {list(q_reduced)}')

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

    def get_frame_task_error(self, task_name):
        """Return the task's error: the scripted value, linear then angular."""
        error = self._error()
        return np.array([error, 0.0, 0.0, error, 0.0, 0.0])

    @property
    def reduced_configuration_error(self):
        """Return the IK joints' error from the named configuration."""
        return np.array([0.0, -self._error()])

    def _call(self, *entry):
        """Record a call."""
        self.trace.append(entry)

    def init_steady_state(self):
        """Start the hold."""
        self._call('init_steady_state')

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

    def set_reduced_configuration_target(self, q_reduced):
        """Set the named configuration's target."""
        self._call('set_target', list(q_reduced))

    def _plan(self, entry):
        """Record a plan request; return a 3-waypoint path or raise plan_error."""
        self._call(*entry)
        if self._plan_error is not None:
            raise self._plan_error
        return [np.zeros(2)] * 3

    def plan_to_ik_target(self, ik_timeout):
        """Plan to the IK solution of the frame tasks."""
        return self._plan(('plan_to_ik_target', ik_timeout))

    def plan_to_configuration(self, q_reduced):
        """Plan to q_reduced."""
        return self._plan(('plan_to_configuration', list(q_reduced)))

    def execute_path(self, path):
        """Run a path for 0.5 s of robot time, or raise execute_error."""
        self._call('execute_path', len(path))
        self.clock.t += 0.5
        if self._execute_error is not None:
            raise self._execute_error


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
        if hasattr(feedback, 'joint_error'):
            self._trace.append(('feedback', feedback.joint_error))
        else:
            self._trace.append(('feedback', list(feedback.errors_linear),
                                list(feedback.errors_angular)))

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
    server.profile = SimpleNamespace(named_configs={'home': np.array([0.0, 0.5])})
    server.controller = controller
    server.timeout = TIMEOUT
    server.threshold_ik = 1e-3
    server.threshold_joint = 0.01
    server.threshold_linear = 0.002
    server.threshold_angular = 0.01
    server.frame_names = []
    server.frame_targets = []
    server._controller_lock = threading.Lock()
    return server


def frame_request(plan=False, slow_mode=False, duration=0.0):
    """Return a FrameTask goal pulling left_frame to the origin."""
    from custom_ros_messages.action import FrameTask
    goal = FrameTask.Goal()
    goal.frame_names = ['left_frame']
    pose = Pose()
    pose.orientation.w = 1.0
    goal.frame_targets = [pose]
    goal.plan, goal.slow_mode = plan, slow_mode
    goal.duration.sec = int(duration)
    return goal


def named_request(name='home', plan=False, slow_mode=False, duration=0.0):
    """Return a NamedConfig goal to name."""
    from custom_ros_messages.action import NamedConfig
    goal = NamedConfig.Goal()
    goal.config_name = name
    goal.plan, goal.slow_mode = plan, slow_mode
    goal.duration.sec = int(duration)
    return goal


def run_goal(action, plan=False, cancel_at=None, request=None, **script):
    """Run one goal through the server; return (trace, goal handle, result, controller)."""
    trace = []
    controller = FakeController(trace, **script)
    server = server_of(controller, trace)
    if request is None:
        request = (frame_request if action == 'frame' else named_request)(plan=plan)
    handle = FakeGoalHandle(trace, request, cancel_at)
    callback = server.frame_task_callback if action == 'frame' else server.named_config_callback
    result = callback(handle)
    return trace, handle, result, controller


def entries(trace, *kinds):
    """Return the trace's entries whose first field is one of kinds."""
    return [entry for entry in trace if entry[0] in kinds]


def steps(trace):
    """Return the kind of each IK or hold step, in order."""
    return [entry[0].split(' ')[0] for entry in trace
            if entry[0] in ('ik_step', 'hold_step') or entry[0].startswith('goto ')]


def messages(trace, level):
    """Return the log messages of a level."""
    return [entry[1] for entry in trace if entry[0] == level]


ACTIONS = ['frame', 'named']
IK_STEP = {'frame': 'ik_step', 'named': 'goto'}


@pytest.mark.parametrize('action', ACTIONS)
def test_direct_goal_steps_the_ik_then_holds_until_reached(action):
    """A direct goal steps the IK until it converges, then holds until the target is reached."""
    from custom_ros_messages.action import FrameTask, NamedConfig
    trace, handle, result, _ = run_goal(action, ik=(FAR, FAR, NEAR), errors=(FAR,) * 4 + (NEAR,))
    assert steps(trace) == [IK_STEP[action]] * 3 + ['hold_step'] * 2
    assert handle.status == 'succeeded' and result.success
    assert isinstance(result, FrameTask.Result if action == 'frame' else NamedConfig.Result)
    # the hold starts after the feedback of the step whose velocity was under threshold_ik
    start = trace.index(('cancel?', 3))
    assert [entry[0] for entry in trace[start:start + 5]] == [
        'cancel?', 'feedback', 'init_steady_state', 'info', 'sleep']
    assert len(entries(trace, 'feedback')) == 5
    assert entries(trace, 'feedback')[-1] == (
        ('feedback', [NEAR], [NEAR]) if action == 'frame' else ('feedback', NEAR))
    # each period sleeps what is left of it after the step's work; the last ends at once
    assert entries(trace, 'sleep') == [('sleep', DT - WORK)] * 4
    reached = ['Named config reached'] if action == 'named' else []
    assert messages(trace, 'info')[2:] == [
        'IK converged; entering steady-state hold', *reached, 'Goal reached']
    assert messages(trace, 'warn') == []


@pytest.mark.parametrize('action', ACTIONS)
def test_direct_goal_times_out_before_ik_convergence(action):
    """An IK that never converges ends at the timeout, succeeded with success false."""
    trace, handle, result, _ = run_goal(action)
    assert steps(trace) == [IK_STEP[action]] * ITERATIONS
    assert handle.status == 'succeeded' and not result.success
    assert entries(trace, 'init_steady_state') == []
    assert messages(trace, 'warn') == ['Timed out before IK convergence']


@pytest.mark.parametrize('plan', [False, True])
@pytest.mark.parametrize('action', ACTIONS)
def test_goal_times_out_in_the_hold(action, plan):
    """A hold that never reaches the target ends at the timeout with success false.

    A planned goal's timeout counts from the end of its path.
    """
    trace, handle, result, _ = run_goal(action, plan=plan, ik=(NEAR,))
    assert steps(trace).count('hold_step') == (ITERATIONS if plan else ITERATIONS - 1)
    assert handle.status == 'succeeded' and not result.success
    assert messages(trace, 'warn') == ['Timed out during steady-state hold']


@pytest.mark.parametrize('action', ACTIONS)
def test_goal_duration_replaces_the_controller_timeout(action):
    """A goal's duration, s of robot time, replaces controller.timeout."""
    request = (frame_request if action == 'frame' else named_request)(duration=2.0)
    trace, _, result, _ = run_goal(action, request=request)
    assert len(steps(trace)) == 2 * ITERATIONS
    assert not result.success


@pytest.mark.parametrize('action', ACTIONS)
def test_direct_goal_canceled_after_its_step(action):
    """A direct goal checks for a cancel after each step, and ends canceled at once."""
    trace, handle, result, _ = run_goal(action, ik=(FAR, NEAR), cancel_at=3)
    assert steps(trace) == [IK_STEP[action], IK_STEP[action], 'hold_step']
    assert handle.status == 'canceled' and not result.success
    assert [entry[0] for entry in trace[-5:]] == [
        'hold_step', 'cancel?', 'info', 'status', 'speed_scale']
    assert trace[-3] == ('info', 'Goal cancelled')
    assert len(entries(trace, 'feedback')) == 2


@pytest.mark.parametrize('action', ACTIONS)
def test_planned_goal_canceled_before_its_step(action):
    """A planned goal checks for a cancel before each hold step, and ends canceled at once."""
    trace, handle, result, _ = run_goal(action, plan=True, cancel_at=3)
    assert steps(trace) == ['hold_step'] * 2
    assert handle.status == 'canceled' and not result.success
    assert [entry[0] for entry in trace[-5:]] == [
        'sleep', 'cancel?', 'info', 'status', 'speed_scale']
    assert trace[-3] == ('info', 'Goal cancelled')


@pytest.mark.parametrize('plan', [False, True])
@pytest.mark.parametrize('action', ACTIONS)
def test_hold_blocked_at_a_limit_ends_the_goal(action, plan):
    """A hold pinned at a joint limit ends the goal at once, success false, naming the joint."""
    limited = ((), (('left_elbow_joint', 1.5, 1.5),))
    trace, handle, result, _ = run_goal(action, plan=plan, ik=(NEAR,), limited=limited)
    assert steps(trace).count('hold_step') == 2
    assert handle.status == 'succeeded' and not result.success
    assert messages(trace, 'warn') == [
        'Hold blocked at a joint position limit: left_elbow_joint q=+1.500 (limit +1.500)']
    # the blocked step does not sleep out its period
    names = [entry[0] for entry in trace]
    last_step = len(names) - 1 - names[::-1].index('hold_step')
    assert names[last_step:] == (['hold_step', 'feedback', 'warn', 'status', 'speed_scale']
                                 if plan else
                                 ['hold_step', 'cancel?', 'feedback', 'warn', 'status',
                                  'speed_scale'])


@pytest.mark.parametrize('action, plan, ends', [
    ('frame', False, True), ('named', False, True), ('named', True, True),
    ('frame', True, False)])
def test_hold_step_convergence(action, plan, ends):
    """The hold step's own convergence ends a goal, except a planned frame goal's.

    A planned frame goal is reached only when its frames are within the thresholds.
    """
    trace, handle, result, _ = run_goal(action, plan=plan, ik=(NEAR,), hold=(False, True))
    assert handle.status == 'succeeded'
    if ends:
        assert steps(trace).count('hold_step') == 2 and result.success
        assert messages(trace, 'info')[-1] == 'Goal reached'
    else:
        assert steps(trace).count('hold_step') == ITERATIONS and not result.success


@pytest.mark.parametrize('failing', ['plan_error', 'execute_error'])
@pytest.mark.parametrize('action', ACTIONS)
def test_plan_or_execution_failure_aborts(action, failing):
    """A planned goal whose plan or execution fails aborts before any hold step."""
    trace, handle, result, _ = run_goal(action, plan=True,
                                        **{failing: RuntimeError('no path found')})
    assert handle.status == 'aborted' and not result.success
    assert steps(trace) == [] and entries(trace, 'init_steady_state') == []
    assert messages(trace, 'warn') == ['Planning or execution failed: no path found']
    assert trace[-2:] == [('status', 'aborted'), ('speed_scale', 1.0)]


@pytest.mark.parametrize('action', ACTIONS)
def test_planned_goal_plans_executes_then_holds(action):
    """A planned goal plans, executes the path, then starts the hold, in that order."""
    trace, handle, result, _ = run_goal(action, plan=True, errors=(FAR, NEAR))
    names = [entry[0] for entry in trace]
    start = names.index('execute_path')
    if action == 'frame':
        assert ('plan_to_ik_target', TIMEOUT) in trace
        assert names[start + 1:start + 4] == ['info', 'init_steady_state', 'cancel?']
        target = 'target frame poses'
    else:
        assert ('plan_to_configuration', [0.0, 0.5]) in trace
        # the feedback's joint error is measured against the planned goal
        assert names[start + 1:start + 5] == ['info', 'set_target', 'init_steady_state',
                                              'cancel?']
        target = 'named configuration'
    assert messages(trace, 'info')[2:] == [
        f'Planning path to {target}',
        'Executing planned path with 3 waypoints',
        f'Path execution finished in 0.50s; holding steady state (timeout {TIMEOUT:.1f}s)',
        'Goal reached']
    assert steps(trace) == ['hold_step'] * 2
    assert handle.status == 'succeeded' and result.success


@pytest.mark.parametrize('plan', [False, True])
@pytest.mark.parametrize('action', ACTIONS)
def test_slow_mode_lasts_the_goal(action, plan):
    """slow_mode slows every step of the goal, and full speed returns after it."""
    request = (frame_request if action == 'frame' else named_request)(plan=plan,
                                                                      slow_mode=True)
    trace, _, _, _ = run_goal(action, request=request, ik=(NEAR,))
    scales = entries(trace, 'speed_scale')
    assert scales == [('speed_scale', 0.25), ('speed_scale', 1.0)]
    names = [entry[0] for entry in trace]
    assert names.index('speed_scale') < min(
        i for i, name in enumerate(names) if name in ('plan_to_ik_target',
                                                      'plan_to_configuration', 'hold_step')
        or name.startswith('goto') or name == 'ik_step')
    assert len(names) - 1 - names[::-1].index('speed_scale') == names.index('status') + 1
    assert ('info', 'Slow mode enabled (speed_scale=0.25)') in trace


@pytest.mark.parametrize('action', ACTIONS)
def test_full_speed_goal_logs_no_slow_mode(action):
    """A goal without slow_mode runs at full speed throughout."""
    trace, _, _, _ = run_goal(action)
    assert entries(trace, 'speed_scale') == [('speed_scale', 1.0), ('speed_scale', 1.0)]
    assert all('Slow mode' not in m for m in messages(trace, 'info'))


@pytest.mark.parametrize('plan', [False, True])
@pytest.mark.parametrize('action', ACTIONS)
def test_stale_state_aborts_and_restores_full_speed(action, plan):
    """A stale /lowstate mid-goal aborts it, after full speed is restored."""
    request = (frame_request if action == 'frame' else named_request)(plan=plan,
                                                                      slow_mode=True)
    trace, handle, result, _ = run_goal(action, request=request, ik=(NEAR,), stale_at=2)
    assert handle.status == 'aborted' and not result.success
    names = [entry[0] for entry in trace]
    assert names[-3:] == ['speed_scale', 'error', 'status']
    assert trace[-3] == ('speed_scale', 1.0)
    assert messages(trace, 'error') == [
        'Aborting goal: /lowstate is 80 ms stale (> 50 ms); holding the last command']


def test_unknown_named_config_aborts():
    """A named configuration the profile does not have aborts the goal before any step."""
    trace, handle, result, _ = run_goal('named', request=named_request('no_such_pose'))
    assert handle.status == 'aborted' and not result.success
    assert messages(trace, 'warn') == ['Named config "no_such_pose" not found']
    assert steps(trace) == [] and entries(trace, 'speed_scale') == []


def test_frame_goal_replaces_the_frame_tasks():
    """A frame goal checks the state, replaces the frame tasks and re-seeds the IK first."""
    trace, _, _, _ = run_goal('frame')
    names = [entry[0] for entry in trace]
    assert names[:2] == ['info', 'check_state_fresh']
    assert names.index('clear_frame_tasks') < names.index('add_frame_task') < names.index(
        'update_ik_solver') < names.index('speed_scale')
    assert ('add_frame_task', 'left_frame_task', 'left_frame') in trace


def test_feedback_carries_each_frames_errors():
    """Frame feedback is each frame's position and orientation error; named, the joint error."""
    trace, _, _, _ = run_goal('frame', errors=(0.5,))
    assert entries(trace, 'feedback')[0] == ('feedback', [0.5], [0.5])
    trace, _, _, _ = run_goal('named', errors=(0.5,))
    assert entries(trace, 'feedback')[0] == ('feedback', 0.5)
