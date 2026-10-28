'''
The OMPL reduced-joint planner and its subprocess client.

The ReducedJointPlanner tests run on a two- or three-joint stand-in for the robot model,
planning for the H1-2's planner frames. The client's failure tests run planner_stubs.py's
stand-in workers in place of the real one.
'''
import multiprocessing as mp
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import planner_stubs
from golem_upper_ik.core.planner import (
    PLANNER_UNAVAILABLE,
    PlannerClient,
    PlannerConfig,
    PlannerUnavailableError,
    ReducedJointPlanner,
)


def profile_of(robot):
    if robot == 'g1':
        pytest.importorskip('g1_description')
    from golem_upper_ik.profile import load_profile
    return load_profile(robot)


def config(**kwargs):
    '''A planner config for the H1-2's planner frames'''
    kwargs.setdefault('frame_names', profile_of('h12').planner_frames)
    return PlannerConfig(**kwargs)


class FakeRobotModel:
    def __init__(self, lower=None, upper=None, collision_free=True):
        lower = np.asarray(lower if lower is not None else [-1.0, -2.0])
        upper = np.asarray(upper if upper is not None else [1.0, 2.0])
        self.has_collision = True
        self.collision_free = collision_free
        self.model_body_reduced = SimpleNamespace(
            nq=lower.shape[0],
            lowerPositionLimit=lower,
            upperPositionLimit=upper,
        )
        self.reduced_mask = np.ones(lower.shape[0], dtype=bool)
        self.state = {'q': np.zeros(lower.shape[0])}

    def check_within_limits_reduced(self, q_reduced):
        lower = self.model_body_reduced.lowerPositionLimit
        upper = self.model_body_reduced.upperPositionLimit
        return bool(np.all(q_reduced >= lower) and np.all(q_reduced <= upper))

    def reduced_limit_violations(self, q_reduced):
        lower = self.model_body_reduced.lowerPositionLimit
        upper = self.model_body_reduced.upperPositionLimit
        violations = []
        for i, value in enumerate(q_reduced):
            if value < lower[i] or value > upper[i]:
                bound = lower[i] if value < lower[i] else upper[i]
                violations.append({
                    'name': f'q[{i}]', 'value': value, 'bound': bound,
                    'overshoot': abs(value - bound),
                })
        return violations

    def check_collision_free_reduced(self, q_reduced):
        return self.collision_free

    def get_frame_position(self, frame_name, q):
        return np.array([q[1], q[1], q[0]])


def test_requires_initialized_reduced_model():
    robot_model = FakeRobotModel()
    robot_model.model_body_reduced = None

    with pytest.raises(ValueError, match='reduced model'):
        ReducedJointPlanner(robot_model, config=config())


def test_requires_initialized_collision_model():
    robot_model = FakeRobotModel()
    robot_model.has_collision = False

    with pytest.raises(ValueError, match='collision model'):
        ReducedJointPlanner(robot_model, config=config())


def test_wrong_vector_shape_raises_when_ompl_available():
    planner = ReducedJointPlanner(FakeRobotModel(), config=config())

    with pytest.raises(ValueError, match='start must have shape'):
        planner.plan([0.0, 0.0, 0.0], [0.0, 0.0])


def test_invalid_start_fails_with_clear_reason():
    planner = ReducedJointPlanner(FakeRobotModel(), config=config())

    result = planner.plan([2.0, 0.0], [0.0, 0.0])

    assert not result.success
    assert result.path.shape == (0, 2)
    assert result.reason.startswith('start is outside reduced joint limits:')


def test_invalid_goal_fails_with_clear_reason():
    planner = ReducedJointPlanner(FakeRobotModel(), config=config())

    result = planner.plan([0.0, 0.0], [0.0, 3.0])

    assert not result.success
    assert result.path.shape == (0, 2)
    assert result.reason.startswith('goal is outside reduced joint limits:')


def test_collision_state_fails_with_clear_reason():
    planner = ReducedJointPlanner(FakeRobotModel(collision_free=False), config=config())

    result = planner.plan([0.0, 0.0], [0.5, 0.5])

    assert not result.success
    assert result.path.shape == (0, 2)
    assert result.reason == 'start is in self-collision'


def test_bounds_are_copied_to_ompl_space():
    lower = np.array([-1.0, -0.5, -0.25])
    upper = np.array([1.0, 0.5, 0.25])
    planner = ReducedJointPlanner(FakeRobotModel(lower=lower, upper=upper), config=config())
    bounds = planner.space.getBounds()

    assert planner.nq == 3
    for idx in range(planner.nq):
        assert bounds.low[idx] == pytest.approx(lower[idx])
        assert bounds.high[idx] == pytest.approx(upper[idx])


def test_plan_between_fake_valid_states():
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=config(timeout=0.2, max_interpolation_steps=10),
    )

    result = planner.plan([0.0, 0.0], [0.5, 0.5])

    assert result.success
    assert result.path.shape[1] == 2
    assert result.path.shape[0] <= 10
    assert np.allclose(result.path[0], [0.0, 0.0])
    assert np.allclose(result.path[-1], [0.5, 0.5])


def test_plan_pins_idle_right_arm_with_home_tolerance():
    lower = -np.ones(14)
    upper = np.ones(14)
    planner = ReducedJointPlanner(
        FakeRobotModel(lower=lower, upper=upper),
        config=config(timeout=0.2, max_interpolation_steps=10),
    )
    start = np.zeros(14)
    goal = np.zeros(14)
    start[7:] = 0.01
    goal[:7] = 0.5

    result = planner.plan(start, goal)

    assert result.success, result.reason
    assert np.allclose(result.path[:, 7:], start[7:])


def test_path_validation_rejects_z_corridor_dip():
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=config(
            constraint_check_steps=2,
            frame_z_max=None,
        ),
    )
    start = np.array([1.0, 0.0])
    goal = np.array([1.0, 0.0])
    path = np.array([
        start,
        [0.9, 0.0],
        goal,
    ])

    planner._set_workspace_corridor(start, goal)
    reason = planner._path_validity_failure(path)

    assert 'below z=' in reason


def test_z_corridor_uses_its_own_margin():
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=config(
            frame_limit_margin=0.01,
            frame_z_corridor_margin=0.05,
        ),
    )
    start = np.array([1.0, 0.0])

    planner._set_workspace_corridor(start, start)

    assert planner.corridor_min_z['left_grasp_frame'] == pytest.approx(0.95)


def test_state_validation_rejects_y_limit_violation():
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=config(
            # FakeRobotModel maps both x and y to q[1], so the x limit has to
            # be off for this to isolate the y constraint
            frame_x_max=None,
            frame_y_min=-0.5,
            frame_y_max=0.5,
            frame_limit_margin=0.01,
        ),
    )

    result = planner.plan([0.0, 0.52], [0.0, 0.0])

    assert not result.success
    assert result.reason.startswith('start moves left_grasp_frame above y=0.5000')


def test_state_validation_allows_y_limit_margin():
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=config(
            # see above: x must be unconstrained to isolate the y margin
            frame_x_max=None,
            frame_y_min=-0.5,
            frame_y_max=0.5,
            frame_limit_margin=0.01,
        ),
    )

    reason = planner._validity_failure(np.array([0.0, 0.51]), 'state')

    assert reason == ''


def test_state_validation_rejects_x_limit_violation():
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=config(
            frame_x_max=0.7,
            frame_limit_margin=0.01,
        ),
    )

    result = planner.plan([0.0, 0.72], [0.0, 0.0])

    assert not result.success
    assert result.reason.startswith('start moves left_grasp_frame beyond x=0.7000')


def test_state_validation_rejects_z_limit_violation():
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=config(
            frame_z_max=0.5,
            frame_limit_margin=0.01,
        ),
    )

    result = planner.plan([0.52, 0.0], [0.0, 0.0])

    assert not result.success
    assert result.reason.startswith('start moves left_grasp_frame above z=0.5000')
    assert planner._validity_failure(np.array([0.51, 0.0]), 'state') == ''


def test_planner_needs_its_frame_names():
    with pytest.raises(ValueError, match='frame_names'):
        ReducedJointPlanner(FakeRobotModel(), config=PlannerConfig())
    planner = ReducedJointPlanner(FakeRobotModel(), config=config(frame_names=['a', 'b']))
    assert planner.frame_names == ('a', 'b')


def test_planner_client_resolves_the_frame_names():
    profile = profile_of('h12')
    # default: the profile's planner frames
    assert PlannerClient(profile).planner_config.frame_names == profile.planner_frames
    # the frame_names argument (Task 11 passes the controller config's planner.frame_names)
    given = PlannerConfig(moving_speed=0.1)
    client = PlannerClient(profile, given, frame_names=['c', 'd'])
    assert client.planner_config.frame_names == ('c', 'd')
    assert client.planner_config.moving_speed == 0.1
    assert given.frame_names is None  # the caller's config is not changed
    # a config naming the same frames is accepted; other frames would be ambiguous
    assert PlannerClient(profile, PlannerConfig(frame_names=('c', 'd')),
                         frame_names=('c', 'd')).planner_config.frame_names == ('c', 'd')
    with pytest.raises(ValueError, match='frame_names'):
        PlannerClient(profile, PlannerConfig(frame_names=('a', 'b')))


def test_planner_unavailable_without_collision():
    from golem_upper_ik.robots import g1
    profile = profile_of('g1')
    assert profile.collision is None
    client = PlannerClient(profile)

    assert client.available is False
    assert g1.DISPLAY_NAME == 'G1'
    assert client.unavailable_reason == PLANNER_UNAVAILABLE.format(robot='G1', module='g1')
    assert client.unavailable_reason == (
        'Planning is unavailable: the G1 has no self-collision model (its profile, '
        'golem_upper_ik/robots/g1.py, names no sphere URDF and SRDF); send the goal with '
        'plan=false')
    with pytest.raises(PlannerUnavailableError) as error:
        client.plan_configuration(np.zeros(len(profile.body_joints)), np.zeros(14),
                                  np.zeros(14))
    assert str(error.value) == client.unavailable_reason
    assert isinstance(error.value, RuntimeError)
    assert client._worker is None  # no planner process was started
    client.shutdown()


def test_planner_client_plans_on_h12():
    '''The subprocess builds the H1-2 model from its profile and plans a direct path'''
    profile = profile_of('h12')
    named = profile.named_configs
    client = PlannerClient(profile, PlannerConfig(timeout=5.0))
    assert client.available is True
    assert client.unavailable_reason is None
    try:
        result = client.plan_configuration(
            current_q=np.zeros(len(profile.body_joints)),
            start=named['init_3'],
            goal=named['prep'],
        )
        assert result.success, result.reason
        assert result.reason == 'direct path'
        np.testing.assert_allclose(result.path[0], named['init_3'], atol=1e-9)
        np.testing.assert_allclose(result.path[-1], named['prep'], atol=1e-9)
        assert set(result.metadata['frame_path_lengths']) == set(profile.planner_frames)

        # a per-request speed reaches the worker: half the speed, about twice the waypoints
        slow = client.plan_configuration(
            current_q=np.zeros(len(profile.body_joints)),
            start=named['init_3'],
            goal=named['prep'],
            moving_speed=client.planner_config.moving_speed / 2,
            max_interpolation_steps=2 * client.planner_config.max_interpolation_steps,
        )
        assert slow.success, slow.reason
        assert slow.path.shape[0] > 1.8 * result.path.shape[0]

        # an override is checked in the worker and reported back; the worker stays
        worker = client._worker
        with pytest.raises(RuntimeError, match='moving_speed must be positive'):
            client.plan_configuration(np.zeros(len(profile.body_joints)), named['init_3'],
                                      named['prep'], moving_speed=0.0)
        assert client._worker is worker and worker.process.is_alive()
    finally:
        client.shutdown()
    assert client._worker is None
    assert not worker.process.is_alive()


# The client's failure handling, on stand-in workers (planner_stubs.py)

def stub_client(worker_main, startup_timeout=30.0):
    client = PlannerClient(profile_of('h12'), startup_timeout=startup_timeout)
    client._worker_main = worker_main
    return client


def plan(client, goal):
    return client.plan_configuration(np.zeros(27), np.zeros(14), np.full(14, goal))


def planner_processes():
    return [process for process in mp.active_children() if process.name == 'planner_process']


def test_requests_are_answered_by_their_own_reply():
    client = stub_client(planner_stubs.stale_answer_first)
    try:
        # the worker sends a reply for the previous request before each answer
        for goal in (0.1, 0.2, 0.3):
            assert plan(client, goal) == [goal] * 14
    finally:
        client.shutdown()


def test_start_timeout_stops_the_worker_and_the_next_goal_starts_afresh():
    client = stub_client(planner_stubs.ready_after_a_second, startup_timeout=0.3)
    try:
        with pytest.raises(RuntimeError, match='did not start in time'):
            plan(client, 0.1)
        # the slow worker was stopped: its late 'ready' cannot answer a later goal
        assert client._worker is None
        assert planner_processes() == []

        client.startup_timeout = 10.0
        assert plan(client, 0.2) == [0.2] * 14
        assert plan(client, 0.3) == [0.3] * 14
    finally:
        client.shutdown()
    assert planner_processes() == []


class Abandoned(Exception):
    pass


class AbandoningQueue:
    '''Fails the first wait for a reply, as an interrupted goal would'''

    def __init__(self, queue):
        self.queue = queue
        self.failed = False

    def get(self, *args, **kwargs):
        if not self.failed:
            self.failed = True
            raise Abandoned()
        return self.queue.get(*args, **kwargs)


def test_an_abandoned_request_stops_the_worker():
    client = stub_client(planner_stubs.echo)
    try:
        assert plan(client, 0.1) == [0.1] * 14
        worker = client._worker
        worker.response_queue = AbandoningQueue(worker.response_queue)
        with pytest.raises(Abandoned):
            plan(client, 0.2)
        # its reply to the abandoned goal would be waiting for the next one: it is gone
        assert client._worker is None
        assert not worker.process.is_alive()
        assert plan(client, 0.3) == [0.3] * 14
        assert client._worker is not worker
    finally:
        client.shutdown()


def test_a_worker_dying_at_start_fails_at_once():
    client = stub_client(planner_stubs.dies_at_start, startup_timeout=30.0)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='exit code 3'):
        plan(client, 0.1)
    assert time.monotonic() - start < 5.0
    assert client._worker is None


def test_requests_from_several_threads_take_turns():
    client = stub_client(planner_stubs.echo_after_a_while)
    results = {}

    def request(goal):
        try:
            results[goal] = plan(client, goal)
        except Exception as error:  # the outcome under test
            results[goal] = error

    goals = (0.1, 0.2, 0.3, 0.4)
    threads = [threading.Thread(target=request, args=(goal,), daemon=True) for goal in goals]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        # each got the answer to its own request, from one worker
        assert not any(thread.is_alive() for thread in threads), results
        assert results == {goal: [goal] * 14 for goal in goals}
        assert len(planner_processes()) == 1
    finally:
        client.shutdown()


def test_shutdown_during_a_request_raises_cleanly():
    client = stub_client(planner_stubs.never_answers)
    client._start()
    errors = []

    def request():
        try:
            plan(client, 0.1)
        except Exception as error:  # the outcome under test
            errors.append(error)

    thread = threading.Thread(target=request)
    thread.start()
    time.sleep(1.0)  # the request is waiting on the worker
    client.shutdown()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == 'planner shut down'
    assert planner_processes() == []
