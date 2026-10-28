'''
The OMPL reduced-joint planner and its subprocess client.

The ReducedJointPlanner tests run on a two- or three-joint stand-in for the robot model
that carries the H1-2 profile, so the planner frames are the H1-2's grasp frames.
'''
from types import SimpleNamespace

import numpy as np
import pytest

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


class FakeRobotModel:
    def __init__(self, lower=None, upper=None, collision_free=True):
        lower = np.asarray(lower if lower is not None else [-1.0, -2.0])
        upper = np.asarray(upper if upper is not None else [1.0, 2.0])
        self.profile = profile_of('h12')
        self.init_reduced = True
        self.init_collision = True
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
    robot_model.init_reduced = False

    with pytest.raises(ValueError, match='reduced model'):
        ReducedJointPlanner(robot_model)


def test_requires_initialized_collision_model():
    robot_model = FakeRobotModel()
    robot_model.init_collision = False

    with pytest.raises(ValueError, match='collision model'):
        ReducedJointPlanner(robot_model)


def test_wrong_vector_shape_raises_when_ompl_available():
    pytest.importorskip('ompl')
    planner = ReducedJointPlanner(FakeRobotModel())

    with pytest.raises(ValueError, match='start must have shape'):
        planner.plan([0.0, 0.0, 0.0], [0.0, 0.0])


def test_invalid_start_fails_with_clear_reason():
    pytest.importorskip('ompl')
    planner = ReducedJointPlanner(FakeRobotModel())

    result = planner.plan([2.0, 0.0], [0.0, 0.0])

    assert not result.success
    assert result.path.shape == (0, 2)
    assert result.reason.startswith('start is outside reduced joint limits:')


def test_invalid_goal_fails_with_clear_reason():
    pytest.importorskip('ompl')
    planner = ReducedJointPlanner(FakeRobotModel())

    result = planner.plan([0.0, 0.0], [0.0, 3.0])

    assert not result.success
    assert result.path.shape == (0, 2)
    assert result.reason.startswith('goal is outside reduced joint limits:')


def test_collision_state_fails_with_clear_reason():
    pytest.importorskip('ompl')
    planner = ReducedJointPlanner(FakeRobotModel(collision_free=False))

    result = planner.plan([0.0, 0.0], [0.5, 0.5])

    assert not result.success
    assert result.path.shape == (0, 2)
    assert result.reason == 'start is in self-collision'


def test_bounds_are_copied_to_ompl_space():
    pytest.importorskip('ompl')
    lower = np.array([-1.0, -0.5, -0.25])
    upper = np.array([1.0, 0.5, 0.25])
    planner = ReducedJointPlanner(FakeRobotModel(lower=lower, upper=upper))
    bounds = planner.space.getBounds()

    assert planner.nq == 3
    for idx in range(planner.nq):
        assert bounds.low[idx] == pytest.approx(lower[idx])
        assert bounds.high[idx] == pytest.approx(upper[idx])


def test_plan_between_fake_valid_states():
    pytest.importorskip('ompl')
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=PlannerConfig(timeout=0.2, max_interpolation_steps=10),
    )

    result = planner.plan([0.0, 0.0], [0.5, 0.5])

    assert result.success
    assert result.path.shape[1] == 2
    assert result.path.shape[0] <= 10
    assert np.allclose(result.path[0], [0.0, 0.0])
    assert np.allclose(result.path[-1], [0.5, 0.5])


def test_plan_pins_idle_right_arm_with_home_tolerance():
    pytest.importorskip('ompl')
    lower = -np.ones(14)
    upper = np.ones(14)
    planner = ReducedJointPlanner(
        FakeRobotModel(lower=lower, upper=upper),
        config=PlannerConfig(timeout=0.2, max_interpolation_steps=10),
    )
    start = np.zeros(14)
    goal = np.zeros(14)
    start[7:] = 0.01
    goal[:7] = 0.5

    result = planner.plan(start, goal)

    assert result.success, result.reason
    assert np.allclose(result.path[:, 7:], start[7:])


def test_path_validation_rejects_z_corridor_dip():
    pytest.importorskip('ompl')
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=PlannerConfig(
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
    pytest.importorskip('ompl')
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=PlannerConfig(
            frame_limit_margin=0.01,
            frame_z_corridor_margin=0.05,
        ),
    )
    start = np.array([1.0, 0.0])

    planner._set_workspace_corridor(start, start)

    assert planner.corridor_min_z['left_grasp_frame'] == pytest.approx(0.95)


def test_state_validation_rejects_y_limit_violation():
    pytest.importorskip('ompl')
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=PlannerConfig(
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
    pytest.importorskip('ompl')
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=PlannerConfig(
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
    pytest.importorskip('ompl')
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=PlannerConfig(
            frame_x_max=0.7,
            frame_limit_margin=0.01,
        ),
    )

    result = planner.plan([0.0, 0.72], [0.0, 0.0])

    assert not result.success
    assert result.reason.startswith('start moves left_grasp_frame beyond x=0.7000')


def test_state_validation_rejects_z_limit_violation():
    pytest.importorskip('ompl')
    planner = ReducedJointPlanner(
        FakeRobotModel(),
        config=PlannerConfig(
            frame_z_max=0.5,
            frame_limit_margin=0.01,
        ),
    )

    result = planner.plan([0.52, 0.0], [0.0, 0.0])

    assert not result.success
    assert result.reason.startswith('start moves left_grasp_frame above z=0.5000')
    assert planner._validity_failure(np.array([0.51, 0.0]), 'state') == ''


def test_planner_frame_names_default_to_the_profile():
    pytest.importorskip('ompl')
    robot_model = FakeRobotModel()
    assert ReducedJointPlanner(robot_model).frame_names == robot_model.profile.planner_frames
    planner = ReducedJointPlanner(robot_model, config=PlannerConfig(frame_names=('a', 'b')))
    assert planner.frame_names == ('a', 'b')


def test_planner_client_frame_names():
    profile = profile_of('h12')
    assert PlannerClient(profile).planner_config.frame_names == profile.planner_frames
    config = PlannerConfig(moving_speed=0.1, frame_names=('a', 'b'))
    assert PlannerClient(profile, config).planner_config.frame_names == ('a', 'b')
    # the frame_names argument (the controller config's planner.frame_names) wins
    client = PlannerClient(profile, config, frame_names=['c', 'd'])
    assert client.planner_config.frame_names == ('c', 'd')
    assert client.planner_config.moving_speed == 0.1
    assert config.frame_names == ('a', 'b')  # the caller's config is not changed


def test_planner_unavailable_without_collision():
    profile = profile_of('g1')
    assert profile.collision is None
    client = PlannerClient(profile)

    assert client.available is False
    assert client.unavailable_reason == PLANNER_UNAVAILABLE.format(robot='G1')
    assert 'self-collision model' in client.unavailable_reason
    with pytest.raises(PlannerUnavailableError) as error:
        client.plan_configuration(np.zeros(len(profile.body_joints)), np.zeros(14),
                                  np.zeros(14))
    assert str(error.value) == client.unavailable_reason
    assert isinstance(error.value, RuntimeError)
    assert client._process is None  # no planner process was started
    client.shutdown()


def test_planner_client_plans_on_h12():
    '''The subprocess builds the H1-2 model from its profile and plans a direct path'''
    pytest.importorskip('ompl')
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

        # an override is checked in the worker and reported back
        with pytest.raises(RuntimeError, match='moving_speed must be positive'):
            client.plan_configuration(np.zeros(len(profile.body_joints)), named['init_3'],
                                      named['prep'], moving_speed=0.0)
    finally:
        client.shutdown()
    assert client._process is None
