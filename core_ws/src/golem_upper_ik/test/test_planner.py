"""Check the OMPL reduced-joint planner.

The tests run on a two- or three-joint stand-in for the robot model, planning for the H1-2's
planner frames.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from golem_upper_ik.core.planner import PlannerConfig, ReducedJointPlanner


def profile_of(robot):
    """Return robot's RobotProfile."""
    from golem_upper_ik.profile import load_profile
    return load_profile(robot)


def config(**kwargs):
    """Return a PlannerConfig for the H1-2's planner frames, with kwargs."""
    kwargs.setdefault('frame_names', profile_of('h12').planner_frames)
    return PlannerConfig(**kwargs)


class FakeRobotModel:
    """Stand in for RobotModel: box limits, a fixed collision answer, frames at (q1, q1, q0)."""
    def __init__(self, lower=None, upper=None, collision_free=True):
        """Set the joint limits (default two joints) and whether every state is collision-free."""
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
        """Return whether q_reduced is within the limits."""
        lower = self.model_body_reduced.lowerPositionLimit
        upper = self.model_body_reduced.upperPositionLimit
        return bool(np.all(q_reduced >= lower) and np.all(q_reduced <= upper))

    def reduced_limit_violations(self, q_reduced):
        """Return the joints of q_reduced past their limits."""
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
        """Return the fixed collision answer."""
        return self.collision_free

    def get_frame_position(self, frame_name, q):
        """Return every frame at (q[1], q[1], q[0]), m."""
        return np.array([q[1], q[1], q[0]])

    def get_frame_rotation(self, frame_name, q):
        """Return every frame's rotation: the identity, so a path's frames never turn."""
        return np.eye(3)


def test_requires_initialized_reduced_model():
    """A model without a reduced model is refused."""
    robot_model = FakeRobotModel()
    robot_model.model_body_reduced = None

    with pytest.raises(ValueError, match='reduced model'):
        ReducedJointPlanner(robot_model, config=config())


def test_requires_initialized_collision_model():
    """A model without a self-collision model is refused."""
    robot_model = FakeRobotModel()
    robot_model.has_collision = False

    with pytest.raises(ValueError, match='collision model'):
        ReducedJointPlanner(robot_model, config=config())


def test_wrong_vector_shape_raises_when_ompl_available():
    """A start of the wrong shape is refused."""
    planner = ReducedJointPlanner(FakeRobotModel(), config=config())

    with pytest.raises(ValueError, match='start must have shape'):
        planner.plan([0.0, 0.0, 0.0], [0.0, 0.0])


def test_invalid_start_fails_with_clear_reason():
    """A start outside the joint limits fails, saying so."""
    planner = ReducedJointPlanner(FakeRobotModel(), config=config())

    result = planner.plan([2.0, 0.0], [0.0, 0.0])

    assert not result.success
    assert result.path.shape == (0, 2)
    assert result.reason.startswith('start is outside reduced joint limits:')


def test_invalid_goal_fails_with_clear_reason():
    """A goal outside the joint limits fails, saying so."""
    planner = ReducedJointPlanner(FakeRobotModel(), config=config())

    result = planner.plan([0.0, 0.0], [0.0, 3.0])

    assert not result.success
    assert result.path.shape == (0, 2)
    assert result.reason.startswith('goal is outside reduced joint limits:')


def test_collision_state_fails_with_clear_reason():
    """A start in self-collision fails, saying so."""
    planner = ReducedJointPlanner(FakeRobotModel(collision_free=False), config=config())

    result = planner.plan([0.0, 0.0], [0.5, 0.5])

    assert not result.success
    assert result.path.shape == (0, 2)
    assert result.reason == 'start is in self-collision'


def test_bounds_are_copied_to_ompl_space():
    """The OMPL space takes the reduced model's joint limits."""
    lower = np.array([-1.0, -0.5, -0.25])
    upper = np.array([1.0, 0.5, 0.25])
    planner = ReducedJointPlanner(FakeRobotModel(lower=lower, upper=upper), config=config())
    bounds = planner.space.getBounds()

    assert planner.nq == 3
    for idx in range(planner.nq):
        assert bounds.low[idx] == pytest.approx(lower[idx])
        assert bounds.high[idx] == pytest.approx(upper[idx])


def test_plan_between_fake_valid_states():
    """A plan between valid states runs start to goal within max_interpolation_steps."""
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


def peak_acceleration(path, dt):
    """Return a path's largest joint acceleration, rad/s^2, from rest before to rest after."""
    rest = np.vstack([path[:1], path, path[-1:]])
    return float(np.max(np.abs(np.diff(rest, n=2, axis=0)))) / dt ** 2


def test_paths_keep_max_acceleration():
    """A path too short for max_acceleration gets more waypoints, its acceleration then within it.

    Without max_acceleration the path keeps moving_speed's waypoint count, whose smoothstep
    accelerates the joints faster than that.
    """
    dt = 1.0 / 30.0
    paths = {}
    for limit in (None, 2.0):
        planner = ReducedJointPlanner(
            FakeRobotModel(), config=config(moving_speed=10.0, dt=dt, max_acceleration=limit))
        result = planner.plan([0.0, 0.0], [0.4, 0.4])
        assert result.success
        np.testing.assert_allclose(result.path[0], [0.0, 0.0])
        np.testing.assert_allclose(result.path[-1], [0.4, 0.4])
        paths[limit] = result.path
    assert peak_acceleration(paths[None], dt) > 2.0
    assert len(paths[2.0]) > len(paths[None])
    assert peak_acceleration(paths[2.0], dt) <= 2.0 * (1.0 + 1e-9)


def test_plan_pins_idle_right_arm_with_home_tolerance():
    """An arm whose start and goal differ by less than the tolerance stays pinned."""
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
    """A path dipping below the start-goal z corridor between waypoints is invalid."""
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
    """The corridor floor is the lower end's z less frame_z_corridor_margin."""
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
    """A frame past frame_y_max by more than the margin is invalid."""
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
    """A frame past frame_y_max by less than the margin is valid."""
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
    """A frame past frame_x_max by more than the margin is invalid."""
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
    """A frame past frame_z_max by more than the margin is invalid, within it valid."""
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
    """A config without frame_names is refused; given ones are kept."""
    with pytest.raises(ValueError, match='frame_names'):
        ReducedJointPlanner(FakeRobotModel(), config=PlannerConfig())
    planner = ReducedJointPlanner(FakeRobotModel(), config=config(frame_names=['a', 'b']))
    assert planner.frame_names == ('a', 'b')
