'''IKSolver on both robots: the seed order, the optional collision barrier, the reduced rebind'''
from types import SimpleNamespace

import numpy as np
import pytest

ROBOTS = ('h12', 'g1')


@pytest.fixture(params=ROBOTS)
def solver(request, models):
    '''A fresh solver on the shared model (cheap: the model is the slow part)'''
    from golem_upper_ik.core.ik_solver import IKSolver
    model = models(request.param)
    yield IKSolver(model, model.profile.named_configs, dt=0.02, d_min=0.02)
    model.state_subscriber = None


def measured(model, q):
    from golem_upper_ik.core.robot_model import FixedState
    model.state_subscriber = FixedState(q)


def track_named_config(solver, name):
    '''Frame tasks on both end effectors, targeting their poses at the named posture'''
    model = solver.robot_model
    q_target = model.zero_q_body
    q_target[model.reduced_mask] = model.profile.named_configs[name]
    solver.clear_frame_tasks()
    left, right = model.profile.ee_frames
    solver.add_frame_task('left', left)
    solver.add_frame_task('right', right)
    solver.configuration.update(q_target)
    for task in solver.frame_tasks.values():
        task.set_target_from_configuration(solver.configuration)
    return q_target


def test_ik_seed_order_current_first(solver):
    model = solver.robot_model
    q = np.linspace(-0.1, 0.1, model.num_motor)
    measured(model, q)
    items = solver._default_seed_items()
    # the first seed that converges wins, so the measured pose goes first
    assert [name for name, _ in items] == ['current', 'home', 't_pose', 'arms_front_45']
    np.testing.assert_array_equal(items[0][1], q)
    for name, seed in items[1:]:
        np.testing.assert_array_equal(seed, model.profile.named_configs[name])


def test_named_seeds_are_fallbacks(solver):
    '''With no time to iterate, only a seed already at the target converges'''
    model = solver.robot_model
    track_named_config(solver, 'arms_front_45')
    measured(model, np.zeros(model.num_motor))  # 'home' on both robots
    result = solver.solve_ik_reduced(timeout=0.0)
    assert [attempt.seed_name for attempt in result.attempts] == [
        'current', 'home', 't_pose', 'arms_front_45']
    assert result.success and result.seed_name == 'arms_front_45'
    np.testing.assert_allclose(result.q[model.reduced_mask],
                               model.profile.named_configs['arms_front_45'], atol=1e-12)


def test_seed_configs_must_be_named_configs():
    from golem_upper_ik.core.ik_solver import IKSolver
    named = {'home': np.zeros(14), 'arms_front_45': np.zeros(14)}
    # refused before the model is touched
    with pytest.raises(ValueError, match='t_pose'):
        IKSolver(SimpleNamespace(), named)


def test_collision_barrier_only_with_collision_model(solver):
    model = solver.robot_model
    if model.has_collision:
        assert solver.collision_barrier_reduced is not None
        assert solver.configuration_reduced.collision_model is (
            model.collision_model_body_reduced)
    else:
        assert solver.collision_barrier_reduced is None
        assert solver.configuration_reduced.collision_model is None


def test_solves_a_reachable_target(solver):
    '''Both arms to the end-effector poses of a named posture, from 'home' '''
    model = solver.robot_model
    q_target = track_named_config(solver, 'arms_front_45')
    measured(model, np.zeros(model.num_motor))
    result = solver.solve_ik_reduced(alpha=0.1, timeout=5.0)
    assert result.success, result
    assert result.linear_error < 1e-3 and result.angular_error < 1e-2
    assert result.q.shape == (model.num_motor,)
    # only the IK joints move
    np.testing.assert_array_equal(result.q[~model.reduced_mask], 0.0)
    for frame in model.profile.ee_frames:
        np.testing.assert_allclose(model.get_frame_position(frame, result.q),
                                   model.get_frame_position(frame, q_target), atol=2e-3)


def test_ik_step_reduced_moves_only_the_ik_joints(solver):
    model = solver.robot_model
    left = model.profile.ee_frames[0]
    solver.add_frame_task('left', left)
    solver.update_configurations(model.zero_q_body)
    task = solver.frame_tasks['left']
    task.set_target_from_configuration(solver.configuration_reduced)
    target = task.transform_target_to_world.copy()
    target.translation = target.translation + np.array([0.05, 0.0, 0.05])
    task.set_target(target)
    vel = solver.ik_step_reduced()
    assert vel.shape == (model.num_motor,)
    assert np.any(np.abs(vel[model.left_arm_index]) > 1e-6)
    np.testing.assert_array_equal(vel[~model.reduced_mask], 0.0)


def test_update_configurations_takes_a_seed(solver):
    model = solver.robot_model
    q = np.linspace(-0.2, 0.2, model.num_motor)
    solver.update_configurations(q)
    np.testing.assert_array_equal(solver.q, q)
    np.testing.assert_array_equal(solver.q_reduced, q[model.reduced_mask])

    q_measured = np.linspace(0.1, -0.1, model.num_motor)
    measured(model, q_measured)
    solver.update_configurations()  # default: the measured state
    np.testing.assert_array_equal(solver.q, q_measured)
    np.testing.assert_array_equal(solver.q_reduced, q_measured[model.reduced_mask])


def test_rebuild_reduced_rebinds_the_reduced_model(models):
    from golem_upper_ik.core.ik_solver import IKSolver
    model = models('h12')
    solver = IKSolver(model, model.profile.named_configs)
    torso = model.profile.body_joints.index('torso_joint')
    q = np.zeros(model.num_motor)
    q[torso] = 0.3
    try:
        model.rebuild_reduced_model(q)
        assert solver.configuration_reduced.model is not model.model_body_reduced

        solver.rebuild_reduced()
        assert solver.configuration_reduced.model is model.model_body_reduced
        assert solver.configuration_reduced.collision_model is (
            model.collision_model_body_reduced)
        # the IK's reduced configuration sees the torso where it was re-locked
        frame = model.profile.ee_frames[0]
        solver.update_configurations(q)
        np.testing.assert_allclose(
            solver.configuration_reduced.get_transform_frame_to_world(frame).np,
            model.get_frame_transformation(frame, q), atol=1e-12)
    finally:
        model.rebuild_reduced_model(np.zeros(model.num_motor))


def test_unused_paths_are_gone(solver):
    '''Only what the old nodes call is carried over'''
    for name in ('solve_ik', 'integrate_reduced', 'set_from_configuration',
                 'set_from_configuration_reduced', 'com_task', 'com_task_reduced'):
        assert not hasattr(solver, name), name
