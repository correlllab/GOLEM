'''
IKSolver on both robots: the seed order, the self-collision barrier, the reduced rebind, and
the G1's start-up move to home with the barrier on
'''
from types import SimpleNamespace

import numpy as np
import pytest

ROBOTS = ('h12', 'g1')


def d_min_of(robot):
    '''The self-collision distance of the robot's default controller config, m'''
    from golem_upper_ik.config import load_controller_config
    from golem_upper_ik.robots import robot_module
    config = load_controller_config(robot, robot_module(robot).DEFAULT_CONFIG)
    return config['controller']['d_min']


@pytest.fixture(params=ROBOTS)
def solver(request, models):
    '''A fresh solver on the shared model (cheap: the model is the slow part)'''
    from golem_upper_ik.core.ik_solver import IKSolver
    model = models(request.param)
    yield IKSolver(model, model.profile.named_configs, dt=0.02,
                   d_min=d_min_of(request.param))
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


def test_collision_barrier_on_the_reduced_model(solver):
    model = solver.robot_model
    assert model.has_collision
    assert solver.collision_barrier_reduced is not None
    assert solver.collision_barrier_reduced.d_min == d_min_of(model.profile.robot)
    assert solver.configuration_reduced.collision_model is model.collision_model_body_reduced


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


@pytest.mark.parametrize('robot', ROBOTS)
def test_rebuild_reduced_rebinds_the_reduced_model(models, robot):
    from golem_upper_ik.core.ik_solver import IKSolver
    model = models(robot)
    solver = IKSolver(model, model.profile.named_configs)
    # the H1-2's torso, the G1's waist yaw
    held = model.profile.body_joints.index(model.profile.held_joints[0])
    q = np.zeros(model.num_motor)
    q[held] = 0.3
    try:
        model.rebuild_reduced_model(q)
        assert solver.configuration_reduced.model is not model.model_body_reduced

        solver.rebuild_reduced()
        assert solver.configuration_reduced.model is model.model_body_reduced
        assert solver.configuration_reduced.collision_model is (
            model.collision_model_body_reduced)
        # the IK's reduced configuration sees the held joint where it was re-locked
        frame = model.profile.ee_frames[0]
        solver.update_configurations(q)
        np.testing.assert_allclose(
            solver.configuration_reduced.get_transform_frame_to_world(frame).np,
            model.get_frame_transformation(frame, q), atol=1e-12)
    finally:
        model.rebuild_reduced_model(np.zeros(model.num_motor))


def g1_move(models, start, goal, steps):
    '''
    The start-up's joint-space IK steps (goto_reduced_configuration at the sim config's
    ctrl_hz and dq_lim) from start to goal on the G1, the barrier on, the legs at the sim's
    stance; the reduced configurations it passes through
    '''
    from golem_upper_ik.config import load_controller_config
    from golem_upper_ik.core.ik_solver import IKSolver
    model = models('g1')
    config = load_controller_config('g1', 'sim_split')
    profile = model.profile
    q = np.zeros(model.num_motor)
    for side in ('left', 'right'):
        for joint, angle in (('hip_pitch', -0.1), ('knee', 0.3), ('ankle_pitch', -0.2)):
            q[profile.body_joints.index(f'{side}_{joint}_joint')] = angle
    q[model.reduced_mask] = start
    dt = 1.0 / config['frequency']['ctrl_hz']
    dq_lim = config['controller']['dq_lim']
    path = []
    try:
        measured(model, q)
        model.rebuild_reduced_model(q)
        solver = IKSolver(model, profile.named_configs, dt=dt,
                          d_min=config['controller']['d_min'])
        solver.update_configurations(q)
        for _ in range(steps):
            vel = np.clip(solver.goto_reduced_configuration(goal), -dq_lim, dq_lim)
            q = q + vel * dt
            solver.update_configurations(q)
            path.append(q[model.reduced_mask])
            if np.max(np.abs(path[-1] - goal)) < 1e-3:
                break
    finally:
        model.state_subscriber = None
        model.rebuild_reduced_model(np.zeros(model.num_motor))
    return np.array(path)


def test_g1_startup_reaches_home_from_limp_arms(models):
    '''The start-up move, with the barrier on, from where the limp arms hang to home'''
    home = models('g1').profile.named_configs['home']
    from ros_harness import G1_LIMP_ARMS
    path = g1_move(models, np.array(G1_LIMP_ARMS), home, steps=300)
    np.testing.assert_allclose(path[-1], home, atol=1e-3)
    model = models('g1')
    assert all(model.check_collision_free_reduced(q) for q in path)


def test_g1_arms_down_stops_short(models):
    '''arms_down is in self-collision: a move there stops short, collision-free'''
    model = models('g1')
    named = model.profile.named_configs
    path = g1_move(models, named['home'], named['arms_down'], steps=300)
    assert len(path) == 300
    assert np.max(np.abs(path[-1] - named['arms_down'])) > 0.05
    assert all(model.check_collision_free_reduced(q) for q in path)


def test_unused_paths_are_gone(solver):
    '''IKSolver has only what the controllers call: none of these'''
    for name in ('solve_ik', 'integrate_reduced', 'set_from_configuration',
                 'set_from_configuration_reduced', 'com_task', 'com_task_reduced'):
        assert not hasattr(solver, name), name
