"""Check IKSolver on both robots: the seeds, the barrier, the rebind and the G1's start-up."""
from types import SimpleNamespace

import numpy as np
import pytest
from golem_upper_ik.profile import known_robots

ROBOTS = known_robots()


def d_min_of(robot):
    """Return the self-collision distance of the robot's real-robot controller config, m."""
    from golem_upper_ik.config import load_controller_config
    config = load_controller_config(robot, {'h12': 'safety_split', 'g1': 'g1_split'}[robot])
    return config['controller']['d_min']


@pytest.fixture(params=ROBOTS)
def solver(request, models):
    """Yield a fresh solver on the shared model (cheap: the model is the slow part)."""
    from golem_upper_ik.core.ik_solver import IKSolver
    model = models(request.param)
    yield IKSolver(model, model.profile.named_configs, dt=0.02,
                   d_min=d_min_of(request.param))
    model.state_subscriber = None


def measured(model, q):
    """Give model a fixed measured state at q."""
    from golem_upper_ik.core.robot_model import FixedState
    model.state_subscriber = FixedState(q)


def track_named_config(solver, name):
    """Add frame tasks on both end effectors at their poses in the named posture; its body q."""
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
    """The seeds are the measured pose first, then the named fallbacks in order."""
    model = solver.robot_model
    q = np.linspace(-0.1, 0.1, model.num_motor)
    measured(model, q)
    items = solver._default_seed_items()
    # the first seed that converges wins, so the measured pose goes first
    assert [name for name, _ in items] == ['current', 'home', 't_pose', 'arms_front_45']
    np.testing.assert_array_equal(items[0][1], q)
    for name, seed in items[1:]:
        np.testing.assert_array_equal(seed, model.profile.named_configs[name])


@pytest.mark.parametrize('target, tried', [
    ('t_pose', ['current', 'home', 't_pose']),
    ('arms_front_45', ['current', 'home', 't_pose', 'arms_front_45']),
])
def test_named_seeds_are_fallbacks(solver, monkeypatch, target, tried):
    """solve_ik_reduced tries the seeds in order and stops at the first that converges.

    With no time to iterate, only a seed already at the target converges.
    """
    model = solver.robot_model
    track_named_config(solver, target)
    measured(model, np.zeros(model.num_motor))  # 'home' on both robots
    seeds = []
    attempt = solver._solve_ik_reduced_attempt

    def record(seed_name, *args):
        """Note the seed, then run the attempt."""
        seeds.append(seed_name)
        return attempt(seed_name, *args)

    monkeypatch.setattr(solver, '_solve_ik_reduced_attempt', record)
    result = solver.solve_ik_reduced(timeout=0.0)
    assert seeds == tried
    assert result.success and result.seed_name == target
    np.testing.assert_allclose(result.q[model.reduced_mask],
                               model.profile.named_configs[target], atol=1e-12)


def fake_attempts(solver, monkeypatch, converges):
    """Replace the IK attempt with a fake; return its calls as (seed, timeout, start q).

    converges(seed_name, call) says whether a seed's call-th attempt (from 1) converges; each
    attempt ends at its start plus 0.01 rad on every IK joint, at once.
    """
    from golem_upper_ik.core.ik_solver import IKResult
    model = solver.robot_model
    calls = []

    def attempt(seed_name, q_seed, alpha, timeout):
        """Note the call; return the fake outcome."""
        calls.append((seed_name, timeout, np.array(q_seed)))
        q = model.zero_q_body
        q[model.reduced_mask] = np.asarray(q_seed) + 0.01
        success = converges(seed_name, sum(call[0] == seed_name for call in calls))
        return IKResult(q=q, success=success, linear_error=0.0 if success else 1.0,
                        angular_error=0.0, seed_name=seed_name)

    monkeypatch.setattr(solver, '_solve_ik_reduced_attempt', attempt)
    return calls


def test_a_fast_seed_is_not_held_up_by_seeds_that_never_converge(solver, monkeypatch):
    """The first pass gives each seed at most SEED_FIRST_PASS, in order.

    So a seed that converges in its first pass ends the search after the seeds before it have
    run SEED_FIRST_PASS each, not their whole timeout.
    """
    from golem_upper_ik.core.ik_solver import SEED_FIRST_PASS
    measured(solver.robot_model, solver.robot_model.zero_q_body)
    calls = fake_attempts(solver, monkeypatch, lambda seed, call: seed == 't_pose')
    result = solver.solve_ik_reduced(timeout=5.0)
    assert [(seed, timeout) for seed, timeout, _ in calls] == [
        ('current', SEED_FIRST_PASS), ('home', SEED_FIRST_PASS), ('t_pose', SEED_FIRST_PASS)]
    assert result.success and result.seed_name == 't_pose'


def test_the_second_pass_resumes_each_seed_for_the_rest_of_its_timeout(solver, monkeypatch):
    """A seed that converges only after its first pass is resumed where that pass left it.

    The second pass goes through the unconverged seeds in order, each for its timeout less
    what its first pass took, so each seed iterates for its timeout in all.
    """
    from golem_upper_ik.core.ik_solver import SEED_FIRST_PASS
    measured(solver.robot_model, solver.robot_model.zero_q_body)
    calls = fake_attempts(solver, monkeypatch, lambda seed, call: (seed, call) == ('home', 2))
    result = solver.solve_ik_reduced(timeout=5.0)
    assert [seed for seed, _, _ in calls] == [
        'current', 'home', 't_pose', 'arms_front_45', 'current', 'home']
    assert all(timeout == SEED_FIRST_PASS for _, timeout, _ in calls[:4])
    # the fake attempts take no time, so the second pass gets (nearly) the whole timeout
    assert all(5.0 - 0.05 < timeout <= 5.0 for _, timeout, _ in calls[4:])
    np.testing.assert_allclose(calls[5][2], calls[1][2] + 0.01)
    assert result.success and result.seed_name == 'home'
    np.testing.assert_allclose(result.q[solver.robot_model.reduced_mask], calls[1][2] + 0.02)


def test_a_timeout_shorter_than_the_first_pass_bounds_it(solver, monkeypatch):
    """With timeout under SEED_FIRST_PASS each seed gets timeout, once; none converging.

    The best attempt is returned and the configuration left at it.
    """
    measured(solver.robot_model, solver.robot_model.zero_q_body)
    calls = fake_attempts(solver, monkeypatch, lambda seed, call: False)
    result = solver.solve_ik_reduced(timeout=0.2)
    assert [(seed, timeout) for seed, timeout, _ in calls] == [
        ('current', 0.2), ('home', 0.2), ('t_pose', 0.2), ('arms_front_45', 0.2)]
    assert not result.success
    np.testing.assert_allclose(solver.configuration_reduced.q,
                               result.q[solver.robot_model.reduced_mask])


def test_a_refused_seed_passes_the_search_on(solver, monkeypatch):
    """A converged seed accept refuses passes the search on to the next seed to converge.

    accept is called once for each converged seed, in the order they converge.
    """
    measured(solver.robot_model, solver.robot_model.zero_q_body)
    fake_attempts(solver, monkeypatch, lambda seed, call: seed in ('current', 't_pose'))
    asked = []

    def accept(result):
        """Note the seed; take only t_pose's solution."""
        asked.append(result.seed_name)
        return result.seed_name == 't_pose'

    result = solver.solve_ik_reduced(timeout=5.0, accept=accept)
    assert asked == ['current', 't_pose']
    assert result.success and result.seed_name == 't_pose'


def test_seed_configs_must_be_named_configs():
    """A solver whose named configs lack a seed is refused before the model is touched."""
    from golem_upper_ik.core.ik_solver import IKSolver
    named = {'home': np.zeros(14), 'arms_front_45': np.zeros(14)}
    # refused before the model is touched
    with pytest.raises(ValueError, match='t_pose'):
        IKSolver(SimpleNamespace(), named)


def test_collision_barrier_on_the_reduced_model(solver):
    """The barrier keeps the config's d_min on the reduced model's collision pairs."""
    model = solver.robot_model
    assert model.has_collision
    assert solver.collision_barrier_reduced is not None
    assert solver.collision_barrier_reduced.d_min == d_min_of(model.profile.robot)
    assert solver.configuration_reduced.collision_model is model.collision_model_body_reduced


def test_barrier_takes_the_closest_pairs(solver):
    """The barrier constrains the BARRIER_PAIRS closest sphere pairs of each step, or all."""
    from golem_upper_ik.core.ik_solver import BARRIER_PAIRS
    pairs = len(solver.robot_model.collision_model_body_reduced.collisionPairs)
    assert solver.collision_barrier_reduced.dim == min(BARRIER_PAIRS, pairs)


def test_solves_a_reachable_target(solver):
    """Both arms reach a named posture's end-effector poses from home, moving only IK joints."""
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
    """An IK step moves the tasked arm and no joint outside the IK."""
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
    """update_configurations takes a body q, or by default the measured state."""
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
    """rebuild_reduced binds the IK to the re-locked reduced model and its spheres."""
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
            model.get_frame_transformation_body(frame, q), atol=1e-12)
    finally:
        model.rebuild_reduced_model(np.zeros(model.num_motor))


def g1_move(models, start, goal, steps):
    """Return the reduced configurations of the G1's start-up IK steps from start to goal.

    The steps are goto_reduced_configuration at the sim config's ctrl_hz and dq_lim, the
    barrier on, the legs at the sim's stance.
    """
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
    """From where the limp arms hang, the barrier-on start-up move reaches home collision-free."""
    home = models('g1').profile.named_configs['home']
    from ros_harness import G1_LIMP_ARMS
    path = g1_move(models, np.array(G1_LIMP_ARMS), home, steps=300)
    np.testing.assert_allclose(path[-1], home, atol=1e-3)
    model = models('g1')
    assert all(model.check_collision_free_reduced(q) for q in path)


def test_g1_arms_down_stops_short(models):
    """A move to arms_down, which is in self-collision, stops short and stays collision-free."""
    model = models('g1')
    named = model.profile.named_configs
    path = g1_move(models, named['home'], named['arms_down'], steps=300)
    assert len(path) == 300
    assert np.max(np.abs(path[-1] - named['arms_down'])) > 0.05
    assert all(model.check_collision_free_reduced(q) for q in path)



@pytest.mark.parametrize('robot, base', [('h12', 'home'), ('g1', 't_pose')])
def test_solutions_stay_in_the_clip_range(models, robot, base):
    """The IK never returns or steps to a configuration outside the controller's clip range.

    Each target is the pose of the base posture with one IK joint 0.2 rad past its joint
    table limit, and the measured pose is that configuration, outside the range too: the
    solves from every seed and 30 control steps from the measured pose all stay within the
    clip range. A target the self-collision barrier leaves no IK step for (NoSolutionFound)
    returns nothing; at least three in four targets of each robot have one.
    """
    from pink.exceptions import NoSolutionFound
    from golem_upper_ik.config import load_controller_config
    from golem_upper_ik.core.ik_solver import IKSolver
    model = models(robot)
    profile = model.profile
    config = load_controller_config(robot, {'h12': 'safety_split', 'g1': 'g1_split'}[robot])
    clip = config['limits']['q_clip_limits']
    mask = model.reduced_mask
    low = np.maximum(model.model_body.lowerPositionLimit, clip[:, 0])[mask]
    high = np.minimum(model.model_body.upperPositionLimit, clip[:, 1])[mask]
    table = np.stack([profile.q_low, profile.q_high], axis=1)[mask]
    start = np.asarray(profile.named_configs[base], dtype=float)
    left, right = profile.ee_frames
    cases = solved = 0
    try:
        model.set_position_limits(clip)
        solver = IKSolver(model, profile.named_configs, dt=0.02,
                          d_min=config['controller']['d_min'])
        for joint in range(mask.sum()):
            for side, past in ((0, -0.2), (1, 0.2)):
                cases += 1
                q_past = start.copy()
                q_past[joint] = table[joint, side] + past
                q_body = model.zero_q_body
                q_body[mask] = q_past
                measured(model, q_body)
                solver.clear_frame_tasks()
                solver.add_frame_task('left', left)
                solver.add_frame_task('right', right)
                solver.configuration_reduced.update(q_past)
                for task in solver.frame_tasks.values():
                    task.set_target_from_configuration(solver.configuration_reduced)
                where = (profile.enabled_joints[joint], 'low' if side == 0 else 'high')
                try:
                    result = solver.solve_ik_reduced(alpha=0.1, timeout=0.05)
                    q = result.q[mask]
                    assert np.all(q >= low - 1e-9) and np.all(q <= high + 1e-9), where
                    solver.update_configurations()
                    for _ in range(30):
                        solver.integrate(solver.ik_step_reduced())
                        q = solver.q_reduced
                        assert np.all(q >= low - 1e-9) and np.all(q <= high + 1e-9), where
                        np.testing.assert_array_equal(solver.q[mask], q)
                except NoSolutionFound:
                    continue
                solved += 1
    finally:
        model.state_subscriber = None
        model.set_position_limits(None)
        model.rebuild_reduced_model(np.zeros(model.num_motor))
    assert solved >= 0.75 * cases, (solved, cases)


def stepping_solver(models, robot):
    """Return (solver, config) on the robot's real controller config: its clip range and ctrl_hz."""
    from golem_upper_ik.config import load_controller_config
    from golem_upper_ik.core.ik_solver import IKSolver
    model = models(robot)
    config = load_controller_config(robot, {'h12': 'safety_split', 'g1': 'g1_split'}[robot])
    model.set_position_limits(config['limits']['q_clip_limits'])
    solver = IKSolver(model, model.profile.named_configs,
                      dt=1.0 / config['frequency']['ctrl_hz'], d_min=config['controller']['d_min'])
    return solver, config


def step_to(solver, q_start, q_goal, steps, check):
    """Step the IK from reduced q_start toward the grasp-frame poses of reduced q_goal.

    Each step's velocity is integrated as the controller does; check(q_before, vel) runs on
    every step. Returns False when the barrier leaves no IK step (NoSolutionFound).
    """
    from pink.exceptions import NoSolutionFound
    model = solver.robot_model
    q_body = model.zero_q_body
    q_body[model.reduced_mask] = q_start
    measured(model, q_body)
    solver.update_configurations()
    solver.clear_frame_tasks()
    left, right = model.profile.ee_frames
    solver.add_frame_task('left', left)
    solver.add_frame_task('right', right)
    solver.configuration_reduced.update(q_goal)
    for task in solver.frame_tasks.values():
        task.set_target_from_configuration(solver.configuration_reduced)
    solver.configuration_reduced.update(solver.q[model.reduced_mask])
    try:
        for _ in range(steps):
            q_before = solver.q_reduced
            vel = solver.ik_step_reduced()
            check(q_before, vel[model.reduced_mask])
            solver.integrate(vel)
    except NoSolutionFound:
        return False
    return True


@pytest.mark.parametrize('robot, base', [('h12', 'home'), ('g1', 't_pose')])
def test_stepped_approach_to_a_limit_is_braked(models, robot, base):
    """A stepped move closes on a position limit no faster than its distance over the time constant.

    Each target is the pose of the base posture with one IK joint 0.2 rad past its range
    limit. On every control step every IK joint moves toward its nearer limit by at most
    its distance to it times 1 - exp(-dt / LIMIT_APPROACH_TIME), and the bound is the one
    that stops the joint in some step (it binds).
    """
    from golem_upper_ik.core.ik_solver import LIMIT_APPROACH_TIME
    model = models(robot)
    mask = model.reduced_mask
    start = np.asarray(model.profile.named_configs[base], dtype=float)
    try:
        solver, _ = stepping_solver(models, robot)
        low = model.model_body_reduced.lowerPositionLimit.copy()
        high = model.model_body_reduced.upperPositionLimit.copy()
        fraction = -np.expm1(-solver.dt / LIMIT_APPROACH_TIME)
        binding = []

        def check(q, vel):
            step = vel * solver.dt
            toward_low = np.maximum(-step, 0.0)
            toward_high = np.maximum(step, 0.0)
            assert np.all(toward_low <= fraction * (q - low) + 1e-6)
            assert np.all(toward_high <= fraction * (high - q) + 1e-6)
            binding.append(np.any(np.isclose(toward_low, fraction * (q - low), atol=1e-6)
                                  & (toward_low > 1e-4))
                           or np.any(np.isclose(toward_high, fraction * (high - q), atol=1e-6)
                                     & (toward_high > 1e-4)))

        solved = 0
        for joint in range(mask.sum()):
            for side, limit, past in ((0, low, -0.2), (1, high, 0.2)):
                goal = start.copy()
                goal[joint] = limit[joint] + past
                solved += step_to(solver, start, goal, 60, check)
        assert solved >= mask.sum()
        assert any(binding)
    finally:
        model.state_subscriber = None
        model.set_position_limits(None)
        model.rebuild_reduced_model(np.zeros(model.num_motor))


@pytest.mark.parametrize('robot, base', [('h12', 'home'), ('g1', 't_pose')])
def test_stepped_moves_away_from_limits_are_not_braked(models, robot, base):
    """A step whose joints all stay farther from the limits than the brake reaches is Pink's default.

    Targets are the poses of the base posture with every IK joint moved by up to 0.3 rad.
    On each step where the brake's bound on every joint (its distance to its nearer limit
    times 1 - exp(-dt / LIMIT_APPROACH_TIME)) exceeds the default-gain step by 1 mrad, the
    braked step equals the step under Pink's default configuration limit (gain 0.5) and the
    same velocity limits.
    """
    import pink
    from golem_upper_ik.core.ik_solver import LIMIT_APPROACH_TIME
    model = models(robot)
    start = np.asarray(model.profile.named_configs[base], dtype=float)
    rng = np.random.default_rng(3)
    try:
        solver, _ = stepping_solver(models, robot)
        low = model.model_body_reduced.lowerPositionLimit.copy()
        high = model.model_body_reduced.upperPositionLimit.copy()
        fraction = -np.expm1(-solver.dt / LIMIT_APPROACH_TIME)
        compared = []
        default_limits = [pink.limits.ConfigurationLimit(model.model_body_reduced),
                          pink.limits.VelocityLimit(model.model_body_reduced)]

        def check(q, vel):
            default = solver._ik_reduced(list(solver.frame_tasks.values()) + [solver.posture_task],
                                         solver.dt, default_limits)
            distance = np.minimum(q - low, high - q)
            if np.all(distance * fraction > np.abs(default) * solver.dt + 1e-3):
                np.testing.assert_allclose(vel, default, atol=1e-6)
                compared.append(True)

        for _ in range(8):
            goal = np.clip(start + rng.uniform(-0.3, 0.3, size=start.shape), low + 0.05,
                           high - 0.05)
            step_to(solver, start, goal, 30, check)
        assert len(compared) >= 100
    finally:
        model.state_subscriber = None
        model.set_position_limits(None)
        model.rebuild_reduced_model(np.zeros(model.num_motor))


@pytest.mark.parametrize('robot, base', [('h12', 'home'), ('g1', 't_pose')])
def test_start_up_steps_clip_each_joint_at_the_speed_cap(models, robot, base):
    """A capped start-up step is the uncapped one with each joint clipped at JOINT_SPEED_CAP.

    So each joint closes on its posture at up to the cap on its own, the ones with less to go
    arriving first, instead of the arm moving along a straight joint path. The goal is the
    base posture with every IK joint moved by up to 0.5 rad, away from the limits.
    """
    from golem_upper_ik.core.ik_solver import JOINT_SPEED_CAP
    model = models(robot)
    start = np.asarray(model.profile.named_configs[base], dtype=float)
    rng = np.random.default_rng(5)
    try:
        solver, _ = stepping_solver(models, robot)
        low = model.model_body_reduced.lowerPositionLimit
        high = model.model_body_reduced.upperPositionLimit
        mask = model.reduced_mask
        clipped = 0
        for _ in range(5):
            goal = np.clip(start + rng.uniform(-0.5, 0.5, size=start.shape), low + 0.3, high - 0.3)
            q_body = model.zero_q_body
            q_body[mask] = start
            measured(model, q_body)
            solver.update_configurations()
            capped = solver.goto_reduced_configuration(goal)[mask]
            free = solver.goto_reduced_configuration(goal, capped=False)[mask]
            np.testing.assert_allclose(capped, np.clip(free, -JOINT_SPEED_CAP, JOINT_SPEED_CAP),
                                       atol=1e-6)
            clipped += np.sum(np.abs(free) > JOINT_SPEED_CAP)
        assert clipped > 0
    finally:
        model.state_subscriber = None
        model.set_position_limits(None)
        model.rebuild_reduced_model(np.zeros(model.num_motor))


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_limited_step_keeps_the_limits_when_its_bounds_cannot(models, robot):
    """Bounds that would take a joint past its position limit give way; the limit holds.

    The left elbow sits on its upper limit and the bounds ask it to step 5 to 10 mrad further:
    the step stays inside the limit (within the QP's tolerance) and every other joint keeps
    its bounds.
    """
    model = models(robot)
    try:
        solver, _ = stepping_solver(models, robot)
        mask = model.reduced_mask
        elbow = list(model.profile.enabled_joints).index('left_elbow_joint')
        q = np.asarray(model.profile.named_configs['home'], dtype=float).copy()
        q[elbow] = model.model_body_reduced.upperPositionLimit[elbow]
        q_body = model.zero_q_body
        q_body[mask] = q
        measured(model, q_body)
        solver.update_configurations()
        low, high = np.full(q.size, -0.002), np.full(q.size, 0.002)
        low[elbow], high[elbow] = 0.005, 0.01
        step = solver.limited_step(np.zeros(q.size), low, high)
        assert step is not None
        assert step[elbow] <= 1e-6
        others = np.arange(q.size) != elbow
        assert np.all(step[others] >= low[others] - 1e-6)
        assert np.all(step[others] <= high[others] + 1e-6)
    finally:
        model.state_subscriber = None
        model.set_position_limits(None)
        model.rebuild_reduced_model(np.zeros(model.num_motor))
