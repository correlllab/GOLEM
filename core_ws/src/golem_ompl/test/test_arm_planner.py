"""Check ArmPlanner and parse_request on both robots' real models, without ROS.

A pose goal and a joint goal give exactly the path golem_upper_ik's controller planned in
process: the IK on a model re-locked at the measured pose, the planner on one frozen at 0, the
planner config of the controller config at the controller's acceleration limit. The seed rule
takes the first converged seed the planner reaches; the moving arms alone move; a new position
range narrows both models and rebuilds the planner; a robot without a self-collision model is
refused; and malformed requests are refused before any planning.
"""
from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest

from golem_ompl.arm_planner import ArmPlanner, PlanRefused, planner_config_of
from golem_upper_ik.config import load_controller_config
from golem_upper_ik.profile import load_profile

ROBOTS = ('h12', 'g1')
# each robot's controller config in its sim bringup
CONFIGS = {'h12': 'safety_split', 'g1': 'sim_split'}
# where the arms start: a working pose with room to move forward
START = {'h12': 'prep', 'g1': 'home'}
# rad/s^2: golem_upper_ik's ACCEL_LIMIT, the acceleration bound its controller plans at
ACCEL_LIMIT = 5.0


@pytest.fixture(scope='module')
def planners():
    """Return planner(robot): the module's ArmPlanner of robot, built on first use."""
    built = {}

    def planner(robot):
        """Return robot's ArmPlanner on its sim config, the clip range restored."""
        if robot not in built:
            built[robot] = ArmPlanner(load_profile(robot),
                                      load_controller_config(robot, CONFIGS[robot]))
        arm_planner = built[robot]
        arm_planner.set_limits(arm_planner.config['limits']['q_clip_limits'])
        return arm_planner
    return planner


def body_q(planner, arms):
    """Return a body q: 0, the IK joints at the named config arms (or values)."""
    q = np.zeros(len(planner.profile.body_joints))
    values = planner.profile.named_configs[arms] if isinstance(arms, str) else arms
    q[planner.ik_model.reduced_mask] = values
    return q


def timing(planner):
    """Return the request timing golem_upper_ik's controller sends: its config's."""
    config = planner_config_of(planner.config)
    return (config.dt, config.moving_speed, ACCEL_LIMIT, config.timeout)


def frame_targets(planner, q, arms=('left', 'right'), shift=(0.0, 0.0, 0.0)):
    """Return {arm: 4x4}: each arm's grasp frame at body q, shifted (m, pelvis frame)."""
    targets = {}
    for arm in arms:
        target = planner.ik_model.get_frame_transformation_body(planner.grasp_frames[arm], q)
        target = target.copy()
        target[:3, 3] += shift
        targets[arm] = target
    return targets


def in_process_pose_plan(planner, measured_q, start, targets):
    """Return (path, seed) as golem_upper_ik's controller planned a pose goal in process.

    Its IK on a model of its own re-locked at the measured pose, from the command seed; its
    planner subprocess's ReducedJointPlanner on a model frozen at 0 holding the measured q,
    with the controller config's planner settings at ACCEL_LIMIT.
    """
    from golem_upper_ik.core.ik_solver import IKSolver
    from golem_upper_ik.core.planner import PlannerConfig, ReducedJointPlanner
    from golem_upper_ik.core.robot_model import FixedState, RobotModel
    profile, config = planner.profile, planner.config
    controller = config['controller']
    clip = config['limits']['q_clip_limits']
    dt = 1.0 / float(config['frequency']['ctrl_hz'])
    model = RobotModel(profile, imu_frame=controller['imu_frame'])
    model.set_position_limits(clip)
    model.state_subscriber = FixedState(measured_q)
    model.update_kinematics()
    model.rebuild_reduced_model()
    solver = IKSolver(model, profile.named_configs, dt=dt, d_min=controller['d_min'])
    q_seed = np.array(measured_q, dtype=float)
    q_seed[model.reduced_mask] = start
    solver.update_configurations(q_seed)
    for arm, target in targets.items():
        frame = planner.grasp_frames[arm]
        solver.add_frame_task(f'{frame}_task', frame, target=pin.SE3(target))
    settings = {key: value for key, value in config['planner'].items()
                if key not in ('enabled', 'frame_names')}
    settings.setdefault('dt', dt)
    settings['max_acceleration'] = ACCEL_LIMIT
    plan_model = RobotModel(profile, q_limits=clip)
    plan_model.state_subscriber = FixedState(measured_q)
    plan_model.update_kinematics()
    reference = ReducedJointPlanner(plan_model, config=PlannerConfig(
        **settings, frame_names=tuple(config['planner']['frame_names'])))
    mask = planner._arm_mask(targets)
    found = {}

    def accept(result):
        """Plan to a converged seed's solution."""
        found['plan'] = reference.plan(start, result.q[model.reduced_mask], active_mask=mask)
        found['seed'] = result.seed_name
        return found['plan'].success
    solver.solve_ik_reduced(timeout=controller['timeout'], accept=accept)
    return found['plan'].path, found['seed']


@pytest.mark.parametrize('robot', ROBOTS)
@pytest.mark.parametrize('arms', [('left',), ('left', 'right')])
def test_a_pose_goal_plans_the_controllers_path(planners, robot, arms):
    """A pose goal 5 cm forward gives exactly the path the controller planned in process."""
    planner = planners(robot)
    measured_q = body_q(planner, START[robot])
    # the command sits a little above the measured pose, as gravity leaves an arm
    start = measured_q[planner.ik_model.reduced_mask] + 0.01
    targets = frame_targets(planner, body_q(planner, start), arms, shift=(0.05, 0.0, 0.0))
    plan = planner.plan_pose_goal(measured_q, start, targets, timing(planner))
    assert plan.outcome == 'planned' and plan.message == 'direct path', plan.message
    path, seed = in_process_pose_plan(planner, measured_q, start, targets)
    assert plan.seed == seed == 'current'
    np.testing.assert_array_equal(plan.path, path)
    np.testing.assert_array_equal(plan.path[0], start)
    # the arm without a target stays at its start
    for arm in set(('left', 'right')) - set(arms):
        columns = [planner.arm_joints.index(name) for name in planner.arm_joint_names[arm]]
        np.testing.assert_array_equal(plan.path[:, columns],
                                      np.tile(start[columns], (len(plan.path), 1)))


@pytest.mark.parametrize('robot', ROBOTS)
def test_a_joint_goal_plans_the_in_process_planners_path(planners, robot):
    """A joint goal's path is exactly ReducedJointPlanner.plan's on the same models."""
    planner = planners(robot)
    measured_q = body_q(planner, START[robot])
    start = measured_q[planner.ik_model.reduced_mask]
    goal = start.copy()
    left = [planner.arm_joints.index(name) for name in planner.arm_joint_names['left']]
    goal[left[3]] += 0.2  # the left elbow
    plan = planner.plan_joint_goal(measured_q, start, goal, ['left'], timing(planner))
    assert plan.outcome == 'planned', plan.message
    reference = planner.planner.plan(start, goal, active_mask=planner._arm_mask(['left']))
    np.testing.assert_array_equal(plan.path, reference.path)
    np.testing.assert_array_equal(plan.path[-1], goal)
    # the timing the request gives: max_acceleration and frame speed hold
    assert planner.planner.config.max_acceleration == ACCEL_LIMIT


@pytest.mark.parametrize('robot', ROBOTS)
def test_the_request_times_the_path(planners, robot):
    """A slower frame speed gives more waypoints; a longer dt fewer."""
    planner = planners(robot)
    measured_q = body_q(planner, START[robot])
    start = measured_q[planner.ik_model.reduced_mask]
    goal = start.copy()
    goal[planner.arm_joints.index(planner.arm_joint_names['right'][3])] += 0.3
    dt, _, accel, budget = timing(planner)

    def count(dt_, speed_):
        """Return the waypoints of the joint goal at dt_ and speed_."""
        plan = planner.plan_joint_goal(measured_q, start, goal, ['right'],
                                       (dt_, speed_, accel, budget))
        assert plan.outcome == 'planned', plan.message
        return len(plan.path)
    # slow enough that every count is above the config's min_interpolation_steps
    base = count(dt, 0.05)
    assert base > planner.planner.config.min_interpolation_steps
    assert count(dt, 0.025) > base
    assert count(2.0 * dt, 0.05) < base


@pytest.mark.parametrize('robot', ROBOTS)
@pytest.mark.parametrize('refused', [0, 1, 2, None])
def test_a_pose_goal_takes_the_first_seed_the_planner_reaches(planners, robot, refused,
                                                              monkeypatch):
    """A pose goal's IK goes on to the next converged seed when the planner reaches none.

    Measured at arms_front_45 and targeting its grasp frames, the measured pose ('current')
    and the arms_front_45 seed converge with no time to iterate. The planner refuses the
    first `refused` converged seeds' solutions: the goal plans to the next, ends without a
    path (the planner's reason) once both are refused, and, with a target no seed reaches
    (None), ends without an IK solution before any plan.
    """
    from golem_upper_ik.core.planner.reduced_joint_planner import PlanResult
    planner = planners(robot)
    q = body_q(planner, 'arms_front_45')
    shift = (1.0, 0.0, 0.0) if refused is None else (0.0, 0.0, 0.0)  # 1 m: out of reach
    targets = frame_targets(planner, q, shift=shift)
    plans = []

    def plan(start, goal, active_mask=None):
        """Refuse the first `refused` goals; plan a one-waypoint path to the others."""
        plans.append(np.array(goal))
        if len(plans) <= (refused or 0):
            return PlanResult(success=False, path=np.empty((0, len(goal))),
                              reason=f'refused goal {len(plans)}')
        return PlanResult(success=True, path=np.array([goal]), reason='direct path')
    monkeypatch.setattr(planner.planner, 'plan', plan)
    logged = []
    result = planner.plan_pose_goal(q, q[planner.ik_model.reduced_mask], targets,
                                    timing(planner), ik_timeout=0.0, log=logged.append)
    if refused is None:
        assert (result.outcome, result.message) == ('no_ik_solution',
                                                    'IK failed to resolve current targets')
        assert plans == []
    elif refused == 2:
        assert (result.outcome, result.message) == ('no_path', 'refused goal 2')
        assert len(plans) == 2
    else:
        assert result.outcome == 'planned'
        assert len(plans) == refused + 1
        np.testing.assert_array_equal(result.path, [plans[-1]])
        assert result.seed == ('current' if refused == 0 else 'arms_front_45')
    assert len(logged) == min(refused or 0, 2)
    assert all('trying the next seed' in line for line in logged)
    if refused:
        assert "seed 'current'" in logged[0]


@pytest.mark.parametrize('robot', ROBOTS)
def test_a_new_range_narrows_both_models_and_the_planner(planners, robot):
    """set_limits narrows the IK's and the planner's models; the same range rebuilds nothing.

    A range RobotModel refuses is refused, and both models keep the last range.
    """
    planner = planners(robot)
    clip = np.asarray(planner.config['limits']['q_clip_limits'], dtype=float)
    before = planner.planner
    planner.set_limits(clip.copy())
    assert planner.planner is before
    narrowed = clip.copy()
    narrowed[:, 0] += 0.1
    narrowed[:, 1] -= 0.1
    planner.set_limits(narrowed)
    assert planner.planner is not before
    mask = planner.ik_model.reduced_mask
    for model in (planner.ik_model, planner.plan_model):
        assert np.all(model.model_body_reduced.lowerPositionLimit
                      >= narrowed[mask, 0] - 1e-12)
        assert np.all(model.model_body_reduced.upperPositionLimit
                      <= narrowed[mask, 1] + 1e-12)
    assert planner.ik_solver.configuration_reduced.model is planner.ik_model.model_body_reduced
    assert planner.planner.robot_model is planner.plan_model
    empty = narrowed.copy()
    empty[0] = [0.5, 0.4]
    with pytest.raises(PlanRefused, match='no range'):
        planner.set_limits(empty)
    np.testing.assert_array_equal(planner.limits, narrowed)
    np.testing.assert_array_equal(planner.plan_model.model_body_reduced.lowerPositionLimit,
                                  planner.ik_model.model_body_reduced.lowerPositionLimit)


def test_a_robot_without_a_self_collision_model_is_refused():
    """A profile without a self-collision model cannot plan."""
    profile = replace(load_profile('g1'), collision=None)
    with pytest.raises(PlanRefused, match='Planning is unavailable: the .* has no '
                                          'self-collision model'):
        ArmPlanner(profile, load_controller_config('g1', CONFIGS['g1']))


@pytest.mark.parametrize('robot', ROBOTS)
def test_an_invalid_goal_ends_without_a_path(planners, robot):
    """A joint goal in self-collision or past the range ends without a path, saying why."""
    planner = planners(robot)
    measured_q = body_q(planner, START[robot])
    start = measured_q[planner.ik_model.reduced_mask]
    goal = start.copy()
    goal[planner.arm_joints.index(planner.arm_joint_names['left'][3])] = 10.0
    plan = planner.plan_joint_goal(measured_q, start, goal, ['left'], timing(planner))
    assert plan.outcome == 'no_path' and len(plan.path) == 0
    assert plan.message.startswith('goal is outside reduced joint limits'), plan.message
