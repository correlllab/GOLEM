"""Plan one robot's arm paths for any controller: ArmPlanner, the core of planner_server.

ArmPlanner builds the robot's models once and serves one plan at a time (it is not
thread-safe): a pose goal (each moving arm's grasp-frame target in the pelvis frame) or a joint
goal (the moving arms' joint angles), from a start at the caller's last command, in the
caller's position range, timed at the caller's waypoint period, frame speed and acceleration
bound.

It holds two RobotModels, as golem_upper_ik's controller and its planner held them:
- the IK's: the pose goal's IK runs on its reduced model, re-locked for each goal at the
  measured angles of the joints that do not plan (the torso or waist, the legs), from
  golem_upper_ik's IK seeds in turn ('current', the measured arm angles, then the named
  configs, in two passes: IKSolver.solve_ik_reduced). A converged seed's solution is taken
  only once the planner finds a path to it; otherwise the next seed to converge is tried.
- the planner's: ReducedJointPlanner checks limits, self-collision and the planner frames'
  workspace bounds on it. Its reduced model and self-collision model keep the joints that do
  not plan locked at 0; the measured angles place them only for the workspace bounds.

The planner settings are the controller config's planner section; a request sets the waypoint
period, the frame speed, the acceleration bound and OMPL's budget for its plan. A request in a
position range other than the last one's narrows both models to it and rebuilds the planner.
"""
from dataclasses import dataclass, replace

import numpy as np
from pink.exceptions import NoSolutionFound
import pinocchio as pin

from golem_upper_ik.core.ik_solver import IKSolver
from golem_ompl.reduced_joint_planner import PlannerConfig, ReducedJointPlanner
from golem_upper_ik.core.robot_model import FixedState, RobotModel
from golem_upper_ik.profile import RobotProfile, robot_config

__all__ = ['ARMS', 'ArmPlan', 'ArmPlanner', 'NO_IK_SOLUTION', 'NoSolutionFound', 'PlanRefused',
           'planner_config_of', 'unavailable_reason']

# the arms, in the order of the profile's (left, right) grasp frames
ARMS = ('left', 'right')
# why a robot without a self-collision model cannot plan: {robot} is its display name and
# {module} its name
PLANNER_UNAVAILABLE = (
    'Planning is unavailable: the {robot} has no self-collision model (its profile, '
    'golem_upper_ik/config/{module}/robot.yaml, names no sphere URDF and SRDF); send the '
    'goal with plan=false'
)
# a pose goal's reason when no IK seed converged
NO_IK_SOLUTION = 'IK failed to resolve current targets'


class PlanRefused(ValueError):
    """Signal a request ArmPlanner cannot serve; its message says why."""


@dataclass
class ArmPlan:
    """Hold how a plan ended.

    outcome: 'planned', 'no_ik_solution' or 'no_path'. path: (waypoints, IK joints) rad, one
    waypoint per dt, empty unless planned. message: the planner's reason ('direct path' or
    'success' when planned). seed: a pose goal's IK seed whose solution was planned to, or ''.
    """

    outcome: str
    path: np.ndarray
    message: str
    seed: str = ''


def unavailable_reason(profile: RobotProfile) -> str:
    """Return why a robot without a self-collision model cannot plan."""
    return PLANNER_UNAVAILABLE.format(robot=robot_config(profile.robot)['display_name'],
                                      module=profile.robot)


def planner_config_of(config: dict) -> PlannerConfig:
    """Return the PlannerConfig of a controller config's planner section.

    dt defaults to the control period (frequency.ctrl_hz); requests set dt, the frame speed,
    the acceleration bound and the timeout for their own plans.
    """
    settings = {key: value for key, value in config['planner'].items()
                if key not in ('enabled', 'frame_names')}
    settings.setdefault('dt', 1.0 / float(config['frequency']['ctrl_hz']))
    return PlannerConfig(**settings, frame_names=tuple(config['planner']['frame_names']))


class ArmPlanner:
    """Solve and plan one robot's arm goals (module doc)."""

    def __init__(self, profile: RobotProfile, config: dict):
        """Build the IK's and the planner's models in the config's clip range.

        config: the robot's controller config (golem_upper_ik.config.load_controller_config):
        its planner section, controller.d_min, controller.timeout (the default IK budget),
        controller.imu_frame and the clip range. PlanRefused when the robot has no
        self-collision model.
        """
        if config['robot'] != profile.robot:
            raise ValueError(f"the controller config is for robot {config['robot']}, not "
                             f'{profile.robot}')
        if profile.collision is None:
            raise PlanRefused(unavailable_reason(profile))
        self.profile = profile
        self.config = config
        controller = config['controller']
        self.default_ik_timeout = float(controller['timeout'])
        self.base_planner_config = planner_config_of(config)
        # the position range the models are narrowed to, as given: the config's clip range
        # until a request gives another
        self.limits = np.asarray(config['limits']['q_clip_limits'], dtype=float)

        self.ik_model = RobotModel(profile, imu_frame=controller['imu_frame'])
        # (body joints, 2) rad: the URDF's limits, the range of a request that gives none
        self.urdf_limits = np.column_stack((self.ik_model.model_body.lowerPositionLimit,
                                            self.ik_model.model_body.upperPositionLimit))
        self.ik_model.set_position_limits(self.limits)
        self.ik_model.state_subscriber = FixedState(self.ik_model.zero_q_body)
        self.ik_solver = IKSolver(self.ik_model, profile.named_configs,
                                  dt=1.0 / float(config['frequency']['ctrl_hz']),
                                  d_min=controller['d_min'])
        self.plan_model = RobotModel(profile, q_limits=self.limits)
        self.planner = ReducedJointPlanner(self.plan_model, config=self.base_planner_config)

        body = list(profile.body_joints)
        mask = self.ik_model.reduced_mask
        # the IK joints in reduced (slot) order: a path's columns
        self.arm_joints = tuple(name for name, kept in zip(body, mask) if kept)
        self.grasp_frames = {arm: frame.name for arm, frame in zip(ARMS, profile.grasp_frames)}
        # each arm's IK joints, by name
        self.arm_joint_names = {arm: tuple(name for name in self.arm_joints
                                           if name.startswith(f'{arm}_')) for arm in ARMS}

    def set_limits(self, limits):
        """Narrow both models to limits ((body joints, 2) rad, within the URDF's limits).

        Nothing is rebuilt when the range is the last one's; otherwise the IK rebinds to the IK
        model's rebuilt reduced model and the planner is rebuilt on the planner model's.
        PlanRefused for a range RobotModel refuses (mis-shaped, non-finite, or leaving a joint
        no range within the URDF's limits).
        """
        limits = np.asarray(limits, dtype=float)
        if limits.shape == self.limits.shape and np.array_equal(limits, self.limits):
            return
        try:
            self.ik_model.set_position_limits(limits)
            self.plan_model.set_position_limits(limits)
        except ValueError as error:
            # the models keep the last range: put back what either took of the new one
            self.ik_model.set_position_limits(self.limits)
            self.plan_model.set_position_limits(self.limits)
            raise PlanRefused(str(error)) from None
        self.ik_solver.rebuild_reduced()
        self.planner = ReducedJointPlanner(self.plan_model, config=self.planner.config)
        self.limits = limits

    def _set_timing(self, dt, frame_speed, max_acceleration, plan_timeout):
        """Set the planner's waypoint period, speed, acceleration bound and OMPL budget.

        dt in s of robot time (> 0); frame_speed in m/s (None: the config's moving_speed);
        max_acceleration in rad/s^2 (None: no bound); plan_timeout in s of steady time (None:
        the config's timeout).
        """
        base = self.base_planner_config
        self.planner.config = replace(
            base, dt=float(dt),
            moving_speed=base.moving_speed if frame_speed is None else float(frame_speed),
            max_acceleration=None if max_acceleration is None else float(max_acceleration),
            timeout=base.timeout if plan_timeout is None else float(plan_timeout))
        self.planner._check_planner_config()

    def _set_measured(self, measured_q):
        """Hold measured_q (rad, per body joint) as both models' state."""
        measured_q = np.asarray(measured_q, dtype=float)
        self.plan_model.state_subscriber = FixedState(measured_q)
        self.plan_model.update_kinematics()
        self.ik_model.state_subscriber = FixedState(measured_q)
        self.ik_model.update_kinematics()

    def _arm_mask(self, arms):
        """Return the reduced-joint mask of the joints that move the arms' grasp frames."""
        model = self.ik_model.model_body_reduced
        mask = np.zeros(model.nq, dtype=bool)
        for arm in arms:
            joint_id = model.frames[model.getFrameId(self.grasp_frames[arm])].parentJoint
            while joint_id > 0:
                idx_q = model.joints[joint_id].idx_q
                mask[idx_q:idx_q + model.joints[joint_id].nq] = True
                joint_id = model.parents[joint_id]
        return mask

    def plan_pose_goal(self, measured_q, start, targets, timing, ik_timeout=None, log=None):
        """Return an ArmPlan to an IK solution of targets ({arm: 4x4 in the pelvis frame}).

        measured_q: rad per body joint; start: rad per IK joint (the caller's last command);
        timing: (dt, frame_speed, max_acceleration, plan_timeout) as _set_timing takes them;
        ik_timeout: each seed's budget, s of steady time (None: controller.timeout). Only the
        joints that move the targets' grasp frames plan; the others stay at start. log(text),
        when given, hears each converged seed the planner found no path to. NoSolutionFound
        when an IK iteration has no QP solution.
        """
        self._set_timing(*timing)
        self._set_measured(measured_q)
        ik_model, solver = self.ik_model, self.ik_solver
        # re-lock the frozen joints at their measured angles, so the IK's forward kinematics
        # agree with the robot's
        ik_model.rebuild_reduced_model()
        solver.rebuild_reduced()
        q_seed = np.asarray(measured_q, dtype=float).copy()
        q_seed[ik_model.reduced_mask] = start
        solver.update_configurations(q_seed)
        solver.clear_frame_tasks()
        for arm, target in targets.items():
            solver.add_frame_task(f'{self.grasp_frames[arm]}_task', self.grasp_frames[arm],
                                  target=pin.SE3(np.asarray(target, dtype=float)))
        active_mask = self._arm_mask(targets)
        planned = {}

        def plannable(ik_result):
            """Plan to a converged seed's solution; return whether a path was found."""
            result = self.planner.plan(start, ik_result.q[ik_model.reduced_mask],
                                       active_mask=active_mask)
            planned['result'], planned['seed'] = result, ik_result.seed_name
            if not result.success and log is not None:
                log(f"The IK converged from seed '{ik_result.seed_name}', but planning to it "
                    f'failed: {result.reason}; trying the next seed')
            return result.success

        timeout = self.default_ik_timeout if ik_timeout is None else float(ik_timeout)
        solver.solve_ik_reduced(timeout=timeout, accept=plannable)
        if 'result' not in planned:
            return ArmPlan('no_ik_solution', self._empty_path(), NO_IK_SOLUTION)
        return self._arm_plan(planned['result'], planned['seed'])

    def plan_joint_goal(self, measured_q, start, goal, arms, timing):
        """Return an ArmPlan from start to goal (rad per IK joint) moving only arms.

        measured_q: rad per body joint; the joints of the other arm stay at start. timing as
        plan_pose_goal's.
        """
        self._set_timing(*timing)
        self._set_measured(measured_q)
        return self._arm_plan(self.planner.plan(start, goal, active_mask=self._arm_mask(arms)))

    def _arm_plan(self, result, seed=''):
        """Return the ArmPlan of a planner PlanResult."""
        if result.success:
            return ArmPlan('planned', np.asarray(result.path, dtype=float), result.reason, seed)
        return ArmPlan('no_path', self._empty_path(), result.reason, seed)

    def _empty_path(self):
        """Return a path with no waypoints."""
        return np.empty((0, len(self.arm_joints)))

