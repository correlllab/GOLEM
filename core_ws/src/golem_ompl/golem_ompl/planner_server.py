"""Serve one robot's arm planner as a ROS node: planner_server.

    ros2 run golem_ompl planner_server --ros-args -p robot:=h12 -p config:=safety_split

Parameters: robot (required; a robot with data in golem_upper_ik's config/) and config
(required; a golem_upper_ik controller config name under share/golem_upper_ik/config/<robot>/,
or a path), whose planner section, controller.d_min, controller.timeout and clip range the
planner runs on (ArmPlanner).

At start the node builds the robot's models in the config's clip range, then serves
/plan_arm_trajectory (custom_ros_messages/PlanArmTrajectory), so the service appearing means
the planner is ready. The service has a callback group of its own: one plan runs at a time
and later requests wait their turn, while the node's other callbacks go on. A robot without a
self-collision model serves the service and refuses every request (ERROR). Every request's
budgets are s of steady time; the returned path is timed in s of robot time.
"""
import os

# One BLAS thread before numpy loads, as frame_task_server: the planner is single-threaded
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import math  # noqa: E402
import time  # noqa: E402

from builtin_interfaces.msg import Duration  # noqa: E402
from custom_ros_messages.srv import PlanArmTrajectory  # noqa: E402
import numpy as np  # noqa: E402
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup  # noqa: E402
from rclpy.node import Node  # noqa: E402
from trajectory_msgs.msg import JointTrajectoryPoint  # noqa: E402

from golem_ompl.arm_planner import (  # noqa: E402
    ARMS, ArmPlanner, NoSolutionFound, PlanRefused, unavailable_reason,
)
from golem_upper_ik.config import load_controller_config  # noqa: E402
from golem_upper_ik.profile import load_profile  # noqa: E402
from golem_upper_ik.ros2.utility import pose_to_matrix, robot_parameter, run  # noqa: E402

__all__ = ['PLAN_SERVICE', 'PlannerServer', 'main', 'parse_request']

# the service: a global name, which any arm controller calls
PLAN_SERVICE = '/plan_arm_trajectory'
# the frame every pose goal is in
POSE_FRAME = 'pelvis'
# the response's outcome for each ArmPlan outcome
OUTCOMES = {
    'planned': PlanArmTrajectory.Response.PLANNED,
    'no_ik_solution': PlanArmTrajectory.Response.NO_IK_SOLUTION,
    'no_path': PlanArmTrajectory.Response.NO_PATH,
}


def _by_name(state, names, what):
    """Return state's positions (a JointState) in the order of names, rad.

    PlanRefused unless state names every one of names, no other and none
    twice, with one finite position each.
    """
    given = list(state.name)
    positions = list(state.position)
    if len(positions) != len(given):
        raise PlanRefused(f'{what} names {len(given)} joints but gives {len(positions)} '
                          'positions')
    if len(set(given)) != len(given):
        raise PlanRefused(f'{what} names a joint twice')
    unknown = sorted(set(given) - set(names))
    if unknown:
        raise PlanRefused(f'{what} names joints that are not {what} joints of this robot: '
                          f'{unknown}')
    missing = [name for name in names if name not in given]
    if missing:
        raise PlanRefused(f'{what} is missing joints {missing}')
    values = np.array([positions[given.index(name)] for name in names], dtype=float)
    if not np.all(np.isfinite(values)):
        raise PlanRefused(f'{what} has a position that is not finite')
    return values


def _non_negative(request, field):
    """Return request.field as a float, None for 0; PlanRefused if negative or not finite."""
    value = float(getattr(request, field))
    if not math.isfinite(value) or value < 0.0:
        raise PlanRefused(f'{field} must be finite and not negative, not {value}')
    return None if value == 0.0 else value


def parse_request(request, planner: ArmPlanner) -> dict:
    """Return a PlanArmTrajectory request as ArmPlanner's arguments; PlanRefused if malformed.

    Returns measured_q (rad per body joint), start (rad per IK joint), limits ((body joints,
    2) rad), timing (dt, frame_speed, max_acceleration, plan_timeout), ik_timeout, and either
    targets ({arm: 4x4} for a pose goal) or goal and arms (a joint goal).
    """
    profile = planner.profile
    body = list(profile.body_joints)
    measured_q = _by_name(request.measured, body, 'measured')
    start = _by_name(request.start, planner.arm_joints, 'start')

    lower, upper = list(request.position_lower), list(request.position_upper)
    if not lower and not upper:
        limits = planner.urdf_limits
    else:
        if len(lower) != len(body) or len(upper) != len(body):
            raise PlanRefused(f'position_lower and position_upper must give one value per joint '
                              f'of measured.name ({len(body)}), not {len(lower)} and '
                              f'{len(upper)}')
        order = [list(request.measured.name).index(name) for name in body]
        limits = np.column_stack((np.asarray(lower, dtype=float)[order],
                                  np.asarray(upper, dtype=float)[order]))

    dt = float(request.dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise PlanRefused(f'dt must be positive, not {dt}')
    parsed = {
        'measured_q': measured_q, 'start': start, 'limits': limits,
        'timing': (dt, _non_negative(request, 'frame_speed'),
                   _non_negative(request, 'max_acceleration'),
                   _non_negative(request, 'plan_timeout')),
        'ik_timeout': _non_negative(request, 'ik_timeout'),
    }

    poses = {arm: getattr(request, f'{arm}_pose') for arm in ARMS}
    targeted = {arm: pose for arm, pose in poses.items() if pose.header.frame_id}
    named = list(request.goal_joints.name)
    if targeted and named:
        raise PlanRefused('the request gives both a pose goal and a joint goal')
    if not targeted and not named:
        raise PlanRefused('the request gives no goal: a pose (left_pose, right_pose) or '
                          'goal_joints')
    if targeted:
        targets = {}
        for arm, pose in targeted.items():
            if pose.header.frame_id != POSE_FRAME:
                raise PlanRefused(f"the {arm} pose is in '{pose.header.frame_id}', not in "
                                  f"'{POSE_FRAME}'")
            position, orientation = pose.pose.position, pose.pose.orientation
            values = np.array([position.x, position.y, position.z, orientation.w,
                               orientation.x, orientation.y, orientation.z])
            if not np.all(np.isfinite(values)):
                raise PlanRefused(f'the {arm} pose is not finite')
            if np.linalg.norm(values[3:]) < 1e-6:
                raise PlanRefused(f'the {arm} pose has no orientation (a zero quaternion)')
            targets[arm] = pose_to_matrix(pose.pose)
        parsed['targets'] = targets
        return parsed

    arms = [arm for arm in ARMS
            if any(name in planner.arm_joint_names[arm] for name in named)]
    moving = [name for arm in arms for name in planner.arm_joint_names[arm]]
    values = _by_name(request.goal_joints, moving, 'goal_joints')
    goal = start.copy()
    for name, value in zip(moving, values):
        goal[planner.arm_joints.index(name)] = value
    parsed['goal'], parsed['arms'] = goal, arms
    return parsed


def _duration(seconds):
    """Return seconds (>= 0) as a builtin_interfaces Duration."""
    nanoseconds = int(round(seconds * 1e9))
    return Duration(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


class PlannerServer(Node):
    """Serve /plan_arm_trajectory (module doc)."""

    def __init__(self, **node_kwargs):
        """Read the parameters and the config; start() builds the planner and the service."""
        super().__init__('planner_server', **node_kwargs)
        self.robot = robot_parameter(self)
        self.profile = load_profile(self.robot)
        self.config_name = self.declare_parameter('config', '').value
        if not self.config_name:
            raise ValueError(f'robot {self.robot}: the config parameter is required: a '
                             'golem_upper_ik controller config name or path, e.g. '
                             'config:=safety_split')
        self.summary = f'robot {self.robot}, config {self.config_name}'
        self.config = load_controller_config(self.robot, self.config_name)
        self.planner = None
        self.unavailable = None
        self.service = None

    def start(self):
        """Build the planner's models (seconds), then serve the service."""
        started = time.monotonic()
        if self.profile.collision is None:
            self.unavailable = unavailable_reason(self.profile)
        else:
            self.planner = ArmPlanner(self.profile, self.config)
        self.service = self.create_service(
            PlanArmTrajectory, PLAN_SERVICE, self.plan_callback,
            callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f'Planner server ready ({self.summary}; models built in '
                               f'{time.monotonic() - started:.1f} s)')

    def shutdown(self):
        """Do nothing: a plan in progress ends with the process."""

    def plan_callback(self, request, response):
        """Plan one request into response (the service callback); every end sets a message."""
        Response = PlanArmTrajectory.Response
        started = time.monotonic()
        try:
            if self.unavailable is not None:
                raise PlanRefused(self.unavailable)
            parsed = parse_request(request, self.planner)
            self.planner.set_limits(parsed['limits'])
            if 'targets' in parsed:
                kind = 'pose goal (' + ', '.join(parsed['targets']) + ')'
                plan = self.planner.plan_pose_goal(
                    parsed['measured_q'], parsed['start'], parsed['targets'], parsed['timing'],
                    ik_timeout=parsed['ik_timeout'], log=self.get_logger().info)
            else:
                kind = 'joint goal (' + ', '.join(parsed['arms']) + ')'
                plan = self.planner.plan_joint_goal(
                    parsed['measured_q'], parsed['start'], parsed['goal'], parsed['arms'],
                    parsed['timing'])
        except PlanRefused as error:
            return self._refuse(response, str(error))
        except NoSolutionFound as error:
            response.outcome, response.message = Response.NO_IK_SOLUTION, str(error)
            self.get_logger().warn(f'the IK has no solution: {error}')
            return response
        except Exception as error:  # the caller hears it; the node goes on serving
            self.get_logger().error(f'planning failed: {error!r}')
            return self._refuse(response, f'the planner failed: {error!r}')

        response.outcome = OUTCOMES[plan.outcome]
        response.success = plan.outcome == 'planned'
        response.message = plan.message
        response.ik_seed = plan.seed
        if response.success:
            dt = parsed['timing'][0]
            names = list(self.planner.arm_joints)
            response.trajectory.joint_names = names
            response.trajectory.points = [
                JointTrajectoryPoint(positions=[float(value) for value in row],
                                     time_from_start=_duration(index * dt))
                for index, row in enumerate(plan.path)]
            response.goal.name = names
            response.goal.position = [float(value) for value in plan.path[-1]]
        self.get_logger().info(
            f'{kind}: {plan.outcome} ({plan.message}), {len(plan.path)} waypoints, '
            f'{time.monotonic() - started:.2f} s')
        return response

    def _refuse(self, response, message):
        """Return response as an ERROR with message, and log it."""
        response.outcome = PlanArmTrajectory.Response.ERROR
        response.success = False
        response.message = message
        self.get_logger().warn(f'request refused: {message}')
        return response


def main(args=None):
    """Run the node until SIGINT or SIGTERM (golem_upper_ik's utility.run)."""
    run('planner_server', PlannerServer, args=args)


if __name__ == '__main__':
    main()
