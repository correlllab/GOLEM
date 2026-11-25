"""Serve the upper-body controller of one robot as a ROS node: frame_task_server.

    ros2 run golem_upper_ik frame_task_server --ros-args -p robot:=h12 -p config:=safety_split

Parameters: robot (required; a robot with data in golem_upper_ik's config/) and config
(required; a controller config name under share/golem_upper_ik/config/<robot>/, or a path).

Once its start-up is done (UpperController), the node serves:
- /frame_task (custom_ros_messages/FrameTask): pull named frames to target poses in the
  pelvis frame, directly through the IK or along an OMPL plan, then hold them;
- /named_config (custom_ros_messages/NamedConfig): move the IK joints to a named
  configuration, the same two ways;
- /grasp_frame_target (custom_ros_messages/GraspFrameTarget): move each arm's grasp frame
  (the profile's grasp_frames) to a target in the pelvis frame or, resolved through TF when
  the goal starts, in a world frame, the same two ways; it succeeds once every target has
  been held within the thresholds for GRASP_SETTLE_S, and aborts at its timeout;
and publishes at 25 Hz frame_names, frame_targets and frame_poses (the current goal's
frames, their targets and their measured poses), {left,right}_ee_pose (the end effectors'
measured poses, which tell the robot's other nodes the controller is up),
{left,right}_ee_target (the current goal's targets for the end-effector frames) and
/{left,right}_grasp_frame_pose (the grasp frames' measured poses, every body joint measured,
as TF has them). Every pose is in the pelvis frame. A goal's duration or timeout and the
control period are s of robot time (the controller's clock: the sim's /clock under
use_sim_time, the steady clock otherwise).
"""
import os

# One BLAS thread before numpy loads: OpenBLAS's default of a busy-waiting worker per core
# stalls the 500 Hz command publisher by tens of ms
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import threading  # noqa: E402
import time  # noqa: E402
from typing import Callable, NamedTuple  # noqa: E402

from custom_ros_messages.action import FrameTask, GraspFrameTarget, NamedConfig  # noqa: E402
from custom_ros_messages.msg import StringArray  # noqa: E402
from geometry_msgs.msg import PoseArray, PoseStamped  # noqa: E402
import numpy as np  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.time import Time  # noqa: E402
from tf2_ros import Buffer, TransformException, TransformListener  # noqa: E402

from golem_upper_ik.config import load_controller_config  # noqa: E402
from golem_upper_ik.core.controller.frame_controller import FrameController  # noqa: E402
from golem_upper_ik.core.ros_io import StaleStateError  # noqa: E402
from golem_upper_ik.profile import load_profile  # noqa: E402
from golem_upper_ik.ros2.utility import (  # noqa: E402
    matrix_to_pose, pose_to_matrix, robot_parameter, run,
)

__all__ = ['CATCH_UP_STEPS', 'FrameTaskServer', 'GRASP_SETTLE_S', 'GoalError', 'SLOW_MODE_SCALE',
           'main']

# Speed multiplier for a goal that asks for slow_mode. A direct move scales its joint
# velocity by it; a planned move is planned at a proportionally lower planner-frame speed
# (more waypoints, same control period), so the arm stays smooth instead of stepping.
SLOW_MODE_SCALE = 0.25
# The most IK steps one iteration of a direct goal runs. An iteration that outlasts the
# control period (a loaded host, the executor's callbacks holding the GIL) owes the IK one
# step per period it took, so the arm keeps its configured speed in robot time. SAFETY: each
# step is solved and integrated over one period, inside its own joint limits and
# self-collision barrier; this bounds the displacement one command carries to that many
# steps', and a longer stall slows the move instead of jumping it.
CATCH_UP_STEPS = 3
# the pose topics; their forward kinematics share the GIL with the IK loop
PUBLISH_HZ = 25.0
# the frame every published pose is in
POSE_FRAME = 'pelvis'
# s between log lines while publishing the poses keeps failing
PUBLISH_ERROR_LOG_PERIOD = 1.0
# the arms, in the order of the profile's (left, right) grasp frames
ARMS = ('left', 'right')
# the grasp-frame action and pose topics: global names, which every arm controller serves
GRASP_ACTION = '/grasp_frame_target'
GRASP_POSE_TOPICS = ('/left_grasp_frame_pose', '/right_grasp_frame_pose')
# s of robot time every targeted grasp frame must stay within threshold_linear and
# threshold_angular before a /grasp_frame_target goal succeeds
GRASP_SETTLE_S = 0.2
# s of steady time a goal with a world-frame target listens to TF for its transforms
TF_WAIT_S = 1.0
# s of robot time: a world-frame target whose newest transform is older is refused
TF_MAX_AGE_S = 0.5
# s of steady time between TF lookups while waiting
TF_POLL_S = 0.02


class GoalError(Exception):
    """Signal a /grasp_frame_target goal that cannot run; its message says why."""


class _GoalKind(NamedTuple):
    """Hold what one action's goals differ in, for the goal loop both actions share.

    result: the action's Result type. target: what a planned goal plans to, for the log.
    error(): the goal's error now; feedback(goal_handle, error) publishes it;
    reached(error) says whether it is within the action's thresholds.
    planned_step_converges, direct_step_converges: whether the hold step's own convergence
    ends a planned or a direct goal. direct_reached_message: logged when a direct goal
    reaches its target, or None. settle: s of robot time the target must stay reached before
    the goal is.
    """

    result: type
    target: str
    error: Callable
    feedback: Callable
    reached: Callable
    planned_step_converges: bool
    direct_reached_message: str | None
    direct_step_converges: bool = True
    settle: float = 0.0


class _Outcome(NamedTuple):
    """Hold how a goal's drive ended (FrameTaskServer._drive).

    canceled: a cancel ended it. ik_converged: the IK had converged and the hold had started.
    reached: the hold reached the target. blocked: the joints pinning the hold at a position
    limit, or None.
    """

    canceled: bool
    ik_converged: bool
    reached: bool
    blocked: str | None


class FrameTaskServer(Node):
    """Serve /frame_task, /named_config and /grasp_frame_target and publish the poses."""

    def __init__(self, *, robot_model=None, init=True, **node_kwargs):
        """Read the parameters and the config; start() builds the controller and the topics.

        robot_model: a RobotModel of the robot to use instead of building one.
        init: run the controller's start-up moves.
        """
        super().__init__('frame_task_server', **node_kwargs)
        self.robot = robot_parameter(self)
        self.profile = load_profile(self.robot)
        self.config_name = self.declare_parameter('config', '').value
        if not self.config_name:
            raise ValueError(f'robot {self.robot}: the config parameter is required: a '
                             'controller config name or path, e.g. config:=sim_split')
        self.summary = f'robot {self.robot}, config {self.config_name}'
        self.config = load_controller_config(self.robot, self.config_name)
        controller = self.config['controller']
        self.timeout = controller['timeout']
        self.threshold_ik = controller['threshold_ik']
        self.threshold_joint = controller['threshold_joint']
        self.threshold_linear = controller['threshold_linear']
        self.threshold_angular = controller['threshold_angular']
        # the current goal's frames and their targets
        self.frame_names = []
        self.frame_targets = []
        self.controller = None
        self._robot_model = robot_model
        self._init = init
        # one goal at a time drives the controller
        self._controller_lock = threading.Lock()

    def start(self):
        """Build the controller (its start-up included), then the publishers and action servers.

        /left_ee_pose appearing tells the robot's other nodes the controller is up. The node
        must be spinning, since the start-up waits on its /lowstate subscription.
        """
        self.controller = FrameController(self, self.profile, self.config,
                                          robot_model=self._robot_model, init=self._init)

        self.frame_names_publisher = self.create_publisher(StringArray, 'frame_names', 10)
        self.frame_targets_publisher = self.create_publisher(PoseArray, 'frame_targets', 10)
        self.frame_poses_publisher = self.create_publisher(PoseArray, 'frame_poses', 10)
        self.left_ee_pose_publisher = self.create_publisher(PoseStamped, 'left_ee_pose', 10)
        self.right_ee_pose_publisher = self.create_publisher(PoseStamped, 'right_ee_pose', 10)
        self.left_ee_target_publisher = self.create_publisher(PoseStamped, 'left_ee_target', 10)
        self.right_ee_target_publisher = self.create_publisher(PoseStamped, 'right_ee_target',
                                                               10)
        self.grasp_pose_publishers = tuple(self.create_publisher(PoseStamped, topic, 10)
                                           for topic in GRASP_POSE_TOPICS)
        self.publisher_timer = self.create_timer(1.0 / PUBLISH_HZ, self.publisher_callback)

        self.action_server = ActionServer(
            self, FrameTask, 'frame_task',
            execute_callback=self.frame_task_callback,
            goal_callback=self.frame_task_goal_callback,
            cancel_callback=self.cancel_callback)
        self.named_config_server = ActionServer(
            self, NamedConfig, 'named_config',
            execute_callback=self.named_config_callback,
            cancel_callback=self.cancel_callback)
        self.grasp_frame_target_server = ActionServer(
            self, GraspFrameTarget, GRASP_ACTION,
            execute_callback=self.grasp_frame_target_callback,
            cancel_callback=self.cancel_callback)
        self.get_logger().info(f'Frame Task Server initialized ({self.summary})')

    def shutdown(self):
        """Shut the controller down: its shutdown command, then silence."""
        if self.controller is not None:
            self.controller.shutdown()

    def _stamp_pose(self, pose, stamp):
        """Return pose as a PoseStamped in the pelvis frame."""
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = POSE_FRAME
        pose_stamped.pose = pose
        return pose_stamped

    def _pose_array(self, poses, stamp):
        """Return poses as a PoseArray in the pelvis frame."""
        array = PoseArray()
        array.header.stamp = stamp
        array.header.frame_id = POSE_FRAME
        array.poses = poses
        return array

    def publisher_callback(self):
        """Publish the poses (the timer callback).

        A failure is logged, at most once per PUBLISH_ERROR_LOG_PERIOD, and never reaches the
        executor, which would stop serving every callback.
        """
        try:
            self._publish_poses()
        except Exception as error:
            self.get_logger().error(f'publishing the poses failed: {error!r}',
                                    throttle_duration_sec=PUBLISH_ERROR_LOG_PERIOD)

    def _publish_poses(self):
        """Publish the goal's frames, targets and poses, and the end-effector and grasp poses."""
        frame_names, frame_targets = self.frame_names, self.frame_targets
        names = StringArray()
        names.data = list(frame_names)
        self.frame_names_publisher.publish(names)

        stamp = self.get_clock().now().to_msg()
        self.frame_targets_publisher.publish(self._pose_array(list(frame_targets), stamp))
        self.frame_poses_publisher.publish(self._pose_array(
            [matrix_to_pose(self.controller.get_frame_transformation(name))
             for name in frame_names], stamp))

        left, right = self.controller.ee_transformations()
        self.left_ee_pose_publisher.publish(self._stamp_pose(matrix_to_pose(left), stamp))
        self.right_ee_pose_publisher.publish(self._stamp_pose(matrix_to_pose(right), stamp))
        for frame, publisher in zip(self.profile.grasp_frames, self.grasp_pose_publishers):
            pose = self.controller.get_frame_transformation(frame.name)
            publisher.publish(self._stamp_pose(matrix_to_pose(pose), stamp))
        # an end effector's target goes out only while the goal has a task on its frame
        for frame, publisher in zip(self.profile.ee_frames, (self.left_ee_target_publisher,
                                                             self.right_ee_target_publisher)):
            target = self.controller.frame_task_target(f'{frame}_task')
            if target is not None:
                publisher.publish(self._stamp_pose(matrix_to_pose(target), stamp))

    def _abort_stale(self, goal_handle, result, error):
        """Abort a goal because /lowstate went stale; the last command holds."""
        self.get_logger().error(f'Aborting goal: {error}')
        goal_handle.abort()
        result.success = False
        return result

    def frame_task_goal_callback(self, goal_request):
        """Refuse a goal that names a frame the robot's models do not have."""
        unknown = self.controller.unknown_frames(goal_request.frame_names)
        if unknown:
            self.get_logger().warn(
                f'Refusing the FrameTask goal: {", ".join(repr(name) for name in unknown)} '
                f'is not a frame of the {self.robot} model')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def frame_task_callback(self, goal_handle):
        """Run a FrameTask goal; abort it when /lowstate goes stale."""
        try:
            return self._frame_task_callback(goal_handle)
        except StaleStateError as error:
            return self._abort_stale(goal_handle, FrameTask.Result(), error)

    def _frame_task_callback(self, goal_handle):
        """Replace the frame tasks with the goal's, then move directly or along a plan."""
        with self._controller_lock:
            self.get_logger().info('Received goal')
            goal = goal_handle.request
            # relative targets resolve against the measured pose, which must be fresh
            self.controller.check_state_fresh()

            target_matrices = [
                self._resolve_target(name, pose_to_matrix(target), goal.relative)
                for name, target in zip(goal.frame_names, goal.frame_targets)
            ]

            self.controller.clear_frame_tasks()
            self.frame_names = list(goal.frame_names)
            self.frame_targets = [matrix_to_pose(matrix) for matrix in target_matrices]

            self.get_logger().info(
                'Going to target frame poses' + (' (relative)' if goal.relative else ''))
            for frame_name, target_matrix in zip(goal.frame_names, target_matrices):
                self.controller.add_frame_task(f'{frame_name}_task', frame_name, target_matrix)

            self.controller.update_ik_solver()

            kind = _GoalKind(
                result=FrameTask.Result, target='target frame poses',
                error=self._frame_task_errors, feedback=self._publish_frame_task_feedback,
                reached=self._frame_targets_reached,
                planned_step_converges=False, direct_reached_message=None)
            return self._move(
                goal_handle, kind,
                ik_step=self.controller.control_step_reduced,
                plan=lambda: self.controller.plan_to_ik_target(ik_timeout=self.timeout))

    def _goal_timeout(self, duration):
        """Return a goal's duration or timeout, s of robot time; controller.timeout for 0."""
        seconds = duration.sec + duration.nanosec * 1e-9
        return seconds if seconds > 0.0 else self.timeout

    def _resolve_target(self, frame_name, target, relative):
        """Return one frame's absolute 4x4 target in the pelvis frame.

        A relative target's translation is added to the frame's measured position, and its
        rotation is applied in pelvis-frame axes about the frame's measured origin.
        """
        if not relative:
            return target
        current = self.controller.get_frame_transformation(frame_name)
        absolute = np.eye(4)
        absolute[:3, :3] = target[:3, :3] @ current[:3, :3]
        absolute[:3, 3] = current[:3, 3] + target[:3, 3]
        return absolute

    def _canceled(self, goal_handle, result):
        """Mark a goal canceled and return its failed result."""
        self.get_logger().info('Goal cancelled')
        goal_handle.canceled()
        result.success = False
        return result

    def _move(self, goal_handle, kind, ik_step, plan, after_plan=None):
        """Run a goal at its speed, directly or along a plan as it asks; return its result.

        ik_step() is the direct goal's IK step (_run_goal); plan() returns the planned goal's
        path and after_plan(), when given, runs once it has executed (_planned_goal).
        slow_mode slows this goal only: full speed returns when it ends, however it ends.
        """
        goal = goal_handle.request
        timeout = self._goal_timeout(goal.duration)
        self.controller.speed_scale = SLOW_MODE_SCALE if goal.slow_mode else 1.0
        if goal.slow_mode:
            self.get_logger().info(f'Slow mode enabled (speed_scale={SLOW_MODE_SCALE})')
        try:
            if goal.plan:
                return self._planned_goal(goal_handle, kind, timeout, plan, after_plan)
            return self._run_goal(goal_handle, kind, timeout, ik_step=ik_step)
        finally:
            self.controller.speed_scale = 1.0

    def _planned_goal(self, goal_handle, kind, timeout, plan, after_plan):
        """Plan, execute the path, then hold until reached, blocked or timeout.

        A planning or execution failure aborts the goal. Otherwise after_plan() runs, when
        given, and the hold starts (_run_goal); timeout is in s of robot time from the end of
        the path.
        """
        self.get_logger().info(f'Planning path to {kind.target}')
        clock = self.controller.clock
        try:
            path = plan()
            self.get_logger().info(f'Executing planned path with {len(path)} waypoints')
            execute_start = clock.now()
            self.controller.execute_path(path)
            self.get_logger().info(
                f'Path execution finished in {clock.now() - execute_start:.2f}s; '
                f'holding steady state (timeout {timeout:.1f}s)')
        except Exception as error:
            self.get_logger().warn(f'Planning or execution failed: {error}')
            goal_handle.abort()
            result = kind.result()
            result.success = False
            return result

        if after_plan is not None:
            after_plan()
        self.controller.init_steady_state()
        return self._run_goal(goal_handle, kind, timeout)

    def _run_goal(self, goal_handle, kind, timeout, ik_step=None):
        """Drive the goal (_drive), then end it; return its result.

        Unless canceled the goal succeeds, its result's success saying whether the IK
        converged and the hold reached the target.
        """
        outcome = self._drive(goal_handle, kind, timeout, ik_step=ik_step)
        if outcome.canceled:
            return self._canceled(goal_handle, kind.result())
        if not outcome.ik_converged:
            self.get_logger().warn('Timed out before IK convergence')
        else:
            self._log_hold_outcome(outcome.reached, outcome.blocked)

        goal_handle.succeed()
        result = kind.result()
        result.success = bool(outcome.ik_converged and outcome.reached)
        return result

    def _drive(self, goal_handle, kind, timeout, ik_step=None):
        """Step to the goal's target, then hold until reached, blocked, canceled or timeout.

        ik_step: a direct goal's IK step, returning the body joint velocity (rad/s); the hold
            starts once its largest |velocity| is under threshold_ik. None for a planned goal,
            whose hold starts at once. An iteration that took longer than the control period
            runs the next iteration's IK steps it owes, one per period, at most CATCH_UP_STEPS
            (stopping at the step under threshold_ik).
        A direct goal checks for a cancel after each step, a planned one before. The hold is
        reached when kind.reached(error) holds, or when the hold step itself converges (as
        kind.planned_step_converges or kind.direct_step_converges say), and has stayed so for
        kind.settle s of robot time. timeout is in s of robot time. Returns an _Outcome; the
        goal is left for the caller to end.
        """
        clock = self.controller.clock
        planned = ik_step is None
        step_converges = (kind.planned_step_converges if planned
                          else kind.direct_step_converges)
        start_time = clock.now()
        ik_converged = planned
        steady_state_converged = False
        reached = False
        # robot time the target was first reached since it last was not, or None
        reached_since = None
        blocked = None
        steps_owed = 1
        while clock.now() - start_time < timeout:
            step_start = clock.now()
            if planned and goal_handle.is_cancel_requested:
                return _Outcome(True, ik_converged, False, None)
            if not ik_converged:
                for _ in range(steps_owed):
                    vel_error = np.max(np.abs(ik_step()))
                    if vel_error < self.threshold_ik:
                        break
            else:
                held = self.controller.steady_state_step()
                steady_state_converged = step_converges and held
            if not planned and goal_handle.is_cancel_requested:
                return _Outcome(True, ik_converged, False, None)

            error = kind.error()
            kind.feedback(goal_handle, error)

            if not ik_converged:
                if vel_error < self.threshold_ik:
                    ik_converged = True
                    self.controller.init_steady_state()
                    self.get_logger().info('IK converged; entering steady-state hold')
            else:
                blocked = self._limit_block_reason()
                if blocked is not None:
                    break
                if steady_state_converged or kind.reached(error):
                    now = clock.now()
                    reached_since = now if reached_since is None else reached_since
                    if now - reached_since >= kind.settle:
                        reached = True
                        if not planned and kind.direct_reached_message is not None:
                            self.get_logger().info(kind.direct_reached_message)
                        break
                else:
                    reached_since = None

            dt = self.controller.dt
            clock.sleep(max(0.0, dt - (clock.now() - step_start)))
            steps_owed = min(CATCH_UP_STEPS, max(1, round((clock.now() - step_start) / dt)))

        return _Outcome(False, ik_converged, reached, blocked)

    def _limit_block_reason(self):
        """Return the joints pinning the steady-state hold at a position limit, or None."""
        limited = self.controller.steady_state_limited_joints
        if not limited:
            return None
        return ', '.join(f'{name} q={q:+.3f} (limit {limit:+.3f})' for name, q, limit in limited)

    def _log_hold_outcome(self, converged, blocked):
        """Log how the hold ended: blocked at a limit, reached, or timed out."""
        if blocked is not None:
            self.get_logger().warn(f'Hold blocked at a joint position limit: {blocked}')
        elif converged:
            self.get_logger().info('Goal reached')
        else:
            self.get_logger().warn('Timed out during steady-state hold')

    def _frame_task_errors(self):
        """Return each goal frame's position (m) and orientation (rad) error, as two lists."""
        errors_linear = []
        errors_angular = []
        for frame_name in self.frame_names:
            error = self.controller.get_frame_task_error(f'{frame_name}_task')
            errors_linear.append(float(np.linalg.norm(error[:3])))
            errors_angular.append(float(np.linalg.norm(error[3:])))
        return errors_linear, errors_angular

    def _frame_targets_reached(self, errors):
        """Return whether every goal frame is within threshold_linear and threshold_angular.

        errors: _frame_task_errors()'s two lists; a goal with no frame is never reached.
        """
        errors_linear, errors_angular = errors
        return (len(errors_linear) > 0 and len(errors_angular) > 0 and
                max(errors_linear) < self.threshold_linear and
                max(errors_angular) < self.threshold_angular)

    def _publish_frame_task_feedback(self, goal_handle, errors):
        """Publish a FrameTask goal's frame errors (_frame_task_errors()) as feedback."""
        feedback = FrameTask.Feedback()
        feedback.errors_linear, feedback.errors_angular = errors
        goal_handle.publish_feedback(feedback)

    def _named_config_joint_error(self):
        """Return the largest IK joint error from the named configuration, rad."""
        return float(np.max(np.abs(self.controller.reduced_configuration_error)))

    def _named_config_reached(self, joint_error):
        """Return whether every IK joint is within threshold_joint of the named configuration."""
        return joint_error < self.threshold_joint

    def _publish_named_config_feedback(self, goal_handle, joint_error):
        """Publish a NamedConfig goal's joint error as feedback."""
        feedback = NamedConfig.Feedback()
        feedback.joint_error = joint_error
        goal_handle.publish_feedback(feedback)

    def named_config_callback(self, goal_handle):
        """Run a NamedConfig goal; abort it when /lowstate goes stale."""
        try:
            return self._named_config_callback(goal_handle)
        except StaleStateError as error:
            return self._abort_stale(goal_handle, NamedConfig.Result(), error)

    def _named_config_callback(self, goal_handle):
        """Move to the named configuration directly or along a plan; abort an unknown name."""
        with self._controller_lock:
            self.get_logger().info('Received named config goal')
            goal = goal_handle.request

            config_name = goal.config_name
            if config_name not in self.profile.named_configs:
                self.get_logger().warn(f'Named config "{config_name}" not found')
                goal_handle.abort()
                result = NamedConfig.Result()
                result.success = False
                return result

            self.get_logger().info(f'Going to named config: {config_name}')
            q_reduced = self.profile.named_configs[config_name]

            self.controller.update_ik_solver()

            kind = _GoalKind(
                result=NamedConfig.Result, target='named configuration',
                error=self._named_config_joint_error,
                feedback=self._publish_named_config_feedback, reached=self._named_config_reached,
                planned_step_converges=True, direct_reached_message='Named config reached')
            return self._move(
                goal_handle, kind,
                ik_step=lambda: self.controller.goto_reduced_configuration(q_reduced),
                plan=lambda: self.controller.plan_to_configuration(q_reduced),
                # the feedback's joint error is measured against the planned goal
                after_plan=lambda: self.controller.set_reduced_configuration_target(q_reduced))

    def grasp_frame_target_callback(self, goal_handle):
        """Run a GraspFrameTarget goal; every way it ends gives the result a message.

        Its targets are resolved into the pelvis frame first; then it waits for the goal in
        flight, if any, to end.
        """
        self.get_logger().info('Received grasp frame target goal')
        try:
            targets = self._grasp_targets(goal_handle.request)
            with self._controller_lock:
                return self._grasp_frame_target(goal_handle, targets)
        except (GoalError, StaleStateError) as error:
            return self._end_grasp(goal_handle, 'abort', str(error))

    def _end_grasp(self, goal_handle, how, message, success=False):
        """End a GraspFrameTarget goal ('succeed', 'abort' or 'canceled'); return its result."""
        if success:
            self.get_logger().info(f'Grasp frame target reached: {message}')
        else:
            self.get_logger().warn(f'Grasp frame target {how}: {message}')
        getattr(goal_handle, how)()
        result = GraspFrameTarget.Result()
        result.success = success
        result.message = message
        return result

    def _grasp_targets(self, goal):
        """Return {arm: 4x4 target in the pelvis frame} for each arm the goal targets.

        An arm whose header.frame_id is empty has no target. A target in another frame is
        resolved through TF now (_pelvis_transforms). GoalError for a goal without a target,
        a non-finite position or orientation, a zero orientation, or a transform TF does not
        give.
        """
        stamped = {arm: getattr(goal, arm) for arm in ARMS}
        targeted = {arm: pose for arm, pose in stamped.items() if pose.header.frame_id}
        if not targeted:
            raise GoalError('neither arm has a target: give left or right a header.frame_id '
                            f'({POSE_FRAME} or a world frame)')
        targets = {}
        for arm, pose in targeted.items():
            position, orientation = pose.pose.position, pose.pose.orientation
            values = np.array([position.x, position.y, position.z, orientation.w,
                               orientation.x, orientation.y, orientation.z])
            if not np.all(np.isfinite(values)):
                raise GoalError(f'the {arm} target is not finite: {values.tolist()}')
            if np.linalg.norm(values[3:]) < 1e-6:
                raise GoalError(f'the {arm} target has no orientation (a zero quaternion)')
            # pose_to_matrix normalises the quaternion
            targets[arm] = pose_to_matrix(pose.pose)
        frames = sorted({pose.header.frame_id for pose in targeted.values()} - {POSE_FRAME})
        transforms = self._pelvis_transforms(frames) if frames else {}
        for arm, pose in targeted.items():
            frame = pose.header.frame_id
            if frame != POSE_FRAME:
                targets[arm] = transforms[frame] @ targets[arm]
        return targets

    def _pelvis_transforms(self, frames):
        """Return {frame: its 4x4 pose in the pelvis frame}, from TF's newest transforms.

        Listens to TF (/tf, /tf_static) for up to TF_WAIT_S s of steady time, then stops. A
        transform older than TF_MAX_AGE_S s of robot time is not taken; one of static
        transforms only (stamp 0) is. GoalError, naming each frame, when a transform is
        missing or too old by then.

        The listener runs on a node and executor of its own, shut down before the node is
        destroyed: subscriptions destroyed on this node while its executor waits on them stop
        that executor (InvalidHandle), and with it every callback of the controller.
        """
        tf_node = Node(f'{self.get_name()}_tf', context=self.context,
                       use_global_arguments=False, start_parameter_services=False)
        executor = SingleThreadedExecutor(context=self.context)
        executor.add_node(tf_node)
        tf_buffer = Buffer()
        TransformListener(tf_buffer, tf_node)
        spinner = threading.Thread(target=executor.spin, name='grasp_tf', daemon=True)
        spinner.start()
        try:
            found, problems = {}, {}
            deadline = time.monotonic() + TF_WAIT_S
            while True:
                now = self.get_clock().now()
                for frame in frames:
                    if frame in found:
                        continue
                    try:
                        transform = tf_buffer.lookup_transform(POSE_FRAME, frame, Time())
                    except TransformException as error:
                        problems[frame] = f'no transform from {frame} to {POSE_FRAME} ({error})'
                        continue
                    stamp = Time.from_msg(transform.header.stamp, clock_type=now.clock_type)
                    age = (now - stamp).nanoseconds * 1e-9
                    if stamp.nanoseconds > 0 and age > TF_MAX_AGE_S:
                        problems[frame] = (f'the newest transform from {frame} to {POSE_FRAME} '
                                           f'is {age:.2f} s old (> {TF_MAX_AGE_S} s)')
                        continue
                    found[frame] = _transform_matrix(transform.transform)
                if len(found) == len(frames):
                    return found
                if time.monotonic() >= deadline:
                    raise GoalError('; '.join(problems[frame] for frame in frames
                                              if frame not in found)
                                    + f', after {TF_WAIT_S:.1f} s of listening to TF')
                time.sleep(TF_POLL_S)
        finally:
            executor.shutdown()
            spinner.join()
            tf_node.destroy_node()

    def _grasp_frame_target(self, goal_handle, targets):
        """Pull each targeted grasp frame to its target, directly or along a plan; end the goal.

        Only the targeted arms' joints change their torque bias (controller.bias_joints), so
        the other arm and the torso or waist keep their commands and hold where they are.
        The timeout counts robot time from here, planning and the path included. The goal
        succeeds once every target has been within the thresholds for GRASP_SETTLE_S, and
        aborts at the timeout, on a planning or execution failure, or on a hold pinned at a
        joint limit; a cancel or the timeout stops the targeted arms where they are
        (controller.stop_joints).
        """
        joints = self.controller.arm_joint_ids(targets)
        self.controller.bias_joints = joints
        try:
            return self._move_grasp_frames(goal_handle, targets, joints)
        finally:
            self.controller.bias_joints = None

    def _move_grasp_frames(self, goal_handle, targets, joints):
        """Run _grasp_frame_target's goal on the targeted arms' body joints; end the goal."""
        goal = goal_handle.request
        self.controller.check_state_fresh()
        frames = {arm: frame.name for arm, frame in zip(ARMS, self.profile.grasp_frames)}
        self.controller.clear_frame_tasks()
        self.frame_names = [frames[arm] for arm in targets]
        self.frame_targets = [matrix_to_pose(target) for target in targets.values()]
        for arm, target in targets.items():
            self.controller.add_frame_task(f'{frames[arm]}_task', frames[arm], target)
        self.get_logger().info('Moving the grasp frames: ' + '; '.join(
            f'{arm} to {np.round(target[:3, 3], 3).tolist()} m'
            for arm, target in targets.items()) + (' along a plan' if goal.plan else ''))
        self.controller.update_ik_solver()
        # how far the targeted arms rest off their commands now, for a cancel's stop
        stop_reference = self.controller.stop_reference(joints)

        kind = _GoalKind(
            result=GraspFrameTarget.Result, target='grasp frame targets',
            error=lambda: self._grasp_errors(frames, targets),
            feedback=self._publish_grasp_feedback, reached=self._grasp_targets_reached,
            planned_step_converges=False, direct_reached_message=None,
            direct_step_converges=False, settle=GRASP_SETTLE_S)
        clock = self.controller.clock
        timeout = self._goal_timeout(goal.timeout)
        deadline = clock.now() + timeout
        ik_step = self.controller.control_step_reduced
        if goal.plan:
            ended = self._grasp_path(goal_handle, kind, deadline, timeout,
                                     lambda: self.controller.stop_joints(joints, stop_reference))
            if ended is not None:
                return ended
            self.controller.init_steady_state()
            ik_step = None

        outcome = self._drive(goal_handle, kind, deadline - clock.now(), ik_step=ik_step)
        if outcome.canceled:
            self.controller.stop_joints(joints, stop_reference)
            return self._end_grasp(goal_handle, 'canceled', 'canceled; the arms hold where they '
                                   f'are ({self._error_text(kind.error())})')
        errors = self._error_text(kind.error())
        if outcome.reached:
            return self._end_grasp(goal_handle, 'succeed', f'reached ({errors})', success=True)
        if outcome.blocked is not None:
            return self._end_grasp(goal_handle, 'abort',
                                   'the hold is pinned at a joint position limit: '
                                   f'{outcome.blocked} ({errors})')
        self.controller.stop_joints(joints, stop_reference)
        stage = '' if outcome.ik_converged else ' before the IK converged'
        return self._end_grasp(goal_handle, 'abort', f'timed out after {timeout:.1f} s{stage}; '
                               f'the arms hold where they are ({errors})')

    def _grasp_path(self, goal_handle, kind, deadline, timeout, stop_arms):
        """Plan to the IK solution of the targets and execute the path; None once executed.

        Otherwise the goal's result: aborted when planning or execution fails or the deadline
        (robot time) passes during the path, canceled on a cancel; once the path has started,
        each of those ends calls stop_arms() to stop the targeted arms where they are. Feedback goes out before each waypoint, and a
        cancel or the deadline ends the path at the next one. StaleStateError propagates.
        """
        clock = self.controller.clock

        def stop():
            """Publish the feedback; return whether a cancel or the deadline ends the path."""
            kind.feedback(goal_handle, kind.error())
            return goal_handle.is_cancel_requested or clock.now() >= deadline
        try:
            path = self.controller.plan_to_ik_target(ik_timeout=self.timeout)
        except StaleStateError:
            raise
        except Exception as error:
            return self._end_grasp(goal_handle, 'abort', f'planning failed: {error}')
        self.get_logger().info(f'Executing planned path with {len(path)} waypoints')
        try:
            executed = self.controller.execute_path(path, stop=stop)
        except StaleStateError:
            raise
        except Exception as error:
            stop_arms()
            return self._end_grasp(goal_handle, 'abort', f'execution failed: {error}; the arms '
                                   'hold where they are')
        if executed:
            return None
        if goal_handle.is_cancel_requested:
            stop_arms()
            return self._end_grasp(goal_handle, 'canceled', 'canceled during the path; the arms '
                                   f'hold where they are ({self._error_text(kind.error())})')
        stop_arms()
        errors = self._error_text(kind.error())
        return self._end_grasp(goal_handle, 'abort', f'timed out after {timeout:.1f} s during the '
                               f'path; the arms hold where they are ({errors})')

    def _grasp_errors(self, frames, targets):
        """Return {arm: (m, rad)}: each grasp frame's error from its target; 0s without one."""
        return {arm: (self.controller.frame_pose_error(frames[arm], targets[arm])
                      if arm in targets else (0.0, 0.0)) for arm in ARMS}

    def _grasp_targets_reached(self, errors):
        """Return whether every arm is within threshold_linear and threshold_angular."""
        return all(distance < self.threshold_linear and angle < self.threshold_angular
                   for distance, angle in errors.values())

    def _publish_grasp_feedback(self, goal_handle, errors):
        """Publish a GraspFrameTarget goal's errors (_grasp_errors()) as feedback."""
        feedback = GraspFrameTarget.Feedback()
        feedback.left_error_m, feedback.left_error_rad = errors['left']
        feedback.right_error_m, feedback.right_error_rad = errors['right']
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _error_text(errors):
        """Return errors (_grasp_errors()) as text: mm and rad per arm."""
        return ', '.join(f'{arm} {distance * 1000.0:.1f} mm {angle:.4f} rad'
                         for arm, (distance, angle) in errors.items())

    def cancel_callback(self, goal_handle):
        """Accept every cancel request; the goal loop ends at its next step."""
        self.get_logger().info('Canceling goal')
        return CancelResponse.ACCEPT


def _transform_matrix(transform):
    """Return a geometry_msgs Transform as a 4x4 transformation."""
    rotation = transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rotation.w, rotation.x, rotation.y, rotation.z).rotation_matrix
    matrix[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
    return matrix


def main(args=None):
    """Run the node until SIGINT or SIGTERM (utility.run)."""
    run('frame_task_server', FrameTaskServer, args=args)


if __name__ == '__main__':
    main()
