"""Serve the upper-body controller of one robot as a ROS node: frame_task_server.

    ros2 run golem_upper_ik frame_task_server --ros-args -p robot:=h12 -p config:=safety_split

Parameters: robot (required; a robot with data in golem_upper_ik's config/) and config
(required; a controller config name under share/golem_upper_ik/config/<robot>/, or a path).

Once its start-up is done (UpperController), the node serves /grasp_frame_target
(custom_ros_messages/GraspFrameTarget): move each arm's grasp frame (the profile's
grasp_frames) to a target in the pelvis frame or, resolved through TF when the goal starts, in
a world frame, directly through the IK or along an OMPL plan; it succeeds once every target
has been held within the thresholds for GRASP_SETTLE_S, and aborts at its timeout. It
publishes at 25 Hz frame_names, frame_targets and frame_poses (the current goal's frames,
their targets and their measured poses) and /{left,right}_grasp_frame_pose (the grasp frames'
measured poses, every body joint measured, as TF has them; their first message tells the
robot's other nodes the controller is up). Every pose is in the pelvis frame. A goal's timeout
and the control period are s of robot time (the controller's clock: the sim's /clock under
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

from custom_ros_messages.action import GraspFrameTarget  # noqa: E402
from custom_ros_messages.msg import StringArray  # noqa: E402
from geometry_msgs.msg import PoseArray, PoseStamped  # noqa: E402
import numpy as np  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402
from rclpy.action import ActionServer, CancelResponse  # noqa: E402
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

__all__ = ['CATCH_UP_STEPS', 'FrameTaskServer', 'GRASP_SETTLE_S', 'GoalError', 'main']

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
    """Hold how the goal loop reads and judges a goal.

    error(): the goal's error now; feedback(goal_handle, error) publishes it;
    reached(error) says whether it is within the thresholds. settle: s of robot time the
    target must stay reached before the goal is.
    """

    error: Callable
    feedback: Callable
    reached: Callable
    settle: float


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
    """Serve /grasp_frame_target and publish the poses."""

    def __init__(self, *, robot_model=None, init=True, **node_kwargs):
        """Read the parameters and the config; start() builds the controller and the topics.

        robot_model: a RobotModel of the robot to use instead of building one.
        init: run the controller's start-up moves, and start the planner subprocess in the
            background with them (UpperController's warm_planner).
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

        /left_grasp_frame_pose appearing tells the robot's other nodes the controller is up.
        The node must be spinning, since the start-up waits on its /lowstate subscription.
        """
        # with the start-up moves the planner starts in the background, so the first planned
        # goal does not wait for it
        self.controller = FrameController(self, self.profile, self.config,
                                          robot_model=self._robot_model, init=self._init,
                                          warm_planner=self._init)

        self.frame_names_publisher = self.create_publisher(StringArray, 'frame_names', 10)
        self.frame_targets_publisher = self.create_publisher(PoseArray, 'frame_targets', 10)
        self.frame_poses_publisher = self.create_publisher(PoseArray, 'frame_poses', 10)
        self.grasp_pose_publishers = tuple(self.create_publisher(PoseStamped, topic, 10)
                                           for topic in GRASP_POSE_TOPICS)
        self.publisher_timer = self.create_timer(1.0 / PUBLISH_HZ, self.publisher_callback)

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
        """Publish the goal's frames, targets and poses, and the grasp poses."""
        frame_names, frame_targets = self.frame_names, self.frame_targets
        names = StringArray()
        names.data = list(frame_names)
        self.frame_names_publisher.publish(names)

        stamp = self.get_clock().now().to_msg()
        self.frame_targets_publisher.publish(self._pose_array(list(frame_targets), stamp))
        self.frame_poses_publisher.publish(self._pose_array(
            [matrix_to_pose(self.controller.get_frame_transformation(name))
             for name in frame_names], stamp))

        for frame, publisher in zip(self.profile.grasp_frames, self.grasp_pose_publishers):
            pose = self.controller.get_frame_transformation(frame.name)
            publisher.publish(self._stamp_pose(matrix_to_pose(pose), stamp))

    def _goal_timeout(self, duration):
        """Return a goal's duration or timeout, s of robot time; controller.timeout for 0."""
        seconds = duration.sec + duration.nanosec * 1e-9
        return seconds if seconds > 0.0 else self.timeout

    def _drive(self, goal_handle, kind, timeout, ik_step=None):
        """Step to the goal's target, then hold until reached, blocked, canceled or timeout.

        ik_step: a direct goal's IK step, returning its body joint speeds (rad/s,
            UpperController._step_ik); the hold starts once the largest is under threshold_ik. None for a planned goal,
            whose hold starts at once. An iteration that took longer than the control period
            runs the next iteration's IK steps it owes, one per period, at most CATCH_UP_STEPS
            (stopping at the step under threshold_ik).
        A direct goal checks for a cancel after each step, a planned one before. The hold is
        reached when kind.reached(error) has held for kind.settle s of robot time; the hold
        step's own convergence ends nothing. timeout is in s of robot time. Returns an
        _Outcome; the goal is left for the caller to end.
        """
        clock = self.controller.clock
        planned = ik_step is None
        start_time = clock.now()
        ik_converged = planned
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
                self.controller.steady_state_step()
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
                if kind.reached(error):
                    now = clock.now()
                    reached_since = now if reached_since is None else reached_since
                    if now - reached_since >= kind.settle:
                        reached = True
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
            return self._end_grasp(goal_handle, GraspFrameTarget.Result.FAILED, str(error))

    def _end_grasp(self, goal_handle, outcome, message):
        """End a GraspFrameTarget goal with outcome (a Result constant); return its result.

        REACHED succeeds the goal, CANCELED marks it canceled, NOT_REACHED and FAILED abort it.
        """
        Result = GraspFrameTarget.Result
        if outcome == Result.REACHED:
            self.get_logger().info(f'Grasp frame target reached: {message}')
            goal_handle.succeed()
        elif outcome == Result.CANCELED:
            self.get_logger().warn(f'Grasp frame target canceled: {message}')
            goal_handle.canceled()
        else:
            self.get_logger().warn(f'Grasp frame target abort: {message}')
            goal_handle.abort()
        result = Result()
        result.outcome = outcome
        result.success = outcome == Result.REACHED
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
            error=lambda: self._grasp_errors(frames, targets),
            feedback=self._publish_grasp_feedback, reached=self._grasp_targets_reached,
            settle=GRASP_SETTLE_S)
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

        Result = GraspFrameTarget.Result
        outcome = self._drive(goal_handle, kind, deadline - clock.now(), ik_step=ik_step)
        if outcome.canceled:
            self.controller.stop_joints(joints, stop_reference)
            return self._end_grasp(goal_handle, Result.CANCELED, 'canceled; the arms hold where '
                                   f'they are ({self._error_text(kind.error())})')
        errors = self._error_text(kind.error())
        if outcome.reached:
            return self._end_grasp(goal_handle, Result.REACHED, f'reached ({errors})')
        if outcome.blocked is not None:
            return self._end_grasp(goal_handle, Result.NOT_REACHED,
                                   'the hold is pinned at a joint position limit: '
                                   f'{outcome.blocked} ({errors})')
        self.controller.stop_joints(joints, stop_reference)
        stage = '' if outcome.ik_converged else ' before the IK converged'
        return self._end_grasp(goal_handle, Result.NOT_REACHED, f'timed out after {timeout:.1f} '
                               f's{stage}; the arms hold where they are ({errors})')

    def _grasp_path(self, goal_handle, kind, deadline, timeout, stop_arms):
        """Plan to the IK solution of the targets and execute the path; None once executed.

        Otherwise the goal's result: FAILED when planning or execution fails, NOT_REACHED when
        the deadline (robot time) passes during the path, CANCELED on a cancel; once the path
        has started, each of those ends calls stop_arms() to stop the targeted arms where they
        are. Feedback goes out before each waypoint, and a cancel or the deadline ends the path
        at the next one. StaleStateError propagates.
        """
        clock = self.controller.clock
        Result = GraspFrameTarget.Result

        def stop():
            """Publish the feedback; return whether a cancel or the deadline ends the path."""
            kind.feedback(goal_handle, kind.error())
            return goal_handle.is_cancel_requested or clock.now() >= deadline
        try:
            path = self.controller.plan_to_ik_target(ik_timeout=self.timeout)
        except StaleStateError:
            raise
        except Exception as error:
            return self._end_grasp(goal_handle, Result.FAILED, f'planning failed: {error}')
        self.get_logger().info(f'Executing planned path with {len(path)} waypoints')
        try:
            executed = self.controller.execute_path(path, stop=stop)
        except StaleStateError:
            raise
        except Exception as error:
            stop_arms()
            return self._end_grasp(goal_handle, Result.FAILED, f'execution failed: {error}; the '
                                   'arms hold where they are')
        if executed:
            return None
        if goal_handle.is_cancel_requested:
            stop_arms()
            return self._end_grasp(goal_handle, Result.CANCELED, 'canceled during the path; the '
                                   f'arms hold where they are ({self._error_text(kind.error())})')
        stop_arms()
        errors = self._error_text(kind.error())
        return self._end_grasp(goal_handle, Result.NOT_REACHED, f'timed out after {timeout:.1f} s '
                               f'during the path; the arms hold where they are ({errors})')

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
