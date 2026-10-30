'''
frame_task_server: the upper-body controller of one robot as a ROS node.

    ros2 run golem_upper_ik frame_task_server --ros-args -p robot:=h12 -p config:=safety_split

Parameters: robot (h12 or g1, required) and config (a controller config name under
share/golem_upper_ik/config/<robot>/, or a path; default the robot's DEFAULT_CONFIG).

Once its start-up is done (UpperController), the node serves:
- /frame_task (custom_ros_messages/FrameTask): pull named frames to target poses in the pelvis
  frame, directly through the IK or along an OMPL plan, then hold them;
- /named_config (custom_ros_messages/NamedConfig): move the IK joints to a named
  configuration, the same two ways;
and publishes at 100 Hz frame_names, frame_targets and frame_poses (the current goal's
frames, their targets and their measured poses), {left,right}_ee_pose (the end effectors'
measured poses, which tell the robot's other nodes the controller is up) and
{left,right}_ee_target (the current goal's targets for the end-effector frames).
'''
import os

# One BLAS thread before numpy loads: OpenBLAS's default of a busy-waiting worker per core
# stalls the 500 Hz command publisher by tens of ms
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import threading  # noqa: E402
import time  # noqa: E402

from custom_ros_messages.action import FrameTask, NamedConfig  # noqa: E402
from custom_ros_messages.msg import StringArray  # noqa: E402
from geometry_msgs.msg import PoseArray, PoseStamped  # noqa: E402
import numpy as np  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.node import Node  # noqa: E402

from golem_upper_ik.config import load_controller_config  # noqa: E402
from golem_upper_ik.core.controller.frame_controller import FrameController  # noqa: E402
from golem_upper_ik.core.ros_io import StaleStateError  # noqa: E402
from golem_upper_ik.profile import load_profile  # noqa: E402
from golem_upper_ik.ros2.utility import (  # noqa: E402
    matrix_to_pose, node_parameters, pose_to_matrix, run,
)

__all__ = ['FrameTaskServer', 'SLOW_MODE_SCALE', 'main']

# Speed multiplier applied when a goal asks for slow_mode. The controller applies it per
# execution mode: a direct move scales joint velocity, a planned move is planned at a
# proportionally lower planner-frame speed (more waypoints, same control period), so the arm
# stays smooth instead of stepping.
SLOW_MODE_SCALE = 0.25
PUBLISH_HZ = 100.0
# the frame every published pose is in
POSE_FRAME = 'pelvis'
# s between log lines while publishing the poses keeps failing
PUBLISH_ERROR_LOG_PERIOD = 1.0


class FrameTaskServer(Node):
    def __init__(self, *, robot_model=None, init=True, **node_kwargs):
        '''
        Reads the parameters and the config; start() builds the controller and the ROS
        surface. robot_model: a RobotModel of the robot to use instead of building one.
        init: run the controller's start-up moves
        '''
        super().__init__('frame_task_server', **node_kwargs)
        self.robot, self.config_name = node_parameters(self)
        self.profile = load_profile(self.robot)
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
        '''
        Build the controller (its start-up included), then the publishers and the action
        servers: /left_ee_pose appearing tells the robot's other nodes the controller is up.
        The node must be spinning
        '''
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
        self.get_logger().info(
            f'Frame Task Server initialized (robot {self.robot}, config {self.config_name})')

    def shutdown(self):
        '''The controller's shutdown: its shutdown command, then silence'''
        if self.controller is not None:
            self.controller.shutdown()

    def _stamp_pose(self, pose, stamp):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = POSE_FRAME
        pose_stamped.pose = pose
        return pose_stamped

    def _pose_array(self, poses, stamp):
        array = PoseArray()
        array.header.stamp = stamp
        array.header.frame_id = POSE_FRAME
        array.poses = poses
        return array

    def publisher_callback(self):
        '''
        Publish the poses. A failure is logged, at most once per PUBLISH_ERROR_LOG_PERIOD,
        and never reaches the executor, which would stop serving every callback
        '''
        try:
            self._publish_poses()
        except Exception as error:
            self.get_logger().error(f'publishing the poses failed: {error!r}',
                                    throttle_duration_sec=PUBLISH_ERROR_LOG_PERIOD)

    def _publish_poses(self):
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
        # the targets of the current goal's tasks on the end-effector frames
        for frame, publisher in zip(self.profile.ee_frames, (self.left_ee_target_publisher,
                                                             self.right_ee_target_publisher)):
            target = self.controller.frame_task_target(f'{frame}_task')
            if target is not None:
                publisher.publish(self._stamp_pose(matrix_to_pose(target), stamp))

    def _abort_stale(self, goal_handle, result, error):
        '''Abort a goal because /lowstate went stale; the last command holds'''
        self.get_logger().error(f'Aborting goal: {error}')
        goal_handle.abort()
        result.success = False
        return result

    def frame_task_goal_callback(self, goal_request):
        '''Refuse a goal that names a frame the robot's models do not have'''
        unknown = self.controller.unknown_frames(goal_request.frame_names)
        if unknown:
            self.get_logger().warn(
                f'Refusing the FrameTask goal: {", ".join(repr(name) for name in unknown)} '
                f'is not a frame of the {self.robot} model')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def frame_task_callback(self, goal_handle):
        try:
            return self._frame_task_callback(goal_handle)
        except StaleStateError as error:
            return self._abort_stale(goal_handle, FrameTask.Result(), error)

    def _frame_task_callback(self, goal_handle):
        with self._controller_lock:
            self.get_logger().info('Received goal')
            goal = goal_handle.request
            # relative targets are resolved against the measured pose
            self.controller.check_state_fresh()

            # resolve targets to absolute pelvis-frame poses before the current tasks are
            # cleared
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

            timeout = self._goal_timeout(goal)
            # slow the arm for this goal only; full speed again afterwards
            self.controller.speed_scale = SLOW_MODE_SCALE if goal.slow_mode else 1.0
            if goal.slow_mode:
                self.get_logger().info(f'Slow mode enabled (speed_scale={SLOW_MODE_SCALE})')
            try:
                if goal.plan:
                    return self._plan_frame_task_callback(goal_handle, timeout)
                return self._direct_frame_task_callback(goal_handle, timeout)
            finally:
                self.controller.speed_scale = 1.0

    def _goal_timeout(self, goal):
        '''The goal's duration, s, or controller.timeout without one'''
        duration = goal.duration.sec + goal.duration.nanosec * 1e-9
        return duration if duration > 0.0 else self.timeout

    def _resolve_target(self, frame_name, target, relative):
        '''
        The absolute pelvis-frame target of one frame. A relative target's translation is
        added to the frame's current position, and its rotation is applied in pelvis-frame
        axes about the frame's current origin
        '''
        if not relative:
            return target
        current = self.controller.get_frame_transformation(frame_name)
        absolute = np.eye(4)
        absolute[:3, :3] = target[:3, :3] @ current[:3, :3]
        absolute[:3, 3] = current[:3, 3] + target[:3, 3]
        return absolute

    def _canceled(self, goal_handle, result):
        self.get_logger().info('Goal cancelled')
        goal_handle.canceled()
        result.success = False
        return result

    def _direct_frame_task_callback(self, goal_handle, timeout):
        start_time = time.monotonic()
        ik_converged = False
        steady_state_converged = False
        blocked = None
        while time.monotonic() - start_time < timeout:
            step_start = time.monotonic()
            if not ik_converged:
                vel = self.controller.control_step_reduced()
                vel_error = np.max(np.abs(vel))
            else:
                steady_state_converged = self.controller.steady_state_step()

            if goal_handle.is_cancel_requested:
                return self._canceled(goal_handle, FrameTask.Result())

            errors_linear, errors_angular = self._frame_task_errors()
            self._publish_frame_task_feedback(goal_handle, errors_linear, errors_angular)

            if not ik_converged:
                if vel_error < self.threshold_ik:
                    ik_converged = True
                    self.controller.init_steady_state()
                    self.get_logger().info('IK converged; entering steady-state hold')
            else:
                blocked = self._limit_block_reason()
                if blocked is not None:
                    break
                steady_state_converged = steady_state_converged or (
                    len(errors_linear) > 0 and
                    len(errors_angular) > 0 and
                    max(errors_linear) < self.threshold_linear and
                    max(errors_angular) < self.threshold_angular
                )
                if steady_state_converged:
                    break

            time.sleep(max(0.0, self.controller.dt - (time.monotonic() - step_start)))

        if not ik_converged:
            self.get_logger().warn('Timed out before IK convergence')
        else:
            self._log_hold_outcome(steady_state_converged, blocked)

        result = FrameTask.Result()
        result.success = bool(ik_converged and steady_state_converged)
        goal_handle.succeed()
        return result

    def _plan_frame_task_callback(self, goal_handle, timeout):
        self.get_logger().info('Planning path to target frame poses')
        try:
            # solve the IK, plan a reduced-joint path, then execute it
            path = self.controller.plan_to_ik_target(ik_timeout=self.timeout)
            self.get_logger().info(f'Executing planned path with {len(path)} waypoints')
            execute_start = time.monotonic()
            self.controller.execute_path(path)
            self.get_logger().info(
                f'Path execution finished in {time.monotonic() - execute_start:.2f}s; '
                f'holding steady state (timeout {timeout:.1f}s)')
        except Exception as error:
            self.get_logger().warn(f'Planning or execution failed: {error}')
            goal_handle.abort()
            result = FrameTask.Result()
            result.success = False
            return result

        # hold the final target with the steady-state controller
        self.controller.init_steady_state()
        start_time = time.monotonic()
        steady_state_converged = False
        blocked = None
        while time.monotonic() - start_time < timeout:
            step_start = time.monotonic()
            if goal_handle.is_cancel_requested:
                return self._canceled(goal_handle, FrameTask.Result())

            self.controller.steady_state_step()
            errors_linear, errors_angular = self._frame_task_errors()
            self._publish_frame_task_feedback(goal_handle, errors_linear, errors_angular)

            blocked = self._limit_block_reason()
            if blocked is not None:
                break
            steady_state_converged = (
                len(errors_linear) > 0 and
                len(errors_angular) > 0 and
                max(errors_linear) < self.threshold_linear and
                max(errors_angular) < self.threshold_angular
            )
            if steady_state_converged:
                break

            time.sleep(max(0.0, self.controller.dt - (time.monotonic() - step_start)))

        self._log_hold_outcome(steady_state_converged, blocked)

        goal_handle.succeed()
        result = FrameTask.Result()
        result.success = bool(steady_state_converged)
        return result

    def _limit_block_reason(self):
        '''The joints pinning the steady-state hold at a position limit, or None'''
        limited = self.controller.steady_state_limited_joints
        if not limited:
            return None
        return ', '.join(f'{name} q={q:+.3f} (limit {limit:+.3f})' for name, q, limit in limited)

    def _log_hold_outcome(self, converged, blocked):
        if blocked is not None:
            self.get_logger().warn(f'Hold blocked at a joint position limit: {blocked}')
        elif converged:
            self.get_logger().info('Goal reached')
        else:
            self.get_logger().warn('Timed out during steady-state hold')

    def _frame_task_errors(self):
        errors_linear = []
        errors_angular = []
        for frame_name in self.frame_names:
            error = self.controller.get_frame_task_error(f'{frame_name}_task')
            errors_linear.append(float(np.linalg.norm(error[:3])))
            errors_angular.append(float(np.linalg.norm(error[3:])))
        return errors_linear, errors_angular

    def _publish_frame_task_feedback(self, goal_handle, errors_linear, errors_angular):
        feedback = FrameTask.Feedback()
        feedback.errors_linear = errors_linear
        feedback.errors_angular = errors_angular
        goal_handle.publish_feedback(feedback)

    def _named_config_joint_error(self):
        return float(np.max(np.abs(self.controller.reduced_configuration_error)))

    def _publish_named_config_feedback(self, goal_handle, joint_error):
        feedback = NamedConfig.Feedback()
        feedback.joint_error = joint_error
        goal_handle.publish_feedback(feedback)

    def named_config_callback(self, goal_handle):
        try:
            return self._named_config_callback(goal_handle)
        except StaleStateError as error:
            return self._abort_stale(goal_handle, NamedConfig.Result(), error)

    def _named_config_callback(self, goal_handle):
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

            timeout = self._goal_timeout(goal)
            # slow the arm for this goal only; full speed again afterwards
            self.controller.speed_scale = SLOW_MODE_SCALE if goal.slow_mode else 1.0
            if goal.slow_mode:
                self.get_logger().info(f'Slow mode enabled (speed_scale={SLOW_MODE_SCALE})')
            try:
                if goal.plan:
                    return self._plan_named_config_callback(goal_handle, q_reduced, timeout)
                return self._direct_named_config_callback(goal_handle, q_reduced, timeout)
            finally:
                self.controller.speed_scale = 1.0

    def _direct_named_config_callback(self, goal_handle, q_reduced, timeout):
        start_time = time.monotonic()
        ik_converged = False
        steady_state_converged = False
        blocked = None
        while time.monotonic() - start_time < timeout:
            step_start = time.monotonic()
            if not ik_converged:
                vel = self.controller.goto_reduced_configuration(q_reduced)
                vel_error = np.max(np.abs(vel))
            else:
                steady_state_converged = self.controller.steady_state_step()

            if goal_handle.is_cancel_requested:
                return self._canceled(goal_handle, NamedConfig.Result())

            joint_error = self._named_config_joint_error()
            self._publish_named_config_feedback(goal_handle, joint_error)

            if not ik_converged:
                if vel_error < self.threshold_ik:
                    ik_converged = True
                    self.controller.init_steady_state()
                    self.get_logger().info('IK converged; entering steady-state hold')
            else:
                blocked = self._limit_block_reason()
                if blocked is not None:
                    break
                steady_state_converged = (steady_state_converged or
                                          joint_error < self.threshold_joint)
                if steady_state_converged:
                    self.get_logger().info('Named config reached')
                    break

            time.sleep(max(0.0, self.controller.dt - (time.monotonic() - step_start)))

        if not ik_converged:
            self.get_logger().warn('Timed out before IK convergence')
        else:
            self._log_hold_outcome(steady_state_converged, blocked)

        goal_handle.succeed()
        result = NamedConfig.Result()
        result.success = bool(ik_converged and steady_state_converged)
        return result

    def _plan_named_config_callback(self, goal_handle, q_reduced, timeout):
        self.get_logger().info('Planning path to named configuration')
        try:
            # plan directly in reduced joint space for named configs
            path = self.controller.plan_to_configuration(q_reduced)
            self.get_logger().info(f'Executing planned path with {len(path)} waypoints')
            execute_start = time.monotonic()
            self.controller.execute_path(path)
            self.get_logger().info(
                f'Path execution finished in {time.monotonic() - execute_start:.2f}s; '
                f'holding steady state (timeout {timeout:.1f}s)')
        except Exception as error:
            self.get_logger().warn(f'Planning or execution failed: {error}')
            goal_handle.abort()
            result = NamedConfig.Result()
            result.success = False
            return result

        # feedback and the error against the planned joint goal
        self.controller.set_reduced_configuration_target(q_reduced)

        # hold the final joint target with steady-state control
        self.controller.init_steady_state()
        start_time = time.monotonic()
        steady_state_converged = False
        blocked = None
        while time.monotonic() - start_time < timeout:
            step_start = time.monotonic()
            if goal_handle.is_cancel_requested:
                return self._canceled(goal_handle, NamedConfig.Result())

            steady_state_converged = self.controller.steady_state_step()
            joint_error = self._named_config_joint_error()
            self._publish_named_config_feedback(goal_handle, joint_error)

            blocked = self._limit_block_reason()
            if blocked is not None:
                break
            steady_state_converged = steady_state_converged or joint_error < self.threshold_joint
            if steady_state_converged:
                break

            time.sleep(max(0.0, self.controller.dt - (time.monotonic() - step_start)))

        self._log_hold_outcome(steady_state_converged, blocked)

        goal_handle.succeed()
        result = NamedConfig.Result()
        result.success = bool(steady_state_converged)
        return result

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Canceling goal')
        return CancelResponse.ACCEPT


def main(args=None):
    run('frame_task_server', FrameTaskServer, args=args)


if __name__ == '__main__':
    main()
