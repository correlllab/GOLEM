import rclpy
import argparse
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from geometry_msgs.msg import PoseStamped

import time
import threading
import numpy as np

from custom_ros_messages.action import DualArm, NamedConfig
from g1_control.core.controller.arm_controller import ArmController
from g1_control.utility.controller_config import (
    init_channel_factory,
    load_controller_config,
    maybe_start_controller_logging,
)
from g1_control.utility.named_config import NAMED_CONFIGS
from g1_control.utility.goal_checks import unsupported_named_config_fields
from g1_control.utility.path_definition_ros import (
    URDF_PIN_PATH,
    CONFIG_DIR
)
from g1_control.ros2.utility import pose_to_matrix, matrix_to_pose

class DualArmServer(Node):
    def __init__(self,
                 timeout=None,
                 threshold_ik=None,
                 threshold_joint=None,
                 threshold_linear=None,
                 threshold_angular=None,
                 config_name='g1.yaml'):
        super().__init__('dual_arm_server')
        config = load_controller_config(config_name, config_dir=CONFIG_DIR)
        controller_cfg = config['controller']
        self.timeout = controller_cfg['timeout'] if timeout is None else timeout
        self.threshold_ik = (
            controller_cfg['threshold_ik']
            if threshold_ik is None else threshold_ik
        )
        self.threshold_joint = (
            controller_cfg['threshold_joint']
            if threshold_joint is None else threshold_joint
        )
        self.threshold_linear = (
            controller_cfg['threshold_linear']
            if threshold_linear is None else threshold_linear
        )
        self.threshold_angular = (
            controller_cfg['threshold_angular']
            if threshold_angular is None else threshold_angular
        )
        self.config_name = config_name

        self.controller = ArmController(
            URDF_PIN_PATH,
            init=True,
            visualize=False,
            config=config,
        )
        maybe_start_controller_logging(self.controller)

        # publisher of left and right end-effector poses
        self.left_ee_pose_publisher = self.create_publisher(
            PoseStamped,
            'left_ee_pose',
            10
        )
        self.right_ee_pose_publisher = self.create_publisher(
            PoseStamped,
            'right_ee_pose',
            10
        )
        # publisher of left and right end-effector target poses
        self.left_ee_target_publisher = self.create_publisher(
            PoseStamped,
            'left_ee_target',
            10
        )
        self.right_ee_target_publisher = self.create_publisher(
            PoseStamped,
            'right_ee_target',
            10
        )
        self.publisher_timer = self.create_timer(1.0 / 100, self.publisher_callback)

        # action server to control dual arms
        self._controller_lock = threading.Lock()
        self.dual_arm_server = ActionServer(
            self,
            DualArm,
            'dual_arm',
            execute_callback=self.dual_arm_callback,
            cancel_callback=self.cancel_callback
        )
        self.named_config_server = ActionServer(
            self,
            NamedConfig,
            'named_config',
            execute_callback=self.named_config_callback,
            cancel_callback=self.cancel_callback
        )
        self.get_logger().info('Dual Arm Server initialized')

    def _stamp_pose(self, pose):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = 'pelvis'
        pose_stamped.pose = pose
        return pose_stamped

    def publisher_callback(self):
        self.controller.update_robot_model()
        # transform and publish the pose
        left_ee_pose = matrix_to_pose(self.controller.left_ee_transformation)
        right_ee_pose = matrix_to_pose(self.controller.right_ee_transformation)
        left_ee_target = matrix_to_pose(self.controller.left_ee_target_transformation)
        right_ee_target = matrix_to_pose(self.controller.right_ee_target_transformation)
        self.left_ee_pose_publisher.publish(self._stamp_pose(left_ee_pose))
        self.right_ee_pose_publisher.publish(self._stamp_pose(right_ee_pose))
        self.left_ee_target_publisher.publish(self._stamp_pose(left_ee_target))
        self.right_ee_target_publisher.publish(self._stamp_pose(right_ee_target))

    def dual_arm_callback(self, goal_handle):
        with self._controller_lock:
            self.get_logger().info('Received goal')
            goal = goal_handle.request
            self.get_logger().info(str(goal))

            self.get_logger().info('Going to target end-effector poses')
            # set target
            self.controller.left_ee_target_transformation = pose_to_matrix(goal.left_target)
            self.controller.right_ee_target_transformation = pose_to_matrix(goal.right_target)

            # update ik solver with current state
            self.controller.update_ik_solver()

            duration = goal.duration.sec + goal.duration.nanosec * 1e-9
            timeout = duration if duration > 0.0 else self.timeout
            plan = goal.plan
            if plan:
                return self._plan_dual_arm_callback(goal_handle, timeout)
            else:
                return self._direct_dual_arm_callback(goal_handle, timeout)

    def _direct_dual_arm_callback(self, goal_handle, timeout):
        # main loop
        start_time = time.time()
        ik_converged = False
        steady_state_converged = False
        while time.time() - start_time < timeout:
            frame_start_time = time.time()
            if not ik_converged:
                vel = self.controller.control_step_reduced()
                vel_error = np.max(np.abs(vel))
            else:
                steady_state_converged = self.controller.steady_state_step()

            # handle cancel event
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal cancelled')
                goal_handle.canceled()
                result = DualArm.Result()
                result.success = False
                return result

            # compute errors
            errors = self._dual_arm_errors()
            # send feedback
            self._publish_dual_arm_feedback(goal_handle, errors)

            if not ik_converged:
                if vel_error < self.threshold_ik:
                    ik_converged = True
                    self.controller.init_steady_state()
                    self.get_logger().info(
                        'IK converged; entering steady-state hold'
                    )
            else:
                steady_state_converged = steady_state_converged or (
                    errors['left_linear'] < self.threshold_linear and
                    errors['left_angular'] < self.threshold_angular and
                    errors['right_linear'] < self.threshold_linear and
                    errors['right_angular'] < self.threshold_angular
                )
                if steady_state_converged:
                    break

            time.sleep(max(
                0.0,
                self.controller.dt - (time.time() - frame_start_time)
            ))

        if not ik_converged:
            self.get_logger().warn('Timed out before IK convergence')
        elif not steady_state_converged:
            self.get_logger().warn('Timed out during steady-state hold')
        else:
            self.get_logger().info('Goal reached')

        goal_handle.succeed()
        result = DualArm.Result()
        result.success = bool(ik_converged and steady_state_converged)
        return result

    def _plan_dual_arm_callback(self, goal_handle, timeout):
        self.get_logger().info('Planning path to target end-effector poses')
        try:
            # solve ik, plan a reduced-joint path, then execute it
            path = self.controller.plan_to_ik_target(ik_timeout=self.timeout)
            self.get_logger().info(
                f'Executing planned path with {len(path)} waypoints'
            )
            self.controller.execute_path(path)
        except Exception as err:
            self.get_logger().warn(f'Planning or execution failed: {err}')
            goal_handle.abort()
            result = DualArm.Result()
            result.success = False
            return result

        # hold the final target using the existing steady-state controller
        self.controller.init_steady_state()
        start_time = time.time()
        steady_state_converged = False
        while time.time() - start_time < timeout:
            frame_start_time = time.time()

            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal cancelled')
                goal_handle.canceled()
                result = DualArm.Result()
                result.success = False
                return result

            self.controller.steady_state_step()
            # publish feedback while waiting for final pose convergence
            errors = self._dual_arm_errors()
            self._publish_dual_arm_feedback(goal_handle, errors)

            steady_state_converged = (
                errors['left_linear'] < self.threshold_linear and
                errors['left_angular'] < self.threshold_angular and
                errors['right_linear'] < self.threshold_linear and
                errors['right_angular'] < self.threshold_angular
            )
            if steady_state_converged:
                break

            time.sleep(max(
                0.0,
                self.controller.dt - (time.time() - frame_start_time)
            ))

        if steady_state_converged:
            self.get_logger().info('Goal reached')
        else:
            self.get_logger().warn('Timed out during steady-state hold')

        goal_handle.succeed()
        result = DualArm.Result()
        result.success = bool(steady_state_converged)
        return result

    def _dual_arm_errors(self):
        return {
            'left_linear': float(np.linalg.norm(self.controller.left_ee_error[:3])),
            'left_angular': float(
                np.linalg.norm(self.controller.left_ee_error[3:])
            ),
            'right_linear': float(
                np.linalg.norm(self.controller.right_ee_error[:3])
            ),
            'right_angular': float(
                np.linalg.norm(self.controller.right_ee_error[3:])
            ),
        }

    def _publish_dual_arm_feedback(self, goal_handle, errors):
        feedback_msg = DualArm.Feedback()
        feedback_msg.left_error_linear = errors['left_linear']
        feedback_msg.left_error_angular = errors['left_angular']
        feedback_msg.right_error_linear = errors['right_linear']
        feedback_msg.right_error_angular = errors['right_angular']
        goal_handle.publish_feedback(feedback_msg)

    def _named_config_joint_error(self):
        return float(np.max(
            np.abs(self.controller.reduced_configuration_error)
        ))

    def _publish_named_config_feedback(self, goal_handle, joint_error):
        feedback_msg = NamedConfig.Feedback()
        feedback_msg.joint_error = joint_error
        goal_handle.publish_feedback(feedback_msg)

    def named_config_callback(self, goal_handle):
        unsupported = unsupported_named_config_fields(goal_handle.request)
        if unsupported:
            fields = ', '.join(unsupported)
            self.get_logger().warn(f'{fields} not supported on the G1')
            goal_handle.abort()
            result = NamedConfig.Result()
            result.success = False
            return result

        with self._controller_lock:
            self.get_logger().info('Received named config goal')
            goal = goal_handle.request
            self.get_logger().info(str(goal))

            # check if the named config exists
            config_name = goal.config_name
            if config_name not in NAMED_CONFIGS:
                self.get_logger().warn(f'Named config "{config_name}" not found')
                goal_handle.abort()
                result = NamedConfig.Result()
                result.success = False
                return result

            self.get_logger().info(f'Going to named config: {config_name}')
            q_reduced = NAMED_CONFIGS[config_name]

            # update ik solver with current state
            self.controller.update_ik_solver()

            duration = goal.duration.sec + goal.duration.nanosec * 1e-9
            timeout = duration if duration > 0.0 else self.timeout
            plan = goal.plan
            if plan:
                return self._plan_named_config_callback(
                    goal_handle,
                    q_reduced,
                    timeout,
                )
            else:
                return self._direct_named_config_callback(
                    goal_handle,
                    q_reduced,
                    timeout,
                )

    def _direct_named_config_callback(self, goal_handle, q_reduced, timeout):
        # main loop
        start_time = time.time()
        ik_converged = False
        steady_state_converged = False
        while time.time() - start_time < timeout:
            frame_start_time = time.time()
            if not ik_converged:
                vel = self.controller.goto_reduced_configuration(q_reduced)
                vel_error = np.max(np.abs(vel))
            else:
                steady_state_converged = self.controller.steady_state_step()

            # handle cancel event
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal cancelled')
                goal_handle.canceled()
                result = NamedConfig.Result()
                result.success = False
                return result

            # compute error
            joint_error = self._named_config_joint_error()
            # send feedback
            self._publish_named_config_feedback(goal_handle, joint_error)

            if not ik_converged:
                if vel_error < self.threshold_ik:
                    ik_converged = True
                    self.controller.init_steady_state()
                    self.get_logger().info(
                        'IK converged; entering steady-state hold'
                    )
            else:
                steady_state_converged = (
                    steady_state_converged or
                    joint_error < self.threshold_joint
                )
                if steady_state_converged:
                    break

            time.sleep(max(
                0.0,
                self.controller.dt - (time.time() - frame_start_time)
            ))

        if not ik_converged:
            self.get_logger().warn('Timed out before IK convergence')
        elif not steady_state_converged:
            self.get_logger().warn('Timed out during steady-state hold')
        else:
            self.get_logger().info('Goal reached')

        goal_handle.succeed()
        result = NamedConfig.Result()
        result.success = bool(ik_converged and steady_state_converged)
        return result

    def _plan_named_config_callback(self, goal_handle, q_reduced, timeout):
        self.get_logger().info('Planning path to named configuration')
        try:
            # plan directly in reduced joint space for named configs
            path = self.controller.plan_to_configuration(q_reduced)
            self.get_logger().info(
                f'Executing planned path with {len(path)} waypoints'
            )
            self.controller.execute_path(path)
        except Exception as err:
            self.get_logger().warn(f'Planning or execution failed: {err}')
            goal_handle.abort()
            result = NamedConfig.Result()
            result.success = False
            return result

        # match feedback/error target to the planned joint goal
        self.controller.set_reduced_configuration_target(q_reduced)

        # hold the final joint target using steady-state control
        self.controller.init_steady_state()
        start_time = time.time()
        steady_state_converged = False
        while time.time() - start_time < timeout:
            frame_start_time = time.time()

            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal cancelled')
                goal_handle.canceled()
                result = NamedConfig.Result()
                result.success = False
                return result

            steady_state_converged = self.controller.steady_state_step()
            joint_error = self._named_config_joint_error()

            # publish joint-space feedback during the hold phase
            self._publish_named_config_feedback(goal_handle, joint_error)

            steady_state_converged = (
                steady_state_converged or
                joint_error < self.threshold_joint
            )
            if steady_state_converged:
                break

            time.sleep(max(
                0.0,
                self.controller.dt - (time.time() - frame_start_time)
            ))

        if steady_state_converged:
            self.get_logger().info('Goal reached')
        else:
            self.get_logger().warn('Timed out during steady-state hold')

        goal_handle.succeed()
        result = NamedConfig.Result()
        result.success = bool(steady_state_converged)
        return result

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Canceling goal')
        return CancelResponse.ACCEPT


def main(args=None):
    parser = argparse.ArgumentParser(description='Dual arm ROS2 action server')
    parser.add_argument('--config', type=str, default='g1.yaml', help='YAML file name under config/')
    parsed_args, ros_args = parser.parse_known_args(args=args)

    config = load_controller_config(parsed_args.config, config_dir=CONFIG_DIR)
    init_channel_factory(config)
    rclpy.init(args=ros_args)
    node = DualArmServer(config_name=parsed_args.config)
    try:
        rclpy.spin(node, executor=rclpy.executors.MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass
    finally:
        node.controller.shutdown()
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
