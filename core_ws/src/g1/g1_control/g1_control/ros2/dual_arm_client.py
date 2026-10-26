import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped

from pynput import keyboard

from custom_ros_messages.action import DualArm, NamedConfig
from g1_control.utility.named_config import NAMED_CONFIGS
from g1_control.ros2.utility import input_pose

class DualArmClient(Node):
    def __init__(self):
        super().__init__('dual_arm_client')
        self.dual_arm_client = ActionClient(
            self,
            DualArm,
            'dual_arm'
        )
        self.named_config_client = ActionClient(
            self,
            NamedConfig,
            'named_config'
        )
        self.goal_handle = None

        # subscriber of left and right end-effector poses
        self.left_ee_pose_subscriber = self.create_subscription(
            PoseStamped,
            'left_ee_pose',
            self.subscribe_left_ee_pose,
            10
        )
        self.right_ee_pose_subscriber = self.create_subscription(
            PoseStamped,
            'right_ee_pose',
            self.subscribe_right_ee_pose,
            10
        )
        # subscriber of left and right end-effector target poses
        self.left_ee_target_subscriber = self.create_subscription(
            PoseStamped,
            'left_ee_target',
            self.subscribe_left_ee_target,
            10
        )
        self.right_ee_target_subscriber = self.create_subscription(
            PoseStamped,
            'right_ee_target',
            self.subscribe_right_ee_target,
            10
        )
        self.left_ee_pose = None
        self.right_ee_pose = None
        self.left_ee_target = None
        self.right_ee_target = None

    def subscribe_left_ee_pose(self, msg):
        self.left_ee_pose = msg.pose

    def subscribe_right_ee_pose(self, msg):
        self.right_ee_pose = msg.pose

    def subscribe_left_ee_target(self, msg):
        self.left_ee_target = msg.pose

    def subscribe_right_ee_target(self, msg):
        self.right_ee_target = msg.pose

    def send_dual_arm_goal(self, left_target: Pose=None, right_target: Pose=None,
                           plan=False):
        goal_msg = DualArm.Goal()

        # Convert poses to transformation matrices
        if left_target is None:
            left_target = Pose()
        if right_target is None:
            right_target = Pose()
        goal_msg.left_target = left_target
        goal_msg.right_target = right_target
        goal_msg.plan = plan

        self.get_logger().info('Waiting for action server...')
        self.dual_arm_client.wait_for_server()

        # send action
        self.get_logger().info('Sending goal...')
        future = self.dual_arm_client.send_goal_async(
            goal_msg,
            feedback_callback=self.dual_arm_feedback
        )
        rclpy.spin_until_future_complete(self, future)
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().warn('Goal was rejected')
            return

        # start a cancel listener thread
        self.get_logger().info('Goal accepted, waiting for result...')
        self.get_logger().info('Press BACKSPACE to cancel the goal')
        listener = keyboard.Listener(
            on_press=self._keyboard_cancel
        )
        listener.start()

        # wait till finish
        future_result = self.goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, future_result)
        result = future_result.result().result
        self.get_logger().info(f'Final result: success = {result.success}')
        # stop the cancel listener thread
        listener.stop()

    def dual_arm_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Left Error Linear: {feedback.left_error_linear:.4f}, ' +
                               f'Left Error Angular: {feedback.left_error_angular:.4f}')
        self.get_logger().info(f'Right Error Linear: {feedback.right_error_linear:.4f}, ' +
                               f'Right Error Angular: {feedback.right_error_angular:.4f}')

    def send_named_config_goal(self, config_name: str, plan=False):
        goal_msg = NamedConfig.Goal()
        goal_msg.config_name = config_name
        goal_msg.plan = plan

        self.get_logger().info('Waiting for named config action server...')
        self.named_config_client.wait_for_server()

        # send action
        self.get_logger().info(f'Sending named config goal: {config_name}...')
        future = self.named_config_client.send_goal_async(
            goal_msg,
            feedback_callback=self.named_config_feedback
        )
        rclpy.spin_until_future_complete(self, future)
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().warn('Named config goal was rejected')
            return

        # start a cancel listener thread
        self.get_logger().info('Named config goal accepted, waiting for result...')
        self.get_logger().info('Press BACKSPACE to cancel the goal')
        listener = keyboard.Listener(
            on_press=self._keyboard_cancel
        )
        listener.start()

        # wait till finish
        future_result = self.goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, future_result)
        result = future_result.result().result
        self.get_logger().info(f'Final result: success = {result.success}')
        # stop the cancel listener thread
        listener.stop()

    def named_config_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Joint Error: {feedback.joint_error:.4f}')

    def _keyboard_cancel(self, key):
        if key == keyboard.Key.backspace:
            if self.goal_handle is not None:
                self.get_logger().info('Cancelling goal...')
                self.goal_handle.cancel_goal()


def input_plan():
    while True:
        choice = input('Use OMPL path planning? (y/n): ').strip().lower()
        if choice in ('y', 'n'):
            return choice == 'y'
        print('Please enter y or n')


def input_config_name_or_poses():
    '''
    Ask user for config name or manual pose input
    Returns (config_name, left_pose, right_pose, plan) tuple
    If config_name is provided, poses will be None
    If no config_name, poses will be provided
    '''
    print(f'Available config names: {list(NAMED_CONFIGS.keys())}')

    while True:
        choice = input(
            'Enter config name (or press Enter for manual poses): '
        ).strip()
        if choice == '' or choice in NAMED_CONFIGS:
            break
        print(f'Unknown config name: {choice}')

    if choice:
        # user entered a config name
        plan = input_plan()
        return choice, None, None, plan
    else:
        print('Enter left end-effector pose:')
        left_pose = input_pose()
        print('Enter right end-effector pose:')
        right_pose = input_pose()
        plan = input_plan()
        return '', left_pose, right_pose, plan


def main(args=None):
    rclpy.init(args=args)
    node = DualArmClient()

    try:
        while rclpy.ok():
            config_name, left_pose, right_pose, plan = (
                input_config_name_or_poses()
            )
            if config_name != '':
                node.send_named_config_goal(config_name, plan=plan)
            else:
                node.send_dual_arm_goal(left_pose, right_pose, plan=plan)

            input('Press any key to continue...') # flush the input buffer
            cont = input('Do you want to send another goal? (y/n): ').lower()
            if cont != 'y':
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
