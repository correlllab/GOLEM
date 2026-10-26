import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from pynput import keyboard

from custom_ros_messages.action import FrameTask, NamedConfig
from g1_control.utility.named_config import NAMED_CONFIGS
from g1_control.ros2.utility import input_pose

class FrameTaskClient(Node):
    def __init__(self):
        super().__init__('frame_task_client')

        # create the action client
        self.action_client = ActionClient(
            self,
            FrameTask,
            'frame_task'   # action name
        )
        self.named_config_client = ActionClient(
            self,
            NamedConfig,
            'named_config'
        )
        self.goal_handle = None

    def send_frame_task_goal(self, frame_names=None, frame_targets=None,
                             plan=False):
        goal_msg = FrameTask.Goal()

        goal_msg.frame_names = frame_names if frame_names is not None else []
        goal_msg.frame_targets = frame_targets if frame_targets is not None else []
        goal_msg.plan = plan

        self.get_logger().info('Waiting for action server...')
        self.action_client.wait_for_server()

        # send action
        self.get_logger().info('Sending goal...')
        future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.frame_task_feedback
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

    def frame_task_feedback(self, feedback_msg):
        errors_linear = list(feedback_msg.feedback.errors_linear)
        errors_angular = list(feedback_msg.feedback.errors_angular)
        self.get_logger().info(f'Linear errors: {errors_linear}')
        self.get_logger().info(f'Angular errors: {errors_angular}')

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


def input_config_name_or_frame_task():
    '''
    Ask user for config name or manual frame task input
    Returns (config_name, frame_names, frame_poses, plan) tuple
    If config_name is provided, frame_name and pose will be None
    If no config_name, frame_name and pose will be provided
    '''
    print(f'Available config names: {list(NAMED_CONFIGS.keys())}')

    while True:
        choice = input(
            'Enter config name (or press Enter for manual frame task): '
        ).strip()
        if choice == '' or choice in NAMED_CONFIGS:
            break
        print(f'Unknown config name: {choice}')

    if choice:
        # user entered a config name
        plan = input_plan()
        return choice, [], [], plan
    else:
        # user wants manual frame task input
        frame_names = []
        frame_poses = []
        while True:
            frame_name, pose = input_frame_task()
            frame_names.append(frame_name)
            frame_poses.append(pose)
            cont = input('Do you want to add another frame task? (y/n): ').lower()
            if cont != 'y':
                break
        plan = input_plan()
        return '', frame_names, frame_poses, plan


def input_frame_task():
    '''Get frame name and pose from user'''
    frame_name = input('Enter frame name: ')
    target_pose = input_pose()

    return frame_name, target_pose


def main(args=None):
    rclpy.init(args=args)
    node = FrameTaskClient()

    try:
        while rclpy.ok():
            config_name, frame_names, frame_poses, plan = (
                input_config_name_or_frame_task()
            )
            if config_name != '':
                node.send_named_config_goal(config_name, plan=plan)
            else:
                node.send_frame_task_goal(frame_names, frame_poses, plan=plan)

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

if __name__ == '__main__':
    main()
