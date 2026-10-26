import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

import numpy as np

from g1_control.core.robot_model import RobotModel
from g1_control.utility.controller_config import (
    init_channel_factory,
    load_controller_config,
)
from g1_control.utility.path_definition_ros import URDF_PIN_PATH
from g1_control.utility.joint_definition import BODY_JOINTS

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # initialize robot model
        # publishes the 29 body joints; the Dex3-1 hand joints come from the hand driver
        self.robot_model = RobotModel(URDF_PIN_PATH)
        self.robot_model.init_subscriber()
        self.get_logger().info('robot_model successfully initialized')

        # create publisher for joint states
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(1.0 / 100, self.publish_joint_states)

    def publish_joint_states(self):
        # update robot model
        self.robot_model.update_kinematics()
        # create JointState message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = BODY_JOINTS
        msg.position = self.robot_model.state['q'].tolist()

        self.publisher.publish(msg)

def main():
    config = load_controller_config()
    init_channel_factory(config)
    rclpy.init()
    node = JointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
