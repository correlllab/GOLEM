'''ROS 2 node running HOMIE's walking policy on the G1's legs'''

import argparse

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

from g1_locomotion.config import init_channel_factory, load_config, resolve_config_path
from g1_locomotion.driver import LocomotionDriver
from g1_locomotion.locomotion import LocomotionController, load_policy

# where the ros container mounts this package's policy/ (deploy.onnx, g1.yaml, homie_policy.py)
DEFAULT_POLICY_DIR = '/home/code/g1_locomotion'


class LocomotionNode(Node):
    '''
    - /cmd_vel (geometry_msgs/Twist): linear.x, linear.y (m/s) and angular.z
      (rad/s) in the base frame; older than commands.timeout means stop
    - /locomotion/height (std_msgs/Float32): the pelvis height target, m
      (standing, 0.74, until one arrives)
    The legs go to g1_safety_layer's lower-body input over Unitree DDS; see
    LocomotionController for the startup and the faults.

    use_sim_time is always false, whatever the launch passes: the node times
    everything by rt/lowstate's tick and the monotonic clock, and sim time
    would only add a /clock subscription (500 Hz in the sim) in Python
    '''

    def __init__(self, config_name='g1.yaml'):
        no_sim_time = Parameter('use_sim_time', Parameter.Type.BOOL, False)
        super().__init__('locomotion_node', parameter_overrides=[no_sim_time])
        self.declare_parameter('policy_dir', DEFAULT_POLICY_DIR)
        policy_dir = self.get_parameter('policy_dir').value
        policy = load_policy(policy_dir)
        config_path = resolve_config_path(config_name)
        config = load_config(config_path, policy.config['cmd_range'])

        self.get_logger().info(
            f'Loaded {config_path} and the policy in {policy_dir}. Silent until '
            f'{config["topics"]["low_state"]} is steady and upright and the arm controller '
            f'commands on {config["topics"]["upper_cmd"]}; then the legs move to the '
            f'default pose over {config["startup"]["ramp_seconds"]:.1f} s and the policy starts')

        domain_id = init_channel_factory(config)
        controller = LocomotionController(policy, config, log=self._log)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        self.create_subscription(Float32, '/locomotion/height', self._on_height, 10)
        self._driver = LocomotionDriver(controller, config['topics'], domain_id)

    def _log(self, level, message):
        # one call per severity: rclpy refuses a change of severity at one call site
        if level == 'error':
            self.get_logger().error(message)
        elif level == 'warn':
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)

    def _on_cmd_vel(self, msg: Twist):
        self._driver.set_cmd_vel(msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_height(self, msg: Float32):
        self._driver.set_height(msg.data)

    def destroy_node(self):
        # stop publishing; the safety layer's leg watchdog then estops
        self._driver.shutdown()
        return super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description="HOMIE's walking policy on the G1's legs")
    parser.add_argument('--config', type=str, default='g1.yaml',
                        help='config name under config/ (g1.yaml or sim.yaml), or a path')
    parsed_args, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = LocomotionNode(config_name=parsed_args.config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
