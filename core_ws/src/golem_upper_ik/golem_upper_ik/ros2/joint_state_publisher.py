'''
joint_state_publisher: /joint_states for the robot's body joints at 100 Hz, from /lowstate.

    ros2 run golem_upper_ik joint_state_publisher --ros-args -p robot:=h12

Parameters: robot (h12 or g1, required) and config (a controller config name or path, read
for topics.low_state; default the robot's DEFAULT_CONFIG). Every message names the profile's
body joints in /lowcmd slot order and carries the newest /lowstate's positions, all 0 before
the first sample. Joints off the body (the hands', the grippers') come from their own nodes.
'''
import os

# One BLAS thread before numpy loads, as in every node of this package
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from golem_upper_ik.config import load_controller_config  # noqa: E402
from golem_upper_ik.core.ros_io import StateSubscriber  # noqa: E402
from golem_upper_ik.profile import load_profile  # noqa: E402
from golem_upper_ik.ros2.utility import node_parameters, run  # noqa: E402

__all__ = ['JointStatePublisher', 'PUBLISH_HZ', 'main']

PUBLISH_HZ = 100.0


class JointStatePublisher(Node):
    def __init__(self, **node_kwargs):
        '''Reads the parameters, the profile and the config; start() starts publishing'''
        super().__init__('joint_state_publisher', **node_kwargs)
        self.robot, self.config_name = node_parameters(self)
        profile = load_profile(self.robot)
        config = load_controller_config(self.robot, self.config_name)
        self._names = list(profile.body_joints)
        self._state = StateSubscriber(self, profile, topic=config['topics']['low_state'])
        self._publisher = None

    def start(self):
        self._publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.publish_joint_states)
        self.get_logger().info(
            f'publishing /joint_states (robot {self.robot}, config {self.config_name})')

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._names
        msg.position = self._state.state['q'].tolist()
        self._publisher.publish(msg)

    def shutdown(self):
        self._state.shutdown()


def main(args=None):
    run('joint_state_publisher', JointStatePublisher, args=args)


if __name__ == '__main__':
    main()
