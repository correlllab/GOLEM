"""Publish /joint_states for the robot's body joints at 100 Hz, from /lowstate.

    ros2 run golem_upper_ik joint_state_publisher --ros-args -p robot:=h12

Parameters: robot (required; a robot with data in golem_upper_ik's config/) and
low_state_topic (the robot's state, default /lowstate). Every message names the profile's
body joints in /lowcmd slot order and carries the newest state's positions, all 0 before
the first sample. Joints off the body (the hands', the grippers') come from their own nodes.
"""
import os

# One BLAS thread, set before numpy loads: by default OpenBLAS starts a busy-waiting worker
# per core
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from golem_upper_ik.core.ros_io import StateSubscriber  # noqa: E402
from golem_upper_ik.profile import load_profile  # noqa: E402
from golem_upper_ik.ros2.utility import robot_parameter, run  # noqa: E402

__all__ = ['JointStatePublisher', 'LOW_STATE_TOPIC', 'PUBLISH_HZ', 'main']

PUBLISH_HZ = 100.0
# the low_state_topic parameter's default, the topic both robots publish their state on
LOW_STATE_TOPIC = '/lowstate'


class JointStatePublisher(Node):
    """Republish the newest /lowstate's body-joint positions as /joint_states."""

    def __init__(self, **node_kwargs):
        """Read the parameters and the profile; start() starts publishing."""
        super().__init__('joint_state_publisher', **node_kwargs)
        self.robot = robot_parameter(self)
        profile = load_profile(self.robot)
        topic = self.declare_parameter('low_state_topic', LOW_STATE_TOPIC).value
        self.summary = f'robot {self.robot}, state from {topic}'
        self._names = list(profile.body_joints)
        self._state = StateSubscriber(self, profile, topic=topic)
        self._publisher = None

    def start(self):
        """Start publishing /joint_states at PUBLISH_HZ."""
        self._publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.publish_joint_states)
        self.get_logger().info(f'publishing /joint_states ({self.summary})')

    def publish_joint_states(self):
        """Publish the newest measured body-joint positions, rad."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._names
        msg.position = self._state.state['q'].tolist()
        self._publisher.publish(msg)

    def shutdown(self):
        """Stop taking /lowstate samples."""
        self._state.shutdown()


def main(args=None):
    """Run the node until SIGINT or SIGTERM (utility.run)."""
    run('joint_state_publisher', JointStatePublisher, args=args)


if __name__ == '__main__':
    main()
