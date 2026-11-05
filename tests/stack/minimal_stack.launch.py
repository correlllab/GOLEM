"""Start the H1-2's real safety, arm and ALMI leg controllers for the isolated stack tests.

Starts, at once and in this order: golem_safety's safety_node, golem_upper_ik's
frame_task_server and joint_state_publisher, robot_state_publisher on the H1-2 URDF and, for
the locomotion case only, h12_lowerbody_rl's lowerbody_controller_node pinned to ALMI. Every
node runs on sim time with the H1-2's sim configs. If any node exits, the whole launch shuts
down.

Reads the environment, not launch arguments: ROS_DOMAIN_ID (1..232), TEST_CASE
(arms|locomotion, default arms) and SIMULATOR (isaac|robocasa, default isaac). Anything else
raises before a node starts.
"""
import os
from pathlib import Path

from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    """Return the stack test's nodes, each guarded so that its exit shuts the launch down."""
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    case = os.environ.get('TEST_CASE', 'arms')
    simulator = os.environ.get('SIMULATOR', 'isaac')
    # SAFETY: never on domain 0, the real robots' command bus.
    if not 1 <= domain <= 232 or case not in ('arms', 'locomotion') or simulator not in ('isaac', 'robocasa'):
        raise RuntimeError('Controller integration tests require an isolated domain and supported test case/simulator')
    sim_time = {'use_sim_time': True}
    nodes = [
        # SAFETY: the only writer of /lowcmd; the controllers below write its /safety/* inputs.
        Node(package='golem_safety', executable='safety_node', name='safety_node',
             parameters=[sim_time, {'robot': 'h12', 'config': 'sim_safety_split.yaml', 'sim': True}],
             output='screen'),
        # The arm controller under test in the arms case, and the torso and arm hold in locomotion;
        # its config pairs with the safety config above.
        Node(package='golem_upper_ik', executable='frame_task_server', name='frame_task_server',
             parameters=[sim_time, {'robot': 'h12', 'config': 'sim_safety_split.yaml'}], output='screen'),
        # /joint_states from /lowstate, for robot_state_publisher's TF.
        Node(package='golem_upper_ik', executable='joint_state_publisher', name='joint_state_publisher',
             parameters=[sim_time, {'robot': 'h12'}], output='screen'),
        Node(package='robot_state_publisher', executable='robot_state_publisher', name='robot_state_publisher',
             parameters=[sim_time, {'robot_description': Path(urdf_path('h12', ros=True)).read_text()}],
             output='screen'),
    ]
    if case == 'locomotion':
        # The leg controller under test: ALMI with no operator confirm, as the sim bringups run it.
        nodes.append(Node(
            package='h12_lowerbody_rl', executable='lowerbody_controller_node', name='lowerbody_controller_node',
            parameters=[sim_time, {
                'active_policy': 'almi',
                'engage_wait_for_confirm': False,
                'imu_offset_roll_deg': 0., 'imu_offset_pitch_deg': 0., 'imu_offset_yaw_deg': 0.,
                'max_state_gap': 0.5, 'cmd_vel_timeout': 0.5,
            }], output='screen'))
    # A dead controller invalidates the test, so no run continues with part of the stack.
    guards = [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=[
        EmitEvent(event=Shutdown(reason='A required integration-test ROS node exited')),
    ])) for node in nodes]
    return LaunchDescription(guards + nodes)
