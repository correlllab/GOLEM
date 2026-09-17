"""Actual upper-body/safety/ALMI stack for isolated simulator integration tests."""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    case = os.environ.get('TEST_CASE', 'arms')
    simulator = os.environ.get('SIMULATOR', 'isaac')
    if not 1 <= domain <= 232 or case not in ('arms', 'locomotion') or simulator not in ('isaac', 'robocasa'):
        raise RuntimeError('Controller integration tests require an isolated domain and supported test case/simulator')
    sim_time = {'use_sim_time': True}
    nodes = [
        Node(package='h12_safety_layer', executable='safety_node', name='safety_node',
             arguments=['--config', 'sim_safety_split.yaml'], parameters=[sim_time], output='screen'),
        Node(package='h12_ros2_controller', executable='frame_task_server', name='frame_task_server',
             arguments=['--config', 'sim_safety_split.yaml'], parameters=[sim_time], output='screen'),
        Node(package='h12_ros2_controller', executable='joint_state_publisher', name='joint_state_publisher',
             parameters=[sim_time], output='screen'),
        Node(package='robot_state_publisher', executable='robot_state_publisher', name='robot_state_publisher',
             parameters=[sim_time, {'robot_description': Path('/home/code/CL_Assets/ros_assets/h1_2_magpie_ros.urdf').read_text()}],
             output='screen'),
    ]
    if case == 'locomotion':
        nodes.append(Node(
            package='h12_lowerbody_rl', executable='lowerbody_controller_node', name='lowerbody_controller_node',
            parameters=[sim_time, {
                'active_policy': 'almi',
                'engage_wait_for_confirm': False,
                'imu_offset_roll_deg': 0., 'imu_offset_pitch_deg': 0., 'imu_offset_yaw_deg': 0.,
            }], output='screen'))
    # A dead controller invalidates a test; never continue with a partial stack.
    guards = [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=[
        EmitEvent(event=Shutdown(reason='A required integration-test ROS node exited')),
    ])) for node in nodes]
    return LaunchDescription(guards + nodes)
