from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from g1_control.utility.path_definition_ros import URDF_ROS_PATH

def generate_launch_description():
    package_name = 'g1_control'
    config = LaunchConfiguration('config')

    with open(URDF_ROS_PATH, 'r') as urdf_file:
        robot_description = urdf_file.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value='g1.yaml',
            description='Controller config file name under config/'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen'
        ),
        Node(
            package=package_name,
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen'
        ),
        # TimerAction(
        #     period=2.0,
        #     actions=[
        #         Node(
        #             package=package_name,
        #             executable='dual_arm_server',
        #             name='dual_arm_server',
        #             arguments=['--config', config],
        #             output='screen'
        #         ),
        #     ]
        # ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package=package_name,
                    executable='frame_task_server',
                    name='frame_task_server',
                    arguments=['--config', config],
                    output='screen'
                ),
            ]
        ),
    ])
