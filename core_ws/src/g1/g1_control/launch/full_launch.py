from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import os
from g1_control.utility.path_definition_ros import PACKAGE_PATH, URDF_ROS_PATH

def generate_launch_description():
    package_name = 'g1_control'
    config = LaunchConfiguration('config')

    with open(URDF_ROS_PATH, 'r') as urdf_file:
        robot_description = urdf_file.read()

    rviz_config_path = f'{PACKAGE_PATH}/rviz/default.rviz'

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
        TimerAction( # slight delay to ensure robot_description is set
            period=1.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', rviz_config_path] if os.path.exists(rviz_config_path) else [],
                    output='screen'
                )
            ]
        ),
        # TimerAction( # slight delay to ensure server is ready
        #     period=3.0,
        #     actions=[
        #         Node(
        #             package=package_name,
        #             executable='dual_arm_client',
        #             name='dual_arm_client',
        #             output='screen'
        #         )
        #     ]
        # ),
    ])
