"""Show the G1 in RViz with joint sliders.

    ros2 launch g1_description display.launch.py                # body + Dex3-1 hands
    ros2 launch g1_description display.launch.py hands:=false   # body only
    ros2 launch g1_description display.launch.py gui:=false rviz:=false   # headless TF
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from g1_description.paths import urdf_path


def _setup(context):
    hands = LaunchConfiguration('hands').perform(context).lower() == 'true'
    with open(urdf_path('with_hand' if hands else 'body', ros=True)) as f:
        robot_description = f.read()
    rviz_config = os.path.join(get_package_share_directory('g1_description'), 'rviz', 'g1.rviz')
    gui = LaunchConfiguration('gui')
    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            condition=IfCondition(gui),
            output='screen',
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            condition=UnlessCondition(gui),
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen',
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('hands', default_value='true',
                              description='true: body + Dex3-1 hands; false: body only'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='joint sliders (false: all joints held at 0)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
