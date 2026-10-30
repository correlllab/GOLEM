'''Start the Dex3-1 hand driver.

    ros2 launch dex_3 hands.launch.py                  # both hands
    ros2 launch dex_3 hands.launch.py sides:="[left]"  # one hand
    ros2 launch dex_3 hands.launch.py kp:=1.0 kd:=0.1
    ros2 launch dex_3 hands.launch.py on_shutdown:=silent  # don't release on exit

The DDS domain is ROS_DOMAIN_ID's (0 is the real robot). With ROS_DOMAIN_ID
unset, domain_id must be given and match the domain ROS itself then runs on
(0), or the driver refuses to start: set ROS_DOMAIN_ID instead.
'''

from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('sides', default_value='[left, right]'),
        DeclareLaunchArgument('kp', default_value='1.5'),
        DeclareLaunchArgument('kd', default_value='0.1'),
        DeclareLaunchArgument('max_speed', default_value='2.0',
                              description='rad/s cap on any finger joint'),
        DeclareLaunchArgument('on_shutdown', default_value='release',
                              description='release (fingers go limp) or silent (stop publishing)'),
        DeclareLaunchArgument('domain_id', default_value='-1',
                              description='DDS domain if ROS_DOMAIN_ID is unset (-1: none)'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(
            package='dex_3',
            executable='dex3_hands',
            name='dex3_hands',
            output='screen',
            parameters=[{
                'sides': ParameterValue(LaunchConfiguration('sides'), value_type=List[str]),
                'kp': ParameterValue(LaunchConfiguration('kp'), value_type=float),
                'kd': ParameterValue(LaunchConfiguration('kd'), value_type=float),
                'max_speed': ParameterValue(LaunchConfiguration('max_speed'), value_type=float),
                'on_shutdown': ParameterValue(LaunchConfiguration('on_shutdown'), value_type=str),
                'domain_id': ParameterValue(LaunchConfiguration('domain_id'), value_type=int),
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            }],
        ),
    ])
