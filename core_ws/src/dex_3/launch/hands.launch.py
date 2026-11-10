"""Start the Dex3-1 hand driver, dex3_hands, on its own.

Starts one node, dex3_hands, which drives the hands named in `sides` over
rt/dex3/<side>/cmd and stays silent until a hand's first service call. The
G1 bringups start the same node themselves (golem_bringup's g1_core.launch.py,
use_hands); this file is for running the hands alone.

    ros2 launch dex_3 hands.launch.py                  # both hands
    ros2 launch dex_3 hands.launch.py sides:="[left]"  # one hand
    ros2 launch dex_3 hands.launch.py kp:=1.0 kd:=0.1
    ros2 launch dex_3 hands.launch.py on_shutdown:=silent  # no release on exit

The DDS domain is ROS_DOMAIN_ID's (0 is the real robot). With ROS_DOMAIN_ID
unset, domain_id must be given and match the domain ROS itself then runs on
(0), or the driver refuses to start: set ROS_DOMAIN_ID instead.
"""

from typing import List

from dex_3.dex3 import DEFAULT_KD, DEFAULT_KP, DEFAULT_MAX_SPEED
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Declare the hand driver's arguments and start dex3_hands."""
    return LaunchDescription([
        DeclareLaunchArgument('sides', default_value='[left, right]',
                              description='hands to drive: a list of left and/or right'),
        DeclareLaunchArgument('kp', default_value=str(DEFAULT_KP),
                              description='finger PD stiffness, >= 0'),
        DeclareLaunchArgument('kd', default_value=str(DEFAULT_KD),
                              description='finger PD damping, >= 0'),
        DeclareLaunchArgument('max_speed', default_value=str(DEFAULT_MAX_SPEED),
                              description='cap on any finger joint, rad/s, > 0'),
        DeclareLaunchArgument('on_shutdown', default_value='release',
                              description='on exit, release (fingers go limp) or silent '
                                          '(stop publishing; the hand keeps its last command)'),
        DeclareLaunchArgument('domain_id', default_value='-1',
                              description='Unitree DDS domain if ROS_DOMAIN_ID is unset '
                                          '(-1: none); must equal the ROS domain'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true or false: run on /clock (a sim): the moves, '
                                          'the command rate and the /joint_states stamps '
                                          'count sim time'),
        # The hand driver; it writes rt/dex3/<side>/cmd itself, outside golem_safety.
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
