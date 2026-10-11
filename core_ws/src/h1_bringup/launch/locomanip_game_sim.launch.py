"""The Locomanipulation_game policy in the sim, alone: the split safety layer and locomotion_game_deploy's
marl_deploy_node, as h1_sim_bringup.launch.py lowerbody:=marl_ik runs them, without the rest of the bringup
(no model servers, rviz, nav or skills).

    export ROS_DOMAIN_ID=<nonzero>     # the same domain as the RoboCasa container; 0 is the real robot
    ros2 launch h1_bringup locomanip_game_sim.launch.py [policy_mode:=torch|zero] [config:=<deploy yaml>]

ROS_DOMAIN_ID must be exported in the launching shell: the safety layer's DDS half reads it from the environment
(see h1_sim_bringup.launch.py). marl_deploy_node pre-poses the robot, engages once it settles and releases the
elastic band; then it walks on /cmd_vel and reaches on ~/arm_goal (locomotion_game_deploy/README.md).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory('h1_bringup')
    deploy_share = get_package_share_directory('locomotion_game_deploy')
    sim_time_param = {'use_sim_time': True}

    return LaunchDescription([
        # 'torch': the exported networks next to the config; 'zero': stand-ins for plumbing tests (the robot falls
        # once the band releases)
        DeclareLaunchArgument('policy_mode', default_value='torch'),
        # the deploy config; export.yaml beside it holds the checkpoint's per-run values
        DeclareLaunchArgument('config', default_value=os.path.join(deploy_share, 'policies', 'marl_ik', 'marl_ik.yaml')),

        Node(
            package='h12_safety_layer',
            executable='safety_node',
            name='safety_node',
            parameters=[sim_time_param],
            arguments=['--config', 'sim_safety_split.yaml'],
            output='screen',
        ),
        Node(
            package='locomotion_game_deploy',
            executable='marl_deploy_node',
            name='marl_deploy_node',   # MUST match the yaml key
            parameters=[sim_time_param,
                        os.path.join(bringup_share, 'config', 'locomotion_game_deploy_sim.yaml'),
                        {'policy_mode': LaunchConfiguration('policy_mode'),
                         'config': LaunchConfiguration('config')}],
            output='screen',
        ),
    ])
