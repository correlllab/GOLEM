r"""
Simulated G1: the same core stack on DDS domain 1 with sim time and the sim configs.

    ros2 launch golem_bringup g1_sim.launch.py
    ros2 launch golem_bringup g1_sim.launch.py use_locomotion:=true   # walk with HOMIE
    ros2 launch golem_bringup g1_sim.launch.py use_locomotion:=true use_nav:=true
    ros2 launch golem_bringup g1_sim.launch.py use_locomotion:=true use_nav:=true \
        odom_source:=sim_gt   # the sim started with GOLEM_SIM_ODOM=1
    ros2 launch golem_bringup g1_sim.launch.py use_locomotion:=true use_nav:=true \
        frontier_explore:=true   # also serves /skill/frontier_explore

The G1 RoboCasa sim (golem_robocasa/g1/) emulates rt/arm_sdk (weight slot 29
blending with a leg/waist/arm hold), rt/dex3/* and, for use_locomotion:=true,
which switches the safety layer to sim_split, rt/lowcmd. The controller config
follows the safety config (sim with sim_arm_sdk, sim_split with sim_split).
g1_core refuses the real robot's configs here (arm_sdk, split, g1, g1_split);
the sim configs keep the faster arm speeds (1 rad/s, 0.5 m/s).

use_nav:=true (with use_locomotion:=true) adds g1_navigation on the sim's
MID-360 (/livox/lidar, /livox/imu), with sim time and g1_fastlio_sim.yaml, and
Nav2 drives the walking node on /cmd_vel; send goals to /navigate_to_pose.
odom_source:=fastlio (the default) needs the sim started without
GOLEM_SIM_ODOM, and odom_source:=sim_gt with GOLEM_SIM_ODOM=1. Otherwise
odom -> pelvis has two publishers (TF breaks) or none. frontier_explore:=true
adds g1_navigation's frontier explorer, which walks the robot through Nav2 to
the frontiers of /map.
tests/g1/test_sim_nav.sh is the closed-loop test.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    core = os.path.join(get_package_share_directory('golem_bringup'), 'launch', 'g1',
                        'g1_core.launch.py')
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_hands', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'safety_config', default_value='',
            description='sim_arm_sdk, or sim_split (the default with use_locomotion)'),
        DeclareLaunchArgument(
            'use_locomotion', default_value='false',
            description="HOMIE's walking policy on the legs, through sim_split"),
        DeclareLaunchArgument(
            'use_nav', default_value='false',
            description='g1_navigation: odometry, SLAM and Nav2 on /cmd_vel '
                        '(needs use_locomotion)'),
        DeclareLaunchArgument(
            'odom_source', default_value='fastlio',
            description="with use_nav: fastlio, or sim_gt (the sim's ground truth; start "
                        'the sim with GOLEM_SIM_ODOM=1)'),
        DeclareLaunchArgument(
            'frontier_explore', default_value='false',
            description='with use_nav: the frontier explorer, serving '
                        '/skill/frontier_explore'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(core),
            launch_arguments={
                'sim': 'true',
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'use_rviz': LaunchConfiguration('use_rviz'),
                'use_hands': LaunchConfiguration('use_hands'),
                'safety_config': LaunchConfiguration('safety_config'),
                # sim, or sim_split (a stiff waist) with use_locomotion
                'controller_config': '',
                'use_locomotion': LaunchConfiguration('use_locomotion'),
                'use_nav': LaunchConfiguration('use_nav'),
                'odom_source': LaunchConfiguration('odom_source'),
                'frontier_explore': LaunchConfiguration('frontier_explore'),
            }.items(),
        ),
    ])
