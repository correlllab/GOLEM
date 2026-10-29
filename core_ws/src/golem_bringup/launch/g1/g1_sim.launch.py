r"""
Simulated G1: the same core stack on DDS domain 1 with sim time and the sim configs.

    ros2 launch golem_bringup g1_sim.launch.py
    ros2 launch golem_bringup g1_sim.launch.py use_nav:=true
    ros2 launch golem_bringup g1_sim.launch.py use_nav:=true \
        odom_source:=sim_gt   # the sim started with GOLEM_SIM_ODOM=1
    ros2 launch golem_bringup g1_sim.launch.py use_nav:=true \
        frontier_explore:=true   # also serves /skill/frontier_explore

The G1 RoboCasa sim (golem_robocasa/g1/) takes rt/lowcmd and rt/dex3/*, and
holds the robot up on an elastic band until /elastic_band/toggle releases it.
golem_safety runs sim_split and the controller sim_split: golem_safety streams
zero gains on every motor from boot and engages each input on its own first
command. HOMIE's walking node always runs, with g1_homie's sim.yaml: once
golem_safety beats and the controller commands, it moves the legs to the
policy's default pose, starts the policy at zero velocity (no operator
confirm in the sim) and releases the band once the controller has finished its
start-up moves (or band.max_wait after the policy started without them). It
then walks on /cmd_vel. g1_core refuses the real robot's configs here (split,
g1_split, the walking node's g1); the sim configs keep the faster arm speeds
(1 rad/s, 0.5 m/s).

use_nav:=true adds g1_navigation on the sim's MID-360 (/livox/lidar,
/livox/imu), with sim time and g1_fastlio_sim.yaml, and Nav2 drives the
walking node on /cmd_vel; send goals to /navigate_to_pose.
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
            description='golem_safety config (default: sim_split)'),
        DeclareLaunchArgument(
            'use_nav', default_value='false',
            description='g1_navigation: odometry, SLAM and Nav2 on /cmd_vel'),
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
                # the one safety_config runs with: sim_split
                'controller_config': '',
                'use_nav': LaunchConfiguration('use_nav'),
                'odom_source': LaunchConfiguration('odom_source'),
                'frontier_explore': LaunchConfiguration('frontier_explore'),
            }.items(),
        ),
    ])
