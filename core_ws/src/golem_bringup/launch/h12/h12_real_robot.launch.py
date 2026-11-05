import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


# Real robot, onboard PC. Every node joins the launching shell's ROS_DOMAIN_ID,
# which must be 0, the real robot's command bus; ROS uses 0 when it is unset
# (see h12_control.launch.py).
#
# Leg control (h12_lowerbody.launch.py) runs here, behind two operator gates:
#   1. start_position_verified:=true starts the lowerbody:= RL policy, which
#      ramps the legs to the policy's pre-pose crouch and holds there.
#   2. ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger
#      engages the policy.
#
# Before the first real run, besides measuring the leg node's worst /lowstate
# gap (h12_lowerbody_rl's README), replay at least 3 recorded real-robot bags
# (record_slam_bag recordings) through navigation.launch.py robot:=h12 and
# check FAST-LIO, level_odom and the map. Their points were recorded with the
# driver's roll 180 and the IMU upright, so rotate both back to the raw frame
# (180 degrees about x) first. Keep the MID-360 unsynced (no PTP: the driver
# then stamps with the host clock, so lidar and joint-state stamps share one
# clock), or check that level_odom logs no "waiting for body->pelvis".


def _include(package, launch_file, launch_arguments):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
    )


def generate_launch_description():
    # Real robot runs on wall-clock time (no simulator /clock).
    wall_time = {'use_sim_time': 'false'}

    bringup_share = get_package_share_directory('golem_bringup')

    return LaunchDescription([
        # Defaults to false so a bare launch never commands the legs.
        DeclareLaunchArgument('start_position_verified', default_value='false'),
        DeclareLaunchArgument('lowerbody', default_value='almi'),

        _include('golem_bringup', 'h12/h12_real_drivers.launch.py', wall_time),
        _include('golem_slam', 'navigation.launch.py', {**wall_time, 'robot': 'h12'}),
        _include('golem_bringup', 'h12/h12_state.launch.py', wall_time),
        _include('golem_bringup', 'h12/h12_control.launch.py', {
            **wall_time,
            'sim': 'false',
            'use_estop': 'true',
        }),
        _include('golem_bringup', 'h12/h12_lowerbody.launch.py', {
            **wall_time,
            'start_position_verified': LaunchConfiguration('start_position_verified'),
            'lowerbody': LaunchConfiguration('lowerbody'),
            'lowerbody_params': os.path.join(bringup_share, 'config', 'h12',
                                             'lowerbody_real.yaml'),
            'use_estimator': 'false',
        }),
    ])
