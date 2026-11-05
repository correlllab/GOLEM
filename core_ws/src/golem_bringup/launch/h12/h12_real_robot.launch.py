"""
Start the whole real H1-2 on its onboard PC, on the wall clock.

    ros2 launch golem_bringup h12_real_robot.launch.py [start_position_verified:=true]

Every node joins the launching shell's ROS_DOMAIN_ID, which must be 0, the real robot's
command bus; ROS uses 0 when it is unset. The includes start at once: the sensor drivers
and grippers (h12_real_drivers), navigation, the state publishers (h12_state), the
command path (h12_control with the hardware estop: estop_node, then safety_node and
frame_task_server after h12_control's default delays) and the leg controller
(h12_lowerbody, no estimator).

SAFETY: the legs are behind two operator gates:
  1. start_position_verified:=true starts the lowerbody:= policy, which ramps the legs
     to the policy's pre-pose crouch and holds there;
  2. ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger engages it.

Before the first real run, besides measuring the leg node's worst /lowstate gap
(h12_lowerbody_rl's README), replay at least 3 recorded real-robot bags
(record_slam_bag recordings) through navigation.launch.py robot:=h12 and check
FAST-LIO, level_odom and the map. Their points were recorded with the driver's roll
180 and the IMU upright, so rotate both back to the raw frame (180 degrees about x)
first. Keep the MID-360 unsynced (no PTP: the driver then stamps with the host clock,
so lidar and joint-state stamps share one clock), or check that level_odom logs no
"waiting for body->pelvis".
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _include(package, launch_file, launch_arguments):
    """Include a package's launch file with the given arguments."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
    )


def generate_launch_description():
    """Declare the real bringup's arguments and include its blocks."""
    # The real robot runs on the wall clock: there is no /clock.
    wall_time = {'use_sim_time': 'false'}

    bringup_share = get_package_share_directory('golem_bringup')

    return LaunchDescription([
        # SAFETY: false by default, so a bare launch never commands the legs.
        DeclareLaunchArgument('start_position_verified', default_value='false',
                              description='true only after checking the robot is in a '
                                          'safe start pose: starts the leg controller'),
        DeclareLaunchArgument('lowerbody', default_value='almi',
                              description='The policy h12_lowerbody_rl runs on the legs: '
                                          'almi, fame or walk; mjpc starts no leg node'),

        # The MID-360, the head RealSense and the grippers.
        _include('golem_bringup', 'h12/h12_real_drivers.launch.py', wall_time),
        # SLAM and Nav2, driving the legs on /cmd_vel.
        _include('golem_slam', 'navigation.launch.py', {**wall_time, 'robot': 'h12'}),
        # TF of the body from /lowstate.
        _include('golem_bringup', 'h12/h12_state.launch.py', wall_time),
        # The command path, with the hardware estop.
        _include('golem_bringup', 'h12/h12_control.launch.py', {
            **wall_time,
            'sim': 'false',
            'use_estop': 'true',
        }),
        # The legs, behind the operator gates.
        _include('golem_bringup', 'h12/h12_lowerbody.launch.py', {
            **wall_time,
            'start_position_verified': LaunchConfiguration('start_position_verified'),
            'lowerbody': LaunchConfiguration('lowerbody'),
            'lowerbody_params': os.path.join(bringup_share, 'config', 'h12',
                                             'lowerbody_real.yaml'),
            'use_estimator': 'false',
        }),
    ])
