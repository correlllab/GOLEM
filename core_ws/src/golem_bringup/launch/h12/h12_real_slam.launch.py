"""
Start the real H1-2's sensors and SLAM/navigation only, on the wall clock.

    ros2 launch golem_bringup h12_real_slam.launch.py

The includes start at once: the sensor drivers and grippers (h12_real_drivers), the
state publishers (h12_state) and golem_slam's navigation. No estop, safety layer or IK
starts, so no joint is commanded. Export the robot's ROS_DOMAIN_ID (0) in the launching
shell. Record runs with ros2 run golem_slam record_slam_bag --name <name>.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def _include(package, launch_file, launch_arguments):
    """Include a package's launch file with the given arguments."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
    )


def generate_launch_description():
    """Include the drivers, the state publishers and navigation."""
    # The real robot runs on the wall clock: there is no /clock.
    wall_time = {'use_sim_time': 'false'}

    return LaunchDescription([
        # The MID-360, the head RealSense and the grippers.
        _include('golem_bringup', 'h12/h12_real_drivers.launch.py', wall_time),
        # TF of the body from /lowstate, which golem_slam's body_to_pelvis needs.
        _include('golem_bringup', 'h12/h12_state.launch.py', wall_time),
        # FAST-LIO, OctoMap, slam_toolbox and Nav2.
        _include('golem_slam', 'navigation.launch.py', {**wall_time, 'robot': 'h12'}),
    ])
