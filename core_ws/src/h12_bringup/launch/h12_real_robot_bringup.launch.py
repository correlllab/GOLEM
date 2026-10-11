import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


# Real robot, onboard PC. ROS_DOMAIN_ID must be exported in the launching shell
# (0 on the real robot; see h12_control.launch.py).


def _include(package, launch_file, launch_arguments):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
    )


def generate_launch_description():
    # Real robot runs on wall-clock time (no simulator /clock).
    wall_time = {'use_sim_time': 'false'}

    return LaunchDescription([
        _include('h12_bringup', 'h12_real_drivers.launch.py', wall_time),
        _include('h12_slam', 'h12_navigation.launch.py', wall_time),
        _include('h12_bringup', 'h12_state.launch.py', wall_time),
        _include('h12_bringup', 'h12_control.launch.py', {**wall_time, 'use_estop': 'true'}),
    ])
