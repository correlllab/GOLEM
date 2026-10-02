import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


# Real robot SLAM + navigation only: sensor drivers, state publishers and the
# h12_slam nav stack; no estop, safety layer or IK. ROS_DOMAIN_ID must be
# exported in the launching shell (0 on the real robot).


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
        _include('h1_bringup', 'h1_real_drivers.launch.py', wall_time),
        _include('h1_bringup', 'h1_state.launch.py', wall_time),
        _include('h12_slam', 'h1_navigation.launch.py', wall_time),
    ])
