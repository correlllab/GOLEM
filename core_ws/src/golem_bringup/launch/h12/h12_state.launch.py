import os

from ament_index_python.packages import get_package_share_directory
from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


# The H1-2's state for TF: golem_upper_ik's robot_state.launch.py, which starts
# robot_state_publisher on urdf_file and joint_state_publisher (the 27 body
# joints from /lowstate).


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use the simulator /clock if true'),
        DeclareLaunchArgument(
            'urdf_file',
            default_value=urdf_path('h12', ros=True),
            description='Absolute path to the robot URDF'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('golem_upper_ik'), 'launch',
                'robot_state.launch.py')),
            launch_arguments={
                'robot': 'h12',
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'urdf_file': LaunchConfiguration('urdf_file'),
            }.items(),
        ),
    ])
