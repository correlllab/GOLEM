"""
Start the H1-2's state for TF: robot_state_publisher and joint_state_publisher.

Both come from golem_upper_ik's robot_state.launch.py, at once: robot_state_publisher
on urdf_file, and golem_upper_ik's joint_state_publisher, which publishes the 27 body
joints from /lowstate.
"""
import os

from ament_index_python.packages import get_package_share_directory
from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Declare the state block's arguments and include golem_upper_ik's robot_state."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true: both nodes run on the simulator /clock'),
        DeclareLaunchArgument(
            'urdf_file',
            default_value=urdf_path('h12', ros=True),
            description="Absolute path of the URDF robot_state_publisher publishes "
                        "(default: golem_ros2_models' H1-2 ROS URDF)"),
        # The robot's TF from its measured joints.
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
