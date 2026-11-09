"""
Start the H1-2's state for TF: robot_state_publisher and joint_state_publisher.

Both come from golem_upper_ik's robot_state.launch.py, at once: robot_state_publisher
on urdf_file, and golem_upper_ik's joint_state_publisher, which publishes the 27 body
joints from /lowstate.
"""
from golem_bringup.include import include
from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
        include(
            'golem_upper_ik', 'robot_state.launch.py',
            {
                'robot': 'h12',
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'urdf_file': LaunchConfiguration('urdf_file'),
            },
        ),
    ])
