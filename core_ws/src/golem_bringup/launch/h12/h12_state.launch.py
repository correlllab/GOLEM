from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use the simulator /clock if true'
    )
    declare_urdf_file = DeclareLaunchArgument(
        'urdf_file',
        default_value=urdf_path('h12', ros=True),
        description='Absolute path to the robot URDF'
    )

    sim_time_param = {'use_sim_time': ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)}
    robot_description = ParameterValue(
        Command(['cat ', LaunchConfiguration('urdf_file')]), value_type=str)

    return LaunchDescription([
        declare_use_sim_time,
        declare_urdf_file,
        Node(
            package='golem_upper_ik',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[{'robot': 'h12'}, sim_time_param],
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}, sim_time_param],
            output='screen',
            emulate_tty=True,
        ),
    ])
