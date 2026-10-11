import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# Default URDF: the real robot's onboard PC checkout, or GOLEM_ASSETS_DIR when
# set. Inside the containers the assets are at /home/code/CL_Assets, so the sim
# bringup passes urdf_file explicitly.
ASSETS_DIR = os.environ.get('GOLEM_ASSETS_DIR', '/home/unitree/GOLEM/CL_Assets')


def generate_launch_description():
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use the simulator /clock if true'
    )
    declare_urdf_file = DeclareLaunchArgument(
        'urdf_file',
        default_value=os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'),
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
            package='h12_ros2_controller',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[sim_time_param],
            output='screen',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}, sim_time_param],
            output='screen',
        ),
    ])
