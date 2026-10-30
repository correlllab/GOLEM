'''
frame_task_server and joint_state_publisher on their own, for manual runs (the bringups start
them themselves):

    ros2 launch golem_upper_ik upper_ik.launch.py robot:=h12 config:=sim_safety_split
'''

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    def value(name: str, value_type: type) -> ParameterValue:
        return ParameterValue(LaunchConfiguration(name), value_type=value_type)

    parameters = [{
        'robot': value('robot', str),
        'config': value('config', str),
        'use_sim_time': value('use_sim_time', bool),
    }]
    return LaunchDescription([
        DeclareLaunchArgument('robot', description='h12 or g1'),
        DeclareLaunchArgument(
            'config', default_value='',
            description='a controller config name under share/golem_upper_ik/config/<robot>/, '
                        "or a path; empty for the robot's default"),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true to stamp and time the published poses with a simulator\'s /clock'),
        Node(package='golem_upper_ik', executable='frame_task_server',
             name='frame_task_server', output='screen', parameters=parameters),
        Node(package='golem_upper_ik', executable='joint_state_publisher',
             name='joint_state_publisher', output='screen', parameters=parameters),
    ])
