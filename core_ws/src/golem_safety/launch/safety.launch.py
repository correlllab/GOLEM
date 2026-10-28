'''
safety_node on its own, for manual runs (the bringups start it themselves):

    ros2 launch golem_safety safety.launch.py robot:=h12 config:=sim_safety_split sim:=true
'''

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    def value(name: str, value_type: type) -> ParameterValue:
        return ParameterValue(LaunchConfiguration(name), value_type=value_type)

    return LaunchDescription([
        DeclareLaunchArgument('robot', description='h12 or g1'),
        DeclareLaunchArgument(
            'config',
            description='a config name under share/golem_safety/config/<robot>/, or a path'),
        DeclareLaunchArgument(
            'sim', default_value='false',
            description='true for a simulator; refused on ROS domain 0'),
        DeclareLaunchArgument(
            'use_estop', default_value='false',
            description='true to read the hardware estop on the config\'s estop topics'),
        Node(
            package='golem_safety',
            executable='safety_node',
            name='safety_node',
            output='screen',
            parameters=[{
                'robot': value('robot', str),
                'config': value('config', str),
                'sim': value('sim', bool),
                'use_estop': value('use_estop', bool),
            }],
        ),
    ])
