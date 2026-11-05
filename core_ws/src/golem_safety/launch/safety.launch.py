'''
safety_node, after start_delay seconds (wall clock). Both robots' bringups include this file
(golem_bringup's h12_control.launch.py and g1_core.launch.py); on its own:

    ros2 launch golem_safety safety.launch.py robot:=h12 config:=sim_safety_split sim:=true
'''

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import evaluate_condition_expression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _start(context: LaunchContext) -> list:
    '''
    The node, with every argument read now: a TimerAction's actions are evaluated when it
    fires, and by then a later include of the bringup may have set robot, config or sim
    for itself (Humble includes are not scoped)
    '''
    def text(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    def flag(name: str) -> bool:
        return evaluate_condition_expression(context, [LaunchConfiguration(name)])

    return [TimerAction(period=float(text('start_delay')), actions=[
        Node(
            package='golem_safety',
            executable='safety_node',
            name='safety_node',
            output='screen',
            emulate_tty=True,
            # typed strings: launch_ros reads a plain string parameter as YAML, which would
            # make a config named 'on' a bool
            parameters=[{
                'robot': ParameterValue(text('robot'), value_type=str),
                'config': ParameterValue(text('config'), value_type=str),
                'sim': flag('sim'),
                'use_estop': flag('use_estop'),
                'use_sim_time': flag('use_sim_time'),
            }],
        ),
    ])]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('robot', description='a robot with a profile in share/golem_safety/robots/'),
        DeclareLaunchArgument(
            'config',
            description='a config name under share/golem_safety/config/<robot>/, or a path'),
        DeclareLaunchArgument(
            'sim', default_value='false',
            description='true for a simulator; refused on ROS domain 0'),
        DeclareLaunchArgument(
            'use_estop', default_value='false',
            description='true to read the hardware estop on the config\'s estop topics'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true to run on a simulator\'s /clock'),
        DeclareLaunchArgument(
            'start_delay', default_value='0.0',
            description='s (wall clock) before the node starts'),
        OpaqueFunction(function=_start),
    ])
