"""
Start golem_safety's safety_node, the only writer of /lowcmd, after start_delay s of wall
time.

It starts one node and nothing else, with no condition: robot and config are required, and
the node itself refuses a bad combination (an unknown robot, a config that does not load for
it, sim:=true on ROS domain 0) by exiting 1 before it publishes. Both robots' bringups
include this file (golem_bringup's h12_control.launch.py and g1_core.launch.py); on its own:

    ros2 launch golem_safety safety.launch.py robot:=h12 config:=sim_safety_split sim:=true
"""

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import evaluate_condition_expression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _start(context: LaunchContext) -> list:
    """
    Return the delayed node, with every argument read now.

    A TimerAction's actions are evaluated when it fires, and by then a later include of the
    bringup may have set robot, config or sim for itself (Humble includes are not scoped).
    """
    def text(name: str) -> str:
        """Return the launch argument `name` as text."""
        return LaunchConfiguration(name).perform(context)

    def flag(name: str) -> bool:
        """Return the launch argument `name` as a boolean."""
        return evaluate_condition_expression(context, [LaunchConfiguration(name)])

    # The bringups stagger their nodes with start_delay; the node streams zero-gain /lowcmd
    # from the moment it starts.
    return [TimerAction(period=float(text('start_delay')), actions=[
        Node(
            package='golem_safety',
            executable='safety_node',
            name='safety_node',
            output='screen',
            emulate_tty=True,
            # Typed strings: launch_ros reads a plain string parameter as YAML, which would
            # make a config named 'on' a bool.
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
    """Declare the arguments and start safety_node after start_delay."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot',
            description='required: the robot, one with a profile in '
                        'share/golem_safety/robots/ (the file name without .yaml)'),
        DeclareLaunchArgument(
            'config',
            description='required: the safety config, a name under '
                        'share/golem_safety/config/<robot>/ (with or without .yaml), or a path'),
        DeclareLaunchArgument(
            'sim', default_value='false',
            description='true|false: true for a simulator; the node refuses it on ROS '
                        'domain 0, the real robots\' command bus'),
        DeclareLaunchArgument(
            'use_estop', default_value='false',
            description='true|false: true to require the hardware estop on the config\'s '
                        'estop topics; no input engages before its first sample, and a '
                        'triggered, unplugged or silent estop latches'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true|false: true to run ROS time on a simulator\'s /clock; the '
                        'relay and its timeouts run on the boot clock either way'),
        DeclareLaunchArgument(
            'start_delay', default_value='0.0',
            description='s of wall time before the node starts, >= 0'),
        OpaqueFunction(function=_start),
    ])
