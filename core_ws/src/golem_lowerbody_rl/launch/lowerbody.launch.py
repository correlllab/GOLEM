"""
Start the leg node, lowerbody_controller_node, once its robot, run config and policy pass.

In order:
1. The refusal (golem_lowerbody_rl.launch_check.lowerbody_refusal): the robot must be known,
   the run config's sim must equal sim, a real-robot run config must wait for the
   operator's confirm and release no band, and the policy must be one the robot lists. A
   refusal raises "lowerbody: not starting: <reason>" before any node starts.
2. lowerbody_controller_node, start_delay seconds (wall clock) later.

Both robots' bringups include this file (golem_bringup's h12_lowerbody.launch.py and
g1_core.launch.py); a bringup that includes it after other nodes runs the same check
(lowerbody_check) first. On its own:

    ros2 launch golem_lowerbody_rl lowerbody.launch.py robot:=h12 config:=sim sim:=true \
        use_sim_time:=true
"""

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import evaluate_condition_expression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from golem_lowerbody_rl.launch_check import lowerbody_check
from golem_lowerbody_rl.profile import known_robots


def _text(context: LaunchContext, name: str) -> str:
    """Return launch argument name as text."""
    return LaunchConfiguration(name).perform(context)


def _flag(context: LaunchContext, name: str) -> bool:
    """Return launch argument name as a bool (true/false, 1/0, ...)."""
    return evaluate_condition_expression(context, [LaunchConfiguration(name)])


def _start(context: LaunchContext) -> list:
    """
    Return the leg node behind a start_delay timer, every argument read now.

    A TimerAction's actions are evaluated when it fires, and by then a later include of
    the bringup may have set robot, config or policy for itself (Humble includes are not
    scoped).
    """
    return [TimerAction(period=float(_text(context, 'start_delay')), actions=[
        # the legs' policy, commanding golem_safety's lower-body input
        Node(
            package='golem_lowerbody_rl',
            executable='lowerbody_controller_node',
            name='lowerbody_controller_node',
            output='screen',
            emulate_tty=True,
            # typed strings: launch_ros reads a plain string parameter as YAML, which would
            # make an empty policy None
            parameters=[{
                'robot': ParameterValue(_text(context, 'robot'), value_type=str),
                'config': ParameterValue(_text(context, 'config'), value_type=str),
                'policy': ParameterValue(_text(context, 'policy'), value_type=str),
                'use_sim_time': _flag(context, 'use_sim_time'),
            }],
        ),
    ])]


def generate_launch_description() -> LaunchDescription:
    """Declare the arguments, then check them and start the leg node."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot',
            description='required: the robot whose legs run, one of '
                        + ', '.join(known_robots())),
        DeclareLaunchArgument(
            'config',
            description='required: the run config, sim or real '
                        '(share/golem_lowerbody_rl/config/<robot>/<name>.yaml) or the path of '
                        'a .yaml file; its sim must equal sim'),
        DeclareLaunchArgument(
            'policy', default_value='',
            description="the policy, one of the robot's policies (config/<robot>/robot.yaml) "
                        "by name or the path of its spec; empty: the robot's default_policy"),
        DeclareLaunchArgument(
            'sim',
            description='required, true|false: true for a simulator; a run config for the '
                        'other platform is refused'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description="true|false: true to run the node's clock on a simulator's /clock"),
        DeclareLaunchArgument(
            'start_delay', default_value='0.0',
            description='s of wall time (>= 0) from the check to starting the node'),
        # the check runs first, so a refusal starts nothing
        lowerbody_check(LaunchConfiguration('robot'), LaunchConfiguration('config'),
                        LaunchConfiguration('policy'), LaunchConfiguration('sim')),
        OpaqueFunction(function=_start),
    ])
