r"""Start the arm controller, frame_task_server, once its config pairs with golem_safety's.

In order:
1. The pairing check (golem_upper_ik.pairing): config must be the controller config paired
   with safety_config in config/<robot>/pairing.yaml, and with sim neither may be meant for
   the other platform. A refused pairing raises before any node starts.
2. frame_task_server, start_delay seconds (wall clock) later.

Both robots' bringups include this file first (golem_bringup's h12_control.launch.py and
g1_core.launch.py), so a refused pairing stops the launch before any of their nodes starts.
On its own:

    ros2 launch golem_upper_ik upper_ik.launch.py robot:=h12 config:=sim_safety_split \
        safety_config:=sim_safety_split sim:=true use_sim_time:=true
"""

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import evaluate_condition_expression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from golem_upper_ik.pairing import pairing_refusal
from golem_upper_ik.profile import known_robots


def _text(context: LaunchContext, name: str) -> str:
    """Return launch argument name as text."""
    return LaunchConfiguration(name).perform(context)


def _flag(context: LaunchContext, name: str) -> bool:
    """Return launch argument name as a bool (true/false, 1/0, ...)."""
    return evaluate_condition_expression(context, [LaunchConfiguration(name)])


def _check_pairing(context: LaunchContext) -> list:
    """Raise RuntimeError, starting nothing, when config does not run with safety_config."""
    refusal = pairing_refusal(_text(context, 'robot'), _text(context, 'config'),
                              _text(context, 'safety_config'), sim=_flag(context, 'sim'))
    if refusal is not None:
        raise RuntimeError(f'upper_ik: not starting: {refusal}')
    return []


def _start(context: LaunchContext) -> list:
    """Return frame_task_server behind a start_delay timer, every argument read now.

    A TimerAction's actions are evaluated when it fires, and by then a later include of the
    bringup may have set robot or config for itself (Humble includes are not scoped).
    """
    return [TimerAction(period=float(_text(context, 'start_delay')), actions=[
        # the controller: /frame_task and /named_config, commanding golem_safety's input
        Node(
            package='golem_upper_ik',
            executable='frame_task_server',
            name='frame_task_server',
            output='screen',
            emulate_tty=True,
            # typed strings: launch_ros reads a plain string parameter as YAML, which would
            # make a config named 'on' a bool
            parameters=[{
                'robot': ParameterValue(_text(context, 'robot'), value_type=str),
                'config': ParameterValue(_text(context, 'config'), value_type=str),
                'use_sim_time': _flag(context, 'use_sim_time'),
            }],
        ),
    ])]


def generate_launch_description() -> LaunchDescription:
    """Declare the arguments, then check the pairing and start the controller."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot',
            description='the robot whose controller runs: ' + ' or '.join(known_robots())),
        DeclareLaunchArgument(
            'config',
            description='the controller config: a name under '
                        'share/golem_upper_ik/config/<robot>/ (with or without .yaml), or a '
                        'path; it must be the one paired with safety_config'),
        DeclareLaunchArgument(
            'safety_config',
            description='the golem_safety config safety_node runs, by name or path: config '
                        'must be the one paired with it in config/<robot>/pairing.yaml'),
        DeclareLaunchArgument(
            'sim',
            description='true for a simulator, false for the real robot: a sim config '
                        '(named sim*) is refused on the real robot and a real one in the sim'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description="true to stamp and time the published poses with a simulator's "
                        '/clock; false for wall time'),
        DeclareLaunchArgument(
            'start_delay', default_value='0.0',
            description='seconds (wall clock, >= 0) from the pairing check to starting the '
                        'controller'),
        # the pairing check runs first, so a refusal starts nothing
        OpaqueFunction(function=_check_pairing),
        OpaqueFunction(function=_start),
    ])
