r'''
The arm controller, frame_task_server, after start_delay seconds (wall clock), once its
config is checked against the safety config golem_safety runs (golem_upper_ik.pairing).
Both robots' bringups include this file first (golem_bringup's h12_control.launch.py and
g1_core.launch.py), so a refused pairing stops the launch before any of their nodes starts.
On its own:

    ros2 launch golem_upper_ik upper_ik.launch.py robot:=h12 config:=sim_safety_split \
        safety_config:=sim_safety_split sim:=true use_sim_time:=true
'''

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import evaluate_condition_expression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from golem_upper_ik.pairing import pairing_refusal


def _text(context: LaunchContext, name: str) -> str:
    return LaunchConfiguration(name).perform(context)


def _flag(context: LaunchContext, name: str) -> bool:
    return evaluate_condition_expression(context, [LaunchConfiguration(name)])


def _check_pairing(context: LaunchContext) -> list:
    '''Raises RuntimeError, starting nothing, when config does not run with safety_config'''
    refusal = pairing_refusal(_text(context, 'robot'), _text(context, 'config'),
                              _text(context, 'safety_config'), sim=_flag(context, 'sim'))
    if refusal is not None:
        raise RuntimeError(f'upper_ik: not starting: {refusal}')
    return []


def _start(context: LaunchContext) -> list:
    '''
    The controller, with every argument read now: a TimerAction's actions are evaluated when
    it fires, and by then a later include of the bringup may have set robot or config for
    itself (Humble includes are not scoped)
    '''
    return [TimerAction(period=float(_text(context, 'start_delay')), actions=[
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
    return LaunchDescription([
        DeclareLaunchArgument('robot', description='h12 or g1'),
        DeclareLaunchArgument(
            'config',
            description='a controller config name under share/golem_upper_ik/config/<robot>/, '
                        'or a path'),
        DeclareLaunchArgument(
            'safety_config',
            description="the golem_safety config safety_node runs: config must be the one "
                        "paired with it in config/<robot>/pairing.yaml"),
        DeclareLaunchArgument(
            'sim',
            description='true for a simulator, false for the real robot: neither config may '
                        'be meant for the other'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true to stamp and time the published poses with a simulator\'s /clock'),
        DeclareLaunchArgument(
            'start_delay', default_value='0.0',
            description='s (wall clock) before the controller starts'),
        OpaqueFunction(function=_check_pairing),
        OpaqueFunction(function=_start),
    ])
