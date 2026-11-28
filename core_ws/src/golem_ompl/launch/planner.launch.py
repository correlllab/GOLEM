r"""Start the arm planner, arm_planner_server, which serves /plan_arm_trajectory.

It starts at once: it builds the robot's models (seconds) and then serves the service, so a
bringup that includes it before the arm controller has it ready by the first planned goal.
golem_bringup's h12_control.launch.py (with use_frame_task) and g1_core.launch.py include it,
with the controller config frame_task_server runs. On its own:

    ros2 launch golem_ompl planner.launch.py robot:=h12 config:=safety_split use_sim_time:=true
"""

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import evaluate_condition_expression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from golem_upper_ik.profile import known_robots


def _start(context: LaunchContext) -> list:
    """Return arm_planner_server, every argument read now.

    Humble includes are not scoped: a later include of the bringup may set robot or config
    for itself, so the values are taken when this file is included.
    """
    def text(name):
        """Return launch argument name as text."""
        return LaunchConfiguration(name).perform(context)
    return [
        # the planner: /plan_arm_trajectory for the arm controller
        Node(
            package='golem_ompl',
            executable='arm_planner_server',
            name='arm_planner_server',
            output='screen',
            emulate_tty=True,
            # typed strings: launch_ros reads a plain string parameter as YAML, which would
            # make a config named 'on' a bool
            parameters=[{
                'robot': ParameterValue(text('robot'), value_type=str),
                'config': ParameterValue(text('config'), value_type=str),
                'use_sim_time': evaluate_condition_expression(
                    context, [LaunchConfiguration('use_sim_time')]),
            }],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """Declare the arguments, then start the planner."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot',
            description='the robot whose arms are planned: ' + ' or '.join(known_robots())),
        DeclareLaunchArgument(
            'config',
            description="the golem_upper_ik controller config whose planner settings, IK "
                        "d_min and timeout and clip range the planner runs on: a name under "
                        "share/golem_upper_ik/config/<robot>/ (with or without .yaml), or a "
                        "path; the arm controller's own"),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description="true to run on a simulator's /clock (the node's logs; its budgets "
                        "are steady time either way); false for wall time"),
        OpaqueFunction(function=_start),
    ])
