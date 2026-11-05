'''
The robot's state for TF: robot_state_publisher on the robot's URDF and golem_upper_ik's
joint_state_publisher, which publishes the body joints from /lowstate on /joint_states. Both
robots' bringups include this file (golem_bringup's h12_state.launch.py and
g1_core.launch.py); on its own:

    ros2 launch golem_upper_ik robot_state.launch.py robot:=g1 use_sim_time:=true
'''

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import evaluate_condition_expression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from golem_ros2_models.paths import urdf_path


def _start(context: LaunchContext) -> list:
    robot = LaunchConfiguration('robot').perform(context)
    urdf_file = LaunchConfiguration('urdf_file').perform(context) or urdf_path(robot, ros=True)
    with open(urdf_file, encoding='utf-8') as handle:
        robot_description = handle.read()
    use_sim_time = evaluate_condition_expression(context, [LaunchConfiguration('use_sim_time')])
    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'robot_description': ParameterValue(robot_description, value_type=str),
                'use_sim_time': use_sim_time,
            }],
        ),
        Node(
            package='golem_upper_ik',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'robot': ParameterValue(robot, value_type=str),
                'use_sim_time': use_sim_time,
            }],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('robot', description='h12 or g1'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true to run on a simulator\'s /clock'),
        DeclareLaunchArgument(
            'urdf_file', default_value='',
            description="the URDF robot_state_publisher publishes; empty for the robot's ROS "
                        'URDF from golem_ros2_models'),
        OpaqueFunction(function=_start),
    ])
