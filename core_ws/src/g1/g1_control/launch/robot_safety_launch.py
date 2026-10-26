from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from g1_control.utility.config_pairing import pairing_refusal
from g1_control.utility.path_definition_ros import URDF_ROS_PATH


def _validate_config_compatibility(context):
    # the same pairing rules as g1_bringup's g1_core.launch.py
    refusal = pairing_refusal(
        LaunchConfiguration('config').perform(context),
        LaunchConfiguration('safety_config').perform(context),
    )
    if refusal is not None:
        raise ValueError(refusal)
    return []


def generate_launch_description():
    package_name = 'g1_control'
    config = LaunchConfiguration('config')
    safety_config = LaunchConfiguration('safety_config')

    with open(URDF_ROS_PATH, 'r') as urdf_file:
        robot_description = urdf_file.read()

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'config',
                default_value='g1.yaml',
                description='Controller config file name under config/ '
                            '(g1.yaml, g1_split.yaml, sim.yaml or sim_split.yaml)',
            ),
            DeclareLaunchArgument(
                'safety_config',
                default_value='arm_sdk',
                description='g1_safety_layer config: arm_sdk, split, sim_arm_sdk or sim_split',
            ),
            OpaqueFunction(function=_validate_config_compatibility),
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                parameters=[{'robot_description': robot_description}],
                output='screen',
            ),
            Node(
                package=package_name,
                executable='joint_state_publisher',
                name='joint_state_publisher',
                output='screen',
            ),
            TimerAction(
                period=5.0,
                actions=[
                    Node(
                        package='g1_safety_layer',
                        executable='safety_node',
                        name='safety_node',
                        arguments=['--config', safety_config],
                        output='screen',
                    ),
                ],
            ),
            # TimerAction(
            #     period=2.0,
            #     actions=[
            #         Node(
            #             package=package_name,
            #             executable='dual_arm_server',
            #             name='dual_arm_server',
            #             arguments=['--config', config],
            #             output='screen'
            #         ),
            #     ]
            # ),
            TimerAction(
                period=5.0,
                actions=[
                    Node(
                        package=package_name,
                        executable='frame_task_server',
                        name='frame_task_server',
                        arguments=['--config', config],
                        output='screen',
                    ),
                ],
            ),
        ]
    )
