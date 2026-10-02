from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# Upper-body command path: [estop] -> safety_node -> frame_task_server.
#
# IMPORTANT: ROS_DOMAIN_ID must be exported in the launching shell. The safety
# layer uses unitree_sdk2py's DDS ChannelSubscriber (which honours
# $ROS_DOMAIN_ID and falls back to the YAML's network.domain_id only if the env
# is unset), while the rclpy nodes pick up the env directly. If the variable is
# missing, the DDS half and the rclpy half can end up on different domains and
# the safety layer will see no commands.


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use the simulator /clock if true'),
        DeclareLaunchArgument(
            'use_estop', default_value='false',
            description='Start the serial e-stop bridge (real robot only)'),
        DeclareLaunchArgument(
            'use_frame_task', default_value='true',
            description='Start the frame_task_server arm IK action server'),
        DeclareLaunchArgument(
            'safety_config', default_value='relax_safety_split.yaml',
            description='h12_safety_layer config name or path (resolved against its config/ dir)'),
        DeclareLaunchArgument(
            'frame_task_config', default_value='safety_split.yaml',
            description='h12_ros2_controller config for frame_task_server'),
        # The safety layer latches the estop status it reads at startup, so on
        # the real robot it waits for the estop node to open its serial port and
        # publish, and frame_task_server waits for the safety layer.
        DeclareLaunchArgument(
            'safety_delay', default_value='3.0',
            description='Seconds (wall clock) before starting safety_node'),
        DeclareLaunchArgument(
            'frame_task_delay', default_value='5.0',
            description='Seconds (wall clock) before starting frame_task_server'),
    ]

    sim_time_param = {'use_sim_time': ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)}

    return LaunchDescription(args + [
        Node(
            package='estop',
            executable='estop_node',
            name='estop_node',
            parameters=[sim_time_param],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_estop')),
        ),
        TimerAction(
            period=LaunchConfiguration('safety_delay'),
            actions=[
                Node(
                    package='h12_safety_layer',
                    executable='safety_node',
                    name='safety_node',
                    parameters=[sim_time_param],
                    arguments=['--config', LaunchConfiguration('safety_config')],
                    output='screen',
                ),
            ],
        ),
        TimerAction(
            period=LaunchConfiguration('frame_task_delay'),
            actions=[
                Node(
                    package='h12_ros2_controller',
                    executable='frame_task_server',
                    name='frame_task_server',
                    arguments=['--config', LaunchConfiguration('frame_task_config')],
                    parameters=[sim_time_param],
                    output='screen',
                    condition=IfCondition(LaunchConfiguration('use_frame_task')),
                ),
            ],
        ),
    ])
