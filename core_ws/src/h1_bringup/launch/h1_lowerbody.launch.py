import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import AndSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# Lower-body leg control: the h12_lowerbody_rl controller and, optionally, the
# h12_deploy_mjpc RW-EKF base estimator. Only one lower-body controller may feed
# /safety/lowcmd_lower_in; this bringup does not launch the MJPC controller, so
# lowerbody:=mjpc leaves the legs uncontrolled.


def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('h1_bringup'), 'config')

    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use the simulator /clock if true'),
        # Interlock: nothing here commands the legs until the operator has
        # verified the robot is in a safe start position. The sim passes true.
        DeclareLaunchArgument(
            'start_position_verified', default_value='false',
            description='Start the leg controller and estimator'),
        DeclareLaunchArgument(
            'lowerbody', default_value='almi',
            description='RL policy: almi | almi27 | fame | walk; mjpc starts no RL node'),
        # false keeps the robot on the lowerbody:= policy for standing AND
        # walking; true hands nonzero /cmd_vel to the walk policy and back.
        DeclareLaunchArgument('lowerbody_auto_switch', default_value='false'),
        DeclareLaunchArgument(
            'lowerbody_params',
            default_value=os.path.join(config_dir, 'lowerbody_real.yaml'),
            description='Parameter file for lowerbody_controller_node'),
        DeclareLaunchArgument(
            'use_estimator', default_value='false',
            description='Start the MJPC RW-EKF base estimator'),
        DeclareLaunchArgument(
            'estimator_params',
            default_value=os.path.join(config_dir, 'mjpc_real.yaml'),
            description='Parameter file for the MJPC estimator'),
    ]

    sim_time_param = {'use_sim_time': ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)}

    return LaunchDescription(args + [
        Node(
            package='h12_lowerbody_rl',
            executable='lowerbody_controller_node',
            name='lowerbody_controller_node',  # MUST match the yaml key
            parameters=[sim_time_param,
                        LaunchConfiguration('lowerbody_params'),
                        {'active_policy': LaunchConfiguration('lowerbody'),
                         'auto_switch': ParameterValue(
                             LaunchConfiguration('lowerbody_auto_switch'), value_type=bool)}],
            output='screen',
            condition=IfCondition(AndSubstitution(
                LaunchConfiguration('start_position_verified'),
                PythonExpression(
                    ["'", LaunchConfiguration('lowerbody'), "' != 'mjpc'"]),
            )),
        ),
        Node(
            package='h12_deploy_mjpc',
            executable='estimator_node',
            name='h12_deploy_mjpc_estimator',  # MUST match the yaml key
            parameters=[sim_time_param, LaunchConfiguration('estimator_params')],
            # 'log' keeps the node's periodic [est] stdout off the console; it
            # still goes to the ROS log files.
            output='log',
            condition=IfCondition(AndSubstitution(
                LaunchConfiguration('start_position_verified'),
                LaunchConfiguration('use_estimator'),
            )),
        ),
    ])
