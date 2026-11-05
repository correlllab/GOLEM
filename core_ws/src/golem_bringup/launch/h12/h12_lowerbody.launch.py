"""
Start the H1-2's leg control: h12_lowerbody_rl's controller and, optionally, an estimator.

Both nodes start at once, and only with start_position_verified:=true:
lowerbody_controller_node runs the lowerbody:= policy (not with lowerbody:=mjpc), and
deploy_mjpc's RW-EKF base estimator runs with use_estimator:=true. Only one lower-body
controller may feed /safety/lowcmd_lower_in; no bringup starts the MJPC controller, so
lowerbody:=mjpc leaves the legs uncommanded.

SAFETY: start_position_verified is the operator's word that the robot is in a safe start
pose; without it nothing here starts, and safety_node keeps the legs on its zero-gain
stream. The controller then waits for safety_node's heartbeat, pre-poses the legs and,
on the real robot, holds the pre-pose until /lowerbody/confirm_engage.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import AndSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Declare the leg block's arguments and start its nodes behind the interlock."""
    config_dir = os.path.join(get_package_share_directory('golem_bringup'), 'config', 'h12')

    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true: both nodes run on the simulator /clock'),
        # SAFETY: the leg interlock; the sim bringups pass true.
        DeclareLaunchArgument(
            'start_position_verified', default_value='false',
            description='true only after checking the robot is in a safe start pose: '
                        'starts the leg controller (and the estimator with use_estimator)'),
        DeclareLaunchArgument(
            'lowerbody', default_value='almi',
            description="The controller's active_policy: almi, fame or walk; mjpc starts "
                        'no leg node'),
        DeclareLaunchArgument(
            'lowerbody_params',
            default_value=os.path.join(config_dir, 'lowerbody_real.yaml'),
            description='Parameter file of lowerbody_controller_node (path): '
                        'lowerbody_real.yaml or lowerbody_sim.yaml'),
        DeclareLaunchArgument(
            'use_estimator', default_value='false',
            description="true: start deploy_mjpc's RW-EKF base estimator"),
        DeclareLaunchArgument(
            'estimator_params',
            default_value=os.path.join(config_dir, 'mjpc_real.yaml'),
            description='Parameter file of the estimator (path)'),
    ]

    sim_time_param = {'use_sim_time': ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)}

    return LaunchDescription(args + [
        # The RL policy on the 12 leg joints, through safety_node's lower input.
        Node(
            package='h12_lowerbody_rl',
            executable='lowerbody_controller_node',
            name='lowerbody_controller_node',  # the parameter file's top-level key
            parameters=[sim_time_param,
                        LaunchConfiguration('lowerbody_params'),
                        {'active_policy': LaunchConfiguration('lowerbody')}],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(AndSubstitution(
                LaunchConfiguration('start_position_verified'),
                PythonExpression(
                    ["'", LaunchConfiguration('lowerbody'), "' != 'mjpc'"]),
            )),
        ),
        # The MJPC controller's base state, from leg odometry and the IMU.
        Node(
            package='deploy_mjpc',
            executable='estimator_node',
            name='deploy_mjpc_estimator',  # the parameter file's top-level key
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
