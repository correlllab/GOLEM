"""
Start the H1-2's leg control: golem_lowerbody_rl's leg node and, optionally, an estimator.

Both start at once, and only with start_position_verified:=true:
golem_lowerbody_rl's lowerbody.launch.py (robot h12) runs the policy:= policy on the
lowerbody_config run config (not with policy:=mjpc), and deploy_mjpc's RW-EKF base
estimator runs with use_estimator:=true. Only one lower-body controller may feed
/safety/lowcmd_lower_in; no bringup starts the MJPC controller, so policy:=mjpc leaves the
legs uncommanded.

lowerbody.launch.py refuses, before any node starts, a run config whose sim differs from
sim, a real-robot run config without the operator's confirm or with the elastic band's
release, and a policy the H1-2 does not list (golem_lowerbody_rl's config/h12/robot.yaml).

SAFETY: start_position_verified is the operator's word that the robot is in a safe start
pose; without it nothing here starts, and safety_node keeps the legs on its zero-gain
stream. The leg node then waits for safety_node's heartbeat, pre-poses the legs and, on the
real robot, holds the pre-pose until /lowerbody/confirm_engage.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import AndSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Declare the leg block's arguments and start its nodes behind the interlock."""
    config_dir = os.path.join(get_package_share_directory('golem_bringup'), 'config', 'h12')
    lowerbody_launch = os.path.join(get_package_share_directory('golem_lowerbody_rl'),
                                    'launch', 'lowerbody.launch.py')

    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true: both nodes run on the simulator /clock'),
        # SAFETY: the leg interlock; the sim bringups pass true.
        DeclareLaunchArgument(
            'start_position_verified', default_value='false',
            description='true only after checking the robot is in a safe start pose: '
                        'starts the leg node (and the estimator with use_estimator)'),
        DeclareLaunchArgument(
            'policy', default_value='almi',
            description="The leg node's policy: almi, fame or walk (golem_lowerbody_rl's "
                        'config/h12/robot.yaml); mjpc starts no leg node'),
        DeclareLaunchArgument(
            'sim',
            description='Required, true|false: true for a simulator; the leg node refuses a '
                        'run config for the other platform'),
        DeclareLaunchArgument(
            'lowerbody_config',
            description="Required: the leg node's run config, sim or real "
                        "(golem_lowerbody_rl's config/h12/<name>.yaml) or a path"),
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
        # The policy on the 12 leg joints, through safety_node's lower input. Every
        # argument is passed: Humble includes are not scoped, so an earlier include's
        # config, sim or start_delay would otherwise stand in for one left out.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(lowerbody_launch),
            launch_arguments={
                'robot': 'h12',
                'config': LaunchConfiguration('lowerbody_config'),
                'policy': LaunchConfiguration('policy'),
                'sim': LaunchConfiguration('sim'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'start_delay': '0.0',
            }.items(),
            condition=IfCondition(AndSubstitution(
                LaunchConfiguration('start_position_verified'),
                PythonExpression(["'", LaunchConfiguration('policy'), "' != 'mjpc'"]),
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
