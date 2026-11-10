"""
Start lmg_deploy's marl_deploy_node: the loco-manipulation game's MARL policy.

The policy commands all 27 joints through safety_node's split inputs, so frame_task_server
and the lower-body RL node must not run beside it. lmg_sim.launch.py and
lmg_real.launch.py include this file and start the state publishers, the safety layer
and SLAM. The node runs lmg_deploy's policies/h12/, the H1-2's one policy. It starts only
with start_position_verified:=true; it ramps the joints to the policy's default pose,
holds it until ~/confirm_engage (with engage_wait_for_confirm) and holds its arm goals in
the world by SLAM's /Odometry.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Declare the policy's arguments and start marl_deploy_node behind the interlock."""
    deploy_share = get_package_share_directory('lmg_deploy')
    # The H1-2's one policy (lmg_deploy's policies/README.md: one iteration per robot).
    policy_config = os.path.join(deploy_share, 'policies', 'h12', 'h12.yaml')
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    policy_on = LaunchConfiguration('start_position_verified')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true: the node runs on the simulator /clock'),
        DeclareLaunchArgument(
            'urdf_path',
            description='Required. Pinocchio URDF for the wrist kinematics (path of '
                        "golem_ros2_models' h1_2_magpie.urdf)"),
        # SAFETY: the interlock; the policy node starts only once the operator has
        # checked the robot is in a safe start pose.
        DeclareLaunchArgument('start_position_verified', default_value='false',
                              description='true only after checking the start pose: starts '
                                          'marl_deploy_node'),
        DeclareLaunchArgument('engage_wait_for_confirm', default_value='true',
                              description='true: hold the pre-pose until ~/confirm_engage'),
        DeclareLaunchArgument('prepose_ramp_s', default_value='6.0',
                              description='s, the shortest ramp into the default pose'),
        DeclareLaunchArgument('prepose_max_vel', default_value='0.25',
                              description='rad/s, the fastest joint target during that ramp; '
                                          'it lengthens the ramp'),
        DeclareLaunchArgument('disable_elastic_band', default_value='false',
                              description="true: release the sim's elastic band through "
                                          '/elastic_band/toggle once engaged (sim only)'),
        *[DeclareLaunchArgument(f'imu_offset_{axis}_deg', default_value='0.0',
                                description=f'startup IMU {axis} trim [deg]: its reading with the '
                                            'torso level (live: ros2 param set)')
          for axis in ('roll', 'pitch', 'yaw')],
        # The policy on all 27 joints, through safety_node's split inputs.
        Node(
            package='lmg_deploy',
            executable='marl_deploy_node',
            name='marl_deploy_node',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(policy_on),
            parameters=[{
                'use_sim_time': use_sim_time,
                'config': policy_config,
                'urdf_path': LaunchConfiguration('urdf_path'),
                'engage_wait_for_confirm': ParameterValue(
                    LaunchConfiguration('engage_wait_for_confirm'), value_type=bool),
                'disable_elastic_band': ParameterValue(
                    LaunchConfiguration('disable_elastic_band'), value_type=bool),
                'prepose_ramp_s': ParameterValue(
                    LaunchConfiguration('prepose_ramp_s'), value_type=float),
                'prepose_max_vel': ParameterValue(
                    LaunchConfiguration('prepose_max_vel'), value_type=float),
                **{f'imu_offset_{axis}_deg': ParameterValue(
                    LaunchConfiguration(f'imu_offset_{axis}_deg'), value_type=float)
                   for axis in ('roll', 'pitch', 'yaw')},
            }],
        ),
    ])
