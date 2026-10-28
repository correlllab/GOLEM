from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Loco-manipulation game: the Locomanipulation_game MARL policy
# (lmg_deploy's marl_deploy_node, all 27 joints through the split
# safety layer). Included by lmg_sim.launch.py
# and lmg_real.launch.py, which start the state publishers, the safety
# layer and SLAM.
#
# marl_deploy_node owns both safety channels, so frame_task_server and the
# lower-body RL node must not run beside it.


def generate_launch_description():
    deploy_share = get_package_share_directory('lmg_deploy')
    policy_dir = PathJoinSubstitution([deploy_share, 'policies', LaunchConfiguration('policy')])
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    policy_on = LaunchConfiguration('start_position_verified')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'urdf_path',
            description='Pinocchio URDF for the wrist kinematics (h1_2_magpie.urdf)'),
        # Interlock: the policy node doesn't start until the operator has
        # verified the robot is in a safe start position.
        DeclareLaunchArgument('start_position_verified', default_value='false',
                              description='Start marl_deploy_node'),
        DeclareLaunchArgument(
            'policy',
            description='policies/<name>/ of lmg_deploy: <name>.yaml plus the export (required: '
                        'lmg_deploy ships none)'),
        DeclareLaunchArgument('engage_wait_for_confirm', default_value='true',
                              description='hold the pre-pose until ~/confirm_engage'),
        DeclareLaunchArgument('prepose_ramp_s', default_value='6.0',
                              description='minimum duration [s] of the ramp into the default '
                                          'pose'),
        DeclareLaunchArgument('prepose_max_vel', default_value='0.25',
                              description='peak joint target speed [rad/s] during that ramp; '
                                          'lengthens it'),
        DeclareLaunchArgument('disable_elastic_band', default_value='false',
                              description="release the sim's elastic band through "
                                          '/elastic_band/toggle once engaged'),
        *[DeclareLaunchArgument(f'imu_offset_{axis}_deg', default_value='0.0',
                                description=f'startup IMU {axis} trim [deg]: its reading with the '
                                            'torso level (live: ros2 param set)')
          for axis in ('roll', 'pitch', 'yaw')],
        Node(
            package='lmg_deploy',
            executable='marl_deploy_node',
            name='marl_deploy_node',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(policy_on),
            parameters=[{
                'use_sim_time': use_sim_time,
                'config': ParameterValue(
                    [policy_dir, '/', LaunchConfiguration('policy'), '.yaml'], value_type=str),
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
