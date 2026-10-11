from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import AndSubstitution, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Loco-manipulation game: the Locomanipulation_game MARL policy
# (lmg_deploy's marl_deploy_node, all 27 joints through the split
# safety layer) and the game commander. Included by lmg_sim_bringup.launch.py
# and lmg_real_bringup.launch.py, which start the state publishers, the safety
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
        DeclareLaunchArgument('urdf_path', description='Pinocchio URDF for the wrist kinematics (h1_2_magpie.urdf)'),
        # Interlock: neither the policy node nor the commander starts until the
        # operator has verified the robot is in a safe start position.
        DeclareLaunchArgument('start_position_verified', default_value='false',
                              description='Start marl_deploy_node and the commander'),
        DeclareLaunchArgument('policy', default_value='marl_golem',
                              description='policies/<name>/ of lmg_deploy: <name>.yaml plus the export'),
        DeclareLaunchArgument('engage_wait_for_confirm', default_value='true',
                              description='hold the pre-pose until ~/confirm_engage'),
        DeclareLaunchArgument('disable_elastic_band', default_value='false',
                              description="release the sim's elastic band through /elastic_band/toggle once engaged"),
        DeclareLaunchArgument('use_commander', default_value='false'),
        DeclareLaunchArgument('commander_mode', default_value='training', description='training | sequence'),
        DeclareLaunchArgument('goals', default_value='arm_goals_4_10.npz',
                              description='arm goal samples in the policy directory (spread_drop levels)'),
        DeclareLaunchArgument('commander_seed', default_value='0'),
        DeclareLaunchArgument('commander_duration_s', default_value='0.0', description='0: run until stopped'),
        DeclareLaunchArgument('commander_log', default_value='', description='JSON-lines log of events and status'),

        Node(
            package='lmg_deploy',
            executable='marl_deploy_node',
            name='marl_deploy_node',
            output='screen',
            condition=IfCondition(policy_on),
            parameters=[{
                'use_sim_time': use_sim_time,
                'config': ParameterValue([policy_dir, '/', LaunchConfiguration('policy'), '.yaml'], value_type=str),
                'urdf_path': LaunchConfiguration('urdf_path'),
                'engage_wait_for_confirm': ParameterValue(LaunchConfiguration('engage_wait_for_confirm'), value_type=bool),
                'disable_elastic_band': ParameterValue(LaunchConfiguration('disable_elastic_band'), value_type=bool),
            }],
        ),

        Node(
            package='lmg_deploy',
            executable='game_commander_node',
            name='game_commander',
            output='screen',
            condition=IfCondition(AndSubstitution(policy_on, LaunchConfiguration('use_commander'))),
            parameters=[{
                'use_sim_time': use_sim_time,
                'mode': LaunchConfiguration('commander_mode'),
                'goals_file': PathJoinSubstitution([policy_dir, LaunchConfiguration('goals')]),
                'seed': ParameterValue(LaunchConfiguration('commander_seed'), value_type=int),
                'duration_s': ParameterValue(LaunchConfiguration('commander_duration_s'), value_type=float),
                'log_file': LaunchConfiguration('commander_log'),
                'cloud_frame': 'livox_link',
            }],
        ),
    ])
