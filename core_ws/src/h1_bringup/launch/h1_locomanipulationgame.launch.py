import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import AndSubstitution, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Loco-manipulation game: the Locomanipulation_game MARL policy
# (locomotion_game_deploy's marl_deploy_node, all 27 joints through the split
# safety layer), the game commander, and SLAM without nav2. Included by
# locomanipulationgame_sim_bringup.launch.py and
# locomanipulationgame_real_bringup.launch.py, which start the state publishers
# and the safety layer.
#
# marl_deploy_node owns both safety channels, so frame_task_server and the
# lower-body RL node must not run beside it. nav2 is not started: the commander
# (or the operator) is the only /cmd_vel publisher.


def generate_launch_description():
    deploy_share = get_package_share_directory('locomotion_game_deploy')
    slam_share = get_package_share_directory('h12_slam')
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
                              description='policies/<name>/ of locomotion_game_deploy: <name>.yaml plus the export'),
        DeclareLaunchArgument('policy_mode', default_value='torch',
                              description='torch: the exported networks; zero: plumbing stand-ins'),
        DeclareLaunchArgument('engage_wait_for_confirm', default_value='true',
                              description='hold the pre-pose until ~/confirm_engage'),
        DeclareLaunchArgument('disable_elastic_band', default_value='false',
                              description="release the sim's elastic band through /elastic_band/toggle once engaged"),
        DeclareLaunchArgument('use_slam', default_value='true'),
        DeclareLaunchArgument('use_commander', default_value='false'),
        DeclareLaunchArgument('commander_mode', default_value='training', description='training | sequence'),
        DeclareLaunchArgument('goals', default_value='arm_goals_4_10.npz',
                              description='arm goal samples in the policy directory (spread_drop levels)'),
        DeclareLaunchArgument('commander_seed', default_value='0'),
        DeclareLaunchArgument('commander_duration_s', default_value='0.0', description='0: run until stopped'),
        DeclareLaunchArgument('commander_log', default_value='', description='JSON-lines log of events and status'),

        Node(
            package='locomotion_game_deploy',
            executable='marl_deploy_node',
            name='marl_deploy_node',
            output='screen',
            condition=IfCondition(policy_on),
            parameters=[{
                'use_sim_time': use_sim_time,
                'config': PathJoinSubstitution([policy_dir, [LaunchConfiguration('policy'), '.yaml']]),
                'urdf_path': LaunchConfiguration('urdf_path'),
                'policy_mode': LaunchConfiguration('policy_mode'),
                'engage_wait_for_confirm': ParameterValue(LaunchConfiguration('engage_wait_for_confirm'), value_type=bool),
                'disable_elastic_band': ParameterValue(LaunchConfiguration('disable_elastic_band'), value_type=bool),
            }],
        ),

        Node(
            package='locomotion_game_deploy',
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

        # SLAM: FAST-LIO (/livox/lidar + /livox/imu -> /Odometry, camera_init -> body), its body -> pelvis
        # mount, and slam_toolbox's 2-D map (map -> camera_init) from the pelvis-height scan. Same nodes and
        # configs as h12_slam/h1_navigation.launch.py, without nav2.
        Node(
            package='fast_lio',
            executable='fastlio_mapping',
            parameters=[os.path.join(slam_share, 'config', 'mid360.yaml'), {'use_sim_time': use_sim_time}],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_slam')),
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='body_to_pelvis_broadcaster',
            arguments=['0.11433', '0.0', '-0.67188', '0.0', '-0.24015730507441985', '0.0', 'body', 'pelvis'],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_slam')),
        ),
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='cloud_registered_to_laserscan',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_slam')),
            parameters=[{
                'target_frame': 'pelvis', 'use_sim_time': use_sim_time,
                'min_height': -0.65, 'max_height': 0.65,
                'angle_min': -3.14159, 'angle_max': 3.14159, 'angle_increment': 0.0087,
                'range_min': 0.9, 'range_max': 3.0, 'use_inf': True,
                'scan_time': 0.0333, 'transform_tolerance': 0.05, 'queue_size': 20,
            }],
            remappings=[('cloud_in', '/cloud_registered_body'), ('scan', '/converted_scan')],
        ),
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_slam')),
            parameters=[os.path.join(slam_share, 'config', 'slam_toolbox_h1.yaml'), {
                'base_frame': 'pelvis', 'odom_frame': 'camera_init', 'scan_topic': '/converted_scan',
                'odom_topic': '/Odometry', 'use_sim_time': use_sim_time,
            }],
        ),
    ])
