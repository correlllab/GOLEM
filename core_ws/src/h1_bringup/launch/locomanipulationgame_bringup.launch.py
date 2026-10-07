import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Loco-manipulation game bringup (x86 sim): the Locomanipulation_game MARL policy
# (locomotion_game_deploy) driving all 27 joints through the split safety layer,
# SLAM, and the game commander that sends it walking and arm-goal commands the
# way training did. ROS_DOMAIN_ID must be exported in the launching shell (see
# h1_control.launch.py); start the simulator first.
#
# marl_deploy_node owns both safety channels, so frame_task_server and the
# lower-body RL node are not started. The safety layer runs the real robot's
# preset (relax_safety_split): the policy was trained against its e-stops, and
# the sim is where to see that it doesn't trip them. Navigation (nav2) is not
# started: the commander is the only /cmd_vel publisher.
ASSETS_DIR = '/home/code/CL_Assets'


def _include(package, launch_file, launch_arguments, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    bringup_share = get_package_share_directory('h1_bringup')
    deploy_share = get_package_share_directory('locomotion_game_deploy')
    slam_share = get_package_share_directory('h12_slam')
    policy_dir = PathJoinSubstitution([deploy_share, 'policies', LaunchConfiguration('policy')])
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('policy', default_value='marl_golem',
                              description='policies/<name>/ of locomotion_game_deploy: <name>.yaml plus the export'),
        DeclareLaunchArgument('policy_mode', default_value='torch',
                              description='torch: the exported networks; zero: plumbing stand-ins'),
        DeclareLaunchArgument('safety_config', default_value='relax_safety_split.yaml'),
        DeclareLaunchArgument('engage_wait_for_confirm', default_value='false',
                              description='hold the pre-pose until ~/confirm_engage (true on the real robot)'),
        DeclareLaunchArgument('use_slam', default_value='true'),
        DeclareLaunchArgument('use_commander', default_value='true'),
        DeclareLaunchArgument('commander_mode', default_value='training', description='training | sequence'),
        DeclareLaunchArgument('goals', default_value='arm_goals_4_10.npz',
                              description='arm goal samples in the policy directory (spread_drop levels)'),
        DeclareLaunchArgument('commander_seed', default_value='0'),
        DeclareLaunchArgument('commander_duration_s', default_value='0.0', description='0: run until stopped'),
        DeclareLaunchArgument('commander_log', default_value='', description='JSON-lines log of events and status'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(bringup_share, 'rviz', 'nav.rviz')),

        _include('h1_bringup', 'h1_state.launch.py', {
            **sim_time,
            'urdf_file': os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'),
        }),
        _include('h1_bringup', 'h1_control.launch.py', {
            **sim_time,
            'use_frame_task': 'false',
            'safety_config': LaunchConfiguration('safety_config'),
            # no hardware e-stop node in sim to wait for
            'safety_delay': '0.0',
        }),

        Node(
            package='locomotion_game_deploy',
            executable='marl_deploy_node',
            name='marl_deploy_node',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'config': PathJoinSubstitution([policy_dir, [LaunchConfiguration('policy'), '.yaml']]),
                'urdf_path': os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie.urdf'),
                'policy_mode': LaunchConfiguration('policy_mode'),
                'engage_wait_for_confirm': ParameterValue(LaunchConfiguration('engage_wait_for_confirm'), value_type=bool),
                'disable_elastic_band': True,
            }],
        ),

        Node(
            package='locomotion_game_deploy',
            executable='game_commander_node',
            name='game_commander',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_commander')),
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

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_locomanipulationgame',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ])
