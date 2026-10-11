import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Loco-manipulation game bringup (x86 sim): the state publishers, the split
# safety layer with the real robot's preset (relax_safety_split: the policy was
# trained against its e-stops, and the sim is where to see that it doesn't trip
# them), h12_locomanipulationgame.launch.py with the commander running and no
# operator confirm, and the h12_slam navigation stack (use_slam). Start the
# simulator first. The real-robot counterpart is
# lmg_real_bringup.launch.py.
#
# ROS_DOMAIN_ID must be exported in the launching shell (see
# h12_control.launch.py), and not as 0: this launch engages and walks without an
# operator, and domain 0 (also the DDS default when the variable is unset) is
# the real robot's command bus.
ASSETS_DIR = '/home/code/CL_Assets'

GAME_ARGS = ('policy', 'policy_mode', 'engage_wait_for_confirm', 'use_commander',
             'commander_mode', 'goals', 'commander_seed', 'commander_duration_s', 'commander_log')


def _include(package, launch_file, launch_arguments, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    if os.environ.get('ROS_DOMAIN_ID', '').strip() in ('', '0'):
        raise RuntimeError(
            'lmg_sim_bringup: ROS_DOMAIN_ID is unset or 0, the real robot\'s bus. '
            'Export a sim domain (e.g. ROS_DOMAIN_ID=42), or use lmg_real_bringup.launch.py '
            'on the robot.')

    bringup_share = get_package_share_directory('h12_bringup')
    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('policy', default_value='marl_golem'),
        DeclareLaunchArgument('policy_mode', default_value='torch'),
        DeclareLaunchArgument('safety_config', default_value='relax_safety_split.yaml'),
        DeclareLaunchArgument('engage_wait_for_confirm', default_value='false'),
        DeclareLaunchArgument('use_slam', default_value='true'),
        DeclareLaunchArgument('use_commander', default_value='true'),
        DeclareLaunchArgument('commander_mode', default_value='training'),
        DeclareLaunchArgument('goals', default_value='arm_goals_4_10.npz'),
        DeclareLaunchArgument('commander_seed', default_value='0'),
        DeclareLaunchArgument('commander_duration_s', default_value='0.0'),
        DeclareLaunchArgument('commander_log', default_value=''),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(bringup_share, 'rviz', 'nav.rviz')),

        _include('h12_bringup', 'h12_state.launch.py', {
            **sim_time,
            'urdf_file': os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'),
        }),
        _include('h12_bringup', 'h12_control.launch.py', {
            **sim_time,
            'use_frame_task': 'false',
            'safety_config': LaunchConfiguration('safety_config'),
            # no hardware e-stop node in sim to wait for
            'safety_delay': '0.0',
        }),
        _include('h12_bringup', 'h12_locomanipulationgame.launch.py', {
            **sim_time,
            **{name: LaunchConfiguration(name) for name in GAME_ARGS},
            'urdf_path': os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie.urdf'),
            'start_position_verified': 'true',
            'disable_elastic_band': 'true',
        }),
        _include('h12_slam', 'h12_navigation.launch.py', sim_time,
                 condition=IfCondition(LaunchConfiguration('use_slam'))),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_locomanipulationgame',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ])
