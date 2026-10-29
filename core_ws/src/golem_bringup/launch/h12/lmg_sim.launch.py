import os

from ament_index_python.packages import get_package_share_directory
from golem_ros2_models.paths import urdf_path
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
# them), h12_locomanipulationgame.launch.py with no operator confirm, and the
# h12_slam navigation stack (use_slam), with rviz (use_rviz) for nav2 goals. Start the
# simulator first. The real-robot counterpart is
# lmg_real.launch.py.
#
# ROS_DOMAIN_ID must be exported in the launching shell (see
# h12_control.launch.py), and not as 0: this launch engages and walks without an
# operator, and domain 0 (also the DDS default when the variable is unset) is
# the real robot's command bus.

GAME_ARGS = ('policy', 'prepose_ramp_s', 'prepose_max_vel')


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
            'lmg_sim: ROS_DOMAIN_ID is unset or 0, the real robot\'s bus. '
            'Export a sim domain (e.g. ROS_DOMAIN_ID=42), or use lmg_real.launch.py '
            'on the robot.')

    bringup_share = get_package_share_directory('golem_bringup')
    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('policy', description='policies/<name>/ of lmg_deploy (required)'),
        DeclareLaunchArgument('prepose_ramp_s', default_value='6.0'),
        DeclareLaunchArgument('prepose_max_vel', default_value='0.25'),
        DeclareLaunchArgument('safety_config', default_value='relax_safety_split.yaml'),
        DeclareLaunchArgument('use_slam', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument(
            'rviz_config', default_value=os.path.join(bringup_share, 'rviz', 'h12', 'main.rviz')),

        _include('golem_bringup', 'h12/h12_state.launch.py', {
            **sim_time,
            'urdf_file': urdf_path('h12', ros=True),
        }),
        _include('golem_bringup', 'h12/h12_control.launch.py', {
            **sim_time,
            'sim': 'true',
            'use_frame_task': 'false',
            'safety_config': LaunchConfiguration('safety_config'),
            # no hardware e-stop node in sim to wait for
            'safety_delay': '0.0',
        }),
        _include('golem_bringup', 'h12/h12_locomanipulationgame.launch.py', {
            **sim_time,
            **{name: LaunchConfiguration(name) for name in GAME_ARGS},
            'urdf_path': urdf_path('h12'),
            'start_position_verified': 'true',
            # engages as soon as the pre-pose settles, with no operator confirm
            'engage_wait_for_confirm': 'false',
            'disable_elastic_band': 'true',
        }),
        _include('h12_slam', 'h12_navigation.launch.py', sim_time,
                 condition=IfCondition(LaunchConfiguration('use_slam'))),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_locomanipulationgame',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'),
                                                        value_type=bool)}],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ])
