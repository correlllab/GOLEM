import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# Loco-manipulation game, real robot, onboard PC. Run it INSTEAD of
# h12_real_robot_bringup.launch.py, never beside it: it starts the same drivers,
# state publishers, estop, safety layer and h12_slam navigation stack
# (use_slam), but without frame_task_server,
# because marl_deploy_node owns both safety channels. Keep the lower-body RL
# and MJPC controllers (h12_real_desktop_bringup.launch.py's leg control) off.
#
# Nothing commands the robot until both operator gates pass:
#   1. start_position_verified:=true starts marl_deploy_node (and the commander,
#      if use_commander:=true); it ramps all 27 joints to the policy's default
#      pose and holds there.
#   2. ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger
#      engages the policy.
# With use_commander:=false (the default) the robot only walks on /cmd_vel and
# reaches on /marl_deploy_node/arm_goal from the operator.
#
# ROS_DOMAIN_ID must be exported in the launching shell (0 on the real robot;
# see h12_control.launch.py). Assets resolve as in h12_state.launch.py.
ASSETS_DIR = os.environ.get('GOLEM_ASSETS_DIR', '/home/unitree/GOLEM/CL_Assets')

GAME_ARGS = ('start_position_verified', 'policy', 'prepose_ramp_s', 'prepose_max_vel', 'use_commander',
             'commander_mode', 'goals', 'commander_seed', 'commander_duration_s', 'commander_log')


def _include(package, launch_file, launch_arguments, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    # Real robot runs on wall-clock time (no simulator /clock).
    wall_time = {'use_sim_time': 'false'}

    return LaunchDescription([
        DeclareLaunchArgument('start_position_verified', default_value='false'),
        DeclareLaunchArgument('policy', default_value='marl_golem'),
        DeclareLaunchArgument('prepose_ramp_s', default_value='6.0'),
        DeclareLaunchArgument('prepose_max_vel', default_value='0.25'),
        DeclareLaunchArgument('safety_config', default_value='relax_safety_split.yaml'),
        DeclareLaunchArgument('use_slam', default_value='true'),
        # The game reads only the lidar (SLAM and the commander's obstacle guard).
        DeclareLaunchArgument('use_lidar', default_value='true',
                              description='MID360 driver and livox_imu_upright'),
        DeclareLaunchArgument('use_head_camera', default_value='false', description='head RealSense'),
        DeclareLaunchArgument('use_grippers', default_value='false', description='left and right gripper nodes'),
        DeclareLaunchArgument('use_commander', default_value='false'),
        DeclareLaunchArgument('commander_mode', default_value='training'),
        DeclareLaunchArgument('goals', default_value='arm_goals_4_10.npz'),
        DeclareLaunchArgument('commander_seed', default_value='0'),
        DeclareLaunchArgument('commander_duration_s', default_value='0.0'),
        DeclareLaunchArgument('commander_log', default_value=''),

        _include('h12_bringup', 'h12_real_drivers.launch.py', {
            **wall_time,
            **{name: LaunchConfiguration(name) for name in ('use_lidar', 'use_head_camera', 'use_grippers')},
        }),
        _include('h12_bringup', 'h12_state.launch.py', wall_time),
        _include('h12_bringup', 'h12_control.launch.py', {
            **wall_time,
            'use_estop': 'true',
            'use_frame_task': 'false',
            'safety_config': LaunchConfiguration('safety_config'),
        }),
        _include('h12_bringup', 'h12_locomanipulationgame.launch.py', {
            **wall_time,
            **{name: LaunchConfiguration(name) for name in GAME_ARGS},
            'urdf_path': os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie.urdf'),
            # The operator gate and the absent elastic band are fixed on the real robot, not arguments.
            'engage_wait_for_confirm': 'true',
            'disable_elastic_band': 'false',
        }),
        _include('h12_slam', 'h12_navigation.launch.py', wall_time,
                 condition=IfCondition(LaunchConfiguration('use_slam'))),
    ])
