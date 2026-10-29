import os

from ament_index_python.packages import get_package_share_directory
from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# Loco-manipulation game, real robot, onboard PC. Run it INSTEAD of
# h12_real_robot.launch.py, never beside it: it starts the same drivers,
# state publishers, estop, safety layer and h12_slam navigation stack
# (use_slam), but without frame_task_server,
# because marl_deploy_node owns both safety channels. Keep the lower-body RL
# and MJPC controllers (h12_real_robot.launch.py's leg control) off.
#
# Nothing commands the robot until both operator gates pass:
#   1. start_position_verified:=true starts marl_deploy_node; it ramps all 27
#      joints to the policy's default pose and holds there.
#   2. ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger
#      engages the policy.
# The robot then walks on /cmd_vel and reaches on the arm goals the operator sets
# through /marl_deploy_node/set_mode and set_arm_goals.
#
# Every node joins the launching shell's ROS_DOMAIN_ID, which must be 0, the real
# robot's command bus; ROS uses 0 when it is unset (see h12_control.launch.py).

GAME_ARGS = ('start_position_verified', 'policy', 'prepose_ramp_s', 'prepose_max_vel',
             'imu_offset_roll_deg', 'imu_offset_pitch_deg', 'imu_offset_yaw_deg')


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
        DeclareLaunchArgument('policy', description='policies/<name>/ of lmg_deploy (required)'),
        DeclareLaunchArgument('prepose_ramp_s', default_value='6.0'),
        DeclareLaunchArgument('prepose_max_vel', default_value='0.25'),
        # Startup trim of the policy's IMU, the MID360 (/livox/imu): its reading with the torso
        # level. The URDF livox_joint carries the measured mount, so it starts at 0. Tune live
        # with ros2 param set /marl_deploy_node imu_offset_pitch_deg <deg>.
        DeclareLaunchArgument('imu_offset_roll_deg', default_value='0.0'),
        DeclareLaunchArgument('imu_offset_pitch_deg', default_value='0.0'),
        DeclareLaunchArgument('imu_offset_yaw_deg', default_value='0.0'),
        DeclareLaunchArgument('safety_config', default_value='relax_safety_split.yaml'),
        DeclareLaunchArgument('use_slam', default_value='true'),
        # The game reads the MID360 for SLAM and as the policy's IMU; marl_deploy_node waits
        # in the pre-pose until /livox/imu is up.
        DeclareLaunchArgument('use_lidar', default_value='true',
                              description='MID360 driver and livox_imu_upright'),
        DeclareLaunchArgument('use_head_camera', default_value='false',
                              description='head RealSense'),
        DeclareLaunchArgument('use_grippers', default_value='false',
                              description='left and right gripper nodes'),

        _include('golem_bringup', 'h12/h12_real_drivers.launch.py', {
            **wall_time,
            **{name: LaunchConfiguration(name)
               for name in ('use_lidar', 'use_head_camera', 'use_grippers')},
        }),
        _include('golem_bringup', 'h12/h12_state.launch.py', wall_time),
        _include('golem_bringup', 'h12/h12_control.launch.py', {
            **wall_time,
            'sim': 'false',
            'use_estop': 'true',
            'use_frame_task': 'false',
            'safety_config': LaunchConfiguration('safety_config'),
        }),
        _include('golem_bringup', 'h12/h12_locomanipulationgame.launch.py', {
            **wall_time,
            **{name: LaunchConfiguration(name) for name in GAME_ARGS},
            'urdf_path': urdf_path('h12'),
            # The operator gate and the absent elastic band are fixed on the real robot, not
            # arguments.
            'engage_wait_for_confirm': 'true',
            'disable_elastic_band': 'false',
        }),
        _include('h12_slam', 'h12_navigation.launch.py', wall_time,
                 condition=IfCondition(LaunchConfiguration('use_slam'))),
    ])
