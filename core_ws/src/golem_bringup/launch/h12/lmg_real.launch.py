"""
Start the loco-manipulation game on the real H1-2's onboard PC; refuses to start today.

    ros2 launch golem_bringup lmg_real.launch.py policy:=mid360v2

SAFETY: this launch refuses to start, before any node: lmg_deploy reads /livox/imu as
upright, and the H1-2's MID-360 is mounted upside down and publishes it in its raw
frame, so the policy would get an upside-down IMU. The refusal goes once lmg_deploy
handles the raw frame (the strict xfail test_urdf_mount_is_the_livox_pitch marks it).

What it starts otherwise, all at once and INSTEAD of h12_real_robot.launch.py, never
beside it: the drivers (h12_real_drivers: the MID-360, and the head camera and grippers
when asked), the state publishers, the command path with the hardware estop but without
frame_task_server, because marl_deploy_node owns both safety inputs, the policy
(h12_locomanipulationgame) and navigation (use_slam). Keep the lower-body RL controller
off.

SAFETY: nothing commands the robot until both operator gates pass:
  1. start_position_verified:=true starts marl_deploy_node; it ramps all 27 joints to
     the policy's default pose and holds there;
  2. ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger engages it.
The robot then walks on /cmd_vel and reaches for the arm goals the operator sets
through /marl_deploy_node/set_mode and set_arm_goals.

Every node joins the launching shell's ROS_DOMAIN_ID, which must be 0, the real robot's
command bus; ROS uses 0 when it is unset.
"""
import os

from ament_index_python.packages import get_package_share_directory
from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

NOT_STARTING = ('lmg_real: not starting: lmg_deploy reads /livox/imu as upright, and the H1-2 '
                'MID-360 is in its raw (upside-down) frame; update lmg_deploy first (see the '
                'strict xfail test_urdf_mount_is_the_livox_pitch)')

# The arguments passed through to h12_locomanipulationgame.launch.py as they are.
GAME_ARGS = ('start_position_verified', 'policy', 'prepose_ramp_s', 'prepose_max_vel',
             'imu_offset_roll_deg', 'imu_offset_pitch_deg', 'imu_offset_yaw_deg')


def _include(package, launch_file, launch_arguments, condition=None):
    """Include a package's launch file with the given arguments."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
        condition=condition,
    )


def _refuse(context):
    """Stop the launch: lmg_deploy cannot read the raw-frame IMU yet."""
    raise RuntimeError(NOT_STARTING)


def generate_launch_description():
    """Refuse first, then declare the game's arguments and blocks."""
    # The real robot runs on the wall clock: there is no /clock.
    wall_time = {'use_sim_time': 'false'}

    return LaunchDescription([
        # SAFETY: first, so the launch stops before any node starts.
        OpaqueFunction(function=_refuse),
        # SAFETY: false by default, so a bare launch never commands the robot.
        DeclareLaunchArgument('start_position_verified', default_value='false',
                              description='true only after checking the robot is in a '
                                          'safe start pose: starts marl_deploy_node'),
        DeclareLaunchArgument('policy',
                              description='Required. The policy, policies/<name>/ of '
                                          'lmg_deploy (it ships mid360v2)'),
        DeclareLaunchArgument('prepose_ramp_s', default_value='6.0',
                              description='s, the shortest ramp into the default pose'),
        DeclareLaunchArgument('prepose_max_vel', default_value='0.25',
                              description='rad/s, the fastest joint target during that ramp; '
                                          'it lengthens the ramp'),
        # The URDF's livox_joint carries the measured mount, so the trims start at 0;
        # tune them live with ros2 param set /marl_deploy_node imu_offset_pitch_deg <deg>.
        DeclareLaunchArgument('imu_offset_roll_deg', default_value='0.0',
                              description="deg, the start-up roll trim of the policy's IMU "
                                          '(/livox/imu): its reading with the torso level'),
        DeclareLaunchArgument('imu_offset_pitch_deg', default_value='0.0',
                              description="deg, the start-up pitch trim of the policy's IMU"),
        DeclareLaunchArgument('imu_offset_yaw_deg', default_value='0.0',
                              description="deg, the start-up yaw trim of the policy's IMU"),
        DeclareLaunchArgument('safety_config', default_value='relax_safety_split.yaml',
                              description='golem_safety config: a name under its config/h12/, '
                                          'or a path'),
        DeclareLaunchArgument('use_slam', default_value='true',
                              description="true: start golem_slam's navigation, which also "
                                          'gives the policy /Odometry for its arm goals'),
        # The game reads the MID-360 for SLAM and as the policy's IMU;
        # marl_deploy_node waits in the pre-pose until /livox/imu is up.
        DeclareLaunchArgument('use_lidar', default_value='true',
                              description='true: start the MID-360 driver'),
        DeclareLaunchArgument('use_head_camera', default_value='false',
                              description='true: start the head RealSense'),
        DeclareLaunchArgument('use_grippers', default_value='false',
                              description='true: start the left and right gripper nodes'),

        # The MID-360 and, when asked, the head camera and the grippers.
        _include('golem_bringup', 'h12/h12_real_drivers.launch.py', {
            **wall_time,
            **{name: LaunchConfiguration(name)
               for name in ('use_lidar', 'use_head_camera', 'use_grippers')},
        }),
        # TF of the body from /lowstate.
        _include('golem_bringup', 'h12/h12_state.launch.py', wall_time),
        # safety_node with the hardware estop; the policy owns both of its inputs,
        # so no frame_task_server runs.
        _include('golem_bringup', 'h12/h12_control.launch.py', {
            **wall_time,
            'sim': 'false',
            'use_estop': 'true',
            'use_frame_task': 'false',
            'safety_config': LaunchConfiguration('safety_config'),
        }),
        # The policy, on all 27 joints, behind the operator gates.
        _include('golem_bringup', 'h12/h12_locomanipulationgame.launch.py', {
            **wall_time,
            **{name: LaunchConfiguration(name) for name in GAME_ARGS},
            'urdf_path': urdf_path('h12'),
            # SAFETY: the operator gate is fixed on the real robot, not an argument;
            # the real robot has no elastic band to release.
            'engage_wait_for_confirm': 'true',
            'disable_elastic_band': 'false',
        }),
        # SLAM and Nav2, driving the policy on /cmd_vel.
        _include('golem_slam', 'navigation.launch.py', {**wall_time, 'robot': 'h12'},
                 condition=IfCondition(LaunchConfiguration('use_slam'))),
    ])
