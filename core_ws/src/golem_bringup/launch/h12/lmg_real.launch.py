"""
Start the loco-manipulation game on the real H1-2's onboard PC.

    ros2 launch golem_bringup lmg_real.launch.py

The policy reads the torso IMU as /lowstate reports it, with the real robot's calibration
removed (lmg_deploy's README, "IMU calibration").

What it starts, all at once and INSTEAD of h12_real_robot.launch.py, never
beside it: the drivers (h12_real_drivers: the MID-360, and the head camera and grippers
when asked), the state publishers, the command path with the hardware estop but without
frame_task_server, because marl_deploy_node owns both safety inputs, the policy
(h12_locomanipulationgame) and navigation (use_slam). Keep the lower-body RL controller
off.

SAFETY: nothing commands the robot until both operator gates pass:
  1. start_position_verified:=true starts marl_deploy_node; it ramps all 27 joints to
     the policy's default pose and holds there;
  2. ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger engages it.
The robot then walks on /cmd_vel.

SAFETY: marl_deploy_node has no fall check, and a stale /lowstate only
pauses its publishing; golem_safety then keeps relaying its last targets, which do not
balance the robot. The hardware estop is the stop: keep it in hand.

Every node joins the launching shell's ROS_DOMAIN_ID, which must be 0, the real robot's
command bus; ROS uses 0 when it is unset.
"""

from golem_bringup.include import include
from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

# The arguments passed through to h12_locomanipulationgame.launch.py as they are.
GAME_ARGS = ('start_position_verified', 'prepose_ramp_s', 'prepose_max_vel')


def generate_launch_description():
    """Declare the game's arguments and blocks."""
    # The real robot runs on the wall clock: there is no /clock.
    wall_time = {'use_sim_time': 'false'}

    return LaunchDescription([
        # SAFETY: false by default, so a bare launch never commands the robot.
        DeclareLaunchArgument('start_position_verified', default_value='false',
                              description='true only after checking the robot is in a '
                                          'safe start pose: starts marl_deploy_node'),
        DeclareLaunchArgument('prepose_ramp_s', default_value='6.0',
                              description='s, the shortest ramp into the default pose'),
        DeclareLaunchArgument('prepose_max_vel', default_value='0.25',
                              description='rad/s, the fastest joint target during that ramp; '
                                          'it lengthens the ramp'),
        DeclareLaunchArgument('safety_config', default_value='relax_safety_split.yaml',
                              description='golem_safety config: a name under its config/h12/, '
                                          'or a path'),
        DeclareLaunchArgument('use_slam', default_value='true',
                              description="true: start golem_slam's navigation, which drives "
                                          'the policy on /cmd_vel'),
        # The MID-360 is SLAM's lidar and IMU.
        DeclareLaunchArgument('use_lidar', default_value='true',
                              description='true: start the MID-360 driver'),
        DeclareLaunchArgument('use_head_camera', default_value='false',
                              description='true: start the head RealSense'),
        DeclareLaunchArgument('use_grippers', default_value='false',
                              description='true: start the left and right gripper nodes'),

        # The MID-360 and, when asked, the head camera and the grippers.
        include('golem_bringup', 'h12/h12_real_drivers.launch.py', {
            name: LaunchConfiguration(name)
            for name in ('use_lidar', 'use_head_camera', 'use_grippers')}),
        # TF of the body from /lowstate.
        include('golem_bringup', 'h12/h12_state.launch.py', wall_time),
        # safety_node with the hardware estop; the policy owns both of its inputs,
        # so no frame_task_server runs.
        include('golem_bringup', 'h12/h12_control.launch.py', {
            **wall_time,
            'sim': 'false',
            'use_frame_task': 'false',
            'safety_config': LaunchConfiguration('safety_config'),
        }),
        # The policy, on all 27 joints, behind the operator gates.
        include('golem_bringup', 'h12/h12_locomanipulationgame.launch.py', {
            **wall_time,
            **{name: LaunchConfiguration(name) for name in GAME_ARGS},
            'urdf_path': urdf_path('h12'),
            # SAFETY: the operator gate is fixed on the real robot, not an argument;
            # the real robot has no elastic band to release.
            'engage_wait_for_confirm': 'true',
            'disable_elastic_band': 'false',
        }),
        # SLAM and Nav2, driving the policy on /cmd_vel.
        include('golem_slam', 'navigation.launch.py', {**wall_time, 'robot': 'h12'},
                condition=IfCondition(LaunchConfiguration('use_slam'))),
    ])
