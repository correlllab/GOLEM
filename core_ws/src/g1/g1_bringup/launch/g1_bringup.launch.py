r"""
Real G1, with the whole ROS stack running on the laptop wired to the robot.

    ros2 launch g1_bringup g1_bringup.launch.py
    ros2 launch g1_bringup g1_bringup.launch.py use_rviz:=false use_hands:=false

Before launching:
- ROS_DOMAIN_ID=0 (the robot's DDS domain) and G1_DDS_INTERFACE set to the
  laptop's wired port (ours: enp130s0), both in docker/.env.
- The robot is standing under Unitree's own controller: in arm_sdk mode the
  legs stay Unitree's, and only the waist + arms are taken over.
- Nothing else may be commanding rt/lowcmd, rt/arm_sdk or rt/dex3/*: two
  publishers fight over the motors.
- Clear the space around the arms: once the controller has a fresh
  rt/lowstate, golem_safety ramps control over from Unitree for 1 s, then
  the arms move to `home` (forearms forward at waist height) at up to 0.3
  rad/s per joint.

This launch refuses sim configs (any sim_* safety_config), and the controller
config follows safety_config (g1 with arm_sdk, g1_split with split); g1_core's
docstring lists every pairing it refuses. When the controller exits it sends
one last command, with zero gains on the arms. In arm_sdk mode it zeroes the
waist too, so golem_safety refuses it (its waist kp is below
arm_sdk.waist_kp_min), latches its estop and hands the waist and arms back to
Unitree, as it does on a command that is not finite; in split mode the waist
keeps its hold and golem_safety relays it, so only the arms go limp. The controller
stops with an error if rt/lowstate does not arrive within 5 s, if
golem_safety has not relayed its commands within 30 s, or if the move to home
is not done within 30 s after that. Sensors and skills are not part of the G1
bringup yet, and it refuses use_nav:=true: navigation runs in the sim only
until the real MID-360 is connected (g1_navigation's README, The real lidar).

Hazards not handled in code:
- A controller that dies without its last command leaves the waist + arms
  held at weight 1 on its last command: golem_safety holds a quiet input and
  only reports it (input_quiet). Stopping the bringup hands them back to
  Unitree over arm_sdk.release_seconds (3 s).
- After any estop, restart the whole bringup, not only the safety node: the
  estop stays latched until restart, and the controller's follow gate opens
  for good once golem_safety relays its commands, so a restarted safety node
  would meet a command held wherever the robot used to be.
- Until rt/arm_sdk reaches weight 1 (the end of the 1 s ramp), a waist or arm
  joint pushed more than 0.05 rad from its command drags the command along;
  Unitree's share of the blend holds the joint meanwhile. From weight 1 on the
  commands hold.
- The arm IK has no self-collision model (the arms can hit the torso or each
  other). The gains are not tuned on our G1: the waist's damping ratio is
  about 0.46 at home and 0.26 with the arms out to the sides.

Walking with HOMIE's policy on the legs instead of Unitree's controller:

    ros2 launch g1_bringup g1_bringup.launch.py safety_config:=split \
        use_locomotion:=true start_position_verified:=true

- The legs then go through golem_safety's split mode on rt/lowcmd, so
  Unitree's controller must already be released; that is a manual operator
  step, and nothing in this stack does it.
- safety_config:=split runs g1_split.yaml (a stiff waist), walking or not;
  g1_core refuses any other controller config with it.
- Support the robot (a gantry): split.yaml's estop zeroes every motor.
- Split mode engages only once the walking node commands too; until then the
  controller's follow gate keeps the arm and waist commands within 0.05 rad of
  wherever the robot hangs, and the move to home starts when the controller
  sees golem_safety's first rt/lowcmd.
- Set start_position_verified:=true only after checking the robot is in a safe
  start pose; without it the launch refuses to start.
- The walking node waits for upright rt/lowstate and the arm controller's
  commands, moves the legs from where they are to the policy's default pose
  over 3 s, then starts the policy at zero velocity and 0.74 m height. It
  walks on /cmd_vel. If it stops on a stale or fallen state, a non-finite IMU
  value or a late command, golem_safety keeps relaying its last leg command,
  which does not balance the robot, and reports the quiet input (input_quiet)
  0.2 s later. It does not estop on it: the command stands until golem_safety
  estops for another reason (a fall that drives a joint past split.yaml's
  estop limits, say) or the bringup stops. A non-finite joint value (q, dq or
  tau_est) in rt/lowstate is different: golem_safety latches its estop on
  that sample, and split.yaml's estop (mode zero, keep_writing) turns every
  motor off and keeps writing that command.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    core = os.path.join(get_package_share_directory('g1_bringup'), 'launch', 'g1_core.launch.py')
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_hands', default_value='true'),
        DeclareLaunchArgument('safety_config', default_value='arm_sdk',
                              description='arm_sdk (Unitree balances) or split (own lower body)'),
        DeclareLaunchArgument(
            'use_locomotion', default_value='false',
            description="HOMIE's walking policy on the legs (needs safety_config:=split)"),
        # Leg-control interlock: g1_core refuses use_locomotion:=true unless
        # start_position_verified:=true, set only after checking the robot is in
        # a safe start pose
        DeclareLaunchArgument('start_position_verified', default_value='false'),
        # refused for now: the real lidar is not connected yet
        DeclareLaunchArgument('use_nav', default_value='false',
                              description='g1_navigation; refused until the real MID-360 '
                                          'is connected'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(core),
            launch_arguments={
                'sim': 'false',
                'use_sim_time': 'false',
                'use_rviz': LaunchConfiguration('use_rviz'),
                'use_hands': LaunchConfiguration('use_hands'),
                'safety_config': LaunchConfiguration('safety_config'),
                # the one safety_config runs with: g1 with arm_sdk, g1_split (a
                # stiff waist) with split, walking or not
                'controller_config': '',
                'use_locomotion': LaunchConfiguration('use_locomotion'),
                'start_position_verified': LaunchConfiguration('start_position_verified'),
                'use_nav': LaunchConfiguration('use_nav'),
            }.items(),
        ),
    ])
