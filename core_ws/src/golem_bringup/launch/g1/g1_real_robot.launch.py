r"""
Real G1, with the whole ROS stack running on the laptop wired to the robot.

    ros2 launch golem_bringup g1_real_robot.launch.py   # the arms; the legs stay off
    ros2 launch golem_bringup g1_real_robot.launch.py start_position_verified:=true
    ros2 launch golem_bringup g1_real_robot.launch.py use_rviz:=false use_hands:=false

Before launching:
- The operator puts the G1 into Unitree's debug mode, on a gantry. In debug
  mode Unitree's own controller is released and this stack drives every motor
  on rt/lowcmd, from the moment golem_safety starts. Nothing in this stack
  enters debug mode or checks that the robot is in it.
- The gantry carries the robot whenever the legs are off: they hang limp until
  the walking node commands them, and split.yaml's estop turns every motor off.
- ROS_DOMAIN_ID=0 (the robot's DDS domain) and G1_DDS_INTERFACE set to the
  laptop's wired port (ours: enp130s0), both in docker/.env.
- Nothing else may be commanding rt/lowcmd or rt/dex3/*: two publishers fight
  over the motors.
- Clear the space around the arms: once golem_safety relays the controller's
  commands, the arms move to `home` (forearms forward at waist height) at up to
  0.3 rad/s per joint.

golem_safety runs split.yaml: it writes every motor on rt/lowcmd, the legs
from the walking node's input and the waist and arms from the arm
controller's (g1_split.yaml). From boot it streams zero gains on every motor
(every motor off) and engages each input on its own first command: the arms on
the controller's first command, and the move to home starts once the
controller sees golem_safety relay it on rt/lowcmd.

The legs:
- start_position_verified gates the legs. Without it (the default) HOMIE's
  walking node does not start: the leg motors stay off on golem_safety's
  zero-gain stream while the arms run. Set start_position_verified:=true only
  after checking the robot is in a safe start pose.
- With it, the walking node (g1_locomotion's g1.yaml) waits for golem_safety's
  heartbeat, upright rt/lowstate and the arm controller's commands, moves the
  legs from where they are to the policy's default pose over 2 s, and holds
  that pose. golem_safety engages the leg input on its first command.
- /lowerbody/confirm_engage starts the policy. Check the stance, then
  `ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger`; until
  then the legs hold the default pose. The policy starts at zero velocity and
  0.74 m height and walks on /cmd_vel.
- A false or missing heartbeat (golem_safety's estop latched, or it stopped)
  stops the walking node for good. If the node stops on a stale or fallen
  state, a non-finite IMU value or a late command, golem_safety keeps relaying
  its last leg command, which does not balance the robot, and reports the
  quiet input (input_quiet) 0.2 s later. It does not estop on it: the command
  stands until golem_safety estops for another reason (a fall that drives a
  joint past split.yaml's estop limits, say) or the bringup stops. A
  non-finite joint value (q, dq or tau_est) in rt/lowstate is different:
  golem_safety latches its estop on that sample, and split.yaml's estop (mode
  zero, keep_writing) turns every motor off and keeps writing that command.

This launch refuses sim configs (any sim_* safety_config), and the controller
config follows safety_config (g1_split with split); g1_core's docstring lists
every setting it refuses. When the controller exits it sends one last command,
with zero gains on the arms; the waist keeps its hold and golem_safety relays
it, so only the arms go limp. The controller stops with an error if
rt/lowstate does not arrive within 5 s, if golem_safety has not relayed its
commands within 30 s, or if the move to home is not done within 30 s after
that. Sensors and skills are not part of the G1 bringup yet, and it refuses
use_nav:=true: navigation runs in the sim only until the real MID-360 is
connected (g1_navigation's README, The real lidar).

Hazards not handled in code:
- A controller that dies without its last command leaves the waist + arms
  held on its last command: golem_safety holds a quiet input and only reports
  it (input_quiet), until the bringup stops.
- After any estop, restart the whole bringup, not only the safety node: the
  estop stays latched until restart, and the controller's follow gate opens
  for good once golem_safety relays its commands, so a restarted safety node
  would meet a command held wherever the robot used to be.
- The arm IK has no self-collision model (the arms can hit the torso or each
  other). The gains are not tuned on our G1: the waist's damping ratio is
  about 0.42-0.45 at home (g1_split.yaml).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    core = os.path.join(get_package_share_directory('golem_bringup'), 'launch', 'g1',
                        'g1_core.launch.py')
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_hands', default_value='true'),
        DeclareLaunchArgument('safety_config', default_value='split',
                              description='golem_safety config, a name under its config/g1/ '
                                          'or a path'),
        # the leg-control interlock
        DeclareLaunchArgument(
            'start_position_verified', default_value='false',
            description="HOMIE's walking node on the legs: true only after checking the "
                        'robot is in a safe start pose'),
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
                # the one safety_config runs with: g1_split with split
                'controller_config': '',
                'start_position_verified': LaunchConfiguration('start_position_verified'),
                'use_nav': LaunchConfiguration('use_nav'),
            }.items(),
        ),
    ])
