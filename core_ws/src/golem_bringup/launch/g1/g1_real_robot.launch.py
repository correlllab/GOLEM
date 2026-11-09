r"""
Start the real G1, with the whole ROS stack on the laptop wired to the robot.

    ros2 launch golem_bringup g1_real_robot.launch.py   # the arms; the legs stay off
    ros2 launch golem_bringup g1_real_robot.launch.py start_position_verified:=true
    ros2 launch golem_bringup g1_real_robot.launch.py use_rviz:=false use_hands:=false

Before launching:
- The operator puts the G1 into Unitree's debug mode, on a gantry. In debug
  mode Unitree's own controller is released and this stack drives every motor
  on rt/lowcmd, from the moment golem_safety starts. Nothing in this stack
  enters debug mode or checks that the robot is in it.
- The gantry carries the robot whenever the legs are off: they hang limp until
  the leg node commands them, and split.yaml's estop turns every motor off.
- ROS_DOMAIN_ID=0 (the robot's DDS domain) in docker/.env.
- Nothing else may be commanding rt/lowcmd or rt/dex3/*: two publishers fight
  over the motors.
- Clear the space around the arms: once golem_safety relays the controller's
  commands, the arms move to `home` (forearms forward at waist height) at up to
  0.3 rad/s per joint.

golem_safety runs split.yaml: it writes every motor on rt/lowcmd, the legs
from the leg node's input and the waist and arms from the arm
controller's (g1_split.yaml). From boot it streams zero gains on every motor
(every motor off) and engages each input on its own first command: the arms on
the controller's first command, and the move to home starts once the
controller sees golem_safety relay it on rt/lowcmd.

The legs:
- start_position_verified gates the legs. Without it (the default) the leg
  node does not start: the leg motors stay off on golem_safety's zero-gain
  stream while the arms run. Set start_position_verified:=true only after
  checking the robot is in a safe start pose.
- With it, the leg node (golem_lowerbody_rl's lowerbody_controller_node,
  running HOMIE on its config/g1/real.yaml) waits for golem_safety's heartbeat
  and upright rt/lowstate, moves the legs from where they are to the policy's
  default pose over 2 s, and holds that pose. golem_safety engages the leg
  input on its first command.
- /lowerbody/confirm_engage starts the policy, once rt/lowstate has been
  steady for 1 s and the arm controller commands (the node logs why it waits).
  Check the stance, then
  `ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger`; until
  then the legs hold the default pose. The policy starts at zero velocity and
  0.74 m height and walks on /cmd_vel.
- A false or missing heartbeat (golem_safety's estop latched, or it stopped)
  stops the leg node for good. If the node stops on a stale or fallen state, a
  non-finite IMU value or a failed policy step, golem_safety keeps relaying
  its last leg command, which does not balance the robot, and reports the
  quiet input (input_quiet) 0.2 s later. It does not estop on it: the command
  stands until golem_safety estops for another reason (a fall that drives a
  joint past split.yaml's estop limits, say) or the bringup stops. A
  non-finite joint value (q, dq or tau_est) in rt/lowstate is different:
  golem_safety latches its estop on that sample, and split.yaml's estop (mode
  zero, keep_writing) turns every motor off and keeps writing that command.

This launch refuses sim configs (any sim_* safety_config), and the controller
config follows safety_config (golem_upper_ik's config/g1/pairing.yaml);
g1_core's docstring lists every setting it refuses. When the controller exits
it sends one last command, with zero gains on the arms; the waist keeps its
hold and golem_safety relays it, so only the arms go limp. The controller
stops with an error if rt/lowstate does not arrive within 5 s, if golem_safety
has not relayed its commands within 30 s, or if the move to home is not done
within 30 s after that. use_nav (default true) starts the MID-360 driver
(g1_real_drivers.launch.py) and golem_slam on config/g1/fastlio.yaml: FAST_LIO,
slam_toolbox and Nav2, which drives the leg node through /cmd_vel once it
runs. Unmeasured on the real G1: the driver and navigation have not run on it.
Skills are not part of the real G1 bringup.

Hazards not handled in code:
- A controller that dies without its last command leaves the waist + arms
  held on its last command: golem_safety holds a quiet input and only reports
  it (input_quiet), until the bringup stops.
- After any estop, restart the whole bringup, not only the safety node: the
  estop stays latched until restart, and the controller's follow gate opens
  for good once golem_safety relays its commands, so a restarted safety node
  would meet a command held wherever the robot used to be.
- The arm IK's self-collision model was packed on the rubber-hand G1: it has
  no spheres on the Dex3-1 fingers, which can hit the torso, the legs or the
  other arm. It keeps its spheres 20 mm apart (controller.d_min), as on the
  H1-2, but its torso spheres stand up to 19 mm inside the torso's flanks
  (CL_Assets/README.md): an upper arm rolled in toward the torso stops a few
  millimetres from it.
- The arms must start clear of the hips and the torso. In debug mode
  golem_safety's zero-gain stream leaves the arms limp in the gantry until the
  controller's first command, so the start-up begins wherever they hang. If a
  sphere pair starts about 10 mm apart or closer (half the keep-out; a hand or
  wrist in its hip-pitch sphere, or an upper arm rolled in against the torso),
  the start-up's first IK step has no solution and frame_task_server stops at
  once: it logs "the IK has no solution; closest self-collision pairs
  (controller.d_min 20 mm): ..." with each pair's distance, and its "did not
  start" error says how many of them are inside controller.d_min. Hanging
  freely in the sim, the wrists rest on the hips with their spheres 20 mm
  from the hip spheres; arms_down (elbows at pi/2) is inside them.
- The gains are unmeasured on the real G1: the waist's damping ratio is
  about 0.42-0.45 at home (g1_split.yaml).
"""

from golem_bringup.include import include
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Declare the real bringup's arguments and include g1_core with sim:=false."""
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='true: start RViz'),
        DeclareLaunchArgument('use_hands', default_value='true',
                              description='true: start dex3_hands, the Dex3-1 hand driver, '
                                          'which writes rt/dex3/* outside golem_safety'),
        DeclareLaunchArgument('safety_config', default_value='split',
                              description='golem_safety config, a name under its config/g1/ '
                                          'or a path'),
        # SAFETY: the leg-control interlock, false by default.
        DeclareLaunchArgument(
            'start_position_verified', default_value='false',
            description="true only after checking the robot is in a safe start pose: "
                        "starts the leg node, HOMIE on the legs"),
        DeclareLaunchArgument('use_nav', default_value='true',
                              description='true: start the MID-360 driver and golem_slam '
                                          '(FAST-LIO, slam_toolbox, Nav2)'),
        # The whole stack.
        include(
            'golem_bringup', 'g1/g1_core.launch.py',
            {
                'sim': 'false',
                'use_sim_time': 'false',
                'use_rviz': LaunchConfiguration('use_rviz'),
                'use_hands': LaunchConfiguration('use_hands'),
                'safety_config': LaunchConfiguration('safety_config'),
                # Empty: the one golem_upper_ik's config/g1/pairing.yaml pairs with
                # safety_config.
                'controller_config': '',
                'start_position_verified': LaunchConfiguration('start_position_verified'),
                'use_nav': LaunchConfiguration('use_nav'),
            },
        ),
    ])
