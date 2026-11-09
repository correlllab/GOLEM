r"""
Start the simulated G1: the core stack with sim time and the sim configs.

    ros2 launch golem_bringup g1_sim.launch.py
    ros2 launch golem_bringup g1_sim.launch.py use_nav:=false   # no SLAM or Nav2
    ros2 launch golem_bringup g1_sim.launch.py \
        use_skills:=true   # also serves /skill/* including /skill/frontier_explore

The G1 RoboCasa sim (golem_robocasa/g1/) takes rt/lowcmd and rt/dex3/*, and
holds the robot up on an elastic band until /elastic_band/toggle releases it.
golem_safety runs sim_split and the controller sim_split: golem_safety streams
zero gains on every motor from boot and engages each input on its own first
command. The leg node (golem_lowerbody_rl's lowerbody_controller_node) always
runs HOMIE, on golem_lowerbody_rl's config/g1/sim.yaml: once golem_safety beats
and rt/lowstate is upright, it moves the legs to the policy's default pose;
once the controller commands and has finished its start-up moves, it starts
the policy at zero velocity (no operator confirm in the sim) and releases the
band. It then walks on /cmd_vel. g1_core refuses the real robot's configs here
(its docstring lists what it refuses); the sim configs keep the faster arm
speeds (1 rad/s, 0.5 m/s).

use_nav (default true) adds golem_slam on the sim's MID-360 (/livox/lidar,
/livox/imu), with sim time and config/g1/fastlio_sim.yaml, and Nav2 drives the
leg node on /cmd_vel; send goals to /navigate_to_pose.
FAST-LIO is the odometry. Start the sim with GOLEM_SIM_ODOM unset;
otherwise odom -> pelvis has two publishers. use_skills:=true adds
golem_skills, which serves /skill/frontier_explore to walk the robot through
Nav2 to the frontiers of /map.
tests/g1/test_sim_nav.sh is the closed-loop test. Start the sim first; every node joins
the launching shell's ROS_DOMAIN_ID, which must be a sim domain (safety_node refuses
sim:=true on domain 0).
"""

from golem_bringup.include import include
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Declare the sim bringup's arguments and include g1_core with sim:=true."""
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='true: start RViz'),
        DeclareLaunchArgument('use_hands', default_value='true',
                              description='true: start dex3_hands, the Dex3-1 hand driver'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true: every node runs on the simulator /clock'),
        DeclareLaunchArgument(
            'safety_config', default_value='',
            description='golem_safety config, a name under its config/g1/ or a path '
                        '(default: sim_split)'),
        DeclareLaunchArgument(
            'use_nav', default_value='true',
            description="true: golem_slam's odometry, SLAM and Nav2 on /cmd_vel"),
        DeclareLaunchArgument(
            'use_skills', default_value='false',
            description='true: start the golem_skills action servers (/skill/*)'),
        # The whole stack.
        include(
            'golem_bringup', 'g1/g1_core.launch.py',
            {
                'sim': 'true',
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'use_rviz': LaunchConfiguration('use_rviz'),
                'use_hands': LaunchConfiguration('use_hands'),
                'safety_config': LaunchConfiguration('safety_config'),
                # Empty: the one golem_upper_ik's config/g1/pairing.yaml pairs with
                # safety_config.
                'controller_config': '',
                'use_nav': LaunchConfiguration('use_nav'),
                'use_skills': LaunchConfiguration('use_skills'),
            },
        ),
    ])
