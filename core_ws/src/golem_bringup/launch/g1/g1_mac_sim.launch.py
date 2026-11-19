"""
Start the G1 against the Mac's RoboCasa sim: g1_sim without skills.

    ros2 launch golem_bringup g1_mac_sim.launch.py [use_hands:=false] [use_rviz:=false]

The Mac sim (docker/docker-compose.mac.yml, docs/MAC.md) renders on the CPU and casts its
MID-360 on the CPU only with GOLEM_LIDAR=1, so this includes g1_sim.launch.py with
use_skills false and use_nav as given: by default GOLEM_LIDAR (0 or 1, from
docker-compose.mac.yml), so navigation runs exactly when the sim has its lidar. Everything
else is g1_sim's: g1_core with sim:=true (golem_safety and the controller on sim_split,
the leg node running HOMIE, which releases the sim's elastic band once the policy runs),
the Dex3-1 hand driver (use_hands) and, with use_rviz, RViz.

Start the simulator first; every node joins the launching shell's ROS_DOMAIN_ID, which
must be a sim domain (safety_node refuses sim:=true on domain 0).
"""

from golem_bringup.include import include
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def generate_launch_description():
    """Declare the Mac bringup's arguments and include g1_sim without skills."""
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='true: start RViz (needs a display in golem_ros)'),
        DeclareLaunchArgument('use_hands', default_value='true',
                              description='true: start dex3_hands, the Dex3-1 hand driver'),
        DeclareLaunchArgument('use_nav', default_value=EnvironmentVariable('GOLEM_LIDAR',
                                                                           default_value='0'),
                              description="true or 1: golem_slam's FAST-LIO, SLAM and Nav2 "
                                          "on the sim's CPU-cast MID-360 (needs GOLEM_LIDAR=1 "
                                          'on the sim); default: GOLEM_LIDAR, else 0'),
        # The whole G1 sim bringup, without the parts the Mac cannot run.
        include('golem_bringup', 'g1/g1_sim.launch.py', {
            'use_rviz': LaunchConfiguration('use_rviz'),
            'use_hands': LaunchConfiguration('use_hands'),
            'use_nav': LaunchConfiguration('use_nav'),
            'use_skills': 'false',
        }),
    ])
