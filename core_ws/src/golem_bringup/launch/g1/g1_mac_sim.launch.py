"""
Start the G1 against the Mac's RoboCasa sim: g1_sim without navigation or skills.

    ros2 launch golem_bringup g1_mac_sim.launch.py [use_hands:=false] [use_rviz:=false]

The Mac sim (docker/docker-compose.mac.yml, docs/MAC.md) renders on the CPU and has no
MID-360, so this includes g1_sim.launch.py with use_nav and use_skills false. Everything
else is g1_sim's: g1_core with sim:=true (golem_safety and the controller on sim_split,
the leg node running HOMIE, which releases the sim's elastic band once the policy runs),
the Dex3-1 hand driver (use_hands) and, with use_rviz, RViz.

Start the simulator first; every node joins the launching shell's ROS_DOMAIN_ID, which
must be a sim domain (safety_node refuses sim:=true on domain 0).
"""

from golem_bringup.include import include
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Declare the Mac bringup's arguments and include g1_sim without nav or skills."""
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='true: start RViz (needs a display in golem_ros)'),
        DeclareLaunchArgument('use_hands', default_value='true',
                              description='true: start dex3_hands, the Dex3-1 hand driver'),
        # The whole G1 sim bringup, without the parts the Mac cannot run.
        include('golem_bringup', 'g1/g1_sim.launch.py', {
            'use_rviz': LaunchConfiguration('use_rviz'),
            'use_hands': LaunchConfiguration('use_hands'),
            'use_nav': 'false',
            'use_skills': 'false',
        }),
    ])
