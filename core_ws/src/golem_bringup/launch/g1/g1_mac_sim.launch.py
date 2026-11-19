"""
Start the G1 against the Mac's RoboCasa sim: g1_sim without skills or the hand driver.

    ros2 launch golem_bringup g1_mac_sim.launch.py [use_hands:=true] [use_rviz:=false]

The Mac sim (docker/docker-compose.mac.yml, docs/MAC.md) renders on the CPU and casts its
MID-360 on the CPU, and an 8-core Mac holds real time only without the Dex3-1 hand
driver, so this includes g1_sim.launch.py with use_skills false and use_hands false by
default. Everything else is g1_sim's: g1_core with sim:=true (golem_safety and the
controller on sim_split, the leg node running HOMIE, which releases the sim's elastic band
once the policy runs), navigation (golem_slam on FAST-LIO) and, with use_rviz, RViz.

Start the simulator first; every node joins the launching shell's ROS_DOMAIN_ID, which
must be a sim domain (safety_node refuses sim:=true on domain 0).
"""

from golem_bringup.include import include
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Declare the Mac bringup's arguments and include g1_sim without skills."""
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='true: start RViz, on the Mac golem_ros display '
                                          '(http://localhost:6080/vnc.html); false: none'),
        DeclareLaunchArgument('use_hands', default_value='false',
                              description='true: start dex3_hands, the Dex3-1 hand driver '
                                          '(about a quarter of a core: the stack then falls '
                                          'below real time on an 8-core Mac)'),
        # The whole G1 sim bringup, without the parts the Mac cannot run.
        include('golem_bringup', 'g1/g1_sim.launch.py', {
            'use_rviz': LaunchConfiguration('use_rviz'),
            'use_hands': LaunchConfiguration('use_hands'),
            'use_skills': 'false',
        }),
    ])
