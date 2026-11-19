"""
Start the H1-2 against the Mac's RoboCasa sim: h12_sim without model servers or skills.

    ros2 launch golem_bringup h12_mac_sim.launch.py [policy:=almi] [use_rviz:=false]

The Mac sim (docker/docker-compose.mac.yml, docs/MAC.md) renders on the CPU and casts its
MID-360 on the CPU only with GOLEM_LIDAR=1, and the Mac's golem_ros image has no
model-server stack, so this includes h12_sim.launch.py with use_vision and use_skills false
and use_nav as given: by default GOLEM_LIDAR (0 or 1, from docker-compose.mac.yml), so
navigation runs exactly when the sim has its lidar. Everything else is
h12_sim's, in its order: the leg node's check first, then the state publishers, the
command path (estop_node on the sim's estop box, safety_node, frame_task_server), the
legs and, with use_rviz, RViz.

SAFETY: the operator gates are h12_sim's: the sim's estop box (ENTER on the sim's
terminal; golem_safety then latches until restart), and the leg policy holds its
pre-pose until
    ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger

Start the simulator first; every node joins the launching shell's ROS_DOMAIN_ID, which
must be a sim domain (safety_node refuses sim:=true on domain 0).
"""

from golem_bringup.include import include
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def generate_launch_description():
    """Declare the Mac bringup's arguments and include h12_sim without vision."""
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='true: start RViz (needs a display in golem_ros)'),
        DeclareLaunchArgument('policy', default_value='almi',
                              description='The policy golem_lowerbody_rl runs on the legs: '
                                          'almi, fame or walk; any other is refused'),
        DeclareLaunchArgument('use_nav', default_value=EnvironmentVariable('GOLEM_LIDAR',
                                                                           default_value='0'),
                              description="true or 1: golem_slam's FAST-LIO, SLAM and Nav2 "
                                          "on the sim's CPU-cast MID-360 (needs GOLEM_LIDAR=1 "
                                          'on the sim); default: GOLEM_LIDAR, else 0'),
        # The whole H1-2 sim bringup, without the parts the Mac cannot run.
        include('golem_bringup', 'h12/h12_sim.launch.py', {
            'use_rviz': LaunchConfiguration('use_rviz'),
            'policy': LaunchConfiguration('policy'),
            'use_nav': LaunchConfiguration('use_nav'),
            'use_vision': 'false',
            'use_skills': 'false',
        }),
    ])
