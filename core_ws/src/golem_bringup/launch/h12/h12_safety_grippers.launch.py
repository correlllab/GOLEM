"""
Start the real H1-2's actuation only: estop, safety layer and both grippers.

    ros2 launch golem_bringup h12_safety_grippers.launch.py [safety_config:=<config>]

Both includes start at once, on the wall clock: h12_control with the hardware estop and
no frame_task_server (estop_node, then safety_node after its safety_delay), and
h12_grippers. No lidar, cameras, state publishers or arm IK: the pieces a command needs
to reach a motor safely, for gripper and low-level actuation work. h12_real_robot is the
normal bringup. Every node joins the launching shell's ROS_DOMAIN_ID, which must be 0,
the real robot's command bus; ROS uses 0 when it is unset.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Declare safety_config and include the command path and the grippers."""
    bringup_share = get_package_share_directory('golem_bringup')

    # The full real bringup's default, overridable so a session can tighten or
    # relax the limits without editing a launch file.
    safety_config_arg = DeclareLaunchArgument(
        'safety_config',
        default_value='relax_safety_split.yaml',
        description='golem_safety config: a name under its config/h12/, or a path. No '
                    'controller starts here (use_frame_task is false), so no controller '
                    'pairing applies',
    )

    # The command path with the hardware estop and no arm IK.
    control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h12', 'h12_control.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'sim': 'false',
            'use_estop': 'true',
            'use_frame_task': 'false',
            'safety_config': LaunchConfiguration('safety_config'),
        }.items(),
    )

    # Both Magpie grippers.
    grippers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h12', 'h12_grippers.launch.py')
        ),
    )

    return LaunchDescription([
        safety_config_arg,
        control,
        grippers,
    ])
