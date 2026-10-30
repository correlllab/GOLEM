import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


# Minimal real-robot actuation bringup: estop + safety layer + both grippers.
# No lidar, no cameras, no state publishers, no arm IK -- just the pieces that
# have to be alive for a command to reach a motor safely. Use it when working on
# gripper/low-level actuation without paying for the full driver stack; the full
# onboard bringup (h12_real_robot.launch.py) is still the normal path.
# Every node joins the launching shell's ROS_DOMAIN_ID, which must be 0, the real
# robot's command bus; ROS uses 0 when it is unset (see h12_control.launch.py).


def generate_launch_description():
    bringup_share = get_package_share_directory('golem_bringup')

    # Same default config as the full real bringup; overridable so a session
    # can tighten/relax limits without editing launch.
    safety_config_arg = DeclareLaunchArgument(
        'safety_config',
        default_value='relax_safety_split.yaml',
        description='golem_safety config: a name under its config/h12/, or a path',
    )

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
