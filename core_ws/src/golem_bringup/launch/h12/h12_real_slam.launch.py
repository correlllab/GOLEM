"""
Start the real H1-2's sensors and SLAM/navigation only, on the wall clock.

    ros2 launch golem_bringup h12_real_slam.launch.py

The includes start at once: the sensor drivers and grippers (h12_real_drivers), the
state publishers (h12_state) and golem_slam's navigation. No estop, safety layer or IK
starts, so no joint is commanded. Export the robot's ROS_DOMAIN_ID (0) in the launching
shell. Record runs with ros2 run golem_slam record_slam_bag --name <name>.
"""

from golem_bringup.include import include
from launch import LaunchDescription


def generate_launch_description():
    """Include the drivers, the state publishers and navigation."""
    # The real robot runs on the wall clock: there is no /clock.
    wall_time = {'use_sim_time': 'false'}

    return LaunchDescription([
        # The MID-360, the head RealSense and the grippers.
        include('golem_bringup', 'h12/h12_real_drivers.launch.py'),
        # TF of the body from /lowstate, which golem_slam's body_to_pelvis needs.
        include('golem_bringup', 'h12/h12_state.launch.py', wall_time),
        # FAST-LIO, OctoMap, slam_toolbox and Nav2.
        include('golem_slam', 'navigation.launch.py', {**wall_time, 'robot': 'h12'}),
    ])
