'''
Show one robot's model in RViz with joint sliders.

    ros2 launch golem_ros2_models display.launch.py              # the H1-2
    ros2 launch golem_ros2_models display.launch.py robot:=g1    # the G1
'''
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from golem_ros2_models.paths import ROBOTS, urdf_path


def launch_setup(context):
    robot = LaunchConfiguration('robot').perform(context)
    with open(urdf_path(robot, ros=True)) as f:
        robot_description = f.read()
    rviz_config = os.path.join(get_package_share_directory('golem_ros2_models'), 'rviz',
                               'model.rviz')
    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot', default_value=ROBOTS[0], choices=list(ROBOTS),
                              description='h12: the H1-2 with Magpie grippers; '
                                          'g1: the G1 with Dex3-1 hands'),
        OpaqueFunction(function=launch_setup),
    ])
