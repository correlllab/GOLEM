"""
Start the real H1-2's companion-desktop half: model servers, skills, RViz, hand cameras.

    ros2 launch golem_bringup h12_real_desktop.launch.py [use_hand_cameras:=true]

All start at once, on the wall clock: the hand RealSense cameras (use_hand_cameras), the
model servers with YOLO and golem_skills (use_skills), and RViz. The state publishers,
safety layer, frame_task_server and leg control run on the onboard PC
(h12_real_robot.launch.py). Export the robot's ROS_DOMAIN_ID (0) in the launching shell.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _include(package, launch_file, launch_arguments=None, condition=None):
    """Include a package's launch file with the given arguments."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=(launch_arguments or {}).items(),
        condition=condition,
    )


def generate_launch_description():
    """Declare the desktop bringup's arguments and start its nodes."""
    bringup_share = get_package_share_directory('golem_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'h12', 'main.rviz')

    # The real robot runs on the wall clock: there is no /clock.
    wall_time = {'use_sim_time': 'false'}

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_hand_cameras', default_value='false',
            description='true: start both hand RealSense cameras and their static TFs'),
        DeclareLaunchArgument('use_skills', default_value='true',
                              description='true: start graspgen_server and golem_skills '
                                          '(/skill/*)'),
        DeclareLaunchArgument('model_logging', default_value='true',
                              description="true: the model servers and golem_skills save "
                                          "each call's inputs and outputs (enable_logging)"),
        DeclareLaunchArgument('model_visualization', default_value='true',
                              description='true: the model servers and golem_skills save '
                                          'annotated overlay images (enable_visualization)'),
        DeclareLaunchArgument('model_clear_logs', default_value='true',
                              description="true: each model's log directory is wiped at "
                                          'start-up (clear_logs)'),
        DeclareLaunchArgument('record_runs', default_value='true',
                              description='true: golem_skills records each grasp and '
                                          'pick_place run under its logs/runs/<skill>/'),

        # Both hand RealSense cameras and their camera -> link static TFs. They
        # run here because the wrists' USB cables do; the head camera starts
        # with the onboard drivers (h12_real_drivers.launch.py).
        _include('cl_realsense', 'h12_hand_cameras.launch.py',
                 condition=IfCondition(LaunchConfiguration('use_hand_cameras'))),

        # Perception servers, YOLO and the LLM-callable skills.
        _include('golem_bringup', 'h12/h12_models.launch.py', {
            **wall_time,
            'use_yolo': 'true',
            'use_skills': LaunchConfiguration('use_skills'),
            'model_logging': LaunchConfiguration('model_logging'),
            'model_visualization': LaunchConfiguration('model_visualization'),
            'model_clear_logs': LaunchConfiguration('model_clear_logs'),
            'record_runs': LaunchConfiguration('record_runs'),
        }),

        # The operator's view, and where Nav2 goals are set.
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', default_rviz],
            parameters=[{'use_sim_time': False}],
            output='screen',
            emulate_tty=True,
        ),
    ])
