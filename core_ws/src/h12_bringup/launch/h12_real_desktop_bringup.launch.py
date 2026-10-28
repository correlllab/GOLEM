import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Companion-desktop bringup: model servers + skills, rviz, and optionally the
# hand cameras (use_hand_cameras).
# The state publishers, safety layer, frame_task_server and lower-body control
# run on the onboard PC (h12_real_robot_bringup.launch.py). ROS_DOMAIN_ID must
# be exported in the launching shell (see h12_control.launch.py).


def _include(package, launch_file, launch_arguments=None, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=(launch_arguments or {}).items(),
        condition=condition,
    )


def generate_launch_description():
    bringup_share = get_package_share_directory('h12_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'main.rviz')

    # Real robot runs on wall-clock time (no simulator /clock).
    wall_time = {'use_sim_time': 'false'}

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_hand_cameras', default_value='false',
            description='Start both hand RealSense cameras and their static TFs'),
        DeclareLaunchArgument('use_skills', default_value='true'),
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),
        DeclareLaunchArgument('record_runs', default_value='true'),

        # Both hand/wrist RealSense cameras and their camera->link static TFs.
        # These live on the companion desktop (the wrists' USB runs here), while
        # the head camera comes up with the onboard driver bringup
        # (h12_real_drivers.launch.py).
        _include('cl_realsense', 'h12_hand_cameras.launch.py',
                 condition=IfCondition(LaunchConfiguration('use_hand_cameras'))),

        _include('h12_bringup', 'h12_models.launch.py', {
            **wall_time,
            'use_yolo': 'true',
            'use_skills': LaunchConfiguration('use_skills'),
            'model_logging': LaunchConfiguration('model_logging'),
            'model_visualization': LaunchConfiguration('model_visualization'),
            'model_clear_logs': LaunchConfiguration('model_clear_logs'),
            'record_runs': LaunchConfiguration('record_runs'),
        }),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_sim',
            arguments=['-d', default_rviz],
            parameters=[{'use_sim_time': False}],
            output='screen',
            emulate_tty=True,
        ),

        # slider_debugger waits up to 5s on /left_ee_pose & /right_ee_pose,
        # which frame_task_server publishes only after its IK solver finishes
        # initialising (URDF load + 150-step torso init — empirically ~7s).
        # 10s leaves headroom so the sliders seed from the live pose.
        #
        # Intentionally NOT using sim_time: the GUI's wait_for_initial_poses
        # measures wall-clock; with use_sim_time=True a fast sim that's
        # already past 5s makes get_clock().now() jump and trip the timeout
        # immediately, falling back to all-zero targets that drive the IK
        # toward unreachable poses inside the body.
        # TimerAction(
        #     period=1.0,
        #     actions=[
        #         Node(
        #             package='h12_bringup',
        #             executable='slider_debugger.py',
        #             name='slider_debugger',
        #             output='screen',
        #         ),
        #     ],
        # ),
    ])
