import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Companion-desktop bringup: hand cameras, model servers + skills, lower-body
# control, and rviz. The state publishers, safety layer and frame_task_server
# run on the onboard PC (h1_real_robot_bringup.launch.py). ROS_DOMAIN_ID must
# be exported in the launching shell (see h1_control.launch.py).


def _include(package, launch_file, launch_arguments=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=(launch_arguments or {}).items(),
    )


def generate_launch_description():
    bringup_share = get_package_share_directory('h1_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'sim.rviz')

    # Real robot runs on wall-clock time (no simulator /clock).
    wall_time = {'use_sim_time': 'false'}

    return LaunchDescription([
        # Leg control engages only once the operator has verified the robot is
        # in a safe start position. Defaults to false so a bare launch never
        # auto-commands the legs on the real robot.
        DeclareLaunchArgument('start_position_verified', default_value='false'),
        # Which RL policy drives the legs once start_position_verified:=true
        # (see h1_lowerbody.launch.py); the RL node additionally holds at the
        # pre-pose crouch until the operator confirms:
        #   ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger
        DeclareLaunchArgument('lowerbody', default_value='almi'),
        DeclareLaunchArgument('lowerbody_auto_switch', default_value='false'),
        DeclareLaunchArgument('use_skills', default_value='true'),
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),
        DeclareLaunchArgument('record_runs', default_value='true'),

        # Both hand/wrist RealSense cameras and their camera->link static TFs.
        # These live on the companion desktop (the wrists' USB runs here), while
        # the head camera comes up with the onboard driver bringup
        # (h1_real_drivers.launch.py).
        _include('cl_realsense', 'h12_hand_cameras.launch.py'),

        _include('h1_bringup', 'h1_models.launch.py', {
            **wall_time,
            'use_yolo': 'true',
            'use_skills': LaunchConfiguration('use_skills'),
            'model_logging': LaunchConfiguration('model_logging'),
            'model_visualization': LaunchConfiguration('model_visualization'),
            'model_clear_logs': LaunchConfiguration('model_clear_logs'),
            'record_runs': LaunchConfiguration('record_runs'),
        }),
        _include('h1_bringup', 'h1_lowerbody.launch.py', {
            **wall_time,
            'start_position_verified': LaunchConfiguration('start_position_verified'),
            'lowerbody': LaunchConfiguration('lowerbody'),
            'lowerbody_auto_switch': LaunchConfiguration('lowerbody_auto_switch'),
            'lowerbody_params': os.path.join(bringup_share, 'config', 'lowerbody_real.yaml'),
            'use_estimator': 'true',
            'estimator_params': os.path.join(bringup_share, 'config', 'mjpc_real.yaml'),
        }),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_sim',
            arguments=['-d', default_rviz],
            parameters=[{'use_sim_time': False}],
            output='screen',
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
        #             package='h1_bringup',
        #             executable='slider_debugger.py',
        #             name='slider_debugger',
        #             output='screen',
        #         ),
        #     ],
        # ),
    ])
