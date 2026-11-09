"""
Start the whole H1-2 against the RoboCasa or Isaac sim, on sim time.

    ros2 launch golem_bringup h12_sim.launch.py [use_rviz:=false] [use_nav:=false] ...

Start the simulator first: every node runs on its /clock. The leg node's check
(golem_bringup.h12_legs.leg_check) runs first, so a refused policy starts no node; then
everything starts at once: the state publishers (h12_state), the command path
(h12_control: the safety/controller pairing check before any node, then safety_node and
frame_task_server, with no estop and no stagger), the leg controller (h12_lowerbody, with
the start pose taken as verified), the model servers and golem_skills (h12_models),
navigation (use_nav), the static camera_link -> camera_color_optical_frame TF and RViz
(use_rviz).

Every node joins the launching shell's ROS_DOMAIN_ID; safety_node refuses sim:=true on
domain 0, the real robot's command bus, so export a sim domain first.
"""
import os

from ament_index_python.packages import get_package_share_directory
from golem_bringup.h12_legs import leg_check
from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _include(package, launch_file, launch_arguments, condition=None):
    """Include a package's launch file with the given arguments."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    """Declare the sim bringup's arguments and include its blocks."""
    bringup_share = get_package_share_directory('golem_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'h12', 'main.rviz')

    # The simulator publishes /clock; every node uses it so TF lookups and sensor
    # stamps agree with the simulation.
    sim_time = {'use_sim_time': 'true'}
    # The legs; the sim needs no operator check of the start pose.
    legs = {
        **sim_time,
        'start_position_verified': 'true',
        'policy': LaunchConfiguration('policy'),
        'sim': 'true',
        'lowerbody_config': 'sim',
    }

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='true: start RViz on rviz_config'),
        DeclareLaunchArgument('use_nav', default_value='true',
                              description="true: start golem_slam's navigation (FAST-LIO, "
                                          'OctoMap, slam_toolbox, Nav2) for the H1-2'),
        DeclareLaunchArgument('use_skills', default_value='true',
                              description='true: start graspgen_server and golem_skills '
                                          '(/skill/*)'),
        DeclareLaunchArgument('record_runs', default_value='true',
                              description="true: golem_skills records each grasp and "
                                          "pick_place run under its logs/runs/<skill>/"),
        DeclareLaunchArgument('policy', default_value='almi',
                              description='The policy golem_lowerbody_rl runs on the legs: '
                                          'almi, fame or walk. mjpc starts no leg node and '
                                          'no bringup starts the MJPC controller, so it '
                                          'leaves the legs uncommanded'),
        DeclareLaunchArgument('model_logging', default_value='true',
                              description="true: the model servers and golem_skills save "
                                          "each call's inputs and outputs (enable_logging)"),
        DeclareLaunchArgument('model_visualization', default_value='true',
                              description='true: the model servers and golem_skills save '
                                          'annotated overlay images (enable_visualization)'),
        DeclareLaunchArgument('model_clear_logs', default_value='true',
                              description="true: each model's log directory is wiped at "
                                          'start-up (clear_logs)'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz,
                              description='RViz config file (path)'),

        # SAFETY: first, so a refused policy or run config starts no node: the leg
        # block, which checks them too, is included after the others.
        leg_check(legs),
        # TF of the body from /lowstate.
        _include('golem_bringup', 'h12/h12_state.launch.py', {
            **sim_time,
            'urdf_file': urdf_path('h12', ros=True),
        }),
        # The command path: safety_node, the only writer of /lowcmd, and the arm IK.
        _include('golem_bringup', 'h12/h12_control.launch.py', {
            **sim_time,
            'sim': 'true',
            'safety_config': 'sim_safety_split.yaml',
            'frame_task_config': 'sim_safety_split.yaml',
            # No estop in the sim, so nothing to wait for.
            'safety_delay': '0.0',
            'frame_task_delay': '0.0',
        }),
        # The legs.
        _include('golem_bringup', 'h12/h12_lowerbody.launch.py', legs),
        # Perception servers and the LLM-callable skills.
        _include('golem_bringup', 'h12/h12_models.launch.py', {
            **sim_time,
            'use_skills': LaunchConfiguration('use_skills'),
            'model_logging': LaunchConfiguration('model_logging'),
            'model_visualization': LaunchConfiguration('model_visualization'),
            'model_clear_logs': LaunchConfiguration('model_clear_logs'),
            'record_runs': LaunchConfiguration('record_runs'),
        }),
        # SLAM and Nav2, driving the legs on /cmd_vel.
        _include('golem_slam', 'navigation.launch.py', {**sim_time, 'robot': 'h12'},
                 condition=IfCondition(LaunchConfiguration('use_nav'))),

        # For Isaac: its head camera's images are stamped
        # camera_color_optical_frame (CL_isaaclab_sim's ros_sensor_bridge.py,
        # REP-103 optical axes: +z forward, +x right, +y down) and it publishes
        # no TF, so this rotates that frame under camera_link (+x forward,
        # +z up). The URDF loaded here has no camera_link (its head camera is
        # head_camera_link), so the frame joins the robot's tree only where
        # something publishes camera_link. RoboCasa stamps
        # head_color_optical_frame and publishes its own camera TF.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_optical_frame_broadcaster',
            arguments=['0', '0', '0',
                       '-1.5707963267948966', '0', '-1.5707963267948966',
                       'camera_link', 'camera_color_optical_frame'],
            parameters=[{'use_sim_time': True}],
            output='screen',
            emulate_tty=True,
        ),

        # The operator's view, and where Nav2 goals are set.
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_sim',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': True}],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ])
