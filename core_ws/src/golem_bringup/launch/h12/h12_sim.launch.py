import os

from ament_index_python.packages import get_package_share_directory
from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# x86 sim bringup. ROS_DOMAIN_ID must be exported in the launching shell (see
# h12_control.launch.py).


def _include(package, launch_file, launch_arguments, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    bringup_share = get_package_share_directory('golem_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'h12', 'main.rviz')

    # MuJoCo publishes /clock with sim time. All nodes use it so TF lookups and
    # sensor timestamps are coherent with the simulation.
    sim_time = {'use_sim_time': 'true'}

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_sliders', default_value='true'),
        DeclareLaunchArgument('use_nav', default_value='true'),
        DeclareLaunchArgument('use_skills', default_value='true'),
        DeclareLaunchArgument('record_runs', default_value='true'),
        # Which RL policy drives the legs (see h12_lowerbody.launch.py). The sim
        # bringup does not launch the MJPC controller, so lowerbody:=mjpc
        # leaves the legs uncontrolled.
        DeclareLaunchArgument('lowerbody', default_value='almi'),  # almi | fame | walk
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),

        _include('golem_bringup', 'h12/h12_state.launch.py', {
            **sim_time,
            'urdf_file': urdf_path('h12', ros=True),
        }),
        _include('golem_bringup', 'h12/h12_control.launch.py', {
            **sim_time,
            'sim': 'true',
            'safety_config': 'sim_safety_split.yaml',
            'frame_task_config': 'sim_safety_split.yaml',
            # No estop in sim, so nothing to wait for.
            'safety_delay': '0.0',
            'frame_task_delay': '0.0',
        }),
        _include('golem_bringup', 'h12/h12_lowerbody.launch.py', {
            **sim_time,
            'start_position_verified': 'true',
            'lowerbody': LaunchConfiguration('lowerbody'),
            'lowerbody_params': os.path.join(bringup_share, 'config', 'h12', 'lowerbody_sim.yaml'),
        }),
        _include('golem_bringup', 'h12/h12_models.launch.py', {
            **sim_time,
            'use_skills': LaunchConfiguration('use_skills'),
            'model_logging': LaunchConfiguration('model_logging'),
            'model_visualization': LaunchConfiguration('model_visualization'),
            'model_clear_logs': LaunchConfiguration('model_clear_logs'),
            'record_runs': LaunchConfiguration('record_runs'),
        }),
        _include('h12_slam', 'h12_navigation.launch.py', sim_time,
                 condition=IfCondition(LaunchConfiguration('use_nav'))),

        # The MuJoCo bridge back-projects depth into 3D using REP-103 optical
        # convention (+z = forward, +x = right, +y = down) but stamps the
        # resulting camera_info / image / depth messages with the optical
        # frame name below. The URDF defines only camera_link (a ROS link
        # frame, +x = forward, +z = up); without this static TF the vision
        # pipeline would treat optical-convention points as if they were in
        # camera_link, rotating every detection ~90 deg out of place.
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
