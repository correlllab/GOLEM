import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# x86 sim bringup. ROS_DOMAIN_ID must be exported in the launching shell (see
# h1_control.launch.py).
ASSETS_DIR = '/home/code/CL_Assets'


def _include(package, launch_file, launch_arguments, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    bringup_share = get_package_share_directory('h1_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'sim.rviz')

    # MuJoCo publishes /clock with sim time. All nodes use it so TF lookups and
    # sensor timestamps are coherent with the simulation.
    sim_time = {'use_sim_time': 'true'}

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_sliders', default_value='true'),
        DeclareLaunchArgument('use_nav', default_value='true'),
        DeclareLaunchArgument('use_skills', default_value='true'),
        DeclareLaunchArgument('record_runs', default_value='true'),
        # Which RL policy drives the legs (see h1_lowerbody.launch.py). The sim
        # bringup does not launch the MJPC controller, so lowerbody:=mjpc
        # leaves the legs uncontrolled.
        DeclareLaunchArgument('lowerbody', default_value='almi'),  # almi | almi27 | fame | walk
        DeclareLaunchArgument('lowerbody_auto_switch', default_value='false'),
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),

        _include('h1_bringup', 'h1_state.launch.py', {
            **sim_time,
            'urdf_file': os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'),
        }),
        _include('h1_bringup', 'h1_control.launch.py', {
            **sim_time,
            'safety_config': 'sim_safety_split.yaml',
            'frame_task_config': 'sim_safety_split.yaml',
            # No estop in sim, so nothing to wait for.
            'safety_delay': '0.0',
            'frame_task_delay': '0.0',
        }),
        _include('h1_bringup', 'h1_lowerbody.launch.py', {
            **sim_time,
            'start_position_verified': 'true',
            'lowerbody': LaunchConfiguration('lowerbody'),
            'lowerbody_auto_switch': LaunchConfiguration('lowerbody_auto_switch'),
            'lowerbody_params': os.path.join(bringup_share, 'config', 'lowerbody_sim.yaml'),
        }),
        _include('h1_bringup', 'h1_models.launch.py', {
            **sim_time,
            'use_skills': LaunchConfiguration('use_skills'),
            'model_logging': LaunchConfiguration('model_logging'),
            'model_visualization': LaunchConfiguration('model_visualization'),
            'model_clear_logs': LaunchConfiguration('model_clear_logs'),
            'record_runs': LaunchConfiguration('record_runs'),
        }),
        _include('h12_slam', 'h1_navigation.launch.py', sim_time,
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
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_sim',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': True}],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_rviz')),
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
        #             condition=IfCondition(LaunchConfiguration('use_sliders')),
        #         ),
        #     ],
        # ),
    ])
