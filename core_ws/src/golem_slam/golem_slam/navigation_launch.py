"""
Build golem_slam's navigation graph, one for every robot (launch/navigation.launch.py).

The robot argument picks a config directory, config/<robot>/: FAST-LIO
(fastlio.yaml), Nav2 (nav2.yaml), OctoMap (octomap.yaml) and the pipeline
profile (pipeline.yaml: the lidar link, the scan's height band and level_odom's
values). The nodes, their executables and remaps are the same on every robot.

Every node starts at once, with no delays or conditions; each waits for its inputs
(TF, clouds, the map) itself. Data flows: MID-360 -> FAST-LIO -> self_filter ->
level_odom, OctoMap and level_cloud -> pointcloud_to_laserscan -> slam_toolbox, and
Nav2 on OctoMap's /projected_map and the scan. The launch stops before any node
starts when the robot has no config directory or its pipeline.yaml lacks a key.
It needs the robot's state (robot_state_publisher with joint states) from the
including bringup.
"""
import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from nav2_common.launch import RewrittenYaml

from golem_slam.body_to_pelvis import LIVOX_FRAME, static_transform_arguments, transform

# pipeline.yaml's keys, every one required.
PIPELINE_KEYS = (('lidar_link',), ('scan', 'min_height'), ('scan', 'max_height'),
                 ('level_odom', 'min_floor_depth'))


def _config_dir():
    """Return the installed config directory."""
    return os.path.join(get_package_share_directory('golem_slam'), 'config')


def load_pipeline(path):
    """Return the pipeline profile at path, refusing one that lacks a key."""
    with open(path) as source:
        profile = yaml.safe_load(source) or {}
    for key in PIPELINE_KEYS:
        value = profile
        for part in key:
            if not isinstance(value, dict) or part not in value:
                raise RuntimeError(f'golem_slam: {path} has no {".".join(key)}')
            value = value[part]
    return profile


def _pipeline(context):
    """Return the graph's nodes and the Nav2 include for the robot argument."""
    robot = LaunchConfiguration('robot').perform(context)
    robot_config = os.path.join(_config_dir(), robot)
    if not robot or not os.path.isdir(robot_config):
        robots = sorted(d for d in os.listdir(_config_dir())
                        if os.path.isdir(os.path.join(_config_dir(), d)))
        raise RuntimeError(f'golem_slam: no config directory for robot {robot!r} '
                           f'(config/<robot>/ exists for {", ".join(robots)})')
    profile = load_pipeline(os.path.join(robot_config, 'pipeline.yaml'))
    use_sim_time = LaunchConfiguration('use_sim_time')
    fast_lio_config = LaunchConfiguration('fast_lio_config').perform(context)

    # The driver and the sims stamp the lidar points and the IMU 'livox_frame', the
    # raw sensor frame, which the URDF's lidar link is; this joins it to the TF tree.
    livox_frame_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='livox_frame_tf',
        arguments=static_transform_arguments(transform(), profile['lidar_link'], LIVOX_FRAME),
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )
    # LiDAR-inertial odometry: camera_init -> body, /Odometry and the deskewed
    # body-frame cloud /cloud_registered_body.
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        name='fast_lio',
        parameters=[fast_lio_config, {'use_sim_time': use_sim_time}],
        output='screen',
    )
    # FAST-LIO publishes camera_init -> body, body being its IMU on the torso; this
    # hangs the pelvis off it at the measured torso joints (golem_slam/body_to_pelvis.py).
    body_to_pelvis_node = Node(
        package='golem_slam',
        executable='body_to_pelvis',
        name='body_to_pelvis',
        parameters=[{'fastlio_config': fast_lio_config, 'lidar_frame': LIVOX_FRAME,
                     'use_sim_time': use_sim_time}],
        output='screen',
    )

    # Gravity-aligned, floor-referenced odom -> camera_init (static, computed
    # once at startup: up from the IMU while the robot stands still, floor
    # height from the floor plane in the self-filtered scan), then
    # odom -> base_footprint (pelvis on the floor, yaw only) per FAST-LIO
    # odometry. Costmap and OctoMap height bands are in odom/map, so they need
    # it; see golem_slam/level_odom.py.
    level_odom_node = Node(
        package='golem_slam',
        executable='level_odom',
        name='level_odom',
        output='screen',
        parameters=[profile['level_odom'], {'use_sim_time': use_sim_time}],
        remappings=[('imu', '/livox/imu'), ('cloud', '/cloud_body_filtered'),
                    ('odometry', '/Odometry')],
    )

    # Drops the robot's own body (URDF collision boxes posed by TF) from the
    # deskewed FAST-LIO scan, so the maps never mark the robot. Everything
    # downstream (OctoMap, costmaps, slam_toolbox's scan) reads /cloud_body_filtered.
    self_filter_node = Node(
        package='golem_slam',
        executable='self_filter',
        name='self_filter',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('cloud_in', '/cloud_registered_body'),
            ('cloud_out', '/cloud_body_filtered'),
        ],
    )

    # The scan's input in base_footprint (level and on the floor, so body sway
    # does not tilt the slice into the floor), transformed without a TF message
    # filter (golem_slam/level_cloud.py).
    level_cloud_node = Node(
        package='golem_slam',
        executable='level_cloud',
        name='level_cloud',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[('cloud_in', '/cloud_body_filtered'),
                    ('cloud_out', '/cloud_base_filtered')],
        output='screen',
    )

    # 3D occupancy (RViz: /occupied_cells_vis_array) and its 2D projection
    # /projected_map, the static layer of both Nav2 costmaps.
    octomap_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        parameters=[LaunchConfiguration('octomap_params_file'),
                    {'use_sim_time': use_sim_time}],
        remappings=[('cloud_in', '/cloud_body_filtered')],
    )

    # 2D scan for slam_toolbox and the local costmap obstacle layer.
    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='cloud_registered_to_laserscan',
        output='screen',
        parameters=[
            # No target frame: level_cloud already puts /cloud_base_filtered in base_footprint.
            {'target_frame': ''},
            {'use_sim_time': use_sim_time},
            # m above the floor: clears floor returns and FAST-LIO z drift,
            # and stays below the robot's height.
            {'min_height': profile['scan']['min_height']},
            {'max_height': profile['scan']['max_height']},
            {'angle_min': -3.14159},  # rad: the MID-360's full 360 deg
            {'angle_max': 3.14159},
            {'angle_increment': 0.0087},  # rad, 0.5 deg
            # m. The self filter already removes the body; range_min only drops
            # the MID-360's near-field noise. range_max is nav2.yaml's raytrace range.
            {'range_min': 0.3},
            {'range_max': 3.0},
            {'use_inf': True},  # a beam with no return is inf, which clears the costmap
            {'scan_time': 0.0333},  # s
            {'transform_tolerance': 0.05},  # s
            {'queue_size': 20},
        ],
        remappings=[
            ('cloud_in', '/cloud_base_filtered'),
            ('scan', '/converted_scan'),
        ]
    )

    # 2D SLAM on the scan: the map frame and map -> odom.
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            LaunchConfiguration('slam_params_file'),
            {
                # Must be level and on the floor: slam_toolbox folds its base
                # frame's height and tilt into map -> odom.
                'base_frame': 'base_footprint',
                'odom_frame': 'odom',
                'scan_topic': '/converted_scan',
                'odom_topic': '/Odometry',
                'use_sim_time': use_sim_time,
            }
        ]
    )

    # Nav2's navigation half only: slam_toolbox provides map -> odom and
    # octomap_server the map (/projected_map), so no map_server or AMCL runs.
    # nav2_bringup's navigation_launch.py starts the servers under
    # lifecycle_manager_navigation with autostart.
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    # navigation_launch.py takes no BT argument; bt_navigator reads its tree
    # from the params file, so point that key at the installed humanoid tree.
    nav2_params = RewrittenYaml(
        source_file=LaunchConfiguration('nav2_params_file'),
        param_rewrites={'default_nav_to_pose_bt_xml': os.path.join(
            get_package_share_directory('golem_slam'), 'behavior_trees',
            'navigate_to_pose_humanoid.xml')},
        convert_types=True,
    )
    nav2_launch = GroupAction(actions=[
        # Humble's navigation_launch remaps the controller's cmd_vel to the
        # velocity smoother's input but not the recoveries'; this node-scoped rule
        # sends every Nav2 command through the smoother and its limits.
        SetRemap(src='behavior_server:cmd_vel', dst='cmd_vel_nav'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': nav2_params,
                'autostart': 'true',
                'namespace': '',
            }.items(),
        ),
    ])

    return [livox_frame_tf, fast_lio_node, body_to_pelvis_node, level_odom_node,
            self_filter_node, level_cloud_node, octomap_node, pointcloud_to_laserscan_node,
            slam_toolbox_node, nav2_launch]


def _robot_file(name):
    """Return config/<robot>/<name> for the robot argument, as a substitution."""
    return PathJoinSubstitution([_config_dir(), LaunchConfiguration('robot'), name])


def generate_launch_description():
    """Declare the arguments and build the graph for the robot argument's config directory."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot', description="Required. The robot, which picks golem_slam's "
                                 'config/<robot>/: a directory name there (h12 or g1)'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true to run every node on /clock (sims, bag '
                                          'replay); false on the real robot'),
        DeclareLaunchArgument('fast_lio_config', default_value=_robot_file('fastlio.yaml'),
                              description='FAST-LIO parameter file, also read by '
                                          "body_to_pelvis for the extrinsic; a sim may "
                                          'need its own (the G1 sim: fastlio_sim.yaml)'),
        DeclareLaunchArgument('nav2_params_file', default_value=_robot_file('nav2.yaml'),
                              description='Nav2 parameter file (path)'),
        DeclareLaunchArgument('octomap_params_file', default_value=_robot_file('octomap.yaml'),
                              description='octomap_server parameter file (path)'),
        DeclareLaunchArgument('slam_params_file', default_value=os.path.join(
            _config_dir(), 'slam_toolbox.yaml'),
            description='slam_toolbox parameter file (path), shared by both robots'),
        OpaqueFunction(function=_pipeline),
    ])
