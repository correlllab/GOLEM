import os

import yaml

from ament_index_python.packages import get_package_share_directory
from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from nav2_common.launch import RewrittenYaml

from golem_slam import g1_body_to_pelvis as frames
from golem_slam.robot_geometry import fixed_transform, matrix_to_quat

LIDAR_FRAME = 'livox_link'    # /livox/lidar's points
IMU_FRAME = 'livox_imu_link'  # /livox/imu, FAST-LIO's 'body'


def _fast_lio(context, config_dir, fast_lio_config_file, use_sim_time):
    """FAST-LIO and its body -> pelvis TF, both from the URDF's MID360 frames."""
    with open(LaunchConfiguration('urdf_file').perform(context)) as f:
        urdf = f.read()
    # FAST-LIO's extrinsic: the lidar in the IMU frame
    r_il, t_il = fixed_transform(urdf, IMU_FRAME, LIDAR_FRAME)
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=[PathJoinSubstitution([config_dir, fast_lio_config_file]),
                    {'use_sim_time': use_sim_time,
                     'mapping.extrinsic_T': [float(v) for v in t_il],
                     'mapping.extrinsic_R': [float(v) for v in r_il.flatten()]}],
        output='screen'
    )
    # FAST-LIO publishes camera_init -> body, body being the IMU frame. Hang the robot
    # off it by the pelvis's pose in that frame, torso_joint at zero.
    r_bp, t_bp = fixed_transform(urdf, IMU_FRAME, 'pelvis')
    qx, qy, qz, qw = matrix_to_quat(r_bp)
    static_tf_broadcaster = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='body_to_pelvis_broadcaster',
        arguments=['--x', str(t_bp[0]), '--y', str(t_bp[1]), '--z', str(t_bp[2]),
                   '--qx', str(qx), '--qy', str(qy), '--qz', str(qz), '--qw', str(qw),
                   '--frame-id', 'body', '--child-frame-id', 'pelvis'],
        output='screen'
    )
    return [fast_lio_node, static_tf_broadcaster]


def _as_bool(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in ('true', '1', 'yes')


def _g1_fast_lio(context, config_dir, use_sim_time):
    config = os.path.join(config_dir, LaunchConfiguration('fast_lio_config_file').perform(context))
    return [
        Node(package='tf2_ros', executable='static_transform_publisher', name='livox_frame_tf',
             arguments=frames.static_transform_arguments(
                 frames.transform(), frames.LIDAR_LINK, frames.LIVOX_FRAME), output='screen'),
        Node(package='fast_lio', executable='fastlio_mapping', name='fast_lio',
             parameters=[config, {'use_sim_time': use_sim_time}], output='screen'),
        Node(package='golem_slam', executable='g1_body_to_pelvis', name='body_to_pelvis',
             parameters=[{'fastlio_config': config, 'use_sim_time': use_sim_time}],
             output='screen'),
    ]


def _pipeline(context, robot):
    # Refuse unsupported G1 modes before constructing any executable action.
    if robot == 'g1':
        if LaunchConfiguration('odom_source').perform(context) != 'fastlio':
            raise RuntimeError('golem_slam requires FAST-LIO for its deskewed cloud')
    config_dir = os.path.join(get_package_share_directory('golem_slam'), 'config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    fast_lio_config_file = LaunchConfiguration('fast_lio_config_file')
    nav2_params_file = LaunchConfiguration('nav2_params_file')
    slam_params_file = LaunchConfiguration('slam_params_file')
    profile = {}
    if robot == 'g1':
        with open(LaunchConfiguration('pipeline_params_file').perform(context)) as source:
            profile = yaml.safe_load(source)
        fast_lio = _g1_fast_lio(context, config_dir, use_sim_time)
    else:
        fast_lio = _fast_lio(context, config_dir, fast_lio_config_file, use_sim_time)

    # Gravity-aligned, floor-referenced odom -> camera_init (static, computed
    # once at startup: up from the IMU while the robot stands still, floor
    # height from the floor plane in the self-filtered scan), then
    # odom -> base_footprint (pelvis on the floor, yaw only) per FAST-LIO
    # odometry. Costmap and octomap height bands are in odom/map; see
    # golem_slam/level_odom.py.
    level_odom_node = Node(
        package='golem_slam',
        executable='level_odom',
        name='level_odom',
        output='screen',
        parameters=[profile.get('level_odom', {}), {'use_sim_time': use_sim_time}],
        remappings=[('imu', '/livox/imu'), ('cloud', '/cloud_body_filtered'),
                    ('odometry', '/Odometry')],
    )

    # Drops the robot's own body (URDF collision boxes posed by TF) from the
    # deskewed FAST-LIO scan. Everything downstream (octomap, costmaps,
    # slam_toolbox's scan) consumes /cloud_body_filtered.
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
            # Level frame on the floor, so body sway doesn't tilt the slice
            # into the floor.
            {'target_frame': '' if robot == 'g1' else 'base_footprint'},
            {'use_sim_time': use_sim_time},
            # m above the floor: clears floor returns and FAST-LIO z drift,
            # and stays below the robot's height.
            {'min_height': profile.get('scan', {}).get('min_height', 0.25)},
            {'max_height': profile.get('scan', {}).get('max_height', 1.7)},
            {'angle_min': -3.14159},
            {'angle_max': 3.14159},
            {'angle_increment': 0.0087},
            # The self filter already removes the body; range_min only drops
            # the MID360's near-field noise.
            {'range_min': 0.3},
            {'range_max': 3.0},
            {'use_inf': True},
            {'scan_time': 0.0333},
            {'transform_tolerance': 0.05},
            {'queue_size': 20},
        ],
        remappings=[
            ('cloud_in', '/cloud_base_filtered' if robot == 'g1' else '/cloud_body_filtered'),
            ('scan', '/converted_scan'),
        ]
    )

    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            slam_params_file,
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

    # Nav2: bring up the navigation half only. slam_toolbox above provides the
    # map -> odom transform and octomap_server the map (/projected_map), so we
    # don't need map_server or AMCL. navigation_launch.py runs
    # controller_server, planner_server, behavior_server, bt_navigator,
    # smoother_server, waypoint_follower, and velocity_smoother under
    # lifecycle_manager_navigation.
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    # navigation_launch.py takes no BT argument; bt_navigator reads its tree
    # from the params file, so point that key at the installed humanoid tree.
    nav2_params = RewrittenYaml(
        source_file=nav2_params_file,
        param_rewrites={'default_nav_to_pose_bt_xml': os.path.join(
            get_package_share_directory('golem_slam'), 'behavior_trees',
            'navigate_to_pose_humanoid.xml')},
        convert_types=True,
    )
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': nav2_params,
            'autostart': 'true',
            'namespace': '',
        }.items(),
    )

    if robot == 'g1':
        # Humble's navigation_launch remaps the controller but not recoveries.
        # Scope a node-specific rule so all G1 commands pass through the smoother.
        nav2_launch = GroupAction(actions=[
            SetRemap(src='behavior_server:cmd_vel', dst='cmd_vel_nav'), nav2_launch])
    actions = fast_lio + [level_odom_node, self_filter_node, octomap_node,
                          pointcloud_to_laserscan_node, slam_toolbox_node, nav2_launch]
    if robot == 'g1':
        actions.append(Node(
            package='golem_slam', executable='level_cloud', name='level_cloud',
            parameters=[{'use_sim_time': use_sim_time}],
            remappings=[('cloud_in', '/cloud_body_filtered'),
                        ('cloud_out', '/cloud_base_filtered')], output='screen'))
    return actions


def generate_navigation_launch_description(robot):
    """Build the shared pipeline with explicit robot-specific defaults."""
    if robot not in ('h12', 'g1'):
        raise ValueError(f'unsupported robot {robot!r}')
    config = os.path.join(get_package_share_directory('golem_slam'), 'config')
    g1 = robot == 'g1'
    arguments = [
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('fast_lio_config_file',
                              default_value='g1_fastlio_sim.yaml' if g1 else 'mid360.yaml'),
        DeclareLaunchArgument('nav2_params_file', default_value=os.path.join(
            config, 'g1_nav2.yaml' if g1 else 'nav2_config.yaml')),
        DeclareLaunchArgument('slam_params_file', default_value=os.path.join(
            config, 'slam_toolbox_h1.yaml')),
        DeclareLaunchArgument('octomap_params_file', default_value=os.path.join(
            config, 'g1_octomap.yaml' if g1 else 'octomap.yaml')),
        DeclareLaunchArgument('urdf_file', default_value=urdf_path(robot, ros=True)),
    ]
    if g1:
        arguments += [
            DeclareLaunchArgument('sim', default_value='false'),
            DeclareLaunchArgument('odom_source', default_value='fastlio'),
            DeclareLaunchArgument('pipeline_params_file',
                                  default_value=os.path.join(config, 'g1_pipeline.yaml')),
        ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_pipeline, args=[robot])])
