import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('h12_slam'), 'config')

    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (MuJoCo) clock if true'
    )

    fast_lio_config_file = LaunchConfiguration('fast_lio_config_file')
    declare_fast_lio_config_file = DeclareLaunchArgument(
        'fast_lio_config_file', default_value='mid360.yaml',
        description='Config file for fast_lio (resolved against h12_slam/config)'
    )

    nav2_params_file = LaunchConfiguration('nav2_params_file')
    declare_nav2_params_file = DeclareLaunchArgument(
        'nav2_params_file',
        default_value=os.path.join(config_dir, 'nav2_config.yaml'),
        description='Full path to the Nav2 parameters file to use'
    )

    slam_params_file = LaunchConfiguration('slam_params_file')
    declare_slam_params_file = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(config_dir, 'slam_toolbox_h1.yaml'),
        description='Full path to the slam_toolbox parameters file'
    )

    # FAST-LIO. Subscribes to /livox/lidar and /livox/imu per mid360.yaml.
    # Assumes the Livox driver (or its sim equivalent) is already publishing them.
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=[PathJoinSubstitution([config_dir, fast_lio_config_file]),
                    {'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Attach FAST-LIO's 'body' frame to H1's 'pelvis' frame.
    #
    # FAST-LIO publishes camera_init -> body where 'body' is the IMU frame
    # (livox_imu_site in MJCF, which sits at the origin of livox_link).
    # So body == livox_link at startup. The URDF places livox_link on torso_link at
    #   xyz=(0.04874, 0, 0.67980) rpy=(0, 0.24015730507441985, 0)
    # (see CL_Assets/ros_assets/h1_2_magpie_ros.urdf, livox_joint), and torso_joint
    # sits at the pelvis origin with identity rotation.
    #
    # Inverting that transform (body -> pelvis):
    #   rotation: RotY(-0.24015730507441985)
    #   translation: -RotY(-0.24015730507441985) * (0.04874, 0, 0.67980)
    #              ≈ (0.11433, 0.0, -0.67188)
    static_tf_broadcaster = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='body_to_pelvis_broadcaster',
        arguments=['0.11433', '0.0', '-0.67188',
                   '0.0', '-0.24015730507441985', '0.0',
                   'body', 'pelvis'],
        output='screen'
    )

    # Gravity-aligned, floor-referenced odom -> camera_init (static, computed
    # once at startup: up from the IMU while the robot stands still, floor
    # height from the floor plane in the self-filtered scan), then
    # odom -> base_footprint (pelvis on the floor, yaw only) per FAST-LIO
    # odometry. Costmap and octomap height bands are in odom/map; see
    # h12_slam/level_odom.py.
    level_odom_node = Node(
        package='h12_slam',
        executable='level_odom',
        name='level_odom',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[('imu', '/livox/imu'), ('cloud', '/cloud_body_filtered'),
                    ('odometry', '/Odometry')],
    )

    # Drops the robot's own body (URDF collision boxes posed by TF) from the
    # deskewed FAST-LIO scan. Everything downstream (octomap, costmaps,
    # slam_toolbox's scan) consumes /cloud_body_filtered.
    self_filter_node = Node(
        package='h12_slam',
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
        parameters=[os.path.join(config_dir, 'octomap.yaml'),
                    {'use_sim_time': use_sim_time}],
        remappings=[('cloud_in', '/cloud_body_filtered')],
    )

    # 2D scan for slam_toolbox only (the costmaps use the 3D cloud).
    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='cloud_registered_to_laserscan',
        output='screen',
        parameters=[
            # Level frame on the floor, so body sway doesn't tilt the slice
            # into the floor.
            {'target_frame': 'base_footprint'},
            {'use_sim_time': use_sim_time},
            # m above the floor: clears floor returns and FAST-LIO z drift,
            # and stays below the robot's height.
            {'min_height': 0.25},
            {'max_height': 1.7},
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
            ('cloud_in', '/cloud_body_filtered'),
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
            get_package_share_directory('h12_slam'), 'behavior_trees',
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

    return LaunchDescription([
        declare_use_sim_time,
        declare_fast_lio_config_file,
        declare_nav2_params_file,
        declare_slam_params_file,
        fast_lio_node,
        static_tf_broadcaster,
        level_odom_node,
        self_filter_node,
        octomap_node,
        pointcloud_to_laserscan_node,
        slam_toolbox_node,
        nav2_launch,
    ])
