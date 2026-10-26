'''G1 navigation: odometry, 2D SLAM and Nav2, driving the walking node's /cmd_vel.

    ros2 launch g1_navigation g1_navigation.launch.py                    # real robot, FAST_LIO
    ros2 launch g1_navigation g1_navigation.launch.py sim:=true use_sim_time:=true
    ros2 launch g1_navigation g1_navigation.launch.py odom_source:=sim_gt use_sim_time:=true

It needs robot_state_publisher on the G1 URDF (the core bringup starts it) and
the MID-360 on /livox/lidar (livox_ros_driver2/CustomMsg) and /livox/imu, both
in livox_frame, the sensor's own frame (upside down on the G1).

odom_source:
- fastlio (default): FAST_LIO on /livox/lidar and /livox/imu, hung off a level
  odom frame by a static transform computed from the URDF
  (g1_navigation/frames.py), and off the pelvis by body_to_pelvis, through the
  measured waist. Start it with the robot standing still: odom assumes the
  torso level and the pelvis at pelvis_height when FAST_LIO initialises. The
  sim must not publish its ground-truth odom -> pelvis (GOLEM_SIM_ODOM unset).
- sim_gt: the sim's ground-truth odom -> pelvis TF (it publishes it only with
  GOLEM_SIM_ODOM=1); no FAST_LIO. Sim only.

Both: level_frame publishes pelvis_level (the pelvis with its roll and pitch
removed) and base_footprint (pelvis_level on odom's floor: the base frame of
slam_toolbox and Nav2, so map -> odom is the identity), livox_to_pointcloud
turns /livox/lidar into /lidar_points, deskewed for the sensor's rotation with
/livox/imu's gyro and placed in pelvis_level through TF, and
pointcloud_to_laserscan (no TF) cuts /scan from a level band of it.
slam_toolbox draws /map from /scan at the odometry's poses (no scan matching:
the odometry localises; config/g1_slam_toolbox.yaml); Nav2 serves
/navigate_to_pose. The velocity smoother is the only publisher of /cmd_vel,
capped at vx -0.3..0.5 m/s, vy 0 and |wz| <= 0.6 rad/s (config/g1_nav2.yaml).

TF: map -(slam_toolbox)-> odom -(static)-> camera_init -(FAST_LIO)-> body
-(body_to_pelvis)-> pelvis -(robot_state_publisher)-> ... torso_link -> mid360_link
-(static)-> livox_frame, pelvis -(level_frame)-> pelvis_level -(level_frame)-> base_footprint,
with odom -> pelvis straight from the sim in sim_gt mode.

frontier_explore:=true (default false) adds the frontier explorer
(g1_navigation/frontier_explore.py), which serves /skill/frontier_explore and
walks the robot through Nav2 to the frontiers of /map, with the footprint of
nav2_params' global costmap.
'''

import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from g1_description.paths import urdf_path
from g1_navigation import frames

ODOM_SOURCES = ('fastlio', 'sim_gt')

# mid360_link -> livox_frame: the frame the points arrive in. Identity for the
# raw sensor frame (the sim, and the real driver with a zero extrinsic in
# MID360_config.json); see config/g1_fastlio.yaml.
LIDAR_FRAME = frames.transform()

# (package, executable, remappings); the controller's and the behaviors' commands
# go to the velocity smoother, whose output is the only /cmd_vel
NAV2_NODES = (
    ('nav2_controller', 'controller_server', [('cmd_vel', 'cmd_vel_nav')]),
    ('nav2_planner', 'planner_server', []),
    ('nav2_behaviors', 'behavior_server', [('cmd_vel', 'cmd_vel_nav')]),
    ('nav2_bt_navigator', 'bt_navigator', []),
    ('nav2_velocity_smoother', 'velocity_smoother',
     [('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')]),
)

# the frame the scan is cut in: livox_to_pointcloud publishes /lidar_points in it
SCAN_FRAME = 'pelvis_level'

# pointcloud_to_laserscan: the 2D scan for slam_toolbox and the costmaps, cut in
# the cloud's own frame (SCAN_FRAME), so that it needs no TF
SCAN = {
    'target_frame': '',
    # the band, relative to the pelvis but level: 0.15 m above the floor (pelvis
    # at 0.744 standing, ~0.72 walking) to just under the lidar (0.47 m up).
    # It holds only with livox_to_pointcloud's deskew: in the G1 sim's kitchen
    # test, without it, 1.2-1.7% of the scan's returns within 4 m lay over 0.25 m
    # from anything in the band, floor points (the torso tilts up to 9 deg in one
    # frame); with it, none in 32 runs.
    'min_height': -0.59,
    'max_height': 0.45,
    # horizontal distance from the pelvis; drops the robot's own arms and hands
    # (the Dex3 finger meshes reach 0.445 m in 'home'). The sim's lidar returns
    # nothing from the robot's body, so it cannot check this; the real robot will.
    'range_min': 0.50,
    'range_max': 8.0,
    'angle_min': -math.pi,
    'angle_max': math.pi,
    'angle_increment': 0.0087,
    'scan_time': 0.1,
    'use_inf': True,
    'queue_size': 10,
}


def _as_bool(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in ('true', '1', 'yes')


def _footprint(nav2_params):
    '''The global costmap's footprint in a Nav2 params file, or None'''
    with open(nav2_params) as f:
        params = yaml.safe_load(f)
    try:
        return params['global_costmap']['global_costmap']['ros__parameters']['footprint']
    except (KeyError, TypeError):
        return None


def _static_tf(name, t, parent, child):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=name,
        arguments=frames.static_transform_arguments(t, parent, child),
        output='screen',
    )


def _setup(context):
    share = get_package_share_directory('g1_navigation')
    odom_source = LaunchConfiguration('odom_source').perform(context)
    if odom_source not in ODOM_SOURCES:
        raise RuntimeError(f'odom_source must be one of {ODOM_SOURCES}, got {odom_source!r}')
    use_sim_time = {'use_sim_time': _as_bool(context, 'use_sim_time')}
    fastlio_config = LaunchConfiguration('fastlio_config').perform(context) or (
        'g1_fastlio_sim.yaml' if _as_bool(context, 'sim') else 'g1_fastlio.yaml')
    fastlio_config = os.path.join(share, 'config', fastlio_config)
    nav2_params = LaunchConfiguration('nav2_params').perform(context)
    slam_params = LaunchConfiguration('slam_params').perform(context)

    actions = [
        _static_tf('livox_frame_tf', LIDAR_FRAME, frames.LIDAR_LINK, frames.LIVOX_FRAME),
        # deskewed with /livox/imu, and in SCAN_FRAME at each frame's stamp. TODO(lidar): the
        # deskew is the sim's only. On the robot, MID360_config.json's roll-180 extrinsic turns
        # the points but not the IMU, so the gyro's y and z would have the wrong sign for them.
        Node(
            package='g1_navigation',
            executable='livox_to_pointcloud',
            name='livox_to_pointcloud',
            parameters=[{'target_frame': SCAN_FRAME, 'deskew': _as_bool(context, 'sim')}, use_sim_time],
            output='screen',
        ),
        Node(
            package='g1_navigation',
            executable='level_frame',
            name='level_frame',
            parameters=[use_sim_time],
            output='screen',
        ),
        # no TF and no clock: the cloud comes in SCAN_FRAME, and the scan keeps its
        # stamp. With its own TF message filter (target_frame pelvis_level) it
        # stopped for good in 7 of 27 runs of the G1 sim's nav test, about 1.5 s
        # after the walking node started: 6 of 13 on sim time, 1 of 14 on the wall
        # clock (livox_to_pointcloud's docstring)
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            parameters=[SCAN, {'use_sim_time': False}],
            remappings=[('cloud_in', '/lidar_points'), ('scan', '/scan')],
            output='screen',
        ),
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            parameters=[slam_params, use_sim_time],
            output='screen',
        ),
    ]

    if odom_source == 'fastlio':
        odom_camera_init, _ = frames.fastlio_transforms(
            urdf_path('with_hand'), fastlio_config,
            float(LaunchConfiguration('pelvis_height').perform(context)), LIDAR_FRAME)
        actions += [
            Node(
                package='fast_lio',
                executable='fastlio_mapping',
                name='fast_lio',
                parameters=[fastlio_config, use_sim_time],
                output='screen',
            ),
            _static_tf('odom_camera_init_tf', odom_camera_init, 'odom', 'camera_init'),
            # body -> pelvis through the measured waist, not a fixed transform
            Node(
                package='g1_navigation',
                executable='body_to_pelvis',
                name='body_to_pelvis',
                parameters=[{'fastlio_config': fastlio_config}, use_sim_time],
                output='screen',
            ),
        ]

    for package, executable, remappings in NAV2_NODES:
        actions.append(Node(
            package=package,
            executable=executable,
            name=executable,
            parameters=[nav2_params, use_sim_time],
            remappings=remappings,
            output='screen',
        ))
    actions.append(Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        parameters=[use_sim_time, {
            'autostart': True,
            'node_names': [executable for _, executable, _ in NAV2_NODES],
        }],
        output='screen',
    ))

    if _as_bool(context, 'frontier_explore'):
        footprint = _footprint(nav2_params)
        actions.append(Node(
            package='g1_navigation',
            executable='frontier_explore',
            name='frontier_explore',
            parameters=[use_sim_time] + ([{'footprint': footprint}] if footprint else []),
            output='screen',
        ))
    return actions


def generate_launch_description():
    config = os.path.join(get_package_share_directory('g1_navigation'), 'config')
    return LaunchDescription([
        DeclareLaunchArgument('odom_source', default_value='fastlio',
                              description='fastlio (lidar odometry) or sim_gt (the sim\'s '
                                          'ground truth, GOLEM_SIM_ODOM=1)'),
        DeclareLaunchArgument('sim', default_value='false',
                              description='use the sim\'s FAST_LIO config (g1_fastlio_sim.yaml)'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('fastlio_config', default_value='',
                              description='FAST_LIO config in g1_navigation/config (default: '
                                          'g1_fastlio.yaml; g1_fastlio_sim.yaml with sim:=true)'),
        DeclareLaunchArgument('pelvis_height', default_value='0.744',
                              description='m, pelvis above the floor when FAST_LIO starts '
                                          '(HOMIE standing: 0.744)'),
        DeclareLaunchArgument('nav2_params', default_value=os.path.join(config, 'g1_nav2.yaml')),
        DeclareLaunchArgument('slam_params',
                              default_value=os.path.join(config, 'g1_slam_toolbox.yaml')),
        DeclareLaunchArgument('frontier_explore', default_value='false',
                              description='the frontier explorer, serving /skill/frontier_explore'),
        OpaqueFunction(function=_setup),
    ])
