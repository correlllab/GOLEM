"""Native marker/standing experiment. Actuation is opt-in, never a default."""

import os
from pathlib import Path
import shlex

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def _include(filename, arguments):
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(str(
        Path(get_package_share_directory('h12_bringup')) / 'launch' / filename)),
        launch_arguments=arguments.items())


def _launch(context):
    domain = os.environ.get('ROS_DOMAIN_ID', '')
    if not domain.isascii() or not domain.isdecimal() or not 1 <= int(domain) <= 232:
        raise RuntimeError('Native experiment requires ROS_DOMAIN_ID=1..232, never the real robot bus')
    policy = LaunchConfiguration('lowerbody').perform(context)
    if policy not in ('fame_finetune', 'almi_manip_stand'):
        raise RuntimeError('lowerbody must be fame_finetune or almi_manip_stand')
    verified = LaunchConfiguration('start_position_verified').perform(context)
    if verified not in ('true', 'false'):
        raise RuntimeError('start_position_verified must be true or false')
    python = LaunchConfiguration('controller_python').perform(context)
    if not Path(python).is_absolute() or not Path(python).is_file():
        raise RuntimeError('Use scripts/run_experiment.sh after setting up the controller environment')
    bringup = Path(get_package_share_directory('h12_bringup'))
    model = Path(get_package_share_directory('h12_ros2_model'))
    sim_time = {'use_sim_time': 'true'}
    actions = [
        _include('h12_state.launch.py', {
            **sim_time, 'joint_state_prefix': shlex.quote(python),
            'urdf_file': str(model / 'assets/h1_2/h1_2_magpie_ros.urdf'),
        }),
        Node(package='model_server', executable='aruco_localizer', name='aruco_localizer',
             parameters=[{'use_sim_time': True, 'target_frame': 'pelvis',
                          'stale_timeout_sec': 0.5, 'max_observation_age_sec': 0.5,
                           'max_reprojection_px': 2.0,
                           'publish_annotated': LaunchConfiguration('publish_annotated'),
                           'publish_visualization': True,
                           'visualization_catalog_path': LaunchConfiguration('marker_catalog_path')}], output='screen'),
    ]
    # Pre-pose and upper-controller initialization can command before engagement.
    # Do not even include their process actions until the global interlock passes.
    if verified == 'true':
        from h12_safety_layer.core.config import load_config
        safety_path = bringup / 'config/safety_experiment.yaml'
        safety = load_config(safety_path)
        if safety.get('estop', {}).get('lower_stale_seconds', 0.0) != 0.2:
            raise RuntimeError('Experiment requires the opt-in lower-command watchdog implementation')
        catalog = LaunchConfiguration('marker_catalog_path').perform(context)
        if not Path(catalog).is_absolute() or not Path(catalog).is_file():
            raise RuntimeError('Experiment requires an absolute marker catalog path; use scripts/run_experiment.sh')
        actions.extend([
            _include('h12_control.launch.py', {
                **sim_time, 'use_estop': 'false', 'use_frame_task': 'true',
                'safety_config': str(safety_path),
                'frame_task_config': 'sim_safety_split.yaml',
                'frame_task_prefix': shlex.quote(python),
                'safety_delay': '0.0', 'frame_task_delay': '2.0',
            }),
            _include('h12_lowerbody.launch.py', {
                **sim_time, 'start_position_verified': 'true', 'lowerbody': policy,
                'lowerbody_params': str(bringup / 'config/lowerbody_experiment.yaml'),
                'use_estimator': 'false',
            }),
            Node(package='h12_skills', executable='skills', name='h12_skills',
                 parameters=[{'use_sim_time': True, 'target_backend': 'marker',
                              'marker_catalog_path': LaunchConfiguration('marker_catalog_path'),
                              'marker_profile_path': LaunchConfiguration('marker_profile_path')}],
                 output='screen'),
        ])
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('start_position_verified', default_value='false',
                              description='Allow safety, upper/lower actuation and marker skill servers'),
        DeclareLaunchArgument('lowerbody', default_value='fame_finetune',
                              description='fame_finetune | almi_manip_stand; standing only'),
        DeclareLaunchArgument('controller_python', default_value=EnvironmentVariable(
            'GOLEM_CONTROLLER_PYTHON', default_value='')),
        DeclareLaunchArgument('marker_catalog_path', default_value=EnvironmentVariable(
            'GOLEM_MARKER_CATALOG', default_value='')),
        DeclareLaunchArgument('marker_profile_path', default_value='',
                               description='Optional explicit active peg/source calibration profile'),
        DeclareLaunchArgument('publish_annotated', default_value='true',
                              description='Publish calibrated head RGB with observed/projected board geometry'),
        OpaqueFunction(function=_launch),
    ])
