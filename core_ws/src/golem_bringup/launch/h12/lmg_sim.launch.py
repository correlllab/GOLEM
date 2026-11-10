"""
Start the loco-manipulation game against the sim: lmg_deploy's MARL policy on all 27 joints.

    ros2 launch golem_bringup lmg_sim.launch.py [use_skills:=true]

Start the simulator first. Everything starts at once: the state publishers
(h12_state), the command path (h12_control without frame_task_server, no estop, no
delay), the policy (h12_locomanipulationgame: lmg_deploy's policies/h12/, with the start
pose taken as verified and no operator confirm), navigation (use_slam), golem_skills
(use_skills: /skill/*, among them /skill/frontier_explore, which walks the policy through
Nav2) and RViz for Nav2 goals (use_rviz). golem_skills does not wait for the perception
and gripper services or frame_task_server, none of which run here, and plans its frontier
goals with the H1-2's Nav2 footprint (golem_slam's config/h12/nav2.yaml).
safety_config defaults to the real robot's preset, relax_safety_split: the policy was
trained against its estops, and the sim is where to see that it does not trip them.
lmg_real.launch.py is the real-robot counterpart.

SAFETY: this launch engages and walks without an operator, so it refuses an unset or 0
ROS_DOMAIN_ID (domain 0, also the DDS default, is the real robot's command bus) before
any node starts.
"""
import os

from ament_index_python.packages import get_package_share_directory
from golem_bringup.include import include
from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml

# The arguments passed through to h12_locomanipulationgame.launch.py as they are.
GAME_ARGS = ('prepose_ramp_s', 'prepose_max_vel')


def h12_footprint():
    """Return the H1-2's Nav2 footprint, the global costmap's in golem_slam's config/h12."""
    path = os.path.join(get_package_share_directory('golem_slam'), 'config', 'h12', 'nav2.yaml')
    with open(path) as f:
        config = yaml.safe_load(f)
    return config['global_costmap']['global_costmap']['ros__parameters']['footprint']


def generate_launch_description():
    """Refuse the real robot's domain, then declare the game's arguments and blocks."""
    if os.environ.get('ROS_DOMAIN_ID', '').strip() in ('', '0'):
        raise RuntimeError(
            'lmg_sim: ROS_DOMAIN_ID is unset or 0, the real robot\'s bus. '
            'Export a sim domain (e.g. ROS_DOMAIN_ID=42), or use lmg_real.launch.py '
            'on the robot.')

    bringup_share = get_package_share_directory('golem_bringup')
    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true: every node runs on the simulator /clock'),
        DeclareLaunchArgument('prepose_ramp_s', default_value='6.0',
                              description='s, the shortest ramp into the default pose'),
        DeclareLaunchArgument('prepose_max_vel', default_value='0.25',
                              description='rad/s, the fastest joint target during that ramp; '
                                          'it lengthens the ramp'),
        DeclareLaunchArgument('safety_config', default_value='relax_safety_split.yaml',
                              description='golem_safety config: a name under its config/h12/, '
                                          'or a path'),
        DeclareLaunchArgument('use_slam', default_value='true',
                              description="true: start golem_slam's navigation, which also "
                                          "gives the policy /Odometry for its arm goals"),
        DeclareLaunchArgument('use_skills', default_value='false',
                              description='true: start golem_skills (/skill/*, among them '
                                          '/skill/frontier_explore); needs use_slam'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='true: start RViz on rviz_config'),
        DeclareLaunchArgument(
            'rviz_config', default_value=os.path.join(bringup_share, 'rviz', 'h12', 'main.rviz'),
            description='RViz config file (path)'),

        # TF of the body from /lowstate.
        include('golem_bringup', 'h12/h12_state.launch.py', {
            **sim_time,
            'urdf_file': urdf_path('h12', ros=True),
        }),
        # safety_node alone: the policy owns both of its inputs, so no
        # frame_task_server runs.
        include('golem_bringup', 'h12/h12_control.launch.py', {
            **sim_time,
            'sim': 'true',
            'use_frame_task': 'false',
            'safety_config': LaunchConfiguration('safety_config'),
            # No hardware estop in the sim, so nothing to wait for.
            'safety_delay': '0.0',
        }),
        # The policy, on all 27 joints.
        include('golem_bringup', 'h12/h12_locomanipulationgame.launch.py', {
            **sim_time,
            **{name: LaunchConfiguration(name) for name in GAME_ARGS},
            'urdf_path': urdf_path('h12'),
            'start_position_verified': 'true',
            # Engages as soon as the pre-pose settles, with no operator confirm.
            'engage_wait_for_confirm': 'false',
            'disable_elastic_band': 'true',
        }),
        # SLAM and Nav2, driving the policy on /cmd_vel.
        include('golem_slam', 'navigation.launch.py', {**sim_time, 'robot': 'h12'},
                condition=IfCondition(LaunchConfiguration('use_slam'))),
        # The /skill/* action servers, without waiting for the H1-2's perception and
        # gripper services or frame_task_server, which the game does not run.
        Node(
            package='golem_skills',
            executable='skills',
            name='golem_skills',
            parameters=[{'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'),
                                                        value_type=bool),
                         'wait_for_services': False,
                         'footprint': h12_footprint()}],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('use_skills')),
        ),

        # The operator's view, and where Nav2 goals are set.
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_locomanipulationgame',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'),
                                                        value_type=bool)}],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ])
