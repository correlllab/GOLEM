"""
Core G1 stack, shared by g1_real_robot.launch.py (real robot) and g1_sim.launch.py.

Starts robot_state_publisher, golem_upper_ik's joint_state_publisher (the 29
body joints), golem_safety's safety_node, the only writer of /lowcmd, the arm
controller (golem_upper_ik's frame_task_server), HOMIE's walking node
(g1_homie's locomotion_node; on the real robot only with
start_position_verified:=true) and, optionally, navigation (golem_slam),
the Dex3-1 hands and RViz. The golem nodes get robot g1 as a ROS parameter,
and safety_node and frame_task_server their configs (joint_state_publisher
reads its default, g1_split). safety_node also gets sim (it refuses sim:=true
on ROS domain 0) and use_estop false: the G1 stack has no hardware estop node.

safety_node runs split mode (split, or sim_split in the sim): it writes every
motor on /lowcmd, the legs from the walking node's input and the waist and
arms from the controller's. From boot it streams zero gains on every motor
(every motor off), and engages each input on that input's own first valid
command once rt/lowstate reports the G1's mode_machine (5); until the walking
node commands, the leg motors stay off.

The safety node starts first; the controller follows after `controller_delay`
seconds. It waits for a fresh /lowstate (or stops with an error) and starts
every command at the measured pose. Its follow gate then keeps each waist and
arm command within controller.follow_tolerance (0.05 rad) of the measured
joint until it sees golem_safety relay the commands, reading (never writing)
golem_safety's output, /lowcmd, which carries them from the controller's first
command on, with or without the walking node. Only then does the controller
move the arms to `home` and the waist to its hold angles, and from then on no
command follows the measured pose. The follow gate must open within
controller.startup_timeout (30 s), and the moves to home and the waist hold
must finish within controller.startup_move_timeout (30 s) of the gate opening;
otherwise the controller stops with an error.

The walking node starts `locomotion_delay` seconds after the controller, with
g1_homie's config for the platform (locomotion_config: sim in the sim, g1
on the real robot), and drives the legs through golem_safety's split mode. It
sends nothing until golem_safety beats on /safety/heartbeat, rt/lowstate is
upright and the controller commands. Then it ramps the legs to the policy's
default pose. In the sim it starts the policy and releases the sim's elastic
band once the controller has finished its start-up moves (or band.max_wait
after the policy started without them); on the real robot it holds the pose
until the operator calls /lowerbody/confirm_engage (g1_homie's README has
the details). The controller holds the waist as
stiffly as golem_safety allows (g1_split.yaml, sim_split.yaml), as the policy
expects.

The sim always starts the walking node. The real robot starts it only with
start_position_verified:=true (default false), as the H1-2's leg node
(h12_lowerbody.launch.py): without it nothing commands the legs, so their
motors stay off on golem_safety's zero-gain stream while the arms run.

use_nav (default true, as the H1-2's bringups start navigation) adds
golem_slam (odometry, slam_toolbox and Nav2, which drives the walking node
through /cmd_vel), with the core: on the sim's FAST_LIO config and clock in
the sim, and on the real robot g1_fastlio.yaml, wall time and the MID-360
driver (g1_real_drivers.launch.py). odom_source must be fastlio: the shared
pipeline needs its deskewed cloud. Leave GOLEM_SIM_ODOM unset to avoid a second
odom -> pelvis publisher.
FAST_LIO starts while the sim's elastic band still holds the robot up: the
walking node releases it once its policy runs and the controller's start-up
moves are done (or band.max_wait after the policy started).
use_skills:=true starts golem_skills (serving /skill/*, including
/skill/frontier_explore).

The launch refuses to start, before any node, on these settings
(startup_refusal; the config rules are golem_upper_ik.pairing's for robot g1):
- The safety config must relay the upper-body channel (split, or sim_split in
  the sim).
- Each safety config runs with one controller config: split with g1_split and
  sim_split with sim_split. controller_config defaults to the one the safety
  config runs with.
- Sim and real configs do not mix: the real robot (sim:=false) refuses every
  sim_* safety or controller config and the walking node's sim config, and the
  sim refuses the real robot's (split, g1_split, and the walking node's g1).
  Since g1_split is the real robot's and sim_split the sim's, this also keeps
  each controller config to its own safety config.
- With use_nav, odom_source must be fastlio.
start_position_verified refuses nothing: it decides whether the real robot's
walking node starts.

Not checked: configs are told apart by name only, so a renamed copy of a sim
config passes, and a custom controller config is not matched to the safety
config, nor checked for the stiff waist the walking policy expects;
start_position_verified is the operator's word, not a measurement.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from golem_ros2_models.paths import urdf_path
from golem_upper_ik.pairing import default_controller_config, pairing_refusal

ROBOT = 'g1'
# real / sim defaults: golem_safety config, controller config; the controller
# config defaults to the one the safety config runs with (default_controller_config)
DEFAULT_CONFIGS = {
    False: ('split', 'g1_split'),
    True: ('sim_split', 'sim_split'),
}
# real / sim: the walking node's config (g1_homie's config/<name>.yaml)
LOCOMOTION_CONFIGS = {
    False: 'g1',
    True: 'sim',
}
# Shared pipeline's odom sources (its launch file checks them too)
ODOM_SOURCES = ('fastlio',)


def _as_bool(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in ('true', '1', 'yes')


def _config(context, name):
    return LaunchConfiguration(name).perform(context)


def locomotion_refusal(sim, locomotion_config):
    """Why the walking node's config does not suit the platform, or None."""
    for config_sim, name in LOCOMOTION_CONFIGS.items():
        if locomotion_config == name and config_sim != sim:
            meant_for = 'the sim' if config_sim else 'the real robot'
            return f'locomotion_config:={name} is for {meant_for}; use {LOCOMOTION_CONFIGS[sim]}'
    return None


def nav_refusal(sim, odom_source):
    """Why shared navigation must not start with these settings, or None."""
    if odom_source not in ODOM_SOURCES:
        return 'golem_slam requires FAST-LIO for its deskewed cloud'
    return None


def startup_refusal(sim, safety_config, controller_config, locomotion_config,
                    use_nav=False, odom_source='fastlio'):
    """Why g1_core must not start with these settings, or None."""
    refusal = nav_refusal(sim, odom_source) if use_nav else None
    # golem_upper_ik's pairing rules for the G1
    if refusal is None:
        refusal = pairing_refusal(ROBOT, controller_config, safety_config, sim=sim)
    if refusal is None:
        refusal = locomotion_refusal(sim, locomotion_config)
    return refusal


def _setup(context):
    sim = _as_bool(context, 'sim')
    use_nav = _as_bool(context, 'use_nav')
    odom_source = _config(context, 'odom_source')
    use_sim_time = {'use_sim_time': _as_bool(context, 'use_sim_time')}
    default_safety, default_controller = DEFAULT_CONFIGS[sim]
    safety_config = _config(context, 'safety_config') or default_safety
    # the controller config the safety config runs with: g1_split with split
    default_controller = default_controller_config(ROBOT, safety_config) or default_controller
    controller_config = _config(context, 'controller_config') or default_controller
    locomotion_config = _config(context, 'locomotion_config') or LOCOMOTION_CONFIGS[sim]
    delay = float(LaunchConfiguration('controller_delay').perform(context))
    locomotion_delay = float(LaunchConfiguration('locomotion_delay').perform(context))
    refusal = startup_refusal(sim, safety_config, controller_config, locomotion_config,
                              use_nav, odom_source)
    if refusal is not None:
        raise RuntimeError(f'g1_core: not starting: {refusal}')

    with open(urdf_path('g1', ros=True)) as urdf_file:
        robot_description = urdf_file.read()
    rviz_config = os.path.join(get_package_share_directory('golem_bringup'), 'rviz', 'g1',
                               'g1.rviz')

    actions = [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}, use_sim_time],
            output='screen',
        ),
        # body joints (29) from /lowstate; the finger joints come from dex3_hands
        Node(
            package='golem_upper_ik',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[{'robot': ROBOT}, use_sim_time],
            output='screen',
        ),
        # the config names as typed strings: launch_ros reads a plain string
        # parameter as YAML, which would make a config named 'on' a bool
        Node(
            package='golem_safety',
            executable='safety_node',
            name='safety_node',
            parameters=[{
                'robot': ROBOT,
                'config': ParameterValue(safety_config, value_type=str),
                'sim': sim,
                'use_estop': False,
            }, use_sim_time],
            output='screen',
        ),
        TimerAction(
            period=delay,
            actions=[
                Node(
                    package='golem_upper_ik',
                    executable='frame_task_server',
                    name='frame_task_server',
                    parameters=[{
                        'robot': ROBOT,
                        'config': ParameterValue(controller_config, value_type=str),
                    }, use_sim_time],
                    output='screen',
                ),
            ],
        ),
        Node(
            package='dex_3',
            executable='dex3_hands',
            name='dex3_hands',
            parameters=[use_sim_time],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_hands')),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            parameters=[use_sim_time],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ]
    # the leg-control interlock, as the H1-2's: the sim always starts the walking
    # node, the real robot only once the operator has verified its start pose
    if sim or _as_bool(context, 'start_position_verified'):
        actions.append(TimerAction(
            period=delay + locomotion_delay,
            actions=[
                Node(
                    package='g1_homie',
                    executable='locomotion_node',
                    name='locomotion_node',
                    arguments=['--config', f'{locomotion_config}.yaml'],
                    parameters=[use_sim_time],
                    output='screen',
                ),
            ],
        ))
    if _as_bool(context, 'use_skills'):
        actions.append(Node(
            package='golem_skills',
            executable='skills',
            name='golem_skills',
            parameters=[use_sim_time, {'wait_for_services': False}],
            output='screen',
        ))
    if use_nav and not sim:
        # the real MID-360's driver; the sim publishes the lidar itself
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('golem_bringup'), 'launch', 'g1',
                'g1_real_drivers.launch.py')),
            launch_arguments={'use_lidar': 'true'}.items(),
        ))
    if use_nav:
        # with the core: FAST_LIO initialises on its first scans, in the sim
        # while its elastic band still holds the robot up (the walking node
        # releases it once its policy runs and the controller's start-up moves
        # are done)
        nav = os.path.join(get_package_share_directory('golem_slam'), 'launch',
                           'g1_navigation.launch.py')
        # the arguments of this launch that golem_slam also declares are passed
        # explicitly: Humble includes are not scoped, so an included launch file sees
        # every launch configuration set so far, and an argument left out would take a
        # value set elsewhere
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav),
            launch_arguments={
                'odom_source': odom_source,
                'sim': str(sim).lower(),
                'fast_lio_config_file': 'g1_fastlio_sim.yaml' if sim else 'g1_fastlio.yaml',
                'use_sim_time': str(use_sim_time['use_sim_time']).lower(),
            }.items(),
        ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_hands', default_value='true'),
        DeclareLaunchArgument(
            'safety_config', default_value='',
            description='golem_safety config, a name under its config/g1/ or a path '
                        '(default: split, or sim_split in sim)'),
        DeclareLaunchArgument(
            'controller_config', default_value='',
            description='golem_upper_ik controller config, a name under its config/g1/ or '
                        'a path (default: the one the safety config runs with: g1_split or '
                        'sim_split)'),
        DeclareLaunchArgument('controller_delay', default_value='2.0',
                              description='s between the safety node and the controller'),
        DeclareLaunchArgument(
            'locomotion_config', default_value='',
            description='g1_homie config (default: g1, or sim in sim)'),
        DeclareLaunchArgument('locomotion_delay', default_value='2.0',
                              description='s between the controller and the walking node'),
        DeclareLaunchArgument(
            'use_nav', default_value='true',
            description='golem_slam: odometry, SLAM and Nav2 on /cmd_vel, and on the real '
                        'robot the MID-360 driver (g1_real_drivers.launch.py)'),
        DeclareLaunchArgument(
            'odom_source', default_value='fastlio',
            description='with use_nav: fastlio (shared pipeline requires its deskewed cloud)'),
        DeclareLaunchArgument(
            'use_skills', default_value='false',
            description='Start the golem_skills action servers (/skill/*)'),
        # the leg-control interlock; the sim always starts the walking node
        DeclareLaunchArgument(
            'start_position_verified', default_value='false',
            description="the real robot's walking node: true only after checking the robot "
                        'is in a safe start pose'),
        OpaqueFunction(function=_setup),
    ])
