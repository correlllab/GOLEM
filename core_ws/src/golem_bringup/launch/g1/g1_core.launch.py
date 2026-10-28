"""
Core G1 stack, shared by g1_real_robot.launch.py (real robot) and g1_sim.launch.py.

Starts robot_state_publisher, golem_upper_ik's joint_state_publisher (the 29
body joints), golem_safety's safety_node, the only writer of /lowcmd, the arm
controller (golem_upper_ik's frame_task_server) and, optionally, the walking
node (locomotion_node), navigation (g1_navigation), the Dex3-1 hands and RViz.
The golem nodes get robot g1 as a ROS parameter, and safety_node and
frame_task_server their configs (joint_state_publisher reads its default,
g1_split). safety_node also gets sim (it refuses sim:=true on ROS domain 0) and
use_estop false: the G1 stack has no hardware estop node.

safety_node runs split mode (split, or sim_split in the sim): it writes every
motor on /lowcmd, the legs from the walking node's input and the waist and
arms from the controller's.

The safety node starts first; the controller follows after `controller_delay`
seconds. It waits for a fresh /lowstate (or stops with an error) and starts
every command at the measured pose. Its follow gate then keeps each waist and
arm command within controller.follow_tolerance (0.05 rad) of the measured
joint until it sees golem_safety relay the commands, reading (never writing)
golem_safety's output, /lowcmd, which split mode writes once its engage gate
passes: once the walking node commands too. Only then does the controller move
the arms to `home` and the waist to its hold angles, and from then on no
command follows the measured pose. The follow gate must open within
controller.startup_timeout (30 s), and the moves to home and the waist hold
must finish within controller.startup_move_timeout (30 s) of the gate opening;
otherwise the controller stops with an error.

use_locomotion:=true adds HOMIE's walking policy on the legs, through
golem_safety's split mode, `locomotion_delay` seconds after the controller.
Without it nothing commands the legs, so the engage gate never passes and the
controller stops after controller.startup_timeout. The controller holds the
waist as stiffly as golem_safety allows (g1_split.yaml, sim_split.yaml), as
the policy expects.

use_nav:=true adds g1_navigation (odometry, slam_toolbox and Nav2, which
drives the walking node through /cmd_vel), with the core, on the sim's
FAST_LIO config and clock in the sim. odom_source picks the odometry:
fastlio (the default; the sim must not publish its ground truth, GOLEM_SIM_ODOM
unset) or sim_gt (the sim's ground-truth odom -> pelvis, GOLEM_SIM_ODOM=1).
FAST_LIO starts while the sim's stand-in holds the robot still, as it needs.
frontier_explore:=true adds g1_navigation's frontier explorer (it serves
/skill/frontier_explore), and is passed to g1_navigation as a launch argument
of its include.

The launch refuses to start on a bad pairing, checked by
golem_upper_ik.pairing for robot g1:
- The safety config must relay the upper-body channel (split, or sim_split in
  the sim).
- Each safety config runs with one controller config: split with g1_split and
  sim_split with sim_split. controller_config defaults to the one the safety
  config runs with.
- The real robot (sim:=false) never takes a sim_* safety or controller config,
  and the sim refuses the real ones.
- use_locomotion:=true needs safety_config split and controller_config
  g1_split (sim_split and sim_split in the sim), and on the real robot
  start_position_verified:=true.
- use_nav:=true is sim only for now (the real MID-360 is not connected yet),
  needs use_locomotion:=true, and odom_source must be fastlio or sim_gt.

Not checked: configs are told apart by name only, so a renamed copy of a sim
config passes, and a custom controller config is not matched to the safety
config; start_position_verified is the operator's word, not a measurement.
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

from g1_description.paths import urdf_path
from golem_upper_ik.pairing import default_controller_config, pairing_refusal

ROBOT = 'g1'
# real / sim defaults: golem_safety config, controller config; the controller
# config defaults to the one the safety config runs with (default_controller_config)
DEFAULT_CONFIGS = {
    False: ('split', 'g1_split'),
    True: ('sim_split', 'sim_split'),
}
# real / sim: the walking node's config (g1_locomotion's config/<name>.yaml)
LOCOMOTION_CONFIGS = {
    False: 'g1',
    True: 'sim',
}
# g1_navigation's odom sources (its launch file checks them too)
ODOM_SOURCES = ('fastlio', 'sim_gt')


def _as_bool(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in ('true', '1', 'yes')


def _config(context, name):
    return LaunchConfiguration(name).perform(context)


def locomotion_refusal(sim, locomotion_config, start_position_verified):
    """Why the walking node must not start with these settings, or None."""
    default_locomotion = LOCOMOTION_CONFIGS[sim]
    if locomotion_config in ('g1', 'sim') and locomotion_config != default_locomotion:
        meant_for = 'the sim' if locomotion_config == 'sim' else 'the real robot'
        return (f'locomotion_config:={locomotion_config} is for {meant_for}; '
                f'use {default_locomotion}')
    if not sim and not start_position_verified:
        return ('use_locomotion:=true on the real robot needs start_position_verified:=true, '
                'set only after checking the robot is in a safe start pose')
    return None


def nav_refusal(sim, use_locomotion, odom_source):
    """Why g1_navigation must not start with these settings, or None."""
    if not sim:
        return ('use_nav:=true is sim only for now: the real MID-360 is not connected to the '
                "bringup yet (g1_navigation's README, The real lidar)")
    if not use_locomotion:
        return "use_nav:=true needs use_locomotion:=true: Nav2 drives the walking node's /cmd_vel"
    if odom_source not in ODOM_SOURCES:
        return f'odom_source must be one of {", ".join(ODOM_SOURCES)}, not {odom_source!r}'
    return None


def startup_refusal(sim, use_locomotion, safety_config, controller_config,
                    locomotion_config, start_position_verified,
                    use_nav=False, odom_source='fastlio'):
    """Why g1_core must not start with these settings, or None."""
    # first, so that the real robot refuses use_nav whatever else is set
    refusal = nav_refusal(sim, use_locomotion, odom_source) if use_nav else None
    # golem_upper_ik's pairing rules for the G1
    if refusal is None:
        refusal = pairing_refusal(ROBOT, controller_config, safety_config,
                                  sim=sim, locomotion=use_locomotion)
    if refusal is None and use_locomotion:
        refusal = locomotion_refusal(sim, locomotion_config, start_position_verified)
    return refusal


def _setup(context):
    sim = _as_bool(context, 'sim')
    use_locomotion = _as_bool(context, 'use_locomotion')
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
    refusal = startup_refusal(sim, use_locomotion, safety_config, controller_config,
                              locomotion_config, _as_bool(context, 'start_position_verified'),
                              use_nav, odom_source)
    if refusal is not None:
        raise RuntimeError(f'g1_core: not starting: {refusal}')

    with open(urdf_path('with_hand', ros=True)) as urdf_file:
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
            package='g1_dex3',
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
    if use_locomotion:
        # silent until rt/lowstate is upright and the controller commands; then
        # it ramps the legs to the policy's default pose and starts the policy
        actions.append(TimerAction(
            period=delay + locomotion_delay,
            actions=[
                Node(
                    package='g1_locomotion',
                    executable='locomotion_node',
                    name='locomotion_node',
                    arguments=['--config', f'{locomotion_config}.yaml'],
                    parameters=[use_sim_time],
                    output='screen',
                ),
            ],
        ))
    if use_nav:
        # with the core: FAST_LIO initialises on its first scans, while the
        # sim's stand-in still holds the robot standing still
        nav = os.path.join(get_package_share_directory('g1_navigation'), 'launch',
                           'g1_navigation.launch.py')
        # the arguments of this launch that g1_navigation also declares are passed
        # explicitly: Humble includes are not scoped, so an included launch file sees
        # every launch configuration set so far, and an argument left out would take a
        # value set elsewhere
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav),
            launch_arguments={
                'odom_source': odom_source,
                'sim': str(sim).lower(),
                'use_sim_time': str(use_sim_time['use_sim_time']).lower(),
                'frontier_explore': str(_as_bool(context, 'frontier_explore')).lower(),
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
            'use_locomotion', default_value='false',
            description="HOMIE's walking policy on the legs (needs split mode, and "
                        'start_position_verified:=true on the real robot)'),
        DeclareLaunchArgument(
            'locomotion_config', default_value='',
            description='g1_locomotion config (default: g1, or sim in sim)'),
        DeclareLaunchArgument('locomotion_delay', default_value='2.0',
                              description='s between the controller and the walking node'),
        DeclareLaunchArgument(
            'use_nav', default_value='false',
            description='g1_navigation: odometry, SLAM and Nav2 on /cmd_vel (needs '
                        'use_locomotion; sim only until the real lidar is connected)'),
        DeclareLaunchArgument(
            'odom_source', default_value='fastlio',
            description="with use_nav: fastlio, or sim_gt (the sim's ground truth, "
                        'GOLEM_SIM_ODOM=1)'),
        DeclareLaunchArgument(
            'frontier_explore', default_value='false',
            description='with use_nav: the frontier explorer, serving '
                        '/skill/frontier_explore'),
        # the leg-control interlock; the sim does not need it
        DeclareLaunchArgument(
            'start_position_verified', default_value='false',
            description='true only after checking the real robot is in a safe start pose'),
        OpaqueFunction(function=_setup),
    ])
