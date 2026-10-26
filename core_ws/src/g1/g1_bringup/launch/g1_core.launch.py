'''Core G1 stack, shared by g1_bringup.launch.py (real robot) and g1_sim_bringup.launch.py.

Starts robot_state_publisher, the body joint_state_publisher, g1_safety_layer,
the arm controller (frame_task_server) and, optionally, the walking node
(locomotion_node), navigation (g1_navigation), the Dex3-1 hands and RViz.

The safety layer starts first; the controller follows after `controller_delay`
seconds. It waits for a fresh rt/lowstate (or stops with an error) and starts
every command at the measured pose. Its follow gate then keeps each waist and
arm command within controller.follow_tolerance (0.05 rad) of the measured
joint until it sees the safety layer relay the commands, reading (never
writing) the safety layer's output: rt/arm_sdk at weight 1 in arm_sdk mode,
where the safety layer takes the waist + arms from Unitree over 1 s from the
controller's first command, or rt/lowcmd in split mode, which engages once the
walking node commands too. Only then does the controller move the arms to
`home` and the waist to its hold angles, and from then on no command follows
the measured pose. The follow gate must open within controller.startup_timeout
(30 s), and the moves to home and the waist hold must finish within
controller.startup_move_timeout (30 s) of the gate opening; otherwise the
controller stops with an error.

use_locomotion:=true adds HOMIE's walking policy on the legs, through the
safety layer's split mode, `locomotion_delay` seconds after the controller.
The controller then holds the waist as stiffly as the safety layer allows
(g1_split.yaml, sim_split.yaml), as the policy expects.

use_nav:=true adds g1_navigation (odometry, slam_toolbox and Nav2, which
drives the walking node through /cmd_vel), with the core, on the sim's
FAST_LIO config and clock in the sim. odom_source picks the odometry:
fastlio (the default; the sim must not publish its ground truth, GOLEM_SIM_ODOM
unset) or sim_gt (the sim's ground-truth odom -> pelvis, GOLEM_SIM_ODOM=1).
FAST_LIO starts while the sim's stand-in holds the robot still, as it needs.
frontier_explore:=true adds g1_navigation's frontier explorer (it serves
/skill/frontier_explore), and is passed to g1_navigation as a launch argument
of its include.

The launch refuses to start on a bad pairing, checked by g1_control's
utility/config_pairing.py (its robot_safety_launch.py uses the same rules):
- The safety config must relay the upper-body channel (arm_sdk, split or
  their sim_ versions).
- Each safety config runs with one controller config: arm_sdk with g1, split
  with g1_split (a stiff waist), sim_arm_sdk with sim and sim_split with
  sim_split. controller_config defaults to the one the safety config runs
  with, so the real bringup's safety_config:=split runs g1_split.
- The real robot (sim:=false) never takes a sim_* safety or controller config,
  and the sim refuses the real ones.
- use_locomotion:=true needs safety_config split and controller_config
  g1_split (sim_split and sim_split in the sim), and on the real robot
  start_position_verified:=true.
- use_nav:=true is sim only for now (the real MID-360 is not connected yet),
  needs use_locomotion:=true, and odom_source must be fastlio or sim_gt.

Not checked: configs are told apart by name only, so a renamed copy of a sim
config passes, and a custom controller config is not matched to the safety
config (if its topics.safety_out is not the safety layer's output, the
controller never sees it engage and stops after startup_timeout);
start_position_verified is the operator's word, not a measurement.
'''

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from g1_description.paths import urdf_path
from g1_control.utility.config_pairing import (
    LOCOMOTION_SAFETY_CONFIGS,
    default_controller_config,
    pairing_refusal,
)

# real / sim defaults: safety layer config, controller config; the controller
# config defaults to the one the safety config runs with (default_controller_config)
DEFAULT_CONFIGS = {
    False: ('arm_sdk', 'g1'),
    True: ('sim_arm_sdk', 'sim'),
}
# real / sim: the walking node's config. With use_locomotion only the sim
# switches to split mode by itself; the real robot needs safety_config:=split
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
    '''Why the walking node must not start with these settings, or None'''
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
    '''Why g1_navigation must not start with these settings, or None'''
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
    '''Why g1_core must not start with these settings, or None'''
    # first, so that the real robot refuses use_nav whatever else is set
    refusal = nav_refusal(sim, use_locomotion, odom_source) if use_nav else None
    # the controller package's pairing rules, shared with its robot_safety_launch.py
    if refusal is None:
        refusal = pairing_refusal(controller_config, safety_config,
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
    if use_locomotion and sim:
        default_safety = LOCOMOTION_SAFETY_CONFIGS[sim]
    safety_config = _config(context, 'safety_config') or default_safety
    # the controller config the safety config runs with: g1_split with split
    default_controller = default_controller_config(safety_config) or default_controller
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
    rviz_config = os.path.join(get_package_share_directory('g1_bringup'), 'rviz', 'g1.rviz')

    actions = [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}, use_sim_time],
            output='screen',
        ),
        # body joints (29) from rt/lowstate; the finger joints come from dex3_hands
        Node(
            package='g1_control',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[use_sim_time],
            output='screen',
        ),
        Node(
            package='g1_safety_layer',
            executable='safety_node',
            name='safety_node',
            arguments=['--config', f'{safety_config}.yaml'],
            parameters=[use_sim_time],
            output='screen',
        ),
        TimerAction(
            period=delay,
            actions=[
                Node(
                    package='g1_control',
                    executable='frame_task_server',
                    name='frame_task_server',
                    arguments=['--config', f'{controller_config}.yaml'],
                    parameters=[use_sim_time],
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
            description='g1_safety_layer config (default: arm_sdk, or sim_arm_sdk in sim; '
                        'sim_split in sim with use_locomotion)'),
        DeclareLaunchArgument(
            'controller_config', default_value='',
            description='g1_control config (default: the one the safety config '
                        'runs with: g1, g1_split, sim or sim_split)'),
        DeclareLaunchArgument('controller_delay', default_value='2.0',
                              description='s between the safety layer and the controller'),
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
