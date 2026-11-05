"""
Start the core G1 stack, shared by g1_real_robot.launch.py and g1_sim.launch.py.

Starts robot_state_publisher and golem_upper_ik's joint_state_publisher (the
29 body joints; golem_upper_ik's robot_state.launch.py), golem_safety's
safety_node, the only writer of /lowcmd (golem_safety's safety.launch.py), the
arm controller (golem_upper_ik's frame_task_server, through its
upper_ik.launch.py), HOMIE's walking node (g1_homie's locomotion_node; on the
real robot only with start_position_verified:=true) and, optionally,
navigation (golem_slam), the Dex3-1 hands and RViz. The golem nodes get robot
g1 as a ROS parameter, and safety_node and frame_task_server their configs
(joint_state_publisher reads its default, g1_split). safety_node also gets sim
(it refuses sim:=true on ROS domain 0) and use_estop false: the G1 stack has no
hardware estop node.

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
through /cmd_vel), with the core: on the sim's FAST-LIO config and clock in
the sim (golem_slam's config/g1/fastlio_sim.yaml), and on the real robot
config/g1/fastlio.yaml, wall time and the MID-360 driver
(g1_real_drivers.launch.py). FAST-LIO is the only odometry: the shared
pipeline needs its deskewed cloud. Leave GOLEM_SIM_ODOM unset to avoid a second
odom -> pelvis publisher.
FAST-LIO starts while the sim's elastic band still holds the robot up: the
walking node releases it once its policy runs and the controller's start-up
moves are done (or band.max_wait after the policy started).
use_skills:=true starts golem_skills (serving /skill/*, including
/skill/frontier_explore) without waiting for the H1-2's perception and gripper
services. use_hands starts dex3_hands, the Dex3-1 driver, which writes
rt/dex3/<side>/cmd directly, outside golem_safety. use_rviz starts RViz on
rviz/g1/g1.rviz.

The launch refuses to start, before any node, on these settings: g1_core
refuses the walking node's config (locomotion_refusal) before it returns any
action, and includes golem_upper_ik's upper_ik.launch.py first, whose pairing
check (golem_upper_ik.pairing's rules for robot g1) runs before any node starts:
- The safety config must be one of the G1's pairs (golem_upper_ik's
  config/g1/pairing.yaml): split or sim_split.
- Each safety config runs with its paired controller config, and no other:
  split with g1_split and sim_split with sim_split. controller_config defaults
  to the paired one.
- Sim and real configs do not mix: the real robot (sim:=false) refuses every
  sim_* safety or controller config and the walking node's sim config, and the
  sim refuses the real robot's (split, g1_split, and the walking node's g1).
start_position_verified refuses nothing: it decides whether the real robot's
walking node starts.

Not checked: configs are told apart by name only, so a renamed copy of a sim
config passes, and the paired controller config is not checked for the stiff
waist the walking policy expects; start_position_verified is the operator's
word, not a measurement.
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

from golem_upper_ik.pairing import default_controller_config

ROBOT = 'g1'
# sim -> the default golem_safety config and controller config; the controller
# config defaults to the one the safety config runs with (default_controller_config).
DEFAULT_CONFIGS = {
    False: ('split', 'g1_split'),
    True: ('sim_split', 'sim_split'),
}
# sim -> the walking node's config (g1_homie's config/<name>.yaml).
LOCOMOTION_CONFIGS = {
    False: 'g1',
    True: 'sim',
}


def _as_bool(context, name):
    """Return whether a launch configuration is true, 1 or yes."""
    return LaunchConfiguration(name).perform(context).strip().lower() in ('true', '1', 'yes')


def _config(context, name):
    """Return a launch configuration's text."""
    return LaunchConfiguration(name).perform(context)


def locomotion_refusal(sim, locomotion_config):
    """Return why the walking node's config does not suit the platform, or None."""
    for config_sim, name in LOCOMOTION_CONFIGS.items():
        if locomotion_config == name and config_sim != sim:
            meant_for = 'the sim' if config_sim else 'the real robot'
            return f'locomotion_config:={name} is for {meant_for}; use {LOCOMOTION_CONFIGS[sim]}'
    return None


def _include(package, launch_file, launch_arguments=None):
    """Include a package's launch file with the given arguments."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=(launch_arguments or {}).items(),
    )


def _setup(context):
    """Return the core's actions for the launch configurations; raise RuntimeError to refuse."""
    sim = _as_bool(context, 'sim')
    use_nav = _as_bool(context, 'use_nav')
    use_sim_time = {'use_sim_time': _as_bool(context, 'use_sim_time')}
    # The value passed to included launch files.
    sim_time_arg = str(use_sim_time['use_sim_time']).lower()
    default_safety, default_controller = DEFAULT_CONFIGS[sim]
    safety_config = _config(context, 'safety_config') or default_safety
    # The controller config the safety config runs with: g1_split with split. A
    # safety config in none of the pairs keeps the default, and upper_ik.launch.py
    # refuses it.
    try:
        default_controller = default_controller_config(ROBOT, safety_config)
    except ValueError:
        pass
    controller_config = _config(context, 'controller_config') or default_controller
    locomotion_config = _config(context, 'locomotion_config') or LOCOMOTION_CONFIGS[sim]
    delay = float(LaunchConfiguration('controller_delay').perform(context))
    locomotion_delay = float(LaunchConfiguration('locomotion_delay').perform(context))
    refusal = locomotion_refusal(sim, locomotion_config)
    if refusal is not None:
        raise RuntimeError(f'g1_core: not starting: {refusal}')

    rviz_config = os.path.join(get_package_share_directory('golem_bringup'), 'rviz', 'g1',
                               'g1.rviz')

    # Every argument of the included launch files is passed, even at its default:
    # Humble includes are not scoped, so an included launch file sees every launch
    # configuration set so far, and an argument left out would take a value set
    # elsewhere (upper_ik.launch.py's start_delay would delay safety_node).
    actions = [
        # SAFETY: first, so its pairing check refuses before any node starts; then
        # frame_task_server after controller_delay.
        _include('golem_upper_ik', 'upper_ik.launch.py', {
            'robot': ROBOT,
            'config': controller_config,
            'safety_config': safety_config,
            'sim': str(sim).lower(),
            'use_sim_time': sim_time_arg,
            'start_delay': str(delay),
        }),
        # TF: the 29 body joints from /lowstate; the finger joints come from dex3_hands.
        _include('golem_upper_ik', 'robot_state.launch.py', {
            'robot': ROBOT,
            'use_sim_time': sim_time_arg,
            'urdf_file': '',
        }),
        # The only writer of rt/lowcmd, at once.
        _include('golem_safety', 'safety.launch.py', {
            'robot': ROBOT,
            'config': safety_config,
            'sim': str(sim).lower(),
            'use_estop': 'false',
            'use_sim_time': sim_time_arg,
            'start_delay': '0.0',
        }),
        # The Dex3-1 hands and their finger joint states.
        Node(
            package='dex_3',
            executable='dex3_hands',
            name='dex3_hands',
            parameters=[use_sim_time],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_hands')),
        ),
        # The operator's view.
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
    # SAFETY: the leg-control interlock, as the H1-2's: the sim always starts the
    # walking node, the real robot only once the operator has verified its start
    # pose. It starts locomotion_delay after the controller.
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
        # The /skill/* action servers, without waiting for the H1-2's services.
        actions.append(Node(
            package='golem_skills',
            executable='skills',
            name='golem_skills',
            parameters=[use_sim_time, {'wait_for_services': False}],
            output='screen',
        ))
    if use_nav and not sim:
        # The real MID-360's driver; the sim publishes the lidar itself.
        actions.append(_include('golem_bringup', os.path.join('g1', 'g1_real_drivers.launch.py')))
    if use_nav:
        # SLAM and Nav2, at once with the core: FAST-LIO initialises on its first
        # scans, in the sim while the elastic band still holds the robot up. The
        # arguments golem_slam also declares are passed explicitly, as above.
        # FAST-LIO's config defaults to the real robot's, golem_slam's
        # config/g1/fastlio.yaml; the sim has its own.
        navigation = {'robot': ROBOT, 'use_sim_time': sim_time_arg}
        if sim:
            navigation['fast_lio_config'] = os.path.join(
                get_package_share_directory('golem_slam'), 'config', ROBOT, 'fastlio_sim.yaml')
        actions.append(_include('golem_slam', 'navigation.launch.py', navigation))
    return actions


def generate_launch_description():
    """Declare the core's arguments; _setup builds its actions from them."""
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='false',
                              description='true for the G1 sim, false for the real robot: '
                                          'picks the default configs and the leg interlock'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true: every node runs on the simulator /clock'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='true: start RViz on rviz/g1/g1.rviz'),
        DeclareLaunchArgument('use_hands', default_value='true',
                              description='true: start dex3_hands, the Dex3-1 hand driver'),
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
                              description='s (wall clock) between the safety node and the '
                                          'controller; >= 0'),
        DeclareLaunchArgument(
            'locomotion_config', default_value='',
            description="g1_homie config, a name under its config/: g1 (real) or sim "
                        "(default: the platform's)"),
        DeclareLaunchArgument('locomotion_delay', default_value='2.0',
                              description='s (wall clock) between the controller and the '
                                          'walking node; >= 0'),
        DeclareLaunchArgument(
            'use_nav', default_value='true',
            description='golem_slam: odometry, SLAM and Nav2 on /cmd_vel, and on the real '
                        'robot the MID-360 driver (g1_real_drivers.launch.py)'),
        DeclareLaunchArgument(
            'use_skills', default_value='false',
            description='Start the golem_skills action servers (/skill/*)'),
        # SAFETY: the leg-control interlock; the sim always starts the walking node.
        DeclareLaunchArgument(
            'start_position_verified', default_value='false',
            description="true only after checking the robot is in a safe start pose: "
                        "starts the real robot's walking node"),
        OpaqueFunction(function=_setup),
    ])
