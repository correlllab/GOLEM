"""
Start the H1-2's leg control: golem_lowerbody_rl's leg node.

It starts only with start_position_verified:=true: golem_lowerbody_rl's lowerbody.launch.py
(robot h12) runs the policy:= policy on the lowerbody_config run config, the only
lower-body controller feeding /safety/lowcmd_lower_in.

lowerbody.launch.py refuses, before its node starts, a run config whose sim differs from
sim, a real-robot run config without the operator's confirm or with the elastic band's
release, and a policy the H1-2 does not list (golem_lowerbody_rl's config/h12/robot.yaml).
The top-level files include this block after the others, so each runs the same check
first (golem_bringup.h12_legs.leg_check): a refusal starts no node.

SAFETY: start_position_verified is the operator's word that the robot is in a safe start
pose; without it nothing here starts, and safety_node keeps the legs on its zero-gain
stream. The leg node then waits for safety_node's heartbeat, pre-poses the legs and, on the
real robot, holds the pre-pose until /lowerbody/confirm_engage.
"""
from golem_bringup.h12_legs import leg_node_condition
from golem_bringup.include import include
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Declare the leg block's arguments and start the leg node behind the interlock."""
    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true: the leg node runs on the simulator /clock'),
        # SAFETY: the leg interlock; the sim bringups pass true.
        DeclareLaunchArgument(
            'start_position_verified', default_value='false',
            description='true only after checking the robot is in a safe start pose: '
                        'starts the leg node'),
        DeclareLaunchArgument(
            'policy', default_value='almi',
            description="The leg node's policy: almi, fame or walk (golem_lowerbody_rl's "
                        'config/h12/robot.yaml); any other is refused'),
        DeclareLaunchArgument(
            'sim',
            description='Required, true|false: true for a simulator; the leg node refuses a '
                        'run config for the other platform'),
        DeclareLaunchArgument(
            'lowerbody_config',
            description="Required: the leg node's run config, sim or real "
                        "(golem_lowerbody_rl's config/h12/<name>.yaml) or a path"),
    ]

    return LaunchDescription(args + [
        # The policy on the 12 leg joints, through safety_node's lower input. Every
        # argument is passed: Humble includes are not scoped, so an earlier include's
        # config, sim or start_delay would otherwise stand in for one left out.
        include('golem_lowerbody_rl', 'lowerbody.launch.py', {
            'robot': 'h12',
            'config': LaunchConfiguration('lowerbody_config'),
            'policy': LaunchConfiguration('policy'),
            'sim': LaunchConfiguration('sim'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'start_delay': '0.0',
        }, condition=leg_node_condition(LaunchConfiguration('start_position_verified'))),
    ])
