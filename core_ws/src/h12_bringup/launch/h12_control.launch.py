from golem_upper_ik.pairing import pairing_refusal
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import evaluate_condition_expression, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# The H1-2's command path: [estop_node] -> golem_safety's safety_node, the only
# writer of /lowcmd -> golem_upper_ik's frame_task_server (use_frame_task).
#
# Every node is a ROS node on the launching shell's ROS_DOMAIN_ID, domain 0 when
# it is unset. Domain 0 is the real robot's command bus, and safety_node refuses
# sim:=true on it, so a sim bringup needs a sim domain exported.
#
# sim has no default: every top-level file says whether the robot is simulated.
# check_pairing runs before this file starts any node.

ROBOT = 'h12'


def check_pairing(context):
    """
    Refuse a frame_task_config that does not run with safety_config.

    Only with use_frame_task: the bringups that run no IK pass any safety_config.
    The rules are golem_upper_ik.pairing's. Raises RuntimeError with the reason;
    starts nothing.
    """
    def flag(name):
        return evaluate_condition_expression(context, [LaunchConfiguration(name)])

    if not flag('use_frame_task'):
        return []
    refusal = pairing_refusal(
        ROBOT,
        LaunchConfiguration('frame_task_config').perform(context),
        LaunchConfiguration('safety_config').perform(context),
        sim=flag('sim'))
    if refusal is not None:
        raise RuntimeError(f'h12_control: not starting: {refusal}')
    return []


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use the simulator /clock if true'),
        DeclareLaunchArgument(
            'sim',
            description='true for a simulator (safety_node refuses it on ROS domain 0), '
                        'false for the real robot'),
        DeclareLaunchArgument(
            'use_estop', default_value='false',
            description='Start the serial e-stop bridge, and have safety_node read it '
                        '(real robot only)'),
        DeclareLaunchArgument(
            'use_frame_task', default_value='true',
            description='Start the frame_task_server arm IK action server'),
        DeclareLaunchArgument(
            'safety_config', default_value='relax_safety_split.yaml',
            description='golem_safety config: a name under its config/h12/, or a path'),
        DeclareLaunchArgument(
            'frame_task_config', default_value='safety_split.yaml',
            description='golem_upper_ik config for frame_task_server: a name under its '
                        'config/h12/, or a path'),
        # The H1-2's start-up order, on the wall clock: estop_node at once, then
        # safety_node after safety_delay, then frame_task_server after
        # frame_task_delay.
        #
        # With use_estop, safety_node engages no input before estop_node's first
        # sample, and latches until restart on a sample that says triggered or
        # unplugged. estop_node publishes nothing until its first attempt to open
        # and probe the serial port ends; when that attempt fails it publishes
        # unplugged, and triggered (fail_safe_triggered_on_disconnect), until a
        # retry connects. A safety_node that starts while estop_node is retrying
        # latches, so on the real robot safety_delay must cover the time
        # estop_node takes to connect.
        #
        # frame_task_server's start-up waits up to its controller.startup_timeout
        # for safety_node to relay its commands.
        DeclareLaunchArgument(
            'safety_delay', default_value='3.0',
            description='Seconds (wall clock) before starting safety_node'),
        DeclareLaunchArgument(
            'frame_task_delay', default_value='5.0',
            description='Seconds (wall clock) before starting frame_task_server'),
    ]

    def value(name, value_type):
        return ParameterValue(LaunchConfiguration(name), value_type=value_type)

    sim_time_param = {'use_sim_time': value('use_sim_time', bool)}

    return LaunchDescription(args + [
        OpaqueFunction(function=check_pairing),
        Node(
            package='estop',
            executable='estop_node',
            name='estop_node',
            parameters=[sim_time_param],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('use_estop')),
        ),
        TimerAction(
            period=LaunchConfiguration('safety_delay'),
            actions=[
                Node(
                    package='golem_safety',
                    executable='safety_node',
                    name='safety_node',
                    parameters=[{
                        'robot': ROBOT,
                        'config': value('safety_config', str),
                        'sim': value('sim', bool),
                        'use_estop': value('use_estop', bool),
                    }, sim_time_param],
                    output='screen',
                    emulate_tty=True,
                ),
            ],
        ),
        TimerAction(
            period=LaunchConfiguration('frame_task_delay'),
            actions=[
                Node(
                    package='golem_upper_ik',
                    executable='frame_task_server',
                    name='frame_task_server',
                    parameters=[{
                        'robot': ROBOT,
                        'config': value('frame_task_config', str),
                    }, sim_time_param],
                    output='screen',
                    emulate_tty=True,
                    condition=IfCondition(LaunchConfiguration('use_frame_task')),
                ),
            ],
        ),
    ])
