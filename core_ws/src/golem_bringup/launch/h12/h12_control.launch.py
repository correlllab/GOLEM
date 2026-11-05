import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# The H1-2's command path: [estop_node] -> golem_safety's safety_node, the only
# writer of /lowcmd -> golem_upper_ik's frame_task_server (use_frame_task),
# each package's node from its own launch file (safety.launch.py,
# upper_ik.launch.py).
#
# Every node is a ROS node on the launching shell's ROS_DOMAIN_ID, domain 0 when
# it is unset. Domain 0 is the real robot's command bus, and safety_node refuses
# sim:=true on it, so a sim bringup needs a sim domain exported.
#
# sim has no default: every top-level file says whether the robot is simulated.
# upper_ik.launch.py is included first: it refuses a frame_task_config that does
# not run with safety_config (golem_upper_ik.pairing) before this file starts
# any node. Without use_frame_task it is not included, and any safety_config
# passes.

ROBOT = 'h12'


def _include(package, launch_file, launch_arguments, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
        condition=condition,
    )


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
        # estop_node takes to connect. A first attempt that outlasts safety_delay
        # and then connects only delays the start: safety_node engages nothing and
        # publishes no /safety/heartbeat, which the leg controller waits for before
        # it pre-poses the legs, until that first sample.
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

    use_sim_time = LaunchConfiguration('use_sim_time')
    safety_config = LaunchConfiguration('safety_config')
    sim = LaunchConfiguration('sim')

    return LaunchDescription(args + [
        _include('golem_upper_ik', 'upper_ik.launch.py', {
            'robot': ROBOT,
            'config': LaunchConfiguration('frame_task_config'),
            'safety_config': safety_config,
            'sim': sim,
            'use_sim_time': use_sim_time,
            'start_delay': LaunchConfiguration('frame_task_delay'),
        }, condition=IfCondition(LaunchConfiguration('use_frame_task'))),
        Node(
            package='estop',
            executable='estop_node',
            name='estop_node',
            parameters=[{'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('use_estop')),
        ),
        _include('golem_safety', 'safety.launch.py', {
            'robot': ROBOT,
            'config': safety_config,
            'sim': sim,
            'use_estop': LaunchConfiguration('use_estop'),
            'use_sim_time': use_sim_time,
            'start_delay': LaunchConfiguration('safety_delay'),
        }),
    ])
