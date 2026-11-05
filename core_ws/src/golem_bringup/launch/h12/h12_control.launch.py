"""
Start the H1-2's command path: [estop_node], safety_node, frame_task_server.

golem_safety's safety_node is the only writer of /lowcmd; golem_upper_ik's
frame_task_server (use_frame_task) commands the arms and torso through it. Each comes
from its own package's launch file (safety.launch.py, upper_ik.launch.py). The order,
on the wall clock: estop_node (use_estop) at once, safety_node after safety_delay,
frame_task_server after frame_task_delay.

upper_ik.launch.py is included first: it refuses a frame_task_config that does not run
with safety_config (golem_upper_ik.pairing) before this file starts any node. Without
use_frame_task it is not included, and any safety_config passes. sim has no default:
every top-level file says whether the robot is simulated.

Every node joins the launching shell's ROS_DOMAIN_ID, domain 0 when it is unset.
Domain 0 is the real robot's command bus, and safety_node refuses sim:=true on it, so a
sim bringup needs a sim domain exported.

SAFETY: with use_estop, safety_node engages no input before estop_node's first sample,
and latches until restart on a sample that says triggered or unplugged. estop_node
publishes nothing until its first attempt to open and probe the serial port ends; when
that attempt fails it publishes unplugged and triggered
(fail_safe_triggered_on_disconnect) until a retry connects. A safety_node that starts
while estop_node is retrying latches, so on the real robot safety_delay must cover the
time estop_node takes to connect. A first attempt that outlasts safety_delay and then
connects only delays the start: safety_node engages nothing and publishes no
/safety/heartbeat, which the leg controller waits for before it pre-poses the legs,
until that first sample. frame_task_server waits up to its controller.startup_timeout
for safety_node to relay its commands.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ROBOT = 'h12'


def _include(package, launch_file, launch_arguments, condition=None):
    """Include a package's launch file with the given arguments."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    """Declare the command path's arguments and start it."""
    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true: every node runs on the simulator /clock'),
        DeclareLaunchArgument(
            'sim',
            description='Required. true for a simulator (safety_node refuses it on ROS '
                        'domain 0), false for the real robot'),
        DeclareLaunchArgument(
            'use_estop', default_value='false',
            description='true: start the serial e-stop bridge and have safety_node '
                        'require and obey it (real robot only)'),
        DeclareLaunchArgument(
            'use_frame_task', default_value='true',
            description='true: start frame_task_server, the arm IK action server '
                        '(/frame_task, /named_config)'),
        DeclareLaunchArgument(
            'safety_config', default_value='relax_safety_split.yaml',
            description='golem_safety config: a name under its config/h12/, or a path. '
                        'With use_frame_task only the safety configs in '
                        'golem_upper_ik/config/h12/pairing.yaml start, by name (a path only '
                        'if its file name is one of them), and sim and real configs do not '
                        'mix'),
        DeclareLaunchArgument(
            'frame_task_config', default_value='safety_split.yaml',
            description='golem_upper_ik config for frame_task_server. With use_frame_task '
                        'it must be the one pairing.yaml pairs with safety_config, by name '
                        '(a path only if its file name is that one)'),
        # SAFETY: on the real robot safety_delay must cover the time estop_node
        # takes to connect (the module docstring), or safety_node latches its estop.
        DeclareLaunchArgument(
            'safety_delay', default_value='3.0',
            description='s (wall clock) before safety_node starts; >= 0'),
        DeclareLaunchArgument(
            'frame_task_delay', default_value='5.0',
            description='s (wall clock) before frame_task_server starts; >= 0'),
    ]

    use_sim_time = LaunchConfiguration('use_sim_time')
    safety_config = LaunchConfiguration('safety_config')
    sim = LaunchConfiguration('sim')

    return LaunchDescription(args + [
        # First: its pairing check refuses before any node starts.
        _include('golem_upper_ik', 'upper_ik.launch.py', {
            'robot': ROBOT,
            'config': LaunchConfiguration('frame_task_config'),
            'safety_config': safety_config,
            'sim': sim,
            'use_sim_time': use_sim_time,
            'start_delay': LaunchConfiguration('frame_task_delay'),
        }, condition=IfCondition(LaunchConfiguration('use_frame_task'))),
        # The hardware e-stop box on serial, published for safety_node.
        Node(
            package='estop',
            executable='estop_node',
            name='estop_node',
            parameters=[{'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('use_estop')),
        ),
        # The only writer of /lowcmd.
        _include('golem_safety', 'safety.launch.py', {
            'robot': ROBOT,
            'config': safety_config,
            'sim': sim,
            'use_estop': LaunchConfiguration('use_estop'),
            'use_sim_time': use_sim_time,
            'start_delay': LaunchConfiguration('safety_delay'),
        }),
    ])
