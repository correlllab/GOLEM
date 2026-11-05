"""
Start the perception model servers and the /skill/* action servers.

All start at once: gemini_server and sam_server always, yolo_server with use_yolo, and
graspgen_server and golem_skills with use_skills. golem_skills waits up to 10 s in
total for the perception and gripper services and the arm actions (a missing one is
not fatal), then serves goals.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Declare the model block's arguments and start its nodes."""
    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true: every node runs on the simulator /clock'),
        DeclareLaunchArgument(
            'use_skills', default_value='true',
            description='true: start graspgen_server and the golem_skills action servers'),
        DeclareLaunchArgument(
            'use_yolo', default_value='false',
            description='true: start the YOLO-World detection publisher'),
        # Each node's debug output lands in its package's logs/<model>/.
        DeclareLaunchArgument('model_logging', default_value='true',
                              description="true: every node saves each call's inputs and "
                                          'outputs (enable_logging)'),
        DeclareLaunchArgument('model_visualization', default_value='true',
                              description='true: every node saves annotated overlay images '
                                          '(enable_visualization)'),
        DeclareLaunchArgument('model_clear_logs', default_value='true',
                              description="true: each node wipes its log directory at "
                                          'start-up (clear_logs)'),
        DeclareLaunchArgument('record_runs', default_value='true',
                              description='true: golem_skills records each grasp and '
                                          'pick_place run (base-sway telemetry) under its '
                                          'logs/runs/<skill>/<stamp>/'),
    ]

    sim_time_param = {'use_sim_time': ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)}
    model_log_params = {
        'enable_logging': ParameterValue(
            LaunchConfiguration('model_logging'), value_type=bool),
        'enable_visualization': ParameterValue(
            LaunchConfiguration('model_visualization'), value_type=bool),
        'clear_logs': ParameterValue(
            LaunchConfiguration('model_clear_logs'), value_type=bool),
    }
    skills_params = {
        'record_runs': ParameterValue(
            LaunchConfiguration('record_runs'), value_type=bool),
    }

    return LaunchDescription(args + [
        # Gemini VLM queries (/gemini_query), which the skills use to find objects.
        Node(
            package='model_server',
            executable='gemini_server',
            name='gemini_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            emulate_tty=True,
        ),
        # SAM3 segmentation (/sam_segment) of what Gemini found.
        Node(
            package='model_server',
            executable='sam_server',
            name='sam_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            emulate_tty=True,
        ),

        # YOLO-World open-vocabulary detection on the head and hand cameras, each
        # on <image_topic>/detections; the vocabulary is its live `queries`
        # parameter (yolo_server's docstring).
        Node(
            package='model_server',
            executable='yolo_server',
            name='yolo_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('use_yolo')),
        ),

        # GraspGenX grasp planning loads a heavy GPU model, so it starts only with
        # the skills that use it; the grasp skill chains
        # gemini -> sam -> graspgen -> frame_task.
        Node(
            package='model_server',
            executable='graspgen_server',
            name='graspgen_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('use_skills')),
        ),

        # The LLM-callable /skill/* action servers.
        Node(
            package='golem_skills',
            executable='skills',
            name='golem_skills',
            parameters=[sim_time_param, model_log_params, skills_params],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('use_skills')),
        ),
    ])
