from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# Perception model servers (model_server) and the /skill/* action servers.


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use the simulator /clock if true'),
        DeclareLaunchArgument(
            'use_skills', default_value='true',
            description='Start graspgen_server and the h12_skills action servers'),
        DeclareLaunchArgument(
            'use_yolo', default_value='false',
            description='Start the YOLO-World detection publisher'),
        # Per-model debug logging + visualization, shared by the graspgen,
        # gemini, sam, yolo and skills nodes. clear_logs wipes each model's dir
        # on startup so every run begins fresh. Output lands in each package's
        # logs/<model>/.
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),
        # h12_skills per-run base-sway telemetry (run_telemetry.py): each grasp /
        # pick_place run writes logs/runs/<skill>/<stamp>/ with a 10 Hz
        # telemetry.csv + result.json.
        DeclareLaunchArgument('record_runs', default_value='true'),
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
        Node(
            package='model_server',
            executable='gemini_server',
            name='gemini_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
        ),
        Node(
            package='model_server',
            executable='sam_server',
            name='sam_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
        ),

        # YOLO-World open-vocabulary detection. Subscribes to the head + both
        # hand color cameras (its DEFAULT_IMAGE_TOPICS) and publishes a
        # DetectionBundle on <image_topic>/detections at 5 Hz per camera, using
        # the fine-tuned battery weights (weights/yolo_world_battery_best.pt).
        # The vocabulary is read live from the `queries` param, e.g.:
        #   ros2 param set /yolo_server queries "['Bolt','Nut','Screw']"
        Node(
            package='model_server',
            executable='yolo_server',
            name='yolo_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_yolo')),
        ),

        # GraspGenX grasp planning loads a heavy GPU model, so it is gated with
        # the skills that use it; the grasp skill chains
        # gemini -> sam -> graspgen -> frame_task.
        Node(
            package='model_server',
            executable='graspgen_server',
            name='graspgen_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_skills')),
        ),

        # On startup the skills node waits ~10 s each (non-fatal) on the vision
        # and graspgen services, the grippers, and frame_task, then idles ready
        # for goals.
        Node(
            package='h12_skills',
            executable='skills',
            name='h12_skills',
            parameters=[sim_time_param, model_log_params, skills_params],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_skills')),
        ),
    ])
