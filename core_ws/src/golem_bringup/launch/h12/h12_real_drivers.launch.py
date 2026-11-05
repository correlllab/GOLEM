import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Equivalent of `ros2 launch livox_ros_driver2 msg_MID360_launch.py`,
    # inlined so the MID360 driver comes up as part of this bringup. Params
    # mirror msg_MID360_launch.py; user_config_path points at the
    # humanoid-validated MID360_config.json stored under golem_slam/config/h12
    # (Unitree onboard net 192.168.123.164 host / 192.168.123.120 lidar). The
    # lidar hangs upside down; the driver applies no extrinsic rotation, so the
    # points (frame 'livox_frame') and the IMU are both in the sensor's raw
    # frame, the URDF's livox_link and livox_imu_link.
    bringup_share = get_package_share_directory('golem_bringup')
    mid360_config = os.path.join(
        get_package_share_directory('golem_slam'), 'config', 'h12', 'MID360_config.json')

    livox_mid360_params = [
        {'xfer_format': 1},      # 0-Pointcloud2(PointXYZRTL), 1-customized pointcloud format
        {'multi_topic': 0},      # 0-All LiDARs share the same topic, 1-One LiDAR one topic
        {'data_src': 0},         # 0-lidar, others-Invalid data src
        {'publish_freq': 30.0},  # publish frequency (Hz)
        {'output_data_type': 0},
        {'frame_id': 'livox_frame'},
        {'user_config_path': mid360_config},
        {'cmdline_input_bd_code': 'livox0000000001'},
    ]

    livox_mid360 = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        emulate_tty=True,
        parameters=livox_mid360_params,
        condition=IfCondition(LaunchConfiguration('use_lidar')),
    )

    # Equivalent of `ros2 launch cl_realsense h12_headcamera.launch.py` — brings
    # up only the head RealSense camera and its camera->link static TF. The hand
    # cameras start on the companion desktop (h12_real_desktop.launch.py),
    # so the onboard driver bringup only owns the head cam.
    realsense_share = get_package_share_directory('cl_realsense')
    realsense_cams = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_share, 'launch', 'h12_headcamera.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('use_head_camera')),
    )

    # Magpie grippers, one node per side. Both nodes (ports, per-gripper finger
    # calibration) live in h12_grippers.launch.py so the actuation-only bringup
    # (h12_safety_grippers.launch.py) can reuse the exact same configuration.
    grippers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h12', 'h12_grippers.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('use_grippers')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_lidar', default_value='true',
                              description='Start the MID360 driver'),
        DeclareLaunchArgument('use_head_camera', default_value='true',
                              description='Start the head RealSense and its static TF'),
        DeclareLaunchArgument('use_grippers', default_value='true',
                              description='Start the left and right gripper nodes'),
        livox_mid360,
        realsense_cams,
        grippers,
    ])
