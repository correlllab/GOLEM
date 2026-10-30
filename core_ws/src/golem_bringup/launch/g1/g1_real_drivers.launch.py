"""
The real G1's sensor drivers, run in the laptop's golem_ros container.

The MID-360 driver (livox_ros_driver2), as msg_MID360_launch.py starts it, on
golem_slam/config/g1_MID360_config.json: the laptop's wired address
(192.168.123.222, docs/g1/dds.md) as the host, the lidar at 192.168.123.120,
and no extrinsic, so /livox/lidar and /livox/imu are both in the raw sensor
frame, livox_frame, as the G1 sim publishes them. The URDF's mid360_joint holds
the sensor's upside-down mounting, and g1_fastlio.yaml's extrinsic_R is the
identity for that frame. With 10 Hz frames in CustomMsg (xfer_format 1), which
FAST_LIO's scan_rate and lidar_type expect.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mid360_config = os.path.join(
        get_package_share_directory('golem_slam'), 'config', 'g1_MID360_config.json')
    return LaunchDescription([
        DeclareLaunchArgument('use_lidar', default_value='true',
                              description='Start the MID-360 driver'),
        Node(
            package='livox_ros_driver2',
            executable='livox_ros_driver2_node',
            name='livox_lidar_publisher',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'xfer_format': 1,      # Livox CustomMsg
                'multi_topic': 0,      # /livox/lidar and /livox/imu for every lidar
                'data_src': 0,         # the lidar
                'publish_freq': 10.0,  # Hz: g1_fastlio.yaml's scan_rate
                'output_data_type': 0,
                'frame_id': 'livox_frame',
                'user_config_path': mid360_config,
                'cmdline_input_bd_code': 'livox0000000001',
            }],
            condition=IfCondition(LaunchConfiguration('use_lidar')),
        ),
    ])
