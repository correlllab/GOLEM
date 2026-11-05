"""
Start the real H1-2's onboard sensor drivers and grippers.

All start at once, each behind its argument: the MID-360 driver (use_lidar), the head
RealSense and its static TF (use_head_camera) and both Magpie gripper nodes
(use_grippers). The hand cameras start on the companion desktop
(h12_real_desktop.launch.py).

The MID-360 hangs upside down; the driver applies no extrinsic rotation
(golem_slam/config/h12/MID360_config.json), so the points (frame livox_frame) and the
IMU are both in the sensor's raw frame, the URDF's livox_link and livox_imu_link, which
golem_slam's H1-2 configs expect.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Declare the driver toggles and start the drivers."""
    bringup_share = get_package_share_directory('golem_bringup')
    mid360_config = os.path.join(
        get_package_share_directory('golem_slam'), 'config', 'h12', 'MID360_config.json')

    # livox_ros_driver2's msg_MID360_launch.py parameters, on the onboard network's
    # addresses in golem_slam/config/h12/MID360_config.json.
    livox_mid360_params = [
        {'xfer_format': 1},      # Livox CustomMsg, which FAST-LIO reads
        {'multi_topic': 0},      # one points topic and one IMU topic for every lidar
        {'data_src': 0},         # the lidar
        {'publish_freq': 30.0},  # Hz
        {'output_data_type': 0},
        {'frame_id': 'livox_frame'},
        {'user_config_path': mid360_config},
        {'cmdline_input_bd_code': 'livox0000000001'},
    ]

    # The MID-360's points and IMU.
    livox_mid360 = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        emulate_tty=True,
        parameters=livox_mid360_params,
        condition=IfCondition(LaunchConfiguration('use_lidar')),
    )

    # The head RealSense and its camera -> link static TF.
    realsense_share = get_package_share_directory('cl_realsense')
    realsense_cams = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_share, 'launch', 'h12_headcamera.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('use_head_camera')),
    )

    # Both Magpie grippers, from the file h12_safety_grippers.launch.py includes
    # too, so their ports and calibration live in one place.
    grippers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h12', 'h12_grippers.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('use_grippers')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_lidar', default_value='true',
                              description='true: start the MID-360 driver'),
        DeclareLaunchArgument('use_head_camera', default_value='true',
                              description='true: start the head RealSense and its static TF'),
        DeclareLaunchArgument('use_grippers', default_value='true',
                              description='true: start the left and right gripper nodes'),
        livox_mid360,
        realsense_cams,
        grippers,
    ])
