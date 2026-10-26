from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

import os
from g1_control.utility.path_definition_ros import PACKAGE_PATH

def generate_launch_description():
    rviz_config_path = f'{PACKAGE_PATH}/rviz/default.rviz'

    return LaunchDescription([
        TimerAction(  # slight delay to ensure robot_description is set
            period=1.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', rviz_config_path] if os.path.exists(rviz_config_path) else [],
                    output='screen'
                )
            ]
        )
    ])
