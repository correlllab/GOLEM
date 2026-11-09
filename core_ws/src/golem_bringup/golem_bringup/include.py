"""Include another package's launch file: the one helper every golem_bringup launch file uses."""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.condition import Condition
from launch.launch_description_sources import PythonLaunchDescriptionSource

__all__ = ['include']


def include(package: str, launch_file: str, launch_arguments: dict | None = None,
            condition: Condition | None = None) -> IncludeLaunchDescription:
    """
    Include <package's share>/launch/<launch_file> with the given arguments.

    launch_file may hold a subdirectory ('h12/h12_control.launch.py'); condition, when
    given, gates the whole include.
    """
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=(launch_arguments or {}).items(),
        condition=condition,
    )
