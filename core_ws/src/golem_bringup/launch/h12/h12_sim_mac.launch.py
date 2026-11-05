"""
The H1-2 sim bringup on the Mac (docker/mac: arm64, CPU only).

It includes the blocks h12_sim.launch.py includes, with the arguments that file
passes them: h12_state, h12_control, h12_lowerbody and golem_slam's navigation.
The GOLEM_* environment variables (docker/mac/docker-compose.yml passes them)
choose the optional ones:

  GOLEM_LOWERBODY   fame | walk | almi: h12_lowerbody with that policy, which
                    pre-poses the legs and releases the sim's elastic band; almi
                    stands and walks on /cmd_vel. Unset or empty: no leg node, and
                    the band holds the robot. Any other value is refused.
  GOLEM_SLAM        1 | true | on (either one): golem_slam's navigation graph for
  GOLEM_NAV2        the H1-2 (FAST-LIO, OctoMap, slam_toolbox, Nav2).

It starts no model servers, golem_skills or rviz: the slim arm64 image does not
build the first two, and launch_ros_mac.sh starts RViz itself (GOLEM_RVIZ=vnc).
Nor h12_sim's camera_link optical frame, which only Isaac's camera needs.
launch_ros_mac.sh launches this file by path:
  ros2 launch /home/code/core_ws/src/golem_bringup/launch/h12/h12_sim_mac.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from golem_ros2_models.paths import urdf_path
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

POLICIES = ('fame', 'walk', 'almi')
ON = ('1', 'true', 'on')


def _include(package, launch_file, launch_arguments):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=launch_arguments.items(),
    )


def _env(name):
    return os.environ.get(name, '').strip().lower()


def generate_launch_description():
    bringup_share = get_package_share_directory('golem_bringup')
    lowerbody = _env('GOLEM_LOWERBODY')
    if lowerbody and lowerbody not in POLICIES:
        raise RuntimeError(
            f'h12_sim_mac: GOLEM_LOWERBODY={lowerbody!r} is not one of '
            f'{", ".join(POLICIES)} (unset or empty starts no leg node)')

    # MuJoCo publishes /clock with sim time. All nodes use it so TF lookups and
    # sensor timestamps are coherent with the simulation.
    sim_time = {'use_sim_time': 'true'}

    actions = [
        _include('golem_bringup', 'h12/h12_state.launch.py', {
            **sim_time,
            'urdf_file': urdf_path('h12', ros=True),
        }),
        _include('golem_bringup', 'h12/h12_control.launch.py', {
            **sim_time,
            'sim': 'true',
            'safety_config': 'sim_safety_split.yaml',
            'frame_task_config': 'sim_safety_split.yaml',
            # No estop in sim, so nothing to wait for.
            'safety_delay': '0.0',
            'frame_task_delay': '0.0',
        }),
    ]
    if lowerbody:
        actions.append(_include('golem_bringup', 'h12/h12_lowerbody.launch.py', {
            **sim_time,
            'start_position_verified': 'true',
            'lowerbody': lowerbody,
            'lowerbody_params': os.path.join(bringup_share, 'config', 'h12', 'lowerbody_sim.yaml'),
        }))
    if _env('GOLEM_SLAM') in ON or _env('GOLEM_NAV2') in ON:
        actions.append(
            _include('golem_slam', 'navigation.launch.py', {**sim_time, 'robot': 'h12'}))
    return LaunchDescription(actions)
