'''
Robot profiles: load_profile(robot) gives everything robot-specific the controller needs.

The joint table (names in slot order, limits, slot ranges, mode_machine) is golem_safety's
share/golem_safety/robots/<robot>.yaml, the one copy both packages read. The rest (model
files, frames, held and IK joints, named configurations) is golem_upper_ik.robots.<robot>.
'''
from collections.abc import Mapping
from dataclasses import dataclass
import os
from types import MappingProxyType

from ament_index_python.packages import get_package_share_directory
import numpy as np
import yaml

from golem_upper_ik.robots import ROBOTS, robot_module

__all__ = ['ROBOTS', 'RobotProfile', 'load_profile']


# eq=False: compared and hashed by identity, since an array field has no truth value
@dataclass(frozen=True, eq=False)
class RobotProfile:
    robot: str
    mode_machine: int  # the value /lowstate reports and every command carries
    mode_pr: int
    modes: tuple[str, ...]  # golem_safety's modes for this robot
    body_joints: tuple[str, ...]  # in LowCmd slot order
    # per body joint, slot order (read-only): position limits in rad, velocity limits in
    # rad/s, torque limits in N*m
    q_low: np.ndarray
    q_high: np.ndarray
    dq_max: np.ndarray
    tau_max: np.ndarray
    lower_slots: range
    upper_slots: range
    urdf_path: str  # for Pinocchio: relative mesh paths
    urdf_ros_path: str  # for robot_state_publisher and RViz: package:// mesh URIs
    collision: tuple[str, str] | None  # (sphere URDF, SRDF); None: no self-collision model
    ee_frames: tuple[str, str]  # (left, right)
    planner_frames: tuple[str, str]  # (left, right)
    imu_frame: str
    held_joints: tuple[str, ...]  # upper-body joints the IK does not solve for
    # the controller's shutdown command keeps the held joints at their hold, gains and torque
    # when golem_safety's output is /lowcmd; otherwise it turns them off with the IK joints
    shutdown_keeps_held_joints_on_lowcmd: bool
    enabled_joints: tuple[str, ...]  # joints the IK solves for, the named configs' order
    # read-only: name -> read-only array over enabled_joints, rad
    named_configs: Mapping[str, np.ndarray]
    startup_sequence: tuple[str, ...]  # named configs the controller moves through at start


def _read_only(values) -> np.ndarray:
    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array


def _robot_data(robot: str) -> dict:
    path = os.path.join(get_package_share_directory('golem_safety'), 'robots', f'{robot}.yaml')
    with open(path, encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if data.get('robot') != robot:
        raise ValueError(f"{path} is the data of robot '{data.get('robot')}', not '{robot}'")
    return data


def load_profile(robot: str) -> RobotProfile:
    '''The profile of robot ('h12' or 'g1'); ValueError for any other name'''
    module = robot_module(robot)
    data = _robot_data(robot)
    joints = sorted(data['joints'], key=lambda joint: joint['slot'])
    slots = [joint['slot'] for joint in joints]
    if slots != list(range(len(joints))):
        raise ValueError(f"robot {robot}: golem_safety's joint slots must be "
                         f'0..{len(joints) - 1}, each once, not {slots}')
    body_joints = tuple(joint['name'] for joint in joints)
    lower_first, lower_last = data['lower_slots']
    upper_first, upper_last = data['upper_slots']
    upper_slots = range(upper_first, upper_last + 1)

    upper_joints = tuple(body_joints[slot] for slot in upper_slots)
    held_joints = tuple(module.HELD_JOINTS)
    enabled_joints = tuple(module.ENABLED_JOINTS)
    # the controller drives the upper slots and nothing else: each upper joint is either
    # solved by the IK or held
    if sorted(held_joints + enabled_joints) != sorted(upper_joints):
        raise ValueError(
            f'robot {robot}: the held joints {held_joints} and IK joints {enabled_joints} '
            f'must be the upper-body joints {upper_joints}, each once')
    named_configs = MappingProxyType(
        {name: _read_only(q) for name, q in module.NAMED_CONFIGS.items()})
    for name, q in named_configs.items():
        if q.shape != (len(enabled_joints),):
            raise ValueError(f'robot {robot}: named config {name} has {q.size} values, '
                             f'not one per IK joint ({len(enabled_joints)})')
    startup_sequence = tuple(module.STARTUP_SEQUENCE)
    missing = [name for name in startup_sequence if name not in named_configs]
    if missing:
        raise ValueError(f'robot {robot}: start-up postures {missing} are not named configs')

    paths = module.model_paths()
    return RobotProfile(
        robot=robot,
        mode_machine=int(data['mode_machine']),
        mode_pr=int(data['mode_pr']),
        modes=tuple(data['modes']),
        body_joints=body_joints,
        q_low=_read_only([joint['position'][0] for joint in joints]),
        q_high=_read_only([joint['position'][1] for joint in joints]),
        dq_max=_read_only([joint['velocity'] for joint in joints]),
        tau_max=_read_only([joint['effort'] for joint in joints]),
        lower_slots=range(lower_first, lower_last + 1),
        upper_slots=upper_slots,
        urdf_path=paths['urdf_path'],
        urdf_ros_path=paths['urdf_ros_path'],
        collision=None if paths['collision'] is None else tuple(paths['collision']),
        ee_frames=tuple(module.EE_FRAMES),
        planner_frames=tuple(module.PLANNER_FRAMES),
        imu_frame=module.IMU_FRAME,
        held_joints=held_joints,
        shutdown_keeps_held_joints_on_lowcmd=bool(module.SHUTDOWN_KEEPS_HELD_JOINTS_ON_LOWCMD),
        enabled_joints=enabled_joints,
        named_configs=named_configs,
        startup_sequence=startup_sequence,
    )
