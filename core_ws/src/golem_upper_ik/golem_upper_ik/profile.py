"""Load each robot's profile: everything robot-specific the controller needs.

The joint table (names in slot order, limits, upper slots, mode_machine) is golem_safety's
share/golem_safety/robots/<robot>.yaml, the one copy both packages read. The rest (frames,
held and IK joints, named configurations, controller-config defaults) is this package's
share/golem_upper_ik/config/<robot>/robot.yaml (its keys: config/<robot>/example.robot.yaml),
and the model files are golem_ros2_models'.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import os
from types import MappingProxyType

from ament_index_python.packages import get_package_share_directory
from golem_ros2_models import paths as model_files
import numpy as np
import yaml

__all__ = ['RobotProfile', 'known_robots', 'load_profile', 'robot_config']

# robot.yaml must hold exactly these keys
ROBOT_CONFIG_KEYS = (
    'display_name', 'enabled_joints', 'held_joints', 'shutdown_keeps_held_joints', 'ee_frames',
    'planner_frames', 'imu_frame', 'named_configs', 'startup_sequence', 'config_defaults',
)


# eq=False: compared and hashed by identity, since an array field has no truth value
@dataclass(frozen=True, eq=False)
class RobotProfile:
    """Hold one robot's joint table, model files, frames and named configurations.

    The joint arrays (q_low, q_high in rad; dq_max in rad/s; tau_max in N*m) hold one
    read-only value per body joint in /lowcmd slot order. named_configs maps each name to a
    read-only array over enabled_joints, rad.
    """

    robot: str
    mode_machine: int  # the value /lowstate reports and every command carries
    mode_pr: int
    body_joints: tuple[str, ...]  # in /lowcmd slot order
    q_low: np.ndarray
    q_high: np.ndarray
    dq_max: np.ndarray
    tau_max: np.ndarray
    upper_slots: range
    urdf_path: str  # the Pinocchio URDF, whose mesh paths are relative to it
    collision: tuple[str, str] | None  # (sphere URDF, SRDF); None: no self-collision model
    ee_frames: tuple[str, str]  # (left, right)
    planner_frames: tuple[str, str]  # (left, right)
    imu_frame: str
    held_joints: tuple[str, ...]  # upper-body joints the IK does not solve for
    # SAFETY: true keeps the held joints at their hold, gains and torque in the controller's
    # shutdown command (a walking policy may balance on them); false turns them off with the
    # IK joints
    shutdown_keeps_held_joints: bool
    enabled_joints: tuple[str, ...]  # joints the IK solves for, the named configs' order
    named_configs: Mapping[str, np.ndarray]
    startup_sequence: tuple[str, ...]  # named configs the controller moves through at start


def _read_only(values) -> np.ndarray:
    """Return values as a read-only float64 array."""
    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array


def _robot_data(robot: str) -> dict:
    """Return golem_safety's robots/<robot>.yaml; ValueError if it names another robot."""
    path = os.path.join(get_package_share_directory('golem_safety'), 'robots', f'{robot}.yaml')
    with open(path, encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if data.get('robot') != robot:
        raise ValueError(f"{path} is the data of robot '{data.get('robot')}', not '{robot}'")
    return data


def known_robots() -> tuple[str, ...]:
    """Return the robots with data in both packages, sorted.

    A robot is known when golem_safety has robots/<robot>.yaml and this package has
    config/<robot>/robot.yaml.
    """
    safety = os.path.join(get_package_share_directory('golem_safety'), 'robots')
    upper = os.path.join(get_package_share_directory('golem_upper_ik'), 'config')
    names = (name[:-5] for name in os.listdir(safety) if name.endswith('.yaml'))
    return tuple(sorted(name for name in names
                        if os.path.isfile(os.path.join(upper, name, 'robot.yaml'))))


def _unknown_robot(robot: str) -> ValueError:
    """Return the error for a robot that is not one of known_robots()."""
    return ValueError(f"unknown robot '{robot}': one of {', '.join(known_robots())}")


def _strings(where: str, key: str, value, count: int | None = None) -> list[str]:
    """Return value if it is a list (of count, when given) of non-empty names."""
    ok = (isinstance(value, list) and all(isinstance(item, str) and item for item in value)
          and (count is None or len(value) == count))
    if not ok:
        size = '' if count is None else f' of {count}'
        raise ValueError(f"{where}: '{key}' must be a list{size} of names, not {value!r}")
    return value


def _number(where: str, key: str, value) -> float:
    """Return value if it is an int or float (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: '{key}' must be a number, not {value!r}")
    return value


def robot_config(robot: str) -> dict:
    """Return the checked data of config/<robot>/robot.yaml, keyed by ROBOT_CONFIG_KEYS.

    Raises ValueError, naming the robot and the key, for an unknown robot, a missing or
    unknown key, or a value of the wrong type.
    """
    if robot not in known_robots():
        raise _unknown_robot(robot)
    path = os.path.join(get_package_share_directory('golem_upper_ik'), 'config', robot,
                        'robot.yaml')
    with open(path, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    where = f'robot {robot} ({path})'
    if not isinstance(raw, dict):
        raise ValueError(f'{where}: the file must hold a mapping')
    for key in raw:
        if key not in ROBOT_CONFIG_KEYS:
            raise ValueError(f"{where}: unknown key '{key}'")
    for key in ROBOT_CONFIG_KEYS:
        if key not in raw:
            raise ValueError(f"{where}: missing key '{key}'")

    config = dict(raw)
    if not isinstance(raw['display_name'], str) or not raw['display_name']:
        raise ValueError(f"{where}: 'display_name' must be a name, not {raw['display_name']!r}")
    for key in ('enabled_joints', 'held_joints', 'startup_sequence'):
        _strings(where, key, raw[key])
    for key in ('ee_frames', 'planner_frames'):
        _strings(where, key, raw[key], 2)
    if not isinstance(raw['imu_frame'], str) or not raw['imu_frame']:
        raise ValueError(f"{where}: 'imu_frame' must be a name, not {raw['imu_frame']!r}")
    if not isinstance(raw['shutdown_keeps_held_joints'], bool):
        raise ValueError(f"{where}: 'shutdown_keeps_held_joints' must be true or false, not "
                         f"{raw['shutdown_keeps_held_joints']!r}")

    named = raw['named_configs']
    if not isinstance(named, dict) or not all(isinstance(name, str) for name in named):
        raise ValueError(f"{where}: 'named_configs' must map names to lists of numbers")
    for name, values in named.items():
        if not isinstance(values, list):
            raise ValueError(f"{where}: 'named_configs': '{name}' must be a list of numbers, "
                             f'not {values!r}')
    config['named_configs'] = {
        name: [float(_number(where, f'named_configs.{name}', value)) for value in values]
        for name, values in named.items()}

    defaults = raw['config_defaults']
    if not isinstance(defaults, dict):
        raise ValueError(f"{where}: 'config_defaults' must map config keys to numbers")
    config['config_defaults'] = {
        str(key): _number(where, f'config_defaults.{key}', value)
        for key, value in defaults.items()}
    return config


def load_profile(robot: str) -> RobotProfile:
    """Return the profile of robot, one of known_robots(); ValueError for any other name."""
    config = robot_config(robot)
    data = _robot_data(robot)
    joints = sorted(data['joints'], key=lambda joint: joint['slot'])
    slots = [joint['slot'] for joint in joints]
    if slots != list(range(len(joints))):
        raise ValueError(f"robot {robot}: golem_safety's joint slots must be "
                         f'0..{len(joints) - 1}, each once, not {slots}')
    body_joints = tuple(joint['name'] for joint in joints)
    upper_first, upper_last = data['upper_slots']
    upper_slots = range(upper_first, upper_last + 1)

    upper_joints = tuple(body_joints[slot] for slot in upper_slots)
    held_joints = tuple(config['held_joints'])
    enabled_joints = tuple(config['enabled_joints'])
    # SAFETY: the controller drives the upper slots and nothing else, so each upper joint is
    # either solved by the IK or held, never left uncommanded
    if sorted(held_joints + enabled_joints) != sorted(upper_joints):
        raise ValueError(
            f'robot {robot}: the held joints {held_joints} and IK joints {enabled_joints} '
            f'must be the upper-body joints {upper_joints}, each once')
    named_configs = MappingProxyType(
        {name: _read_only(q) for name, q in config['named_configs'].items()})
    for name, q in named_configs.items():
        if q.shape != (len(enabled_joints),):
            raise ValueError(f'robot {robot}: named config {name} has {q.size} values, '
                             f'not one per IK joint ({len(enabled_joints)})')
    startup_sequence = tuple(config['startup_sequence'])
    missing = [name for name in startup_sequence if name not in named_configs]
    if missing:
        raise ValueError(f'robot {robot}: start-up postures {missing} are not named configs')

    files = model_files.model_paths(robot)
    return RobotProfile(
        robot=robot,
        mode_machine=int(data['mode_machine']),
        mode_pr=int(data['mode_pr']),
        body_joints=body_joints,
        q_low=_read_only([joint['position'][0] for joint in joints]),
        q_high=_read_only([joint['position'][1] for joint in joints]),
        dq_max=_read_only([joint['velocity'] for joint in joints]),
        tau_max=_read_only([joint['effort'] for joint in joints]),
        upper_slots=upper_slots,
        urdf_path=files['urdf_path'],
        collision=None if files['collision'] is None else tuple(files['collision']),
        ee_frames=tuple(config['ee_frames']),
        planner_frames=tuple(config['planner_frames']),
        imu_frame=config['imu_frame'],
        held_joints=held_joints,
        shutdown_keeps_held_joints=bool(config['shutdown_keeps_held_joints']),
        enabled_joints=enabled_joints,
        named_configs=named_configs,
        startup_sequence=startup_sequence,
    )
