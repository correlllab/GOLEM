"""
Load each robot's leg profile: its joint table and the leg data this package keeps.

The joint table (names in /lowcmd slot order, mode_machine, mode_pr, the lower-body slots)
is golem_safety's share/golem_safety/robots/<robot>.yaml, the one copy every package reads.
The rest (leg joints in action order, the policies the robot may run) is this
package's share/golem_lowerbody_rl/config/<robot>/robot.yaml, whose keys are documented in
config/example.robot.yaml.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import os

from ament_index_python.packages import get_package_share_directory
import numpy as np
import yaml

__all__ = ['ROBOT_KEYS', 'RobotProfile', 'known_robots', 'load_profile']

# robot.yaml must hold exactly these keys
ROBOT_KEYS = ('legs', 'policies', 'default_policy')


@dataclass(frozen=True)
class RobotProfile:
    """Hold one robot's joint table and leg data."""

    robot: str
    mode_machine: int  # the value /lowstate reports and every command carries
    mode_pr: int
    body_joints: tuple[str, ...]  # in /lowcmd slot order
    legs: tuple[str, ...]  # the commanded leg joints, in the policies' action order
    policies: tuple[str, ...]  # the policies this robot may run
    default_policy: str

    def slots(self, names: Sequence[str]) -> np.ndarray:
        """
        Return the /lowcmd slot of each joint in names, in order.

        Raises KeyError naming the first joint that is not one of body_joints.
        """
        index = {name: slot for slot, name in enumerate(self.body_joints)}
        for name in names:
            if name not in index:
                raise KeyError(f"'{name}' is not a joint of robot {self.robot} "
                               "(golem_safety's robots/<robot>.yaml)")
        return np.array([index[name] for name in names], dtype=np.intp)


def _safety_dir() -> str:
    """Return golem_safety's installed robots/ directory."""
    return os.path.join(get_package_share_directory('golem_safety'), 'robots')


def _config_dir() -> str:
    """Return this package's installed config/ directory."""
    return os.path.join(get_package_share_directory('golem_lowerbody_rl'), 'config')


def known_robots() -> tuple[str, ...]:
    """
    Return the robots with data in both packages, sorted.

    A robot is known when golem_safety has robots/<robot>.yaml and this package has
    config/<robot>/robot.yaml.
    """
    names = (name[:-5] for name in os.listdir(_safety_dir()) if name.endswith('.yaml'))
    return tuple(sorted(name for name in names
                        if os.path.isfile(os.path.join(_config_dir(), name, 'robot.yaml'))))


def _safety_data(robot: str) -> dict:
    """Return golem_safety's robots/<robot>.yaml; ValueError if it names another robot."""
    path = os.path.join(_safety_dir(), f'{robot}.yaml')
    with open(path, encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if data.get('robot') != robot:
        raise ValueError(f"{path} is the data of robot '{data.get('robot')}', not '{robot}'")
    return data


def _names(where: str, key: str, value) -> tuple[str, ...]:
    """Return value as a tuple if it is a non-empty list of distinct non-empty names."""
    ok = (isinstance(value, list) and value
          and all(isinstance(item, str) and item for item in value))
    if not ok:
        raise ValueError(f"{where}: '{key}' must be a non-empty list of names, not {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{where}: '{key}' must name each entry once, not {value!r}")
    return tuple(value)


def _robot_config(robot: str, path: str) -> dict:
    """
    Return the checked data of a robot.yaml, keyed by ROBOT_KEYS.

    Raises ValueError, naming the file and the key, for a missing or unknown key or a
    value of the wrong type.
    """
    with open(path, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    where = f'robot {robot} ({path})'
    if not isinstance(raw, dict):
        raise ValueError(f'{where}: the file must hold a mapping')
    for key in raw:
        if key not in ROBOT_KEYS:
            raise ValueError(f"{where}: unknown key '{key}'")
    for key in ROBOT_KEYS:
        if key not in raw:
            raise ValueError(f"{where}: missing key '{key}'")
    if not isinstance(raw['default_policy'], str) or not raw['default_policy']:
        raise ValueError(f"{where}: 'default_policy' must be a name, not "
                         f"{raw['default_policy']!r}")
    config = dict(raw)
    config['legs'] = _names(where, 'legs', raw['legs'])
    config['policies'] = _names(where, 'policies', raw['policies'])
    if raw['default_policy'] not in config['policies']:
        raise ValueError(f"{where}: default_policy '{raw['default_policy']}' is not one of "
                         f"policies {list(config['policies'])}")
    return config


def load_profile(robot: str, robot_yaml: str | None = None) -> RobotProfile:
    """
    Return the profile of robot, one of known_robots().

    robot_yaml is the robot.yaml to read instead of the installed
    config/<robot>/robot.yaml. Raises ValueError for an unknown robot, a robot.yaml that
    does not hold exactly ROBOT_KEYS, or leg joints that are not the robot's lower-body
    joints.
    """
    if robot not in known_robots():
        raise ValueError(f"unknown robot '{robot}': one of {', '.join(known_robots())}")
    path = robot_yaml or os.path.join(_config_dir(), robot, 'robot.yaml')
    config = _robot_config(robot, path)
    data = _safety_data(robot)
    joints = sorted(data['joints'], key=lambda joint: joint['slot'])
    slots = [joint['slot'] for joint in joints]
    if slots != list(range(len(joints))):
        raise ValueError(f"robot {robot}: golem_safety's joint slots must be "
                         f'0..{len(joints) - 1}, each once, not {slots}')
    body_joints = tuple(joint['name'] for joint in joints)
    lower_first, lower_last = data['lower_slots']
    lower_joints = body_joints[lower_first:lower_last + 1]
    # SAFETY: the leg node writes golem_safety's lower-body input, which relays the lower
    # slots only, so every leg is a lower-slot joint and every lower slot is commanded
    if sorted(config['legs']) != sorted(lower_joints):
        raise ValueError(f'robot {robot} ({path}): legs {list(config["legs"])} must be the '
                         f'lower-body joints {list(lower_joints)}, each once')
    return RobotProfile(
        robot=robot,
        mode_machine=int(data['mode_machine']),
        mode_pr=int(data['mode_pr']),
        body_joints=body_joints,
        legs=config['legs'],
        policies=config['policies'],
        default_policy=config['default_policy'],
    )
