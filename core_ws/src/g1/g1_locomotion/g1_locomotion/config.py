'''Load and check the walking node's YAML config, and start DDS from it'''

import math
import os
from pathlib import Path

import yaml

from g1_locomotion.locomotion import POLICY_DT

# every key is required; an unknown one is an error, so a typo cannot fall
# back to a default
SECTIONS = {
    'topics': ('low_state', 'low_cmd', 'upper_cmd'),
    'network': ('domain_id', 'interface'),
    'startup': ('ramp_seconds', 'tick_rate'),
    'commands': ('vx', 'vy', 'yaw', 'accel', 'timeout', 'height_rate'),
    'faults': ('max_state_gap', 'max_latency'),
}
TOP_LEVEL = tuple(SECTIONS) + ('imu_frame', 'tick_seconds')
COMMANDS = ('vx', 'vy', 'yaw')
# HOMIE was trained on the pelvis IMU and deployed with rt/lowstate's IMU as it
IMU_FRAMES = ('pelvis',)


def _check_keys(mapping, keys, where):
    if not isinstance(mapping, dict):
        raise ValueError(f'{where} must be a mapping')
    missing = [k for k in keys if k not in mapping]
    unknown = [k for k in mapping if k not in keys]
    if missing or unknown:
        raise ValueError(f'{where}: missing keys {missing}, unknown keys {unknown}')


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number, got {value!r}')
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f'{name} must be positive and finite, got {value}')
    return float(value)


def _numbers(value, count, name):
    if (not isinstance(value, list) or len(value) != count or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
        raise ValueError(f'{name} must be a list of {count} numbers, got {value!r}')
    if not all(math.isfinite(v) for v in value):
        raise ValueError(f'{name} must be finite, got {value}')
    return [float(v) for v in value]


def check_config(raw, cmd_range):
    '''
    The config as plain values, checked; raises ValueError. cmd_range is the
    policy's trained [low, high] for vx, vy and yaw (cmd_range in the policy's
    g1.yaml), which the deployment limits must stay inside
    '''
    _check_keys(raw, TOP_LEVEL, 'config')
    for section, keys in SECTIONS.items():
        _check_keys(raw[section], keys, section)

    topics = {key: str(value) for key, value in raw['topics'].items()}
    if not topics['low_cmd'].startswith('rt/safety/'):
        # rt/lowcmd is only ever written by golem_safety
        raise ValueError('topics.low_cmd must be a golem_safety input (rt/safety/...), '
                         f'got {topics["low_cmd"]}')
    if topics['low_cmd'] == topics['upper_cmd']:
        raise ValueError('topics.low_cmd and topics.upper_cmd must differ: upper_cmd is only read')

    if raw['imu_frame'] not in IMU_FRAMES:
        raise ValueError(
            f'imu_frame must be one of {list(IMU_FRAMES)}, got {raw["imu_frame"]!r}: HOMIE was '
            "trained on the pelvis IMU, and a torso IMU's orientation would have to be "
            'carried through the waist joints, which the node does not do')

    commands = raw['commands']
    limits = {}
    for name, (low_trained, high_trained) in zip(COMMANDS, cmd_range):
        low, high = _numbers(commands[name], 2, f'commands.{name}')
        # zero must stay reachable: the node starts at zero velocity and falls
        # back to it when /cmd_vel goes quiet
        if not low_trained <= low <= 0.0 <= high <= high_trained:
            raise ValueError(
                f'commands.{name} {[low, high]} must include 0 and lie inside the trained range '
                f'{[float(low_trained), float(high_trained)]}')
        limits[name] = (low, high)
    accel = _numbers(commands['accel'], len(COMMANDS), 'commands.accel')
    for name, value in zip(COMMANDS, accel):
        _positive(value, f'commands.accel ({name})')

    tick_seconds = _positive(raw['tick_seconds'], 'tick_seconds')
    ticks = POLICY_DT / tick_seconds
    if not (ticks >= 1.0 and abs(ticks - round(ticks)) < 1e-6):
        raise ValueError(f'tick_seconds must divide the {POLICY_DT} s policy period, '
                         f'got {tick_seconds}')
    tick_rate = _numbers(raw['startup']['tick_rate'], 2, 'startup.tick_rate')
    # a tick that keeps to the wall clock must always be accepted
    if not 0.0 < tick_rate[0] <= 1.0 <= tick_rate[1]:
        raise ValueError(f'startup.tick_rate {tick_rate} must be [low, high] with '
                         '0 < low <= 1 <= high')

    domain_id = raw['network']['domain_id']
    if isinstance(domain_id, bool) or not isinstance(domain_id, int) or domain_id < 0:
        raise ValueError(f'network.domain_id must be a nonnegative integer, got {domain_id!r}')
    interface = raw['network']['interface']

    return {
        'topics': topics,
        'network': {
            'domain_id': domain_id,
            'interface': None if interface in (None, '') else str(interface),
        },
        'imu_frame': raw['imu_frame'],
        'tick_seconds': tick_seconds,
        'startup': {
            'ramp_seconds': _positive(raw['startup']['ramp_seconds'], 'startup.ramp_seconds'),
            'tick_rate': tuple(tick_rate),
        },
        'commands': {
            **limits,
            'accel': accel,
            'timeout': _positive(commands['timeout'], 'commands.timeout'),
            'height_rate': _positive(commands['height_rate'], 'commands.height_rate'),
        },
        'faults': {
            key: _positive(raw['faults'][key], f'faults.{key}') for key in SECTIONS['faults']
        },
    }


def load_config(path, cmd_range):
    '''Load the YAML config at path and check it (see check_config)'''
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    try:
        return check_config(raw, cmd_range)
    except ValueError as e:
        raise ValueError(f'{path}: {e}') from None


def resolve_config_path(config_name):
    '''A bare config name (g1.yaml) from the installed config/, or a path as given'''
    path = Path(config_name)
    if not path.suffix:
        path = path.with_suffix('.yaml')
    if path.is_absolute():
        return path
    from ament_index_python.packages import get_package_share_directory
    share = Path(get_package_share_directory('g1_locomotion'))
    if path.parts[0] == 'config':
        return share / path
    return share / 'config' / path


def init_channel_factory(config):
    '''
    Start Unitree DDS on the config's domain and interface, and return the
    domain; ValueError for a config written for another domain when that
    domain is 0, the real robot
    '''
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    # ROS_DOMAIN_ID overrides network.domain_id, as for the other G1 nodes.
    # G1_DDS_INTERFACE (e.g. enp130s0, the laptop's wired port to the robot)
    # overrides network.interface. The Unitree SDK ignores CYCLONEDDS_URI and
    # otherwise auto-picks an interface, which may be WiFi
    env_domain = os.environ.get('ROS_DOMAIN_ID')
    domain_id = int(env_domain) if env_domain is not None else config['network']['domain_id']
    if domain_id == 0 and config['network']['domain_id'] != 0:
        # DDS domain 0 is the real robot: only a config written for it drives it
        raise ValueError(
            f'ROS_DOMAIN_ID=0 is the real robot, but this config is for DDS domain '
            f'{config["network"]["domain_id"]} (sim.yaml is for the sim); use g1.yaml')
    interface = os.environ.get('G1_DDS_INTERFACE') or config['network']['interface']
    if interface:
        ChannelFactoryInitialize(domain_id, interface)
    else:
        ChannelFactoryInitialize(domain_id)
    return domain_id
