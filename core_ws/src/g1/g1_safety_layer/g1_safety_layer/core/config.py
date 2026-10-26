'''YAML config loader and DDS channel initialization'''

import os
import time

import yaml
import numpy as np
from typing import Any
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_safety_layer.core.joint_limits import (
    MOTOR_COUNT,
    JOINT_NAMES,
    WAIST_SLOTS,
    URDF_POSITION_LIMITS,
    URDF_TORQUE_LIMITS,
    URDF_VELOCITY_LIMITS,
)

# split_mode and full_body_mode estop commands; arm_sdk_mode hands back to Unitree
ESTOP_RESPONSES = ('zero', 'damp')
# what each config is written for; a sim config refuses DDS domain 0, the real robot
ROBOTS = ('real', 'sim')
# a mounted directory that keeps the logs; each run writes its own subdirectory
LOG_DIR_ENV = 'G1_SAFETY_LOG_DIR'


def _load_yaml(path: Path) -> dict[str, Any]:
    '''Load yaml file and ensure it's a dict'''
    with path.open('r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError('config root must be a mapping')
    return loaded


def _load_joint_config(value: Any, field_name: str) -> np.ndarray:
    '''Expand a scalar value to a vector of MOTOR_COUNT, or validate list input'''
    # single value for all joints
    if isinstance(value, (int, float)):
        return np.full((MOTOR_COUNT,), float(value), dtype=np.float64)
    # list of values for each joint
    if isinstance(value, list) and len(value) == MOTOR_COUNT:
        return np.asarray(value, dtype=np.float64)
    # mismatch
    raise ValueError(
        f'{field_name} must be a number or a list with {MOTOR_COUNT} entries'
    )


def _positive_finite(value: Any, field_name: str) -> float:
    '''Parse a positive, finite number or raise'''
    number = float(value)
    if not (np.isfinite(number) and number > 0.0):
        raise ValueError(f'{field_name} must be positive and finite')
    return number


def _derive_q_limits(position_offset: np.ndarray) -> np.ndarray:
    '''Derive low and high position limits for each joint from URDF limits and config offset'''
    q_limits = np.zeros((MOTOR_COUNT, 2), dtype=np.float64)
    for i in range(MOTOR_COUNT):
        low = float(URDF_POSITION_LIMITS[i]['low']) + position_offset[i]
        high = float(URDF_POSITION_LIMITS[i]['high']) - position_offset[i]
        if low >= high:
            raise ValueError(
                f'position_offset too large for joint {i} {JOINT_NAMES[i]}'
            )
        q_limits[i, 0] = low
        q_limits[i, 1] = high
    return q_limits


def _process_clip_limits(policy: dict[str, Any]) -> dict[str, np.ndarray]:
    '''Process clip policy from config and derive per-joint limits'''
    position_offset = _load_joint_config(
        policy.get('position_offset', 0.01), 'position_offset'
    )
    velocity_ratio = _load_joint_config(
        policy.get('velocity_ratio', 0.5), 'velocity_ratio'
    )
    torque_ratio = _load_joint_config(policy.get('torque_ratio', 0.5), 'torque_ratio')
    kp_max = _load_joint_config(
        policy.get('kp_max', policy.get('kp_abs', 300.0)), 'kp_max'
    )
    kd_max = _load_joint_config(
        policy.get('kd_max', policy.get('kd_abs', 30.0)), 'kd_max'
    )
    if np.any(kp_max < 0.0):
        raise ValueError('kp_max must be nonnegative')
    if np.any(kd_max < 0.0):
        raise ValueError('kd_max must be nonnegative')

    return {
        'q_limits': _derive_q_limits(position_offset),
        'dq_limits': np.asarray(URDF_VELOCITY_LIMITS, dtype=np.float64)
        * velocity_ratio,
        'tau_limits': np.asarray(URDF_TORQUE_LIMITS, dtype=np.float64) * torque_ratio,
        'kp_max': kp_max,
        'kd_max': kd_max,
    }


def _process_estop_limits(policy: dict[str, Any]) -> dict[str, np.ndarray]:
    '''Process estop config and derive per-joint limits'''
    position_offset = _load_joint_config(
        policy.get('position_offset', 0.01), 'position_offset'
    )
    velocity_ratio = _load_joint_config(
        policy.get('velocity_ratio', 0.5), 'velocity_ratio'
    )
    torque_ratio = _load_joint_config(policy.get('torque_ratio', 0.5), 'torque_ratio')

    return {
        'q_limits': _derive_q_limits(position_offset),
        'dq_limits': np.asarray(URDF_VELOCITY_LIMITS, dtype=np.float64)
        * velocity_ratio,
        'tau_limits': np.asarray(URDF_TORQUE_LIMITS, dtype=np.float64) * torque_ratio,
    }


def _process_estop_response(
    response: dict[str, Any], kd_max: np.ndarray
) -> dict[str, Any]:
    '''
    Process the rt/lowcmd estop command: zero turns every motor off, damp keeps
    every motor on with kp = 0 and the configured kd
    '''
    mode = str(response.get('mode', 'zero')).strip().lower()
    if mode not in ESTOP_RESPONSES:
        raise ValueError(f'estop_response.mode must be zero or damp, not {mode!r}')
    kd = None
    if response.get('kd') is not None:
        kd = _load_joint_config(response['kd'], 'estop_response.kd')
        if not np.all(np.isfinite(kd)) or np.any(kd < 0.0):
            raise ValueError('estop_response.kd must be nonnegative and finite')
        if np.any(kd > kd_max):
            raise ValueError('estop_response.kd must not exceed limits.clip.kd_max')
    if mode == 'damp' and kd is None:
        raise ValueError('estop_response.kd is required when estop_response.mode is damp')
    return {'mode': mode, 'kd': kd}


def _channel_factory_settings(
    config: dict[str, Any]
) -> tuple[int, str | None]:
    '''Get DDS channel factory settings from environment and config'''
    env_domain = os.environ.get('ROS_DOMAIN_ID')
    network = config['network']
    domain_id = (
        int(env_domain)
        if env_domain is not None
        else int(network['domain_id'])
    )
    if domain_id == 0 and config['robot'] == 'sim':
        # its estop limits are relaxed for sim contact spikes
        raise ValueError(
            'this is a sim config (robot: sim) and refuses DDS domain 0, the real '
            'robot; unset ROS_DOMAIN_ID=0 or use a real config'
        )
    # G1_DDS_INTERFACE (e.g. enp130s0, the laptop's wired port to the robot)
    # overrides network.interface. The Unitree SDK ignores CYCLONEDDS_URI and
    # otherwise auto-picks an interface, which may be WiFi
    interface = os.environ.get('G1_DDS_INTERFACE') or network['interface']
    return domain_id, interface


def dds_domain_id(config: dict[str, Any]) -> int:
    '''The DDS domain config runs on; raises for a sim config on domain 0'''
    domain_id, _ = _channel_factory_settings(config)
    return domain_id


def init_channel_factory(config: dict[str, Any]) -> None:
    '''Initialize the DDS channel factory from environment and config'''
    domain_id, interface = _channel_factory_settings(config)
    if interface:
        ChannelFactoryInitialize(domain_id, interface)
    else:
        ChannelFactoryInitialize(domain_id)


def init_channel_factory_guard(config: dict[str, Any]) -> None:
    '''Confirm before initializing the real robot DDS domain'''
    domain_id, _ = _channel_factory_settings(config)
    if domain_id == 0:
        try:
            confirmation = input(
                'WARNING: ROS_DOMAIN_ID=0 -> DDS domain 0 is the REAL ROBOT '
                'command bus.\n'
                '         Nodes will publish/subscribe on the live robot.\n'
                'Proceed on DDS domain 0 (real robot)? [y/N] '
            )
        except EOFError:
            confirmation = ''
        if confirmation.strip().lower() != 'y':
            raise SystemExit('DDS channel factory initialization cancelled')
    init_channel_factory(config)


def load_config(path: str | Path) -> dict[str, Any]:
    '''Load YAML config as plain dict'''
    raw = _load_yaml(Path(path))

    topics = raw.get('topics', {})
    network = raw.get('network', {})
    logging = raw.get('logging', {})
    control = raw.get('control', {})
    engage = raw.get('engage', {})
    arm_sdk = raw.get('arm_sdk', {})
    estop_response = raw.get('estop_response', {})
    limits = raw.get('limits', {})

    if 'estop' in raw:
        # the H1's serial estop box and its DDS bridge are gone; fail loudly
        # rather than let an old config believe an external estop is wired
        raise ValueError('the external estop section is no longer supported; remove it')

    mode = str(raw.get('mode', 'arm_sdk_mode'))
    publish_hz = float(control.get('publish_hz', 500.0))
    if publish_hz <= 0.0:
        raise ValueError('control.publish_hz must be positive')
    # split_mode and full_body_mode: longest gap allowed on the leg input
    lower_stale_seconds = float(control.get('lower_stale_estop_seconds', 0.2))
    if not (np.isfinite(lower_stale_seconds) and lower_stale_seconds > 0.0):
        raise ValueError('control.lower_stale_estop_seconds must be positive and finite')
    # once engaged: longest gap allowed on rt/lowstate
    state_stale_seconds = _positive_finite(
        control.get('state_stale_estop_seconds', 0.1), 'control.state_stale_estop_seconds'
    )
    # take control only with a fresh rt/lowstate, the expected mode_machine and a
    # first command close to the measured q
    engage_gate = {
        'max_state_age_seconds': _positive_finite(
            engage.get('max_state_age_seconds', 0.1), 'engage.max_state_age_seconds'
        ),
        'mode_machine': engage.get('mode_machine', 5),
        'max_position_error': _positive_finite(
            engage.get('max_position_error', 0.15), 'engage.max_position_error'
        ),
    }
    if isinstance(engage_gate['mode_machine'], bool) or not isinstance(
        engage_gate['mode_machine'], int
    ):
        raise ValueError('engage.mode_machine must be an integer')

    engage_seconds = float(arm_sdk.get('engage_seconds', 1.0))
    release_seconds = float(arm_sdk.get('release_seconds', 1.0))
    if engage_seconds < 0.0 or release_seconds < 0.0:
        raise ValueError('arm_sdk ramp times must be nonnegative')
    default_cmd_out = 'rt/arm_sdk' if mode.strip().lower() == 'arm_sdk_mode' else 'rt/lowcmd'

    clip_policy = limits.get('clip', limits.get('enforce', {}))
    estop_policy = limits.get('estop', {})

    clip_limits = _process_clip_limits(clip_policy)
    estop_limits = _process_estop_limits(estop_policy)
    # damp's kd may not exceed what the clip allows a controller: kd_max
    response = _process_estop_response(estop_response, clip_limits['kd_max'])
    # arm_sdk_mode rejects a command with a waist kp below this
    waist_kp_min = float(arm_sdk.get('waist_kp_min', 40.0))
    if not (np.isfinite(waist_kp_min) and waist_kp_min >= 0.0):
        raise ValueError('arm_sdk.waist_kp_min must be nonnegative and finite')
    if waist_kp_min > float(np.min(clip_limits['kp_max'][list(WAIST_SLOTS)])):
        raise ValueError('arm_sdk.waist_kp_min must not exceed the waist limits.clip.kp_max')

    if 'robot' not in raw:
        raise ValueError('the config must declare robot: real or robot: sim')
    robot = str(raw['robot']).strip().lower()
    if robot not in ROBOTS:
        raise ValueError(f'robot must be real or sim, not {robot!r}')
    if robot == 'sim' and int(network.get('domain_id', 0)) == 0:
        raise ValueError('a sim config (robot: sim) must not set network.domain_id 0')
    log_root = os.environ.get(LOG_DIR_ENV)
    if log_root:
        # ChunkLogger clears its directory's chunks at start: a fresh one per
        # run keeps every earlier run's logs
        base_dir = os.path.join(log_root, f'{time.strftime("%Y%m%d-%H%M%S")}-{os.getpid()}')
    else:
        base_dir = str(logging.get('base_dir', '/tmp/g1_safety_layer'))

    return {
        'mode': mode,
        'robot': robot,
        'topics': {
            'low_cmd_in': str(topics.get('low_cmd_in', 'rt/safety/lowcmd_in')),
            'low_cmd_lower_in': str(
                topics.get('low_cmd_lower_in', 'rt/safety/lowcmd_lower_in')
            ),
            'low_cmd_upper_in': str(
                topics.get('low_cmd_upper_in', 'rt/safety/lowcmd_upper_in')
            ),
            'low_cmd_out': str(topics.get('low_cmd_out', default_cmd_out)),
            'low_state': str(topics.get('low_state', 'rt/lowstate')),
        },
        'network': {
            'domain_id': int(network.get('domain_id', 0)),
            'interface': None
            if network.get('interface') in (None, '')
            else str(network.get('interface')),
        },
        'control': {
            'publish_hz': publish_hz,
            'lower_stale_estop_seconds': lower_stale_seconds,
            'state_stale_estop_seconds': state_stale_seconds,
        },
        'engage': engage_gate,
        'arm_sdk': {
            'engage_seconds': engage_seconds,
            'release_seconds': release_seconds,
            'waist_kp_min': waist_kp_min,
        },
        'estop_response': response,
        'limits': {
            'q_clip_limits': clip_limits['q_limits'],
            'dq_clip_limits': clip_limits['dq_limits'],
            'tau_clip_limits': clip_limits['tau_limits'],
            'kp_clip_max': clip_limits['kp_max'],
            'kd_clip_max': clip_limits['kd_max'],
            'q_estop_limits': estop_limits['q_limits'],
            'dq_estop_limits': estop_limits['dq_limits'],
            'tau_estop_limits': estop_limits['tau_limits'],
        },
        'logging': {
            'enabled': bool(logging.get('enabled', True)),
            'base_dir': base_dir,
            'chunk_prefix': str(logging.get('chunk_prefix', 'chunk_')),
        },
    }
