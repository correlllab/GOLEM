import os
from pathlib import Path
from typing import Any

import yaml
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_control.utility.joint_definition import NUM_MOTOR, UPPER_BODY_INDEX, WAIST_INDEX
from g1_control.utility.joint_limits import (
    JOINT_POSITION_LIMITS,
    JOINT_VELOCITY_LIMITS,
    JOINT_TORQUE_LIMITS,
)

DEFAULT_CONFIG_NAME = 'g1.yaml'
# g1_safety_layer's outputs, which only it writes: the controller reads one
# (topics.safety_out) to see the safety layer relay its commands
ARM_SDK_TOPIC = 'rt/arm_sdk'
SAFETY_OUTPUT_TOPICS = (ARM_SDK_TOPIC, 'rt/lowcmd')
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / 'config'

def _ensure_yaml_extension(config_path: Path) -> Path:
    '''Append .yaml when config path has no suffix'''
    if config_path.suffix:
        return config_path
    return config_path.with_suffix('.yaml')

def _resolve_config_path(config_name: str,
                         config_dir: Path = DEFAULT_CONFIG_DIR) -> Path:
    '''Resolve config from either bare name or config/<name> path'''
    config_path = _ensure_yaml_extension(Path(config_name))
    if config_path.is_absolute():
        return config_path
    if config_path.parts and config_path.parts[0] == 'config':
        return REPO_ROOT / config_path
    return config_dir / config_path

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError('config root must be a mapping')
    return loaded

def _load_waist_target(value: Any) -> list[float]:
    '''Waist hold angles (yaw, roll, pitch) from a scalar or a 3-entry list'''
    if isinstance(value, (int, float)):
        return [float(value)] * len(WAIST_INDEX)
    if isinstance(value, list) and len(value) == len(WAIST_INDEX):
        return [float(v) for v in value]
    raise ValueError(f'controller.waist_target must be a number or {len(WAIST_INDEX)} values')

def _load_joint_config(value: Any, field_name: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full((NUM_MOTOR,), float(value), dtype=np.float64)
    if isinstance(value, list) and len(value) == NUM_MOTOR:
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, list) and len(value) == len(UPPER_BODY_INDEX):
        values = np.zeros((NUM_MOTOR,), dtype=np.float64)
        values[UPPER_BODY_INDEX] = np.asarray(value, dtype=np.float64)
        return values
    raise ValueError(
        f'{field_name} must be a number, a list with {NUM_MOTOR} entries, '
        f'or a list with {len(UPPER_BODY_INDEX)} upper-body entries'
    )

def _load_upper_body_gain_config(value: Any, field_name: str) -> np.ndarray:
    gains = np.zeros((NUM_MOTOR,), dtype=np.float64)
    if isinstance(value, (int, float)):
        gains[UPPER_BODY_INDEX] = float(value)
        return gains
    if isinstance(value, list) and len(value) == len(UPPER_BODY_INDEX):
        gains[UPPER_BODY_INDEX] = np.asarray(value, dtype=np.float64)
        return gains
    raise ValueError(
        f'{field_name} must be a number or a list with '
        f'{len(UPPER_BODY_INDEX)} upper-body entries'
    )

def _derive_q_limits(position_offset: np.ndarray) -> np.ndarray:
    q_limits = np.zeros((NUM_MOTOR, 2), dtype=np.float64)
    for i in range(NUM_MOTOR):
        low = float(JOINT_POSITION_LIMITS[i]['low']) + position_offset[i]
        high = float(JOINT_POSITION_LIMITS[i]['high']) - position_offset[i]
        if low >= high:
            raise ValueError(f'position_offset too large for joint {i}')
        q_limits[i, 0] = low
        q_limits[i, 1] = high
    return q_limits

def _process_clip_limits(policy: dict[str, Any]) -> dict[str, np.ndarray]:
    position_offset = _load_joint_config(policy.get('position_offset', 0.001), 'limits.clip.position_offset')
    velocity_ratio = _load_joint_config(policy.get('velocity_ratio', 0.10), 'limits.clip.velocity_ratio')
    torque_ratio = _load_joint_config(policy.get('torque_ratio', 0.25), 'limits.clip.torque_ratio')

    return {
        'q_limits': _derive_q_limits(position_offset),
        'dq_limits': np.asarray(JOINT_VELOCITY_LIMITS, dtype=np.float64) * velocity_ratio,
        'tau_limits': np.asarray(JOINT_TORQUE_LIMITS, dtype=np.float64) * torque_ratio,
    }

def _process_estop_limits(policy: dict[str, Any]) -> dict[str, np.ndarray]:
    position_offset = _load_joint_config(policy.get('position_offset', 0.01), 'limits.estop.position_offset')
    velocity_ratio = _load_joint_config(policy.get('velocity_ratio', 0.5), 'limits.estop.velocity_ratio')
    torque_ratio = _load_joint_config(policy.get('torque_ratio', 0.5), 'limits.estop.torque_ratio')

    return {
        'q_limits': _derive_q_limits(position_offset),
        'dq_limits': np.asarray(JOINT_VELOCITY_LIMITS, dtype=np.float64) * velocity_ratio,
        'tau_limits': np.asarray(JOINT_TORQUE_LIMITS, dtype=np.float64) * torque_ratio,
    }

def build_processed_limits(limits_cfg: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    limits_cfg = {} if limits_cfg is None else limits_cfg
    clip_policy = limits_cfg.get('clip', limits_cfg.get('enforce', {}))
    estop_policy = limits_cfg.get('estop', {})

    clip_limits = _process_clip_limits(clip_policy)
    estop_limits = _process_estop_limits(estop_policy)

    return {
        'q_clip_limits': clip_limits['q_limits'],
        'dq_clip_limits': clip_limits['dq_limits'],
        'tau_clip_limits': clip_limits['tau_limits'],
        'q_estop_limits': estop_limits['q_limits'],
        'dq_estop_limits': estop_limits['dq_limits'],
        'tau_estop_limits': estop_limits['tau_limits'],
    }

def get_publisher_clip_limits(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = {} if config is None else config
    limits_cfg = config.get('limits', {})

    if all(key in limits_cfg for key in ('q_clip_limits', 'dq_clip_limits', 'tau_clip_limits')):
        q_clip_limits = limits_cfg['q_clip_limits']
        dq_clip_limits = limits_cfg['dq_clip_limits']
        tau_clip_limits = limits_cfg['tau_clip_limits']
    else:
        processed = build_processed_limits(limits_cfg)
        q_clip_limits = processed['q_clip_limits']
        dq_clip_limits = processed['dq_clip_limits']
        tau_clip_limits = processed['tau_clip_limits']

    return {
        'position_clip': [
            {'low': float(limit[0]), 'high': float(limit[1])}
            for limit in q_clip_limits
        ],
        'velocity_clip': dq_clip_limits,
        'torque_clip': tau_clip_limits,
    }

def _process_gains(raw_gains: dict[str, Any]) -> dict[str, Any]:
    kp = raw_gains.get('kp')
    kd = raw_gains.get('kd')
    ki = raw_gains.get('ki')
    if kp is None or kd is None or ki is None:
        raise ValueError('gains.kp, gains.kd, and gains.ki must be provided')

    return {
        'kp': _load_upper_body_gain_config(kp, 'gains.kp'),
        'kd': _load_upper_body_gain_config(kd, 'gains.kd'),
        'ki': _load_upper_body_gain_config(ki, 'gains.ki'),
    }

def load_controller_config(config_name=DEFAULT_CONFIG_NAME,
                           config_dir=DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    config_path = _resolve_config_path(config_name, config_dir=Path(config_dir))
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    raw = _load_yaml(config_path)

    if 'mode' in raw:
        # the H1-2 debug/sport modes published straight to rt/lowcmd or
        # rt/arm_sdk; on the G1 every command goes through g1_safety_layer
        raise ValueError(f'{config_path}: `mode` is no longer supported; remove it')

    topics = raw.get('topics', {})
    network = raw.get('network', {})
    controller = raw.get('controller', {})
    planner = raw.get('planner', {})
    frequency = raw.get('frequency', {})
    limits = raw.get('limits', {})
    gains = raw.get('gains', {})
    logging = raw.get('logging', {})
    if not isinstance(planner, dict):
        raise ValueError('planner must be a mapping')

    ctrl_hz = float(frequency.get('ctrl_hz', 50.0))
    pub_hz = float(frequency.get('pub_hz', 500.0))
    check_hz = float(frequency.get('check_hz', 1000.0))
    if ctrl_hz <= 0.0:
        raise ValueError('frequency.ctrl_hz must be positive')
    if pub_hz <= 0.0:
        raise ValueError('frequency.pub_hz must be positive')
    if check_hz <= 0.0:
        raise ValueError('frequency.check_hz must be positive')

    processed_limits = build_processed_limits(limits)
    processed_gains = _process_gains(gains)

    v_lim = float(controller.get('v_lim', 1.0))
    w_lim = float(controller.get('w_lim', 2.0))
    dq_lim = float(controller.get('dq_lim', 1.0))
    d_min = float(controller.get('d_min', 0.02))
    timeout = float(controller.get('timeout', 10.0))
    low_cmd_topic = str(topics.get('low_cmd', 'rt/safety/lowcmd_upper_in'))
    if not low_cmd_topic.startswith('rt/safety/'):
        # rt/lowcmd and rt/arm_sdk are only ever written by g1_safety_layer
        raise ValueError(
            f'topics.low_cmd must be a g1_safety_layer input (rt/safety/...), got {low_cmd_topic}'
        )
    safety_out_topic = str(topics.get('safety_out', ARM_SDK_TOPIC))
    if safety_out_topic not in SAFETY_OUTPUT_TOPICS:
        raise ValueError(
            'topics.safety_out must be the g1_safety_layer output the controller reads, '
            f'one of {", ".join(SAFETY_OUTPUT_TOPICS)}; got {safety_out_topic}'
        )
    waist_target = _load_waist_target(controller.get('waist_target', 0.0))
    state_timeout = float(controller.get('state_timeout', 5.0))
    startup_timeout = float(controller.get('startup_timeout', 30.0))
    startup_move_timeout = float(controller.get('startup_move_timeout', 30.0))
    follow_tolerance = float(controller.get('follow_tolerance', 0.05))
    for name, value in (('state_timeout', state_timeout), ('startup_timeout', startup_timeout),
                        ('startup_move_timeout', startup_move_timeout)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f'controller.{name} must be positive and finite')
    if not (np.isfinite(follow_tolerance) and follow_tolerance >= 0.0):
        raise ValueError('controller.follow_tolerance must be 0 (off) or positive, and finite')
    imu_frame = str(controller.get('imu_frame', 'imu_in_pelvis'))
    threshold_ik = float(controller.get('threshold_ik', 0.001))
    threshold_joint = float(controller.get('threshold_joint', 0.01))
    threshold_linear = float(controller.get('threshold_linear', 0.002))
    threshold_angular = float(controller.get('threshold_angular', 0.01))

    return {
        'topics': {
            'low_cmd': low_cmd_topic,
            'low_state': str(topics.get('low_state', 'rt/lowstate')),
            'safety_out': safety_out_topic,
        },
        'network': {
            'domain_id': int(network.get('domain_id', 0)),
            'interface': None if network.get('interface') in (None, '') else str(network.get('interface')),
        },
        'controller': {
            'v_lim': v_lim,
            'w_lim': w_lim,
            'dq_lim': dq_lim,
            'd_min': d_min,
            'timeout': timeout,
            'waist_target': waist_target,
            'state_timeout': state_timeout,
            'startup_timeout': startup_timeout,
            'startup_move_timeout': startup_move_timeout,
            'follow_tolerance': follow_tolerance,
            'imu_frame': imu_frame,
            'threshold_ik': threshold_ik,
            'threshold_joint': threshold_joint,
            'threshold_linear': threshold_linear,
            'threshold_angular': threshold_angular,
        },
        'planner': dict(planner),
        'frequency': {
            'ctrl_hz': ctrl_hz,
            'pub_hz': pub_hz,
            'check_hz': check_hz,
        },
        'gains': {
            'kp': None if processed_gains['kp'] is None else processed_gains['kp'].astype(np.float32),
            'kd': None if processed_gains['kd'] is None else processed_gains['kd'].astype(np.float32),
            'ki': None if processed_gains['ki'] is None else processed_gains['ki'].astype(np.float32),
        },
        'limits': {
            'q_clip_limits': processed_limits['q_clip_limits'],
            'dq_clip_limits': processed_limits['dq_clip_limits'],
            'tau_clip_limits': processed_limits['tau_clip_limits'],
            'q_estop_limits': processed_limits['q_estop_limits'],
            'dq_estop_limits': processed_limits['dq_estop_limits'],
            'tau_estop_limits': processed_limits['tau_estop_limits'],
        },
        'logging': {
            'enabled': bool(logging.get('enabled', False)),
            'base_dir': str(logging.get('base_dir', logging.get('save_path', 'data/control_record'))),
            'filename': str(logging.get('filename', logging.get('save_filename', 'record'))),
            'record_interval': float(logging.get('record_interval', 1.0)),
        },
    }

def _channel_factory_settings(config: dict[str, Any]) -> tuple[int, str | None]:
    '''Get DDS channel factory settings from environment and config'''
    env_domain = os.environ.get('ROS_DOMAIN_ID')
    network = config['network']
    domain_id = (
        int(env_domain)
        if env_domain is not None
        else int(network['domain_id'])
    )
    # G1_DDS_INTERFACE (e.g. enp130s0, the laptop's wired port to the robot)
    # overrides network.interface. The Unitree SDK ignores CYCLONEDDS_URI and
    # otherwise auto-picks an interface, which may be WiFi
    interface = os.environ.get('G1_DDS_INTERFACE') or network['interface']
    return domain_id, interface


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


def init_channel_factory_for_commands(config: dict[str, Any], real_robot: bool,
                                      publishes: str) -> None:
    '''
    Initialize DDS for an example or debug tool that publishes commands:
    refuse DDS domain 0 (the real robot) unless real_robot (--real-robot) is
    set, and say why; then confirm as init_channel_factory_guard does
    '''
    domain_id, _ = _channel_factory_settings(config)
    if domain_id == 0 and not real_robot:
        raise SystemExit(
            'Refusing to start on DDS domain 0, the real robot\'s command bus: '
            f'this tool publishes {publishes}. Set ROS_DOMAIN_ID to the sim domain '
            '(1), or pass --real-robot to run it on the robot on purpose'
        )
    init_channel_factory_guard(config)


def maybe_start_controller_logging(controller):
    logging_cfg = controller.config.get('logging', {})
    if not logging_cfg.get('enabled', False):
        return

    filename = logging_cfg.get('filename', '').strip()
    if not filename:
        raise ValueError('logging.filename must be set when logging.enabled is true')

    controller.start_recording(
        save_path=logging_cfg.get('base_dir', 'data/control_record'),
        filename=filename,
        record_interval=float(logging_cfg.get('record_interval', 1.0)),
    )
