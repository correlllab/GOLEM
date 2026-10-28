'''
Controller configs: load_controller_config(robot, name_or_path) reads one YAML file and
returns the values the controller runs on, every default filled in and the joint arrays
derived for the robot's body slots.

A bare name ('safety_split' or 'safety_split.yaml') is a file of
share/golem_upper_ik/config/<robot>/. Anything with a '/' is a path. Every key is checked:
an unknown key, a key from an older config, a non-finite number or a value of the wrong type
raises ValueError naming the robot, the file and the key.
'''
import math
import os

from ament_index_python.packages import get_package_share_directory
import numpy as np
import yaml

from golem_upper_ik.profile import ROBOTS, RobotProfile, load_profile
from golem_upper_ik.robots import robot_module

__all__ = ['load_controller_config', 'SAFETY_INPUT_PREFIX', 'SAFETY_OUTPUT_TOPICS']

# The controller writes only golem_safety's inputs: golem_safety is the only writer of
# /lowcmd and /arm_sdk
SAFETY_INPUT_PREFIX = '/safety/'
# golem_safety's output topic in each of its modes; the follow gate reads one of them
SAFETY_OUTPUT_TOPICS = {
    'arm_sdk_mode': '/arm_sdk',
    'split_mode': '/lowcmd',
    'full_body_mode': '/lowcmd',
}

TOPIC_DEFAULTS = {
    'low_cmd': '/safety/lowcmd_upper_in',
    'low_state': '/lowstate',
}
CONTROLLER_DEFAULTS = {
    'v_lim': 1.0,  # end-effector linear speed limit, m/s
    'w_lim': 2.0,  # end-effector angular speed limit, rad/s
    'dq_lim': 1.0,  # IK joint speed limit, rad/s
    'd_min': 0.02,  # self-collision distance the IK keeps, m
    'timeout': 10.0,  # s per goal without a duration
    'threshold_ik': 0.001,
    'threshold_joint': 0.01,  # rad
    'threshold_linear': 0.002,  # m
    'threshold_angular': 0.01,  # rad
    'state_timeout': 5.0,  # s to wait for a fresh /lowstate before enabling anything
    'startup_timeout': 30.0,  # s for golem_safety to be seen relaying the commands
    'startup_move_timeout': 30.0,  # s for the start-up moves, from the follow gate opening
    'follow_tolerance': 0.05,  # rad; 0 turns the follow gate off
}
# must be > 0
POSITIVE_CONTROLLER_KEYS = ('state_timeout', 'startup_timeout', 'startup_move_timeout')
FREQUENCY_DEFAULTS = {
    'ctrl_hz': 50.0,  # IK
    'pub_hz': 500.0,  # command publishing
    'check_hz': 1000.0,  # the follow gate's check between control steps
}
# limits.clip: q within [low + position_offset, high - position_offset], |dq| at most
# velocity_ratio * the velocity limit, |tau| at most torque_ratio * the torque limit
CLIP_DEFAULTS = {
    'position_offset': 0.001,  # rad
    'velocity_ratio': 0.10,
    'torque_ratio': 0.25,
}
LOGGING_DEFAULTS = {
    'enabled': False,
    'base_dir': 'data/control_record',
    'filename': 'record',
    'record_interval': 1.0,  # s
}
# logging keys and the older spelling each also takes; giving both is an error
LOGGING_ALIASES = {'base_dir': 'save_path', 'filename': 'save_filename'}
# The planner section's keys. Its values are passed through as written; the planner's own
# config supplies the defaults, except enabled (true) and frame_names (the profile's planner
# frames), which the loader fills in. The planner runs only when the profile has a
# self-collision model and enabled is true.
PLANNER_KEYS = frozenset((
    'enabled', 'planner', 'timeout', 'range', 'try_direct', 'simplify', 'simplify_time',
    'shortcut', 'moving_speed', 'dt', 'min_interpolation_steps', 'max_interpolation_steps',
    'validity_resolution', 'constraint_check_steps', 'frame_names', 'frame_x_max',
    'frame_y_min', 'frame_y_max', 'frame_z_min', 'frame_z_max', 'frame_limit_margin',
    'frame_z_corridor_margin', 'start_limit_tolerance',
))
SECTION_KEYS = {
    'topics': frozenset(TOPIC_DEFAULTS) | {'safety_out'},
    'controller': frozenset(CONTROLLER_DEFAULTS) | {'hold_targets', 'imu_frame',
                                                    'max_state_lag_ms'},
    'planner': PLANNER_KEYS,
    'frequency': frozenset(FREQUENCY_DEFAULTS),
    'gains': frozenset(('kp', 'kd', 'ki')),
    'limits': frozenset(('clip', 'enforce')),
    'logging': frozenset(LOGGING_DEFAULTS) | frozenset(LOGGING_ALIASES.values()),
}
# What the error says about a key an older config may still carry
REPLACED_KEYS = {
    'mode': 'removed: the controller always writes a golem_safety input (topics.low_cmd)',
    'network': "removed: ROS supplies the domain, and the G1's interface is pinned in "
               'cyclonedds_local.xml',
    'controller.torso_target': 'renamed controller.hold_targets: {torso_joint: rad}',
    'controller.waist_target': 'renamed controller.hold_targets: {waist_yaw_joint: rad, '
                               'waist_roll_joint: rad, waist_pitch_joint: rad}',
    'limits.estop': "removed: the controller has no estop of its own; golem_safety's estop "
                    'limits are the check',
    'planner.frame_z_min_margin': 'renamed planner.frame_limit_margin (x, y and z)',
}


class _Where:
    '''Builds the errors of one file: the robot, the file and the key'''

    def __init__(self, robot: str, source: str):
        self.robot = robot
        self.source = source

    def error(self, key: str, what: str) -> ValueError:
        where = f"key '{key}': " if key else ''
        return ValueError(f'controller config {self.source} (robot {self.robot}): {where}{what}')


def _section(where: _Where, raw: dict, name: str) -> dict:
    '''The mapping under name, its keys checked; {} when absent'''
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise where.error(name, 'must be a mapping')
    for key in value:
        dotted = f'{name}.{key}'
        if dotted in REPLACED_KEYS:
            raise where.error(dotted, REPLACED_KEYS[dotted])
        if key not in SECTION_KEYS[name]:
            raise where.error(dotted, 'unknown key')
    return value


def _number(where: _Where, key: str, value) -> float:
    '''A finite number. A string is parsed: PyYAML reads 1e-3 (no dot) as a string'''
    if isinstance(value, bool):
        raise where.error(key, f'must be a number, not {value!r}')
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise where.error(key, f'must be a number, not {value!r}') from None
    if not isinstance(value, (int, float)):
        raise where.error(key, f'must be a number, not {value!r}')
    value = float(value)
    if not math.isfinite(value):
        raise where.error(key, f'must be finite, not {value}')
    return value


def _boolean(where: _Where, key: str, value) -> bool:
    if not isinstance(value, bool):
        raise where.error(key, f'must be true or false, not {value!r}')
    return value


def _text(where: _Where, key: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise where.error(key, f'must be a non-empty string, not {value!r}')
    return value


def _topic(where: _Where, key: str, value) -> str:
    '''A ROS topic name; the DDS spelling rt/x is refused with its ROS name /x'''
    topic = _text(where, key, value)
    if topic.startswith('rt/'):
        raise where.error(key, f"'{topic}' is a DDS name: give the ROS topic "
                               f"'/{topic[len('rt/'):]}'")
    if not topic.startswith('/'):
        raise where.error(key, f"'{topic}' must be a ROS topic name starting with '/'")
    return topic


def _joint_values(where: _Where, key: str, value, profile: RobotProfile) -> np.ndarray:
    '''
    One value per body slot from a number (every slot), a list over the body slots, or a
    list over the upper slots (the lower slots 0)
    '''
    count = len(profile.body_joints)
    upper = list(profile.upper_slots)
    if isinstance(value, list):
        values = np.array([_number(where, key, v) for v in value], dtype=np.float64)
        if len(values) == count:
            return values
        if len(values) == len(upper):
            full = np.zeros(count, dtype=np.float64)
            full[upper] = values
            return full
    elif not isinstance(value, dict):
        return np.full(count, _number(where, key, value), dtype=np.float64)
    raise where.error(key, f'must be a number, a list of {count} (every body joint) or a '
                           f'list of {len(upper)} (the upper body), not {value!r}')


def _gains(where: _Where, gains: dict, profile: RobotProfile) -> dict:
    '''kp, kd and ki on the upper slots (a number, or a list over them); the legs get 0'''
    missing = [name for name in ('kp', 'kd', 'ki') if name not in gains]
    if missing:
        raise where.error('gains', f'kp, kd and ki must be given; missing {", ".join(missing)}')
    upper = list(profile.upper_slots)
    out = {}
    for name in ('kp', 'kd', 'ki'):
        key = f'gains.{name}'
        value = gains[name]
        full = np.zeros(len(profile.body_joints), dtype=np.float64)
        if isinstance(value, list) and len(value) == len(upper):
            full[upper] = [_number(where, key, v) for v in value]
        elif isinstance(value, (list, dict)):
            raise where.error(key, f'must be a number or a list of {len(upper)} '
                                   f'(the upper body), not {value!r}')
        else:
            full[upper] = _number(where, key, value)
        out[name] = full.astype(np.float32)
    return out


def _clip_limits(where: _Where, limits: dict, profile: RobotProfile) -> dict:
    if 'clip' in limits and 'enforce' in limits:
        raise where.error('limits.enforce', 'give limits.clip or limits.enforce, not both')
    name = 'enforce' if 'enforce' in limits else 'clip'
    policy = limits.get(name, {})
    if not isinstance(policy, dict):
        raise where.error(f'limits.{name}', 'must be a mapping')
    for key in policy:
        if key not in CLIP_DEFAULTS:
            raise where.error(f'limits.{name}.{key}', 'unknown key')
    values = {key: _joint_values(where, f'limits.{name}.{key}', policy.get(key, default),
                                 profile)
              for key, default in CLIP_DEFAULTS.items()}

    offset = values['position_offset']
    q_limits = np.stack([profile.q_low + offset, profile.q_high - offset], axis=1)
    for slot, (low, high) in enumerate(q_limits):
        if low >= high:
            raise where.error(
                f'limits.{name}.position_offset',
                f'too large for joint {slot} ({profile.body_joints[slot]}): it leaves '
                f'[{low}, {high}]')
    return {
        'q_clip_limits': q_limits,
        'dq_clip_limits': profile.dq_max * values['velocity_ratio'],
        'tau_clip_limits': profile.tau_max * values['torque_ratio'],
    }


def _hold_targets(where: _Where, value, profile: RobotProfile) -> dict:
    '''{held joint: rad} over every held joint, in the profile's order; a joint left out is 0'''
    key = 'controller.hold_targets'
    if not isinstance(value, dict):
        raise where.error(key, f'must be a mapping {{joint: rad}} over the held joints '
                               f'({", ".join(profile.held_joints)}), not {value!r}')
    for joint in value:
        if joint not in profile.held_joints:
            raise where.error(key, f"'{joint}' is not a held joint of robot {profile.robot} "
                                   f'({", ".join(profile.held_joints)})')
    return {joint: _number(where, f'{key}.{joint}', value.get(joint, 0.0))
            for joint in profile.held_joints}


def _resolve(robot: str, name_or_path: str) -> str:
    path = str(name_or_path)
    if '/' not in path:
        name = path if path.endswith('.yaml') else f'{path}.yaml'
        path = os.path.join(get_package_share_directory('golem_upper_ik'), 'config', robot, name)
    # A config of one robot can parse for the other (a scalar gain fits either), so a file
    # in another robot's config directory (config/g1/ for the h12) is refused by that name
    directory = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if directory != robot and directory in ROBOTS:
        raise ValueError(f'the controller config {path} is in the config directory of robot '
                         f'{directory}, not of robot {robot}')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no controller config '{name_or_path}' for robot {robot}: "
                                f'{path} does not exist')
    return path


def load_controller_config(robot: str, name_or_path: str) -> dict:
    '''The controller config name_or_path of robot, checked and with its defaults filled in'''
    robot_module(robot)  # ValueError for an unknown robot, before any file is read
    path = _resolve(robot, name_or_path)
    profile = load_profile(robot)
    robot_defaults = robot_module(robot).CONFIG_DEFAULTS
    where = _Where(robot, path)

    with open(path, encoding='utf-8') as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as error:
            raise where.error('', f'not valid YAML: {error}') from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise where.error('', 'the file must hold a mapping')
    for key in raw:
        if key in REPLACED_KEYS:
            raise where.error(key, REPLACED_KEYS[key])
        if key not in SECTION_KEYS:
            raise where.error(key, 'unknown key')
    sections = {name: _section(where, raw, name) for name in SECTION_KEYS}

    topics_raw = sections['topics']
    topics = {key: _topic(where, f'topics.{key}', topics_raw.get(key, default))
              for key, default in TOPIC_DEFAULTS.items()}
    if not topics['low_cmd'].startswith(SAFETY_INPUT_PREFIX):
        raise where.error('topics.low_cmd', f"'{topics['low_cmd']}' is not a golem_safety "
                                            f'input ({SAFETY_INPUT_PREFIX}...)')
    outputs = sorted({SAFETY_OUTPUT_TOPICS[mode] for mode in profile.modes})
    safety_out = _topic(where, 'topics.safety_out',
                        topics_raw.get('safety_out', robot_defaults['topics.safety_out']))
    if safety_out not in outputs:
        raise where.error('topics.safety_out', f"'{safety_out}' is not a golem_safety output "
                                               f'of robot {robot} ({", ".join(outputs)})')
    topics['safety_out'] = safety_out

    controller_raw = sections['controller']
    controller = {key: _number(where, f'controller.{key}', controller_raw.get(key, default))
                  for key, default in CONTROLLER_DEFAULTS.items()}
    for key in POSITIVE_CONTROLLER_KEYS:
        if controller[key] <= 0.0:
            raise where.error(f'controller.{key}', f'must be positive, not {controller[key]}')
    if controller['follow_tolerance'] < 0.0:
        raise where.error('controller.follow_tolerance',
                          f"must be 0 (off) or positive, not {controller['follow_tolerance']}")
    # /lowstate tick lag at which goals stop, ms; <= 0 turns the check off
    controller['max_state_lag_ms'] = _number(
        where, 'controller.max_state_lag_ms',
        controller_raw.get('max_state_lag_ms', robot_defaults['controller.max_state_lag_ms']))
    controller['imu_frame'] = _text(where, 'controller.imu_frame',
                                    controller_raw.get('imu_frame', profile.imu_frame))
    controller['hold_targets'] = _hold_targets(
        where, controller_raw.get('hold_targets', {}), profile)

    frequency_raw = sections['frequency']
    frequency = {key: _number(where, f'frequency.{key}', frequency_raw.get(key, default))
                 for key, default in FREQUENCY_DEFAULTS.items()}
    for key, value in frequency.items():
        if value <= 0.0:
            raise where.error(f'frequency.{key}', f'must be positive, not {value}')

    planner = dict(sections['planner'])
    planner['enabled'] = _boolean(where, 'planner.enabled', planner.get('enabled', True))
    planner.setdefault('frame_names', list(profile.planner_frames))

    logging_raw = sections['logging']
    for key, alias in LOGGING_ALIASES.items():
        if key in logging_raw and alias in logging_raw:
            raise where.error(f'logging.{alias}', f'give logging.{key} or logging.{alias}, '
                                                  'not both')
    logging = {
        'enabled': _boolean(where, 'logging.enabled',
                            logging_raw.get('enabled', LOGGING_DEFAULTS['enabled'])),
        'record_interval': _number(
            where, 'logging.record_interval',
            logging_raw.get('record_interval', LOGGING_DEFAULTS['record_interval'])),
    }
    for key, alias in LOGGING_ALIASES.items():
        given = key if key in logging_raw else alias
        logging[key] = _text(where, f'logging.{given}',
                             logging_raw.get(given, LOGGING_DEFAULTS[key]))

    return {
        'robot': robot,
        'source': path,
        'topics': topics,
        'controller': controller,
        'planner': planner,
        'frequency': frequency,
        'gains': _gains(where, sections['gains'], profile),
        'limits': _clip_limits(where, sections['limits'], profile),
        'logging': {key: logging[key] for key in LOGGING_DEFAULTS},
    }
