'''
Controller configs: load_controller_config(robot, name_or_path) reads one YAML file and
returns the values the controller runs on, every default filled in and the joint arrays
derived for the robot's body slots.

A bare name ('safety_split' or 'safety_split.yaml') is a file of
share/golem_upper_ik/config/<robot>/. Anything with a '/' is a path. Every key and every
value is checked: an unknown key, a key from an older config, a value of the wrong type, a
non-finite number or a number out of its range raises ValueError naming the robot, the file
and the key.
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
# /lowcmd
SAFETY_INPUT_PREFIX = '/safety/'
# golem_safety's output topic in each of its modes; the follow gate reads one of them
SAFETY_OUTPUT_TOPICS = {
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
    'threshold_ik': 0.001,  # IK velocity error at which a goal has converged
    'threshold_joint': 0.01,  # rad
    'threshold_linear': 0.002,  # m
    'threshold_angular': 0.01,  # rad
    'state_timeout': 5.0,  # s to wait for a fresh /lowstate before enabling anything
    'startup_timeout': 30.0,  # s for golem_safety to be seen relaying the commands
    'startup_move_timeout': 30.0,  # s for the start-up moves, from the follow gate opening
    'follow_tolerance': 0.05,  # rad; 0 turns the follow gate off
}
# The values above must be > 0 (a speed limit of 0 never moves, a timeout of 0 fails at
# once, a goal never converges to a threshold of 0), apart from these, which may be 0
NON_NEGATIVE_CONTROLLER_KEYS = ('d_min', 'follow_tolerance')
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
# The planner section's keys, by kind: the fields of the H1-2 planner's PlannerConfig, and
# enabled. The loader checks the keys a file gives and fills in only enabled (true) and
# frame_names (the profile's planner frames); PlannerConfig supplies every other default.
# The planner runs only when the profile has a self-collision model and enabled is true.
PLANNER_TEXT_KEYS = ('planner',)  # the OMPL planner's name
PLANNER_BOOLEAN_KEYS = ('enabled', 'try_direct', 'simplify', 'shortcut')
PLANNER_POSITIVE_KEYS = (
    'timeout',  # s to plan
    'moving_speed',  # planner-frame Cartesian speed, m/s
    'dt',  # s between path samples
    'validity_resolution',  # validity-check spacing in joint space, rad
)
PLANNER_NON_NEGATIVE_KEYS = (
    'range',  # extension length in joint space, rad; 0 lets OMPL choose
    'simplify_time',  # s
    'frame_limit_margin',  # m
    'frame_z_corridor_margin',  # m
    'start_limit_tolerance',  # rad
)
PLANNER_COUNT_KEYS = ('min_interpolation_steps', 'max_interpolation_steps',
                      'constraint_check_steps')
PLANNER_BOUND_KEYS = ('frame_x_max', 'frame_y_min', 'frame_y_max', 'frame_z_min',
                      'frame_z_max')  # m, or null for no bound
PLANNER_KEYS = frozenset(
    PLANNER_TEXT_KEYS + PLANNER_BOOLEAN_KEYS + PLANNER_POSITIVE_KEYS + PLANNER_NON_NEGATIVE_KEYS
    + PLANNER_COUNT_KEYS + PLANNER_BOUND_KEYS + ('frame_names',))
SECTION_KEYS = {
    'topics': frozenset(TOPIC_DEFAULTS) | {'safety_out'},
    'controller': frozenset(CONTROLLER_DEFAULTS) | {'hold_targets', 'imu_frame',
                                                    'max_state_lag_ms'},
    'planner': PLANNER_KEYS,
    'frequency': frozenset(FREQUENCY_DEFAULTS),
    'gains': frozenset(('kp', 'kd', 'ki')),
    'limits': frozenset(('clip', 'enforce')),
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


def _positive(where: _Where, key: str, value) -> float:
    number = _number(where, key, value)
    if number <= 0.0:
        raise where.error(key, f'must be positive, not {number}')
    return number


def _non_negative(where: _Where, key: str, value) -> float:
    number = _number(where, key, value)
    if number < 0.0:
        raise where.error(key, f'must be 0 or positive, not {number}')
    return number


def _count(where: _Where, key: str, value) -> int:
    '''A positive whole number; 30.0 is 30'''
    number = _number(where, key, value)
    if not number.is_integer() or number < 1:
        raise where.error(key, f'must be a whole number of at least 1, not {value!r}')
    return int(number)


def _boolean(where: _Where, key: str, value) -> bool:
    if not isinstance(value, bool):
        raise where.error(key, f'must be true or false, not {value!r}')
    return value


def _text(where: _Where, key: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise where.error(key, f'must be a non-empty string, not {value!r}')
    return value


def _frame_names(where: _Where, key: str, value) -> list:
    '''The two planner frames, [left, right]'''
    if not isinstance(value, list) or len(value) != 2:
        raise where.error(key, f'must be a list of two frame names, [left, right], not {value!r}')
    return [_text(where, key, name) for name in value]


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
    for key, value in values.items():
        if np.any(value < 0.0):
            slot = int(np.argmax(value < 0.0))
            raise where.error(f'limits.{name}.{key}',
                              f'must be 0 or positive, not {value[slot]} '
                              f'(joint {profile.body_joints[slot]})')

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


def _planner(where: _Where, planner: dict, profile: RobotProfile) -> dict:
    '''The keys the file gives, checked, with enabled and frame_names filled in'''
    checks = {key: _text for key in PLANNER_TEXT_KEYS}
    checks.update({key: _boolean for key in PLANNER_BOOLEAN_KEYS})
    checks.update({key: _positive for key in PLANNER_POSITIVE_KEYS})
    checks.update({key: _non_negative for key in PLANNER_NON_NEGATIVE_KEYS})
    checks.update({key: _count for key in PLANNER_COUNT_KEYS})
    checks['frame_names'] = _frame_names
    out = {}
    for key, value in planner.items():
        if key in PLANNER_BOUND_KEYS:
            out[key] = None if value is None else _number(where, f'planner.{key}', value)
        else:
            out[key] = checks[key](where, f'planner.{key}', value)
    out.setdefault('enabled', True)
    out.setdefault('frame_names', list(profile.planner_frames))
    return out


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
    unmapped = [mode for mode in profile.modes if mode not in SAFETY_OUTPUT_TOPICS]
    if unmapped:
        raise where.error('', f'robot {robot} has golem_safety modes with no known output '
                              f'topic: {", ".join(unmapped)}')
    outputs = sorted({SAFETY_OUTPUT_TOPICS[mode] for mode in profile.modes})
    safety_out = _topic(where, 'topics.safety_out',
                        topics_raw.get('safety_out', robot_defaults['topics.safety_out']))
    if safety_out not in outputs:
        raise where.error('topics.safety_out', f"'{safety_out}' is not a golem_safety output "
                                               f'of robot {robot} ({", ".join(outputs)})')
    topics['safety_out'] = safety_out

    controller_raw = sections['controller']
    controller = {
        key: (_non_negative if key in NON_NEGATIVE_CONTROLLER_KEYS else _positive)(
            where, f'controller.{key}', controller_raw.get(key, default))
        for key, default in CONTROLLER_DEFAULTS.items()
    }
    # /lowstate tick lag at which goals stop, ms; <= 0 turns the check off
    controller['max_state_lag_ms'] = _number(
        where, 'controller.max_state_lag_ms',
        controller_raw.get('max_state_lag_ms', robot_defaults['controller.max_state_lag_ms']))
    controller['imu_frame'] = _text(where, 'controller.imu_frame',
                                    controller_raw.get('imu_frame', profile.imu_frame))
    controller['hold_targets'] = _hold_targets(
        where, controller_raw.get('hold_targets', {}), profile)

    frequency_raw = sections['frequency']
    frequency = {key: _positive(where, f'frequency.{key}', frequency_raw.get(key, default))
                 for key, default in FREQUENCY_DEFAULTS.items()}

    return {
        'robot': robot,
        'source': path,
        'topics': topics,
        'controller': controller,
        'planner': _planner(where, sections['planner'], profile),
        'frequency': frequency,
        'gains': _gains(where, sections['gains'], profile),
        'limits': _clip_limits(where, sections['limits'], profile),
    }
