"""
Load a run config: what differs between the sim and the real robot for the leg node.

Each robot has share/golem_lowerbody_rl/config/<robot>/sim.yaml and real.yaml; their keys
are documented in config/example.run.yaml. Every key is required and an unknown one is
refused, so a typo cannot fall back to a default.
"""

from dataclasses import dataclass
import math
import os

from ament_index_python.packages import get_package_share_directory
import yaml

from golem_lowerbody_rl.profile import known_robots

__all__ = [
    'Band', 'Commands', 'Faults', 'Heartbeat', 'RUN_NAMES', 'RunConfig', 'SECTION_KEYS',
    'Startup', 'TOP_LEVEL_KEYS', 'load_run_config',
]

# section -> the keys it must hold
SECTION_KEYS = {
    'startup': ('wait_for_confirm', 'ramp_seconds', 'settle_tolerance', 'settle_ticks',
                'settle_max_ticks', 'steady_seconds', 'tick_rate', 'upper_fresh_seconds'),
    'heartbeat': ('timeout',),
    'band': ('release', 'max_wait'),
    'commands': ('timeout', 'vx', 'vy', 'yaw', 'accel', 'height_rate'),
    'faults': ('max_state_gap', 'max_latency'),
}
TOP_LEVEL_KEYS = ('sim',) + tuple(SECTION_KEYS) + ('imu_trim_deg',)
# the installed run configs of each robot, by name; a file named after one must have
# sim true exactly for 'sim'
RUN_NAMES = ('sim', 'real')
# golem_safety's /safety/heartbeat period (s); heartbeat.timeout must be longer
HEARTBEAT_PERIOD = 0.1


@dataclass(frozen=True)
class Startup:
    """Hold how the legs pre-pose and when the policy may start."""

    wait_for_confirm: bool  # hold the pre-pose until /lowerbody/confirm_engage
    ramp_seconds: float  # s of the pre-pose ramp
    settle_tolerance: float  # rad, largest leg error that counts as posed
    settle_ticks: int  # consecutive policy steps within settle_tolerance
    settle_max_ticks: int  # policy steps after the ramp before holding anyway
    steady_seconds: float  # s the start conditions must hold
    tick_rate: tuple[float, float]  # [low, high] /lowstate tick s per s of the node clock
    upper_fresh_seconds: float  # s, the arm controller's command age that counts as live


@dataclass(frozen=True)
class Heartbeat:
    """Hold how long the node may go without golem_safety's heartbeat."""

    timeout: float  # s


@dataclass(frozen=True)
class Band:
    """Hold whether the node releases the sim's elastic band and how long it waits."""

    release: bool
    max_wait: float  # s


@dataclass(frozen=True)
class Commands:
    """Hold the deployment limits on /cmd_vel and the height command."""

    timeout: float  # s, a /cmd_vel older than this is zero velocity
    vx: tuple[float, float]  # m/s
    vy: tuple[float, float]  # m/s
    yaw: tuple[float, float]  # rad/s
    accel: tuple[float, float, float] | None  # m/s^2, m/s^2, rad/s^2; None: unlimited
    height_rate: float | None  # m/s; None: unlimited


@dataclass(frozen=True)
class Faults:
    """Hold the timing limits that stop the legs for good."""

    max_state_gap: float  # s between /lowstate samples
    max_latency: float | None  # s of tick lag while the policy runs; None: unchecked


@dataclass(frozen=True)
class RunConfig:
    """Hold one run config's checked values."""

    sim: bool
    startup: Startup
    heartbeat: Heartbeat
    band: Band
    commands: Commands
    faults: Faults
    imu_trim_deg: tuple[float, float, float]  # deg, roll, pitch, yaw (imu.trim_quaternion)


def _number(where: str, value) -> float:
    """Return value as a float if it is a finite int or float (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{where} must be a finite number, not {value!r}')
    return float(value)


def _positive(where: str, value) -> float:
    """Return value as a float if it is a positive finite number."""
    number = _number(where, value)
    if number <= 0.0:
        raise ValueError(f'{where} must be positive, not {value!r}')
    return number


def _count(where: str, value) -> int:
    """Return value if it is a positive int (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{where} must be a positive integer, not {value!r}')
    return value


def _flag(where: str, value) -> bool:
    """Return value if it is true or false."""
    if not isinstance(value, bool):
        raise ValueError(f'{where} must be true or false, not {value!r}')
    return value


def _numbers(where: str, value, count: int) -> tuple[float, ...]:
    """Return value as a tuple of floats if it is a list of count finite numbers."""
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f'{where} must be a list of {count} numbers, not {value!r}')
    return tuple(_number(where, item) for item in value)


def _range(where: str, value) -> tuple[float, float]:
    """Return value if it is a [low, high] list containing zero."""
    low, high = _numbers(where, value, 2)
    # SAFETY: the node starts at zero velocity and falls back to it on a stale or
    # non-finite /cmd_vel, so zero must lie inside every limit
    if not low <= 0.0 <= high:
        raise ValueError(f'{where} {[low, high]} must be [low, high] with low <= 0 <= high')
    return low, high


def _check_keys(where: str, mapping, keys) -> None:
    """Raise ValueError naming the first missing or unknown key of mapping."""
    if not isinstance(mapping, dict):
        raise ValueError(f'{where or "the file"} must be a mapping')
    prefix = f'{where}.' if where else ''
    for key in mapping:
        if key not in keys:
            raise ValueError(f"unknown key '{prefix}{key}'")
    for key in keys:
        if key not in mapping:
            raise ValueError(f"missing key '{prefix}{key}'")


def _check(raw, role: str | None) -> RunConfig:
    """Return the RunConfig of raw; role 'sim' or 'real' sets the value sim must have."""
    _check_keys('', raw, TOP_LEVEL_KEYS)
    for section, keys in SECTION_KEYS.items():
        _check_keys(section, raw[section], keys)
    sim = _flag('sim', raw['sim'])
    if role is not None and sim != (role == 'sim'):
        raise ValueError(f"'sim' must be {str(role == 'sim').lower()} in a {role}.yaml")

    startup = raw['startup']
    tick_rate = _numbers('startup.tick_rate', startup['tick_rate'], 2)
    # a tick that keeps to the node's clock must always be accepted
    if not 0.0 < tick_rate[0] <= 1.0 <= tick_rate[1]:
        raise ValueError(f'startup.tick_rate {list(tick_rate)} must be [low, high] with '
                         '0 < low <= 1 <= high')
    heartbeat_timeout = _positive('heartbeat.timeout', raw['heartbeat']['timeout'])
    # a timeout within one beat period would stop the node between two beats
    if heartbeat_timeout <= HEARTBEAT_PERIOD:
        raise ValueError(f"heartbeat.timeout must be longer than golem_safety's beat period "
                         f'({HEARTBEAT_PERIOD} s), not {heartbeat_timeout}')
    commands = raw['commands']
    accel = None
    if commands['accel'] is not None:
        accel = tuple(_positive('commands.accel', value)
                      for value in _numbers('commands.accel', commands['accel'], 3))
    height_rate = commands['height_rate']
    trim = _numbers('imu_trim_deg', raw['imu_trim_deg'], 3)
    max_latency = raw['faults']['max_latency']
    return RunConfig(
        sim=sim,
        startup=Startup(
            wait_for_confirm=_flag('startup.wait_for_confirm', startup['wait_for_confirm']),
            ramp_seconds=_positive('startup.ramp_seconds', startup['ramp_seconds']),
            settle_tolerance=_positive('startup.settle_tolerance', startup['settle_tolerance']),
            settle_ticks=_count('startup.settle_ticks', startup['settle_ticks']),
            settle_max_ticks=_count('startup.settle_max_ticks', startup['settle_max_ticks']),
            steady_seconds=_positive('startup.steady_seconds', startup['steady_seconds']),
            tick_rate=(tick_rate[0], tick_rate[1]),
            upper_fresh_seconds=_positive('startup.upper_fresh_seconds',
                                          startup['upper_fresh_seconds']),
        ),
        heartbeat=Heartbeat(timeout=heartbeat_timeout),
        band=Band(
            release=_flag('band.release', raw['band']['release']),
            max_wait=_positive('band.max_wait', raw['band']['max_wait']),
        ),
        commands=Commands(
            timeout=_positive('commands.timeout', commands['timeout']),
            vx=_range('commands.vx', commands['vx']),
            vy=_range('commands.vy', commands['vy']),
            yaw=_range('commands.yaw', commands['yaw']),
            accel=None if accel is None else (accel[0], accel[1], accel[2]),
            height_rate=(None if height_rate is None
                         else _positive('commands.height_rate', height_rate)),
        ),
        faults=Faults(
            max_state_gap=_positive('faults.max_state_gap', raw['faults']['max_state_gap']),
            max_latency=(None if max_latency is None
                         else _positive('faults.max_latency', max_latency)),
        ),
        imu_trim_deg=(trim[0], trim[1], trim[2]),
    )


def load_run_config(robot: str, name_or_path: str) -> RunConfig:
    """
    Return the checked run config of robot.

    name_or_path is one of RUN_NAMES (the installed config/<robot>/<name>.yaml) or the path
    of a .yaml file. A file named sim.yaml or real.yaml must say sim true or false to match.
    Raises ValueError, naming the file and the key, for an unknown robot or name, a missing
    or unknown key, or a value of the wrong type or out of range.
    """
    if robot not in known_robots():
        raise ValueError(f"unknown robot '{robot}': one of {', '.join(known_robots())}")
    if name_or_path.endswith('.yaml'):
        path = name_or_path
    elif name_or_path in RUN_NAMES:
        path = os.path.join(get_package_share_directory('golem_lowerbody_rl'), 'config', robot,
                            f'{name_or_path}.yaml')
    else:
        raise ValueError(f"unknown run config '{name_or_path}' of robot {robot}: one of "
                         f"{', '.join(RUN_NAMES)}, or the path of a .yaml file")
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    try:
        return _check(raw, stem if stem in RUN_NAMES else None)
    except ValueError as error:
        raise ValueError(f'run config of robot {robot} ({path}): {error}') from None
