"""
Load a run config: what differs between the sim and the real robot for the leg node.

Each robot has share/golem_lowerbody_rl/config/<robot>/sim.yaml and real.yaml; their keys
are documented in config/example.run.yaml. Every key is required and an unknown one is
refused, so a typo cannot fall back to a default.
"""

from dataclasses import dataclass
import os

from ament_index_python.packages import get_package_share_directory
import yaml

from golem_lowerbody_rl.checks import check_keys, count, flag, numbers, positive
from golem_lowerbody_rl.profile import known_robots

__all__ = [
    'Band', 'Commands', 'Faults', 'Heartbeat', 'RUN_NAMES', 'RunConfig', 'SECTION_KEYS',
    'Startup', 'TOP_LEVEL_KEYS', 'load_run_config', 'run_config_path',
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
# s: the period of golem_safety's /safety/heartbeat, which safety_node publishes at 10 Hz
# (golem_safety's README); heartbeat.timeout must be longer. Kept here, not read from
# golem_safety, which publishes the rate in no file
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


def _range(where: str, value) -> tuple[float, float]:
    """Return value if it is a [low, high] list containing zero."""
    low, high = numbers(where, value, 2)
    # SAFETY: the node starts at zero velocity and falls back to it on a stale or
    # non-finite /cmd_vel, so zero must lie inside every limit
    if not low <= 0.0 <= high:
        raise ValueError(f'{where} {[low, high]} must be [low, high] with low <= 0 <= high')
    return low, high


def _check(raw, role: str | None) -> RunConfig:
    """Return the RunConfig of raw; role 'sim' or 'real' sets the value sim must have."""
    check_keys('', raw, TOP_LEVEL_KEYS)
    for section, keys in SECTION_KEYS.items():
        check_keys(section, raw[section], keys)
    sim = flag('sim', raw['sim'])
    if role is not None and sim != (role == 'sim'):
        raise ValueError(f"'sim' must be {str(role == 'sim').lower()} in a {role}.yaml")

    startup = raw['startup']
    tick_rate = numbers('startup.tick_rate', startup['tick_rate'], 2)
    # a tick that keeps to the node's clock must always be accepted
    if not 0.0 < tick_rate[0] <= 1.0 <= tick_rate[1]:
        raise ValueError(f'startup.tick_rate {list(tick_rate)} must be [low, high] with '
                         '0 < low <= 1 <= high')
    heartbeat_timeout = positive('heartbeat.timeout', raw['heartbeat']['timeout'])
    # a timeout within one beat period would stop the node between two beats
    if heartbeat_timeout <= HEARTBEAT_PERIOD:
        raise ValueError(f"heartbeat.timeout must be longer than golem_safety's beat period "
                         f'({HEARTBEAT_PERIOD} s), not {heartbeat_timeout}')
    commands = raw['commands']
    accel = None
    if commands['accel'] is not None:
        accel = tuple(positive('commands.accel', value)
                      for value in numbers('commands.accel', commands['accel'], 3))
    height_rate = commands['height_rate']
    trim = numbers('imu_trim_deg', raw['imu_trim_deg'], 3)
    max_latency = raw['faults']['max_latency']
    # SAFETY: tick lag is the tick's age on robot time, which is the steady clock without
    # use_sim_time, and a sim's tick counts sim time: there a sim slower than real time
    # builds up lag and would stop the legs within seconds
    if sim and max_latency is not None:
        raise ValueError('faults.max_latency must be null in a sim run config (sim true): '
                         "a sim's tick counts sim time, and without use_sim_time the lag is "
                         f'measured against the steady clock; not {max_latency!r}')
    return RunConfig(
        sim=sim,
        startup=Startup(
            wait_for_confirm=flag('startup.wait_for_confirm', startup['wait_for_confirm']),
            ramp_seconds=positive('startup.ramp_seconds', startup['ramp_seconds']),
            settle_tolerance=positive('startup.settle_tolerance', startup['settle_tolerance']),
            settle_ticks=count('startup.settle_ticks', startup['settle_ticks']),
            settle_max_ticks=count('startup.settle_max_ticks', startup['settle_max_ticks']),
            steady_seconds=positive('startup.steady_seconds', startup['steady_seconds']),
            tick_rate=(tick_rate[0], tick_rate[1]),
            upper_fresh_seconds=positive('startup.upper_fresh_seconds',
                                         startup['upper_fresh_seconds']),
        ),
        heartbeat=Heartbeat(timeout=heartbeat_timeout),
        band=Band(
            release=flag('band.release', raw['band']['release']),
            max_wait=positive('band.max_wait', raw['band']['max_wait']),
        ),
        commands=Commands(
            timeout=positive('commands.timeout', commands['timeout']),
            vx=_range('commands.vx', commands['vx']),
            vy=_range('commands.vy', commands['vy']),
            yaw=_range('commands.yaw', commands['yaw']),
            accel=None if accel is None else (accel[0], accel[1], accel[2]),
            height_rate=(None if height_rate is None
                         else positive('commands.height_rate', height_rate)),
        ),
        faults=Faults(
            max_state_gap=positive('faults.max_state_gap', raw['faults']['max_state_gap']),
            max_latency=(None if max_latency is None
                         else positive('faults.max_latency', max_latency)),
        ),
        imu_trim_deg=(trim[0], trim[1], trim[2]),
    )


def run_config_path(robot: str, name_or_path: str) -> str:
    """
    Return the file of robot's run config name_or_path.

    name_or_path is one of RUN_NAMES (the installed config/<robot>/<name>.yaml) or the path
    of a .yaml file. Raises ValueError for an unknown robot or name.
    """
    if robot not in known_robots():
        raise ValueError(f"unknown robot '{robot}': one of {', '.join(known_robots())}")
    if name_or_path.endswith('.yaml'):
        return name_or_path
    if name_or_path in RUN_NAMES:
        return os.path.join(get_package_share_directory('golem_lowerbody_rl'), 'config', robot,
                            f'{name_or_path}.yaml')
    raise ValueError(f"unknown run config '{name_or_path}' of robot {robot}: one of "
                     f"{', '.join(RUN_NAMES)}, or the path of a .yaml file")


def load_run_config(robot: str, name_or_path: str) -> RunConfig:
    """
    Return the checked run config of robot.

    name_or_path is one of RUN_NAMES (the installed config/<robot>/<name>.yaml) or the path
    of a .yaml file. A file named sim.yaml or real.yaml must say sim true or false to match.
    Raises ValueError, naming the file and the key, for an unknown robot or name, a missing
    or unknown key, or a value of the wrong type or out of range.
    """
    path = run_config_path(robot, name_or_path)
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    try:
        return _check(raw, stem if stem in RUN_NAMES else None)
    except ValueError as error:
        raise ValueError(f'run config of robot {robot} ({path}): {error}') from None
