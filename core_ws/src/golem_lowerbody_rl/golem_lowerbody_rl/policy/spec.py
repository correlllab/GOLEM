"""
Load a policy spec: how one policy's observation is built and its action is applied.

Each robot's policies are share/golem_lowerbody_rl/config/<robot>/policies/<name>.yaml,
one per name in its robot.yaml's policies; their keys are documented in
config/example.policy.yaml. Every key is required and an unknown one is refused, and
every joint is checked against the robot's joint table, so a typo cannot fall back to a
default or reach a wrong slot.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import os
import types
from typing import Any

from ament_index_python.packages import get_package_share_directory
import yaml

from golem_lowerbody_rl.checks import (
    check_keys, count, flag, name, names, number, numbers, positive)
from golem_lowerbody_rl.policy.observation import TERMS
from golem_lowerbody_rl.profile import load_profile, RobotProfile

__all__ = [
    'ActionSpec', 'CLIP_TARGET_KEYS', 'CommandSpec', 'GROUP_KEYS', 'GroupSpec', 'HEIGHT_KEYS',
    'HeightSpec', 'ONSET_KEYS', 'OnsetReset', 'PolicySpec', 'RANGE_KEYS', 'RESET_EVENTS',
    'RUNTIME_KINDS', 'RuntimeSpec', 'SECTION_KEYS', 'TOP_LEVEL_KEYS', 'TermSpec',
    'load_policy_spec',
]

TOP_LEVEL_KEYS = ('runtime', 'rate_hz', 'observed_joints', 'default_pose', 'groups',
                  'command', 'action', 'reset_on')
# section -> the keys it must hold
SECTION_KEYS = {
    'runtime': ('kind', 'file', 'reset', 'threads', 'warmup'),
    'command': ('trained_range', 'gain', 'clip', 'smooth_alpha', 'onset_reset', 'height'),
    'action': ('scale', 'clip_raw', 'clip_targets', 'kp', 'kd'),
}
GROUP_KEYS = ('history', 'terms')
RANGE_KEYS = ('vx', 'vy', 'yaw')
ONSET_KEYS = ('rise', 'fall')
HEIGHT_KEYS = ('default', 'range')
CLIP_TARGET_KEYS = ('lower', 'upper')
RUNTIME_KINDS = ('torchscript', 'onnx')
# the events at which the node resets the policy
RESET_EVENTS = ('engage', 'band_release')


@dataclass(frozen=True)
class RuntimeSpec:
    """Hold which network to run and how."""

    kind: str  # one of RUNTIME_KINDS
    file: str  # relative to the package share directory
    reset: str | None  # the TorchScript method that clears the network's state
    threads: int  # intra-op threads
    warmup: bool  # run the network once at load, then reset it


@dataclass(frozen=True)
class TermSpec:
    """Hold one observation term: its TERMS name and checked parameters."""

    term: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class GroupSpec:
    """Hold one observation group: its frames' history length and terms."""

    history: int
    terms: tuple[TermSpec, ...]


@dataclass(frozen=True)
class OnsetReset:
    """Hold the processed command norms that start (rise) and re-arm (fall) an onset."""

    rise: float
    fall: float


@dataclass(frozen=True)
class HeightSpec:
    """Hold the height command's start value and the range it is clipped to."""

    default: float
    range: tuple[float, float] | None


@dataclass(frozen=True)
class CommandSpec:
    """Hold how the policy's velocity and height commands are processed."""

    trained_range: tuple[tuple[float, float], ...] | None  # vx, vy, yaw (low, high)
    gain: tuple[float, float, float] | None
    clip: float | None  # per axis, after the gain
    smooth_alpha: float | None  # the step's weight in an exponential moving average
    onset_reset: OnsetReset | None
    height: HeightSpec


@dataclass(frozen=True)
class ActionSpec:
    """Hold how the network's output becomes leg targets and gains, in leg order."""

    scale: float
    clip_raw: float | None  # the raw output is clipped to +-clip_raw
    clip_targets: tuple[tuple[float, ...], tuple[float, ...]] | None  # rad (lower, upper)
    kp: tuple[float, ...]
    kd: tuple[float, ...]


@dataclass(frozen=True)
class PolicySpec:
    """Hold one policy spec's checked values."""

    name: str  # the spec file's name without .yaml
    path: str
    runtime: RuntimeSpec
    rate_hz: float
    observed_joints: tuple[str, ...]
    default_pose: Mapping[str, float]  # rad; an observed joint not in it is 0
    groups: tuple[GroupSpec, ...]
    command: CommandSpec
    action: ActionSpec
    reset_on: tuple[str, ...]


def _choice(where: str, value, choices: tuple[str, ...]) -> str:
    """Return value if it is one of choices."""
    if value not in choices:
        raise ValueError(f'{where} must be one of {", ".join(choices)}, not {value!r}')
    return value


def _range(where: str, value) -> tuple[float, float]:
    """Return value if it is a [low, high] list with low <= high."""
    low, high = numbers(where, value, 2)
    if low > high:
        raise ValueError(f'{where} {[low, high]} must be [low, high] with low <= high')
    return low, high


def _optional(value, check):
    """Return None for a null value, else check(value)."""
    return None if value is None else check(value)


def _runtime(raw) -> RuntimeSpec:
    """Return the checked runtime section."""
    kind = _choice('runtime.kind', raw['kind'], RUNTIME_KINDS)
    reset = _optional(raw['reset'], lambda value: name('runtime.reset', value))
    # an ONNX network is a function of its input: there is no state to clear
    if kind == 'onnx' and reset is not None:
        raise ValueError('runtime.reset must be null for an onnx network')
    return RuntimeSpec(kind=kind, file=name('runtime.file', raw['file']), reset=reset,
                       threads=count('runtime.threads', raw['threads']),
                       warmup=flag('runtime.warmup', raw['warmup']))


def _trained_range(raw) -> tuple[tuple[float, float], ...]:
    """Return the checked trained range per axis, vx, vy, yaw."""
    check_keys('command.trained_range', raw, RANGE_KEYS)
    ranges = tuple(_range(f'command.trained_range.{axis}', raw[axis]) for axis in RANGE_KEYS)
    for axis, (low, high) in zip(RANGE_KEYS, ranges):
        # SAFETY: a stale or non-finite /cmd_vel becomes zero velocity, which the
        # policy must then get unchanged
        if not low <= 0.0 <= high:
            raise ValueError(f'command.trained_range.{axis} {[low, high]} must contain 0')
    return ranges


def _onset(raw) -> OnsetReset:
    """Return the checked onset thresholds."""
    check_keys('command.onset_reset', raw, ONSET_KEYS)
    rise = positive('command.onset_reset.rise', raw['rise'])
    fall = positive('command.onset_reset.fall', raw['fall'])
    if fall > rise:
        raise ValueError(f'command.onset_reset.fall must not exceed rise ({rise}), not {fall}')
    return OnsetReset(rise=rise, fall=fall)


def _height(raw) -> HeightSpec:
    """Return the checked height command."""
    check_keys('command.height', raw, HEIGHT_KEYS)
    default = number('command.height.default', raw['default'])
    height_range = _optional(raw['range'], lambda value: _range('command.height.range', value))
    if height_range is not None and not height_range[0] <= default <= height_range[1]:
        raise ValueError(f'command.height.default {default} must lie in range '
                         f'{list(height_range)}')
    return HeightSpec(default=default, range=height_range)


def _command(raw) -> CommandSpec:
    """Return the checked command section."""
    alpha = raw['smooth_alpha']
    if alpha is not None:
        alpha = number('command.smooth_alpha', alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f'command.smooth_alpha must be in (0, 1], not {alpha}')
    gain = _optional(raw['gain'], lambda value: numbers('command.gain', value, 3))
    return CommandSpec(
        trained_range=_optional(raw['trained_range'], _trained_range),
        gain=None if gain is None else (gain[0], gain[1], gain[2]),
        clip=_optional(raw['clip'], lambda value: positive('command.clip', value)),
        smooth_alpha=alpha,
        onset_reset=_optional(raw['onset_reset'], _onset),
        height=_height(raw['height']),
    )


def _gains(where: str, value, legs: int, zero_ok: bool) -> tuple[float, ...]:
    """Return one gain per leg, each positive (or zero, with zero_ok)."""
    gains = numbers(where, value, legs)
    if min(gains) < 0.0 or (not zero_ok and min(gains) == 0.0):
        kind = 'non-negative' if zero_ok else 'positive'
        raise ValueError(f'{where} must each be {kind}, not {list(gains)}')
    return gains


def _action(raw, legs: int) -> ActionSpec:
    """Return the checked action section for legs leg joints."""
    clip_targets = raw['clip_targets']
    if clip_targets is not None:
        check_keys('action.clip_targets', clip_targets, CLIP_TARGET_KEYS)
        lower = numbers('action.clip_targets.lower', clip_targets['lower'], legs)
        upper = numbers('action.clip_targets.upper', clip_targets['upper'], legs)
        if any(low > high for low, high in zip(lower, upper)):
            raise ValueError('action.clip_targets.lower must not exceed upper')
        clip_targets = (lower, upper)
    return ActionSpec(
        scale=positive('action.scale', raw['scale']),
        clip_raw=_optional(raw['clip_raw'], lambda value: positive('action.clip_raw', value)),
        clip_targets=clip_targets,
        kp=_gains('action.kp', raw['kp'], legs, zero_ok=False),
        kd=_gains('action.kd', raw['kd'], legs, zero_ok=True),
    )


def _groups(raw) -> tuple[GroupSpec, ...]:
    """Return the checked observation groups, each term's parameters by its TERMS entry."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f'groups must be a non-empty list, not {raw!r}')
    groups = []
    for index, group in enumerate(raw):
        where = f'groups.{index}'
        check_keys(where, group, GROUP_KEYS)
        if not isinstance(group['terms'], list) or not group['terms']:
            raise ValueError(f'{where}.terms must be a non-empty list of terms')
        terms = []
        for position, term in enumerate(group['terms']):
            term_where = f'{where}.terms.{position}'
            if not isinstance(term, dict) or 'term' not in term:
                raise ValueError(f"{term_where} must be a mapping with a 'term' name")
            if term['term'] not in TERMS:
                raise ValueError(f"{term_where}: unknown term '{term['term']}': one of "
                                 f'{", ".join(TERMS)}')
            params = {key: value for key, value in term.items() if key != 'term'}
            terms.append(TermSpec(term=term['term'], params=types.MappingProxyType(
                TERMS[term['term']].parse(term_where, params))))
        groups.append(GroupSpec(history=count(f'{where}.history', group['history']),
                                terms=tuple(terms)))
    return tuple(groups)


def _joints(where: str, joint_names, profile: RobotProfile) -> None:
    """Raise ValueError naming the first of joint_names that is not a joint of the robot."""
    for joint in joint_names:
        if joint not in profile.body_joints:
            raise ValueError(f"{where}: '{joint}' is not a joint of robot {profile.robot} "
                             "(golem_safety's robots/<robot>.yaml)")


def _default_pose(raw, observed: tuple[str, ...], profile: RobotProfile) -> Mapping[str, float]:
    """Return the checked default pose: observed joints or legs, every leg given."""
    if not isinstance(raw, dict):
        raise ValueError(f'default_pose must be a mapping of joint names to rad, not {raw!r}')
    _joints('default_pose', raw, profile)
    for joint in raw:
        if joint not in observed and joint not in profile.legs:
            raise ValueError(f"default_pose: '{joint}' is neither an observed joint nor a leg")
    for leg in profile.legs:
        # SAFETY: a leg's default is its pre-pose and its action offset; none is implied
        if leg not in raw:
            raise ValueError(f"default_pose has no angle for leg '{leg}'")
    return types.MappingProxyType(
        {joint: number(f'default_pose.{joint}', value) for joint, value in raw.items()})


def _check(raw, profile: RobotProfile, spec_name: str, path: str) -> PolicySpec:
    """Return the PolicySpec of raw for profile's robot."""
    check_keys('', raw, TOP_LEVEL_KEYS)
    for section, keys in SECTION_KEYS.items():
        check_keys(section, raw[section], keys)
    observed = names('observed_joints', raw['observed_joints'])
    _joints('observed_joints', observed, profile)
    groups = _groups(raw['groups'])
    for index, group in enumerate(groups):
        for position, term in enumerate(group.terms):
            if 'joints' in term.params:
                _joints(f'groups.{index}.terms.{position}.joints', term.params['joints'],
                        profile)
    reset_on = raw['reset_on']
    if (not isinstance(reset_on, list) or len(set(reset_on)) != len(reset_on)
            or not set(reset_on) <= set(RESET_EVENTS)):
        raise ValueError(f'reset_on must be a list of distinct events from '
                         f'{", ".join(RESET_EVENTS)}, not {reset_on!r}')
    return PolicySpec(
        name=spec_name,
        path=path,
        runtime=_runtime(raw['runtime']),
        rate_hz=positive('rate_hz', raw['rate_hz']),
        observed_joints=observed,
        default_pose=_default_pose(raw['default_pose'], observed, profile),
        groups=groups,
        command=_command(raw['command']),
        action=_action(raw['action'], len(profile.legs)),
        reset_on=tuple(reset_on),
    )


def load_policy_spec(robot: str, name_or_path: str) -> PolicySpec:
    """
    Return the checked spec of one of robot's policies.

    name_or_path is one of the robot's policies (the installed
    config/<robot>/policies/<name>.yaml) or the path of a .yaml file. Raises ValueError,
    naming the file and the key, for an unknown robot or policy, a missing or unknown key,
    a value of the wrong type or out of range, or a joint that is not the robot's.
    """
    profile = load_profile(robot)
    if name_or_path.endswith('.yaml'):
        path = name_or_path
    elif name_or_path in profile.policies:
        path = os.path.join(get_package_share_directory('golem_lowerbody_rl'), 'config',
                            robot, 'policies', f'{name_or_path}.yaml')
    else:
        raise ValueError(f"unknown policy '{name_or_path}' of robot {robot}: one of "
                         f"{', '.join(profile.policies)}, or the path of a .yaml file")
    with open(path, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    spec_name = os.path.splitext(os.path.basename(path))[0]
    try:
        return _check(raw, profile, spec_name, path)
    except ValueError as error:
        raise ValueError(f'policy spec of robot {robot} ({path}): {error}') from None
