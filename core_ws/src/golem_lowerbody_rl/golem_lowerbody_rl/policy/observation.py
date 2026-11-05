"""
Build a policy's network input from its spec's observation groups and terms.

A term (TERMS, by the name a spec gives it) turns one step's data into a few values; a
group concatenates its terms into one frame and keeps its last `history` frames, oldest
first and zero-filled at a reset; the network input is the groups' histories in order.
Values are computed in float64 and each frame is stored in float32, the networks' type.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from golem_lowerbody_rl.checks import (
    check_keys, count, flag, name, names, number, numbers, positive)
from golem_lowerbody_rl.imu import projected_gravity
from golem_lowerbody_rl.profile import RobotProfile

__all__ = ['ENCODER_KEYS', 'GAIT_FORMS', 'Observation', 'PolicyInput', 'StepData',
           'TERMS', 'TermContext']

# the keys of the latent term's encoder mapping
ENCODER_KEYS = ('file', 'hidden', 'activation', 'latent')
# gait_phase forms: (sin, cos) of the phase, or the sine of the left and right feet's phases
GAIT_FORMS = ('sin_cos', 'left_right_sin')


@dataclass(frozen=True)
class PolicyInput:
    """Hold one step's measured state and commands, as the supervisor hands them on."""

    t: float  # s of the node's clock: the gait phase's time
    q: np.ndarray  # rad, every body slot of /lowstate, in slot order
    dq: np.ndarray  # rad/s, every body slot
    quat_wxyz: np.ndarray  # the IMU orientation (trimmed), w first
    gyro: np.ndarray  # rad/s, the IMU's angular velocity in its frame
    cmd: np.ndarray  # vx (m/s), vy (m/s), yaw (rad/s), zero when stale or not finite
    height: float  # the height command, in the policy's own units


@dataclass(frozen=True)
class StepData:
    """Hold what the terms read in one step."""

    inp: PolicyInput
    command: np.ndarray  # the processed velocity command (CommandProcessor.process)
    height: float  # the processed height command
    phase_time: float  # s since the gait phase's origin (the first step after a reset)
    last_action: np.ndarray  # the network's previous output, zero after a reset
    last_action_clipped: np.ndarray  # the same, clipped to action.clip_raw


@dataclass(frozen=True)
class TermContext:
    """Hold what a term needs to build: the robot, the observed joints, the files."""

    profile: RobotProfile
    observed_slots: np.ndarray  # /lowstate slots of the spec's observed_joints, in order
    defaults: np.ndarray  # float32 rad, the default angle of each observed joint
    action_size: int  # the network's outputs: one per leg
    share_dir: str  # the package share directory files are relative to


class Term:
    """Turn one step's data into values of the observation (a base for TERMS' entries)."""

    NAME = ''
    # the keys the spec gives the term, besides 'term'
    KEYS: tuple[str, ...] = ()
    size = 0

    def __init__(self, params: Mapping[str, Any], context: TermContext | None):
        """Build the term from its parsed parameters and the robot's context."""

    @classmethod
    def parse(cls, where: str, raw) -> dict[str, Any]:
        """Return the term's checked parameters from raw, its spec mapping without 'term'."""
        check_keys(where, raw, cls.KEYS)
        return cls._parse(where, raw)

    @classmethod
    def _parse(cls, where: str, raw: dict) -> dict[str, Any]:
        """Return raw's values checked; a term with keys overrides it."""
        return {}

    def __call__(self, step: StepData) -> np.ndarray:
        """Return the term's values for step."""
        raise NotImplementedError


class _Scaled(Term):
    """Read a term's one parameter, a number 'scale' (a base for such terms)."""

    KEYS = ('scale',)

    @classmethod
    def _parse(cls, where: str, raw: dict) -> dict[str, Any]:
        """Return the checked scale."""
        return {'scale': number(f'{where}.scale', raw['scale'])}


class AngVel(_Scaled):
    """Observe the IMU's angular velocity times scale (rad/s)."""

    NAME = 'ang_vel'
    size = 3

    def __init__(self, params: Mapping[str, Any], context: TermContext | None):
        """Keep the scale."""
        self._scale = params['scale']

    def __call__(self, step: StepData) -> np.ndarray:
        """Return gyro * scale."""
        return np.asarray(step.inp.gyro, dtype=np.float64) * self._scale


class ProjectedGravity(Term):
    """Observe gravity's unit direction in the IMU frame, (0, 0, -1) when level."""

    NAME = 'projected_gravity'
    size = 3

    def __init__(self, params: Mapping[str, Any], context: TermContext | None):
        """Keep nothing: the term has no parameters."""

    def __call__(self, step: StepData) -> np.ndarray:
        """Return gravity from the normalised IMU quaternion."""
        return projected_gravity(step.inp.quat_wxyz)


class VelocityCommand(Term):
    """Observe the processed velocity command times a scale per axis."""

    NAME = 'velocity_command'
    KEYS = ('scale',)
    size = 3

    @classmethod
    def _parse(cls, where: str, raw: dict) -> dict[str, Any]:
        """Return the three per-axis scales."""
        return {'scale': numbers(f'{where}.scale', raw['scale'], 3)}

    def __init__(self, params: Mapping[str, Any], context: TermContext | None):
        """Keep the scales."""
        self._scale = np.array(params['scale'], dtype=np.float64)

    def __call__(self, step: StepData) -> np.ndarray:
        """Return command * scale."""
        return step.command * self._scale


class HeightCommand(_Scaled):
    """Observe the processed height command times scale."""

    NAME = 'height_command'
    size = 1

    def __init__(self, params: Mapping[str, Any], context: TermContext | None):
        """Keep the scale."""
        self._scale = params['scale']

    def __call__(self, step: StepData) -> np.ndarray:
        """Return [height * scale]."""
        return np.array([step.height * self._scale])


class JointPosRel(_Scaled):
    """Observe each observed joint's position less its default angle, times scale."""

    NAME = 'joint_pos_rel'

    def __init__(self, params: Mapping[str, Any], context: TermContext):
        """Keep the scale, the observed slots and their default angles."""
        self._scale = params['scale']
        self._slots = context.observed_slots
        self._defaults = context.defaults.astype(np.float64)
        self.size = self._slots.size

    def __call__(self, step: StepData) -> np.ndarray:
        """Return (q - default) * scale over the observed joints."""
        q = np.asarray(step.inp.q, dtype=np.float64)[self._slots]
        return (q - self._defaults) * self._scale


class JointVel(_Scaled):
    """Observe each observed joint's velocity times scale."""

    NAME = 'joint_vel'

    def __init__(self, params: Mapping[str, Any], context: TermContext):
        """Keep the scale and the observed slots."""
        self._scale = params['scale']
        self._slots = context.observed_slots
        self.size = self._slots.size

    def __call__(self, step: StepData) -> np.ndarray:
        """Return dq * scale over the observed joints."""
        return np.asarray(step.inp.dq, dtype=np.float64)[self._slots] * self._scale


class LastAction(Term):
    """Observe the network's previous output, raw or clipped to action.clip_raw."""

    NAME = 'last_action'
    KEYS = ('clipped',)

    @classmethod
    def _parse(cls, where: str, raw: dict) -> dict[str, Any]:
        """Return whether the clipped action is observed."""
        return {'clipped': flag(f'{where}.clipped', raw['clipped'])}

    def __init__(self, params: Mapping[str, Any], context: TermContext):
        """Keep which action is observed and its size."""
        self._clipped = params['clipped']
        self.size = context.action_size

    def __call__(self, step: StepData) -> np.ndarray:
        """Return the previous action."""
        action = step.last_action_clipped if self._clipped else step.last_action
        return np.asarray(action, dtype=np.float64)


class GaitPhase(Term):
    """
    Observe a gait clock: two values of the phase, a fraction of period since the origin.

    sin_cos gives (sin, cos) of 2 pi phase; left_right_sin gives the sines of 2 pi phase
    and 2 pi (phase + offset), the left and right feet's clocks. With stand_below, both
    values are 0 while the processed command's norm is below it: the policy stands.
    """

    NAME = 'gait_phase'
    KEYS = ('period', 'form', 'offset', 'stand_below')
    size = 2

    @classmethod
    def _parse(cls, where: str, raw: dict) -> dict[str, Any]:
        """Return the checked period (s), form, offset (periods) and stand threshold."""
        form = raw['form']
        if form not in GAIT_FORMS:
            raise ValueError(f'{where}.form must be one of {", ".join(GAIT_FORMS)}, '
                             f'not {form!r}')
        offset = raw['offset']
        if form == 'sin_cos':
            if offset is not None:
                raise ValueError(f'{where}.offset must be null for form sin_cos')
        else:
            offset = number(f'{where}.offset', offset)
        stand_below = raw['stand_below']
        return {
            'period': positive(f'{where}.period', raw['period']),
            'form': form,
            'offset': offset,
            'stand_below': (None if stand_below is None
                            else positive(f'{where}.stand_below', stand_below)),
        }

    def __init__(self, params: Mapping[str, Any], context: TermContext | None):
        """Keep the clock's parameters."""
        self._period = params['period']
        self._form = params['form']
        self._offset = params['offset']
        self._stand_below = params['stand_below']

    def __call__(self, step: StepData) -> np.ndarray:
        """Return the two clock values for step's phase time and command."""
        if (self._stand_below is not None
                and float(np.linalg.norm(step.command)) < self._stand_below):
            return np.zeros(2)
        phase = (step.phase_time % self._period) / self._period
        if self._form == 'sin_cos':
            return np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
        return np.array([np.sin(2 * np.pi * phase),
                         np.sin(2 * np.pi * ((phase + self._offset) % 1.0))])


class Latent(Term):
    """
    Observe an encoder's latent of the given joints' raw positions and fixed forces.

    The encoder is an EnvFactorEncoder (policy/encoder.py) loaded from its state_dict,
    whose input is the joints' positions (rad, unscaled) then each force (3 values).
    last_input and last_latent hold the latest step's encoder input and output.
    """

    NAME = 'latent'
    KEYS = ('encoder', 'joints', 'forces')

    @classmethod
    def _parse(cls, where: str, raw: dict) -> dict[str, Any]:
        """Return the checked encoder mapping, joint names and forces."""
        encoder = raw['encoder']
        check_keys(f'{where}.encoder', encoder, ENCODER_KEYS)
        hidden = encoder['hidden']
        if not isinstance(hidden, list) or not hidden:
            raise ValueError(f'{where}.encoder.hidden must be a non-empty list of widths, '
                             f'not {hidden!r}')
        forces = raw['forces']
        if not isinstance(forces, list):
            raise ValueError(f'{where}.forces must be a list of [x, y, z] forces, '
                             f'not {forces!r}')
        return {
            'encoder': {
                'file': name(f'{where}.encoder.file', encoder['file']),
                'hidden': tuple(count(f'{where}.encoder.hidden', width) for width in hidden),
                'activation': name(f'{where}.encoder.activation', encoder['activation']),
                'latent': count(f'{where}.encoder.latent', encoder['latent']),
            },
            'joints': names(f'{where}.joints', raw['joints']),
            'forces': tuple(numbers(f'{where}.forces', force, 3) for force in forces),
        }

    def __init__(self, params: Mapping[str, Any], context: TermContext):
        """Load the encoder; raise ValueError if its file is missing or does not fit."""
        import torch

        from golem_lowerbody_rl.policy.encoder import EnvFactorEncoder, EnvFactorEncoderCfg

        self._torch = torch
        encoder = params['encoder']
        self.slots = context.profile.slots(params['joints'])
        self._forces = np.array(params['forces'], dtype=np.float32).reshape(-1)
        path = os.path.join(context.share_dir, encoder['file'])
        if not os.path.isfile(path):
            raise ValueError(f'the latent encoder {path} does not exist')
        try:
            self._encoder = EnvFactorEncoder(EnvFactorEncoderCfg(
                in_dim=self.slots.size + self._forces.size, latent_dim=encoder['latent'],
                hidden_dims=encoder['hidden'], activation=encoder['activation']))
            self._encoder.load_state_dict(
                torch.load(path, map_location='cpu', weights_only=True))
        except (RuntimeError, ValueError) as error:
            raise ValueError(f'the latent encoder {path} does not fit its spec: {error}') \
                from None
        self._encoder.eval()
        self.size = encoder['latent']
        self.last_input = np.zeros(self.slots.size + self._forces.size, dtype=np.float32)
        self.last_latent = np.zeros(self.size, dtype=np.float32)

    def __call__(self, step: StepData) -> np.ndarray:
        """Return the encoder's latent (float32) of step's joint positions and the forces."""
        e_t = np.concatenate([np.asarray(step.inp.q, dtype=np.float32)[self.slots],
                              self._forces])
        with self._torch.no_grad():
            z = self._encoder(self._torch.from_numpy(e_t).unsqueeze(0)).numpy().reshape(-1)
        self.last_input, self.last_latent = e_t, z
        return z


# every term a spec may name, by that name
TERMS: dict[str, type[Term]] = {
    term.NAME: term for term in (
        AngVel, ProjectedGravity, VelocityCommand, HeightCommand, JointPosRel, JointVel,
        LastAction, GaitPhase, Latent)
}


class Observation:
    """Build the network input from a spec's groups; hold each group's history."""

    def __init__(self, groups, context: TermContext):
        """Build each group's terms; groups is a PolicySpec's groups."""
        self._groups = []
        for group in groups:
            terms = tuple(TERMS[term.term](term.params, context) for term in group.terms)
            size = sum(term.size for term in terms)
            self._groups.append((terms, np.zeros((group.history, size), dtype=np.float32)))
        self.terms: tuple[Term, ...] = tuple(
            term for terms, _ in self._groups for term in terms)
        self.size = sum(history.size for _, history in self._groups)
        # the joint slots the terms read, for the runner's input check
        self.slots = np.unique(np.concatenate(
            [context.observed_slots] + [term.slots for term in self.terms
                                        if isinstance(term, Latent)]))

    def newest(self, term_name: str) -> list[slice]:
        """Return where each term called term_name lies in the input's newest frames."""
        found, offset = [], 0
        for terms, history in self._groups:
            start = offset + history.size - history.shape[1]
            for term in terms:
                if term.NAME == term_name:
                    found.append(slice(start, start + term.size))
                start += term.size
            offset += history.size
        return found

    def reset(self) -> None:
        """Zero every group's history."""
        for _, history in self._groups:
            history[:] = 0.0

    def build(self, step: StepData) -> np.ndarray:
        """
        Return the network input (float32) with step's frames appended to the histories.

        Raises ValueError, naming the term, for a value that is not finite; the histories
        are then unchanged.
        """
        frames = []
        for index, (terms, _) in enumerate(self._groups):
            values = [np.asarray(term(step), dtype=np.float64).reshape(-1) for term in terms]
            for term, value in zip(terms, values):
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"observation term '{term.NAME}' of group {index} is "
                                     f'not finite: {value.tolist()}')
            frames.append(np.concatenate(values).astype(np.float32))
        for (_, history), frame in zip(self._groups, frames):
            history[:-1] = history[1:]
            history[-1] = frame
        return np.concatenate([history.reshape(-1) for _, history in self._groups])
