"""
Run one policy from its spec: commands in, leg targets and gains out, every step.

The runner owns the policy's command processing, observation history, network and
action mapping; the supervisor decides when it steps and when it resets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from golem_lowerbody_rl.policy.action import ActionMap
from golem_lowerbody_rl.policy.commands import CommandProcessor
from golem_lowerbody_rl.policy.observation import (
    Observation, PolicyInput, StepData, TermContext)
from golem_lowerbody_rl.policy.runtime import load_runtime
from golem_lowerbody_rl.policy.spec import PolicySpec
from golem_lowerbody_rl.profile import RobotProfile
from golem_lowerbody_rl.run_config import RunConfig

__all__ = ['LegTargets', 'PolicyError', 'PolicyRunner']


class PolicyError(RuntimeError):
    """Report a policy step that cannot give leg targets: an input or output not finite."""


@dataclass(frozen=True)
class LegTargets:
    """Hold one step's leg commands, in the robot profile's leg order."""

    q: np.ndarray  # float32 rad, position targets
    kp: np.ndarray  # position gains
    kd: np.ndarray  # velocity gains


class PolicyRunner:
    """
    Step one policy: process the commands, build the observation, run the network.

    Not thread-safe: the caller serialises step and reset. After a step, last_input and
    last_output hold what the network got and returned, command and height what the
    policy observed, and onset whether the command processing reset the policy (ALMI's
    command onset) before this step's observation.
    """

    def __init__(self, profile: RobotProfile, run: RunConfig, spec: PolicySpec,
                 share_dir: str):
        """
        Build the policy of spec for profile's robot under run's command limits.

        share_dir is the package share directory the spec's files are relative to.
        Raises ValueError for a joint the robot lacks, a missing file, or a network whose
        input or output size differs from the spec's.
        """
        self.spec = spec
        self.name = spec.name
        self._dt = 1.0 / spec.rate_hz
        self._body_size = len(profile.body_joints)
        observed_slots = profile.slots(spec.observed_joints)
        defaults = np.array([spec.default_pose.get(joint, 0.0)
                             for joint in spec.observed_joints], dtype=np.float32)
        legs = len(profile.legs)
        self._commands = CommandProcessor(spec.command, run.commands)
        self.observation = Observation(spec.groups, TermContext(
            profile=profile, observed_slots=observed_slots, defaults=defaults,
            action_size=legs, share_dir=share_dir))
        self._action = ActionMap(spec.action, np.array(
            [spec.default_pose[leg] for leg in profile.legs], dtype=np.float32))
        self.runtime = load_runtime(spec.runtime, share_dir, self.observation.size, legs)
        self._last_raw = np.zeros(legs, dtype=np.float32)
        self._last_clipped = np.zeros(legs, dtype=np.float32)
        self._phase_origin: float | None = None
        self._last_cmd = np.zeros(3)
        self.last_input = np.zeros(self.observation.size, dtype=np.float32)
        self.last_output = np.zeros(legs, dtype=np.float32)

    @property
    def prepose(self) -> np.ndarray:
        """Return the legs' pre-pose (rad, leg order): their default angles, clipped."""
        return self._action.prepose

    @property
    def kp(self) -> np.ndarray:
        """Return the legs' position gains."""
        return self._action.kp.copy()

    @property
    def kd(self) -> np.ndarray:
        """Return the legs' velocity gains."""
        return self._action.kd.copy()

    @property
    def onset(self) -> bool:
        """Return whether the last step's command processing reset the policy."""
        return self._commands.onset

    @property
    def command(self) -> np.ndarray:
        """Return the velocity command the last step observed, before its scale."""
        return self._commands.command

    @property
    def height(self) -> float:
        """Return the height command the last step observed, before its scale."""
        return self._commands.height

    def _reset_policy(self) -> None:
        """Clear the network's state, the histories and the last action; restart the phase."""
        self.runtime.reset()
        self.observation.reset()
        self._last_raw = np.zeros_like(self._last_raw)
        self._last_clipped = np.zeros_like(self._last_clipped)
        self._phase_origin = None

    def reset(self, seed_cmd=None) -> None:
        """
        Reset the policy as at the start of an episode.

        The network's state, the observation histories and the last action are cleared,
        and the gait phase restarts at the next step. The command processing is re-seeded
        from seed_cmd (vx, vy, yaw), or from the last command step was given (zero before
        the first) when it is None. Raises PolicyError for a seed that is not finite.
        """
        try:
            self._commands.reset(self._last_cmd if seed_cmd is None else seed_cmd)
        except ValueError as error:
            raise PolicyError(f'policy {self.name}: {error}') from None
        self._reset_policy()

    def _check_input(self, inp: PolicyInput) -> None:
        """Raise PolicyError naming the first input the policy uses that is not finite."""
        for field in ('q', 'dq'):
            values = np.asarray(getattr(inp, field), dtype=np.float64)
            if values.shape != (self._body_size,):
                raise PolicyError(f'policy {self.name}: {field} has shape {values.shape}, '
                                  f'not one value per body slot ({self._body_size})')
            bad = [int(slot) for slot in self.observation.slots
                   if not np.isfinite(values[slot])]
            if bad:
                raise PolicyError(f'policy {self.name}: {field} is not finite at slots {bad}')
        for field, size in (('gyro', 3), ('quat_wxyz', 4), ('cmd', 3)):
            values = np.asarray(getattr(inp, field), dtype=np.float64)
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise PolicyError(f'policy {self.name}: {field} is not {size} finite values: '
                                  f'{values.tolist()}')
        if not np.any(np.asarray(inp.quat_wxyz)):
            raise PolicyError(f'policy {self.name}: quat_wxyz is zero, not a rotation')
        if not np.isfinite(inp.height) or not np.isfinite(inp.t):
            raise PolicyError(f'policy {self.name}: height {inp.height} or t {inp.t} '
                              'is not finite')

    def step(self, inp: PolicyInput) -> LegTargets:
        """
        Return this step's leg targets and gains for inp.

        Raises PolicyError, before changing any state, for an input the policy uses that
        is not finite; and for an observation, network output or target that is not
        finite (the caller then stops the legs for good).
        """
        self._check_input(inp)
        cmd = self._commands.process(inp.cmd, self._dt)
        height = self._commands.process_height(inp.height, self._dt)
        self._last_cmd = np.asarray(inp.cmd, dtype=np.float64).copy()
        if self._commands.onset:
            self._reset_policy()
        if self._phase_origin is None:
            self._phase_origin = float(inp.t)
        step = StepData(inp=inp, command=cmd, height=height,
                        phase_time=float(inp.t) - self._phase_origin,
                        last_action=self._last_raw, last_action_clipped=self._last_clipped)
        try:
            x = self.observation.build(step)
        except ValueError as error:
            raise PolicyError(f'policy {self.name}: {error}') from None
        out = np.asarray(self.runtime(x), dtype=np.float32).reshape(-1)
        if out.shape != self._last_raw.shape or not np.all(np.isfinite(out)):
            raise PolicyError(f'policy {self.name}: the network output is not '
                              f'{self._last_raw.size} finite values: {out.tolist()}')
        clipped, targets = self._action(out)
        if not np.all(np.isfinite(targets)):
            raise PolicyError(f'policy {self.name}: the leg targets are not finite: '
                              f'{targets.tolist()}')
        self._last_raw, self._last_clipped = out, clipped
        self.last_input, self.last_output = x, out
        return LegTargets(q=targets, kp=self._action.kp.copy(), kd=self._action.kd.copy())
