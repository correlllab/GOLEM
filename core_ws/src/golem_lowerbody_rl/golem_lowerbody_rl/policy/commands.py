"""
Turn the supervisor's velocity and height commands into the ones a policy observes.

The velocity command goes, in order, through the run config's limits, the policy's
trained range, its gain, its clip, its smoothing (with ALMI's onset detection) and the run
config's acceleration limit; the height command through the policy's range and the run
config's rate limit. Each stage is skipped where its config is null.
"""

from __future__ import annotations

import numpy as np

from golem_lowerbody_rl.policy.spec import CommandSpec
from golem_lowerbody_rl.run_config import Commands

__all__ = ['CommandProcessor']


def _finite_command(raw) -> np.ndarray:
    """Return raw as a float64 (vx, vy, yaw); ValueError if it is not 3 finite values."""
    cmd = np.asarray(raw, dtype=np.float64).reshape(-1)
    if cmd.shape != (3,) or not np.all(np.isfinite(cmd)):
        raise ValueError(f'the velocity command must be 3 finite values, not {cmd.tolist()}')
    return cmd


class CommandProcessor:
    """
    Process one policy's velocity and height commands, step by step.

    A reset (reset) re-seeds the smoothing filter and the onset detector from a seed
    command; the acceleration-limited command and the height keep their state, so what
    the policy observes moves only as fast as the run config allows.
    """

    def __init__(self, spec: CommandSpec, limits: Commands):
        """Keep the policy's command spec and the run config's limits; start at rest."""
        self._spec = spec
        self._low = np.array([limits.vx[0], limits.vy[0], limits.yaw[0]])
        self._high = np.array([limits.vx[1], limits.vy[1], limits.yaw[1]])
        self._accel = None if limits.accel is None else np.array(limits.accel)
        self._height_rate = limits.height_rate
        self._trained = (None if spec.trained_range is None
                         else np.array(spec.trained_range, dtype=np.float64).T)
        self._gain = None if spec.gain is None else np.array(spec.gain, dtype=np.float64)
        self._smoothed = np.zeros(3)
        self._walking = False
        self._onset = False
        self._command = np.zeros(3)
        self._height = spec.height.default

    @property
    def onset(self) -> bool:
        """Return whether the last process call found an onset, re-seeding the processor."""
        return self._onset

    @property
    def command(self) -> np.ndarray:
        """Return the last processed velocity command; zero before the first."""
        return self._command.copy()

    @property
    def height(self) -> float:
        """Return the last processed height command; the spec's default before the first."""
        return self._height

    def _shape(self, cmd: np.ndarray) -> np.ndarray:
        """Return cmd through the limits, the trained range, the gain and the clip."""
        cmd = np.clip(cmd, self._low, self._high)
        if self._trained is not None:
            cmd = np.clip(cmd, self._trained[0], self._trained[1])
        if self._gain is not None:
            cmd = cmd * self._gain
        if self._spec.clip is not None:
            cmd = np.clip(cmd, -self._spec.clip, self._spec.clip)
        return cmd

    def _seed(self, shaped: np.ndarray) -> None:
        """Seed the smoothing filter and the onset detector with a shaped command."""
        self._smoothed = shaped.copy()
        onset = self._spec.onset_reset
        self._walking = onset is not None and float(np.linalg.norm(shaped)) >= onset.rise

    def reset(self, seed_cmd) -> None:
        """Re-seed the smoothing filter and the onset detector from seed_cmd (vx, vy, yaw)."""
        self._seed(self._shape(_finite_command(seed_cmd)))

    def process(self, raw_cmd, dt: float) -> np.ndarray:
        """
        Return the velocity command (float64) the policy observes this step.

        dt (s) is the policy step. When the smoothed command's norm rises through the
        spec's onset_reset.rise from below fall, the processor re-seeds itself with this
        step's shaped command, hands that on, and onset is true until the next call.
        Raises ValueError for a command that is not 3 finite values, changing nothing.
        """
        shaped = self._shape(_finite_command(raw_cmd))
        cmd = shaped
        if self._spec.smooth_alpha is not None:
            self._smoothed = self._smoothed + self._spec.smooth_alpha * (shaped - self._smoothed)
            cmd = self._smoothed.copy()
        self._onset = False
        onset = self._spec.onset_reset
        if onset is not None:
            norm = float(np.linalg.norm(cmd))
            if not self._walking and norm >= onset.rise:
                self._seed(shaped)
                cmd = shaped
                self._onset = True
            elif self._walking and norm < onset.fall:
                self._walking = False
        if self._accel is not None:
            step = self._accel * dt
            cmd = self._command + np.clip(cmd - self._command, -step, step)
        self._command = np.array(cmd, dtype=np.float64)
        return self._command.copy()

    def process_height(self, raw: float, dt: float) -> float:
        """
        Return the height command the policy observes this step.

        dt (s) is the policy step. Raises ValueError for a value that is not finite,
        changing nothing.
        """
        height = float(raw)
        if not np.isfinite(height):
            raise ValueError(f'the height command must be finite, not {raw!r}')
        height_range = self._spec.height.range
        if height_range is not None:
            height = min(max(height, height_range[0]), height_range[1])
        if self._height_rate is not None:
            step = self._height_rate * dt
            height = self._height + min(max(height - self._height, -step), step)
        self._height = height
        return height
