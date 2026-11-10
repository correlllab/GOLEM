"""Turn a policy network's output into leg position targets and gains, in leg order."""

from __future__ import annotations

import numpy as np

from golem_lowerbody_rl.policy.spec import ActionSpec

__all__ = ['ActionMap']


class ActionMap:
    """
    Map the network's raw action to leg targets: clip, scale, offset, clip.

    targets = clip(raw, +-clip_raw) * scale + default, then clipped to clip_targets, in
    float32 (rad); default is each leg's default_pose angle.
    """

    def __init__(self, spec: ActionSpec, default: np.ndarray):
        """Keep the spec and the legs' default angles (rad, leg order)."""
        self._scale = np.float32(spec.scale)
        self._clip_raw = None if spec.clip_raw is None else np.float32(spec.clip_raw)
        self._default = np.asarray(default, dtype=np.float32)
        self._limits = (None if spec.clip_targets is None
                        else tuple(np.array(side, dtype=np.float32)
                                   for side in spec.clip_targets))
        self.kp = np.array(spec.kp, dtype=np.float64)
        self.kd = np.array(spec.kd, dtype=np.float64)

    def clip_targets(self, targets: np.ndarray) -> np.ndarray:
        """Return targets clipped to the spec's clip_targets; unchanged without them."""
        if self._limits is None:
            return targets
        return np.clip(targets, self._limits[0], self._limits[1])

    @property
    def prepose(self) -> np.ndarray:
        """Return the legs' default angles clipped as targets are: the pre-pose (rad)."""
        return self.clip_targets(self._default.copy())

    def __call__(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (the raw action clipped to clip_raw, the leg targets), both float32."""
        action = np.asarray(raw, dtype=np.float32)
        if self._clip_raw is not None:
            action = np.clip(action, -self._clip_raw, self._clip_raw)
        return action, self.clip_targets(action * self._scale + self._default)
