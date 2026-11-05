"""Pinned standing-only h12_mjlab adapters; no ROS, DDS, or training dependency."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .policy import (
    AlmiPolicy, FamePolicy, LegCommand, RobotState, _clip_leg_positions,
    _load_yaml, quat_mul,
)


class _StandingContract:
    control_hz = 50.0

    def _validate_config(self, config_path):
        cfg = _load_yaml(config_path)
        root = Path(config_path).resolve().parent
        for key in ("policy_path", "encoder_path"):
            if key not in cfg:
                continue
            path = root / cfg[key]
            expected = cfg["artifact_sha256"][key]
            if sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"{self.name}: artifact SHA256 mismatch: {path}")
        for key, size in (("kps", 12), ("kds", 12),
                          ("legs_motor_pos_lower_limit_list", 12),
                          ("legs_motor_pos_upper_limit_list", 12),
                          ("default_angles", 12 if self.name == "fame_finetune" else 21)):
            value = np.asarray(cfg[key], dtype=np.float32)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} for {self.name}")
        lower = np.asarray(cfg["legs_motor_pos_lower_limit_list"])
        upper = np.asarray(cfg["legs_motor_pos_upper_limit_list"])
        nominal = np.asarray(cfg["default_angles"][:12])
        if (np.any(lower >= upper) or np.any(nominal < lower)
                or np.any(nominal > upper)
                or np.any(np.asarray(cfg["kps"]) <= 0)
                or np.any(np.asarray(cfg["kds"]) < 0)):
            raise ValueError("Invalid standing gains or position limits")
        for key, expected in (("ang_vel_scale", .25), ("dof_pos_scale", 1.),
                              ("dof_vel_scale", .05), ("action_scale", .25),
                              ("cmd_scale", [2., 2., .25]), ("num_actions", 12),
                              ("num_obs", 252 if self.name == "fame_finetune" else 65)):
            if not np.array_equal(cfg[key], expected):
                raise ValueError(f"Unsupported standing observation contract: {key}")
        if self.name == "fame_finetune":
            if cfg["obs_history_len"] != 3 or cfg["default_angles_arms"] != [0.] * 15:
                raise ValueError("FAME requires three frames and zero upper defaults")
            if any(cfg.get(key, [0.] * 3) != [0.] * 3
                   for key in ("left_hand_force", "right_hand_force")):
                raise ValueError("FAME standing contract requires zero force factors")
        return cfg

    @staticmethod
    def validate_state(state: RobotState):
        for name, shape in (("q", (27,)), ("dq", (27,)), ("quat", (4,)),
                            ("gyro", (3,)), ("cmd", (3,))):
            value = np.asarray(getattr(state, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid standing state: {name}")
        if (not np.isfinite(state.t) or not np.isfinite(state.height_cmd)
                or np.linalg.norm(state.quat) < 1.e-6):
            raise ValueError("Invalid standing time, height, or quaternion")

    def _pelvis_state(self, state):
        self.validate_state(state)
        yaw = state.q[12]
        c, s = np.cos(yaw), np.sin(yaw)
        # LowState IMU is torso-mounted. Remove waist yaw and its angular rate.
        gyro = np.array([c * state.gyro[0] - s * state.gyro[1],
                         s * state.gyro[0] + c * state.gyro[1],
                         state.gyro[2] - state.dq[12]], dtype=np.float32)
        inv_waist = np.array([np.cos(yaw / 2), 0., 0., -np.sin(yaw / 2)],
                             dtype=np.float32)
        quat = quat_mul(state.quat / np.linalg.norm(state.quat), inv_waist)
        return replace(state, quat=quat, gyro=gyro,
                       cmd=np.zeros(3, dtype=np.float32), height_cmd=1.)

    def _checked_command(self, command):
        for value in (self._action, command.target_q, command.kp, command.kd):
            if value.shape != (12,) or not np.isfinite(value).all():
                raise ValueError(f"{self.name}: invalid policy output")
        return command


class FameFinetunePolicy(_StandingContract, FamePolicy):
    """FAME step 4599: right-padded proprio history, zero-padded latent history."""

    name = "fame_finetune"

    def __init__(self, config_path):
        self._validate_config(config_path)
        super().__init__(config_path)
        if not self.has_encoder:
            raise ValueError("FAME finetune requires its encoder")
        self._policy.cpu()
        with torch.no_grad():
            output = self._policy(torch.zeros(1, 252))
        if output.shape != (1, 12) or not torch.isfinite(output).all():
            raise ValueError("FAME actor must map 252 observations to 12 finite actions")

    def reset(self, state):
        self.validate_state(state)
        self._action.fill(0.)
        self._obs_history.clear()
        self._z_history.fill(0.)

    def compute(self, state):
        state = self._pelvis_state(state)
        self._obs_history.append(self._single_obs(state))
        proprio = np.zeros((3, 76), dtype=np.float32)
        proprio[:len(self._obs_history)] = np.asarray(self._obs_history)
        self._z_history[:-1] = self._z_history[1:].copy()
        self._z_history[-1] = self._encode(state)
        actor_obs = np.concatenate((proprio.ravel(), self._z_history.ravel()))
        with torch.no_grad():
            self._action = self._policy(torch.from_numpy(actor_obs).unsqueeze(0)).numpy()[0]
        target = _clip_leg_positions(self.nominal_lower + self._action * .25,
                                     self._lower_limit, self._upper_limit)
        return self._checked_command(LegCommand(target, self._kps, self._kds))


class _ManipStandActor(nn.Module):
    """Deterministic actor portion of the upstream RSL-RL checkpoint."""

    def __init__(self, path):
        super().__init__()
        self.rnn = nn.Module()
        self.rnn.rnn = nn.LSTM(65, 64, 1)
        self.mlp = nn.Sequential(nn.Linear(64, 64), nn.ELU(),
                                 nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 12))
        state = torch.load(path, map_location="cpu", weights_only=True)
        actor = state["actor_state_dict"]
        allowed = set(self.state_dict()) | {"distribution.std_param"}
        if set(actor) != allowed:
            raise ValueError("Unexpected ALMI standing actor keys or normalization")
        self.load_state_dict({k: v for k, v in actor.items()
                              if k != "distribution.std_param"}, strict=True)
        if any(not torch.isfinite(v).all() for v in self.state_dict().values()):
            raise ValueError("Nonfinite ALMI standing actor weights")
        self.reset_memory()

    def reset_memory(self):
        self._memory = None

    def forward(self, observation):
        latent, self._memory = self.rnn.rnn(observation.unsqueeze(0), self._memory)
        return self.mlp(latent.squeeze(0))


class AlmiManipStandPolicy(_StandingContract, AlmiPolicy):
    """ALMI manipulation standing only; velocity commands and gait phase are zero."""

    name = "almi_manip_stand"

    def __init__(self, config_path):
        self._validate_config(config_path)
        super().__init__(config_path)

    def _load_actor(self, policy_path):
        return _ManipStandActor(policy_path)

    def reset(self, state):
        super().reset(self._pelvis_state(state))

    def compute(self, state):
        return self._checked_command(super().compute(self._pelvis_state(state)))
