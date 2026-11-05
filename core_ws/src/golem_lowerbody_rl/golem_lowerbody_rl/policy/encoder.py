"""
Rebuild FAME's environment-factor encoder (Rapid Motor Adaptation) for inference.

RMA (Kumar et al., RSS 2021) trains an encoder that maps privileged environment factors
e_t to a compact latent z_t, and conditions the policy on (observation, z_t). The robot
runs the encoder on the e_t it can observe (FAME: upper-body joint positions and hand
forces). The weights are a state_dict of EnvFactorEncoder, so the module is rebuilt here
and loaded into; the observation's latent term (policy/observation.py) does that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int,
               activation: str = "elu") -> nn.Sequential:
    """Return Linear layers of the given widths with activation between them."""
    if activation == "elu":
        act: type[nn.Module] = nn.ELU
    elif activation == "relu":
        act = nn.ReLU
    elif activation == "tanh":
        act = nn.Tanh
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers: list[nn.Module] = []
    last_dim = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(act())
        last_dim = h
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


@dataclass
class EnvFactorEncoderCfg:
    """Hold the encoder's layer sizes and activation."""

    in_dim: int = 21  # e_t: 15 upper-body DoF + 3 left hand + 3 right hand (no torso)
    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "elu"


class EnvFactorEncoder(nn.Module):
    """Encode environment factors e_t into a compact latent z_t."""

    def __init__(self, cfg: EnvFactorEncoderCfg | None = None):
        """Build the encoder's layers from cfg (the defaults are FAME's)."""
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvFactorEncoderCfg()
        self.net = _build_mlp(
            in_dim=self.cfg.in_dim,
            hidden_dims=list(self.cfg.hidden_dims),
            out_dim=self.cfg.latent_dim,
            activation=self.cfg.activation,
        )

    def forward(self, env_factors: torch.Tensor) -> torch.Tensor:
        """
        Encode env_factors (N, in_dim) into z (N, latent_dim).

        Raises ValueError for any other shape.
        """
        if env_factors.ndim != 2 or env_factors.shape[-1] != self.cfg.in_dim:
            raise ValueError(
                f"env_factors must be (N, {self.cfg.in_dim}); got {tuple(env_factors.shape)}"
            )
        return self.net(env_factors)
