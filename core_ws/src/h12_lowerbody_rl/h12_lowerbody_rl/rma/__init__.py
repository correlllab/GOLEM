"""
Minimal RMA modules kept for FAME *inference*.

Rapid Motor Adaptation (Kumar et al., RSS 2021) trains an env-factor encoder
that maps privileged factors ``e_t`` to a compact latent ``z_t``; the policy is
conditioned on ``(obs, z_t)``. At deploy time we run the encoder directly on the
``e_t`` we can observe (upper-body joint positions + wrist forces).

Only the encoder is needed to run the policy; FamePolicy (policy.py) builds
the 21-dim ``e_t`` it encodes.
"""

from .env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg

__all__ = [
    "EnvFactorEncoder",
    "EnvFactorEncoderCfg",
]
