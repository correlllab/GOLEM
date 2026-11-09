"""
Run a leg policy from its spec: the spec loader, term registry, runtimes and runner.

config/<robot>/policies/<name>.yaml says how the policy's observation is built from
/lowstate and the commands, which network runs it, and how its action becomes leg
targets; nothing here names a robot or a policy. Only the active policy's runtime
library is imported (torch or onnxruntime), when its runner is built.
"""

from golem_lowerbody_rl.policy.commands import CommandProcessor
from golem_lowerbody_rl.policy.observation import PolicyInput, TERMS
from golem_lowerbody_rl.policy.runner import LegTargets, PolicyError, PolicyRunner
from golem_lowerbody_rl.policy.spec import load_policy_spec, PolicySpec

__all__ = [
    'CommandProcessor', 'LegTargets', 'PolicyError', 'PolicyInput', 'PolicyRunner',
    'PolicySpec', 'TERMS', 'load_policy_spec',
]
