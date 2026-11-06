"""
Say when the H1-2's leg block starts its leg node, and check that node before any starts.

h12_lowerbody.launch.py starts golem_lowerbody_rl's leg node under leg_node_condition. The
top-level files include that block after the others, so each puts leg_check, with the
arguments it passes the block, before its first include: a refused policy or run config
then starts no node, as lowerbody.launch.py's own check does when it runs alone.
"""

from golem_lowerbody_rl.launch_check import lowerbody_check
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
from launch.some_substitutions_type import SomeSubstitutionsType
from launch.substitutions import AndSubstitution, PythonExpression

__all__ = ['leg_check', 'leg_node_condition']

# the policy:= that starts no leg node: no bringup starts the MJPC controller
MJPC = 'mjpc'


def leg_node_condition(start_position_verified: SomeSubstitutionsType,
                       policy: SomeSubstitutionsType) -> IfCondition:
    """
    Return the condition under which the leg block starts the leg node.

    SAFETY: only once the operator has verified the start pose (start_position_verified),
    and not for policy:=mjpc.
    """
    return IfCondition(AndSubstitution(
        start_position_verified, PythonExpression(["'", policy, f"' != '{MJPC}'"])))


def leg_check(legs: dict) -> OpaqueFunction:
    """
    Return the leg node's check for the arguments legs passes h12_lowerbody.launch.py.

    legs holds start_position_verified, policy, sim and lowerbody_config. The check runs
    where the block starts the leg node, and raises "lowerbody: not starting: ..." on a
    refusal.
    """
    return lowerbody_check(
        'h12', legs['lowerbody_config'], legs['policy'], legs['sim'],
        condition=leg_node_condition(legs['start_position_verified'], legs['policy']))
