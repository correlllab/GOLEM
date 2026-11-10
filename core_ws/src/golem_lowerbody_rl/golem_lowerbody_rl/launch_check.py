"""
Refuse a leg node that must not start: the robot, run config and policy it would run.

launch/lowerbody.launch.py applies lowerbody_refusal through lowerbody_check before any
node starts, and the node applies it again at start-up, so a node started without the
launch file is held to the same rules. A bringup that includes lowerbody.launch.py after
other nodes puts its own lowerbody_check first, so a refusal still starts no node.
"""

from launch import LaunchContext
from launch.actions import OpaqueFunction
from launch.condition import Condition
from launch.conditions import evaluate_condition_expression
from launch.some_substitutions_type import SomeSubstitutionsType
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions

from golem_lowerbody_rl.policy import load_policy_spec
from golem_lowerbody_rl.profile import known_robots, load_profile
from golem_lowerbody_rl.run_config import load_run_config

__all__ = ['lowerbody_check', 'lowerbody_refusal']


def lowerbody_refusal(robot: str, config: str, policy: str, sim: bool | None) -> str | None:
    """
    Return why the leg node must not run policy under run config config, or None.

    robot must be one of known_robots(); config is a run config name or path, policy a
    policy name or path ('' for the robot's default_policy). sim (None: not checked) must
    equal the run config's sim. Refused too: a run config that does not load, a real-robot
    run config without the operator's confirm or with the elastic band release, and a
    policy whose spec does not load or that the robot's robot.yaml does not list.
    """
    if robot not in known_robots():
        return f"unknown robot '{robot}': one of {', '.join(known_robots())}"
    try:
        run = load_run_config(robot, config)
    except (OSError, ValueError) as error:
        return str(error)
    if sim is not None and run.sim != sim:
        meant = 'a simulator' if run.sim else 'the real robot'
        return (f"run config '{config}' is for {meant} (sim: {str(run.sim).lower()}), and "
                f'sim is {str(sim).lower()}')
    if not run.sim:
        # SAFETY: on a real robot the policy takes the legs only on the operator's confirm,
        # and no call flips a band the robot does not have
        if not run.startup.wait_for_confirm:
            return (f"run config '{config}' is for the real robot and has "
                    'startup.wait_for_confirm false: the policy would take the legs without '
                    "the operator's /lowerbody/confirm_engage")
        if run.band.release:
            return (f"run config '{config}' is for the real robot and has band.release "
                    'true: a real robot has no elastic band to release')
    profile = load_profile(robot)
    try:
        spec = load_policy_spec(robot, policy or profile.default_policy)
    except (OSError, ValueError) as error:
        return str(error)
    if spec.name not in profile.policies:
        return (f"policy '{spec.name}' ({spec.path}) is not one of robot {robot}'s policies: "
                f"{', '.join(profile.policies)} (config/{robot}/robot.yaml)")
    return None


def lowerbody_check(robot: SomeSubstitutionsType, config: SomeSubstitutionsType,
                    policy: SomeSubstitutionsType, sim: SomeSubstitutionsType,
                    condition: Condition | None = None) -> OpaqueFunction:
    """
    Return a launch action that raises "lowerbody: not starting: <reason>" on a refusal.

    robot, config, policy and sim (true|false) are text or substitutions, read when the
    action runs, and checked by lowerbody_refusal. It starts nothing: put it before the
    first node, so that a refusal starts none. With condition, it checks only where the
    condition holds (where the leg node will start).
    """
    def check(context: LaunchContext) -> list:
        """Raise RuntimeError, starting nothing, when the leg node must not run."""
        def text(value: SomeSubstitutionsType) -> str:
            """Return value's text in context."""
            return perform_substitutions(context, normalize_to_list_of_substitutions(value))
        refusal = lowerbody_refusal(
            text(robot), text(config), text(policy),
            sim=evaluate_condition_expression(context, normalize_to_list_of_substitutions(sim)))
        if refusal is not None:
            raise RuntimeError(f'lowerbody: not starting: {refusal}')
        return []
    return OpaqueFunction(function=check, condition=condition)
