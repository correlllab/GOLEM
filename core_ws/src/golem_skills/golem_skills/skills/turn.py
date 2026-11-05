"""Stub the shared /skill/turn_lever and /skill/twist_knob executor: every goal aborts."""


class TurnSkill:
    """Provide the execute callback SkillsNode binds to turn_lever and twist_knob."""

    def _exec_turn(self, action_type, label, motion_phase, gh):
        """
        Raise NotImplementedError, which SkillsNode's wrapper turns into an aborted goal.

        SkillsNode binds `action_type`, `label` and `motion_phase` per skill with
        functools.partial, so the action server calls this with the goal handle only.
        """
        raise NotImplementedError('turn skill not implemented')
