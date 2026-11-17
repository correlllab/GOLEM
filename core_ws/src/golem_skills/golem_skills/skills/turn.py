"""Stub the shared /skill/turn_lever and /skill/twist_knob executor: every goal aborts."""


class TurnSkill:
    """Provide the execute callback SkillsNode binds to turn_lever and twist_knob."""

    def _exec_turn(self, gh):
        """Raise NotImplementedError, which SkillsNode's wrapper turns into an aborted goal."""
        raise NotImplementedError('turn skill not implemented')
