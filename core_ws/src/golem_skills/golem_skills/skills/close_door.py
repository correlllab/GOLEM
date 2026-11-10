"""Stub the /skill/close_door executor: every goal aborts as not implemented."""


class CloseDoorSkill:
    """Provide /skill/close_door's execute callback to SkillsNode."""

    def _exec_close_door(self, gh):
        """Raise NotImplementedError, which SkillsNode's wrapper turns into an aborted goal."""
        raise NotImplementedError('close_door skill not implemented')
