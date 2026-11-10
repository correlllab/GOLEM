"""Stub the /skill/open_door executor: every goal aborts as not implemented."""


class OpenDoorSkill:
    """Provide /skill/open_door's execute callback to SkillsNode."""

    def _exec_open_door(self, gh):
        """Raise NotImplementedError, which SkillsNode's wrapper turns into an aborted goal."""
        raise NotImplementedError('open_door skill not implemented')
