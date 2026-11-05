"""Stub the /skill/press executor: every goal aborts as not implemented."""


class PressSkill:
    """Provide /skill/press's execute callback to SkillsNode."""

    def _exec_press(self, gh):
        """Raise NotImplementedError, which SkillsNode's wrapper turns into an aborted goal."""
        raise NotImplementedError('press skill not implemented')
