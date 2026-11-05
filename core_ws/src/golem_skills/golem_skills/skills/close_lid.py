"""Stub the /skill/close_lid executor: every goal aborts as not implemented."""


class CloseLidSkill:
    """Provide /skill/close_lid's execute callback to SkillsNode."""

    def _exec_close_lid(self, gh):
        """Raise NotImplementedError, which SkillsNode's wrapper turns into an aborted goal."""
        raise NotImplementedError('close_lid skill not implemented')
