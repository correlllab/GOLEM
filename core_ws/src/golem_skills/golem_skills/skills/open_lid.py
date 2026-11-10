"""Stub the /skill/open_lid executor: every goal aborts as not implemented."""


class OpenLidSkill:
    """Provide /skill/open_lid's execute callback to SkillsNode."""

    def _exec_open_lid(self, gh):
        """Raise NotImplementedError, which SkillsNode's wrapper turns into an aborted goal."""
        raise NotImplementedError('open_lid skill not implemented')
