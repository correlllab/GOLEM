"""Stub the /skill/navigate executor: every goal aborts as not implemented."""


class NavigateSkill:
    """Provide /skill/navigate's execute callback to SkillsNode."""

    def _exec_navigate(self, gh):
        """Raise NotImplementedError, which SkillsNode's wrapper turns into an aborted goal."""
        raise NotImplementedError('navigate skill not implemented')
