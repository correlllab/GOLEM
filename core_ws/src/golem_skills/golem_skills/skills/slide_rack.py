"""Stub the /skill/slide_rack executor: every goal aborts as not implemented."""


class SlideRackSkill:
    """Provide /skill/slide_rack's execute callback to SkillsNode."""

    def _exec_slide_rack(self, gh):
        """Raise NotImplementedError, which SkillsNode's wrapper turns into an aborted goal."""
        raise NotImplementedError('slide_rack skill not implemented')
