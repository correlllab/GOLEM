"""SkillNavigate: planning -> navigating (preset or detected target)."""


class NavigateSkill:
    def _exec_navigate(self, gh):
        """planning -> navigating to the named or detected target."""
        raise NotImplementedError('navigate skill not implemented')
