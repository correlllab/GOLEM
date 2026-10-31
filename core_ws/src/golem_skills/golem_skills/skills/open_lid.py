"""SkillOpenLid: detect -> grasp (composed) -> open along a hinge arc."""


class OpenLidSkill:
    def _exec_open_lid(self, gh):
        """detect -> grasp (composed via SkillGrasp) -> open along an arc about
        the lid's hinge."""
        raise NotImplementedError('open_lid skill not implemented')
