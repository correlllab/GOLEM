"""
Collect the per-skill mixin classes SkillsNode inherits from.

Each module defines one mixin whose `_exec_<name>` method runs on the
SkillsNode instance, so it calls SkillsBase's clients and primitives through
`self`.
"""

from .open_door import OpenDoorSkill
from .close_door import CloseDoorSkill
from .open_lid import OpenLidSkill
from .close_lid import CloseLidSkill
from .navigate import NavigateSkill
from .grasp import GraspSkill
from .pick_place import PickPlaceSkill
from .press import PressSkill
from .slide_rack import SlideRackSkill
from .turn import TurnSkill
from .frontier_explore import FrontierExploreSkill

__all__ = [
    'OpenDoorSkill', 'CloseDoorSkill', 'OpenLidSkill', 'CloseLidSkill',
    'NavigateSkill', 'GraspSkill', 'PickPlaceSkill', 'PressSkill',
    'SlideRackSkill', 'TurnSkill', 'FrontierExploreSkill',
]
