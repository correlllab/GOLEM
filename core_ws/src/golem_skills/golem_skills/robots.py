"""
Hold what the manipulation skills need to know about each robot they drive.

The skills node's `robot` parameter (default 'h12') picks one RobotProfile from
ROBOTS, once at start-up. Everything robot-specific the grasp and pick_place
skills read comes from it; the rest of the package is the same for every robot.

This module imports no ROS package.
"""

from dataclasses import dataclass
from typing import Optional


# the arms a goal may name, and the keys of every per-arm table below
ARMS = ('left', 'right')


@dataclass(frozen=True)
class RobotProfile:
    """Name what the grasp and pick_place skills drive on one robot."""

    # golem_upper_ik's robot name, also the node's `robot` parameter value
    name: str
    # arm -> frame the grasp poses are for: the GraspGenX gripper-base
    # convention (+Z the approach, +X the finger-closing axis, the finger
    # contact point grasp_synthesis.common.GRIPPER_BASE_TO_CONTACT_M along +Z)
    grasp_frames: dict
    # arm -> pinch-point frame, at the finger contact point; pick_place's lift
    # drives it
    pinch_frames: dict
    # golem_upper_ik named configuration the arms go to before the head camera
    # looks: out of its view
    prep_config: str
    # arm -> named configuration a staged synthesis method starts from
    staging_configs: dict
    # gripper_name the graspgen service plans for; None: no GraspGenX model
    graspgen_gripper: Optional[str]
    # the hand the grasp closes, one of hands.HANDS
    hand: str


H12 = RobotProfile(
    name='h12',
    # URDF frames of the Magpie grippers
    grasp_frames={'left': 'left_graspgenx_frame', 'right': 'right_graspgenx_frame'},
    pinch_frames={'left': 'left_grasp_frame', 'right': 'right_grasp_frame'},
    prep_config='prep',
    staging_configs={'left': 'staging_left', 'right': 'staging_right'},
    graspgen_gripper='magpie',
    hand='magpie',
)

ROBOTS = {p.name: p for p in (H12,)}

DEFAULT_ROBOT = 'h12'


def robot_profile(name):
    """Return the RobotProfile named `name`; raise ValueError for an unknown robot."""
    key = str(name).strip().lower()
    if key not in ROBOTS:
        raise ValueError(f'unknown robot {name!r}; golem_skills drives {sorted(ROBOTS)}')
    return ROBOTS[key]
