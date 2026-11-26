"""
Hold what the manipulation skills need to know about each robot they drive.

The skills node's `robot` parameter (default 'h12') picks one RobotProfile from
ROBOTS, once at start-up. Everything robot-specific the grasp and pick_place
skills read comes from it; the rest of the package is the same for every robot.

This module imports no ROS package.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# the arms a goal may name, and the keys of every per-arm table below
ARMS = ('left', 'right')


@dataclass(frozen=True)
class ToolFrame:
    """A frame the skills define, fixed on a frame the robot's model has: a pose in it."""

    # frame it is fixed to
    parent: str
    # m, its origin in `parent`
    xyz: tuple
    # its rotation in `parent`, a 3x3 matrix by rows (columns: its x, y, z axes)
    rot: tuple

    def matrix(self):
        """Return its 4x4 pose in `parent`."""
        T = np.eye(4)
        T[:3, :3] = np.asarray(self.rot, dtype=float)
        T[:3, 3] = np.asarray(self.xyz, dtype=float)
        return T


@dataclass(frozen=True)
class RobotProfile:
    """Name what the grasp and pick_place skills drive on one robot."""

    # golem_upper_ik's robot name, also the node's `robot` parameter value
    name: str
    # arm -> frame the grasp poses are for: the GraspGenX gripper-base
    # convention (+Z the approach, +X the finger-closing axis, the finger
    # contact point grasp_synthesis.common.GRIPPER_BASE_TO_CONTACT_M along +Z)
    grasp_frames: dict
    # arm -> pinch-point frame, at the finger contact point: the arm's grasp
    # frame in golem_upper_ik's config/<robot>/robot.yaml (grasp_frames), the
    # frame /grasp_frame_target drives. pick_place's lift drives it.
    pinch_frames: dict
    # arm -> pinch-frame pose (x, y, z m, qx, qy, qz, qw; pelvis frame) the
    # arms go to before the head camera looks: out of its view
    prep_posture: dict
    # arm a staged synthesis method grasps with -> the posture it starts from
    # (arm -> pinch-frame pose, as prep_posture)
    staging_postures: dict
    # gripper_name the graspgen service plans for; None: no GraspGenX model
    graspgen_gripper: Optional[str]
    # the hand the grasp closes, one of hands.HANDS
    hand: str
    # whether each hand carries a camera (yolo_server's hand-camera bundles),
    # which goal.visual_servo needs
    hand_cameras: bool
    # frame name -> ToolFrame, for grasp frames neither the robot's URDF nor
    # golem_upper_ik has: SkillsBase broadcasts them as static TF
    tool_frames: dict = field(default_factory=dict)


# The postures are the pinch frames' poses at golem_upper_ik's named
# configurations of the same name (its config/<robot>/robot.yaml), by forward
# kinematics on golem_upper_ik's model of the robot with the torso or waist at
# its hold (0 rad), rounded to 0.1 mm and 1e-4.
# H1-2 'prep': init_3 with the elbows bent, the grasp frames level and raised.
H12_PREP = {
    'left': (0.2498, 0.2810, 0.1802, 0.1638, 0.0037, -0.0282, 0.9861),
    'right': (0.2498, -0.2810, 0.1802, -0.1638, 0.0037, 0.0282, 0.9861),
}

H12 = RobotProfile(
    name='h12',
    # URDF frames of the Magpie grippers
    grasp_frames={'left': 'left_graspgenx_frame', 'right': 'right_graspgenx_frame'},
    pinch_frames={'left': 'left_grasp_frame', 'right': 'right_grasp_frame'},
    prep_posture=H12_PREP,
    # 'staging_left' and 'staging_right': the grasping arm above the workspace
    # for a steep top-down approach, the other arm at 'prep'. The IK reaches
    # them with the wrist yaw at its limit, where golem_upper_ik's goal ends
    # pinned at the limit about 1-3 mm short.
    staging_postures={
        'left': {'left': (0.3030, 0.2030, 0.0557, -0.0017, 0.6429, -0.0032, 0.7660),
                 'right': H12_PREP['right']},
        'right': {'left': H12_PREP['left'],
                  'right': (0.3030, -0.2030, 0.0557, 0.0017, 0.6429, 0.0032, 0.7660)},
    },
    graspgen_gripper='magpie',
    hand='magpie',
    hand_cameras=True,
)

# m from a grasp frame's origin to its finger contact point along its +Z: the
# GraspGenX convention's (grasp_synthesis.common.GRIPPER_BASE_TO_CONTACT_M, the
# Magpie's), which every synthesis method plans with. This module imports no ROS
# package, so the value is repeated here; test_robots checks the two agree.
CONVENTION_CONTACT_M = 0.1146


def _convention_frame(arm):
    """
    Return `arm`'s GraspGenX-convention frame on a pinch frame with the convention's axes.

    The G1's pinch frames (golem_upper_ik's grasp frames on the Dex3-1 palms)
    already have the convention's axes: +Z the palm face's normal (the
    approach), +X the closing direction. The convention frame sits
    CONVENTION_CONTACT_M behind the pinch point along +Z.
    """
    return ToolFrame(f'{arm}_grasp_frame', (0.0, 0.0, -CONVENTION_CONTACT_M),
                     ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


# G1 'home': forearms forward, the pinch points about 0.32 m in front of the
# pelvis at waist height; its start-up posture.
G1_HOME = {
    'left': (0.3213, 0.1116, 0.0952, 0.7071, 0.0, -0.0001, 0.7071),
    'right': (0.3213, -0.1116, 0.0952, 0.7071, 0.0, -0.0001, -0.7071),
}

G1 = RobotProfile(
    name='g1',
    # the GraspGenX-convention frames on the pinch frames: neither the G1's
    # URDF nor golem_upper_ik has them
    grasp_frames={arm: f'{arm}_dex3_grasp_frame' for arm in ARMS},
    # golem_upper_ik's grasp frames on the Dex3-1 palms, which the G1's URDF
    # lacks: SkillsBase broadcasts them from golem_upper_ik's profile
    pinch_frames={arm: f'{arm}_grasp_frame' for arm in ARMS},
    # the G1 has no 'prep' or staging posture; 'home' holds the hands in
    # front of the pelvis
    prep_posture=G1_HOME,
    staging_postures={arm: G1_HOME for arm in ARMS},
    # GraspGenX has no Dex3-1 model, and the G1 no hand cameras
    graspgen_gripper=None,
    hand='dex3',
    hand_cameras=False,
    tool_frames={f'{arm}_dex3_grasp_frame': _convention_frame(arm) for arm in ARMS},
)

ROBOTS = {p.name: p for p in (H12, G1)}

DEFAULT_ROBOT = 'h12'


def robot_profile(name):
    """Return the RobotProfile named `name`; raise ValueError for an unknown robot."""
    key = str(name).strip().lower()
    if key not in ROBOTS:
        raise ValueError(f'unknown robot {name!r}; golem_skills drives {sorted(ROBOTS)}')
    return ROBOTS[key]


def arm_of(profile, frame):
    """Return the arm whose grasp or pinch frame `frame` is, or None."""
    for arm in ARMS:
        if frame in (profile.grasp_frames[arm], profile.pinch_frames[arm]):
            return arm
    return None
