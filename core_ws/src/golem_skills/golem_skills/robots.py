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
    """A frame the skills define on a URDF link the robot's model has: a fixed pose in it."""

    # URDF frame it is fixed to, the one frame_task drives for it
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
    # whether each hand carries a camera (yolo_server's hand-camera bundles),
    # which goal.visual_servo needs
    hand_cameras: bool
    # frame name -> ToolFrame, for grasp and pinch frames the robot's URDF does
    # not have: SkillsBase broadcasts them as static TF and drives frame_task
    # through their parent (drive_pose)
    tool_frames: dict = field(default_factory=dict)


H12 = RobotProfile(
    name='h12',
    # URDF frames of the Magpie grippers
    grasp_frames={'left': 'left_graspgenx_frame', 'right': 'right_graspgenx_frame'},
    pinch_frames={'left': 'left_grasp_frame', 'right': 'right_grasp_frame'},
    prep_config='prep',
    staging_configs={'left': 'staging_left', 'right': 'staging_right'},
    graspgen_gripper='magpie',
    hand='magpie',
    hand_cameras=True,
)

# m from a grasp frame's origin to its finger contact point along its +Z: the
# GraspGenX convention's (grasp_synthesis.common.GRIPPER_BASE_TO_CONTACT_M, the
# Magpie's), which every synthesis method plans with. This module imports no ROS
# package, so the value is repeated here; test_robots checks the two agree.
CONVENTION_CONTACT_M = 0.1146

# The G1's Dex3-1 grasp, in <arm>_hand_palm_link (from the URDF joint origins, not
# measured): the index and middle fingers stand along the palm's +X from
# x = 0.078 m, 57 mm apart in z, and curl toward the palm's face (-Y on the left
# hand, +Y on the right); the thumb stands out of that face at x = 0.023 m and
# swings toward +X. The grasp approaches along the face's normal and closes
# along the palm's +X, with the finger contact point DEX3_CONTACT_M in front of
# the face, between thumb and fingers. Unmeasured on the real G1 and its sim.
DEX3_CONTACT_M = (0.08, 0.04, 0.0)


def _dex3_frames(arm):
    """Return {grasp frame: ToolFrame, pinch frame: ToolFrame} of `arm`'s Dex3-1."""
    side = -1.0 if arm == 'left' else 1.0
    # columns x (closing: palm +X), y (z cross x), z (approach: the palm face's normal)
    z = np.array([0.0, side, 0.0])
    x = np.array([1.0, 0.0, 0.0])
    R = np.column_stack([x, np.cross(z, x), z])
    contact = np.array([DEX3_CONTACT_M[0], side * DEX3_CONTACT_M[1], DEX3_CONTACT_M[2]])
    rot = tuple(tuple(float(v) for v in row) for row in R)
    parent = f'{arm}_hand_palm_link'
    return {
        f'{arm}_dex3_grasp_frame': ToolFrame(
            parent, tuple(float(v) for v in contact - CONVENTION_CONTACT_M * z), rot),
        f'{arm}_dex3_pinch_frame': ToolFrame(parent, tuple(float(v) for v in contact), rot),
    }


G1 = RobotProfile(
    name='g1',
    # tool frames on the Dex3-1 palms (DEX3_CONTACT_M): the G1's URDF has no grasp frame
    grasp_frames={arm: f'{arm}_dex3_grasp_frame' for arm in ARMS},
    pinch_frames={arm: f'{arm}_dex3_pinch_frame' for arm in ARMS},
    # golem_upper_ik's G1 has no 'prep' or staging posture; 'home' holds the
    # forearms forward, the palms about 0.24 m in front of the pelvis
    prep_config='home',
    staging_configs={arm: 'home' for arm in ARMS},
    # GraspGenX has no Dex3-1 model, and the G1 no hand cameras
    graspgen_gripper=None,
    hand='dex3',
    hand_cameras=False,
    tool_frames={**_dex3_frames('left'), **_dex3_frames('right')},
)

ROBOTS = {p.name: p for p in (H12, G1)}

DEFAULT_ROBOT = 'h12'


def robot_profile(name):
    """Return the RobotProfile named `name`; raise ValueError for an unknown robot."""
    key = str(name).strip().lower()
    if key not in ROBOTS:
        raise ValueError(f'unknown robot {name!r}; golem_skills drives {sorted(ROBOTS)}')
    return ROBOTS[key]


def drive_pose(profile, frame, T):
    """
    Return (URDF frame, its 4x4 pose) that puts `frame` at 4x4 pose `T`.

    A tool frame (profile.tool_frames) is driven through its parent; any other
    frame is a URDF frame, returned as given.
    """
    tool = profile.tool_frames.get(frame)
    if tool is None:
        return frame, T
    return tool.parent, T @ np.linalg.inv(tool.matrix())
