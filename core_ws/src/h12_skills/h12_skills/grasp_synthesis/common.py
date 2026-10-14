"""Shared types and grasp-pose geometry for the grasp-synthesis methods.

Every method emits grasps in the GraspGenX gripper-BASE convention, expressed
in the pelvis frame (+X forward, +Y left, +Z up): +Z of the grasp is the
approach into the object, +X the finger-closing axis, and the URDF's
{left,right}_graspgenx_frame matches it, so the grasp skill executes any
method's output by driving that frame to the pose as given.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from geometry_msgs.msg import Pose, PoseStamped

from ..perception_utils import matrix_to_pose, pose_to_matrix


# Gripper base -> finger contact point distance along the grasp's +Z [m]
# (magpie fingertip).
GRIPPER_BASE_TO_CONTACT_M = 0.1146

PLANNING_FRAME = 'pelvis'


@dataclass
class SynthesisInput:
    """What a synthesis method may consume. Every method sees the same
    perception output, so methods differ only in how they turn it into
    grasps."""
    obj: str                     # object name (for log lines and reasons)
    arm: str                     # 'left' | 'right', already resolved
    cloud: np.ndarray            # (N, 3) segmented object cloud, pelvis [m]
    scene: Optional[np.ndarray]  # (M, 3) whole-frame cloud, pelvis; may be None
    min_pitch_deg: float         # goal.min_downward_pitch_deg; 0 = no gate
    plan_grasp: Callable         # SkillsBase.plan_grasp (GraspGenX service)
    log: object                  # rclpy logger

    @property
    def centroid(self):
        return self.cloud.mean(axis=0)


@dataclass
class Candidates:
    """Grasp candidates in EXECUTION order: the grasp skill tries them
    first-to-last and commits to the first whose pre-grasp it reaches. Each
    method owns its ordering."""
    grasps: list            # PoseStamped per candidate, PLANNING_FRAME
    scores: list            # method confidence per candidate (1.0 if none)
    labels: list            # short human-readable tag per candidate
    tiers: list             # ranking.priority_tier per candidate (informational)
    total: int              # size of the pool these were picked from
    gripper_width: float    # planned opening [m]; 0.0 when the method has none


def make_candidates(poses, labels, arm, scores=None, total=None,
                    gripper_width=0.0):
    """Wrap pelvis-frame Poses (already in execution order) as Candidates."""
    from .ranking import priority_tier
    grasps = []
    for p in poses:
        g = PoseStamped()
        g.header.frame_id = PLANNING_FRAME
        g.pose = p
        grasps.append(g)
    return Candidates(
        grasps=grasps,
        scores=list(scores) if scores is not None else [1.0] * len(poses),
        labels=list(labels),
        tiers=[priority_tier(p, arm) for p in poses],
        total=len(poses) if total is None else total,
        gripper_width=float(gripper_width))


# ================================ geometry ===================================
def pose_at(point):
    """Identity-orientation Pose at `point` (x, y, z array-like)."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        float(point[0]), float(point[1]), float(point[2]))
    pose.orientation.w = 1.0
    return pose


def grasp_pose(contact, x_axis, z_axis):
    """Gripper-base Pose whose finger CONTACT point lands on `contact`, with
    closing axis `x_axis` and approach `z_axis` (both pelvis frame, x_axis
    orthogonal to z_axis). +Y completes the right-handed frame."""
    z = np.asarray(z_axis, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_axis, dtype=float)
    x = x / np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, np.cross(z, x), z
    T[:3, 3] = np.asarray(contact, dtype=float) - GRIPPER_BASE_TO_CONTACT_M * z
    return matrix_to_pose(T)


def approach_from_az_pitch(az_deg, pitch_deg):
    """Unit approach axis (pelvis frame) at azimuth `az_deg` in the XY plane
    (0 = +X forward, positive toward +Y) and `pitch_deg` below horizontal."""
    az, pit = math.radians(az_deg), math.radians(pitch_deg)
    return np.array([math.cos(pit) * math.cos(az),
                     math.cos(pit) * math.sin(az),
                     -math.sin(pit)])


def approach_pitch_down(pose):
    """Pitch of `pose`'s approach (+Z) below horizontal [rad]."""
    return math.asin(float(np.clip(-pose_to_matrix(pose)[2, 2], -1.0, 1.0)))


def level_pose(contact, approach_dir):
    """Grasp approaching along `approach_dir` with a LEVEL wrist: +X =
    approach x up (horizontal), +Y = approach x +X with its sign chosen so
    +Y points up. A vertical approach takes +X = pelvis +Y."""
    a = np.asarray(approach_dir, dtype=float)
    a = a / np.linalg.norm(a)
    x = np.cross(a, np.array([0.0, 0.0, 1.0]))
    nx = float(np.linalg.norm(x))
    x = np.array([0.0, 1.0, 0.0]) if nx < 1e-6 else x / nx
    if float(np.cross(a, x)[2]) < 0.0:
        x = -x
    return grasp_pose(contact, x, a)


def roll_to_yup(pose):
    """Return `pose` rolled about its OWN approach axis (+Z) so gripper +Y
    points as close to pelvis +Z (up) as the approach allows — a level wrist.
    Position and approach are preserved; only the finger-closing direction
    changes, so this is a genuine regrasp (it swings the pinch horizontal)."""
    T = pose_to_matrix(pose)
    R = T[:3, :3]
    x_up, y_up = float(R[2, 0]), float(R[2, 1])
    # Y'.up = cos(th)*y_up - sin(th)*x_up, maximised at atan2(-x_up, y_up).
    theta = math.atan2(-x_up, y_up)
    c, s = math.cos(theta), math.sin(theta)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]])
    T[:3, :3] = R @ Rz
    return matrix_to_pose(T)


def tilted_pose(contact, yaw, tilt):
    """Grasp `tilt` rad off straight-down, leaning outward along the
    horizontal robot->object direction (from the pelvis origin), so the
    gripper comes in from above-and-behind. Fingers close along the
    pelvis-XY heading `yaw`, projected perpendicular to the approach."""
    tip = np.asarray(contact, dtype=float)
    u = np.array([tip[0], tip[1], 0.0])
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    z = np.array([math.sin(tilt) * u[0], math.sin(tilt) * u[1], -math.cos(tilt)])
    z /= np.linalg.norm(z)
    c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    x = c - float(c @ z) * z
    if float(np.linalg.norm(x)) < 1e-6:          # yaw parallel to the approach
        c = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        x = c - float(c @ z) * z
    return grasp_pose(tip, x, z)


# ======================== top-surface cloud analysis =========================
# Tilts off straight-down tried, in order, by the top-surface methods
# (centroid, antipodal). The H1-2 wrist pitch is clamped to +/-0.4625 rad
# (+/-26.5 deg): pre-grasp convergence measured on the arm was 20 deg -> 3 mm,
# 30 deg -> 17-44 mm, straight down -> 41 mm, 45 deg -> 128 mm.
SURFACE_GRASP_TILTS = ((math.pi / 9, '20deg'), (math.pi / 6, '30deg'),
                       (0.0, 'topdown'), (math.pi / 4, '45deg'))
# Fingertip depth below the object's top surface [m].
FINGER_SINK_M = 0.02
# Thickness of the top slab analysed by top_layer_pca [m].
TOP_LAYER_M = 0.025


def robust_top_z(cloud):
    """Height of the object's top surface. The 75th percentile, not the max:
    mask-edge pixels pick up background depth and back-project into stray
    points well above the object (2.4 cm above a cheese wedge's real top)."""
    return float(np.percentile(cloud[:, 2], 75))


def surface_contact(xy, top_z, floor_z):
    """Fingertip target over `xy`: FINGER_SINK_M below `top_z`, but never
    below `floor_z` (the object's bottom)."""
    return np.array([xy[0], xy[1], max(top_z - FINGER_SINK_M, floor_z)])


def top_layer_pca(cloud):
    """PCA of the top TOP_LAYER_M slab of `cloud` in the XY plane. Returns
    a dict: slab `centroid`, `top_z`, `major_yaw` / `minor_yaw` (closing
    headings across each principal axis, minor = the narrowest span) and
    `major_extent` / `minor_extent` [m]."""
    top_z = robust_top_z(cloud)
    top = cloud[cloud[:, 2] > top_z - TOP_LAYER_M]
    if len(top) < 3:
        top = cloud
    xy = top[:, :2]
    mu = xy.mean(axis=0)
    cov = np.cov((xy - mu).T) if len(xy) > 2 else np.eye(2)
    vals, vecs = np.linalg.eigh(cov)
    vecs = vecs[:, np.argsort(vals)[::-1]]          # columns: major, minor
    local = (xy - mu) @ vecs
    extents = local.max(axis=0) - local.min(axis=0)
    major_yaw = float(math.atan2(vecs[1, 0], vecs[0, 0]))
    return dict(
        centroid=np.array([mu[0], mu[1], float(top[:, 2].mean())]),
        top_z=top_z,
        major_yaw=major_yaw,
        minor_yaw=major_yaw + math.pi / 2.0,
        major_extent=float(extents[0]),
        minor_extent=float(extents[1]),
    )
