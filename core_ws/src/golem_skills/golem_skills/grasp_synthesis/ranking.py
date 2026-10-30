"""Candidate gating, priority tiers and diversity selection shared by the
synthesis methods. Pure functions over pelvis-frame Poses."""

import math

import numpy as np

from ..perception_utils import pose_to_matrix


# Priority tiers (priority_tier): methods that rank by tier try tier-by-tier,
# their own score breaking ties within a tier. Tiers 1-3 want the approach
# pitched 20-45 deg below horizontal (steeper than ~45 deg is beyond the H1
# arm's practical reach, flatter rakes the support surface); 4-6 drop the
# pitch requirement; 7 = out-of-fan azimuth (cross-body / behind).
TIER_PITCH_MIN_DEG = 20.0
TIER_PITCH_MAX_DEG = 45.0
# "Forward" tiers (1/4): approach azimuth tolerance off pelvis +X, and the
# finger-closing +X axis tolerance off pelvis +Y.
TIER1_AZ_TOL_DEG = 15.0
TIER1_X_ALIGN_TOL_DEG = 15.0
# "Diagonal" tiers (2/5): azimuth band toward the robot's midline.
TIER2_AZ_CENTER_DEG = 45.0
TIER2_AZ_HALFWIDTH_DEG = 15.0
CROSS_BODY_TIER = 7


def priority_tier(pose, arm):
    """Selection tier of a grasp `pose` (z-up pelvis frame) for `arm` — lower
    tries first. Azimuth classes of the approach axis, mirrored per hand
    ("toward center" = toward +Y for the right hand, -Y for the left):
    forward (azimuth near +X AND fingers near pelvis +Y), diagonal (TIER2
    band toward center), center (the full toward-center fan). Combined with
    pitch: 1/2/3 = those classes pitched TIER_PITCH_MIN..MAX below
    horizontal, 4/5/6 = any other pitch, 7 = out-of-fan azimuth."""
    eps = 1e-6      # absorb Pose->quaternion->matrix round-trip at boundaries
    R = pose_to_matrix(pose)[:3, :3]
    ax, ay, az = float(R[0, 2]), float(R[1, 2]), float(R[2, 2])
    azim = math.degrees(math.atan2(ay, ax))     # 0 = +X, positive toward +Y
    toward_center = azim if arm == 'right' else -azim
    if (abs(azim) <= TIER1_AZ_TOL_DEG + eps
            and float(R[1, 0])
            >= math.cos(math.radians(TIER1_X_ALIGN_TOL_DEG)) - eps):
        az_class = 0                                            # forward
    elif (abs(toward_center - TIER2_AZ_CENTER_DEG)
            <= TIER2_AZ_HALFWIDTH_DEG + eps):
        az_class = 1                                            # diagonal
    elif -TIER1_AZ_TOL_DEG - eps <= toward_center <= 90.0 + eps:
        az_class = 2                                            # center fan
    else:
        return CROSS_BODY_TIER
    pitch_down = math.degrees(math.asin(np.clip(-az, -1.0, 1.0)))
    pitched = (TIER_PITCH_MIN_DEG - eps <= pitch_down
               <= TIER_PITCH_MAX_DEG + eps)
    return (1 if pitched else 4) + az_class


def gate_min_pitch(poses, idxs, min_pitch_deg):
    """Indices from `idxs` whose approach is at least `min_pitch_deg` below
    horizontal (a minimum, not proximity to vertical). <= 0 keeps all."""
    if min_pitch_deg <= 0.0:
        return list(idxs)
    thr = math.sin(math.radians(min_pitch_deg))
    return [i for i in idxs if -pose_to_matrix(poses[i])[2, 2] >= thr]


def tier_histogram(tiers):
    """'tier x count' summary for log lines, e.g. '1x2, 4x3'."""
    hist = {}
    for t in tiers:
        hist[t] = hist.get(t, 0) + 1
    return ', '.join(f'{t}x{n}' for t, n in sorted(hist.items()))


def poses_close(pa, pb, lin_thr, ang_thr):
    """True when Poses `pa` and `pb` are near-duplicate grasps: within
    `lin_thr` metres AND `ang_thr` radians."""
    Ta, Tb = pose_to_matrix(pa), pose_to_matrix(pb)
    if float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])) >= lin_thr:
        return False
    cos = (float(np.trace(Ta[:3, :3].T @ Tb[:3, :3])) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0))) < ang_thr


def select_diverse(cand, poses, max_n, lin_thr, ang_thr):
    """Greedily pick up to `max_n` indices from `cand` (already ranked
    best-first) that are mutually distinct (poses_close). The best grasp is
    always taken first, so a non-empty `cand` yields at least one index."""
    picked = []
    for i in cand:
        if any(poses_close(poses[i], poses[j], lin_thr, ang_thr)
               for j in picked):
            continue
        picked.append(i)
        if len(picked) >= max_n:
            break
    return picked
