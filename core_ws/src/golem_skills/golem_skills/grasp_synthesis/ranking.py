"""Gate, tier and diversify grasp candidates (pure functions over pelvis-frame Poses)."""

import math

import numpy as np

from ..perception_utils import pose_to_matrix


# Priority tiers (priority_tier), lower tried first. Tiers 1-3 want the
# approach pitched TIER_PITCH_MIN..MAX_DEG below horizontal: steeper than about
# 45 deg is beyond the H1-2 arm's practical reach, flatter rakes the support
# surface. Tiers 4-6 are the same azimuth classes at any other pitch; 7 is an
# azimuth outside the toward-center fan (cross-body or behind).
TIER_PITCH_MIN_DEG = 20.0
TIER_PITCH_MAX_DEG = 45.0
# forward tiers (1, 4): approach azimuth tolerance off pelvis +X, and the
# finger-closing +X axis tolerance off pelvis +Y [deg]
TIER1_AZ_TOL_DEG = 15.0
TIER1_X_ALIGN_TOL_DEG = 15.0
# diagonal tiers (2, 5): azimuth band toward the robot's midline [deg]
TIER2_AZ_CENTER_DEG = 45.0
TIER2_AZ_HALFWIDTH_DEG = 15.0
CROSS_BODY_TIER = 7


def priority_tier(pose, arm):
    """
    Return the selection tier (1-7, lower first) of grasp `pose` (z-up pelvis frame) for `arm`.

    The approach axis's azimuth picks a class, mirrored per hand ("toward
    center" is toward +Y for the right hand, -Y for the left): forward
    (azimuth near +X and fingers near pelvis +Y), diagonal (the TIER2 band),
    center (the rest of the toward-center fan). Classes pitched
    TIER_PITCH_MIN..MAX_DEG below horizontal are tiers 1/2/3, any other pitch
    4/5/6, and an azimuth outside the fan is CROSS_BODY_TIER.
    """
    # absorbs the Pose -> quaternion -> matrix round-trip error at band edges
    eps = 1e-6
    R = pose_to_matrix(pose)[:3, :3]
    ax, ay, az = float(R[0, 2]), float(R[1, 2]), float(R[2, 2])
    # deg, 0 = +X, positive toward +Y
    azim = math.degrees(math.atan2(ay, ax))
    toward_center = azim if arm == 'right' else -azim
    if (abs(azim) <= TIER1_AZ_TOL_DEG + eps
            and float(R[1, 0])
            >= math.cos(math.radians(TIER1_X_ALIGN_TOL_DEG)) - eps):
        az_class = 0  # forward
    elif (abs(toward_center - TIER2_AZ_CENTER_DEG)
            <= TIER2_AZ_HALFWIDTH_DEG + eps):
        az_class = 1  # diagonal
    elif -TIER1_AZ_TOL_DEG - eps <= toward_center <= 90.0 + eps:
        az_class = 2  # center fan
    else:
        return CROSS_BODY_TIER
    pitch_down = math.degrees(math.asin(np.clip(-az, -1.0, 1.0)))
    pitched = (TIER_PITCH_MIN_DEG - eps <= pitch_down
               <= TIER_PITCH_MAX_DEG + eps)
    return (1 if pitched else 4) + az_class


def gate_min_pitch(poses, idxs, min_pitch_deg):
    """
    Return the indices from `idxs` whose approach is at least `min_pitch_deg` below horizontal.

    The gate is a minimum pitch, not closeness to vertical; `min_pitch_deg`
    <= 0 keeps every index.
    """
    if min_pitch_deg <= 0.0:
        return list(idxs)
    thr = math.sin(math.radians(min_pitch_deg))
    return [i for i in idxs if -pose_to_matrix(poses[i])[2, 2] >= thr]


def tier_histogram(tiers):
    """Return a 'tier x count' summary for log lines, e.g. '1x2, 4x3'."""
    hist = {}
    for t in tiers:
        hist[t] = hist.get(t, 0) + 1
    return ', '.join(f'{t}x{n}' for t, n in sorted(hist.items()))


def poses_close(pa, pb, lin_thr, ang_thr):
    """Return True when `pa` and `pb` are within `lin_thr` m AND `ang_thr` rad of each other."""
    Ta, Tb = pose_to_matrix(pa), pose_to_matrix(pb)
    if float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])) >= lin_thr:
        return False
    cos = (float(np.trace(Ta[:3, :3].T @ Tb[:3, :3])) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0))) < ang_thr


def select_diverse(cand, poses, max_n, lin_thr, ang_thr):
    """
    Pick up to `max_n` mutually distinct indices from `cand`, which is ranked best-first.

    Greedy: an index is skipped when poses_close to one already picked. The
    best grasp is always picked, so a non-empty `cand` yields at least one.
    """
    picked = []
    for i in cand:
        if any(poses_close(poses[i], poses[j], lin_thr, ang_thr)
               for j in picked):
            continue
        picked.append(i)
        if len(picked) >= max_n:
            break
    return picked
