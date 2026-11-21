"""
Synthesize GraspGenX 6-DOF grasps, raw or reachability-filtered.

Both methods send the same object and scene clouds to the same graspgen
service (model_server/graspgen_server.py owns sampling and collision
filtering), so the difference between them is the client-side selection only:

  graspgenx           the server's grasps executed best-first by model score,
                      with no re-roll, re-rank or diversity pick
  graspgenx_filtered  drop cross-body approaches, re-roll the levelable ones
                      to a level wrist, order tier-major, then pick mutually
                      diverse grasps

Both hand the executor up to MAX_GRASP_CHECKS candidates in that order; its
reach check skips those the arm cannot approach, and it moves the arm for at
most MAX_GRASP_ATTEMPTS of the rest.
"""

import math

from .common import make_candidates, roll_to_yup
from .ranking import (CROSS_BODY_TIER, gate_min_pitch, priority_tier,
                      select_diverse, tier_histogram)
from ..perception_utils import pose_to_matrix

# grasps the arm moves for before giving up
MAX_GRASP_ATTEMPTS = 5
# grasps handed to the executor's reach check (about a second each), in order
MAX_GRASP_CHECKS = 60
# graspgenx_filtered: a candidate within BOTH thresholds of one already picked
# (position AND orientation) is a duplicate and skipped
GRASP_DIVERSITY_LIN_M = 0.02
GRASP_DIVERSITY_ANG_DEG = 20.0
# deg: after a re-roll, +Y's angle from up equals the approach's tilt off
# horizontal, so approaches steeper than this cannot be levelled and keep
# GraspGenX's own roll (a flat part needs them: its side grasps sit at the
# support surface)
GRASP_YUP_TOL_DEG = 55.0


def _plan(inp):
    """Return (graspgen response, None), or (None, reason) when the service gives none."""
    if inp.gripper_name is None:
        return None, "GraspGenX has no model of this robot's gripper"
    resp = inp.plan_grasp(inp.cloud, gripper_name=inp.gripper_name, frame='pelvis',
                          scene_cloud=inp.scene, arm=inp.arm)
    if resp is None:
        return None, f'no grasp planned for {inp.obj!r}'
    return resp, None


def synthesize_raw(inp):
    """Return the server's top MAX_GRASP_CHECKS grasps in its order, after the pitch gate."""
    resp, err = _plan(inp)
    if err:
        return None, err
    poses = [g.pose for g in resp.grasps]
    keep = gate_min_pitch(poses, range(len(poses)), inp.min_pitch_deg)
    if not keep:
        return None, (f'no grasp for {inp.obj!r} pitched >= '
                      f'{inp.min_pitch_deg:.0f} deg down')
    keep = keep[:MAX_GRASP_CHECKS]
    inp.log.info(
        f'grasp[graspgenx]: checking the top {len(keep)} of {len(poses)} '
        f'GraspGenX grasps for {inp.obj!r} as ranked, executing up to '
        f'{MAX_GRASP_ATTEMPTS}')
    return make_candidates(
        [poses[i] for i in keep], [f'ggx-{i}' for i in keep], inp.arm,
        scores=[float(resp.scores[i]) for i in keep], total=len(poses),
        gripper_width=float(resp.gripper_width),
        attempts=MAX_GRASP_ATTEMPTS), None


def synthesize_filtered(inp):
    """
    Return the server's grasps filtered, levelled, tier-ordered and diversified.

    In order: plan; the optional downward-pitch gate; drop cross-body
    approaches and re-roll the levelable ones to a level wrist; sort by tier
    (stable, so GraspGenX confidence breaks ties); pick MAX_GRASP_CHECKS
    mutually diverse grasps.
    """
    resp, err = _plan(inp)
    if err:
        return None, err
    arm, obj = inp.arm, inp.obj
    poses = [g.pose for g in resp.grasps]
    cand = gate_min_pitch(poses, range(len(poses)), inp.min_pitch_deg)
    if inp.min_pitch_deg > 0.0:
        inp.log.info(
            f'grasp[graspgenx_filtered]: downward-pitch gate >= '
            f'{inp.min_pitch_deg:.0f} deg kept {len(cand)}/{len(poses)}')
    if not cand:
        return None, (f'no grasp for {obj!r} pitched >= '
                      f'{inp.min_pitch_deg:.0f} deg down')

    # Roll about the approach is free, so mis-rolled grasps are re-levelled
    # rather than dropped; the gates below test only the approach axis, which
    # the re-roll does not move.
    tier0 = {i: priority_tier(poses[i], arm) for i in cand}
    yup_sin = math.sin(math.radians(GRASP_YUP_TOL_DEG))

    def levelable(i):
        """Return True when grasp `i`'s approach is shallow enough to level the wrist."""
        return abs(float(pose_to_matrix(poses[i])[2, 2])) <= yup_sin

    kept = [i for i in cand if tier0[i] != CROSS_BODY_TIER]
    n_level = sum(1 for i in kept if levelable(i))
    inp.log.info(
        f'grasp[graspgenx_filtered]: {arm}-arm kept {len(kept)}/{len(cand)} '
        f'({n_level} levelled, <={GRASP_YUP_TOL_DEG:.0f}deg pitch; the rest '
        f'keep their roll)')
    if not kept:
        return None, (f'no {arm}-arm grasp for {obj!r}: all {len(cand)} '
                      f'approach cross-body')
    for i in kept:
        if levelable(i):
            poses[i] = roll_to_yup(poses[i])

    # after the re-roll, so the forward tier's finger-axis test sees the levelled fingers
    tier_of = {i: priority_tier(poses[i], arm) for i in kept}
    kept.sort(key=lambda i: tier_of[i])
    inp.log.info(
        f'grasp[graspgenx_filtered]: priority tiers over {len(kept)} kept '
        f'candidate(s) [{arm}]: {tier_histogram(tier_of.values())}')

    picked = select_diverse(kept, poses, MAX_GRASP_CHECKS,
                            GRASP_DIVERSITY_LIN_M,
                            math.radians(GRASP_DIVERSITY_ANG_DEG))
    inp.log.info(
        f'grasp[graspgenx_filtered]: selected {len(picked)} diverse '
        f'candidate(s) for {obj!r} (>= {GRASP_DIVERSITY_LIN_M * 100:.0f}cm or '
        f'{GRASP_DIVERSITY_ANG_DEG:.0f}deg apart) from {len(poses)} ranked')
    return make_candidates(
        [poses[i] for i in picked], [f'ggx-{i}' for i in picked], arm,
        scores=[float(resp.scores[i]) for i in picked], total=len(poses),
        gripper_width=float(resp.gripper_width),
        attempts=MAX_GRASP_ATTEMPTS), None
