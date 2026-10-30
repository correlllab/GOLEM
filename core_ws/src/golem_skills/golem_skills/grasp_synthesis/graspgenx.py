"""GraspGenX 6-DOF grasps, raw and reachability-filtered.

Both methods send the same object + scene clouds to the same graspgen service
(model_server/graspgen_server.py owns sampling, collision filtering and its
own approach envelope), so the difference between them isolates the
client-side selection layer:

  graspgenx           the server's grasps executed strictly best-first by
                      model score — no re-roll, re-rank or diversity pick.
  graspgenx_filtered  the selection chain below: drop cross-body and
                      unlevelable approaches, re-roll the rest to a level
                      wrist, order tier-major, then pick mutually diverse
                      grasps (GGX-F).
"""

import math

from .common import make_candidates, roll_to_yup
from .ranking import (CROSS_BODY_TIER, gate_min_pitch, priority_tier,
                      select_diverse, tier_histogram)
from ..perception_utils import pose_to_matrix

# Grasps to try before giving up — the top grasp may be IK-unreachable.
MAX_GRASP_ATTEMPTS = 5
# Diversity thresholds (graspgenx_filtered): a candidate within BOTH of an
# already-selected one (position AND orientation) is a duplicate and skipped.
GRASP_DIVERSITY_LIN_M = 0.02
GRASP_DIVERSITY_ANG_DEG = 20.0
# The achievable +Y-from-up after a re-roll equals the approach's tilt off
# horizontal, so approaches steeper than this can't be leveled and are dropped.
GRASP_YUP_TOL_DEG = 55.0


def _plan(inp):
    """(resp, None) from the graspgen service, or (None, reason)."""
    resp = inp.plan_grasp(inp.cloud, gripper_name='magpie', frame='pelvis',
                          scene_cloud=inp.scene, arm=inp.arm)
    if resp is None:
        return None, f'no grasp planned for {inp.obj!r}'
    return resp, None


def synthesize_raw(inp):
    resp, err = _plan(inp)
    if err:
        return None, err
    poses = [g.pose for g in resp.grasps]
    keep = gate_min_pitch(poses, range(len(poses)), inp.min_pitch_deg)
    if not keep:
        return None, (f'no grasp for {inp.obj!r} pitched >= '
                      f'{inp.min_pitch_deg:.0f} deg down')
    keep = keep[:MAX_GRASP_ATTEMPTS]
    inp.log.info(
        f'grasp[graspgenx]: executing the top {len(keep)} of {len(poses)} '
        f'GraspGenX grasps for {inp.obj!r} as ranked')
    return make_candidates(
        [poses[i] for i in keep], [f'ggx-{i}' for i in keep], inp.arm,
        scores=[float(resp.scores[i]) for i in keep], total=len(poses),
        gripper_width=float(resp.gripper_width)), None


def synthesize_filtered(inp):
    """Stages, in order: plan; optional downward-pitch gate; arm-side +
    levelable filter, then re-roll survivors to Y-up; tier-major ordering
    (stable, so GraspGenX confidence breaks ties); diversity pick of the
    MAX_GRASP_ATTEMPTS grasps actually tried."""
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

    # Roll about the approach axis is a free parameter, so mis-rolled grasps
    # are re-leveled instead of dropped. Both remaining gates are on the
    # approach axis, which the re-roll never moves.
    tier0 = {i: priority_tier(poses[i], arm) for i in cand}
    yup_sin = math.sin(math.radians(GRASP_YUP_TOL_DEG))

    def levelable(i):
        return abs(float(pose_to_matrix(poses[i])[2, 2])) <= yup_sin

    kept = [i for i in cand if tier0[i] != CROSS_BODY_TIER and levelable(i)]
    n_side = sum(1 for i in cand if tier0[i] != CROSS_BODY_TIER)
    n_level = sum(1 for i in cand if levelable(i))
    inp.log.info(
        f'grasp[graspgenx_filtered]: {arm}-arm + levelable '
        f'(<={GRASP_YUP_TOL_DEG:.0f}deg pitch) kept {len(kept)}/{len(cand)} '
        f'(on-side {n_side}, levelable {n_level})')
    if not kept:
        return None, (
            f'no levelable {arm}-arm grasp for {obj!r}: all {len(cand)} '
            f'filtered (on-side {n_side}, levelable {n_level}) — approaches '
            f'too steep to level, or all cross-body')
    for i in kept:
        poses[i] = roll_to_yup(poses[i])

    # Tiers recomputed AFTER the re-roll so the forward tier's finger-axis
    # check sees the leveled finger direction.
    tier_of = {i: priority_tier(poses[i], arm) for i in kept}
    kept.sort(key=lambda i: tier_of[i])
    inp.log.info(
        f'grasp[graspgenx_filtered]: priority tiers over {len(kept)} kept '
        f'candidate(s) [{arm}]: {tier_histogram(tier_of.values())}')

    picked = select_diverse(kept, poses, MAX_GRASP_ATTEMPTS,
                            GRASP_DIVERSITY_LIN_M,
                            math.radians(GRASP_DIVERSITY_ANG_DEG))
    inp.log.info(
        f'grasp[graspgenx_filtered]: selected {len(picked)} diverse '
        f'candidate(s) for {obj!r} (>= {GRASP_DIVERSITY_LIN_M * 100:.0f}cm or '
        f'{GRASP_DIVERSITY_ANG_DEG:.0f}deg apart) from {len(poses)} ranked')
    return make_candidates(
        [poses[i] for i in picked], [f'ggx-{i}' for i in picked], arm,
        scores=[float(resp.scores[i]) for i in picked], total=len(poses),
        gripper_width=float(resp.gripper_width)), None
