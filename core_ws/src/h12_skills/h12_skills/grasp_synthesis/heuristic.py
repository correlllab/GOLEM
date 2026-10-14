"""heuristic: a fixed set of canonical level-wrist grasps at the centroid.

One grasp per (azimuth toward the working side, pitch below horizontal) at the
object-cloud mean, each with a level wrist (+X horizontal, +Y up) and the
finger contact point on the centroid, ordered by the priority tiers
(ranking.priority_tier) that also rank GraspGenX output in graspgenx_filtered.
No learned model and no shape analysis: it encodes only where this arm can
reach.
"""

from .common import approach_from_az_pitch, level_pose, make_candidates
from .ranking import CROSS_BODY_TIER, gate_min_pitch, tier_histogram

# Approach azimuths in the pelvis XY plane [deg]: 0 = straight forward
# (pelvis +X); positive = toward the robot's midline, mirrored per arm. They
# map to the forward / diagonal / center tier classes.
AZIMUTHS_DEG = (0.0, 45.0, 90.0)
# Approach pitches below horizontal [deg]: 45 sits inside the tier 1-3 band,
# 10 is a gentle downward rake (tiers 4-6).
PITCHES_DEG = (45.0, 10.0)


def synthesize(inp):
    side = 1.0 if inp.arm == 'right' else -1.0
    c = inp.centroid
    specs = [(side * az, p) for p in PITCHES_DEG for az in AZIMUTHS_DEG]
    poses = [level_pose(c, approach_from_az_pitch(az, p)) for az, p in specs]
    labels = [f'heuristic-az{abs(az):.0f}-pitch{p:.0f}' for az, p in specs]

    idxs = gate_min_pitch(poses, range(len(poses)), inp.min_pitch_deg)
    if not idxs:
        return None, (f'no heuristic grasp for {inp.obj!r} pitched >= '
                      f'{inp.min_pitch_deg:.0f} deg down')
    cands = make_candidates([poses[i] for i in idxs], [labels[i] for i in idxs],
                            inp.arm, total=len(specs))
    # Tier-major order (stable), dropping cross-body entries.
    order = sorted((k for k, t in enumerate(cands.tiers) if t != CROSS_BODY_TIER),
                   key=lambda k: cands.tiers[k])
    if not order:
        return None, f'no on-side heuristic grasp for {inp.obj!r}'
    cands = make_candidates([cands.grasps[k].pose for k in order],
                            [cands.labels[k] for k in order], inp.arm,
                            total=len(specs))
    inp.log.info(
        f'grasp[heuristic]: {inp.obj!r} [{inp.arm}] at centroid ({c[0]:.3f}, '
        f'{c[1]:.3f}, {c[2]:.3f}) pelvis — {len(order)} grasp(s), tiers '
        f'{tier_histogram(cands.tiers)}')
    return cands, None
