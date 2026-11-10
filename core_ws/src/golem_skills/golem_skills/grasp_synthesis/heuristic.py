"""
Synthesize a fixed set of level-wrist grasps at the centroid (the `heuristic` method).

One grasp per (azimuth toward the working side, pitch below horizontal) at the
object-cloud mean, each with a level wrist (+X horizontal, +Y up) and the
finger contact point on the centroid, ordered by ranking.priority_tier (the
same tiers graspgenx_filtered ranks GraspGenX output by). There is no learned
model and no shape analysis: it encodes only where the arm can reach.
"""

from .common import approach_from_az_pitch, level_pose, make_candidates
from .ranking import CROSS_BODY_TIER, gate_min_pitch, tier_histogram

# deg in the pelvis XY plane: 0 = forward (pelvis +X), positive = toward the
# robot's midline, mirrored per arm; one each for the forward, diagonal and
# center tier classes
AZIMUTHS_DEG = (0.0, 45.0, 90.0)
# deg below horizontal: 45 is inside the tier 1-3 pitch band, 10 a gentle
# downward rake (tiers 4-6)
PITCHES_DEG = (45.0, 10.0)


def synthesize(inp):
    """Return (Candidates, None) in tier order, or (None, reason) when none survive the gates."""
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
    # stable sort, so equal tiers keep the PITCHES_DEG x AZIMUTHS_DEG order
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
