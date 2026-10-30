"""centroid: fixed grasps at the object's robust centroid.

The cloud's median XY (mask-edge outliers drag the mean off the object), at a
fingertip depth FINGER_SINK_M below its top surface. The fingers always close
across pelvis +Y, whatever the object's own orientation — the naive baseline
the shape-aware methods are measured against. Candidates walk
SURFACE_GRASP_TILTS, then one 45 deg grasp closing across pelvis +X as a
fallback when the primary yaw is IK-blocked.
"""

import math

import numpy as np

from .common import (SURFACE_GRASP_TILTS, make_candidates, robust_top_z,
                     surface_contact, tilted_pose)
from .ranking import gate_min_pitch


def synthesize(inp):
    cloud = inp.cloud
    c = np.median(cloud, axis=0)
    tip = surface_contact(c[:2], robust_top_z(cloud),
                          float(np.percentile(cloud[:, 2], 5)))
    poses = [tilted_pose(tip, math.pi / 2.0, t) for t, _ in SURFACE_GRASP_TILTS]
    labels = [f'centroid-y-{name}' for _, name in SURFACE_GRASP_TILTS]
    poses.append(tilted_pose(tip, 0.0, math.pi / 4.0))
    labels.append('centroid-x-45deg')

    keep = gate_min_pitch(poses, range(len(poses)), inp.min_pitch_deg)
    if not keep:
        return None, (f'no centroid grasp for {inp.obj!r} pitched >= '
                      f'{inp.min_pitch_deg:.0f} deg down')
    inp.log.info(
        f'grasp[centroid]: {inp.obj!r} fingertip at ({tip[0]:.3f}, '
        f'{tip[1]:.3f}, {tip[2]:.3f}) pelvis, {len(keep)} fixed grasp(s)')
    return make_candidates([poses[i] for i in keep], [labels[i] for i in keep],
                           inp.arm, total=len(poses)), None
