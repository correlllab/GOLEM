"""
Synthesize fixed grasps at the object's robust centroid (the `centroid` method).

The fingertips go to the cloud's median XY (mask-edge outliers drag the mean
off the object), FINGER_SINK_M below its top surface, and always close across
pelvis +Y whatever the object's own orientation: the shape-blind baseline the
shape-aware methods are compared with. Candidates walk SURFACE_GRASP_TILTS,
then one 45 deg grasp closing across pelvis +X for when the first yaw is
IK-blocked.
"""

import math

import numpy as np

from .common import (SURFACE_GRASP_TILTS, make_candidates, robust_top_z,
                     surface_contact, tilted_pose)
from .ranking import gate_min_pitch


def synthesize(inp):
    """Return (Candidates, None) at the centroid, or (None, reason) if the pitch gate drops all."""
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
