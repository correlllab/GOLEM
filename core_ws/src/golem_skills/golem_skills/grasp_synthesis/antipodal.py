"""antipodal: close across the object's narrowest top-surface span.

PCA of the cloud's top slab (common.top_layer_pca) gives the principal XY
axes; the fingers close across the MINOR axis — the antipodal pair with the
shortest span — at the slab centroid, FINGER_SINK_M below the top surface.
The closing axis is the method's whole idea, so the wrist is never re-rolled:
candidates walk SURFACE_GRASP_TILTS on the minor axis, then one 45 deg grasp
across the major axis as a fallback. The planned opening is the minor extent.
"""

import math

from .common import (SURFACE_GRASP_TILTS, make_candidates, surface_contact,
                     tilted_pose, top_layer_pca)
from .ranking import gate_min_pitch


def synthesize(inp):
    cloud = inp.cloud
    pca = top_layer_pca(cloud)
    tip = surface_contact(pca['centroid'][:2], pca['top_z'],
                          float(cloud[:, 2].min()))
    poses = [tilted_pose(tip, pca['minor_yaw'], t)
             for t, _ in SURFACE_GRASP_TILTS]
    labels = [f'antipodal-minor-{name}' for _, name in SURFACE_GRASP_TILTS]
    poses.append(tilted_pose(tip, pca['major_yaw'], math.pi / 4.0))
    labels.append('antipodal-major-45deg')

    keep = gate_min_pitch(poses, range(len(poses)), inp.min_pitch_deg)
    if not keep:
        return None, (f'no antipodal grasp for {inp.obj!r} pitched >= '
                      f'{inp.min_pitch_deg:.0f} deg down')
    inp.log.info(
        f'grasp[antipodal]: {inp.obj!r} top slab spans '
        f'{pca["minor_extent"] * 1000:.0f} x {pca["major_extent"] * 1000:.0f} mm, '
        f'closing across the minor axis (yaw '
        f'{math.degrees(pca["minor_yaw"]):.0f} deg), {len(keep)} grasp(s)')
    return make_candidates([poses[i] for i in keep], [labels[i] for i in keep],
                           inp.arm, total=len(poses),
                           gripper_width=pca['minor_extent']), None
