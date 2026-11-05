"""
Synthesize ONE steep grasp at the object-cloud mean (the `top_down` method).

The approach lies in the pelvis XZ plane, pitched TOP_DOWN_PITCH_DEG below
horizontal; the fingers close along pelvis +Y with the wrist as level as the
pitch allows, and the finger contact point is on the centroid. goal.top_down
selects it; it is the small-part (battery screw) grasp, usually paired with
goal.visual_servo. 80 deg is steeper than the H1-2 wrist's practical ~45 deg
approach, so the servo converges best-effort there.
"""

import math

import numpy as np

from .common import grasp_pose, make_candidates
from .ranking import gate_min_pitch

# deg below horizontal; the approach points down and forward (pelvis +X), 90 = straight down
TOP_DOWN_PITCH_DEG = 80.0


def top_down_pose(contact):
    """Return the top_down gripper-base Pose (pelvis) whose finger contact point is `contact`."""
    pitch = math.radians(TOP_DOWN_PITCH_DEG)
    z_axis = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    return grasp_pose(contact, [0.0, 1.0, 0.0], z_axis)


def synthesize(inp):
    """Return (Candidates, None) with the one grasp, or (None, reason) if the gate drops it."""
    c = inp.centroid
    pose = top_down_pose(c)
    if not gate_min_pitch([pose], [0], inp.min_pitch_deg):
        return None, (f'top_down grasp for {inp.obj!r} is pitched only '
                      f'{TOP_DOWN_PITCH_DEG:.0f} deg down, below the requested '
                      f'{inp.min_pitch_deg:.0f} deg')
    inp.log.info(
        f'grasp[top_down]: {inp.obj!r} at centroid ({c[0]:.3f}, {c[1]:.3f}, '
        f'{c[2]:.3f}) pelvis, {TOP_DOWN_PITCH_DEG:.0f} deg below horizontal')
    return make_candidates([pose], [f'top_down-{TOP_DOWN_PITCH_DEG:.0f}deg'],
                           inp.arm), None
