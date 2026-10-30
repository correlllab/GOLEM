"""top_down: ONE steep grasp at the object-cloud mean.

The tier-1 orientation (approach in the pelvis XZ plane, fingers closing
along pelvis +Y, wrist as level as the pitch allows) pitched TOP_DOWN_PITCH_DEG
below horizontal, with the finger contact point on the centroid. This is the
deployed small-part grasp (battery screws) and what goal.top_down selects;
paired with goal.visual_servo it is the closed-loop C-VS method. 80 deg is
past the wrist's practical ~45 deg approach, so expect best-effort
convergence from the servo.
"""

import math

import numpy as np

from .common import grasp_pose, make_candidates
from .ranking import gate_min_pitch

# Pitch of the approach below horizontal [deg]; it points DOWN and FORWARD,
# away from the robot (pelvis +X). 90 = straight down.
TOP_DOWN_PITCH_DEG = 80.0


def top_down_pose(contact):
    """Gripper-base Pose (pelvis) of the top_down grasp whose finger contact
    point lands on `contact`."""
    pitch = math.radians(TOP_DOWN_PITCH_DEG)
    z_axis = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    return grasp_pose(contact, [0.0, 1.0, 0.0], z_axis)


def synthesize(inp):
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
