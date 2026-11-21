"""
Name the hands the grasp skills close, and judge a Dex3-1 grip from its finger joints.

RobotProfile.hand (robots.py) is one of HANDS:

  magpie  the H1-2's Magpie gripper (magpie_control and the H1-2 sims):
          <ns>/set_position, set_force, open, close, reset_overload and the
          <ns>/state aperture, ns = /<arm>/gripper; SkillsBase drives it
  dex3    the G1's Dex3-1 hand (dex_3's dex3_hands): /<arm>/hand/open and
          /<arm>/hand/close (std_srvs/Trigger), and the finger joints on
          /joint_states. No force limit and no aperture: the grip check reads
          how far the fingers closed (dex3_grip_failure).

This module imports no ROS package.
"""

import numpy as np

from dex_3.dex3 import JOINT_NAMES, named_pose

HANDS = ('magpie', 'dex3')

# the Dex3-1 driver's per-arm service namespace: <ns>/open, <ns>/close
DEX3_NS = {'left': '/left/hand', 'right': '/right/hand'}

# Fraction of each closing joint's travel from `open` to `close` (dex_3's named poses)
# the fingers reached after a close: every one at or above DEX3_EMPTY_FRACTION means
# nothing stopped them (a close on nothing leaves the thumb about 1 cm from the finger
# tips); the index and middle fingers' mean below DEX3_NOT_CLOSED_FRACTION means they
# did not close at all. Between the two, something is held. Chosen, not measured:
# unmeasured on the real G1 and its sim.
DEX3_EMPTY_FRACTION = 0.9
DEX3_NOT_CLOSED_FRACTION = 0.2


def dex3_closing_joints(arm):
    """Return the Dex3-1 joints of `arm` that move between `open` and `close`, in driver order."""
    travel = named_pose(arm, 'close') - named_pose(arm, 'open')
    return [n for n, t in zip(JOINT_NAMES[arm], travel) if abs(t) > 1e-9]


def dex3_closure(arm, positions):
    """
    Return {joint: fraction of its open -> close travel reached} for `arm`'s closing joints.

    `positions` maps joint names to rad (at least dex3_closing_joints(arm)).
    0 is open, 1 the `close` pose; a finger pushed back past open reads below 0.
    """
    q_open = dict(zip(JOINT_NAMES[arm], named_pose(arm, 'open')))
    q_close = dict(zip(JOINT_NAMES[arm], named_pose(arm, 'close')))
    return {n: float((positions[n] - q_open[n]) / (q_close[n] - q_open[n]))
            for n in dex3_closing_joints(arm)}


def dex3_grip_failure(arm, positions):
    """
    Return why `arm`'s Dex3-1 holds nothing after a close, given its finger joints, else None.

    `positions` is {joint: rad} after the close, or None when no finger state arrived.
    """
    if positions is None:
        return 'no hand joint state arrived after the close'
    frac = dex3_closure(arm, positions)
    fingers = [f for n, f in frac.items() if '_index_' in n or '_middle_' in n]
    if float(np.mean(fingers)) < DEX3_NOT_CLOSED_FRACTION:
        return f'the fingers did not close ({dex3_closure_text(frac)})'
    if min(frac.values()) >= DEX3_EMPTY_FRACTION:
        return f'the hand closed on nothing ({dex3_closure_text(frac)})'
    return None


def dex3_closure_text(frac):
    """Return the closure fractions as a short log string, e.g. 'thumb_1 0.95, index_0 0.40'."""
    return ', '.join(f'{n.split("_hand_", 1)[-1].removesuffix("_joint")} {f:.2f}'
                     for n, f in frac.items())
