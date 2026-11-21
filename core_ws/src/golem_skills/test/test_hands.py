"""Test the Dex3-1 grip check (hands.py) on joint positions built from dex_3's poses."""

import pytest

from dex_3.dex3 import JOINT_NAMES, named_pose
from golem_skills.hands import (DEX3_EMPTY_FRACTION, dex3_closing_joints, dex3_closure,
                                dex3_grip_failure)


def _at(arm, fraction, override=None):
    """Return {joint: rad} `fraction` of the way from open to close, with `override` fractions."""
    q_open, q_close = named_pose(arm, 'open'), named_pose(arm, 'close')
    out = {}
    for name, o, c in zip(JOINT_NAMES[arm], q_open, q_close):
        f = (override or {}).get(name.split('_hand_')[1], fraction)
        out[name] = float(o + f * (c - o))
    return out


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_thumb_rotation_is_not_a_closing_joint(arm):
    """thumb_0 stays at 0 in both poses, so the check leaves it out; the other six count."""
    joints = dex3_closing_joints(arm)
    assert f'{arm}_hand_thumb_0_joint' not in joints
    assert len(joints) == 6


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_closure_reads_zero_open_and_one_closed(arm):
    """The open pose reads 0 and the close pose 1 on every closing joint, both hands."""
    assert all(abs(f) < 1e-9 for f in dex3_closure(arm, _at(arm, 0.0)).values())
    assert all(abs(f - 1.0) < 1e-9 for f in dex3_closure(arm, _at(arm, 1.0)).values())


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_grip_verdicts(arm):
    """Nothing in the way is empty, fingers stopped short hold, fingers that stayed open failed."""
    assert dex3_grip_failure(arm, None) == 'no hand joint state arrived after the close'
    assert 'closed on nothing' in dex3_grip_failure(arm, _at(arm, 1.0))
    assert 'closed on nothing' in dex3_grip_failure(arm, _at(arm, DEX3_EMPTY_FRACTION))
    assert 'did not close' in dex3_grip_failure(arm, _at(arm, 0.05))
    # an object stops the index and middle fingers half way; the thumb closes fully
    held = _at(arm, 1.0, {'index_0_joint': 0.5, 'index_1_joint': 0.6,
                          'middle_0_joint': 0.5, 'middle_1_joint': 0.6})
    assert dex3_grip_failure(arm, held) is None
    # one finger stopped by the object is enough
    assert dex3_grip_failure(arm, _at(arm, 1.0, {'thumb_2_joint': 0.4})) is None
