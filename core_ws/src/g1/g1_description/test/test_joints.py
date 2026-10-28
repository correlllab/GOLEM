import importlib.util
import pathlib

import pytest

from g1_description import layout
from g1_description.joints import BODY, LEFT_HAND, RIGHT_HAND

GENERATOR = pathlib.Path(__file__).resolve().parents[1] / 'tools' / 'generate_joints.py'


def test_body_slots_are_contiguous_and_unique():
    assert len(BODY) == layout.NUM_BODY_MOTORS
    assert [j.slot for j in BODY] == list(range(layout.NUM_BODY_MOTORS))
    assert len(set(layout.BODY_JOINT_NAMES)) == layout.NUM_BODY_MOTORS


def test_slot_groups_partition_the_body():
    groups = [layout.LEG_SLOTS, layout.WAIST_SLOTS, layout.LEFT_ARM_SLOTS, layout.RIGHT_ARM_SLOTS]
    assert [s for g in groups for s in g] == list(range(layout.NUM_BODY_MOTORS))
    assert all('waist' in BODY[s].name for s in layout.WAIST_SLOTS)
    assert all(BODY[s].name.startswith('left_shoulder') for s in layout.LEFT_ARM_SLOTS[:3])
    assert all(BODY[s].name.startswith('right_shoulder') for s in layout.RIGHT_ARM_SLOTS[:3])


@pytest.mark.parametrize('side, hand', [('left', LEFT_HAND), ('right', RIGHT_HAND)])
def test_hand_tables(side, hand):
    assert len(hand) == layout.NUM_HAND_MOTORS
    assert [j.slot for j in hand] == list(range(layout.NUM_HAND_MOTORS))
    assert all(j.name.startswith(f'{side}_hand_') for j in hand)


def test_limits_are_sane():
    for joint in BODY + LEFT_HAND + RIGHT_HAND:
        assert joint.lower < joint.upper, joint.name
        assert joint.velocity > 0 and joint.effort > 0, joint.name


def test_hands_mirror_except_thumb_0():
    # leprechaun_g1/assets/README.md: six of seven Dex3 joints have negated
    # limits between hands; thumb_0 has a symmetric range.
    for left, right in zip(LEFT_HAND, RIGHT_HAND):
        if left.name.endswith('thumb_0_joint'):
            assert (left.lower, left.upper) == (right.lower, right.upper)
        else:
            assert (right.lower, right.upper) == (-left.upper, -left.lower), left.name


def test_joints_py_matches_the_urdf():
    spec = importlib.util.spec_from_file_location('generate_joints', GENERATOR)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    if not generator.URDF.is_file():
        pytest.skip(f'{generator.URDF} not available (leprechaun_g1 submodule not checked out)')
    assert generator.OUT.read_text() == generator.render()
