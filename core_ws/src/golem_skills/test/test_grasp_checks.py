"""
Test the grasp skill's decisions after an approach: whether to close, and whether the grip holds.

No ROS graph is needed; importing the skill needs the workspace's messages.
"""

import pytest

from golem_skills.skills.grasp import (AXIS_APPROACH_MIN_FRACTION, EMPTY_GRIP_MAX_MM,
                                       OPEN_GRIP_MIN_MM, axis_approach_closes, grip_failure)


def test_an_axis_move_that_stops_at_its_first_step_does_not_close():
    """Nothing covered of a 7.5 cm move is a missed candidate, not a close in the air."""
    assert not axis_approach_closes(0.0, 0.075)


def test_an_axis_move_that_stops_near_the_part_closes():
    """A move stopped on its last step (5 of 7.5 cm covered) closes where it stopped."""
    assert axis_approach_closes(0.05, 0.075)
    assert axis_approach_closes(0.075, 0.075)


def test_the_close_threshold_is_the_minimum_fraction():
    """Exactly AXIS_APPROACH_MIN_FRACTION of the move closes; just under it does not."""
    depth = 0.08
    assert axis_approach_closes(AXIS_APPROACH_MIN_FRACTION * depth, depth)
    assert not axis_approach_closes(AXIS_APPROACH_MIN_FRACTION * depth - 1e-6, depth)


def test_a_move_with_no_depth_closes():
    """A grasp whose pre-grasp is already at the contact closes without an axis move."""
    assert axis_approach_closes(0.0, 0.0)


@pytest.mark.parametrize('aperture_mm', [0.0, 0.1, EMPTY_GRIP_MAX_MM])
def test_pads_together_is_an_empty_grip(aperture_mm):
    """An aperture at or below EMPTY_GRIP_MAX_MM after the close means nothing is held."""
    assert 'closed on nothing' in grip_failure(aperture_mm)


@pytest.mark.parametrize('aperture_mm', [EMPTY_GRIP_MAX_MM + 0.5, 21.4, 47.7, 89.9])
def test_an_aperture_on_a_part_is_a_grip(aperture_mm):
    """An aperture between the empty and the open bounds counts as holding the part."""
    assert grip_failure(aperture_mm) is None


@pytest.mark.parametrize('aperture_mm', [OPEN_GRIP_MIN_MM, 102.9, 108.1])
def test_jaws_left_open_are_no_grip(aperture_mm):
    """Fingers that stayed near fully open after the close hold nothing."""
    assert 'did not close' in grip_failure(aperture_mm)


def test_no_gripper_state_fails_the_grip():
    """Without a gripper state after the close the skill cannot claim a grasp."""
    assert 'no gripper state' in grip_failure(None)
