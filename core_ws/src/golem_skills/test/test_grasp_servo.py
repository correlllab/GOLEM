"""
Test the visual servo's depth floor, its blocked-approach test and its tolerances.

No ROS graph is needed; importing the skill needs the workspace's messages.
"""

import numpy as np
import pytest

from golem_skills.skills.grasp_servo import (
    BLOCKED_MIN_STEP_M, FINGERTIP_REACH_M, PAD_X_SPAN_M, SURFACE_MIN_POINTS,
    VISUAL_SERVO_DEPTH_M, VISUAL_SERVO_RANGE_TOL_M, VISUAL_SERVO_SURFACE_CLEARANCE_M,
    VISUAL_SERVO_TARGET_XY_M, VISUAL_SERVO_X_TOL_M, VISUAL_SERVO_Y_TOL_M, approach_blocked,
    close_reach, part_width, servo_depth_floor, surface_height)

OPEN_GAP_M = FINGERTIP_REACH_M[-1][0]
OPEN_REACH_M = FINGERTIP_REACH_M[-1][1]
DEEPEST_REACH_M = max(r for _, r in FINGERTIP_REACH_M)


def plane(z, x=(-0.08, 0.08), y=(-0.08, 0.08), step=0.002):
    """Return an (N, 3) grid of optical-frame points at depth `z` over the `x` and `y` spans."""
    xs, ys = np.meshgrid(np.arange(x[0], x[1], step), np.arange(y[0], y[1], step))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=-1)


def test_a_part_wider_than_the_open_jaws_gets_the_open_reach():
    """Closing onto a part at or beyond the open gap does not move the pads past fully open."""
    assert close_reach(OPEN_GAP_M) == pytest.approx(OPEN_REACH_M)
    assert close_reach(0.2) == pytest.approx(OPEN_REACH_M)


def test_a_narrow_part_gets_the_deepest_reach_of_the_whole_close():
    """A close onto a 10 mm part passes through the linkage's deepest point near a 30 mm gap."""
    assert close_reach(0.010) == pytest.approx(DEEPEST_REACH_M)
    assert close_reach(0.0) == pytest.approx(DEEPEST_REACH_M)


def test_the_reach_never_shrinks_as_the_part_narrows():
    """A narrower part means a longer close, so the pads reach at least as deep."""
    widths = np.linspace(0.0, OPEN_GAP_M, 50)
    reach = [close_reach(w) for w in widths]
    assert all(a >= b - 1e-9 for a, b in zip(reach, reach[1:]))


def test_the_width_is_the_extent_across_the_jaws():
    """part_width reads the optical-Y span, not X or depth."""
    pts = plane(0.12, x=(-0.03, 0.03), y=(-0.01, 0.01), step=0.0005)
    assert part_width(pts) == pytest.approx(0.018, abs=0.002)


def test_a_flat_table_under_a_bar_reads_the_bar_height():
    """A table 10 mm beyond the bar's near face is read as a 10 mm tall part."""
    p_obj = np.array([0.03, -0.01, 0.12])
    assert surface_height(plane(0.13), p_obj, 0.02, VISUAL_SERVO_TARGET_XY_M) == \
        pytest.approx(0.010, abs=1e-6)


def test_the_band_is_where_the_pads_end_their_close():
    """A deep surface outside the pads' landing band does not count; a near one inside does."""
    p_obj = np.array([0.0, 0.0, 0.12])
    tx = VISUAL_SERVO_TARGET_XY_M[0]
    # a step 30 mm beyond the part, inside the band across the jaws, and a
    # pit 100 mm deep beside it along the pads' X span
    landing = plane(0.15, x=(PAD_X_SPAN_M[0] - tx, PAD_X_SPAN_M[1] - tx), y=(0.012, 0.03))
    beside = plane(0.22, x=(0.03, 0.08), y=(0.012, 0.03))
    assert surface_height(np.vstack([landing, beside]), p_obj, 0.02,
                          VISUAL_SERVO_TARGET_XY_M) == pytest.approx(0.03, abs=1e-6)


def test_no_surface_in_the_band_reads_nothing():
    """Fewer than SURFACE_MIN_POINTS pixels in the landing band give no height."""
    p_obj = np.array([0.0, 0.0, 0.12])
    far = plane(0.13, x=(0.04, 0.08))
    assert surface_height(far, p_obj, 0.02, VISUAL_SERVO_TARGET_XY_M) is None
    assert surface_height(far[:SURFACE_MIN_POINTS - 1], p_obj, 0.02,
                          VISUAL_SERVO_TARGET_XY_M) is None


def test_a_short_bar_moves_the_setpoint_back_from_servo_depth():
    """A 10 mm bus bar cannot be grasped at servo_depth: the closing pads would hit the table."""
    floor = servo_depth_floor(0.010, 0.02)
    assert floor > VISUAL_SERVO_DEPTH_M
    # the pads end the clearance above the table and overlap the bar by the rest
    assert close_reach(0.02) == pytest.approx(
        floor + 0.010 - VISUAL_SERVO_SURFACE_CLEARANCE_M)


def test_a_tall_part_keeps_servo_depth():
    """A part taller than the fingers' sweep leaves servo_depth the deeper setpoint."""
    assert servo_depth_floor(0.08, 0.03) < VISUAL_SERVO_DEPTH_M


def test_an_inward_step_that_did_not_move_is_blocked():
    """A 10 mm inward step that covered 1 mm ran into something."""
    assert approach_blocked(0.010, 0.001)


def test_an_inward_step_that_moved_is_not_blocked():
    """A step covered to within the arm move's residual is free motion."""
    assert not approach_blocked(0.010, 0.008)


@pytest.mark.parametrize('commanded', [BLOCKED_MIN_STEP_M - 1e-4, 0.0, -0.01])
def test_small_or_outward_steps_never_count_as_blocked(commanded):
    """Steps below BLOCKED_MIN_STEP_M or backing off cannot tell contact from noise."""
    assert not approach_blocked(commanded, 0.0)


def test_the_tolerances_sit_above_the_measurement_noise():
    """X is no tighter than the +/-3 mm Gemini+SAM noise, X <= Y, and range below the clearance."""
    assert VISUAL_SERVO_X_TOL_M >= 0.003
    assert VISUAL_SERVO_X_TOL_M <= VISUAL_SERVO_Y_TOL_M
    assert VISUAL_SERVO_RANGE_TOL_M < VISUAL_SERVO_SURFACE_CLEARANCE_M
