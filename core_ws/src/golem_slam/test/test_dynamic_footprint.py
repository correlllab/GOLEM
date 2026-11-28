"""Test the footprint geometry and the arm-goal check of golem_slam.dynamic_footprint."""
from action_msgs.msg import GoalStatus, GoalStatusArray
from golem_slam.dynamic_footprint import (arm_goal_live, body_only_reason, box_corners,
                                          convex_hull, footprint, moved, parse_polygon)
import numpy as np
import pytest

BODY = parse_polygon('[[0.45, 0.275], [0.45, -0.275], [-0.20, -0.275], [-0.20, 0.275]]')


def as_set(polygon):
    """Return a polygon's vertices as a set of rounded tuples."""
    return {tuple(np.round(p, 6)) for p in polygon}


def test_parse_polygon_reads_nav2s_string():
    """Nav2's footprint string becomes an (n, 2) array."""
    assert BODY.shape == (4, 2)
    assert BODY[0].tolist() == [0.45, 0.275]


@pytest.mark.parametrize(
    'text', ['[[0, 0], [1, 1]]', '[0, 1, 2]', '[[0, 0, 0], [1, 1, 1], [2, 0, 0]]'])
def test_parse_polygon_refuses_a_bad_polygon(text):
    """Fewer than three points, or points that are not [x, y], are refused."""
    with pytest.raises(ValueError):
        parse_polygon(text)


def test_convex_hull_drops_inner_points():
    """Points inside the square are not hull vertices."""
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    hull = convex_hull(square + [[0.5, 0.5], [0.2, 0.8]])
    assert as_set(hull) == as_set(np.array(square, dtype=float))


def test_arms_inside_the_body_keep_the_body():
    """Arm points inside the body polygon leave it unchanged."""
    arms = np.array([[0.1, 0.2, 1.0], [-0.1, -0.2, 0.5]])
    assert as_set(footprint(BODY, arms)) == as_set(BODY)


def test_an_arm_ahead_widens_the_footprint_forward():
    """An arm point ahead of the body becomes a vertex; the body's back stays."""
    arms = np.array([[0.8, 0.0, 1.0]])
    hull = footprint(BODY, arms)
    assert (0.8, 0.0) in as_set(hull)
    assert hull[:, 0].max() == pytest.approx(0.8)
    assert hull[:, 0].min() == pytest.approx(-0.20)


def test_no_arm_points_is_the_body():
    """With no arm points the footprint is the body polygon."""
    assert as_set(footprint(BODY, np.empty((0, 3)))) == as_set(BODY)


def test_box_corners_spans_the_box():
    """The 8 corners are every combination of the box's bounds."""
    corners = box_corners(np.array([0, 0, 0]), np.array([1, 2, 3]))
    assert corners.shape == (8, 3)
    assert corners.min(axis=0).tolist() == [0, 0, 0]
    assert corners.max(axis=0).tolist() == [1, 2, 3]


def test_moved_by_tolerance():
    """A vertex moved past the tolerance is a change; a smaller move is not."""
    shifted = BODY + [0.02, 0.0]
    assert not moved(BODY, shifted, 0.03)
    assert moved(BODY, BODY + [0.05, 0.0], 0.03)
    assert moved(None, BODY, 0.03)
    assert moved(BODY, BODY[:3], 0.03)


def status_array(*states):
    """Return a GoalStatusArray with one goal in each state."""
    msg = GoalStatusArray()
    msg.status_list = [GoalStatus(status=s) for s in states]
    return msg


def test_live_goals_drop_the_arms():
    """Accepted, executing and cancelling goals count as live; finished ones do not."""
    assert not arm_goal_live([])
    assert not arm_goal_live([status_array(GoalStatus.STATUS_SUCCEEDED,
                                           GoalStatus.STATUS_ABORTED,
                                           GoalStatus.STATUS_CANCELED)])
    for state in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING,
                  GoalStatus.STATUS_CANCELING):
        assert arm_goal_live([status_array(GoalStatus.STATUS_SUCCEEDED),
                              status_array(state)])


def test_the_arms_count_only_after_the_controller_starts_up():
    """Before the ready topic's first message, or during a live arm goal, the body is alone."""
    done = [status_array(GoalStatus.STATUS_SUCCEEDED)]
    live = [status_array(GoalStatus.STATUS_EXECUTING)]
    assert 'starting up' in body_only_reason(False, done, '/left_grasp_frame_pose')
    assert '/left_grasp_frame_pose' in body_only_reason(False, done, '/left_grasp_frame_pose')
    assert body_only_reason(True, live, '/left_grasp_frame_pose') == 'an arm goal is in flight'
    assert body_only_reason(True, done, '/left_grasp_frame_pose') is None
