"""Test the Dex3-1 hand logic in dex_3.dex3: joints, poses, mode byte, timing and domain."""

import numpy as np
import pytest

from dex_3.dex3 import (
    DEFAULT_MAX_SPEED,
    HAND_JOINTS,
    JOINT_NAMES,
    LOWER,
    NUM_HAND_MOTORS,
    SIDES,
    UPPER,
    PoseInterpolator,
    dds_domain,
    motor_mode,
    move_duration,
    named_pose,
    validate_positions,
)


@pytest.mark.parametrize('side', SIDES)
def test_hand_table_is_in_handcmd_order(side):
    """The joint table lists seven joints in HandCmd_ order, each with lower < upper."""
    order = ['thumb_0', 'thumb_1', 'thumb_2', 'middle_0', 'middle_1', 'index_0', 'index_1']
    assert len(HAND_JOINTS[side]) == NUM_HAND_MOTORS == 7
    assert JOINT_NAMES[side] == [f'{side}_hand_{joint}_joint' for joint in order]
    assert np.all(LOWER[side] < UPPER[side])


def test_hands_mirror_except_thumb_0():
    """The right hand's limits mirror the left's, except thumb_0's, which are equal."""
    for left, right in zip(HAND_JOINTS['left'], HAND_JOINTS['right']):
        if left[0].endswith('thumb_0_joint'):
            assert left[1:] == right[1:]
        else:
            assert right[1:] == (-left[2], -left[1]), left[0]


def test_motor_mode_matches_unitrees_example_encoding():
    """The mode byte has the index in bits 0-3, status 1 in bits 4-6 and timeout in bit 7."""
    assert motor_mode(0) == 0x10
    assert motor_mode(6) == 0x16
    assert motor_mode(3, timeout=True) == 0x93
    with pytest.raises(ValueError):
        motor_mode(7)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_named_poses_are_inside_the_limits(side):
    """Open and close are inside the URDF limits, and open is all zeros."""
    for name in ('open', 'close'):
        q = named_pose(side, name)
        assert q.shape == (7,)
        assert np.all(q >= LOWER[side]) and np.all(q <= UPPER[side]), name
    assert np.all(named_pose(side, 'open') == 0.0)


def test_close_curls_the_fingers_the_right_way_and_mirrors():
    """Close curls the fingers and thumb toward each other, mirrored between the hands."""
    left, right = named_pose('left', 'close'), named_pose('right', 'close')
    assert np.all(left[3:] < 0) and np.all(right[3:] > 0)
    assert left[1] > 0 and left[2] > 0
    assert np.allclose(right, -left)


def test_validate_clips_and_rejects_bad_input():
    """Positions are clipped to the limits; a wrong length, NaN or side is refused."""
    q = validate_positions('left', [5.0, 0, 0, 0, 0, 0, -5.0])
    assert q[0] == UPPER['left'][0] and q[6] == LOWER['left'][6]
    with pytest.raises(ValueError, match='7 joint positions'):
        validate_positions('left', [0.0] * 6)
    with pytest.raises(ValueError, match='finite'):
        validate_positions('right', [0, 0, 0, np.nan, 0, 0, 0])
    with pytest.raises(ValueError, match='side'):
        validate_positions('middle', [0.0] * 7)


def test_interpolator():
    """The interpolator moves linearly over its duration and jumps when it is zero."""
    interp = PoseInterpolator(np.zeros(7), np.ones(7), duration=2.0, t_start=10.0)
    assert np.allclose(interp.value(10.0), 0.0)
    assert np.allclose(interp.value(11.0), 0.5)
    assert np.allclose(interp.value(20.0), 1.0)
    assert np.allclose(PoseInterpolator(np.zeros(7), np.ones(7)).value(0.0), 1.0)


def test_move_duration_caps_the_joint_speed():
    """A move is stretched to the speed cap, and a slower requested duration stands."""
    start, goal = np.zeros(7), np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.5])
    assert move_duration(start, goal, 0.0, 2.0) == pytest.approx(0.75)
    assert move_duration(start, goal, 0.5, 2.0) == pytest.approx(0.75)
    assert move_duration(start, goal, 2.0, 2.0) == pytest.approx(2.0)
    assert move_duration(start, start, 0.0, 2.0) == 0.0


def test_move_duration_rejects_bad_durations():
    """A NaN, infinite or negative duration, which would interpolate to NaN, is refused."""
    assert np.all(np.isnan(PoseInterpolator(np.zeros(7), np.ones(7), float('nan')).value(1.0)))
    for bad in (float('nan'), float('inf'), -0.1):
        with pytest.raises(ValueError, match='duration'):
            move_duration(np.zeros(7), np.ones(7), bad, 2.0)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_default_speed_cap_never_stretches_open_and_close(side):
    """Open and close over 1 s are never stretched by the default speed cap."""
    # Each joint is farthest from its goal at one of its limits, so starting
    # with every joint at its lower, then its upper, limit covers the worst case.
    for name in ('open', 'close'):
        for start in (LOWER[side], UPPER[side]):
            assert move_duration(start, named_pose(side, name), 1.0, DEFAULT_MAX_SPEED) == 1.0


def test_dds_domain_comes_from_ros_domain_id_or_the_parameter():
    """The domain comes from ROS_DOMAIN_ID or the parameter; none, a clash or junk is refused."""
    assert dds_domain('1', -1) == 1
    assert dds_domain('0', -1) == 0
    assert dds_domain(None, 97) == 97
    assert dds_domain('', 97) == 97
    assert dds_domain('97', 97) == 97
    for unset in (None, ''):
        with pytest.raises(ValueError, match='no DDS domain'):
            dds_domain(unset, -1)
    with pytest.raises(ValueError, match='disagree'):
        dds_domain('1', 0)
    for bad in ('-1', 'one', ' 1', '0x0', '1.0'):
        with pytest.raises(ValueError, match='ROS_DOMAIN_ID must be'):
            dds_domain(bad, -1)


def test_dds_domain_must_be_the_nodes_ros_domain():
    """A domain other than the one the node's ROS side runs on is refused."""
    assert dds_domain('1', -1, ros_domain=1) == 1
    assert dds_domain(None, 97, ros_domain=97) == 97
    # With ROS_DOMAIN_ID unset ROS runs on domain 0, so domain_id:=1 would split them.
    with pytest.raises(ValueError,
                       match='runs ROS on domain 0 but would run Unitree DDS on domain 1'):
        dds_domain(None, 1, ros_domain=0)
    # rcl reads '010' as octal 8.
    with pytest.raises(ValueError, match='runs ROS on domain 8'):
        dds_domain('010', -1, ros_domain=8)


def test_launch_defaults_are_the_drivers():
    """hands.launch.py's kp, kd and max_speed default to the driver's own defaults."""
    import importlib.util
    from pathlib import Path

    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    from dex_3.dex3 import DEFAULT_KD, DEFAULT_KP

    path = Path(__file__).resolve().parents[1] / 'launch' / 'hands.launch.py'
    spec = importlib.util.spec_from_file_location('hands_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    defaults = {
        entity.name: ''.join(part.perform(context) for part in entity.default_value)
        for entity in module.generate_launch_description().entities
        if isinstance(entity, DeclareLaunchArgument)}
    assert float(defaults['kp']) == DEFAULT_KP
    assert float(defaults['kd']) == DEFAULT_KD
    assert float(defaults['max_speed']) == DEFAULT_MAX_SPEED
