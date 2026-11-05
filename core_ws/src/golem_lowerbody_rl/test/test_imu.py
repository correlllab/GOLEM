"""Test the IMU helpers: the mounting trim and gravity in the IMU frame."""

import numpy as np
import pytest

from golem_lowerbody_rl.imu import apply_trim, gravity_z, projected_gravity, trim_quaternion

ATOL = 1e-6

# unit quaternions (w, x, y, z): level, 10 deg roll, 30 deg pitch, a mixed tilt, and a
# 90 deg one
QUATS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.9961946980917455, 0.08715574274765817, 0.0, 0.0],
    [0.9659258262890683, 0.0, 0.25881904510252074, 0.0],
    [0.8223631719059994, 0.02226002732222905, 0.4396797395409, 0.3604234056503559],
    [0.5, -0.5, 0.5, 0.5],
]
# golem_lowerbody_rl.policy.gravity_from_quat_walk of each of QUATS
WALK_GRAVITY = [
    [0.0, -0.0, -1.0],
    [0.0, -0.1736481785774231, -0.9848077297210693],
    [0.5, -0.0, -0.8660253882408142],
    [0.7071067690849304, -0.3535533845424652, -0.6123724579811096],
    [1.0, -0.0, 0.0],
]
# golem_lowerbody_rl.policy.gravity_from_quat_fame of each of QUATS
FAME_GRAVITY = [
    [0.0, 0.0, -1.0],
    [0.0, -0.1736481785774231, -0.9848077893257141],
    [0.5, 0.0, -0.8660253882408142],
    [0.7071068286895752, -0.3535533845424652, -0.6123724579811096],
    [1.0, 0.0, 0.0],
]


@pytest.mark.parametrize('index', range(len(QUATS)))
def test_projected_gravity_matches_both_old_formulas_for_unit_quaternions(index):
    """For a unit quaternion, gravity is the WALK and the FAME formulas' value."""
    gravity = projected_gravity(np.array(QUATS[index]))
    np.testing.assert_allclose(gravity, WALK_GRAVITY[index], rtol=0, atol=ATOL)
    np.testing.assert_allclose(gravity, FAME_GRAVITY[index], rtol=0, atol=ATOL)
    assert gravity_z(np.array(QUATS[index])) == pytest.approx(FAME_GRAVITY[index][2], abs=ATOL)


@pytest.mark.parametrize('scale', [0.5, 0.97, 1.03, 3.0])
def test_non_unit_quaternion_is_normalised(scale):
    """A scaled quaternion gives the unit quaternion's gravity, a unit vector."""
    quat = np.array(QUATS[3]) * scale
    gravity = projected_gravity(quat)
    np.testing.assert_allclose(gravity, FAME_GRAVITY[3], rtol=0, atol=ATOL)
    assert np.linalg.norm(gravity) == pytest.approx(1.0, abs=ATOL)
    assert gravity_z(quat) == pytest.approx(FAME_GRAVITY[3][2], abs=ATOL)


@pytest.mark.parametrize('quat', [[0.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 1.0]])
def test_quaternion_that_is_no_rotation_is_refused(quat):
    """A zero or non-finite quaternion raises instead of giving a gravity."""
    with pytest.raises(ValueError, match='quaternion'):
        projected_gravity(np.array(quat))


# golem_lowerbody_rl.policy.imu_offset_quat of each trim (deg, converted with np.radians)
OLD_TRIMS = {
    (0.5, -2.5, 0.0): [0.9997524619102478, -0.004362270701676607, 0.021814676001667976,
                       -9.518508159089833e-05],
    (1.0, 2.0, 3.0): [0.9994710087776184, -0.008265383541584015, -0.01767415925860405,
                      -0.026019718497991562],
}
# golem_lowerbody_rl.policy.apply_imu_offset(float32 QUATS[3], each trim)
OLD_TRIMMED = {
    (0.5, -2.5, 0.0): [0.8126996159553528, 0.010762774385511875, 0.4559403657913208,
                       0.3626595139503479],
    (1.0, 2.0, 3.0): [0.8392612934112549, 0.0103809405118227, 0.4225127398967743,
                      0.34207576513290405],
}


@pytest.mark.parametrize('trim', sorted(OLD_TRIMS))
def test_trim_matches_the_old_node(trim):
    """The trim quaternion and the trimmed IMU reading are the H1-2 node's."""
    correction = trim_quaternion(*trim)
    np.testing.assert_allclose(correction, OLD_TRIMS[trim], rtol=0, atol=ATOL)
    trimmed = apply_trim(np.array(QUATS[3], dtype=np.float32), correction)
    np.testing.assert_allclose(trimmed, OLD_TRIMMED[trim], rtol=0, atol=ATOL)


def test_zero_trim_leaves_the_reading_unchanged():
    """A zero trim is the identity and returns the reading's own values."""
    correction = trim_quaternion(0.0, 0.0, 0.0)
    assert correction.tolist() == [1.0, 0.0, 0.0, 0.0]
    quat = np.array(QUATS[3])
    trimmed = apply_trim(quat, correction)
    assert trimmed.dtype == quat.dtype
    assert trimmed.tolist() == quat.tolist()


def test_trim_cancels_a_mounting_tilt():
    """A reading tilted by the trim's angles at true level reads level once trimmed."""
    trim = (0.5, -2.5, 0.0)
    roll, pitch = np.radians(trim[0]) / 2, np.radians(trim[1]) / 2
    # the raw reading at level is the mounting rotation Ry(pitch) Rx(roll)
    raw = np.array([np.cos(pitch) * np.cos(roll), np.cos(pitch) * np.sin(roll),
                    np.sin(pitch) * np.cos(roll), -np.sin(pitch) * np.sin(roll)])
    gravity = projected_gravity(apply_trim(raw, trim_quaternion(*trim)))
    np.testing.assert_allclose(gravity, [0.0, 0.0, -1.0], rtol=0, atol=ATOL)
