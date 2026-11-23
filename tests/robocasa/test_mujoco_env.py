"""Check the elastic band's force (apply_band), its hoist (BandLowering) and the fall logger both sim loops run.

The closed-loop tests read the fall logger's lines (tests/sim, tests/g1), so their text is
checked here. Needs mujoco and numpy only.
"""
import contextlib
import io
import types
import unittest

import mujoco
import numpy as np

from mujoco_env import BandLowering, ElasticBand, FallLogger, apply_band

# One free body, 1 m up.
SCENE = '<mujoco><worldbody><body name="b" pos="0 0 1"><freejoint/><geom size="0.1"/></body></worldbody></mujoco>'
# One free body, 1 m up and turned 90 deg about z, its centre of mass 0.1 m off its origin along its x.
OFFSET_SCENE = ('<mujoco><worldbody><body name="b" pos="0 0 1" euler="0 0 90"><freejoint/>'
                '<inertial pos="0.1 0 0" mass="2" diaginertia="0.1 0.1 0.1"/></body></worldbody></mujoco>')


def _state(z, upz):
    """Return a stand-in MjData of one body (id 1) at height z whose z axis . world up is upz."""
    xmat = np.zeros((2, 9))
    xmat[1, 8] = upz
    xpos = np.zeros((2, 3))
    xpos[1, 2] = z
    return types.SimpleNamespace(time=2.5, xmat=xmat, xpos=xpos)


class ApplyBandTests(unittest.TestCase):
    """Check apply_band's force on the body."""

    def test_pulls_while_enabled_and_zeroes_once_released(self):
        """An enabled band writes its spring force; a released one zeroes the force and leaves the torque."""
        model = mujoco.MjModel.from_xml_string(SCENE)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        band = ElasticBand(np.array([0.0, 0.0, 1.5]), length=0, stiffness=100.0, damping=10.0)
        apply_band(model, data, band, 1, np.zeros(6))
        np.testing.assert_allclose(data.xfrc_applied[1, :3], [0.0, 0.0, 50.0])
        data.xfrc_applied[1, 3:] = 7.0   # e.g. the viewer's mouse torque
        band.enabled = False
        apply_band(model, data, band, 1, np.zeros(6))
        np.testing.assert_array_equal(data.xfrc_applied[1], [0.0, 0.0, 0.0, 7.0, 7.0, 7.0])


    def test_rope_never_pulls_down(self):
        """A rope band's vertical force is clamped at 0; its sideways pull is the plain band's."""
        rope = ElasticBand(np.array([0.0, 0.0, 1.0]), length=0, stiffness=100.0, damping=10.0, rope=True)
        plain = ElasticBand(np.array([0.0, 0.0, 1.0]), length=0, stiffness=100.0, damping=10.0)
        above = (np.array([0.2, 0.0, 1.5]), np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(plain.evaluate_force(*above), [-20.0, 0.0, -40.0])
        np.testing.assert_allclose(rope.evaluate_force(*above), [-20.0, 0.0, 0.0])
        below = (np.array([0.0, 0.1, 0.5]), np.zeros(3))
        np.testing.assert_allclose(rope.evaluate_force(*below), plain.evaluate_force(*below))

    def test_hooked_band_pulls_its_hook_and_turns_the_body(self):
        """A hooked band pulls the hook's point at its velocity and writes the torque about the centre of mass."""
        model = mujoco.MjModel.from_xml_string(OFFSET_SCENE)
        data = mujoco.MjData(model)
        data.qvel[:] = [0.1, 0.0, 0.0, 0.0, 0.0, 2.0]   # world linear, then the body's angular velocity
        mujoco.mj_forward(model, data)
        band = ElasticBand(np.array([0.0, 0.0, 1.5]), length=0, stiffness=100.0, damping=10.0,
                           hook=(0.0, 0.0, 0.3))
        hook = np.array([0.0, 0.0, 1.3])   # the body's origin is at (0, 0, 1): the hook is straight above
        np.testing.assert_allclose(band.hook_point(data, 1), hook, atol=1e-12)
        com = data.xipos[1].copy()   # (0, 0.1, 1): the body's x points along world y
        np.testing.assert_allclose(com, [0.0, 0.1, 1.0], atol=1e-12)
        # The hook moves at v_com + w x (hook - com); the free joint's linear velocity is the origin's.
        v_hook = np.array([0.1, 0.0, 0.0]) + np.cross([0.0, 0.0, 2.0], hook - [0.0, 0.0, 1.0])
        force = 100.0 * (np.array([0.0, 0.0, 1.5]) - hook) - 10.0 * v_hook
        apply_band(model, data, band, 1, np.zeros(6))
        np.testing.assert_allclose(data.xfrc_applied[1, :3], force, atol=1e-9)
        np.testing.assert_allclose(data.xfrc_applied[1, 3:], np.cross(hook - com, force), atol=1e-9)
        band.enabled = False
        apply_band(model, data, band, 1, np.zeros(6))
        np.testing.assert_array_equal(data.xfrc_applied[1], np.zeros(6))
        data.xfrc_applied[1, 3:] = 7.0   # the viewer's mouse torque after the release is kept
        apply_band(model, data, band, 1, np.zeros(6))
        np.testing.assert_array_equal(data.xfrc_applied[1], [0.0, 0.0, 0.0, 7.0, 7.0, 7.0])

    def test_an_unhooked_band_writes_no_torque(self):
        """hook None pulls the body's origin and leaves the torque, as the fixed band always has."""
        model = mujoco.MjModel.from_xml_string(OFFSET_SCENE)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        data.xfrc_applied[1, 3:] = 7.0
        band = ElasticBand(np.array([0.0, 0.0, 1.5]), length=0, stiffness=100.0, damping=10.0)
        np.testing.assert_array_equal(band.hook_point(data, 1), data.xpos[1])
        apply_band(model, data, band, 1, np.zeros(6))
        np.testing.assert_allclose(data.xfrc_applied[1], [0.0, 0.0, 50.0, 7.0, 7.0, 7.0])


class _Latch:
    """Stand in for a SimInterface: holds() is true for the slots in held."""

    def __init__(self):
        """Hold no slot."""
        self.held = set()
        self.asked = []

    def holds(self, slots):
        """Return whether every slot is held, recording the question."""
        self.asked.append(list(slots))
        return set(slots) <= self.held


class BandLoweringTests(unittest.TestCase):
    """Check the anchor's lift and lowering."""

    def setUp(self):
        """A band anchored at z 1.0: up 0.1 m over 1 s; down 0.3 m over 2 s, 0.5 s after slots 0-2 are held."""
        self.band = ElasticBand(np.array([0.0, 0.0, 1.0]), rope=True)
        self.lowering = BandLowering(self.band, lift=0.1, lift_duration=1.0, stages=((range(3), 0.3, 0.5, 2.0),))
        self.latch = _Latch()

    def at(self, t):
        """Step the lowering at sim time t (s), quietly; return the anchor's z."""
        with contextlib.redirect_stdout(io.StringIO()):
            self.lowering.step(t, self.latch)
        return float(self.band.point[2])

    def test_lifts_then_waits_for_every_leg(self):
        """The anchor rises linearly from the first step and stays up until every leg is held."""
        self.assertAlmostEqual(self.at(3.0), 1.0)   # the first step: the lift starts here
        self.assertAlmostEqual(self.at(3.5), 1.05)
        self.assertAlmostEqual(self.at(4.0), 1.1)
        self.latch.held = {0, 1}
        self.assertAlmostEqual(self.at(9.0), 1.1)
        self.assertEqual(self.latch.asked[-1], [0, 1, 2])
        np.testing.assert_array_equal(self.band.point[:2], [0.0, 0.0])

    def test_lowers_after_the_delay_and_stays_down(self):
        """Once the legs are held the anchor waits delay, comes down linearly over duration, and stays."""
        self.at(0.0)
        self.latch.held = {0, 1, 2}
        self.assertAlmostEqual(self.at(2.0), 1.1)    # held at 2 s: down from 2.5 s
        self.assertAlmostEqual(self.at(2.5), 1.1)
        self.assertAlmostEqual(self.at(3.5), 0.95)
        self.assertAlmostEqual(self.at(4.5), 0.8)
        self.assertTrue(self.lowering.done)
        self.latch.held = set()   # the legs let go: the anchor stays down
        self.assertAlmostEqual(self.at(9.0), 0.8)

    def test_a_release_stops_it(self):
        """A release mid-way stops the anchor where it is, for good."""
        self.at(0.0)
        self.latch.held = {0, 1, 2}
        self.at(1.0)
        self.assertAlmostEqual(self.at(2.5), 1.1 - 0.15)
        self.band.enabled = False
        self.assertAlmostEqual(self.at(2.6), 1.1 - 0.15)
        self.band.enabled = True
        self.assertAlmostEqual(self.at(5.0), 1.1 - 0.15)
        self.assertTrue(self.lowering.done)

    def test_logs_the_hold_and_the_end(self):
        """The hold and the end of the lowering are logged once each."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.lowering.step(0.0, self.latch)
            self.latch.held = {0, 1, 2}
            for t in (1.0, 1.5, 4.0, 5.0):
                self.lowering.step(t, self.latch)
        self.assertEqual(out.getvalue().splitlines(), [
            "[elastic-band] t=1.00s: motors 0-2 are held; the anchor comes down 0.300 m over 2.0s from t=1.50s",
            "[elastic-band] t=4.00s: anchor lowered to z 0.800 m; it stays there until the band is released",
        ])


class StagedLoweringTests(unittest.TestCase):
    """Check a lowering in two stages, each on its own motors."""

    def test_the_second_stage_waits_for_its_motors_and_the_first(self):
        """The second stage starts once its motors are held, and not before the first has ended."""
        band = ElasticBand(np.array([0.0, 0.0, 1.0]), rope=True)
        lowering = BandLowering(band, lift=0.0, lift_duration=1.0,
                                stages=((range(2), 0.2, 0.0, 2.0), (range(3), 0.1, 0.0, 1.0)))
        self.assertAlmostEqual(lowering.drop, 0.3)
        latch = _Latch()
        latch.held = {0, 1, 2}   # both stages' motors at once: the second follows the first
        with contextlib.redirect_stdout(io.StringIO()):
            for t in np.arange(0.0, 3.0, 0.5):
                lowering.step(float(t), latch)
            self.assertAlmostEqual(float(band.point[2]), 0.8 - 0.05)   # t 2.5: the second half way
            lowering.step(3.0, latch)
        self.assertAlmostEqual(float(band.point[2]), 0.7)
        self.assertTrue(lowering.done)

    def test_the_second_stage_waits_for_its_own_motors(self):
        """With only the first stage's motors held the anchor stops after the first stage."""
        band = ElasticBand(np.array([0.0, 0.0, 1.0]), rope=True)
        lowering = BandLowering(band, lift=0.0, lift_duration=1.0,
                                stages=((range(2), 0.2, 0.0, 1.0), (range(3), 0.1, 0.0, 1.0)))
        latch = _Latch()
        latch.held = {0, 1}
        with contextlib.redirect_stdout(io.StringIO()):
            for t in (0.0, 1.0, 5.0):
                lowering.step(t, latch)
        self.assertAlmostEqual(float(band.point[2]), 0.8)
        self.assertFalse(lowering.done)


class FallLoggerTests(unittest.TestCase):
    """Check what the fall logger logs, and when."""

    def lines(self, logger, data, band):
        """Return the lines one step logs."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            logger.step(data, band)
        return out.getvalue().splitlines()

    def test_logs_the_release_and_the_first_fall_once(self):
        """The release is logged once, then the first fall with how long the robot stood after it."""
        logger = FallLogger(1, "pelvis", upright_min=0.5, height_min=0.2)
        band = ElasticBand(np.zeros(3))
        self.assertEqual(self.lines(logger, _state(0.7, 1.0), band), [])
        band.enabled = False
        self.assertEqual(self.lines(logger, _state(0.7, 1.0), band),
                         ["[fall-logger] elastic band released at t=2.50s"])
        self.assertEqual(self.lines(logger, _state(0.7, 1.0), band), [])
        fell = _state(0.1, 1.0)
        fell.time = 4.0
        self.assertEqual(self.lines(logger, fell, band),
                         ["[fall-logger] *** ROBOT FELL at sim t=4.00s — stood 1.50s after band release "
                          "(uprightness=1.00, pelvis_h=0.10) ***"])
        self.assertEqual(self.lines(logger, fell, band), [])

    def test_a_tilt_is_a_fall_under_the_band_too(self):
        """A tilt past upright_min is a fall, logged with the band still on."""
        logger = FallLogger(1, "torso", upright_min=0.5, height_min=0.5)
        self.assertEqual(self.lines(logger, _state(1.0, 0.4), ElasticBand(np.zeros(3))),
                         ["[fall-logger] *** ROBOT FELL at sim t=2.50s — stood 2.50s (band still on) "
                          "(uprightness=0.40, torso_h=1.00) ***"])

    def test_no_body_and_no_band_log_nothing(self):
        """Body -1 watches no fall, and no band logs no release."""
        logger = FallLogger(-1, "torso", upright_min=0.5, height_min=0.5)
        self.assertEqual(self.lines(logger, _state(0.0, 0.0), None), [])


if __name__ == "__main__":
    unittest.main()
