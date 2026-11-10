"""Check the elastic band's force (apply_band) and the fall logger both sim loops run.

The closed-loop tests read the fall logger's lines (tests/sim, tests/g1), so their text is
checked here. Needs mujoco and numpy only.
"""
import contextlib
import io
import types
import unittest

import mujoco
import numpy as np

from mujoco_env import ElasticBand, FallLogger, apply_band

# One free body, 1 m up.
SCENE = '<mujoco><worldbody><body name="b" pos="0 0 1"><freejoint/><geom size="0.1"/></body></worldbody></mujoco>'


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
