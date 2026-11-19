"""Check mid360.CpuCaster, the Mac's lidar caster (GOLEM_RENDER=cpu), against MuJoCo's exact mj_ray.

The reference is exact_rays.ExactCaster (one mj_ray per ray). The scene is a small MJCF: a
closed room, a box obstacle, a hidden (alpha 0) box, and a robot body the sensor rides in, with
an arm the rays may hit first. Needs no GPU:
ROS_DOMAIN_ID=231 tests/robocasa/run_in_container.sh -p test_mid360_cpu.py
"""
import unittest

import mujoco
import numpy as np

import mid360
from exact_rays import ExactCaster

VISUAL = (0, 1, 0, 0, 0, 0)   # geomgroup: visual geoms only, as the G1 casts

SCENE = """
<mujoco>
  <worldbody>
    <geom type="box" group="1" pos="0 0 -1.05" size="3 3 0.05"/>
    <geom type="box" group="1" pos="0 0 2.05" size="3 3 0.05"/>
    <geom type="box" group="1" pos="3.05 0 0.5" size="0.05 3 1.6"/>
    <geom type="box" group="1" pos="-3.05 0 0.5" size="0.05 3 1.6"/>
    <geom type="box" group="1" pos="0 3.05 0.5" size="3 0.05 1.6"/>
    <geom type="box" group="1" pos="0 -3.05 0.5" size="3 0.05 1.6"/>
    <geom type="box" group="1" pos="1.2 0.8 0" size="0.3 0.2 0.6"/>
    <geom type="sphere" group="1" pos="-1.5 -1 0.3" size="0.4"/>
    <geom type="box" group="1" pos="0 -1.5 0" size="0.4 0.4 0.4" rgba="1 1 1 0"/>
    <body name="robot" pos="0 0 0">
      <geom name="torso" type="box" group="1" size="0.08 0.08 0.08"/>
      <body name="arm" pos="0 0.35 -0.1">
        <geom name="arm" type="capsule" group="1" fromto="0 0 0 0.3 0 -0.3" size="0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _scene():
    """Return (model, data, torso body id) of SCENE after mj_forward."""
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")


def _directions(n, seed=0):
    """Return n random unit directions (n, 3)."""
    d = np.random.default_rng(seed).normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


class CpuCasterTests(unittest.TestCase):
    """Check CpuCaster's ranges, robot returns and threading against ExactCaster."""

    def test_matches_mj_ray_ray_for_ray(self):
        """Every ray's range matches mj_ray's to 1e-9 m, and the same rays give no return."""
        model, data, torso = _scene()
        rays = _directions(4000)
        rotation = np.diag([1.0, -1.0, -1.0])   # upside down, as both robots hang the sensor
        expected = ExactCaster(model, torso, VISUAL).cast(np.zeros(3), rotation, rays, data)
        for threads in (1, 3):
            with self.subTest(threads=threads):
                caster = mid360.CpuCaster(mujoco.MjModel.from_xml_string(SCENE), torso, VISUAL,
                                          threads=threads)
                try:
                    got = caster.cast(np.zeros(3), rotation, rays, data)
                finally:
                    caster.close()
                np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
                np.testing.assert_allclose(got[~np.isnan(got)], expected[~np.isnan(expected)],
                                           atol=1e-9)

    def test_robot_hits_and_hidden_geoms_give_no_return(self):
        """A ray that meets the robot's arm first, or only a hidden geom then nothing, gives NaN."""
        model, data, torso = _scene()
        caster = mid360.CpuCaster(model, torso, VISUAL, threads=1)
        try:
            arm = np.array([[0.0, 1.0, -0.15]]) / np.linalg.norm([0.0, 1.0, -0.15])
            wall = np.array([[1.0, 0.0, 0.0]])
            hidden = np.array([[0.0, -1.0, 0.0]])
            got = caster.cast(np.zeros(3), np.eye(3), np.vstack([arm, wall, hidden]), data)
        finally:
            caster.close()
        self.assertTrue(np.isnan(got[0]), got)
        self.assertAlmostEqual(got[1], 3.0, places=9)   # the +x wall's inner face
        self.assertAlmostEqual(got[2], 3.0, places=9)   # through the hidden box to the -y wall


class MakeCasterTests(unittest.TestCase):
    """Check make_caster picks the caster by GOLEM_RENDER's value."""

    def test_cpu_builds_a_cpu_caster_on_a_copy_of_the_model(self):
        """render cpu gives a CpuCaster casting on its own copy, so the physics may step the original."""
        model, _, torso = _scene()
        caster = mid360.make_caster(model, torso, VISUAL, "cpu")
        try:
            self.assertIsInstance(caster, mid360.CpuCaster)
            self.assertIsNot(caster.model, model)
        finally:
            caster.close()

    def test_cpu_casts_on_the_threads_given(self):
        """make_caster's threads sets the CpuCaster's thread count; None is DEFAULT_THREADS."""
        model, _, torso = _scene()
        for threads, expected in ((3, 3), (None, mid360.DEFAULT_THREADS)):
            with self.subTest(threads=threads):
                caster = mid360.make_caster(model, torso, VISUAL, "cpu", threads=threads)
                try:
                    self.assertEqual(caster.threads, expected)
                finally:
                    caster.close()

    def test_points_come_from_golem_lidar_points(self):
        """GOLEM_LIDAR_POINTS sets the points per frame; unset or empty is the real 20,000; bad values raise."""
        self.assertEqual(mid360.points_from_env({}), 20000)
        self.assertEqual(mid360.points_from_env({"GOLEM_LIDAR_POINTS": ""}), 20000)
        self.assertEqual(mid360.points_from_env({"GOLEM_LIDAR_POINTS": "10000"}), 10000)
        for bad in ("0", "-5", "many", "1e4"):
            with self.subTest(GOLEM_LIDAR_POINTS=bad):
                with self.assertRaisesRegex(ValueError, "GOLEM_LIDAR_POINTS"):
                    mid360.points_from_env({"GOLEM_LIDAR_POINTS": bad})

    def test_an_unknown_render_is_refused(self):
        """A render other than gpu or cpu raises ValueError, naming it."""
        model, _, torso = _scene()
        with self.assertRaisesRegex(ValueError, "metal"):
            mid360.make_caster(model, torso, VISUAL, "metal")


if __name__ == "__main__":
    unittest.main()
