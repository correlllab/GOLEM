"""Check golem_robocasa/mj_compat.py: each call behaves the same on the installed MuJoCo.

The x86 image runs MuJoCo 3.17.0 and the Mac's 3.3.1; run this on both (CLAUDE.md §8). The
scenes are small MJCFs built here. Needs MuJoCo and NumPy; the rendering check skips without an
offscreen GL context (MUJOCO_GL=egl or osmesa). Run it with
ROS_DOMAIN_ID=231 tests/robocasa/run_in_container.sh -p test_mj_compat.py
"""
import unittest

import mujoco
import numpy as np

import mj_compat

# Spheres (no box, so mj_multiRay's bounding-box culling cannot differ from mj_ray), one of them
# in geom group 2, which the casts below leave out, one on the excluded body "self" around the
# origin and one past the cutoff. The scene is open above, so some rays hit nothing.
_SCENE = """
<mujoco>
  <worldbody>
    <body name="self"><geom type="sphere" size="0.2" group="1"/></body>
    <geom type="sphere" group="1" pos="2 0 0" size="0.5"/>
    <geom type="sphere" group="1" pos="0 3 0.5" size="0.8"/>
    <geom type="sphere" group="1" pos="-4 -1 -1" size="1.0"/>
    <geom type="sphere" group="1" pos="0 0 -16" size="12"/>
    <geom type="sphere" group="1" pos="25 0 12" size="4"/>
    <geom type="sphere" group="2" pos="0 -2 0" size="0.6"/>
    <geom type="sphere" group="1" pos="0 -5 0" size="1.5"/>
  </worldbody>
</mujoco>
"""


def _directions(n, seed=3):
    """Return n unit directions (n, 3), spread over the sphere."""
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class MultiRayTests(unittest.TestCase):
    """Check multi_ray against mj_ray, ray by ray."""

    def test_multi_ray_same_on_both_versions(self):
        """multi_ray gives every ray mj_ray's geom and distance, and none past the cutoff."""
        model = mujoco.MjModel.from_xml_string(_SCENE)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        origin = np.zeros(3)
        geomgroup = np.array([0, 1, 0, 0, 0, 0], np.uint8)
        exclude = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "self")
        cutoff = 20.0   # m: the sphere at (25, 0, 12) is past it
        world = _directions(2000)
        geomid = np.empty(len(world), np.int32)
        dist = np.empty(len(world))
        mj_compat.multi_ray(model, data, origin.reshape(3, 1), world.reshape(-1, 1), geomgroup, 1,
                            exclude, geomid.reshape(-1, 1), dist.reshape(-1, 1), len(world), cutoff)

        one = np.empty((1, 1), np.int32)
        expected_id = np.empty(len(world), np.int32)
        expected = np.empty(len(world))
        for i, vec in enumerate(world):
            d = mujoco.mj_ray(model, data, origin.reshape(3, 1), vec.reshape(3, 1), geomgroup, 1,
                              exclude, one)
            hit = one[0, 0] >= 0 and d <= cutoff
            expected_id[i] = one[0, 0] if hit else -1
            expected[i] = d if hit else -1.0
        hits = expected_id >= 0
        self.assertGreater(hits.sum(), 200)
        self.assertGreater((~hits).sum(), 200)
        # Every group-1 sphere but "self" and the one past the cutoff.
        self.assertEqual(len(set(expected_id[hits].tolist())), 5)
        np.testing.assert_array_equal(geomid, expected_id)
        np.testing.assert_allclose(dist[hits], expected[hits], rtol=0.0, atol=1e-9)
        np.testing.assert_array_equal(dist[~hits], -1.0)


class PrincipalPixelTests(unittest.TestCase):
    """Check principalpixel against a rendered marker."""

    def test_rendered_marker_lands_at_the_pinhole_projection(self):
        """A camera given principalpixel's values renders a marker where the pinhole (fx, fy, cx, cy) projects it."""
        width, height, fx, fy = 320, 180, 150.0, 140.0
        for cx, cy in ((150.3, 82.7), (171.0, 97.4)):
            px, py = mj_compat.principalpixel(width, height, cx, cy)
            # Camera at the origin looking along -z, image +x along world +x and +y along world -y.
            xml = f"""<mujoco><visual><global offwidth="{width}" offheight="{height}"/>
              <headlight ambient="1 1 1"/></visual><worldbody>
              <camera name="c" pos="0 0 0" xyaxes="1 0 0 0 1 0" resolution="{width} {height}"
                      sensorsize="1 1" focalpixel="{fx} {fy}" principalpixel="{px:.9g} {py:.9g}"/>
              <geom name="m" type="sphere" size="0.03" pos="0.11 0.07 -1" rgba="1 0 0 1"/>
            </worldbody></mujoco>"""
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            try:
                renderer = mujoco.Renderer(model, height, width)
            except Exception as e:  # no offscreen GL context in this environment
                self.skipTest(f"no offscreen renderer: {e}")
            try:
                renderer.update_scene(data, camera="c")
                img = renderer.render().astype(int)
            finally:
                renderer.close()
            ys, xs = np.nonzero((img[:, :, 0] > 120) & (img[:, :, 1] < 60))
            self.assertGreater(len(xs), 5)
            # Optical frame: x right = world x, y down = world -y, z forward = world -z (1 m).
            self.assertAlmostEqual(xs.mean(), fx * 0.11 + cx, delta=0.5)
            self.assertAlmostEqual(ys.mean(), fy * -0.07 + cy, delta=0.5)


class JointTypeTests(unittest.TestCase):
    """Check joint_type."""

    def test_joint_type_keys_a_dict_of_mjtjoint(self):
        """joint_type is an int that finds each mjtJoint key of a dict and equals the enum both ways."""
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <body><freejoint/><geom size="0.1"/></body>
          <body><joint type="ball"/><geom size="0.1"/></body>
          <body><joint type="slide"/><geom size="0.1"/></body>
          <body><joint type="hinge"/><geom size="0.1"/></body>
        </worldbody></mujoco>""")
        kinds = (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL,
                 mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)
        table = {kind: kind for kind in kinds}
        for j, kind in enumerate(kinds):
            got = mj_compat.joint_type(model, j)
            self.assertIs(type(got), int)
            self.assertIs(table[got], kind)
            self.assertTrue(kind == got and got == kind)


if __name__ == "__main__":
    unittest.main()
