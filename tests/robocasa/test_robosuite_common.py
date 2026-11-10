"""Check robosuite_common's placement helpers, which both robots' spawn searches build on.

The tests use spawn_scene's small MjModel (a robot0_-prefixed free body and a fixed counter) and
its stand-in env, so they need only mujoco and numpy.
"""
import math
import unittest

import mujoco
import numpy as np

import robosuite_common as rc
from spawn_scene import FakeEnv, SCENE, free_qadr, load

# SCENE with a visual box (contype and conaffinity 0) on the robot reaching 5 cm below its
# colliding hand box, as the G1's foot meshes reach below its collision spheres.
VISUAL_SCENE = SCENE.replace(
    '<geom name="robot0_hand"',
    '<geom name="robot0_visual" type="box" pos="0.15 0 -0.05" size="0.05 0.05 0.1" contype="0" '
    'conaffinity="0"/><geom name="robot0_hand"')


class BackwardDirTests(unittest.TestCase):
    """Check backward_dir: the level unit vector behind the robot."""

    def test_identity(self):
        """Backward is world -x for a robot at yaw 0, facing +x."""
        b = rc.backward_dir(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(b, [-1.0, 0.0, 0.0], atol=1e-6)

    def test_yaw_90(self):
        """Backward is world -y, unit and level, for a robot yawed +90 deg."""
        q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])  # wxyz
        b = rc.backward_dir(q)
        np.testing.assert_allclose(b, [0.0, -1.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(b)), 1.0, places=6)
        self.assertLess(abs(b[2]), 1e-9)


class RobotEnvContactsTests(unittest.TestCase):
    """Check robot_env_contacts: the robot's penetrating contacts with the environment."""

    def test_detected(self):
        """An overlapping hand and counter give a robot-environment contact."""
        m, d = load()
        self.assertGreaterEqual(len(rc.robot_env_contacts(m, d)), 1)

    def test_clear_when_moved_back(self):
        """No contact remains once the base is 0.5 m back."""
        m, d = load()
        d.qpos[free_qadr(m)] = -0.5
        mujoco.mj_forward(m, d)
        self.assertEqual(rc.robot_env_contacts(m, d), [])


class FloorClearanceTests(unittest.TestCase):
    """Check lowest_robot_point_z and place_robot_clear, with and without the visual geoms."""

    def test_lowest_point_counts_visual_geoms_unless_colliding_only(self):
        """The visual box sets the lowest point, unless colliding_only skips it for the hand box."""
        m, d = load(VISUAL_SCENE)
        env = FakeEnv(m, d)
        self.assertAlmostEqual(rc.lowest_robot_point_z(env), 1.0 - 0.15, places=6)
        self.assertAlmostEqual(rc.lowest_robot_point_z(env, colliding_only=True), 1.0 - 0.1, places=6)

    def test_place_robot_clear_lifts_the_chosen_lowest_point_to_the_clearance(self):
        """place_robot_clear puts the lowest geom it counts the clearance above z = 0."""
        for colliding_only in (False, True):
            m, d = load(VISUAL_SCENE)
            env = FakeEnv(m, d)
            rc.place_robot_clear(env, [-1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], 0.02, colliding_only)
            self.assertAlmostEqual(rc.lowest_robot_point_z(env, colliding_only), 0.02, places=6)
            self.assertAlmostEqual(float(d.qpos[free_qadr(m)]), -1.0, places=9)


class EulerToWxyzTests(unittest.TestCase):
    """Check euler_to_wxyz on the orientation forms RoboCasa and the URDF give."""

    def test_forms(self):
        """None is the identity, a yaw scalar and [0, 0, yaw] agree, and a quaternion passes through."""
        np.testing.assert_allclose(rc.euler_to_wxyz(None), [1.0, 0.0, 0.0, 0.0])
        half = math.sqrt(0.5)
        np.testing.assert_allclose(rc.euler_to_wxyz(math.pi / 2), [half, 0.0, 0.0, half], atol=1e-12)
        np.testing.assert_allclose(rc.euler_to_wxyz([0.0, 0.0, math.pi / 2]), [half, 0.0, 0.0, half],
                                   atol=1e-12)
        np.testing.assert_allclose(rc.euler_to_wxyz([0.0, 1.0, 0.0, 0.0]), [0.0, 1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
