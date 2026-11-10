"""Check the H1-2's spawn search, h1_2_robosuite.place_robot_collision_free: a margin of free space to the fixtures.

The tests use spawn_scene's small MjModel (a robot0_-prefixed free body and a fixed counter) and
its stand-in env, so they need only mujoco and numpy (and robosuite, or spawn_scene's stand-ins,
for h1_2_robosuite's model classes).
"""
import io
import math
import re
import unittest
from contextlib import redirect_stdout

import mujoco
import numpy as np

import robosuite_common as rc
from spawn_scene import FakeEnv, SCENE, free_qadr, import_robot_module, load, total_penetration

hr = import_robot_module("h1_2_robosuite")

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
# SCENE with a second robot geom behind the hand, in the same body.
TWO_GEOMS = SCENE.replace(
    '<geom name="robot0_hand"',
    '<geom name="robot0_back" type="box" pos="-0.1 0 0" size="0.1 0.1 0.1"/><geom name="robot0_hand"')
# TWO_GEOMS with a wall 0.55 m behind the spawn.
BLOCKED = TWO_GEOMS.replace(
    '<body name="robot0_base"',
    '<body name="wall" pos="-0.55 0 0.5"><geom name="wall" type="box" size="0.05 1 0.5"/></body>\n'
    '    <body name="robot0_base"')
# SCENE with a forearm on a hinge above the hand.
ARM_SCENE = SCENE.replace(
    '<geom name="robot0_hand" type="box" pos="0.15 0 0" size="0.15 0.1 0.1"/>',
    '<geom name="robot0_hand" type="box" pos="0.15 0 0" size="0.15 0.1 0.1"/>'
    '<body name="robot0_arm" pos="0 0 0.3"><joint name="robot0_arm_joint" axis="0 1 0"/>'
    '<geom name="robot0_forearm" type="capsule" fromto="0 0 0 0 0 0.4" size="0.03"/></body>')


def _gap(m, d, margin):
    """Return _robot_env_gap after mj_forward."""
    mujoco.mj_forward(m, d)
    return hr._robot_env_gap(m, d, margin)


def _place(m, d, base_pos=(0.0, 0.0, 1.0), **kwargs):
    """Run place_robot_collision_free from base_pos at yaw 0 with its output silenced; return the env."""
    env = FakeEnv(m, d)
    with redirect_stdout(io.StringIO()):
        hr.place_robot_collision_free(env, np.array(base_pos), IDENTITY, **kwargs)
    return env


class PlacementTests(unittest.TestCase):
    """Check the search's back-off, its stop and its give-up."""

    def test_backs_off(self):
        """Placement backs an overlapping robot along -x until it is collision-free."""
        m, d = load()
        _place(m, d, step=0.02, max_iters=50, clearance=0.02)
        mujoco.mj_forward(m, d)
        self.assertEqual(rc.robot_env_contacts(m, d), [])
        self.assertLess(d.qpos[free_qadr(m)], 0.0)

    def test_no_move_when_already_clear(self):
        """Placement leaves a spawn that is already clear where it is."""
        m, d = load()
        _place(m, d, base_pos=(-1.0, 0.0, 1.0), step=0.02, max_iters=50, clearance=0.02)
        self.assertAlmostEqual(float(d.qpos[free_qadr(m)]), -1.0, places=9)

    def test_keeps_best_on_giveup(self):
        """Out of iterations, placement keeps the try with the least penetration."""
        m, d = load()
        env = FakeEnv(m, d)
        rc.place_robot_clear(env, np.array([0.0, 0.0, 1.0]), IDENTITY, 0.02)
        start_pen = total_penetration(m, d)
        self.assertGreater(start_pen, 0, "the scene must start in collision")
        # Clearing the counter takes ~11 steps; 3 forces the give-up path.
        _place(m, d, step=0.02, max_iters=3, clearance=0.02)
        mujoco.mj_forward(m, d)
        self.assertLessEqual(total_penetration(m, d), start_pen + 1e-9)
        self.assertLess(d.qpos[free_qadr(m)], 0.0, "give-up keeps the most-backed-off (least-penetrating) try")


class GapTests(unittest.TestCase):
    """Check _robot_env_gap, the free space the margin is measured in."""

    def test_ignores_floor_and_caps_at_margin(self):
        """The floor does not count toward the gap, which is capped at the margin."""
        m, d = load()
        d.qpos[free_qadr(m)] = -2.0          # far from the counter
        d.qpos[free_qadr(m) + 2] = 0.12      # hand box 2 cm above the floor plane
        self.assertAlmostEqual(_gap(m, d, 0.2), 0.2, places=9)

    def test_sees_near_pairs_inside_multi_geom_bodies(self):
        """The gap sees a pair MuJoCo's midphase culls in a multi-geom body, and restores the flags."""
        m, d = load(TWO_GEOMS)
        d.qpos[free_qadr(m)] = -0.22          # hand front 2 cm from the counter face
        self.assertAlmostEqual(_gap(m, d, 0.2), 0.02, places=6)
        self.assertFalse(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MIDPHASE, "model flags restored")

    def test_negative_when_overlapping(self):
        """The gap is negative while the robot overlaps the counter."""
        m, d = load()
        self.assertLess(_gap(m, d, 0.2), 0.0)


class MarginTests(unittest.TestCase):
    """Check the margin the H1-2's spawn keeps to every non-floor geom."""

    def test_backs_off_until_requested_gap(self):
        """A margin backs the robot off to the first spot with that gap, past mere clearance."""
        m, d = load()
        _place(m, d, margin=0.0)
        no_overlap_x = d.qpos[free_qadr(m)]
        _place(m, d, margin=0.2)
        self.assertGreaterEqual(_gap(m, d, 0.2), 0.2 - 1e-9, "the spawn keeps 20 cm to the counter")
        # The hand's front is 0.3 m ahead of the base; the counter face is at x=0.1.
        self.assertTrue(d.qpos[free_qadr(m)] <= -0.4 + 1e-9 < no_overlap_x)
        self.assertGreater(d.qpos[free_qadr(m)], -0.4 - 0.02 - 1e-9, "stops at the first spot with the margin")

    def test_zero_keeps_no_overlap_behavior(self):
        """Margin 0 only removes the overlap."""
        m, d = load()
        _place(m, d, margin=0.0)
        self.assertEqual(rc.robot_env_contacts(m, d), [])
        self.assertTrue(-0.22 - 1e-9 <= d.qpos[free_qadr(m)] < -0.18, "margin 0 only clears the overlap")

    def test_blocked_behind_keeps_widest_gap(self):
        """With a wall behind, the search stops at the wall and keeps the widest gap in front of it."""
        m, d = load(BLOCKED)
        out = io.StringIO()
        with redirect_stdout(out):
            hr.place_robot_collision_free(FakeEnv(m, d), np.array([0.0, 0.0, 1.0]), IDENTITY, margin=0.2)
        searched = float(re.search(r"searched ([0-9.]+) m back", out.getvalue()).group(1))
        self.assertLess(searched, 0.5, "the search stops at the wall behind")
        gap = _gap(m, d, 0.2)
        self.assertTrue(0.0 < gap < 0.2, gap)
        self.assertGreater(d.qpos[free_qadr(m)], -0.5, "stays in front of the wall")

    def test_holds_for_extra_poses_and_keeps_spawn_joints(self):
        """The margin also holds in extra poses, and the robot keeps its spawn joint state."""
        m, d = load(ARM_SCENE)
        arm = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot0_arm_joint")])])
        _place(m, d, margin=0.2)
        upright_x = d.qpos[free_qadr(m)]
        # Swung 90 deg forward, the forearm reaches 0.4 m ahead of its shoulder.
        _place(m, d, margin=0.2, poses=[(arm, np.array([math.pi / 2]))])
        self.assertEqual(d.qpos[arm[0]], 0.0, "the robot is left in its spawn joint state")
        self.assertLess(d.qpos[free_qadr(m)], upright_x - 0.05, "the swung pose needs more room")
        d.qpos[arm[0]] = math.pi / 2
        self.assertGreaterEqual(_gap(m, d, 0.2), 0.2 - 1e-9, "the margin holds in the extra pose")


if __name__ == "__main__":
    unittest.main()
