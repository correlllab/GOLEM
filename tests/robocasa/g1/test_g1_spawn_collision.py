"""Check g1_robosuite's collision-free spawn, and the joint dynamics and stance it gives the G1 MJCF.

Runs on the host as well as in the container: where robosuite is missing it is stood in for
(tests/robocasa/spawn_scene.py), and spawn_scene's small MjModel (a robot0_-prefixed free body
and a fixed counter) exercises the real geometry, contact filtering and back-off loop. The spawn
in a real kitchen, hanging from the band, is test_g1_band.py's. The placement helpers both robots
share are test_robosuite_common.py's.

Run from the repo root: PYTHONPATH=golem_robocasa python3 tests/robocasa/g1/test_g1_spawn_collision.py
(g1_robosuite reads CL_Assets/g1's MJCF and URDF at import).
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/robocasa
import robosuite_common as rc  # noqa: E402
from spawn_scene import FakeEnv, SCENE, free_qadr, import_robot_module, load, total_penetration  # noqa: E402

gr = import_robot_module("g1.g1_robosuite")

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
# SCENE with a visual box (contype and conaffinity 0) on the robot reaching 5 cm below its hand.
VISUAL_SCENE = SCENE.replace(
    '<geom name="robot0_hand"',
    '<geom name="robot0_visual" type="box" pos="0.15 0 -0.05" size="0.05 0.05 0.1" contype="0" '
    'conaffinity="0"/><geom name="robot0_hand"')
# From Unitree's unitree_rl_gym: g1_12dof.xml's default joint and g1_config.py's
# default_joint_angles (rad).
UNITREE_JOINT = {"damping": 0.001, "armature": 0.01, "frictionloss": 0.1}
UNITREE_LEGS = {"hip_pitch": -0.1, "hip_roll": 0.0, "hip_yaw": 0.0,
                "knee": 0.3, "ankle_pitch": -0.2, "ankle_roll": 0.0}


def _place(m, d, base_pos=(0.0, 0.0, 1.0), **kwargs):
    """Run place_robot_collision_free from base_pos at yaw 0 with its output silenced."""
    with redirect_stdout(io.StringIO()):
        gr.place_robot_collision_free(FakeEnv(m, d), np.array(base_pos), IDENTITY, **kwargs)


class PlacementTests(unittest.TestCase):
    """Check the G1's spawn search: back off until nothing penetrates."""

    def test_backs_off(self):
        """A spawn in collision is backed off along -x until it is clear."""
        m, d = load()
        _place(m, d, step=0.02, max_iters=50, clearance=0.02)
        mujoco.mj_forward(m, d)
        self.assertEqual(rc.robot_env_contacts(m, d), [])
        self.assertLess(d.qpos[free_qadr(m)], 0.0)

    def test_no_move_when_already_clear(self):
        """A spawn already clear is not moved."""
        m, d = load()
        _place(m, d, base_pos=(-1.0, 0.0, 1.0), step=0.02, max_iters=50, clearance=0.02)
        self.assertAlmostEqual(float(d.qpos[free_qadr(m)]), -1.0, places=9)

    def test_keeps_best_on_giveup(self):
        """With too few iterations to clear, the least-penetrating placement tried is kept."""
        m, d = load()
        # The raw placement penetrates; clearing the counter takes about 11 steps.
        gr.place_robot_clear(FakeEnv(m, d), np.array([0.0, 0.0, 1.0]), IDENTITY, 0.02)
        start_pen = total_penetration(m, d)
        self.assertGreater(start_pen, 0, "the scene must start in collision")
        # Too few iterations to clear it, so the give-up path runs.
        _place(m, d, step=0.02, max_iters=3, clearance=0.02)
        mujoco.mj_forward(m, d)
        self.assertLessEqual(total_penetration(m, d), start_pen + 1e-9)
        self.assertLess(d.qpos[free_qadr(m)], 0.0, "give-up keeps the most-backed-off (least-penetrating) try")

    def test_floor_clearance_skips_visual_geoms(self):
        """The G1's floor clearance counts only colliding geoms: the visual box may reach below the floor."""
        m, d = load(VISUAL_SCENE)
        env = FakeEnv(m, d)
        gr.place_robot_clear(env, np.array([-1.0, 0.0, 1.0]), IDENTITY, 0.02)
        self.assertAlmostEqual(gr.lowest_robot_point_z(env), 0.02, places=6)
        self.assertAlmostEqual(rc.lowest_robot_point_z(env), 0.02 - 0.05, places=6)


class ModelTests(unittest.TestCase):
    """Check the joint dynamics and the stance g1_robosuite gives the G1 MJCF."""

    def test_joint_dynamics_are_unitrees(self):
        """The 29 body joints have Unitree's damping, armature and friction; the fingers a tenth of the friction and no damping."""
        self.assertEqual((gr.JOINT_DAMPING, gr.JOINT_ARMATURE, gr.JOINT_FRICTIONLOSS), (0.001, 0.01, 0.1))
        joints = list(gr.build_robot_xml().iter("joint"))
        body = [j for j in joints if "_hand_" not in j.get("name")]
        self.assertEqual(len(body), 29)
        for joint in body:
            self.assertEqual({k: float(joint.get(k)) for k in UNITREE_JOINT}, UNITREE_JOINT, joint.attrib)
        for joint in joints:  # the fingers: the armature, a tenth of the friction, no damping
            if "_hand_" in joint.get("name"):
                self.assertEqual((float(joint.get("damping")), float(joint.get("armature")),
                                  float(joint.get("frictionloss"))), (0.0, 0.01, 0.1 * 0.1), joint.attrib)

    def test_stance_is_unitrees_default(self):
        """The spawn stance is Unitree's default leg pose."""
        expected = {f"{side}_{name}_joint": q for side in ("left", "right") for name, q in UNITREE_LEGS.items()}
        self.assertEqual({n: q for n, q in gr.STANCE.items() if q}, {n: q for n, q in expected.items() if q})
        self.assertLessEqual(set(gr.STANCE), set(expected))


if __name__ == "__main__":
    unittest.main()
