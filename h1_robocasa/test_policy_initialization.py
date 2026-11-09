"""Offline initialization tests: real MuJoCo geometry, mocked transport wiring."""

from contextlib import contextmanager, redirect_stdout
import importlib
import io
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import yaml

import h1_2_robosuite as robot
from policy_initialization import (POLICY_CHOICES, _plane_gap, foot_floor_geometry,
                                   load_leg_pose, make_foot_placer, sole_gaps)
from sim_names import NameResolver


@contextmanager
def mocked_runner():
    stubs = {name: MagicMock() for name in
             ("unitree_interface", "mujoco_ros_bridge", "magpie_hand_bridge",
              "measurement_bridge", "rclpy", "rclpy.executors")}
    stubs["rclpy"].executors = stubs["rclpy.executors"]
    old = sys.modules.pop("h12_mujoco", None)
    try:
        with patch.dict(sys.modules, stubs):
            yield importlib.import_module("h12_mujoco"), stubs
    finally:
        sys.modules.pop("h12_mujoco", None)
        if old is not None:
            sys.modules["h12_mujoco"] = old


class PolicyLoadingTests(unittest.TestCase):
    def test_actual_yaml_sources_not_nominal_fame_or_almi_arms(self):
        root = Path(__file__).resolve().parents[1] / "core_ws/src/h12_lowerbody_rl/policies"
        for policy in POLICY_CHOICES:
            config = yaml.safe_load((root / policy / f"{policy}.yaml").read_text())
            expected = config["lower_start_pos" if policy == "fame_finetune" else "default_angles"][:12]
            self.assertEqual(load_leg_pose(policy), expected)

    def test_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "unknown initialization policy"):
            load_leg_pose("walk")

    def test_invalid_files(self):
        for content in ("{}", "null", "[", "lower_start_pos: [1]",
                        "lower_start_pos: [0,0,0,0,0,0,0,0,0,0,0,.nan]",
                        "lower_start_pos: [0,0,0,0,0,0,0,0,0,0,0,true]"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "fame_finetune"
                path.mkdir()
                (path / "fame_finetune.yaml").write_text(content)
                with self.assertRaisesRegex(ValueError, "invalid initialization policy.*fame_finetune"):
                    load_leg_pose("fame_finetune", directory)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "invalid initialization policy"):
                load_leg_pose("almi_manip_stand", directory)

    def test_load_has_no_torch_or_ros_dependency(self):
        with patch.dict(sys.modules, {"torch": None, "rclpy": None, "h12_mjlab": None}):
            self.assertEqual(len(load_leg_pose("fame_finetune")), 12)


class GroundedSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = robot.make_kitchen_env("PegInHole", hard_reset=False)
        cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def initialize(self, policy=None, overrides=None):
        class BeforeDDS(Exception):
            pass

        with mocked_runner() as (runner, stubs):
            stubs["unitree_interface"].SimInterface.side_effect = BeforeDDS
            env_vars = {"GOLEM_SPAWN_OFFSET": "", "GOLEM_SPAWN_BACKOFF": "0",
                        "GOLEM_SPAWN_MARGIN": "", "GOLEM_SIM_TETHER_ROT_STIFFNESS": "0"}
            env_vars.update(overrides or {})
            output = io.StringIO()
            with patch.dict(os.environ, env_vars), redirect_stdout(output), \
                    patch.object(robot, "make_kitchen_env", return_value=self.env), \
                    patch.object(runner.threading, "Thread", side_effect=AssertionError("no threads")), \
                    patch.object(runner, "ElasticBand", return_value=MagicMock()) as band:
                with self.assertRaises(BeforeDDS):
                    runner.sim_loop("PegInHole", viewer=False, lowerbody=policy)
            stubs["mujoco_ros_bridge"].init_ros.assert_not_called()
            model, data = self.env.sim.model._model, self.env.sim.data._data
            resolver = NameResolver(model)
            feet, floors = foot_floor_geometry(model)
            pelvis = model.joint("robot0_floating_base_joint").qposadr[0]
            snapshot = data.qpos.copy()
            torso = data.body("robot0_torso_link").xpos.copy()
            np.testing.assert_allclose(band.call_args.args[0], torso + [0, 0, .01], atol=1e-12)
            return model, data, resolver, feet, floors, pelvis, snapshot, output.getvalue()

    def test_policy_exact_legs_upper_objects_and_pelvis_forward(self):
        for policy in POLICY_CHOICES:
            with self.subTest(policy=policy):
                self.env.reset()
                model, data = self.env.sim.model._model, self.env.sim.data._data
                resolver = NameResolver(model)
                peg = model.joint("peg_cylinder_freejoint").qposadr[0]
                peg_before = data.qpos[peg:peg + 7].copy()
                intrinsics = model.cam_fovy.copy()
                m, d, r, feet, floors, base, q, logs = self.initialize(policy)
                np.testing.assert_array_equal(q[r.motor_qpos[:12]], load_leg_pose(policy))
                np.testing.assert_array_equal(q[r.motor_qpos[12:]], np.zeros(15))
                np.testing.assert_array_equal(q[peg:peg + 7], peg_before)
                np.testing.assert_array_equal(q[base + 3:base + 7], [1, 0, 0, 0])
                np.testing.assert_array_equal(m.cam_fovy, intrinsics)
                np.testing.assert_array_equal(d.qvel, np.zeros(m.nv))
                np.testing.assert_array_equal(d.ctrl, np.zeros(m.nu))
                np.testing.assert_array_equal(d.act, np.zeros(m.na))
                np.testing.assert_array_equal(d.qacc, np.zeros(m.nv))
                np.testing.assert_array_equal(d.qacc_warmstart, np.zeros(m.nv))
                gaps = sole_gaps(m, d, feet, floors)
                self.assertAlmostEqual(min(gaps.values()), 1e-8, places=9)
                for geoms in feet.values():
                    for g in geoms:
                        self.assertTrue(m.geom_contype[g] or m.geom_conaffinity[g])
                        for f in floors:
                            exact = mujoco.mj_geomDistance(m, d, g, f, 10, None)
                            self.assertAlmostEqual(_plane_gap(m, d, g, f)[0], exact, places=7)
                for side in feet:
                    body = d.body(f"robot0_{side}_ankle_roll_link")
                    heading = math.atan2(body.xmat[3], body.xmat[0])
                    if policy == "fame_finetune":
                        self.assertGreater(abs(heading), .3)
                    else:
                        self.assertLess(abs(heading), 1e-12)
                if policy == "almi_manip_stand":
                    self.assertLess(max(gaps.values()), 1e-7)
                else:
                    self.assertGreater(max(gaps.values()), 1e-4)
                    self.assertIn("cannot put both feet", logs)
                print(policy, "pelvis", q[base:base + 7].tolist(), "gaps", gaps,
                      "headings", {s: math.atan2(d.body(f"robot0_{s}_ankle_roll_link").xmat[3],
                                                  d.body(f"robot0_{s}_ankle_roll_link").xmat[0]) for s in feet})
                if os.environ.get("GOLEM_TEST_POLICY_RENDER") == "1":
                    from PIL import Image
                    option = mujoco.MjvOption()
                    option.geomgroup[0] = 0
                    with mujoco.Renderer(m, height=360, width=640) as renderer:
                        for camera, label in (("robot0_head_cam", "head"), ("pegboard_overview", "overview")):
                            renderer.update_scene(d, camera=camera, scene_option=option)
                            Image.fromarray(renderer.render()).save(f"/tmp/kilo/pose_{policy}_{label}.png")

    def test_omission_exact_existing_placement(self):
        m, d, r, feet, floors, base, q, logs = self.initialize()
        with mocked_runner() as (runner, _):
            np.testing.assert_array_equal(q[r.motor_qpos], runner._initial_motor_qpos())
            robot.place_robot_collision_free(
                self.env, list(self.env.init_robot_base_pos),
                robot._euler_to_wxyz(self.env.init_robot_base_ori),
                margin=getattr(self.env, "spawn_margin", .2),
                poses=runner._limp_upper_body_poses(m, r.motor_qpos, runner._initial_motor_qpos()))
        np.testing.assert_array_equal(d.qpos, q)
        self.assertIn("baked-in sim defaults", logs)
        self.assertNotIn("initial legs from", logs)

    def test_offset_and_backoff_preserve_grounding_and_obstacle_margin(self):
        m, d, r, feet, floors, base, q, _ = self.initialize("almi_manip_stand")
        original = q[base:base + 3].copy()
        m, d, r, feet, floors, base, q, _ = self.initialize(
            "almi_manip_stand", {"GOLEM_SPAWN_OFFSET": "-0.1,0.03", "GOLEM_SPAWN_BACKOFF": ".1",
                                 "GOLEM_SPAWN_MARGIN": ".1"})
        self.assertLess(q[base], original[0] - .09)
        self.assertAlmostEqual(q[base + 1], self.env.init_robot_base_pos[1] + .03)
        self.assertLess(max(sole_gaps(m, d, feet, floors).values()), 1e-7)
        self.assertGreaterEqual(robot._robot_env_gap(m, d, .1), .1 - 1e-8)

    def test_bad_requested_config_fails_before_scene_and_dds(self):
        with mocked_runner() as (runner, stubs), \
                patch.object(robot, "make_kitchen_env") as factory, \
                patch.object(runner, "load_leg_pose", side_effect=ValueError("bad requested YAML")):
            with self.assertRaisesRegex(ValueError, "bad requested YAML"):
                runner.sim_loop("PegInHole", lowerbody="fame_finetune")
            factory.assert_not_called()
            stubs["unitree_interface"].SimInterface.assert_not_called()

    def test_unknown_cli_choice_is_argparse_error_before_scene_or_dds(self):
        with mocked_runner() as (runner, stubs), \
                patch.object(sys, "argv", ["h12_mujoco.py", "--lowerbody", "walk"]), \
                patch.object(robot, "make_kitchen_env") as factory, \
                redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(runner.__file__, run_name="__main__")
            self.assertEqual(error.exception.code, 2)
            self.assertIn("invalid choice", stderr.getvalue())
            factory.assert_not_called()
            stubs["unitree_interface"].SimInterface.assert_not_called()

    def test_invalid_mapping_fails_closed_before_dds(self):
        self.env.reset()
        resolver = NameResolver(self.env.sim.model._model)
        resolver.motor_qpos[1] = resolver.motor_qpos[0]
        with mocked_runner() as (runner, stubs), \
                patch.object(runner, "NameResolver", return_value=resolver), \
                patch.object(robot, "make_kitchen_env", return_value=self.env):
            with self.assertRaisesRegex(ValueError, "12 distinct name-resolved leg joints"):
                runner.sim_loop("PegInHole", lowerbody="fame_finetune")
            stubs["unitree_interface"].SimInterface.assert_not_called()

    def test_task_yaw_is_not_derived_from_toes(self):
        self.env.reset()
        model, data = self.env.sim.model._model, self.env.sim.data._data
        resolver = NameResolver(model)
        data.qpos[resolver.motor_qpos[:12]] = load_leg_pose("fame_finetune")
        place, feet, floors = make_foot_placer(self.env, robot.place_robot_freejoint)
        quat = [math.cos(.25), 0, 0, math.sin(.25)]
        place(self.env, self.env.init_robot_base_pos, quat)
        base = model.joint("robot0_floating_base_joint").qposadr[0]
        np.testing.assert_array_equal(data.qpos[base + 3:base + 7], quat)
        self.assertLess(min(sole_gaps(model, data, feet, floors).values()), 1e-7)

    def test_transformed_floor_and_cylinder_exact_support(self):
        # A long cylinder tilted relative to an offset, sloping floor: rbound/AABB
        # would give a different answer from its analytic support function.
        m = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
          <body name="floor_body" pos="0 0 .3" euler="0 .2 0">
            <geom name="floor" type="plane" size="2 2 .1"/></body>
          <body pos="0 0 1"><geom name="foot" type="cylinder" size=".1 .3" euler="0 .5 0"/></body>
        </worldbody></mujoco>''')
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        gap, nz = _plane_gap(m, d, m.geom("foot").id, m.geom("floor").id)
        self.assertAlmostEqual(gap, .7 * math.cos(.2) - .1 * math.sin(.3) - .3 * math.cos(.3))
        self.assertAlmostEqual(nz, math.cos(.2))

    def test_visual_and_incompatible_floor_planes_cannot_lift_feet(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="floor" type="plane" size="2 2 .1" contype="1" conaffinity="1"/>
          <geom name="floor_visual" type="plane" pos="0 0 .4" size="2 2 .1" contype="0" conaffinity="0"/>
          <geom name="floor_incompatible" type="plane" pos="0 0 .6" size="2 2 .1" contype="2" conaffinity="2"/>
          <body name="robot0_pelvis" pos="0 0 1">
            <freejoint name="robot0_floating_base_joint"/>
            <body name="robot0_left_ankle_roll_link" pos="0 .1 -.5">
              <geom type="box" size=".1 .05 .02" contype="1" conaffinity="1"/></body>
            <body name="robot0_right_ankle_roll_link" pos="0 -.1 -.5">
              <geom type="box" size=".1 .05 .02" contype="1" conaffinity="1"/></body>
          </body></worldbody></mujoco>''')
        d = mujoco.MjData(m)
        env = SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(_model=m),
                                                   data=SimpleNamespace(_data=d)),
                              robots=[SimpleNamespace(robot_model=SimpleNamespace(naming_prefix="robot0_"))])

        def place_freejoint(env, pos, quat):
            d.qpos[:3] = pos
            d.qpos[3:7] = quat
            mujoco.mj_forward(m, d)

        place, feet, floors = make_foot_placer(env, place_freejoint)
        self.assertNotIn(m.geom("floor_visual").id, floors)
        place(env, [0, 0, 1], [1, 0, 0, 0])
        self.assertAlmostEqual(d.qpos[2], .52000001)
        self.assertLess(max(sole_gaps(m, d, feet, floors).values()), 1e-7)
        m.geom_conaffinity[m.geom("floor").id] = 0
        m.geom_contype[m.geom("floor").id] = 0
        with self.assertRaisesRegex(ValueError, "no collision-compatible floor"):
            make_foot_placer(env, place_freejoint)


if __name__ == "__main__":
    unittest.main()
