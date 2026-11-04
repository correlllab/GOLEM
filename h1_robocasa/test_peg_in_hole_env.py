"""Native scene tests; no ROS node, DDS participant or controller is started.

Run with the simulator Python: python -m unittest test_peg_in_hole_env -v
"""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
from robosuite.models.tasks import ManipulationTask

import h1_2_robosuite as robot
from peg_in_hole_env import PegBoardArena, PegBoardConfig, canonical_task


class PegBoardSceneTests(unittest.TestCase):
    def test_arena_geometry_and_free_peg(self):
        for side in ("left", "right"):
            for mode in ("loose", "seated"):
                with self.subTest(side=side, mode=mode):
                    config = PegBoardConfig(side=side, peg_mode=mode)
                    arena = PegBoardArena(config)
                    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[])
                    model = mujoco.MjModel.from_xml_string(task.get_xml())
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)
                    board = model.body("pegboard_board")
                    self.assertEqual(model.body_jntnum[board.id], 0)
                    collision = np.flatnonzero((model.geom_bodyid == board.id) &
                                               (model.geom_contype != 0))
                    self.assertEqual(len(collision), 3005)
                    self.assertEqual(model.njnt, 1)
                    self.assertEqual(model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
                    self.assertAlmostEqual(model.body("peg_cylinder_peg").mass[0],
                                           0.0067836904074)
                    self.assertEqual(model.geom("peg_cylinder_visual").group[0], 2)
                    expected_y = .185369815094 if side == "left" else -.264630184906
                    np.testing.assert_allclose(board.pos, [.503832018731, expected_y, .955172239917], atol=1e-12)
                    table = model.geom("pegboard_table_top")
                    np.testing.assert_allclose(table.size, [.18, .145, .02], atol=1e-12)
                    np.testing.assert_allclose(table.pos, [.503832018731, expected_y, .935172239917], atol=1e-12)
                    for i, (dx, dy) in enumerate(((-.125, -.09), (-.125, .09),
                                                  (.125, -.09), (.125, .09))):
                        leg = model.geom(f"pegboard_table_leg_{i}")
                        np.testing.assert_allclose(leg.pos, [.503832018731 + dx, expected_y + dy,
                                                           (.955172239917 - .04) / 2], atol=1e-12)
                        np.testing.assert_allclose(leg.size, [.025, .025, (.955172239917 - .04) / 2])
                    tip = data.site("peg_cylinder_tip").xpos.copy()
                    entry = data.site("pegboard_hole_cylinder_1_entry").xpos.copy()
                    geom = model.geom("peg_cylinder_collision_0")
                    mesh = geom.dataid[0]
                    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                    vertices = model.mesh_vert[start:start + count]
                    world = vertices @ data.geom_xmat[geom.id].reshape(3, 3).T + data.geom_xpos[geom.id]
                    np.testing.assert_allclose(model.site("peg_cylinder_tip").pos, [0, 0, 0])
                    self.assertAlmostEqual(world[:, 2].min(), tip[2], places=8)
                    self.assertAlmostEqual(world[:, 2].max(), tip[2] + .05, places=8)
                    if mode == "loose":
                        self.assertGreater(np.linalg.norm(tip[:2] - entry[:2]), 0.03)
                        for i in (0, 1):
                            self.assertGreaterEqual(world[:, i].min(), table.pos[i] - table.size[i])
                            self.assertLessEqual(world[:, i].max(), table.pos[i] + table.size[i])
                        self.assertLess(world[:, 0].max(), board.pos[0] - .16)
                    else:
                        np.testing.assert_allclose(tip[:2], entry[:2], atol=1e-10)
                        self.assertAlmostEqual(tip[2] - config.table_height, .003)
                        self.assertAlmostEqual(entry[2] - tip[2], .022)
                        self.assertAlmostEqual(world[:, 2].max() - config.table_height, .053)
                    for _ in range(100):
                        mujoco.mj_step(model, data)
                    self.assertTrue(np.isfinite(data.qpos).all())
                    self.assertGreater(data.qpos[2], config.table_height - 0.001)

    def test_config_and_name(self):
        self.assertEqual(canonical_task("peginhole"), "PegInHole")
        self.assertIsNone(canonical_task("Battery"))
        self.assertEqual(PegBoardConfig().peg_mode, "seated")
        for kwargs in ({"side": "other"}, {"peg_mode": "held"}, {"table_height": 0}):
            with self.assertRaises(ValueError):
                PegBoardConfig(**kwargs)

    def test_native_env_reset_and_mocked_runner(self):
        self._check_native_env("left")

    def test_native_right_env_reset_and_mocked_runner(self):
        self._check_native_env("right")

    def _check_native_env(self, side):
        # Fail closed if the native factory tries to load the kitchen stack.
        with patch.dict(sys.modules, {"robocasa": None}):
            env = robot.make_kitchen_env("PegInHole", hard_reset=False,
                                         scene_config=PegBoardConfig(side=side))
            try:
                env.reset()
                model, data = env.sim.model._model, env.sim.data._data
                self.assertAlmostEqual(model.opt.timestep, 0.001)
                self.assertEqual(model.nq, 53)  # robot 46 + peg freejoint 7
                self.assertEqual(model.nv, 51)
                self.assertFalse(env._check_success())
                self.assertEqual(env.reward(), 0.0)
                self.assertFalse(env.get_ep_meta()["success_implemented"])
                self.assertEqual(env.get_ep_meta()["peg_mode"], "seated")
                self.assertEqual(env.get_ep_meta()["target_hole"], "cylinder_1")
                peg_adr = model.jnt_qposadr[model.joint("peg_cylinder_freejoint").id]
                peg_start = data.qpos[peg_adr:peg_adr + 7].copy()
                env.reset()
                np.testing.assert_allclose(data.qpos[peg_adr:peg_adr + 7], peg_start)
                self._check_runner(env)
                if os.environ.get("GOLEM_TEST_SCENE_RENDER") == "1":
                    self._check_render(env)
            finally:
                env.close()

    def _check_runner(self, env):
        class EndLoop(Exception):
            pass

        # Stub transport modules BEFORE importing the production runner. Nothing
        # can initialize DDS (including domain 0) or publish a motor command.
        stubs = {}
        for name in ("unitree_interface", "mujoco_ros_bridge", "magpie_hand_bridge",
                     "measurement_bridge", "rclpy", "rclpy.executors"):
            stubs[name] = MagicMock()
        stubs["rclpy"].executors = stubs["rclpy.executors"]
        transport = stubs["unitree_interface"].SimInterface
        sensors = stubs["mujoco_ros_bridge"].RosSensorBridge
        sensors.return_value.tick.side_effect = EndLoop
        # battery_env only registers its class, with no kitchen import.
        old_runner = sys.modules.pop("h12_mujoco", None)
        try:
            with patch.dict(sys.modules, stubs), patch.dict(os.environ, {"GOLEM_SIM_ODOM": "0"}):
                runner = importlib.import_module("h12_mujoco")
                with patch.object(runner, "_runnable_tasks", side_effect=AssertionError("kitchen")):
                    self.assertEqual(runner._resolve_task("peginhole"), "PegInHole")
                    self.assertEqual(runner._resolve_task("Battery"), "Battery")
                with patch.object(robot, "make_kitchen_env", return_value=env):
                    with self.assertRaises(EndLoop):
                        runner.sim_loop("PegInHole", viewer=False)
                transport.assert_called_once()
                transport.return_value.write_ctrl.assert_called_once()
                self.assertIs(transport.call_args.args[0], env.sim.model._model)
                self.assertIs(transport.call_args.args[1], env.sim.data._data)
                self.assertEqual(sensors.call_args.kwargs["base_body_id"], -1)
                cameras = sensors.call_args.kwargs["cameras"]
                self.assertEqual(len(cameras), 3)
                stubs["magpie_hand_bridge"].MagpieHandBridge.assert_any_call(
                    env.sim.model._model, env.sim.data._data, side="left",
                    sim_lock=transport.call_args.kwargs["lock"], name_prefix="gripper0_left_")
                sensors.return_value.shutdown.assert_called_once()
        finally:
            sys.modules.pop("h12_mujoco", None)
            if old_runner is not None:
                sys.modules["h12_mujoco"] = old_runner

    def _check_render(self, env):
        """Use clean system OpenCV separately from simulator NumPy 2."""
        from PIL import Image

        model, data = env.sim.model._model, env.sim.data._data
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[1:3] = 1
        option.sitegroup[:] = 0
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 720)
        for camera, width, height, label in (("robot0_head_cam", 640, 360, "head_640"),
                                              ("robot0_head_cam", 1280, 720, "head"),
                                              ("pegboard_overview", 1280, 720, "overview")):
            with mujoco.Renderer(model, height=height, width=width) as renderer:
                renderer.update_scene(data, camera=camera, scene_option=option)
                side_suffix = "" if env.scene_config.side == "left" else "_right"
                path = Path("/tmp/kilo") / f"pegboard_corrected_{label}{side_suffix}.png"
                Image.fromarray(renderer.render()).save(path)
                code = ("import cv2,numpy as np,sys; "
                        "gray=cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE); "
                        "d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50); "
                        "corners,ids,rejected=cv2.aruco.detectMarkers(gray,d); "
                        "print(sys.argv[1],ids); "
                        "assert sys.argv[2]=='optional' or (ids is not None and 7 in ids)")
                # The right gripper occludes this corner in the unchanged spawn
                # posture. Detectability there is reported, not a geometry invariant.
                required = camera == "pegboard_overview" or (width == 1280 and env.scene_config.side == "left")
                subprocess.run(["/usr/bin/python3", "-c", code, str(path),
                                "required" if required else "optional"], check=True,
                                env={**os.environ, "PYTHONNOUSERSITE": "1"})


if __name__ == "__main__":
    unittest.main()
