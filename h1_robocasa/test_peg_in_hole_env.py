"""Native scene tests; no ROS node, DDS participant or controller is started.

Run with the simulator Python: python -m unittest test_peg_in_hole_env -v
"""
import importlib
import copy
import json
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
from peg_in_hole_env import (SCENE_DIR, PegBoardArena, PegBoardConfig,
                             _polygon_clearance, canonical_task)


class PegBoardSceneTests(unittest.TestCase):
    def test_all_catalog_variants_seated_geometry(self):
        with (SCENE_DIR / "manipulation_catalog.json").open() as stream:
            catalog = json.load(stream)
        self.assertEqual(set(catalog["variants"]), {"cylinder", "hex", "l", "rectangle", "triangle"})
        cases = 0
        for shape, rows in catalog["variants"].items():
            self.assertEqual(set(rows), {"0", "1", "2", "3"})
            for row, variant in rows.items():
                peg_id = f"{shape}_{row}"
                with self.subTest(peg_id=peg_id):
                    config = PegBoardConfig(peg_id=peg_id, source_hole=peg_id)
                    arena = PegBoardArena(config)
                    model = mujoco.MjModel.from_xml_string(
                        ManipulationTask(mujoco_arena=arena, mujoco_robots=[]).get_xml())
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)
                    prefix, hole = config.peg_prefix, catalog["holes"][peg_id]
                    self.assertEqual(model.njnt, 1)
                    self.assertEqual(model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
                    self.assertAlmostEqual(model.body(prefix + "peg").mass[0], variant["mass"])
                    np.testing.assert_allclose(model.body(prefix + "peg").quat, hole["quat"], atol=1e-11)
                    tip = data.site(prefix + "tip").xpos
                    target = data.site("pegboard_" + hole["target_site"]).xpos
                    np.testing.assert_allclose(tip, target, atol=1e-11)
                    np.testing.assert_allclose(data.site(prefix + "tip").xmat,
                        data.site("pegboard_" + hole["target_site"]).xmat, atol=1e-11)
                    np.testing.assert_allclose(model.site(prefix + "tip").pos, [0, 0, 0])
                    vertices = []
                    for i in range(variant["collision_part_count"]):
                        geom = model.geom(prefix + f"collision_{i}")
                        mesh = geom.dataid[0]
                        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                        vertices.extend(model.mesh_vert[start:start + count] @
                                        data.geom_xmat[geom.id].reshape(3, 3).T + data.geom_xpos[geom.id])
                    vertices = np.asarray(vertices)
                    self.assertAlmostEqual(vertices[:, 2].min(), config.table_height + .003, places=8)
                    self.assertAlmostEqual(vertices[:, 2].max(), tip[2] + variant["length"], places=8)
                    # Initial contact with the blind floor is allowed; wall penetration is not.
                    self.assertTrue(all(c.dist >= -1e-7 for c in data.contact))
                    self.assertEqual([m["marker_id"] for m in arena.markers_metadata], [7, 8])
                    cases += 1
        self.assertEqual(cases, 20)

    def test_source_hole_is_independent_clearance_row(self):
        # Every source row uses the same shape's peg, with its own placement/yaw.
        for shape in ("cylinder", "hex", "l", "rectangle", "triangle"):
            for peg_row in range(4):
                for hole_row in range(4):
                    with self.subTest(shape=shape, peg_row=peg_row, hole_row=hole_row):
                        PegBoardConfig(peg_id=f"{shape}_{peg_row}", source_hole=f"{shape}_{hole_row}")
        config = PegBoardConfig(peg_id="hex_3", source_hole="hex_0", marker=False)
        arena = PegBoardArena(config)
        body = arena.worldbody.find(f"body[@name='{config.peg_prefix}peg']")
        np.testing.assert_allclose(np.fromstring(body.get("pos"), sep=" "),
                                   np.asarray(config.board_pos) + config.source_hole_spec["target"], atol=1e-11)
        np.testing.assert_allclose(np.fromstring(body.get("quat"), sep=" "),
                                   config.source_hole_spec["quat"], atol=1e-11)

    def test_polygon_fit_rejects_crossing_and_outside_profiles(self):
        square = [[0, 0], [2, 0], [2, 2], [0, 2]]
        self.assertAlmostEqual(_polygon_clearance([[.5, .5], [1, .5], [1, 1]], square), .5)
        self.assertEqual(_polygon_clearance([[1, 1], [3, 1], [1, 1.5]], square), 0)
        self.assertEqual(_polygon_clearance([[0, 0], [1, 0], [1, 1]], square), 0)
        # Both inner endpoints are inside, but their edge crosses the L's notch.
        notch = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]
        self.assertEqual(_polygon_clearance([[.5, 2], [2, .5], [.5, .5]], notch), 0)

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
                    self.assertEqual([m["marker_id"] for m in arena.markers_metadata], [7, 8])
                    for name in ("pegboard_aruco", "pegboard_aruco_8"):
                        self.assertEqual(model.geom(name).contype[0], 0)
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
        self.assertEqual(PegBoardConfig().peg_id, "cylinder_1")
        self.assertEqual(PegBoardConfig().source_hole, "cylinder_1")
        for kwargs in ({"side": "other"}, {"peg_mode": "held"}, {"table_height": 0},
                       {"peg_id": "unknown"}, {"peg_id": "cylinder_4"}, {"peg_id": "L_1"},
                       {"source_hole": "missing"}, {"source_hole": "hex_1"},
                       {"peg_id": []}, {"source_hole": None}):
            with self.assertRaises(ValueError):
                PegBoardConfig(**kwargs)

    def test_loose_footprint_support(self):
        # The authored L is 15 mm wide in X, 25 mm in Y; only X spans the strip.
        config = PegBoardConfig(peg_mode="loose", peg_id="l_1", source_hole="l_1")
        arena = PegBoardArena(config)
        bounds = np.asarray(config.peg_variant["bounds"])
        body = arena.worldbody.find(f"body[@name='{config.peg_prefix}peg']")
        position = np.fromstring(body.get("pos"), sep=" ")
        self.assertGreaterEqual(position[0] + bounds[0, 0], config.table_x - .18)
        self.assertLess(position[0] + bounds[1, 0], config.table_x - .16)
        with (SCENE_DIR / "manipulation_catalog.json").open() as stream:
            catalog = copy.deepcopy(json.load(stream))
        catalog["variants"]["l"]["1"]["bounds"][1][0] = .011
        with patch("peg_in_hole_env._catalog", return_value=catalog):
            with self.assertRaisesRegex(ValueError, "support strip"):
                PegBoardConfig(peg_mode="loose", peg_id="l_1", source_hole="l_1")

    def test_config_checks_actual_polygon_not_just_shape(self):
        with (SCENE_DIR / "manipulation_catalog.json").open() as stream:
            catalog = json.load(stream)
        hole = catalog["holes"]["cylinder_1"]
        center = np.asarray(hole["target"][:2])
        hole["profile_xy"] = (center + (np.asarray(hole["profile_xy"]) - center) * .5).tolist()
        with patch("peg_in_hole_env._catalog", return_value=catalog):
            with self.assertRaisesRegex(ValueError, "positive clearance"):
                PegBoardConfig()

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
                with patch.object(robot, "make_kitchen_env", return_value=env) as factory:
                    selection = ({} if env.scene_config.side == "left" else
                                 {"peg": "hex_3", "source_hole": "hex_0"})
                    with self.assertRaises(EndLoop):
                        runner.sim_loop("PegInHole", viewer=False, **selection)
                    if selection:
                        config = factory.call_args.kwargs["scene_config"]
                        self.assertEqual(config.peg_id, "hex_3")
                        self.assertEqual(config.source_hole, "hex_0")
                    else:
                        self.assertNotIn("scene_config", factory.call_args.kwargs)
                    for task in ("Battery", "TurnOnToasterOven"):
                        with self.assertRaisesRegex(ValueError, "require --task PegInHole"):
                            runner.sim_loop(task, viewer=False, peg="cylinder_1")
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
        import json
        from PIL import Image
        from realsense_profile import camera_specs, load_profile, fovy_intrinsics

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
                if camera == "robot0_head_cam":
                    intr = camera_specs(model, [("head", camera)], load_profile(), width / 1280, (width, height))[0].color
                else:
                    intr = fovy_intrinsics(width, height, model.cam_fovy[model.camera(camera).id])
                world_camera = np.eye(4)
                world_camera[:3, :3] = data.camera(camera).xmat.reshape(3, 3) @ np.diag([1, -1, -1])
                world_camera[:3, 3] = data.camera(camera).xpos
                world_board = np.eye(4)
                world_board[:3, :3] = data.body("pegboard_board").xmat.reshape(3, 3)
                world_board[:3, 3] = data.body("pegboard_board").xpos
                fixture = {"K": [[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]],
                           "T_camera_board": (np.linalg.inv(world_camera) @ world_board).tolist()}
                fixture_path = path.with_suffix(".json")
                fixture_path.write_text(json.dumps(fixture))
                # Ground truth is strictly evaluation-only in this test fixture.
                code = ("import cv2,numpy as np,sys,json; "
                        "from model_server.aruco_geometry import BoardEstimator; "
                        "e=BoardEstimator(); a=cv2.imread(sys.argv[1],0); "
                        "f=json.load(open(sys.argv[2])); c,ids,_=e.detect(a); "
                        "t,q=e.estimate(c,ids,np.array(f['K']),np.zeros(5)); "
                        "gt=np.array(f['T_camera_board']); "
                        "dt=np.linalg.norm(t[:3,3]-gt[:3,3]); "
                        "dr=np.linalg.norm(cv2.Rodrigues(t[:3,:3]@gt[:3,:3].T)[0]); "
                        "print(sys.argv[1],ids,'translation_m',dt,'rotation_rad',dr,q); "
                        "assert dt < .05 and dr < .35")
                if env.scene_config.side == "right" and camera == "robot0_head_cam" and width == 640:
                    code = ("import cv2,numpy as np,sys,json,unittest\n"
                            "from model_server.aruco_geometry import BoardEstimator,PoseRejected\n"
                            "e=BoardEstimator(); a=cv2.imread(sys.argv[1],0)\n"
                            "f=json.load(open(sys.argv[2])); c,ids,_=e.detect(a)\n"
                            "with unittest.TestCase().assertRaisesRegex(PoseRejected,'unresolved planar ambiguity') as failure:\n"
                            "    e.estimate(c,ids,np.array(f['K']),np.zeros(5))\n"
                            "q=failure.exception.quality\n"
                            "assert 0 <= q['ambiguity_gap_px'] < e.min_ambiguity_gap_px\n"
                            "print(sys.argv[1],ids,'REJECTED cold ambiguous view',q)\n")
                source = str(Path(__file__).resolve().parent.parent / "core_ws/src/model_server")
                subprocess.run(["/usr/bin/python3", "-c", code, str(path), str(fixture_path)], check=True,
                                 env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": source})


if __name__ == "__main__":
    unittest.main()
