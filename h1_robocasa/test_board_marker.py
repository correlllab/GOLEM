"""CPU texture / XML checks and optional MuJoCo offscreen detection checks."""

import os
import unittest
import xml.etree.ElementTree as ET

from board_marker import marker_pixels, marker_png, mount_board_marker


def detect(cv2, image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    return cv2.aruco.detectMarkers(image, dictionary)


class BoardMarkerTest(unittest.TestCase):
    def test_deterministic_pixels_and_png(self):
        rows = marker_pixels(1)
        self.assertEqual(rows[0], bytes([255] * 8))
        self.assertEqual(rows[1], bytes([255, 0, 0, 0, 0, 0, 0, 255]))
        self.assertEqual(rows[2], bytes([255, 0, 255, 255, 0, 0, 0, 255]))
        self.assertEqual(marker_png(), marker_png())
        self.assertTrue(marker_png().startswith(b"\x89PNG\r\n\x1a\n"))
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                marker_pixels(value)

    def test_mount_metadata_and_no_physics_changes(self):
        asset = ET.Element("asset")
        board = ET.fromstring('<body name="board"><geom name="solid" type="box" size=".16 .125 .0125"/></body>')
        original = ET.tostring(board[0])
        metadata = mount_board_marker(board, asset)
        self.assertEqual(ET.tostring(board[0]), original)
        self.assertEqual(metadata["marker_id"], 7)
        self.assertAlmostEqual(metadata["black_size_m"], .03)
        self.assertAlmostEqual(metadata["quiet_zone_m"], .005)
        self.assertAlmostEqual(metadata["outer_size_m"], .04)
        self.assertEqual(metadata["position_board_m"], [-.135, .10, .0252])
        self.assertEqual(metadata["T_board_marker"], [[1, 0, 0, -.135], [0, 1, 0, .10],
                                                    [0, 0, 1, .0252], [0, 0, 0, 1]])
        import json
        from pathlib import Path
        catalog = json.loads((Path(__file__).resolve().parent.parent / "CL_Assets" /
                              "mujoco_assets/manipulation_net/manipulation_catalog.json").read_text())
        low, high = (-.155, .08), (-.115, .12)
        for i in (0, 1):
            self.assertGreaterEqual(low[i], -catalog["board"]["footprint"][i] / 2)
            self.assertLessEqual(high[i], catalog["board"]["footprint"][i] / 2)
        for hole in catalog["holes"].values():
            self.assertTrue(any(high[i] <= min(p[i] for p in hole["profile_xy"]) or
                                low[i] >= max(p[i] for p in hole["profile_xy"]) for i in (0, 1)))
        geom = board[-1]
        self.assertEqual(geom.tag, "geom")
        self.assertEqual([geom.get(k) for k in ("group", "mass", "contype", "conaffinity")],
                         ["1", "0", "0", "0"])
        with self.assertRaises(ValueError):
            mount_board_marker(board, asset)
        with self.assertRaises(ValueError):
            mount_board_marker(board, asset, name="other", mount_pos=(-.128, .065, .0252))
        with self.assertRaises(ValueError):
            mount_board_marker(board, asset, name="outside", mount_pos=(-.15, .10, .0252))
        with self.assertRaises(ValueError):
            mount_board_marker(board, asset, black_size_m=float("nan"))

    def test_opencv_texture_detection(self):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            self.skipTest(str(exc))
        if not hasattr(cv2, "aruco"):
            self.skipTest("OpenCV aruco module unavailable")
        image = cv2.imdecode(np.frombuffer(marker_png(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertTrue(np.array_equal(image, np.array([list(row) for row in marker_pixels()], dtype=np.uint8)))
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        generate = (cv2.aruco.generateImageMarker if hasattr(cv2.aruco, "generateImageMarker")
                    else cv2.aruco.drawMarker)
        self.assertTrue(np.array_equal(image[32:-32, 32:-32], generate(dictionary, 7, 192)))
        corners, ids, _ = detect(cv2, image)
        self.assertEqual(ids.reshape(-1).tolist(), [7])
        self.assertTrue(np.allclose(corners[0][0], ((32, 32), (223, 32), (223, 223), (32, 223)), atol=1))

    @unittest.skipUnless(os.environ.get("GOLEM_TEST_MARKER_RENDER") == "1",
                         "set GOLEM_TEST_MARKER_RENDER=1 with a working offscreen GL backend")
    def test_rendered_detection_and_pose(self):
        import cv2
        import mujoco
        import numpy as np

        root = ET.fromstring('<mujoco><asset/><worldbody><light pos="0 0 1"/><body name="board">'
                             '<geom type="box" size=".16 .125 .0125" pos="0 0 .0125"/>'
                             '</body><camera name="test" pos="-.135 .10 .35" fovy="45"/></worldbody></mujoco>')
        baseline = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        metadata = mount_board_marker(root.find("worldbody/body"), root.find("asset"))
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.assertEqual((model.nbody, model.njnt, model.ngeom),
                         (baseline.nbody, baseline.njnt, baseline.ngeom + 1))
        for field in ("body_mass", "body_ipos", "body_pos", "body_quat"):
            np.testing.assert_array_equal(getattr(model, field), getattr(baseline, field))
        # MuJoCo may choose a different principal-axis ordering after adding a
        # mesh; compare the actual inertia tensors in the body frame.
        for body_id in range(model.nbody):
            tensors = []
            for compiled in (baseline, model):
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, compiled.body_iquat[body_id])
                rotation = rotation.reshape(3, 3)
                tensors.append(rotation @ np.diag(compiled.body_inertia[body_id]) @ rotation.T)
            np.testing.assert_allclose(tensors[0], tensors[1], atol=1e-12)
        for field in ("geom_size", "geom_type", "geom_contype", "geom_conaffinity"):
            np.testing.assert_array_equal(getattr(model, field)[:baseline.ngeom],
                                          getattr(baseline, field))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[1:3] = 1
        option.sitegroup[:] = 0
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera="test", scene_option=option)
            image = renderer.render()
        corners, ids, _ = detect(cv2, image)
        self.assertIsNotNone(ids)
        self.assertEqual(ids.reshape(-1).tolist(), [7])
        # Canonical orientation, not just detection: a flipped UV could still
        # decode a rotated code but would invalidate the calibration transform.
        self.assertLess(corners[0][0, 0, 0], corners[0][0, 1, 0])
        self.assertLess(corners[0][0, 0, 1], corners[0][0, 3, 1])
        focal = 240 / np.tan(np.deg2rad(22.5))
        intrinsics = np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1.]])
        h = metadata["black_size_m"] / 2
        points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        ok, rotation, translation = cv2.solvePnP(points, corners[0], intrinsics, np.zeros(5))
        self.assertTrue(ok)
        self.assertAlmostEqual(float(translation[2, 0]), .35 - .0252, delta=.008)
        np.testing.assert_allclose(cv2.Rodrigues(rotation)[0], np.diag([1, -1, -1]), atol=.03)


if __name__ == "__main__":
    unittest.main()
