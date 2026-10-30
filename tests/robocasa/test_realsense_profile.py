"""realsense_profile: real-camera placement, intrinsics, TF and point cloud.

Needs MuJoCo, NumPy and PyYAML only; the rendering checks skip when no
offscreen GL context is available (set MUJOCO_GL=egl or osmesa).
"""
import math
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import realsense_profile as rp

# The robot MJCF's head camera mount: a body standing in for camera_link with
# head_cam as the robot defines it (its optical frame is head_camera_link).
SCENE = '''<mujoco>
  <visual><global offwidth="1280" offheight="720"/><headlight ambient="1 1 1"/></visual>
  <worldbody>
    <body name="camera_link" pos="0.11 0.02 1.6" euler="0 0.8859291283123216 0">
      <camera name="head_cam" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="58"/>
    </body>
    <camera name="hand_cam" pos="0 0 1" xyaxes="1 0 0 0 1 0" fovy="58"/>
  </worldbody>
</mujoco>'''


def _optical(model, data, name):
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    return rp._T(data.cam_xmat[cam].reshape(3, 3) @ rp._F, data.cam_xpos[cam])


def _angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = rp.load_profile()
        cls.head = cls.profile["cameras"]["head"]

    def _placed(self, marker_opt=None):
        """Compile SCENE with the profiled head cameras placed, optionally with a
        red marker at `marker_opt` in the colour optical frame."""
        root = ET.fromstring(SCENE)
        base = mujoco.MjModel.from_xml_string(SCENE)
        base_data = mujoco.MjData(base)
        mujoco.mj_forward(base, base_data)
        anchor = _optical(base, base_data, "head_cam")
        depth = rp.place_profiled_cameras(root, "head_cam", self.head)
        if marker_opt is not None:
            T_w_color = anchor @ rp.frame_transform(self.head, self.head["anchor_link"], self.head["color_frame"])
            world = T_w_color[:3, :3] @ marker_opt + T_w_color[:3, 3]
            ET.SubElement(root.find("worldbody"), "geom", name="marker", type="sphere", size="0.012",
                          pos=" ".join(map(str, world)), rgba="1 0 0 1", group="4",
                          contype="0", conaffinity="0")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data, anchor, depth

    def test_cameras_sit_on_the_captured_tf_chain(self):
        model, data, anchor, depth = self._placed()
        for name, frame in (("head_cam", self.head["color_frame"]), (depth, self.head["depth_frame"])):
            expected = anchor @ rp.frame_transform(self.head, self.head["anchor_link"], frame)
            err = np.linalg.inv(expected) @ _optical(model, data, name)
            self.assertLess(np.abs(err[:3, 3]).max(), 1e-9, frame)
            self.assertLess(_angle_deg(err[:3, :3]), 1e-6, frame)

    def test_specs_scale_the_real_intrinsics(self):
        model, _, _, _ = self._placed()
        names = [("head", "head_cam"), ("left_hand", "hand_cam")]
        head, hand = rp.camera_specs(model, names, self.profile, 0.5, (256, 256))
        fx, fy, cx, cy = self.head["color"]["k"]
        self.assertEqual((head.color.width, head.color.height), (640, 360))
        self.assertAlmostEqual(head.color.fx, fx / 2)
        self.assertAlmostEqual(head.color.cx, (cx + 0.5) / 2 - 0.5)
        self.assertAlmostEqual(head.color.cy, (cy + 0.5) / 2 - 0.5)
        # Depth camera_info at the undecimated size; the image at half of it.
        self.assertEqual((head.depth_info.width, head.depth_info.height), (640, 360))
        self.assertEqual((head.depth_image.width, head.depth_image.height), (320, 180))
        self.assertEqual(head.depth_camera, "head_cam" + rp.DEPTH_CAMERA_SUFFIX)
        self.assertEqual(head.color_frame, "head_color_optical_frame")
        # Hand cameras keep the MJCF fovy at the fallback size; depth from the same camera.
        f = 128 / math.tan(math.radians(29))
        self.assertAlmostEqual(hand.color.fx, f)
        self.assertEqual((hand.color.cx, hand.color.cy), (127.5, 127.5))
        self.assertEqual(hand.depth_camera, "hand_cam")
        self.assertEqual((hand.depth_image.width, hand.depth_image.height), (128, 128))
        np.testing.assert_allclose(hand.T_color_depth, np.eye(4), atol=1e-12)

    def test_tf_tree_matches_the_real_driver(self):
        names = ["head", "left_hand", "right_hand"]
        edges = rp.tf_edges(self.profile, names)
        children = [e[1] for e in edges]
        self.assertEqual(len(children), len(set(children)))
        for ns in names:
            for frame in ("link", "color_frame", "color_optical_frame", "depth_frame", "depth_optical_frame"):
                self.assertIn(f"{ns}_{frame}", children)
            self.assertEqual(sum(e[4] for e in edges if e[1].startswith(ns)), 4, ns)
        mounts = {e[0]: e[1] for e in edges if not e[4]}
        self.assertEqual(mounts, {"head_camera_link": "head_link",
                                  "lg_mount": "left_hand_link", "rg_mount": "right_hand_link"})

    def test_rendered_projection_matches_camera_info(self):
        target = np.array([0.12, 0.07, 1.0])
        model, data, _, depth = self._placed(target)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[4] = 1
        opt.sitegroup[:] = 0
        world = data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "marker")]
        for scale in (0.5, 1.0):
            spec = rp.camera_specs(model, [("head", "head_cam")], self.profile, scale, (256, 256))[0]
            for name, intr in (("head_cam", spec.color), (depth, spec.depth_image)):
                p = np.linalg.inv(_optical(model, data, name)) @ np.append(world, 1.0)
                expected = (intr.fx * p[0] / p[2] + intr.cx, intr.fy * p[1] / p[2] + intr.cy)
                try:
                    renderer = mujoco.Renderer(model, intr.height, intr.width)
                except Exception as e:  # no offscreen GL context in this environment
                    self.skipTest(f"no offscreen renderer: {e}")
                try:
                    renderer.update_scene(data, camera=name, scene_option=opt)
                    img = renderer.render().astype(int)
                finally:
                    renderer.close()
                ys, xs = np.nonzero((img[:, :, 0] > 120) & (img[:, :, 1] < 60))
                self.assertGreater(len(xs), 5, (scale, name))
                self.assertAlmostEqual(xs.mean(), expected[0], delta=0.5, msg=(scale, name))
                self.assertAlmostEqual(ys.mean(), expected[1], delta=0.5, msg=(scale, name))

    def test_colored_cloud_layout_and_texture(self):
        intr = rp.Intrinsics(8, 6, 4.0, 4.0, 3.5, 2.5)
        depth = np.full((6, 8), 2.0, dtype=np.float32)
        depth[0, 0] = 0.0                                # invalid pixel: dropped
        rgb = np.zeros((6, 8, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(8)[None, :] * 30         # red encodes the column
        rgb[..., 2] = 7
        cloud = rp.colored_cloud(depth, intr, rgb, intr, np.eye(4))
        self.assertEqual(rp.CLOUD_DTYPE.itemsize, 20)
        self.assertEqual(cloud.size, 47)
        col = cloud["rgb"].view(np.uint32)
        u = np.rint(cloud["x"] / cloud["z"] * intr.fx + intr.cx).astype(int)
        np.testing.assert_array_equal((col >> 16) & 255, u * 30)
        np.testing.assert_array_equal(col & 255, 7)
        np.testing.assert_allclose(cloud["z"], 2.0)
        # A colour camera shifted 0.5 m right sees only part of the cloud.
        shifted = np.eye(4)
        shifted[0, 3] = -0.5
        self.assertLess(rp.colored_cloud(depth, intr, rgb, intr, shifted).size, cloud.size)


if __name__ == "__main__":
    unittest.main()
