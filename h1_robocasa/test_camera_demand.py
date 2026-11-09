"""Camera publication regression tests without ROS, DDS or a GL context."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


def render_method():
    path = Path(__file__).with_name("mujoco_ros_bridge.py")
    source = path.read_text()
    tree = ast.parse(source)
    bridge = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RosSensorBridge")
    method = next(n for n in bridge.body if isinstance(n, ast.FunctionDef) and
                  n.name == "_render_and_publish_cameras")
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    namespace = {"np": np, "realsense_profile": SimpleNamespace(colored_cloud=Mock()),
                 "PointCloud2": SimpleNamespace, "CLOUD_FIELDS": [], "_array": lambda _, data: data}
    namespace["realsense_profile"].CLOUD_DTYPE = np.dtype([("x", "<f4")])
    namespace["realsense_profile"].colored_cloud.return_value = np.array(
        [(1.25,), (2.5,)], dtype=namespace["realsense_profile"].CLOUD_DTYPE)
    code = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace[method.name], namespace


class CameraDemandTests(unittest.TestCase):
    def setUp(self):
        self.render, self.namespace = render_method()
        self.namespace["PointCloud2"] = lambda: SimpleNamespace(header=SimpleNamespace())
        self.cloud = self.namespace["realsense_profile"].colored_cloud
        self.pubs = {name: Mock() for name in
                     ("color", "color_jpeg", "color_info", "aligned", "aligned_png",
                      "aligned_info", "depth", "depth_png", "depth_info", "points")}
        for pub in self.pubs.values():
            pub.get_subscription_count.return_value = 0
        intrinsics = SimpleNamespace(width=4, height=2)
        spec = SimpleNamespace(color=intrinsics, depth_image=intrinsics,
                               color_frame="color_optical", depth_frame="depth_optical",
                               T_color_depth=np.eye(4))
        cam = SimpleNamespace(spec=spec, pubs=self.pubs, color_id=1, depth_id=2,
                              color_info=object(), depth_info=object())
        self.renderer = Mock()
        self.renderer.render.side_effect = lambda out: out
        self.rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        self.depth = np.full((2, 4), 1250, dtype=np.float32)
        self.bridge = SimpleNamespace(
            _ensure_renderer=Mock(), cameras=[cam], timings=Mock(),
            _renderer={(4, 2): self.renderer}, _buffers={(4, 2): (self.rgb, self.depth)},
            _scene_opt=object(), _depth_mm=lambda value: value.astype(np.uint16),
            _image_msg=lambda stamp, frame, pixels, encoding, stride:
                (stamp, frame, pixels.copy(), encoding, stride),
            _publish_info=Mock(), depth_stream=True, pointcloud=True)
        self.stamp = object()

    def test_unsubscribed_cloud_skips_only_cloud(self):
        self.render(self.bridge, self.stamp, object())
        self.cloud.assert_not_called()
        self.pubs["points"].publish.assert_not_called()
        for name in ("color", "aligned", "depth"):
            self.pubs[name].publish.assert_called_once()
        self.assertEqual(self.renderer.render.call_count, 3)
        self.assertEqual(self.bridge._publish_info.call_count, 3)
        color = self.pubs["color"].publish.call_args.args[0]
        self.assertIs(color[0], self.stamp)
        np.testing.assert_array_equal(color[2], self.rgb)

    def test_new_subscriber_gets_cloud_on_next_frame(self):
        self.render(self.bridge, self.stamp, object())
        self.pubs["points"].get_subscription_count.return_value = 1
        self.render(self.bridge, self.stamp, object())
        self.cloud.assert_called_once()
        np.testing.assert_array_equal(self.cloud.call_args.args[0], self.depth / 1000)
        np.testing.assert_array_equal(self.cloud.call_args.args[2], self.rgb)
        message = self.pubs["points"].publish.call_args.args[0]
        self.assertIs(message.header.stamp, self.stamp)
        self.assertEqual(message.header.frame_id, "depth_optical")
        self.assertEqual((message.width, message.height, message.point_step, message.row_step),
                         (2, 1, 4, 8))
        self.assertEqual(message.data, self.cloud.return_value.tobytes())
        self.assertTrue(message.is_dense)
        self.assertFalse(message.is_bigendian)

    def test_cloud_disabled_stays_disabled_with_readers(self):
        self.bridge.pointcloud = False
        self.pubs["points"].get_subscription_count.return_value = 1
        self.render(self.bridge, self.stamp, object())
        self.cloud.assert_not_called()
        self.pubs["points"].publish.assert_not_called()

    def test_disconnected_cloud_stops_conversion_not_depth(self):
        self.pubs["points"].get_subscription_count.return_value = 1
        self.render(self.bridge, self.stamp, object())
        self.pubs["points"].get_subscription_count.return_value = 0
        self.render(self.bridge, self.stamp, object())
        self.cloud.assert_called_once()
        self.pubs["points"].publish.assert_called_once()
        self.assertEqual(self.pubs["depth"].publish.call_count, 2)

    def test_subscribed_cloud_matches_real_projection(self):
        import realsense_profile

        self.namespace["realsense_profile"] = realsense_profile
        spec = self.bridge.cameras[0].spec
        spec.color = realsense_profile.fovy_intrinsics(4, 2, 60)
        spec.depth_image = spec.color
        self.pubs["points"].get_subscription_count.return_value = 1
        self.render(self.bridge, self.stamp, object())
        expected = realsense_profile.colored_cloud(
            self.depth / 1000, spec.depth_image, self.rgb, spec.color, spec.T_color_depth)
        message = self.pubs["points"].publish.call_args.args[0]
        self.assertEqual(message.data, expected.tobytes())
        self.assertEqual(message.point_step, realsense_profile.CLOUD_DTYPE.itemsize)
        self.assertEqual(message.row_step, expected.size * message.point_step)


if __name__ == "__main__":
    unittest.main()
