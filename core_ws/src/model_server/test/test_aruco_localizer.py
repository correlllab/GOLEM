"""Transport contract with fake ROS modules, no nodes or DDS participants."""

from types import SimpleNamespace as NS, ModuleType
from unittest.mock import MagicMock, patch
import unittest
import sys

import numpy as np

from model_server import aruco_localizer as localizer
from model_server.aruco_geometry import BoardEstimator, PoseRejected, marker_points, pose_distance, quaternion_xyzw


def header(sec=10, frame="head_color_optical_frame", nanosec=0):
    return NS(stamp=NS(sec=sec, nanosec=nanosec), frame_id=frame)


class Pose:
    def __init__(self):
        self.header = header()
        self.pose = NS(position=NS(x=0, y=0, z=0), orientation=NS(x=0, y=0, z=0, w=1))


class FakeNode:
    def __init__(self, name):
        self.params = {}
        self.now_ns = 10_000_000_000
        self.publishers = {}
        self.subscriptions = []

    def declare_parameter(self, name, value):
        self.params[name] = value

    def get_parameter(self, name):
        return NS(value=self.params[name])

    def create_publisher(self, kind, topic, depth):
        self.publishers[topic] = MagicMock()
        return self.publishers[topic]

    def create_subscription(self, *args):
        self.subscriptions.append(args)
        return MagicMock()

    def create_timer(self, *args):
        return MagicMock()

    def get_clock(self):
        return NS(now=lambda: NS(nanoseconds=self.now_ns, to_msg=lambda: NS(sec=self.now_ns // 10**9, nanosec=0)))


class LocalizerTransportTests(unittest.TestCase):
    def setUp(self):
        modules = {}
        for name in ("rclpy", "rclpy.node", "rclpy.qos", "rclpy.time", "rclpy.duration",
                     "sensor_msgs", "sensor_msgs.msg", "geometry_msgs", "geometry_msgs.msg",
                     "diagnostic_msgs", "diagnostic_msgs.msg", "cv_bridge", "tf2_ros"):
            modules[name] = ModuleType(name)
        modules["rclpy.node"].Node = FakeNode
        modules["rclpy.qos"].qos_profile_sensor_data = object()
        modules["rclpy.time"].Time = NS(from_msg=lambda stamp: (stamp.sec, stamp.nanosec))
        modules["rclpy.duration"].Duration = lambda **kw: kw
        modules["sensor_msgs.msg"].Image = NS
        modules["sensor_msgs.msg"].CompressedImage = type("CompressedImage", (), {})
        modules["sensor_msgs.msg"].CameraInfo = NS
        modules["geometry_msgs.msg"].PoseStamped = Pose
        modules["diagnostic_msgs.msg"].DiagnosticArray = lambda: NS(header=header(), status=[])
        modules["diagnostic_msgs.msg"].DiagnosticStatus = type("Status", (), {"OK": 0, "WARN": 1})
        modules["diagnostic_msgs.msg"].KeyValue = lambda **kw: NS(**kw)
        bridge = MagicMock()
        bridge.imgmsg_to_cv2.return_value = np.zeros((360, 640, 3), np.uint8)
        modules["cv_bridge"].CvBridge = lambda: bridge
        buffer = MagicMock()
        buffer.lookup_transform.return_value = NS(transform=NS(
            translation=NS(x=1., y=2., z=3.), rotation=NS(x=0., y=0., z=0., w=1.)))
        modules["tf2_ros"].Buffer = lambda: buffer
        modules["tf2_ros"].TransformListener = MagicMock()
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        estimator = MagicMock()
        estimator.detect.return_value = ([], np.array([[7]]), [])
        matrix = np.eye(4)
        matrix[:3, :3] = np.diag([1, -1, -1])
        matrix[2, 3] = .8
        estimator.estimate.return_value = matrix, {"marker_count": 1, "marker_ids": [7]}
        with patch.object(localizer, "BoardEstimator", return_value=estimator):
            self.node = localizer.create_node()
        self.estimator, self.buffer, self.bridge = estimator, buffer, bridge
        self.info = NS(header=header(), width=640, height=360,
                       k=[500., 0, 320, 0, 500, 180, 0, 0, 1], d=[0.] * 5,
                       distortion_model="plumb_bob", p=[500., 0, 320, 0, 0, 500, 180, 0, 0, 0, 1, 0])
        self.image = NS(header=header())

    def observe(self):
        self.node.on_info(self.info)
        self.node.on_image(self.image)

    def advance_frame(self, delta_ns=66_666_667):
        self.node.now_ns += delta_ns
        sec, nanosec = divmod(self.node.now_ns, 10**9)
        self.image.header = header(sec, nanosec=nanosec)
        self.info.header = header(sec, nanosec=nanosec)

    def test_exact_pair_both_arrival_orders_stamp_and_tf(self):
        self.node.on_image(self.image)
        self.node.pose_pub.publish.assert_not_called()
        self.node.on_info(self.info)
        self.assertTrue(self.node.valid)
        message = self.node.pose_pub.publish.call_args.args[0]
        self.assertEqual(message.header.frame_id, "pelvis")
        self.assertIs(message.header.stamp, self.image.header.stamp)
        self.assertAlmostEqual(message.pose.position.z, 3.8)
        self.assertEqual(self.buffer.lookup_transform.call_args.args[2], (10, 0))
        self.assertEqual(self.buffer.lookup_transform.call_args.args[:2], ("pelvis", "head_color_optical_frame"))

    def test_transport_subscription_default_raw_and_explicit_compressed(self):
        raw = self.node.subscriptions[-1]
        self.assertIs(raw[0], NS)
        self.assertEqual(raw[1], '/realsense/head/color/image_raw')
        original = FakeNode.declare_parameter
        def compressed(node, name, value):
            original(node, name, 'compressed' if name == 'image_transport' else value)
        with patch.object(FakeNode, 'declare_parameter', compressed), \
                patch.object(localizer, 'BoardEstimator', return_value=self.estimator):
            node = localizer.create_node()
        subscription = node.subscriptions[-1]
        self.assertEqual(subscription[0].__name__, 'CompressedImage')
        self.assertEqual(subscription[1], '/realsense/head/color/image_raw/compressed')

    def jpeg_image(self):
        import cv2
        bgr = np.full((360, 640, 3), (20, 80, 180), dtype=np.uint8)
        ok, data = cv2.imencode('.jpg', bgr)
        self.assertTrue(ok)
        return NS(header=header(), format='rgb8; jpeg compressed bgr8', data=data.tobytes())

    def test_unknown_transport_rejected_before_subscriptions(self):
        original = FakeNode.declare_parameter
        def invalid(node, name, value):
            original(node, name, 'compressedDepth' if name == 'image_transport' else value)
        with patch.object(FakeNode, 'declare_parameter', invalid):
            with self.assertRaisesRegex(ValueError, 'image_transport must be'):
                localizer.create_node()

    def test_compressed_exact_pair_decodes_rgb_and_preserves_stamp_tf(self):
        self.node.settings['image_transport'] = 'compressed'
        self.image = self.jpeg_image()
        self.node.on_image(self.image)
        self.node.pose_pub.publish.assert_not_called()
        self.node.on_info(self.info)
        self.assertTrue(self.node.valid)
        self.bridge.imgmsg_to_cv2.assert_not_called()
        rgb = self.estimator.detect.call_args.args[0]
        np.testing.assert_allclose(rgb[0, 0], [180, 80, 20], atol=2)
        self.assertIs(self.node.pose_pub.publish.call_args.args[0].header.stamp, self.image.header.stamp)
        self.assertEqual(self.buffer.lookup_transform.call_args.args[2], (10, 0))

    def test_compressed_rejects_bad_format_payload_and_resolution(self):
        for invalid in ('format', 'data', 'resolution'):
            with self.subTest(invalid=invalid):
                self.node.settings['image_transport'] = 'compressed'
                self.node.processed_stamp = None
                self.image = self.jpeg_image()
                if invalid == 'format':
                    self.image.format = '16UC1; compressedDepth png'
                elif invalid == 'data':
                    self.image.data = b'not JPEG'
                else:
                    self.info.width = 1280
                self.observe()
                self.assertFalse(self.node.valid)
                self.node.pose_pub.publish.assert_not_called()

    def test_compressed_does_not_relax_matching_age_or_geometry(self):
        self.node.settings['image_transport'] = 'compressed'
        self.image = self.jpeg_image()
        self.info.header = header(frame='wrong_optical')
        self.observe()
        self.estimator.detect.assert_not_called()
        self.info.header = self.image.header
        self.node.now_ns += 600_000_000
        self.node.on_info(self.info)
        self.assertIn('stale image timestamp', self.node.reason)
        self.node.pose_pub.publish.assert_not_called()
        self.node.now_ns = 10_000_000_000
        self.estimator.estimate.side_effect = PoseRejected('unresolved planar ambiguity', {})
        self.observe()
        self.assertIn('unresolved planar ambiguity', self.node.reason)
        self.node.pose_pub.publish.assert_not_called()

    def test_unmatched_stamp_and_frame_never_publish(self):
        self.node.on_info(self.info)
        self.node.on_image(NS(header=header(sec=9)))
        self.node.on_image(NS(header=header(frame="other_optical")))
        self.node.pose_pub.publish.assert_not_called()
        self.assertFalse(self.node.valid)

    def test_tf_failure_and_bad_calibration_invalidate(self):
        self.buffer.lookup_transform.side_effect = RuntimeError("no transform at image time")
        self.observe()
        self.node.pose_pub.publish.assert_not_called()
        self.assertFalse(self.node.valid)
        self.assertIn("no transform", self.node.reason)
        self.buffer.lookup_transform.side_effect = None
        self.advance_frame()
        self.info.width = 1280
        self.observe()
        self.assertIn("resolution", self.node.reason)

    def test_stale_heartbeat_never_restamps_or_republishes(self):
        self.observe()
        self.node.now_ns += 600_000_000
        self.node.heartbeat()
        self.assertFalse(self.node.valid)
        self.assertIsNone(self.node.prior)
        self.assertEqual(self.node.pose_pub.publish.call_count, 1)
        self.assertEqual(self.node.pose_pub.publish.call_args.args[0].header.stamp.sec, 10)
        values = self.node.diagnostics_pub.publish.call_args.args[0].status[0].values
        self.assertEqual({v.key: v.value for v in values}["valid"], "false")

    def test_future_stale_and_invalid_observations(self):
        for sec in (0, 9, 11):
            self.node.process(NS(header=header(sec)), self.info)
        self.node.pose_pub.publish.assert_not_called()
        self.estimator.estimate.side_effect = ValueError("reprojection bound")
        self.observe()
        self.assertIn("reprojection", self.node.reason)
        self.assertFalse(self.node.valid)

    def test_rectified_requires_explicit_p_and_zero_distortion(self):
        self.node.settings["image_rectified"] = True
        self.info.d = [float("nan")] * 5
        self.observe()
        self.assertTrue(self.node.valid)
        self.assertEqual(len(self.estimator.estimate.call_args.args[3]), 0)

    def test_clock_reset_invalidates_without_publish(self):
        self.observe()
        self.node.now_ns = 1_000_000_000
        self.node.heartbeat()
        self.assertFalse(self.node.valid)
        self.assertIsNone(self.node.last_stamp)
        self.buffer.clear.assert_called_once()
        self.image.header = self.info.header = header(1)
        self.observe()
        self.assertTrue(self.node.valid)
        self.assertEqual(self.node.pose_pub.publish.call_count, 2)

    def test_duplicate_stamp_never_republishes(self):
        self.observe()
        self.node.on_image(self.image)
        self.node.on_image(self.image)
        self.node.pose_pub.publish.assert_called_once()
        self.assertTrue(self.node.valid)

    def test_image_first_15hz_pairs_retain_prior_and_validity(self):
        self.observe()
        accepted = self.node.prior.copy()
        for _ in range(4):
            self.advance_frame()
            last_stamp = self.node.last_stamp
            self.node.on_image(self.image)
            self.assertTrue(self.node.valid)
            self.assertEqual(self.node.last_stamp, last_stamp)
            np.testing.assert_array_equal(self.node.prior, accepted)
            self.node.on_info(self.info)
            self.assertTrue(self.node.valid)
            np.testing.assert_allclose(self.estimator.estimate.call_args.args[4], accepted, atol=1e-14, rtol=0)
            self.assertEqual(self.node.last_stamp, self.node.now_ns)
        for call in self.node.diagnostics_pub.publish.call_args_list:
            values = {v.key: v.value for v in call.args[0].status[0].values}
            if values["observation_stamp_ns"] != "0":
                self.assertEqual(values["valid"], "true")

    def test_info_first_and_delayed_older_pairs_do_not_mix_or_invalidate(self):
        self.observe()
        self.advance_frame()
        self.node.on_info(self.info)
        self.assertTrue(self.node.valid)
        self.node.on_image(self.image)
        stamp = self.node.last_stamp
        for h in (header(9), header(10), self.image.header):
            self.node.on_image(NS(header=h))
            self.info.header = h
            self.node.on_info(self.info)
        self.assertEqual(self.node.last_stamp, stamp)
        self.assertTrue(self.node.valid)
        self.assertEqual(self.node.pose_pub.publish.call_count, 2)
        self.assertEqual(self.buffer.lookup_transform.call_args.args[2], (10, 66_666_667))

    def test_interleaved_future_info_does_not_pair_with_older_image(self):
        self.observe()
        self.advance_frame()
        earlier_image = NS(header=self.image.header)
        earlier_info = NS(**vars(self.info))
        self.advance_frame()
        self.node.on_info(self.info)
        self.node.on_image(earlier_image)
        self.assertEqual(self.estimator.estimate.call_count, 1)
        self.assertTrue(self.node.valid)
        self.node.on_info(earlier_info)
        self.assertEqual(self.buffer.lookup_transform.call_args.args[2], (10, 66_666_667))
        self.node.on_image(self.image)
        self.assertEqual(self.buffer.lookup_transform.call_args.args[2], (10, 133_333_334))
        self.assertEqual(self.node.pose_pub.publish.call_count, 3)

    def test_unmatched_other_frame_cannot_supply_calibration(self):
        self.observe()
        self.advance_frame()
        wrong = NS(**vars(self.info))
        wrong.header = header(10, frame="different_optical", nanosec=66_666_667)
        self.node.on_info(wrong)
        self.node.on_image(self.image)
        self.assertEqual(self.estimator.estimate.call_count, 1)
        self.assertTrue(self.node.valid)
        self.node.on_info(self.info)
        self.assertEqual(self.estimator.estimate.call_count, 2)
        self.assertIsNotNone(self.estimator.estimate.call_args.args[4])

    def test_unpaired_new_frame_ages_latest_without_refresh(self):
        self.observe()
        self.advance_frame()
        self.node.on_image(self.image)
        self.assertTrue(self.node.valid)
        self.node.now_ns += 500_000_000
        self.node.heartbeat()
        self.assertFalse(self.node.valid)
        self.assertIsNone(self.node.prior)
        self.assertEqual(self.node.last_stamp, 10_000_000_000)
        self.node.pose_pub.publish.assert_called_once()

    def test_rejected_ambiguity_or_jump_invalidates_without_refreshing_age(self):
        for reason in ("unresolved planar ambiguity", "pose continuity bound exceeded"):
            with self.subTest(reason=reason):
                self.estimator.estimate.side_effect = None
                self.observe()
                accepted_stamp = self.node.last_stamp
                count = self.node.pose_pub.publish.call_count
                self.advance_frame()
                self.estimator.estimate.side_effect = PoseRejected(reason, {"ambiguity_gap_px": .04})
                self.node.on_image(self.image)
                self.assertTrue(self.node.valid)
                self.node.on_info(self.info)
                self.assertFalse(self.node.valid)
                self.assertIn(reason, self.node.reason)
                self.assertEqual(self.node.last_stamp, accepted_stamp)
                self.assertIsNotNone(self.node.prior)
                self.assertEqual(self.node.pose_pub.publish.call_count, count)
                values = self.node.diagnostics_pub.publish.call_args.args[0].status[0].values
                self.assertEqual({v.key: v.value for v in values}["valid"], "false")
                self.advance_frame()

    def test_repeated_rejected_jump_cannot_become_an_unguarded_cold_start(self):
        self.observe()
        stamp, prior = self.node.last_stamp, self.node.prior.copy()
        self.estimator.estimate.side_effect = PoseRejected("pose continuity bound exceeded", {})
        for _ in range(3):
            self.advance_frame()
            self.observe()
            self.assertFalse(self.node.valid)
            np.testing.assert_allclose(self.estimator.estimate.call_args.args[4], prior, atol=1e-14, rtol=0)
            self.assertEqual(self.node.last_stamp, stamp)
            self.node.pose_pub.publish.assert_called_once()
        self.node.now_ns += 600_000_000
        self.node.heartbeat()
        self.assertIsNone(self.node.prior)

    def test_calibration_change_and_nonfinite_pose_discard_prior(self):
        self.observe()
        self.advance_frame()
        self.info.k[0] += 1
        self.observe()
        self.assertIsNone(self.estimator.estimate.call_args.args[4])
        self.advance_frame()
        pose, quality = self.estimator.estimate.return_value
        bad = pose.copy()
        bad[0, 3] = float("nan")
        self.estimator.estimate.return_value = bad, quality
        self.observe()
        self.assertFalse(self.node.valid)
        self.assertIsNone(self.node.prior)
        self.assertIsNone(self.node.prior_target)

    def test_missing_tf_preserves_bounded_last_accepted_prior(self):
        self.observe()
        self.advance_frame()
        self.buffer.lookup_transform.side_effect = RuntimeError("TF not yet available")
        self.observe()
        self.assertTrue(self.node.valid)
        self.assertIsNotNone(self.node.prior)
        self.advance_frame()
        self.buffer.lookup_transform.side_effect = None
        self.observe()
        self.assertTrue(self.node.valid)
        self.assertIsNotNone(self.estimator.estimate.call_args.args[4])

    def test_delayed_tf_retries_without_redetection_and_keeps_original_stamp(self):
        self.buffer.lookup_transform.side_effect = RuntimeError("future extrapolation")
        self.observe()
        self.assertFalse(self.node.valid)
        self.assertEqual(len(self.node.pending_tf), 1)
        self.node.pose_pub.publish.assert_not_called()
        self.estimator.estimate.assert_not_called()
        self.buffer.lookup_transform.side_effect = None
        self.node.now_ns += 100_000_000
        self.node.heartbeat()
        self.assertTrue(self.node.valid)
        self.node.pose_pub.publish.assert_called_once()
        self.assertEqual(self.node.pose_pub.publish.call_args.args[0].header.stamp.sec, 10)
        self.assertEqual(self.buffer.lookup_transform.call_args.args[2], (10, 0))
        self.estimator.detect.assert_called_once()
        self.bridge.imgmsg_to_cv2.assert_called_once()
        self.estimator.estimate.assert_called_once()
        self.assertEqual(len(self.node.pending_tf), 0)

    def test_pending_tf_expiry_clock_reset_and_rejection_cannot_resurrect_pose(self):
        for mode in ('expiry', 'reset', 'rejection'):
            with self.subTest(mode=mode):
                self.setUp()
                self.buffer.lookup_transform.side_effect = RuntimeError("not yet")
                self.observe()
                if mode == 'expiry':
                    self.node.now_ns += 600_000_000
                    self.node.heartbeat()
                elif mode == 'reset':
                    self.node.now_ns = 1_000_000_000
                    self.node.heartbeat()
                else:
                    self.advance_frame()
                    self.estimator.estimate.side_effect = PoseRejected('ambiguous', {})
                    self.observe()
                self.buffer.lookup_transform.side_effect = None
                self.node.heartbeat()
                self.node.pose_pub.publish.assert_not_called()
                self.assertFalse(self.node.valid)
                self.assertEqual(len(self.node.pending_tf), 0)

    def test_pending_newer_tf_does_not_revoke_fresh_accepted_pose(self):
        self.observe()
        self.advance_frame()
        self.buffer.lookup_transform.side_effect = RuntimeError('future extrapolation')
        self.observe()
        self.assertTrue(self.node.valid)
        self.assertEqual(self.node.last_stamp, 10_000_000_000)
        self.node.pose_pub.publish.assert_called_once()
        self.node.now_ns += 600_000_000
        self.node.heartbeat()
        self.assertFalse(self.node.valid)

    def test_pending_queue_is_bounded(self):
        self.buffer.lookup_transform.side_effect = RuntimeError('not yet')
        for _ in range(20):
            self.observe()
            self.advance_frame(1_000_000)
        self.assertLessEqual(len(self.node.pending_tf), 10)
        self.node.pose_pub.publish.assert_not_called()

    def test_calibration_transition_drops_pending_old_epoch_before_tf_retry(self):
        self.buffer.lookup_transform.side_effect = RuntimeError('pending A')
        self.observe()
        old_stamp = self.node.now_ns
        old_observation = self.node.pending_tf[old_stamp]
        self.advance_frame()
        self.info.k = [510., 0, 320., 0, 510., 240., 0, 0, 1.]
        available = self.buffer.lookup_transform.return_value

        def lookup(*args, **kwargs):
            if args[2] == (10, 0):
                return available
            raise RuntimeError('pending B')

        self.buffer.lookup_transform.side_effect = lookup
        self.observe()
        self.node.pose_pub.publish.assert_not_called()
        self.assertIsNone(self.node.prior_target)
        self.assertNotIn(old_stamp, self.node.pending_tf)
        # The retry path also independently rejects a foreign calibration epoch.
        self.node.pending_tf[old_stamp] = old_observation
        self.node.heartbeat()
        self.node.pose_pub.publish.assert_not_called()
        self.assertNotIn(old_stamp, self.node.pending_tf)
        self.buffer.lookup_transform.side_effect = None
        self.node.heartbeat()
        self.assertTrue(self.node.valid)
        self.assertEqual(self.node.last_stamp, self.node.now_ns)
        self.assertEqual(self.node.prior_key, self.node.active_calibration_key)

    def projected_frame(self, camera, board, indices=(0, 1), noise=None):
        import cv2
        estimator = BoardEstimator()
        optical = np.linalg.inv(camera) @ board
        markers = [estimator.layout['markers'][i] for i in indices]
        corners = [cv2.projectPoints(marker_points(m), cv2.Rodrigues(optical[:3, :3])[0],
                                    optical[:3, 3], np.array(self.info.k).reshape(3, 3),
                                    np.array(self.info.d))[0].reshape(1, 4, 2) for m in markers]
        if noise is not None:
            corners = [c + noise.normal(0, .1, c.shape) for c in corners]
        ids = np.array([[m['id']] for m in markers])
        estimator.detect = MagicMock(return_value=(corners, ids, []))
        estimator.estimate = MagicMock(wraps=estimator.estimate)
        self.node.estimator = estimator
        q = quaternion_xyzw(camera[:3, :3])
        self.buffer.lookup_transform.return_value = NS(transform=NS(
            translation=NS(**dict(zip(('x', 'y', 'z'), camera[:3, 3]))),
            rotation=NS(**dict(zip(('x', 'y', 'z', 'w'), q)))))
        return optical, corners, ids, estimator

    def camera_yaw_and_board(self):
        import cv2
        camera, board = np.eye(4), np.eye(4)
        camera[:3, :3] = cv2.Rodrigues(np.array([0., np.deg2rad(20), 0.]))[0]
        camera[:3, 3] = [.03, -.02, .01]
        board[:3, :3] = cv2.Rodrigues(np.array([.2, -.15, .08]))[0] @ np.diag([1., -1., -1.])
        board[:3, 3] = [.02, -.03, .7]
        return camera, board

    def test_camera_yaw_compensation_accepts_stationary_target_board(self):
        camera, board = self.camera_yaw_and_board()
        self.projected_frame(np.eye(4), board)
        self.observe()
        old_optical = self.node.prior.copy()
        self.advance_frame()
        optical, corners, ids, estimator = self.projected_frame(camera, board)
        self.assertGreater(pose_distance(optical, old_optical)[0], .036)
        with self.assertRaisesRegex(PoseRejected, 'continuity'):
            estimator.estimate(corners, ids, np.array(self.info.k).reshape(3, 3), np.array(self.info.d), old_optical)
        self.observe()
        self.assertTrue(self.node.valid)
        np.testing.assert_allclose(self.node.prior_target, board, atol=1e-10)
        np.testing.assert_allclose(estimator.estimate.call_args.args[4], optical, atol=1e-10)
        self.assertLess(self.node.metrics['prior_translation_delta_m'], 1e-10)
        self.assertLess(self.node.metrics['prior_rotation_delta_rad'], 1e-6)
        self.assertEqual(self.buffer.lookup_transform.call_args.args[2], (10, 66_666_667))
        self.assertEqual(self.node.pose_pub.publish.call_count, 2)

    def test_compensated_prior_still_rejects_target_translation_and_rotation_jumps(self):
        import cv2
        camera, board = self.camera_yaw_and_board()
        self.projected_frame(np.eye(4), board)
        self.observe()
        accepted = self.node.prior_target.copy()
        for shift, angle in ((.021, 0.), (0., .101)):
            with self.subTest(shift=shift, angle=angle):
                self.advance_frame()
                moved = board.copy()
                moved[0, 3] += shift
                moved[:3, :3] = cv2.Rodrigues(np.array([0., 0., angle]))[0] @ moved[:3, :3]
                self.projected_frame(camera, moved)
                self.observe()
                self.assertFalse(self.node.valid)
                self.assertIn('continuity', self.node.reason)
                np.testing.assert_array_equal(self.node.prior_target, accepted)
                self.assertEqual(self.node.last_stamp, 10_000_000_000)
                self.node.pose_pub.publish.assert_called_once()

    def test_ambiguous_single_marker_requires_tf_compensated_qualified_prior(self):
        import cv2
        camera, _ = self.camera_yaw_and_board()
        optical = np.eye(4)
        optical[:3, :3] = cv2.Rodrigues(np.array([.12, .08, 0.]))[0] @ np.diag([1., -1., -1.])
        optical[:3, 3] = [-.135, .10, .8]
        self.info.k = [700., 0., 320., 0., 700., 240., 0., 0., 1.]
        board = camera @ optical
        self.projected_frame(np.eye(4), board)
        self.observe()
        accepted_stamp = self.node.last_stamp
        self.assertTrue(self.node.valid)
        self.advance_frame()
        _, corners, ids, estimator = self.projected_frame(camera, board, (1,), np.random.default_rng(3))
        with self.assertRaisesRegex(PoseRejected, 'no qualified recent prior'):
            estimator.estimate(corners, ids, np.array(self.info.k).reshape(3, 3), np.array(self.info.d))
        self.buffer.lookup_transform.side_effect = RuntimeError('exact TF pending')
        estimator.estimate.reset_mock()
        self.observe()
        estimator.estimate.assert_not_called()
        self.assertEqual(self.node.last_stamp, accepted_stamp)
        self.buffer.lookup_transform.side_effect = None
        self.node.heartbeat()
        self.assertTrue(self.node.valid)
        self.assertEqual(self.node.metrics['acceptance_reason'], 'bounded_prior')
        self.assertLess(self.node.metrics['ambiguity_gap_px'], .15)
        self.assertEqual(self.node.metrics['prior_bounded_candidate_count'], 1)
        self.node.estimator.detect.assert_called_once()

    def test_newer_tf_acceptance_cannot_be_overwritten_by_older_pending_frame(self):
        self.observe()
        self.advance_frame()
        old_stamp = self.node.now_ns
        old_time = (self.image.header.stamp.sec, self.image.header.stamp.nanosec)
        available = self.buffer.lookup_transform.return_value
        def lookup(*args, **kwargs):
            if args[2] == old_time:
                raise RuntimeError('older TF missing')
            return available
        self.buffer.lookup_transform.side_effect = lookup
        self.observe()
        self.advance_frame()
        self.observe()
        self.assertEqual(self.node.last_stamp, self.node.now_ns)
        target = self.node.prior_target.copy()
        self.assertIn(old_stamp, self.node.pending_tf)
        self.buffer.lookup_transform.side_effect = None
        self.node.now_ns = old_stamp + 501_000_000
        self.node.heartbeat()
        self.assertTrue(self.node.valid)
        self.assertEqual(self.node.last_stamp, old_stamp + 66_666_667)
        np.testing.assert_array_equal(self.node.prior_target, target)
        self.assertNotIn(old_stamp, self.node.pending_tf)
        self.assertEqual(self.node.pose_pub.publish.call_count, 2)

    def test_prior_target_clears_with_optical_anchor_on_reset_and_expiry(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                self.setUp()
                self.observe()
                self.assertIsNotNone(self.node.prior_target)
                self.node.now_ns = 1_000_000_000 if reset else 10_600_000_000
                self.node.heartbeat()
                self.assertIsNone(self.node.prior_target)
                self.assertIsNone(self.node.prior)

    def test_queued_detections_use_latest_accepted_target_anchor(self):
        camera, board = self.camera_yaw_and_board()
        self.projected_frame(np.eye(4), board)
        self.observe()
        self.buffer.lookup_transform.side_effect = RuntimeError('exact TF pending')
        transforms = {}
        estimators = []
        for shift in (.015, .030):
            self.advance_frame()
            moved = board.copy()
            moved[0, 3] += shift
            _, _, _, estimator = self.projected_frame(camera, moved)
            estimators.append(estimator)
            transforms[(self.image.header.stamp.sec, self.image.header.stamp.nanosec)] = self.buffer.lookup_transform.return_value
            self.observe()
            estimator.estimate.assert_not_called()
        self.assertEqual(len(self.node.pending_tf), 2)
        self.buffer.lookup_transform.side_effect = lambda *args, **kw: transforms[args[2]]
        self.node.heartbeat()
        self.assertTrue(self.node.valid)
        self.assertEqual(self.node.pose_pub.publish.call_count, 3)
        self.assertEqual(len(self.node.pending_tf), 0)
        np.testing.assert_allclose(self.node.prior_target, moved, atol=1e-10)
        self.assertAlmostEqual(self.node.metrics['prior_translation_delta_m'], .015)
        for estimator in estimators:
            estimator.detect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
