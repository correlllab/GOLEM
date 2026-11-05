"""Transport contract with fake ROS modules, no nodes or DDS participants."""

from types import SimpleNamespace as NS, ModuleType
from unittest.mock import MagicMock, patch
import unittest
import sys

import numpy as np

from model_server import aruco_localizer as localizer
from model_server.aruco_geometry import PoseRejected


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

    def declare_parameter(self, name, value):
        self.params[name] = value

    def get_parameter(self, name):
        return NS(value=self.params[name])

    def create_publisher(self, kind, topic, depth):
        self.publishers[topic] = MagicMock()
        return self.publishers[topic]

    def create_subscription(self, *args):
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
            np.testing.assert_array_equal(self.estimator.estimate.call_args.args[4], accepted)
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
            np.testing.assert_array_equal(self.estimator.estimate.call_args.args[4], prior)
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
        self.buffer.lookup_transform.side_effect = None
        self.node.now_ns += 100_000_000
        self.node.heartbeat()
        self.assertTrue(self.node.valid)
        self.node.pose_pub.publish.assert_called_once()
        self.assertEqual(self.node.pose_pub.publish.call_args.args[0].header.stamp.sec, 10)
        self.assertEqual(self.buffer.lookup_transform.call_args.args[2], (10, 0))
        self.estimator.detect.assert_called_once()
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


if __name__ == "__main__":
    unittest.main()
