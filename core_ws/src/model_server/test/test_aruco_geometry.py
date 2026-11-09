"""Offline geometry and image tests; importing this never initializes ROS."""

import unittest

import cv2
import numpy as np

from model_server.aruco_geometry import (
    BoardEstimator, PoseRejected, calibration, load_layout, marker_points,
    pose_distance, quaternion_xyzw, transform_matrix,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.layout = load_layout()
        self.estimator = BoardEstimator(self.layout)
        self.k = np.array([[700., 0, 320], [0, 710, 240], [0, 0, 1]])
        self.d = np.array([.04, -.02, .001, -.002, .005])
        base = np.diag([1., -1., -1.])
        self.rotation = cv2.Rodrigues(np.array([.2, -.15, .08]))[0] @ base
        self.translation = np.array([.02, -.03, .7])
        self.rvec = cv2.Rodrigues(self.rotation)[0]

    def observations(self, indices=(0, 1)):
        markers = [self.layout["markers"][i] for i in indices]
        corners = [cv2.projectPoints(marker_points(m), self.rvec, self.translation,
                                     self.k, self.d)[0].reshape(1, 4, 2) for m in markers]
        return corners, np.array([[m["id"]] for m in markers])

    def test_joint_and_each_marker_fallback(self):
        # These near, oblique single-marker views have a measurable IPPE gap.
        self.rotation = cv2.Rodrigues(np.array([.6, -.4, .08]))[0] @ np.diag([1., -1., -1.])
        self.rvec = cv2.Rodrigues(self.rotation)[0]
        self.translation[2] = .45
        for indices in ((0, 1), (0,), (1,)):
            corners, ids = self.observations(indices)
            matrix, quality = self.estimator.estimate(corners, ids, self.k, self.d)
            np.testing.assert_allclose(matrix[:3, :3], self.rotation, atol=1e-6)
            np.testing.assert_allclose(matrix[:3, 3], self.translation, atol=1e-6)
            self.assertEqual(quality["marker_count"], len(indices))
            self.assertLess(quality["reprojection_rms_px"], 1e-6)

    def test_noisy_single_marker_ambiguity_requires_unique_bounded_prior(self):
        corners, ids = self.observations((0,))
        corners[0] += np.random.RandomState(6).normal(0, .1, (1, 4, 2))
        with self.assertRaisesRegex(PoseRejected, "no qualified recent prior") as rejected:
            self.estimator.estimate(corners, ids, self.k, self.d)
        quality = rejected.exception.quality
        self.assertEqual(quality["candidate_count"], 2)
        self.assertLess(quality["ambiguity_gap_px"], .15)
        self.assertLess(quality["reprojection_rms_px"], .15)
        prior = np.eye(4)
        prior[:3, :3], prior[:3, 3] = self.rotation, self.translation
        matrix, quality = self.estimator.estimate(corners, ids, self.k, self.d, prior)
        self.assertEqual(quality["acceptance_reason"], "bounded_prior")
        dt, dr = pose_distance(matrix, prior)
        self.assertLessEqual(dt, .02)
        self.assertLessEqual(dr, .1)
        # A loose prior that admits both branches cannot resolve ambiguity.
        estimator = BoardEstimator(max_prior_translation_m=.2, max_prior_rotation_rad=.5)
        with self.assertRaisesRegex(PoseRejected, "multiple branches"):
            estimator.estimate(corners, ids, self.k, self.d, prior)

    def test_two_marker_planar_ambiguity_also_fails_closed(self):
        rotation = cv2.Rodrigues(np.array([.01, -.01, 0.]))[0] @ np.diag([1., -1., -1.])
        rvec, translation = cv2.Rodrigues(rotation)[0], np.array([0., 0., 1.2])
        rng = np.random.RandomState(1)
        corners = [cv2.projectPoints(marker_points(m), rvec, translation, self.k, np.zeros(5))[0].reshape(1, 4, 2)
                   + rng.normal(0, .1, (1, 4, 2)) for m in self.layout["markers"]]
        with self.assertRaisesRegex(PoseRejected, "unresolved planar ambiguity") as rejected:
            self.estimator.estimate(corners, np.array([[7], [8]]), self.k, np.zeros(5))
        self.assertEqual(rejected.exception.quality["marker_count"], 2)
        self.assertLess(rejected.exception.quality["ambiguity_gap_px"], .15)

    def test_near_frontal_marker_noise_does_not_select_lower_rms_wrong_branch(self):
        k = np.array([[700., 0, 320], [0, 700, 240], [0, 0, 1]])
        rotation = cv2.Rodrigues(np.array([.12, .08, 0.]))[0] @ np.diag([1., -1., -1.])
        translation = np.array([-.135, .10, .8])
        points = cv2.projectPoints(marker_points(self.layout["markers"][1]),
                                  cv2.Rodrigues(rotation)[0], translation, k, np.zeros(5))[0].reshape(1, 4, 2)
        points += np.random.default_rng(3).normal(0, .1, points.shape)
        with self.assertRaisesRegex(PoseRejected, "unresolved planar ambiguity") as rejected:
            self.estimator.estimate([points], np.array([[8]]), k, np.zeros(5))
        self.assertAlmostEqual(rejected.exception.quality["ambiguity_gap_px"], .04658, delta=.001)
        prior = np.eye(4)
        prior[:3, :3], prior[:3, 3] = rotation, translation
        matrix, quality = self.estimator.estimate([points], np.array([[8]]), k, np.zeros(5), prior)
        self.assertEqual(quality["acceptance_reason"], "bounded_prior")
        # The qualified branch has a larger pixel residual, not the best RMS.
        self.assertGreater(quality["reprojection_rms_px"], rejected.exception.quality["reprojection_rms_px"])
        dt, dr = pose_distance(matrix, prior)
        self.assertLess(dt, .02)
        self.assertLess(dr, .1)

    def test_translation_rotation_jumps_and_nonfinite_prior_rejected(self):
        corners, ids = self.observations()
        prior, _ = self.estimator.estimate(corners, ids, self.k, self.d)
        for translation, angle in ((.03, 0), (0, .15)):
            bad = prior.copy()
            bad[0, 3] += translation
            bad[:3, :3] = cv2.Rodrigues(np.array([angle, 0., 0.]))[0] @ bad[:3, :3]
            with self.assertRaisesRegex(PoseRejected, "continuity bound"):
                self.estimator.estimate(corners, ids, self.k, self.d, bad)
        prior[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.estimator.estimate(corners, ids, self.k, self.d, prior)

    def test_unknown_duplicate_and_reprojection_rejection(self):
        corners, ids = self.observations()
        for bad_corners, bad_ids in (([], None), (corners[:1], np.array([[49]])),
                                     (corners, np.array([[7], [7]]))):
            with self.assertRaises(ValueError):
                self.estimator.estimate(bad_corners, bad_ids, self.k, self.d)
        corners[1][0, 0] += [25, -30]
        with self.assertRaises(ValueError):
            self.estimator.estimate(corners, ids, self.k, self.d)

    def test_raw_distortion_and_calibration_validation(self):
        calibration(self.k, self.d, 640, 480, (480, 640, 3))
        for k, d, shape, model in ((self.k * 0, self.d, (480, 640), "plumb_bob"),
                                    (self.k, self.d, (360, 640), "plumb_bob"),
                                    (self.k, [float("nan")] * 5, (480, 640), "plumb_bob"),
                                    (self.k, self.d, (480, 640), "equidistant")):
            with self.assertRaises(ValueError):
                calibration(k, d, 640, 480, shape, model)

    def test_transform_and_quaternion_half_turn(self):
        for rotation in (self.rotation, np.diag([1., -1., -1.]), np.eye(3)):
            result = transform_matrix([1., 2., 3.], quaternion_xyzw(rotation))
            np.testing.assert_allclose(result[:3, :3], rotation, atol=1e-7)
        with self.assertRaises(ValueError):
            transform_matrix([0, 0, 0], [0, 0, 0, 0])

    def test_back_facing_board_rejected(self):
        corners = [cv2.projectPoints(marker_points(m), np.array([.1, .2, 0.]),
                                     self.translation, self.k, self.d)[0].reshape(1, 4, 2)
                   for m in self.layout["markers"]]
        with self.assertRaises(ValueError):
            self.estimator.estimate(corners, np.array([[7], [8]]), self.k, self.d)

    def test_synthetic_image_detection_and_pose(self):
        # A rasterized perspective image exercises dictionary orientation and K.
        image = np.full((480, 640), 255, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        generate = (cv2.aruco.generateImageMarker if hasattr(cv2.aruco, "generateImageMarker")
                    else cv2.aruco.drawMarker)
        corners, _ = self.observations()
        for marker, pixels in zip(self.layout["markers"], corners):
            texture = generate(dictionary, marker["id"], 120)
            homography = cv2.getPerspectiveTransform(np.float32([[0, 0], [119, 0], [119, 119], [0, 119]]),
                                                    pixels.reshape(4, 2).astype(np.float32))
            warped = cv2.warpPerspective(texture, homography, (640, 480), borderValue=255)
            image = np.minimum(image, warped)
        detected, ids, _ = self.estimator.detect(image)
        self.assertEqual(sorted(ids.reshape(-1).tolist()), [7, 8])
        matrix, quality = self.estimator.estimate(detected, ids, self.k, self.d)
        np.testing.assert_allclose(matrix[:3, 3], self.translation, atol=.025)
        self.assertEqual(quality["marker_count"], 2)


if __name__ == "__main__":
    unittest.main()
