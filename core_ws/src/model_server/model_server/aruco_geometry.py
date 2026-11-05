"""ROS-independent calibrated board localization; no learned models."""

import json
from pathlib import Path

import cv2
import numpy as np


def load_layout(path=None):
    if path is None:
        source = Path(__file__).resolve().parents[1] / "config/pegboard_layout.json"
        if source.is_file():
            path = source
        else:
            from ament_index_python.packages import get_package_share_directory
            path = Path(get_package_share_directory("model_server")) / "config/pegboard_layout.json"
    layout = json.loads(Path(path).read_text())
    if layout["dictionary"] != "DICT_4X4_50":
        raise ValueError("unsupported dictionary")
    markers = layout["markers"]
    if not markers or len({m["id"] for m in markers}) != len(markers):
        raise ValueError("empty or duplicate marker layout")
    for marker in markers:
        p = np.asarray(marker["position_board_m"], dtype=float)
        if (p.shape != (3,) or not np.isfinite(p).all() or
                not 0 <= marker["id"] < 50 or
                not np.isfinite(marker["black_size_m"]) or marker["black_size_m"] <= 0):
            raise ValueError("invalid marker layout")
    if np.ptp([m["position_board_m"][2] for m in markers]) > 1e-9:
        raise ValueError("IPPE requires a planar layout")
    return layout


def marker_points(marker):
    h = marker["black_size_m"] / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=float) + marker["position_board_m"]


def calibration(k, d, width, height, image_shape, distortion_model="plumb_bob"):
    """Validate raw-image K/D. Rectified callers explicitly supply P and zero D."""
    k = np.asarray(k, dtype=float).reshape(3, 3)
    d = np.asarray(d, dtype=float).reshape(-1)
    if (tuple(image_shape[:2]) != (height, width) or width <= 0 or height <= 0 or
            not np.isfinite(k).all() or not np.isfinite(d).all() or
            k[0, 0] <= 0 or k[1, 1] <= 0 or
            not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-9):
        raise ValueError("invalid calibration or resolution mismatch")
    if distortion_model not in ("plumb_bob", "rational_polynomial") or len(d) not in (0, 4, 5, 8, 12, 14):
        raise ValueError("unsupported distortion model/coefficients")
    return k, d


class PoseRejected(ValueError):
    """A geometrically unresolved observation, with diagnostic measurements."""

    def __init__(self, reason, quality):
        super().__init__(reason)
        self.quality = dict(quality, acceptance_reason=reason)


def pose_distance(matrix, prior):
    """Return translation metres and geodesic rotation radians for proper SE(3)."""
    for pose in (matrix, prior):
        if (np.shape(pose) != (4, 4) or not np.isfinite(pose).all() or
                not np.allclose(pose[3], [0, 0, 0, 1]) or
                not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6) or
                not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6)):
            raise ValueError("nonfinite or invalid pose/prior")
    translation = float(np.linalg.norm(matrix[:3, 3] - prior[:3, 3]))
    rotation = float(np.arccos(np.clip((np.trace(matrix[:3, :3] @ prior[:3, :3].T) - 1) / 2, -1, 1)))
    return translation, rotation


class BoardEstimator:
    def __init__(self, layout=None, max_reprojection_px=2.0, min_marker_area_px=25.0,
                 min_ambiguity_gap_px=.15, max_prior_translation_m=.02,
                 max_prior_rotation_rad=.1):
        self.layout = layout or load_layout()
        if any(not np.isfinite(value) or value <= 0 for value in (
                max_reprojection_px, min_marker_area_px, min_ambiguity_gap_px,
                max_prior_translation_m, max_prior_rotation_rad)):
            raise ValueError("quality bounds must be finite and positive")
        self.max_reprojection_px = max_reprojection_px
        self.min_marker_area_px = min_marker_area_px
        self.min_ambiguity_gap_px = min_ambiguity_gap_px
        self.max_prior_translation_m = max_prior_translation_m
        self.max_prior_rotation_rad = max_prior_rotation_rad
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters")
                  else cv2.aruco.DetectorParameters_create())
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # A large window crosses the entire short edge of an oblique 3 cm print.
        params.cornerRefinementWinSize = 2
        self.detector = (cv2.aruco.ArucoDetector(dictionary, params)
                         if hasattr(cv2.aruco, "ArucoDetector") else None)
        self.dictionary, self.params = dictionary, params

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        def run(pixels):
            if self.detector is not None:
                return self.detector.detectMarkers(pixels)
            return cv2.aruco.detectMarkers(pixels, self.dictionary, parameters=self.params)
        if gray.shape[1] <= 640:
            # Small oblique prints need more sampling for adaptive thresholding.
            # Return original pixel coordinates so K remains unchanged.
            corners, ids, rejected = run(cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            corners = [c / 2 for c in corners]
            rejected = [c / 2 for c in rejected]
        else:
            corners, ids, rejected = run(gray)
        return corners, ids, rejected

    def estimate(self, corners, ids, k, d, prior=None):
        if ids is None:
            raise ValueError("no markers")
        known = {m["id"]: m for m in self.layout["markers"]}
        seen, objects, pixels = [], [], []
        for points, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
            marker_id = int(marker_id)
            if marker_id not in known:
                continue
            if marker_id in seen:
                raise ValueError("duplicate known marker ID")
            points = np.asarray(points, dtype=float).reshape(4, 2)
            if not np.isfinite(points).all() or abs(cv2.contourArea(points.astype(np.float32))) < self.min_marker_area_px:
                raise ValueError("invalid or undersized marker")
            seen.append(marker_id)
            objects.extend(marker_points(known[marker_id]))
            pixels.extend(points)
        if not seen:
            raise ValueError("no known markers")
        objects, pixels = np.asarray(objects), np.asarray(pixels)
        # Centre the coplanar coordinates at z=0 for consistent IPPE versions.
        center = objects.mean(axis=0)
        centered = objects - center
        result = cv2.solvePnPGeneric(centered, pixels, k, d, flags=cv2.SOLVEPNP_IPPE)
        candidates = []
        for rvec, tvec in zip(result[1], result[2]):
            rotation = cv2.Rodrigues(rvec)[0]
            translation = tvec.reshape(3) - rotation @ center
            camera_points = objects @ rotation.T + translation
            if (not np.isfinite(camera_points).all() or np.min(camera_points[:, 2]) <= 0 or
                    np.dot(rotation[:, 2], camera_points.mean(axis=0)) >= 0):
                continue
            projected = cv2.projectPoints(objects, rvec, translation, k, d)[0].reshape(-1, 2)
            errors = np.linalg.norm(projected - pixels, axis=1)
            rms = float(np.sqrt(np.mean(errors ** 2)))
            if float(errors.max()) > self.max_reprojection_px:
                continue
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, translation
            candidates.append((rms, transform, float(errors.max())))
        if not candidates:
            raise ValueError("no front-facing pose within reprojection bound")
        candidates.sort(key=lambda c: c[0])
        # Numerically identical IPPE branches describe only one hypothesis.
        if len(candidates) > 1:
            delta = pose_distance(candidates[0][1], candidates[1][1])
            if delta[0] < 1e-7 and delta[1] < 1e-7:
                candidates = candidates[:1]
        gap = candidates[1][0] - candidates[0][0] if len(candidates) > 1 else -1.0
        quality = {"marker_ids": sorted(seen), "marker_count": len(seen),
                   "candidate_count": len(candidates), "ambiguity_gap_px": gap,
                   "reprojection_rms_px": candidates[0][0], "reprojection_max_px": candidates[0][2],
                   "min_ambiguity_gap_px": self.min_ambiguity_gap_px,
                   "max_prior_translation_m": self.max_prior_translation_m,
                   "max_prior_rotation_rad": self.max_prior_rotation_rad}
        ambiguous = len(candidates) > 1 and gap < self.min_ambiguity_gap_px
        best = candidates[0]
        resolution = "single_candidate" if len(candidates) == 1 else "reprojection_gap"
        if prior is not None:
            quality["prior_translation_delta_m"], quality["prior_rotation_delta_rad"] = pose_distance(best[1], prior)
            plausible = [c for c in candidates if c[0] - best[0] < self.min_ambiguity_gap_px]
            bounded = []
            for candidate in plausible:
                dt, dr = pose_distance(candidate[1], prior)
                if dt <= self.max_prior_translation_m and dr <= self.max_prior_rotation_rad:
                    bounded.append(candidate)
            quality["prior_bounded_candidate_count"] = len(bounded)
            if not bounded:
                raise PoseRejected("pose continuity bound exceeded", quality)
            if ambiguous and len(bounded) != 1:
                raise PoseRejected("unresolved planar ambiguity: multiple branches fit prior bounds", quality)
            if ambiguous:
                best, resolution = bounded[0], "bounded_prior"
            elif bounded[0] is not best:
                raise PoseRejected("best reprojection pose exceeds continuity bounds", quality)
            quality["prior_translation_delta_m"], quality["prior_rotation_delta_rad"] = pose_distance(best[1], prior)
        elif ambiguous:
            raise PoseRejected("unresolved planar ambiguity: no qualified recent prior", quality)
        quality.update(reprojection_rms_px=best[0], reprojection_max_px=best[2],
                       acceptance_reason=resolution)
        return best[1], quality


def quaternion_xyzw(rotation):
    """Convert a proper rotation, including half turns, without ROS dependencies."""
    rvec = cv2.Rodrigues(np.asarray(rotation, dtype=float))[0].reshape(3)
    angle = np.linalg.norm(rvec)
    if angle < 1e-12:
        return np.array([0., 0., 0., 1.])
    return np.r_[rvec * (np.sin(angle / 2) / angle), np.cos(angle / 2)]


def transform_matrix(translation, quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm < 1e-9:
        raise ValueError("invalid TF quaternion")
    x, y, z, w = q / norm
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = translation
    if not np.isfinite(result).all():
        raise ValueError("nonfinite TF")
    return result
