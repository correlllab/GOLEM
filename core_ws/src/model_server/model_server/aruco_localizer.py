"""Independent calibrated ArUco localizer. Never publishes TF or commands."""

import sys
import time
from collections import OrderedDict

import numpy as np

from model_server.aruco_geometry import (
    BoardEstimator, PoseRejected, calibration, load_layout, pose_distance, quaternion_xyzw, transform_matrix,
)


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def create_node():
    # Keep import and --help independent of ROS initialization and all ML servers.
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from rclpy.duration import Duration
    from sensor_msgs.msg import Image, CompressedImage, CameraInfo
    from geometry_msgs.msg import PoseStamped
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from cv_bridge import CvBridge
    from tf2_ros import Buffer, TransformListener

    class ArucoLocalizer(Node):
        def __init__(self):
            super().__init__("aruco_localizer")
            defaults = {
                "image_topic": "/realsense/head/color/image_raw",
                "image_transport": "raw",
                "camera_info_topic": "/realsense/head/color/camera_info",
                "target_frame": "pelvis", "layout_path": "",
                "image_rectified": False, "max_reprojection_px": 2.0,
                "min_marker_area_px": 25.0, "stale_timeout_sec": .5,
                "max_observation_age_sec": .5, "publish_annotated": False,
                "min_ambiguity_gap_px": .15, "max_prior_translation_m": .02,
                "max_prior_rotation_rad": .1,
                "publish_visualization": False, "visualization_catalog_path": "",
            }
            for key, value in defaults.items():
                self.declare_parameter(key, value)
            self.settings = {key: self.get_parameter(key).value for key in defaults}
            if self.settings["image_transport"] not in ("raw", "compressed"):
                raise ValueError("image_transport must be raw or compressed")
            for name in ("max_reprojection_px", "min_marker_area_px", "stale_timeout_sec", "max_observation_age_sec",
                         "min_ambiguity_gap_px", "max_prior_translation_m", "max_prior_rotation_rad"):
                value = self.settings[name]
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
            if not self.settings["target_frame"]:
                raise ValueError("target_frame must not be empty")
            self.estimator = BoardEstimator(load_layout(self.settings["layout_path"] or None),
                                            self.settings["max_reprojection_px"],
                                            self.settings["min_marker_area_px"],
                                            self.settings["min_ambiguity_gap_px"],
                                            self.settings["max_prior_translation_m"],
                                            self.settings["max_prior_rotation_rad"])
            self.bridge = CvBridge()
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.pose_pub = self.create_publisher(PoseStamped, "/perception/pegboard/pose", 10)
            self.optical_pub = self.create_publisher(PoseStamped, "/perception/pegboard/pose_optical", 10)
            self.diagnostics_pub = self.create_publisher(DiagnosticArray, "/perception/pegboard/diagnostics", 10)
            self.annotated_pub = (self.create_publisher(Image, "/perception/pegboard/annotated", 1)
                                  if self.settings["publish_annotated"] else None)
            self.visualization = None
            if self.settings["publish_visualization"]:
                from model_server.aruco_visualization import BoardVisualization
                self.visualization = BoardVisualization(self, self.estimator.layout,
                                                        self.settings["visualization_catalog_path"])
            self.infos, self.images = OrderedDict(), OrderedDict()
            self.pending_tf = OrderedDict()
            self.input_metrics = {}
            self.last_stamp = None
            self.last_frame = None
            self.processed_stamp = None
            self.last_clock_ns = self.get_clock().now().nanoseconds
            self.prior = None
            self.prior_target = None
            self.prior_key = None
            self.active_calibration_key = None
            self.valid = False
            self.reason = "waiting for synchronized image and calibration"
            self.metrics = {}
            self.create_subscription(CameraInfo, self.settings["camera_info_topic"], self.on_info, qos_profile_sensor_data)
            compressed = self.settings["image_transport"] == "compressed"
            image_topic = self.settings["image_topic"]
            if compressed:
                image_topic = image_topic.rstrip("/") + "/compressed"
            self.create_subscription(CompressedImage if compressed else Image,
                                     image_topic, self.on_image, qos_profile_sensor_data)
            self.create_timer(.1, self.heartbeat)

        def on_info(self, info):
            self.heartbeat(retry_pending=False)
            key = (stamp_ns(info.header.stamp), info.header.frame_id)
            if self.processed_stamp is not None and key[0] <= self.processed_stamp:
                return
            self.infos[key] = info
            if key in self.images:
                self.process(self.images.pop(key), self.infos.pop(key))
            while len(self.infos) > 10:
                self.infos.popitem(last=False)

        def on_image(self, image):
            self.heartbeat(retry_pending=False)
            key = (stamp_ns(image.header.stamp), image.header.frame_id)
            if self.processed_stamp is not None and key[0] <= self.processed_stamp:
                return
            if key in self.infos:
                self.process(image, self.infos.pop(key))
            else:
                self.images[key] = image
                while len(self.images) > 10:
                    self.images.popitem(last=False)

        def invalidate(self, reason, discard_prior=True, quality=None):
            self.valid, self.reason = False, reason
            self.pending_tf.clear()
            if discard_prior:
                self.prior, self.prior_key = None, None
                self.prior_target = None
                self.active_calibration_key = None
            self.metrics = quality or {}
            self.heartbeat()

        def pose_message(self, matrix, header, frame):
            message = PoseStamped()
            message.header.stamp = header.stamp
            message.header.frame_id = frame
            p, q = matrix[:3, 3], quaternion_xyzw(matrix[:3, :3])
            message.pose.position.x, message.pose.position.y, message.pose.position.z = map(float, p)
            (message.pose.orientation.x, message.pose.orientation.y,
             message.pose.orientation.z, message.pose.orientation.w) = map(float, q)
            return message

        def process(self, image, info):
            self.heartbeat(retry_pending=False)
            now = self.get_clock().now().nanoseconds
            stamp = stamp_ns(image.header.stamp)
            if self.processed_stamp is not None and stamp <= self.processed_stamp:
                return
            age = (now - stamp) / 1e9
            self.input_metrics = {"input_stamp_ns": stamp, "input_age_sec": age}
            if (not image.header.frame_id or stamp <= 0 or age < 0 or
                    age > self.settings["max_observation_age_sec"]):
                self.invalidate("zero, future, or stale image timestamp",
                                quality={"rejected_image_age_sec": age})
                return
            if (stamp != stamp_ns(info.header.stamp) or
                    image.header.frame_id != info.header.frame_id):
                self.invalidate("image/CameraInfo association mismatch")
                return
            self.processed_stamp = stamp
            started = time.monotonic()
            try:
                if self.settings["image_transport"] == "compressed":
                    import cv2
                    encoding = image.format.strip().lower()
                    parts = encoding.split(";")
                    if not (encoding in ("jpeg", "jpg") or
                            (len(parts) == 2 and parts[0].strip() in ("rgb8", "bgr8") and
                             parts[1].split() in (["jpeg", "compressed", "bgr8"],
                                                  ["jpeg", "compressed", "rgb8"]))):
                        raise ValueError("compressed RGB input requires JPEG color format")
                    pixels = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if pixels is None:
                        raise ValueError("invalid JPEG image data")
                    rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                else:
                    rgb = self.bridge.imgmsg_to_cv2(image, desired_encoding="rgb8")
                if self.settings["image_rectified"]:
                    projection = np.asarray(info.p, dtype=float).reshape(3, 4)
                    if not np.allclose(projection[:, 3], 0):
                        raise ValueError("rectified stereo P translation is unsupported")
                    k, d = calibration(projection[:, :3], [], info.width, info.height, rgb.shape)
                else:
                    k, d = calibration(info.k, info.d, info.width, info.height, rgb.shape, info.distortion_model)
                calibration_key = (image.header.frame_id, info.width, info.height,
                                   tuple(k.reshape(-1)), tuple(d), self.settings["image_rectified"],
                                   info.distortion_model, self.settings["target_frame"], self.settings["layout_path"])
                if calibration_key != self.active_calibration_key:
                    self.pending_tf.clear()
                    self.prior, self.prior_key = None, None
                    self.prior_target = None
                    self.active_calibration_key = calibration_key
                    self.valid, self.reason = False, "camera calibration changed; waiting for observation TF"
                corners, ids, _ = self.estimator.detect(rgb)
            except Exception as exc:
                # Rejected branches cannot replace or refresh the accepted
                # anchor, nor turn the next jump into an unguarded cold start.
                self.invalidate(f"observation rejected: {exc}",
                                discard_prior=not isinstance(exc, PoseRejected),
                                quality=getattr(exc, "quality", None))
                return
            self.input_metrics["processing_wall_sec"] = time.monotonic() - started
            self.pending_tf[stamp] = (image.header, calibration_key, rgb, k, d, corners, ids)
            while len(self.pending_tf) > 10:
                self.pending_tf.popitem(last=False)
            self.heartbeat()

        def retry_pending_tf(self, now):
            for stamp, observation in list(self.pending_tf.items()):
                header, calibration_key, rgb, k, d, corners, ids = observation
                if calibration_key != self.active_calibration_key:
                    self.pending_tf.pop(stamp)
                    continue
                if self.last_stamp is not None and stamp <= self.last_stamp:
                    self.pending_tf.pop(stamp)
                    continue
                age = (now - stamp) / 1e9
                if age < 0 or age > self.settings["max_observation_age_sec"]:
                    self.pending_tf.pop(stamp)
                    self.valid, self.reason = False, "pending TF observation expired"
                    continue
                try:
                    # Never block the executor that must receive the missing TF.
                    tf = self.tf_buffer.lookup_transform(
                        self.settings["target_frame"], header.frame_id,
                        Time.from_msg(header.stamp), timeout=Duration(seconds=0.0))
                    t, q = tf.transform.translation, tf.transform.rotation
                    target_from_camera = transform_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
                except Exception as exc:
                    self.input_metrics["last_tf_error"] = str(exc)
                    # Waiting for exact-time TF does not revoke a fresh accepted pose.
                    if not self.valid:
                        self.reason = f"observation TF pending: {exc}"
                    continue
                self.pending_tf.pop(stamp)
                try:
                    # Compare board motion in one target frame, not two moving
                    # camera frames. Both poses passed to the estimator are C_now_B.
                    prior = None
                    if (self.prior_target is not None and self.prior_key == calibration_key and
                            self.last_stamp is not None and
                            0 < (stamp-self.last_stamp)/1e9 <= self.settings["stale_timeout_sec"] and
                            0 <= (now-self.last_stamp)/1e9 <= self.settings["stale_timeout_sec"]):
                        prior = np.linalg.inv(target_from_camera) @ self.prior_target
                    optical, metrics = self.estimator.estimate(corners, ids, k, d, prior)
                    pose_distance(optical, optical)
                    target = target_from_camera @ optical
                except Exception as exc:
                    # A rejection neither advances the anchor nor leaves queued
                    # observations able to resurrect it in this retry pass.
                    self.invalidate(f"observation rejected: {exc}",
                                    discard_prior=not isinstance(exc, PoseRejected),
                                    quality=getattr(exc, "quality", None))
                    return
                metrics["tf_wait_age_sec"] = age
                self.accept_observation(header, optical, target, metrics, calibration_key, rgb, k, d, corners, ids)

        def accept_observation(self, header, optical, target, metrics, calibration_key, rgb, k, d, corners, ids):
            stamp = stamp_ns(header.stamp)
            # Both output topics retain the original camera observation timestamp.
            self.last_stamp, self.last_frame = stamp, header.frame_id
            self.prior, self.metrics, self.valid = optical, metrics, True
            self.prior_target = target
            self.prior_key = calibration_key
            self.reason = "board observation valid"
            self.optical_pub.publish(self.pose_message(optical, header, header.frame_id))
            self.pose_pub.publish(self.pose_message(target, header, self.settings["target_frame"]))
            if self.visualization is not None:
                self.visualization.observation = (header, target, k, d, rgb.shape[1], rgb.shape[0])
            if self.annotated_pub is not None:
                from model_server.aruco_visualization import annotate
                annotated = self.bridge.cv2_to_imgmsg(annotate(rgb, optical, k, d, corners, ids), encoding="rgb8")
                annotated.header = header
                self.annotated_pub.publish(annotated)

        def heartbeat(self, retry_pending=True):
            now = self.get_clock().now()
            clock_reset = now.nanoseconds < self.last_clock_ns
            self.last_clock_ns = now.nanoseconds
            if not clock_reset and retry_pending:
                self.retry_pending_tf(now.nanoseconds)
            age = None if self.last_stamp is None else (now.nanoseconds-self.last_stamp)/1e9
            if age is not None and age > self.settings["stale_timeout_sec"]:
                self.prior, self.prior_key = None, None
                self.prior_target = None
            if self.valid and (clock_reset or age < 0 or age > self.settings["stale_timeout_sec"]):
                self.valid, self.reason, self.prior = False, "stale observation or clock reset", None
                self.prior_target = None
            if clock_reset:
                # Clear timestamp-keyed caches across simulation clock epochs.
                self.last_stamp, self.processed_stamp = None, None
                self.prior, self.prior_key = None, None
                self.prior_target = None
                self.metrics = {}
                self.active_calibration_key = None
                self.valid, self.reason = False, "clock reset; waiting for a new observation"
                self.infos.clear()
                self.images.clear()
                self.pending_tf.clear()
                self.tf_buffer.clear()
            status = DiagnosticStatus()
            status.name = "pegboard_localization"
            status.hardware_id = "head_camera_aruco"
            status.level = DiagnosticStatus.OK if self.valid else DiagnosticStatus.WARN
            status.message = self.reason
            values = {"valid": str(self.valid).lower(), "observation_stamp_ns": str(self.last_stamp or 0),
                      "observation_age_sec": str(age), "target_frame": self.settings["target_frame"],
                       "pending_tf_count": len(self.pending_tf), **self.input_metrics, **self.metrics}
            if self.pending_tf:
                values["pending_tf_oldest_age_sec"] = (now.nanoseconds - next(iter(self.pending_tf))) / 1e9
            status.values = [KeyValue(key=str(k), value=str(v)) for k, v in values.items()]
            message = DiagnosticArray()
            message.header.stamp = now.to_msg()
            message.status = [status]
            self.diagnostics_pub.publish(message)
            if self.visualization is not None:
                self.visualization.publish(self.valid, self.reason, now.to_msg(), self.settings["target_frame"])

    return ArucoLocalizer()


def main(args=None):
    if "--help" in (sys.argv[1:] if args is None else args):
        print("aruco_localizer [--ros-args -p use_sim_time:=true ...]\n"
              "Independent image/CameraInfo -> /perception/pegboard/pose (pelvis).\n"
              "Parameters: image_topic (base topic), image_transport (raw|compressed; default raw),\n"
              "camera_info_topic, target_frame, layout_path,\n"
              "image_rectified, max_reprojection_px, min_marker_area_px, stale_timeout_sec,\n"
              "max_observation_age_sec, min_ambiguity_gap_px, max_prior_translation_m,\n"
               "max_prior_rotation_rad, publish_annotated, publish_visualization,\n"
               "visualization_catalog_path. No ML imports or actuation.")
        return
    import rclpy
    rclpy.init(args=args)
    node = None
    try:
        node = create_node()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
