#!/usr/bin/env python3
"""
Detect open-vocabulary objects in camera streams with YOLO-World and publish DetectionBundles.

For each CompressedImage topic, a timer (default 5 Hz) runs the latest frame
through YoloWorld and publishes a custom_ros_messages/DetectionBundle on that
topic's own output topic. A bundle echoes the source image in `rgb_image` and
holds every detection as a Detection (class string, xyxy box as
bbox_min/bbox_max points, confidence).

A bundle also carries the camera's latest ALIGNED DEPTH frame and color
camera_info, from topics derived from the image topic by the realsense layout
(DEPTH_STREAM, CAMERA_INFO_STREAM). The depth is on the color grid, so a
subscriber back-projects a box to a 3-D point from the bundle alone; that is
what golem_skills' hand-camera visual servo does. A camera without those
streams gets bundles without them.

Each detection topic has a sibling '<image_topic>/annotated' topic, a raw
sensor_msgs/Image with the boxes drawn, for an RViz Image display.

The class vocabulary is the `queries` parameter, read every cycle, so
`ros2 param set` changes it without a restart; all topics share it.

ROS parameters:
  image_topics      string[]  input CompressedImage topics
  detection_topics  string[]  output DetectionBundle topics, parallel to
                              image_topics; when empty or of another length,
                              each is "<image_topic>/detections"
  queries           string[]  class prompts (e.g. ['bottle', 'cup'])
  publish_rate_hz   double    per-topic inference and publish rate [Hz] (5.0)
  conf_threshold    double    detection confidence threshold (0.90)
  nms_iou           double    NMS IoU threshold (0.01)
  enable_logging, enable_visualization, clear_logs: model_logging
"""
import os
import threading
from functools import partial

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from geometry_msgs.msg import Point

from ultralytics import YOLOWorld

from custom_ros_messages.msg import Detection, DetectionBundle

from model_server.model_logging import ModelLogger, declare_logging_params

# Checkpoint under the weights directory: the YOLO-World fine-tune for the
# battery workcell, a full Ultralytics export loaded directly. A checkpoint dict
# with a 'state_dict' loads onto the base architecture instead; with no file,
# Ultralytics downloads a checkpoint of this name into the working directory.
YOLO_MODEL = "yolo_world_battery_best.pt"
# Detections below this confidence are dropped; high, so only strong
# detections come back (lower it for recall). Default of `conf_threshold`.
YOLO_CONF_THRESHOLD = 0.90
# Boxes overlapping above this IoU are merged; low, so near-duplicate boxes of
# one object collapse. Default of `nms_iou`.
YOLO_NMS_IOU = 0.01
# FP32 inference: set_classes() stores the CLIP text features as FP32, and with
# half=True the FP16 image branch meets them in the detection head and raises
# "expected scalar type Float but found Half". At ~5 Hz on a few cameras the
# FP16 speedup does not matter.
YOLO_HALF = False

# default of `image_topics`: the head and hand RealSense color streams
# (cl_realsense's camera launch files enable their compressed transport)
DEFAULT_IMAGE_TOPICS = [
    "/realsense/head/color/image_raw/compressed",
    "/realsense/left_hand/color/image_raw/compressed",
    "/realsense/right_hand/color/image_raw/compressed",
]
# default of `queries`: the battery-workcell classes YOLO_MODEL was fine-tuned
# on, spelled exactly as its training labels; golem_skills' grasp.BATTERY_OBJECTS
# lists the same names
DEFAULT_QUERIES = ["Bolt", "BusBar", "InteriorScrew", "Nut", "OrageCover",
                   "Screw", "ScrewHole"]
DEFAULT_PUBLISH_RATE_HZ = 5.0
# appended to an image topic for its detection topic when `detection_topics` does not apply
DETECTION_TOPIC_SUFFIX = "/detections"
# The bundle's depth and intrinsics topics, derived from the image topic:
#   /realsense/<cam>/color/image_raw/compressed
#     -> /realsense/<cam>/aligned_depth_to_color/image_raw/compressedDepth
#     -> /realsense/<cam>/color/camera_info
# aligned_depth_to_color is depth resampled onto the COLOR grid, so the boxes
# index it directly. A camera with another layout gets no frames on the derived
# topics, and its bundles go out with those fields empty.
DEPTH_STREAM = "aligned_depth_to_color/image_raw/compressedDepth"
CAMERA_INFO_STREAM = "color/camera_info"
# appended to an image topic for its annotated-image topic
ANNOTATED_TOPIC_SUFFIX = "/annotated"

# The weights directory is the package's weights/, or MODEL_SERVER_WEIGHTS_DIR.
# realpath, not abspath: under `colcon build --symlink-install` this module is
# a symlink into the source tree, where weights/ is.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_WEIGHTS_DIR = os.environ.get("MODEL_SERVER_WEIGHTS_DIR",
                              os.path.join(_PKG_ROOT, "weights"))


class YoloWorld:
    """Detect a text-prompted class vocabulary in images with Ultralytics YOLO-World."""

    def __init__(self):
        """Load YOLO_MODEL and move it to CUDA when available, so GPU errors surface at startup."""
        weight_path = os.path.join(_WEIGHTS_DIR, YOLO_MODEL)
        if os.path.exists(weight_path):
            # A {'state_dict': ...} export loads onto the base architecture,
            # non-strict so the text-encoder and head buffers it lacks are kept.
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.model = YOLOWorld(YOLO_MODEL)
                self.model.model.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                self.model = YOLOWorld(weight_path)
        else:
            self.model = YOLOWorld(YOLO_MODEL)

        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))
        # the module stays FP32 (YOLO_HALF)
        self.model.to(self.device)
        print(f"[YoloWorld init] Initialized on device {self.device}")

    def detect(self, bgr_img, queries, conf=YOLO_CONF_THRESHOLD, iou=YOLO_NMS_IOU):
        """
        Detect the classes `queries` in one image; return (boxes, scores, class_ids) best-first.

        `bgr_img` is an HxWx3 uint8 BGR array (Ultralytics' numpy convention).
        boxes are (N, 4) float32 pixel xyxy, scores (N,) float32 in [0, 1],
        class_ids (N,) int32 indices into `queries`; all empty when nothing is
        detected.
        """
        self.model.set_classes(queries)
        with torch.no_grad():
            results = self.model.predict(bgr_img, show=False, verbose=False,
                                         conf=conf, iou=iou, half=YOLO_HALF)[0]
        boxes = results.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = results.boxes.conf.detach().cpu().numpy().astype(np.float32)
        class_ids = results.boxes.cls.detach().cpu().numpy().astype(np.int32)
        order = np.argsort(-scores)
        return boxes[order], scores[order], class_ids[order]


# s a cycle waits for the model lock. An inference takes well under a second; a
# lock held far longer means a previous call has wedged (CUDA stall or OOM), and
# skipping the cycle with a clear warning beats queueing every later cycle.
LOCK_ACQUIRE_TIMEOUT_SEC = 20.0


class _Channel:
    """Hold one image topic's latest frame, depth and intrinsics, and its publishers."""

    def __init__(self, image_topic, detection_topic, annotated_topic,
                 publisher, annotated_publisher):
        """Store the topics and publishers; no frame is cached yet."""
        self.image_topic = image_topic
        self.detection_topic = detection_topic
        self.annotated_topic = annotated_topic
        self.publisher = publisher                    # DetectionBundle
        self.annotated_publisher = annotated_publisher  # sensor_msgs/Image
        # guards the three caches, written by subscriptions and read by the timer
        self._img_lock = threading.Lock()
        # latest CompressedImage, aligned depth and CameraInfo; None until one arrives
        self._latest = None
        self._latest_depth = None
        self._latest_caminfo = None

    def set_latest(self, msg):
        """Cache the latest color frame."""
        with self._img_lock:
            self._latest = msg

    def take_latest(self):
        """Return the latest color frame, or None."""
        with self._img_lock:
            return self._latest

    def set_depth(self, msg):
        """Cache the latest aligned-depth frame."""
        with self._img_lock:
            self._latest_depth = msg

    def set_caminfo(self, msg):
        """Cache the latest color CameraInfo."""
        with self._img_lock:
            self._latest_caminfo = msg

    def take_depth_and_info(self):
        """
        Return the latest (depth, camera_info) pair; either may be None.

        Read under one lock, so a bundle never pairs a depth frame with
        intrinsics from a different reconfiguration.
        """
        with self._img_lock:
            return self._latest_depth, self._latest_caminfo


class YoloServer(Node):
    """Run YOLO-World on every configured image topic and publish its DetectionBundles."""

    def __init__(self):
        """Read the parameters, load the model, and wire one channel per image topic."""
        super().__init__("yolo_server")
        self.bridge = CvBridge()
        log, viz, clear = declare_logging_params(self)
        self.logger = ModelLogger(self, 'yolo', 'model_server', __file__,
                                  log=log, visualize=viz, clear=clear)

        str_array = ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        image_topics = list(self.declare_parameter(
            "image_topics", DEFAULT_IMAGE_TOPICS, str_array).value)
        detection_topics = list(self.declare_parameter(
            "detection_topics", [], str_array).value)
        # read every cycle (_current_queries), so it can change at runtime
        self.declare_parameter("queries", DEFAULT_QUERIES, str_array)
        self.conf = float(self.declare_parameter(
            "conf_threshold", YOLO_CONF_THRESHOLD).value)
        self.iou = float(self.declare_parameter("nms_iou", YOLO_NMS_IOU).value)
        rate = float(self.declare_parameter(
            "publish_rate_hz", DEFAULT_PUBLISH_RATE_HZ).value)

        if not image_topics:
            raise RuntimeError("yolo_server: 'image_topics' is empty — "
                               "nothing to subscribe to")
        if rate <= 0.0:
            raise RuntimeError(f"yolo_server: 'publish_rate_hz' must be > 0 "
                               f"(got {rate})")
        # an explicit `detection_topics` of matching length wins; otherwise every
        # output topic is derived from its image topic
        if len(detection_topics) != len(image_topics):
            if detection_topics:
                self.get_logger().warn(
                    f"'detection_topics' ({len(detection_topics)}) does not match "
                    f"'image_topics' ({len(image_topics)}); deriving all outputs "
                    f"as <image_topic>{DETECTION_TOPIC_SUFFIX}")
            detection_topics = [self._derive_detection_topic(t)
                                for t in image_topics]

        # set_classes changes model state that predict reads, so the per-topic
        # timers, which fire concurrently, run inference one at a time
        self._lock = threading.RLock()
        self.get_logger().info("yolo_server loading YOLO-World ...")
        self.yolo = YoloWorld()

        self._cb_group = ReentrantCallbackGroup()
        self._channels = []
        period = 1.0 / rate
        for img_topic, det_topic in zip(image_topics, detection_topics):
            ann_topic = self._derive_annotated_topic(img_topic)
            pub = self.create_publisher(DetectionBundle, det_topic, 10)
            ann_pub = self.create_publisher(Image, ann_topic, 10)
            ch = _Channel(img_topic, det_topic, ann_topic, pub, ann_pub)
            self._channels.append(ch)
            self.create_subscription(
                CompressedImage, img_topic, partial(self._on_image, ch),
                qos_profile_sensor_data, callback_group=self._cb_group)
            depth_topic = self._derive_camera_topic(img_topic, DEPTH_STREAM)
            info_topic = self._derive_camera_topic(img_topic, CAMERA_INFO_STREAM)
            if depth_topic and info_topic:
                self.create_subscription(
                    CompressedImage, depth_topic, partial(self._on_depth, ch),
                    qos_profile_sensor_data, callback_group=self._cb_group)
                self.create_subscription(
                    CameraInfo, info_topic, partial(self._on_caminfo, ch),
                    qos_profile_sensor_data, callback_group=self._cb_group)
            self.create_timer(period, partial(self._on_timer, ch),
                              callback_group=self._cb_group)
            self.get_logger().info(
                f"  {img_topic}  ->  {det_topic}  (+ {ann_topic})")
            self.get_logger().info(
                f"      depth {depth_topic or '(none — non-realsense topic layout)'}"
                f"  info {info_topic or '(none)'}")

        self.get_logger().info(
            f"yolo_server ready: {len(self._channels)} topic(s) @ {rate:.2f} Hz")

    @staticmethod
    def _derive_detection_topic(image_topic):
        """Return the default detection topic of `image_topic`: '<image_topic>/detections'."""
        return image_topic.rstrip("/") + DETECTION_TOPIC_SUFFIX

    @staticmethod
    def _derive_annotated_topic(image_topic):
        """Return the annotated-image topic of `image_topic`: '<image_topic>/annotated'."""
        return image_topic.rstrip("/") + ANNOTATED_TOPIC_SUFFIX

    @staticmethod
    def _derive_camera_topic(image_topic, stream):
        """
        Return the camera's `stream` topic beside a color image topic, or None.

        Everything before '/color/' is the camera namespace, so
        '/realsense/head/color/image_raw/compressed' with 'color/camera_info'
        gives '/realsense/head/color/camera_info'. None when the topic does not
        follow the realsense '<ns>/color/...' layout.
        """
        if "/color/" not in image_topic:
            return None
        return f"{image_topic.rsplit('/color/', 1)[0]}/{stream}"

    def _current_queries(self):
        """Return the current `queries` parameter as the class vocabulary."""
        return [str(q) for q in self.get_parameter("queries").value]

    def _on_image(self, channel, msg):
        """Cache the latest frame for `channel`; inference happens on the timer."""
        channel.set_latest(msg)

    def _on_depth(self, channel, msg):
        """Cache the latest aligned-depth frame for `channel`'s camera."""
        channel.set_depth(msg)

    def _on_caminfo(self, channel, msg):
        """Cache the latest color CameraInfo for `channel`'s camera."""
        channel.set_caminfo(msg)

    def _on_timer(self, channel):
        """Run one inference on `channel`'s latest frame and publish its bundle and overlay."""
        msg = channel.take_latest()
        if msg is None:
            self.get_logger().warn(
                f"{channel.image_topic}: no frames received yet — is the camera "
                "publishing on this topic?", throttle_duration_sec=5.0)
            return

        queries = self._current_queries()
        if not queries:
            self.get_logger().warn(
                "no 'queries' set — skipping detection (set the queries param)",
                throttle_duration_sec=10.0)
            return

        rec = self.logger.start()
        rec.set(topic=channel.image_topic, queries=queries, n_queries=len(queries))
        try:
            # Ultralytics reads a numpy source as BGR
            bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(
                f"{channel.image_topic}: failed to decode image: {e}")
            rec.finish(success=False, message=f"decode error: {e}")
            return
        rec.save_image('input', bgr)

        if not self._lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT_SEC):
            self.get_logger().warn(
                f"YOLO busy: model lock held > {LOCK_ACQUIRE_TIMEOUT_SEC:.0f}s "
                "(prior inference wedged or GPU overloaded); skipping cycle")
            rec.finish(success=False, message="model lock timeout")
            return
        try:
            boxes, scores, class_ids = self.yolo.detect(
                bgr, queries, conf=self.conf, iou=self.iou)
        except Exception as e:
            self.get_logger().error(f"YOLO detect failed: {e}")
            rec.finish(success=False, message=f"detection error: {e}")
            return
        finally:
            self._lock.release()

        bundle = self._build_bundle(channel, msg, boxes, scores, class_ids, queries)
        channel.publisher.publish(bundle)

        n = int(boxes.shape[0])
        per_query = {}
        for cid in class_ids.tolist():
            if 0 <= cid < len(queries):
                per_query[queries[cid]] = per_query.get(queries[cid], 0) + 1
        H, W = bgr.shape[:2]
        rec.set(n_detections=n, detections_per_query=per_query,
                best_score=float(scores[0]) if n else 0.0,
                image_hw=[int(H), int(W)])
        # debug level: a line per frame per camera would flood the console
        self.get_logger().debug(
            f"{channel.detection_topic}: {n} detection(s) for "
            f"{len(queries)} quer(ies) per_query={per_query}" if n else
            f"{channel.detection_topic}: no detections for queries={queries}")

        vis = self._draw_overlay(bgr, boxes, scores, class_ids, queries)
        if vis is not None:
            self._publish_annotated(channel, vis, msg.header)
            if self.logger.visualize:
                self._save_overlay_to_disk(rec, vis)
        rec.finish(success=True, message="ok")

    def _build_bundle(self, channel, image_msg, boxes, scores, class_ids, queries):
        """
        Build the DetectionBundle of one frame: the image, depth, intrinsics and detections.

        camera_pose stays at its default: subscribers get the pose from TF by
        the depth's frame_id.
        """
        bundle = DetectionBundle()
        bundle.rgb_image = image_msg
        # the LATEST depth and intrinsics, not stamp-matched to the color frame:
        # valid for a roughly static scene, which the visual servo assumes
        depth_msg, caminfo_msg = channel.take_depth_and_info()
        if depth_msg is not None:
            bundle.depth_image = depth_msg
        if caminfo_msg is not None:
            bundle.camera_info = caminfo_msg
        detections = []
        for i in range(int(boxes.shape[0])):
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            cid = int(class_ids[i])
            det = Detection()
            det.cls = queries[cid] if 0 <= cid < len(queries) else str(cid)
            det.bbox_min = Point(x=x1, y=y1, z=0.0)
            det.bbox_max = Point(x=x2, y=y2, z=0.0)
            det.prob = float(scores[i])
            detections.append(det)
        bundle.detections = detections
        return bundle

    def _draw_overlay(self, bgr, boxes, scores, class_ids, queries):
        """Return a copy of `bgr` with each box and 'label score' drawn, or None on failure."""
        try:
            import cv2
            vis = bgr.copy()
            palette = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
                       (255, 0, 255), (255, 255, 0)]
            for i in range(int(boxes.shape[0])):
                x1, y1, x2, y2 = (int(v) for v in boxes[i])
                cid = int(class_ids[i])
                color = palette[cid % len(palette)]
                label = queries[cid] if 0 <= cid < len(queries) else str(cid)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{label} {float(scores[i]):.2f}",
                            (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1, cv2.LINE_AA)
            return vis
        except Exception as e:
            self.get_logger().warn(f'yolo overlay draw failed: {e}')
            return None

    def _publish_annotated(self, channel, vis, header):
        """Publish the annotated frame as a raw bgr8 Image with the source frame's header."""
        try:
            img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            img_msg.header = header
            channel.annotated_publisher.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(
                f'{channel.annotated_topic}: publish failed: {e}')

    def _save_overlay_to_disk(self, rec, vis):
        """Save the annotated frame to the model's log directory."""
        try:
            import cv2
            cv2.imwrite(rec.path('overlay', 'png'), vis)
        except Exception as e:
            self.get_logger().warn(f'yolo overlay save failed: {e}')


def RunYoloServer(args=None):
    """Spin a YoloServer on a MultiThreadedExecutor until interrupted."""
    rclpy.init(args=args)
    node = YoloServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    RunYoloServer()
