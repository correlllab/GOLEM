#!/usr/bin/env python3
"""
Serve custom_ros_messages/srv/SamSegment on `sam_segment` with SAM3.

A request carries an image and a text prompt and/or positive and negative box
exemplars; the response is the single best mask (mono8) and its score. SAM3's
image API takes text and box prompts only (no click points), so the service
takes boxes.
"""
import os
import threading

import numpy as np
import torch
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from cv_bridge import CvBridge

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from custom_ros_messages.srv import SamSegment

from model_server.model_logging import ModelLogger, declare_logging_params

# checkpoint file under the weights directory; None downloads facebook/sam3
# from HuggingFace instead (needs HF auth)
SAM3_MODEL = "sam3.pt"
SAM3_CONFIDENCE_THRESHOLD = 0.5

# The weights directory is the package's weights/, or MODEL_SERVER_WEIGHTS_DIR.
# realpath, not abspath: under `colcon build --symlink-install` this module is
# a symlink into the source tree, where weights/ is.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_WEIGHTS_DIR = os.environ.get("MODEL_SERVER_WEIGHTS_DIR",
                              os.path.join(_PKG_ROOT, "weights"))


class SAM3:
    """Segment an image with SAM3 from text and box prompts."""

    def __init__(self):
        """Load SAM3 on CUDA when available; an AssertionError names a missing checkpoint."""
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))
        if SAM3_MODEL is not None:
            weight_path = os.path.join(_WEIGHTS_DIR, SAM3_MODEL)
            assert os.path.exists(weight_path), f"[SAM3 init] Weights at {weight_path} not found"
            sam_model = build_sam3_image_model(checkpoint_path=weight_path, load_from_HF=False)
        else:
            sam_model = build_sam3_image_model()
        self.processor = Sam3Processor(sam_model, confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)
        self._inference_state = None
        self._img_hw = None
        print(f"[SAM3 init] Initialized on device {self.device}")

    def _set_image(self, rgb_img):
        """Encode `rgb_img` once into the inference state the prompts then decode against."""
        pil_img = PILImage.fromarray(rgb_img)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            self._inference_state = self.processor.set_image(pil_img)
        self._img_hw = (rgb_img.shape[0], rgb_img.shape[1])

    def segment(self, rgb_img, positive_boxes, negative_boxes, text):
        """
        Segment `rgb_img` with every prompt at once; return (mask, score) of the best mask.

        `text` is a concept prompt, `positive_boxes` / `negative_boxes`
        foreground / background exemplars as pixel [x1, y1, x2, y2]; give a
        text or at least one positive box. The mask is an (H, W) bool array at
        the input resolution; (None, 0.0) when SAM3 produces none.
        """
        self._set_image(rgb_img)
        state = self._inference_state
        self.processor.reset_all_prompts(state)
        H, W = self._img_hw

        def _norm_cxcywh(box):
            """Convert a pixel xyxy box to SAM3's normalized (cx, cy, w, h)."""
            x1, y1, x2, y2 = box
            return [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            if text:
                ret = self.processor.set_text_prompt(prompt=text, state=state)
                if ret is not None:
                    state = ret
            for box in positive_boxes:
                ret = self.processor.add_geometric_prompt(
                    box=_norm_cxcywh(box), label=True, state=state)
                if ret is not None:
                    state = ret
            for box in negative_boxes:
                ret = self.processor.add_geometric_prompt(
                    box=_norm_cxcywh(box), label=False, state=state)
                if ret is not None:
                    state = ret

        masks = state.get("masks")    # (N, 1, H, W) bool
        scores = state.get("scores")  # (N,)
        if masks is None or masks.shape[0] == 0:
            return None, 0.0
        best = int(scores.argmax().item()) if scores is not None else 0
        mask_np = masks[best, 0].cpu().numpy().astype(bool)
        score = float(scores[best].item()) if scores is not None else 1.0
        return mask_np, score


def _group_boxes(flat):
    """Split a flat [x1, y1, x2, y2, ...] array into boxes; raise ValueError on a bad length."""
    if len(flat) % 4 != 0:
        raise ValueError(f"box array length {len(flat)} is not a multiple of 4")
    return [[float(v) for v in flat[i:i + 4]] for i in range(0, len(flat), 4)]


# s a request waits for the model lock. An inference takes a few seconds; a lock
# held far longer means a previous call has wedged (CUDA stall or OOM), and
# failing with a clear error beats queueing every later request behind it.
LOCK_ACQUIRE_TIMEOUT_SEC = 20.0


class SamServer(Node):
    """Serve sam_segment with SAM3."""

    def __init__(self):
        """Load SAM3 at startup, so weight and GPU errors surface then, and create the service."""
        super().__init__("sam_server")
        self.get_logger().info("\n\n\nsam_server beginning SAM3 initialization ...\n\n\n")
        self.bridge = CvBridge()
        log, viz, clear = declare_logging_params(self)
        self.logger = ModelLogger(self, 'sam', 'model_server', __file__,
                                  log=log, visualize=viz, clear=clear)
        # SAM3 keeps one per-image inference state, so calls run one at a time
        self._lock = threading.RLock()
        self.get_logger().info("sam_server loading SAM3 ...")
        self.sam = SAM3()
        self.srv = self.create_service(
            SamSegment, "sam_segment", self.segment_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info("sam_server ready")

    def segment_callback(self, request, response):
        """Answer one SamSegment request; a bad request or a failed inference fails it."""
        rec = self.logger.start()
        rec.set(text=request.text,
                n_positive_boxes=len(request.positive_boxes) // 4,
                n_negative_boxes=len(request.negative_boxes) // 4)
        try:
            positive_boxes = _group_boxes(request.positive_boxes)
            negative_boxes = _group_boxes(request.negative_boxes)
        except ValueError as e:
            return self._fail(response, str(e), rec)
        text = request.text

        if not text and not positive_boxes:
            return self._fail(
                response, "provide a text prompt and/or at least one positive box", rec)

        try:
            rgb = self.bridge.compressed_imgmsg_to_cv2(request.image, desired_encoding="rgb8")
        except Exception as e:
            return self._fail(response, f"failed to decode image: {e}", rec)
        rec.save_image('input', rgb, rgb=True)

        if not self._lock.acquire(timeout=LOCK_ACQUIRE_TIMEOUT_SEC):
            return self._fail(
                response,
                f"SAM busy: model lock held > {LOCK_ACQUIRE_TIMEOUT_SEC:.0f}s "
                "(prior inference wedged or GPU overloaded)", rec)
        try:
            mask, score = self.sam.segment(rgb, positive_boxes, negative_boxes, text)
        except Exception as e:
            self.get_logger().error(f"SAM3 segment failed: {e}")
            return self._fail(response, f"segmentation error: {e}", rec)
        finally:
            self._lock.release()

        if mask is None:
            return self._fail(response, "SAM3 produced no mask for the given prompts", rec)

        mask_u8 = mask.astype(np.uint8) * 255
        response.mask = self.bridge.cv2_to_imgmsg(mask_u8, encoding="mono8")
        response.mask.header = request.image.header
        response.score = float(score)
        response.success = True
        response.message = "ok"
        H, W = mask.shape
        area = int(mask.sum())
        coverage = area / float(mask.size) if mask.size else 0.0
        rec.set(score=float(score), mask_pixels=area,
                mask_coverage=round(coverage, 4), image_hw=[int(H), int(W)])
        self.get_logger().info(
            f"SAM metrics: score={score:.3f} mask={area}px "
            f"({coverage * 100:.1f}% of {W}x{H}) text={text!r} "
            f"+boxes={len(positive_boxes)} -boxes={len(negative_boxes)}")
        rec.save_image('mask', mask_u8)
        self._save_sam_overlay(rec, rgb, mask, positive_boxes, negative_boxes)
        rec.finish(success=True, message="ok")
        return response

    def _save_sam_overlay(self, rec, rgb, mask, positive_boxes, negative_boxes):
        """Save the input, mask tinted and boxes drawn: positive green, negative red."""
        if not self.logger.visualize:
            return
        try:
            import cv2
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            m = mask.astype(bool)
            tint = np.zeros_like(bgr)
            tint[m] = (0, 255, 0)
            bgr = cv2.addWeighted(bgr, 1.0, tint, 0.5, 0.0)
            for boxes, color in ((positive_boxes, (0, 255, 0)),
                                 (negative_boxes, (0, 0, 255))):
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.imwrite(rec.path('overlay', 'png'), bgr)
        except Exception as e:
            self.get_logger().warn(f'sam overlay failed: {e}')

    def _fail(self, response, msg, rec=None):
        """Fill `response` as a failure with `msg` and an empty mask; return it."""
        response.score = 0.0
        response.success = False
        response.message = msg
        self.get_logger().warn(f"sam_segment: {msg}")
        if rec is not None:
            rec.finish(success=False, message=msg)
        return response


def RunSamServer(args=None):
    """Spin a SamServer on a MultiThreadedExecutor until interrupted."""
    rclpy.init(args=args)
    node = SamServer()
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
    RunSamServer()
