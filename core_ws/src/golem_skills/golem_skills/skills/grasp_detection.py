"""
Find /skill/grasp's target in the head camera and lift it to a point cloud.

GraspDetection is a mixin of GraspSkill. Gemini boxes every visible
instance of the target, the instance closest to the grasping arm is kept,
and SAM segments it; the mask's pixels, back-projected with the aligned
depth, are the object cloud. The same path runs in the sim and on the robot.
"""

import numpy as np

from geometry_msgs.msg import Point
from custom_ros_messages.msg import Detection

from ..base import GEMINI_TIMEOUT_SEC, GRASP_FRAMES
from ..grasp_synthesis.common import pose_at
from ..perception_utils import gemini_boxes


# Gemini phrases for the battery-workcell class labels (yolo_server's
# DEFAULT_QUERIES, spelled as its labels, 'OrageCover' included), so a label
# given as goal.target_object is asked about in words; any other target is
# sent as given.
GEMINI_NAMES = {'bolt': 'bolt', 'busbar': 'bus bar', 'interiorscrew': 'interior screw',
                'nut': 'nut', 'oragecover': 'orange cover', 'screw': 'screw',
                'screwhole': 'screw hole'}

# One box per VISIBLE instance (base.GEMINI_GRASP_PROMPT asks for exactly one),
# so look-alikes such as several grey screws can be told apart: the instance
# closest to the working arm is grasped (detect_object). Each box still fully
# encompasses its instance, so it stays a valid SAM prompt.
GEMINI_MULTI_GRASP_PROMPT = (
    'Locate EVERY visible {obj} so a segmentation model can be prompted with '
    'bounding boxes. Return a JSON array with ONE box per distinct {obj} '
    'instance in view, each box FULLY ENCOMPASSING that whole instance — every '
    'visible part of it, edges snug to its outermost extent: '
    '[{{"box_2d": [y1, x1, y2, x2], "label": "{obj}", "score": 0.9}}, ...], '
    'integer coordinates normalized to 0-1000 with y first. Always return '
    '4-number "box_2d" boxes — never single points. Return one entry per '
    'instance; if no {obj} is visible, return [].'
)


def perception_name(obj):
    """Return the phrase perception queries use for `obj`: a class label's GEMINI_NAMES entry."""
    return GEMINI_NAMES.get(obj.strip().lower(), obj)


class GraspDetection:
    """Box, segment and lift the grasp target from the head camera."""

    def detect_object(self, obj, run, gh, arm=None):
        """
        Return the pixel-xyxy Gemini box of the `obj` instance closest to `arm`'s gripper, or None.

        With no arm, the instance closest to the pelvis origin. Falls back to
        base._gemini_box (the single-box query) when the multi-instance query
        returns zero or one box, or no box has usable depth to range. None
        also on a cancel or timeout of `run`; SAM then uses the text alone.
        """
        boxes = self._gemini_boxes(obj, run, gh)
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return None
        if len(boxes) <= 1:
            return boxes[0] if boxes else self._gemini_box(obj, run, gh)
        ref = None
        if arm in GRASP_FRAMES:
            o = self._transform_pose(pose_at((0.0, 0.0, 0.0)), GRASP_FRAMES[arm], 'pelvis')
            if o is not None:
                ref = np.array([o.position.x, o.position.y, o.position.z])
        if ref is None:
            ref = np.zeros(3)
        best_box, best_pt, best_d = None, None, None
        for box in boxes:
            pt = self._box_point_pelvis(box)
            if pt is None:
                continue
            d = float(np.linalg.norm(pt - ref))
            if best_d is None or d < best_d:
                best_box, best_pt, best_d = box, pt, d
        if best_box is None:
            self.get_logger().warn(
                f'grasp: {len(boxes)} {obj!r} instances in view but none had '
                f'usable head-camera depth; falling back to the single-box query')
            return self._gemini_box(obj, run, gh)
        where = f'the {arm} arm' if arm in GRASP_FRAMES else 'the robot'
        self.get_logger().info(
            f'grasp: {len(boxes)} {obj!r} instances in view; picked the one at '
            f'pelvis ({best_pt[0]:.3f}, {best_pt[1]:.3f}, {best_pt[2]:.3f}), '
            f'closest to {where} ({best_d:.3f} m)')
        return best_box

    def _gemini_boxes(self, obj, run, gh):
        """
        Return every Gemini box (pixel xyxy) of `obj` in the head camera, one per instance.

        [] when none parse. Capped at the skill's remaining budget; a cancel
        of `gh` ends the call.
        """
        timeout = min(GEMINI_TIMEOUT_SEC, run.remaining())
        txt = self.query_gemini(GEMINI_MULTI_GRASP_PROMPT.format(obj=obj),
                                timeout_sec=timeout, outer_gh=gh)
        info = self.latest_caminfo()
        if info is None or not info.width or not info.height:
            return []
        return gemini_boxes(txt, info.width, info.height)

    def _box_point_pelvis(self, box):
        """
        Back-project a pixel-xyxy head-camera `box` with the head depth to one pelvis-frame point.

        The median over the box's central quarter, which avoids background
        depth at the edges; None without depth.
        """
        info = self.latest_caminfo()
        if info is None or not info.width or not info.height:
            return None
        w, h = int(info.width), int(info.height)
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        half_w = max(1.0, abs(box[2] - box[0]) * 0.25)
        half_h = max(1.0, abs(box[3] - box[1]) * 0.25)
        u0, u1 = max(0, int(round(cx - half_w))), min(w, int(round(cx + half_w)))
        v0, v1 = max(0, int(round(cy - half_h))), min(h, int(round(cy + half_h)))
        if u1 <= u0 or v1 <= v0:
            return None
        mask = np.zeros((h, w), dtype=bool)
        mask[v0:v1, u0:u1] = True
        pts = self._depth_to_cloud(mask, 'pelvis')
        if pts is None or len(pts) == 0:
            return None
        return np.median(pts, axis=0)

    def segment_object(self, obj, box, gh):
        """
        Segment `obj` with SAM and lift it; return (obj_cloud, scene_cloud, reason or None).

        SAM gets the object name plus the box when there is one, and only the
        masked pixels are back-projected (pelvis frame). The whole-frame scene
        cloud is the obstacle context for graspgen's collision filter.
        """
        mask = self.segment(text=obj, positive_boxes=box, outer_gh=gh)
        if mask is None:
            return None, None, f'no mask for {obj!r}'
        obj_cloud = self.mask_to_cloud(mask, target_frame='pelvis')
        if obj_cloud is None:
            return None, None, f'{obj!r} mask produced no usable cloud'
        return obj_cloud, self.scene_to_cloud(target_frame='pelvis'), None


def box_detection(obj, box):
    """Wrap a pixel-xyxy `box` as a custom_ros_messages Detection of class `obj`."""
    det = Detection()
    det.cls = obj
    det.bbox_min = Point(x=float(box[0]), y=float(box[1]), z=0.0)
    det.bbox_max = Point(x=float(box[2]), y=float(box[3]), z=0.0)
    return det
