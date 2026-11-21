"""
Align the gripper on /skill/grasp's target with the hand camera (goal.visual_servo).

GraspServo is a mixin of GraspSkill. At the pre-grasp, servo() drives the
gripper in pure translation until the target sits at a setpoint in the hand
camera's optical frame, then the caller closes there. Each measurement is a
Gemini box on the hand-camera frame, segmented by SAM and lifted with that
frame's aligned depth; yolo_server's hand-camera bundle supplies the frame,
its depth and intrinsics, so yolo_server must run for the servo.
"""

import cv2
import numpy as np

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, Vector3Stamped
from rclpy.duration import Duration as RclpyDuration
from sensor_msgs.msg import Image

from ..base import DEPTH_MAX_M, FRAME_TASK_STABLE_STOP_MSGS
from ..perception_utils import deproject_mask, gemini_boxes
from .grasp_detection import box_detection
from .grasp_markers import approach_target_tf


# The hand camera looks straight down the approach axis, 4.4 cm ahead of the
# gripper base, so "centered at depth D" means "on the approach ray, D ahead of
# the camera". The jaws close along the optical Y axis. The Magpie fingers are
# a linkage: the pads' far edges are 126 mm ahead of the camera fully open and
# reach up to 150 mm as they close (FINGERTIP_REACH_M), well past the 70.6 mm
# grasp contact point that grasp_synthesis plans with.
#
# m, the deepest camera -> object depth driven to; default of the node's
# `servo_depth` parameter. At 100 mm the open fingertips end 26 mm beyond the
# object's near face, which suits a part standing taller than the fingers'
# sweep. For a shorter part, servo_depth_floor raises the setpoint so the
# closing fingertips stop VISUAL_SERVO_SURFACE_CLEARANCE_M above the surface
# the part stands on. The servo drives depth TO the setpoint, so it is not a
# band centre: a one-sided band such as (0, 105) would centre the setpoint at
# 52.5 mm and make the servo plunge into the part whenever it starts beyond
# 105 mm.
VISUAL_SERVO_DEPTH_M = 0.10
# Per-axis convergence tolerances [m]. Each sits above what one measurement
# resolves: a Gemini box segmented by SAM moves the object's X by about
# +/-3 mm between frames of a still hand, and frame_task settles to ~1.2 mm,
# so a tighter band is met only by chance.
#   X (across the jaws), what the grasp depends on: the pads are 16 mm wide,
#     so 3 mm keeps the part on them.
#   Y (along the jaw span): the jaws open to 108 mm and centre the part as
#     they close, so 5 mm still holds it near the middle of the grip.
#   range: decides how far the pads overlap the part, and the surface
#     clearance is VISUAL_SERVO_SURFACE_CLEARANCE_M, so it stays below that.
# Keep X <= Y: the reverse is backwards for a parallel gripper.
VISUAL_SERVO_X_TOL_M = 0.003
VISUAL_SERVO_Y_TOL_M = 0.005
VISUAL_SERVO_RANGE_TOL_M = 0.002
# (pad gap [m], depth [m] of the pads' far edges ahead of the hand camera) of
# the Magpie gripper, fully open to closed: the sim MJCF's crank swept to rest
# at each angle. Unmeasured on the real H1-2.
FINGERTIP_REACH_M = ((0.000, 0.148), (0.009, 0.149), (0.018, 0.150),
                     (0.027, 0.150), (0.036, 0.150), (0.045, 0.150),
                     (0.054, 0.148), (0.062, 0.147), (0.071, 0.145),
                     (0.078, 0.143), (0.086, 0.140), (0.092, 0.137),
                     (0.098, 0.134), (0.104, 0.130), (0.108, 0.126))
# m: the pads' far edges in the optical frame, X across them (the same for
# both pads) and Y their inner faces fully open, from the same sim sweep.
# Unmeasured on the real H1-2.
PAD_X_SPAN_M = (-0.004, 0.012)
PAD_OPEN_Y_M = 0.054
# m: how far above the surface the part stands on the pads' far edges stop
# once closed on the part
VISUAL_SERVO_SURFACE_CLEARANCE_M = 0.005
# m: the support surface is read in a band this wide beyond the part's edge
# along the jaw span, where the pads end their close, and this far beyond
# the pads' X span: the deepest the fingertips reach in a close
SURFACE_BAND_M = 0.02
SURFACE_X_MARGIN_M = 0.005
# fewest depth pixels in that band for a surface reading
SURFACE_MIN_POINTS = 30
# percentile of the band's depths taken as the surface: low, so the nearest
# surface counts, but not the minimum, which one stray pixel would set
SURFACE_PERCENTILE = 10
# Approach blocked: an inward range step of at least BLOCKED_MIN_STEP_M that
# the next measurement shows covered by less than BLOCKED_FRACTION of it, on
# BLOCKED_STRIKES iterations in a row, means the fingers are against
# something. The setpoint then becomes the depth where the hand stopped plus
# BLOCKED_BACKOFF_M, so the loop stops pushing (a pushed arm leans the robot).
BLOCKED_MIN_STEP_M = 0.003
BLOCKED_FRACTION = 0.5
BLOCKED_STRIKES = 2
BLOCKED_BACKOFF_M = 0.005
# s of robot time for the whole loop; the goal's own timeout still wins
VISUAL_SERVO_TIMEOUT_SEC = 90.0
# s per hand-camera Gemini query. A query takes seconds to minutes, far longer
# than a frame_task iteration, so the servo is slow but works on any object
# Gemini can name; sight counts as lost after 1.5 queries without a usable box.
SERVO_GEMINI_TIMEOUT_SEC = 60.0
VISUAL_SERVO_DETECT_WAIT_SEC = 1.5 * SERVO_GEMINI_TIMEOUT_SEC
# m: lost sight is usually geometry (out of frame, occluded by the fingers,
# inside the hand camera's minimum range), so the hand backs straight up
# (pelvis +Z) by this much and keeps servoing
VISUAL_SERVO_LOST_RETRACT_M = 0.05
# m: depth at or below which the object is too close to steer by: it would sit
# between the finger links, short of the pads
VISUAL_SERVO_MIN_RANGE_M = 0.055
# consecutive too-close readings before backing off, so one frame never
# commands a 5 cm retreat
VISUAL_SERVO_TOO_CLOSE_STRIKES = 2
# m: a measurement's depth must lie within this band around the last depth and
# the one the last correction predicts; rejects fingers, shadows and background
# but not honest motion (VISUAL_SERVO_MAX_STEP_M is inside the band)
VISUAL_SERVO_RANGE_BAND_M = 0.02
# Back-offs per run, lost-sight and too-close together: unbounded, the loop
# would walk the arm out of the workspace 5 cm at a time.
VISUAL_SERVO_MAX_RETRACTS = 2
# Stall detection: an iteration is progress only if it beats the best error by
# STALL_EPS; STALL_ITERS iterations without progress end the run with the
# residual in the message (both reset after a back-off).
VISUAL_SERVO_STALL_ITERS = 5
VISUAL_SERVO_STALL_EPS_M = 0.001
# proportional gain, < 1 so a bad depth sample cannot fling the hand and the
# loop damps rather than oscillates
VISUAL_SERVO_GAIN = 0.6
# m, largest single correction: bounds what one mis-detection can do
VISUAL_SERVO_MAX_STEP_M = 0.05
# s per correction move. Includes the server's steady-state hold: cutting the
# hold short leaves gravity droop in each step (~5 mm per iteration measured).
VISUAL_SERVO_MOVE_SEC = 4.0
# s after each move before the next measurement, so the loop does not act on a
# frame captured mid-move
VISUAL_SERVO_POST_MOVE_SLEEP_SEC = 0.25
# m, the object's setpoint (X, Y) in the camera optical frame; depth is
# `servo_depth`. Fixed values, not derived from TF (the hand-camera mount
# transform is not trusted for this): correct a camera-to-finger offset by
# measuring the mount, not with a trim here. Positive X puts the object further
# right in the image, i.e. the hand further back along pelvis +X.
VISUAL_SERVO_TARGET_XY_M = (0.012, 0.0)
# m, depth floor of the servo's back-projection: base.DEPTH_MIN_M (0.1) is
# above the target depth and would reject the measurements the servo needs
VISUAL_SERVO_MIN_DEPTH_M = 0.04

# The hand-camera box. The servo only translates (it keeps the planned grasp's
# orientation), so no grasp axis is asked for.
GEMINI_SERVO_PROMPT = (
    'A robot gripper is about to pick up the {obj} in this hand-camera image. '
    'Return ONE JSON object and no prose: {{"box_2d": [y1, x1, y2, x2]}} where '
    'box_2d FULLY ENCOMPASSES the {obj}. Coordinates are integers normalized to '
    '0-1000 with y first. If the {obj} is not visible, return {{}}.'
)


class GraspServo:
    """Run the hand-camera visual servo at the pre-grasp."""

    def servo(self, run, gh, arm, obj, targets, idx):
        """
        Align the gripper on `obj` with the hand camera at the pre-grasp.

        Drives the arm's grasp frame in PURE TRANSLATION until `obj` sits at
        (VISUAL_SERVO_TARGET_XY_M, depth setpoint) in the hand camera's
        optical frame; the orientation stays the planned grasp's. The depth
        setpoint is `servo_depth`, raised to the median servo_depth_floor of
        the measurements so far (the closing pads stay off the surface under
        the part) and, once the approach is blocked, to where it stopped.

        One iteration takes a hand-camera measurement newer than the last
        frame used (reusing a frame would apply its correction twice), and
        steps the gripper by GAIN x error, rotated into the pelvis frame and
        capped at MAX_STEP.

        Lost sight and too-close are recoverable: back straight up (pelvis +Z)
        by VISUAL_SERVO_LOST_RETRACT_M and carry on, at most
        VISUAL_SERVO_MAX_RETRACTS times.

        Returns (pose, None), the pelvis-frame gripper Pose it converged at,
        where the caller closes with no contact move, or (None, reason).
        """
        frame = self.robot.grasp_frames[arm]
        planned_q = targets.grasps_p[idx].orientation
        servo_depth = float(self.get_parameter('servo_depth').value)
        target = np.array([VISUAL_SERVO_TARGET_XY_M[0],
                           VISUAL_SERVO_TARGET_XY_M[1], servo_depth])
        # servo_depth_floor of each measurement that read the surface [m],
        # and the depth setpoint a blocked approach set [m]
        floors, blocked_depth = [], None
        blocked_hits = 0
        deadline = self.robot_now() + VISUAL_SERVO_TIMEOUT_SEC
        last_seen = self.robot_now()
        retracts = 0
        best_err, stalled = None, 0
        # consecutive readings inside VISUAL_SERVO_MIN_RANGE_M
        too_close_hits = 0
        # last measured depth and the depth the last correction predicts [m]
        last_z = pred_z = None
        # only frames captured after now count, so the first measurement
        # describes the arm where it is
        after = _stamp_tuple(self.get_clock().now().to_msg())
        self.get_logger().info(
            f'grasp: visual servo for {obj!r} on the {arm} hand camera — setpoint '
            f'({target[0] * 1000:+.0f}, {target[1] * 1000:+.0f}, '
            f'{target[2] * 1000:.0f})mm optical '
            f'(tol {VISUAL_SERVO_X_TOL_M * 1000:.2f}mm across the jaws / '
            f'{VISUAL_SERVO_Y_TOL_M * 1000:.0f}mm along them / '
            f'{VISUAL_SERVO_RANGE_TOL_M * 1000:.0f}mm range)')
        it = 0
        while True:
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return None, 'canceled or timed out during visual servo'
            if self.robot_now() >= deadline:
                return None, (
                    f'visual servo did not converge on {obj!r} within '
                    f'{VISUAL_SERVO_TIMEOUT_SEC:.0f}s ({it} iteration(s), tol '
                    f'{VISUAL_SERVO_X_TOL_M * 1000:.2f}/'
                    f'{VISUAL_SERVO_Y_TOL_M * 1000:.0f}mm X/Y)')
            p_cam, cam_frame, stamp, bundle, floor = self._servo_measure(
                arm, obj, after, gh)
            # A depth outside the band around the last and the predicted depth
            # is not this object (finger, shadow, background): drop the frame.
            # Persistent implausibility ends in the lost-sight back-off.
            if p_cam is not None and pred_z is not None:
                lo = min(last_z, pred_z) - VISUAL_SERVO_RANGE_BAND_M
                hi = max(last_z, pred_z) + VISUAL_SERVO_RANGE_BAND_M
                if not lo <= float(p_cam[2]) <= hi:
                    self.get_logger().warn(
                        f'grasp: visual servo ignoring implausible depth '
                        f'{p_cam[2] * 1000:.0f}mm (expected '
                        f'{lo * 1000:.0f}-{hi * 1000:.0f}mm)',
                        throttle_duration_sec=2.0)
                    # consumed: do not test this frame again
                    after = stamp
                    self.get_clock().sleep_for(RclpyDuration(seconds=0.1))
                    continue
            # Lost (nothing usable for the detect wait) and too close (inside
            # MIN_RANGE for STRIKES frames in a row) are both answered by
            # backing straight up and looking again.
            if p_cam is not None and float(p_cam[2]) < VISUAL_SERVO_MIN_RANGE_M:
                too_close_hits += 1
            else:
                too_close_hits = 0
            lost = (p_cam is None
                    and self.robot_now() - last_seen > VISUAL_SERVO_DETECT_WAIT_SEC)
            too_close = too_close_hits >= VISUAL_SERVO_TOO_CLOSE_STRIKES
            if lost or too_close:
                why = (f'lost sight of {obj!r} in the {arm} hand camera for '
                       f'{VISUAL_SERVO_DETECT_WAIT_SEC:.0f}s' if lost else
                       f'{obj!r} at {p_cam[2] * 1000:.0f}mm is inside the '
                       f'{VISUAL_SERVO_MIN_RANGE_M * 1000:.0f}mm minimum range')
                if retracts >= VISUAL_SERVO_MAX_RETRACTS:
                    return None, (
                        f'{why}; still not servo-able after {retracts} retract(s) of '
                        f'{VISUAL_SERVO_LOST_RETRACT_M * 100:.0f}cm')
                retracts += 1
                self.get_logger().warn(
                    f'grasp: {why} — backing off '
                    f'{VISUAL_SERVO_LOST_RETRACT_M * 100:.0f}cm up (pelvis +Z) '
                    f'({retracts}/{VISUAL_SERVO_MAX_RETRACTS})')
                cur = self._frame_pose_in_pelvis(frame)
                if cur is None:
                    return None, (f'{why} and TF {frame!r} -> pelvis failed, so the '
                                  'back-off pose is unknown')
                up = Pose()
                up.position.x = cur.position.x
                up.position.y = cur.position.y
                up.position.z = cur.position.z + VISUAL_SERVO_LOST_RETRACT_M
                up.orientation = planned_q
                self.move_frame_to(
                    frame, up, outer_gh=gh,
                    duration_sec=VISUAL_SERVO_MOVE_SEC, do_plan=False,
                    stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                    label=f'visual servo back-off {retracts}/'
                          f'{VISUAL_SERVO_MAX_RETRACTS}')
                # The hand is somewhere new: `after` becomes a clock instant
                # (frames from before the move show the old vantage point) and
                # the lost-sight wait, stall counters and depth prediction reset.
                after = _stamp_tuple(self.get_clock().now().to_msg())
                last_seen = self.robot_now()
                best_err, stalled = None, 0
                last_z = pred_z = None
                too_close_hits = 0
                continue
            if p_cam is None:
                self.get_clock().sleep_for(RclpyDuration(seconds=0.1))
                continue
            last_seen = self.robot_now()
            if floor is not None:
                floors.append(floor[0])
            # last_z is None after a back-off, which moved the hand outward
            if last_z is not None:
                covered = last_z - float(p_cam[2])
                if approach_blocked(last_z - pred_z, covered):
                    blocked_hits += 1
                else:
                    blocked_hits = 0
                if blocked_hits >= BLOCKED_STRIKES:
                    blocked_hits = 0
                    blocked_depth = max(blocked_depth or 0.0,
                                        float(p_cam[2]) + BLOCKED_BACKOFF_M)
                    self.get_logger().warn(
                        f'grasp: visual servo approach blocked at '
                        f'{p_cam[2] * 1000:.0f}mm (covered {covered * 1000:.1f} of '
                        f'{(last_z - pred_z) * 1000:.1f}mm inward) — the fingers '
                        f'are against something; depth setpoint no deeper than '
                        f'{blocked_depth * 1000:.0f}mm')
            depth_sp = max([servo_depth]
                           + ([float(np.median(floors))] if floors else [])
                           + ([blocked_depth] if blocked_depth is not None else []))
            if abs(depth_sp - target[2]) > VISUAL_SERVO_STALL_EPS_M:
                if blocked_depth is not None and depth_sp == blocked_depth:
                    why = 'blocked approach'
                elif floor is not None and depth_sp > servo_depth:
                    why = (f'part {floor[1] * 1000:.0f}mm above its surface, '
                           f'{floor[2] * 1000:.0f}mm across the jaws')
                else:
                    why = 'surface readings'
                self.get_logger().info(
                    f'grasp: visual servo depth setpoint {depth_sp * 1000:.0f}mm '
                    f'({why}; servo_depth {servo_depth * 1000:.0f}mm)')
                # the error is measured against a new setpoint
                best_err, stalled = None, 0
            target[2] = depth_sp
            err = p_cam - target
            ex, ey = float(abs(err[0])), float(abs(err[1]))
            rng = float(abs(err[2]))
            # Moving the camera by d shifts the object's camera coordinates by
            # -d, so the camera moves BY the error, gain-scaled and capped. All
            # three axes move together, also when only X/Y are still off.
            step = VISUAL_SERVO_GAIN * err
            norm = float(np.linalg.norm(step))
            if norm > VISUAL_SERVO_MAX_STEP_M:
                step *= VISUAL_SERVO_MAX_STEP_M / norm
                self.get_logger().warn(
                    f'grasp: visual servo step capped at '
                    f'{VISUAL_SERVO_MAX_STEP_M * 100:.0f}cm (wanted {norm * 100:.1f}cm)')
            self._publish_servo_error(stamp, cam_frame, err)
            self._publish_servo_image(bundle, p_cam, target, step)
            self.get_logger().info(
                f'grasp: visual servo iter {it}: object at ({p_cam[0] * 1000:+.0f}, '
                f'{p_cam[1] * 1000:+.0f}, {p_cam[2] * 1000:.0f})mm in {cam_frame} — '
                f'err X {err[0] * 1000:+.1f}mm (tol {VISUAL_SERVO_X_TOL_M * 1000:.2f}) '
                f'Y {err[1] * 1000:+.1f}mm (tol {VISUAL_SERVO_Y_TOL_M * 1000:.0f}) '
                f'range {err[2] * 1000:+.1f}mm (tol {VISUAL_SERVO_RANGE_TOL_M * 1000:.0f})')
            if (ex <= VISUAL_SERVO_X_TOL_M and ey <= VISUAL_SERVO_Y_TOL_M
                    and rng <= VISUAL_SERVO_RANGE_TOL_M):
                pose = self._frame_pose_in_pelvis(frame)
                if pose is None:
                    return None, (f'visual servo converged but TF {frame!r} -> pelvis '
                                  'failed, so the close pose is unknown')
                pose.orientation = planned_q
                self.get_logger().info(
                    f'grasp: visual servo converged for {obj!r} in {it} iteration(s)')
                return pose, None
            # a run that stops improving does not start again by itself: end it
            # with the residual, naming the out-of-tolerance axes
            err_norm = float(np.linalg.norm(err))
            if best_err is None or err_norm < best_err - VISUAL_SERVO_STALL_EPS_M:
                best_err, stalled = err_norm, 0
            else:
                stalled += 1
                if stalled >= VISUAL_SERVO_STALL_ITERS:
                    out = ', '.join(
                        f'{ax} {v * 1000:+.1f}mm (tol {t * 1000:.2f})'
                        for ax, v, t in (('X', err[0], VISUAL_SERVO_X_TOL_M),
                                         ('Y', err[1], VISUAL_SERVO_Y_TOL_M),
                                         ('range', err[2], VISUAL_SERVO_RANGE_TOL_M))
                        if abs(v) > t)
                    return None, (
                        f'visual servo stalled on {obj!r}: no improvement over '
                        f'{stalled} iterations, still out of tolerance on {out} — '
                        'the arm does not reach the setpoint, or the setpoint is off')
            delta = self._rotate_into_pelvis(step, cam_frame)
            cur = self._frame_pose_in_pelvis(frame)
            if delta is None or cur is None:
                return None, (f'visual servo TF lookup for {cam_frame!r}/{frame!r} '
                              'failed mid-loop')
            goal_pose = Pose()
            goal_pose.position.x = cur.position.x + float(delta[0])
            goal_pose.position.y = cur.position.y + float(delta[1])
            goal_pose.position.z = cur.position.z + float(delta[2])
            goal_pose.orientation = planned_q
            # servo targets are in the pelvis frame by construction
            self.publish_tf(approach_target_tf(
                targets.frame, goal_pose, self.get_clock().now().to_msg()))
            # an undershoot is fine: the next measurement sees the error left
            self.move_frame_to(
                frame, goal_pose, outer_gh=gh,
                duration_sec=VISUAL_SERVO_MOVE_SEC,
                do_plan=False, stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                label=f'visual servo correction iter {it}')
            self._log_servo_tracking(frame, cur, delta, it)
            # a fully executed move lands at z - step[2], an ignored one stays at z
            last_z, pred_z = float(p_cam[2]), float(p_cam[2]) - float(step[2])
            after = stamp
            self.get_clock().sleep_for(
                RclpyDuration(seconds=VISUAL_SERVO_POST_MOVE_SLEEP_SEC))
            it += 1

    def _servo_measure(self, arm, obj, after, gh):
        """
        Measure `obj` in `arm`'s hand camera: (optical point, frame_id, stamp, bundle, floor).

        Gemini boxes the latest bundle's rgb frame, SAM segments the box and
        bundle_mask_point lifts the mask to an on-object point with the
        bundle's aligned depth; when SAM gives no point, the box's central
        quarter is used (bundle_detection_point). `floor` is _servo_floor's
        reading of the SAM mask, None without one. A tuple of Nones when
        nothing is usable. Only a bundle stamped strictly after `after` counts:
        `after` is the stamp of the frame last USED, so a frame's correction
        is never applied twice.
        """
        bundle = self.latest_arm_detections(arm)
        if bundle is None or not bundle.rgb_image.data:
            return None, None, None, None, None
        stamp = _stamp_tuple(bundle.rgb_image.header.stamp)
        if stamp <= after:
            return None, None, None, None, None
        box = self._gemini_box_on_image(obj, bundle.rgb_image, bundle.camera_info, gh)
        if box is None:
            return None, None, None, None, None
        mask = self.segment(text=obj, positive_boxes=box,
                            image=bundle.rgb_image, outer_gh=gh)
        p = self.bundle_mask_point(bundle, mask, min_depth=VISUAL_SERVO_MIN_DEPTH_M)
        floor = None if p is None else self._servo_floor(bundle, mask, p)
        if p is None:
            self.get_logger().warn(
                f'grasp: visual servo SAM mask for {obj!r} gave no point; '
                'using the box centre')
            p = self.bundle_detection_point(bundle, box_detection(obj, box),
                                            min_depth=VISUAL_SERVO_MIN_DEPTH_M)
        if p is None:
            return None, None, None, None, None
        return p, self.bundle_camera_frame(bundle), stamp, bundle, floor

    def _servo_floor(self, bundle, mask_msg, p_obj):
        """
        Read the depth floor from one hand-camera frame: (floor, height, width) [m], or None.

        The part's width across the jaws comes from the SAM mask's points,
        its height above the surface from the frame's other pixels
        (surface_height), and the floor from both (servo_depth_floor). None
        when the frame has no depth or the surface band has too few pixels.
        """
        depth, info = self._bundle_depth(bundle, 'visual servo floor')
        if depth is None:
            return None
        mask = (np.frombuffer(bytes(mask_msg.data), dtype=np.uint8)
                .reshape(mask_msg.height, mask_msg.width) > 127)
        if mask.shape != depth.shape:
            return None
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        obj_pts = deproject_mask(mask, depth, fx, fy, cx, cy,
                                 VISUAL_SERVO_MIN_DEPTH_M, DEPTH_MAX_M)
        if len(obj_pts) == 0:
            return None
        width = part_width(obj_pts)
        height = surface_height(
            deproject_mask(~mask, depth, fx, fy, cx, cy,
                           VISUAL_SERVO_MIN_DEPTH_M, DEPTH_MAX_M),
            p_obj, width, VISUAL_SERVO_TARGET_XY_M)
        if height is None:
            return None
        return servo_depth_floor(height, width), height, width

    def _gemini_box_on_image(self, obj, image, info, gh):
        """
        Return Gemini's box (pixel xyxy) of `obj` in the CompressedImage `image`, or None.

        Scaled by `info`'s width and height: the hand-camera counterpart of
        base._gemini_box, which reads the head caches. A cancel of `gh` ends
        the call.
        """
        if info is None or not info.width or not info.height:
            return None
        txt = self.query_gemini(GEMINI_SERVO_PROMPT.format(obj=obj), image=image,
                                timeout_sec=SERVO_GEMINI_TIMEOUT_SEC, outer_gh=gh)
        boxes = gemini_boxes(txt, info.width, info.height)
        return boxes[0] if boxes else None

    def _publish_servo_error(self, stamp, cam_frame, err):
        """Publish the servo residual stamped with its frame's capture time, to align plots."""
        self.servo_error_pub.publish(_vec3_stamped(
            Time(sec=stamp[0], nanosec=stamp[1]), cam_frame, err))

    def _log_servo_tracking(self, frame, cur, delta, it):
        """
        Log and publish the executed servo move and its residual against the correction.

        Debug output only: a failed TF lookup skips it.
        """
        post = self._frame_pose_in_pelvis(frame)
        if post is None:
            return
        executed = np.array([post.position.x - cur.position.x,
                             post.position.y - cur.position.y,
                             post.position.z - cur.position.z])
        residual = executed - delta
        self.get_logger().info(
            f'grasp: visual servo iter {it}: '
            f'commanded ({delta[0] * 1000:+.1f}, {delta[1] * 1000:+.1f}, '
            f'{delta[2] * 1000:+.1f})mm, '
            f'executed ({executed[0] * 1000:+.1f}, {executed[1] * 1000:+.1f}, '
            f'{executed[2] * 1000:+.1f})mm, '
            f'residual ({residual[0] * 1000:+.1f}, {residual[1] * 1000:+.1f}, '
            f'{residual[2] * 1000:+.1f})mm (pelvis frame)')
        now = self.get_clock().now().to_msg()
        self.servo_move_pub.publish(_vec3_stamped(now, 'pelvis', executed))
        self.servo_move_residual_pub.publish(_vec3_stamped(now, 'pelvis', residual))

    def _publish_servo_image(self, bundle, p_cam, target, step=None):
        """
        Publish one annotated visual-servo frame on /skill/grasp/servo_image.

        Draws the measured object (red dot), the setpoint with its per-axis X/Y
        tolerance as a RECTANGLE (exactly the convergence test), and the
        commanded correction as an arrow to the object's predicted next
        position, with per-axis readings in the caption. Best-effort: drawing
        never breaks a grasp.
        """
        try:
            info = bundle.camera_info
            if not bundle.rgb_image.data or not info.width:
                return
            img = cv2.imdecode(
                np.frombuffer(bytes(bundle.rgb_image.data), dtype=np.uint8),
                cv2.IMREAD_COLOR)
            if img is None:
                return
            fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]

            def project(pt):
                """Return the pixel of an optical-frame point, or None behind the camera."""
                z = float(pt[2])
                if z <= 1e-6:
                    return None
                return (int(round(fx * float(pt[0]) / z + cx)),
                        int(round(fy * float(pt[1]) / z + cy)))

            # The setpoint's X/Y and its tolerance box are projected at the
            # OBJECT's current depth, not the depth setpoint: the convergence
            # test is on the 3-D X/Y error, so only at the same depth does
            # "dot inside box" match it.
            z_ref = float(p_cam[2]) if float(p_cam[2]) > 1e-6 else float(target[2])
            tgt_px = project(np.array([target[0], target[1], z_ref]))
            obj_px = project(p_cam)
            if tgt_px is not None:
                # tolerances [m] at that depth, through the same pinhole scale
                ax = max(2, int(round(fx * VISUAL_SERVO_X_TOL_M / z_ref)))
                ay = max(2, int(round(fy * VISUAL_SERVO_Y_TOL_M / z_ref)))
                cv2.rectangle(img, (tgt_px[0] - ax, tgt_px[1] - ay),
                              (tgt_px[0] + ax, tgt_px[1] + ay), (0, 255, 0), 2)
                cv2.drawMarker(img, tgt_px, (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
            if obj_px is not None:
                cv2.circle(img, obj_px, 6, (0, 0, 255), -1)
            if tgt_px is not None and obj_px is not None:
                cv2.line(img, obj_px, tgt_px, (0, 200, 255), 1)
            cmd_color = _step_color(step[2] if step is not None else 0.0)
            if obj_px is not None and step is not None:
                nxt_px = project(np.asarray(p_cam) - np.asarray(step))
                if nxt_px is not None and nxt_px != obj_px:
                    cv2.arrowedLine(img, obj_px, nxt_px, cmd_color, 2, tipLength=0.3)
            if tgt_px is not None and obj_px is not None:
                # one line per axis: the reading and its accepted (min, max)
                # band, green when the reading is inside it
                axes = (('X', p_cam[0], target[0], VISUAL_SERVO_X_TOL_M),
                        ('Y', p_cam[1], target[1], VISUAL_SERVO_Y_TOL_M),
                        ('Z', p_cam[2], target[2], VISUAL_SERVO_RANGE_TOL_M))
                for k, (name, reading, ctr, tol) in enumerate(axes):
                    lo, hi = ctr - tol, ctr + tol
                    ok = lo <= float(reading) <= hi
                    cv2.putText(
                        img, f'{name} {reading * 1000:.0f} mm '
                             f'({lo * 1000:.0f}, {hi * 1000:.0f})',
                        (10, 28 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if ok else (0, 0, 255), 2)
                if step is not None:
                    cv2.putText(
                        img, f'cmd X{step[0] * 1000:+.1f} Y{step[1] * 1000:+.1f} '
                             f'Z{step[2] * 1000:+.1f} mm',
                        (10, 28 + 30 * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cmd_color, 2)

            img = np.ascontiguousarray(img)
            msg = Image()
            msg.header = bundle.rgb_image.header
            msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
            msg.encoding = 'bgr8'
            msg.is_bigendian = 0
            msg.step = int(img.shape[1] * 3)
            msg.data = img.tobytes()
            self.servo_image_pub.publish(msg)
        except Exception as e:                       # noqa: BLE001 - viz only
            self.get_logger().warn(f'servo image publish failed: {e}',
                                   throttle_duration_sec=5.0)


def close_reach(width_m):
    """
    Return how deep [m] past the hand camera the pads reach closing onto a part `width_m` wide.

    The deepest point of the whole close (FINGERTIP_REACH_M), not only of its
    end: the linkage swings the pads deepest near a 30 mm gap, and a close
    onto a narrower part passes through it.
    """
    gaps = [g for g, _ in FINGERTIP_REACH_M]
    reach = [r for _, r in FINGERTIP_REACH_M]
    w = float(np.clip(width_m, gaps[0], gaps[-1]))
    return max([float(np.interp(w, gaps, reach))]
               + [r for g, r in FINGERTIP_REACH_M if g >= w])


def part_width(obj_pts):
    """Return the part's extent [m] across the jaws (optical Y) from its (N, 3) optical points."""
    lo, hi = np.percentile(np.asarray(obj_pts)[:, 1], (5, 95))
    return float(hi - lo)


def surface_height(scene_pts, p_obj, width_m, target_xy):
    """
    Return how far [m] beyond the part's point `p_obj` the surface under the pads lies, or None.

    `scene_pts` is the hand camera's (N, 3) optical-frame cloud without the
    part and `p_obj` the part's point in the same frame. The band read is
    where the pads end their close once the servo puts the part at
    `target_xy`: across the pads' X span (SURFACE_X_MARGIN_M wider each
    side), and along Y from the part's edge, half `width_m` from it,
    SURFACE_BAND_M outward on both sides, no further than the open pads. The
    surface is the band's SURFACE_PERCENTILE depth; None with fewer than
    SURFACE_MIN_POINTS pixels in the band.
    """
    pts = np.asarray(scene_pts, dtype=float).reshape(-1, 3)
    # where each pixel ends up once the part sits at target_xy
    x = pts[:, 0] - float(p_obj[0]) + target_xy[0]
    y = np.abs(pts[:, 1] - float(p_obj[1]) + target_xy[1])
    y_in = width_m / 2.0
    y_out = min(y_in + SURFACE_BAND_M, PAD_OPEN_Y_M)
    band = ((x >= PAD_X_SPAN_M[0] - SURFACE_X_MARGIN_M)
            & (x <= PAD_X_SPAN_M[1] + SURFACE_X_MARGIN_M)
            & (y >= y_in) & (y <= y_out))
    if np.count_nonzero(band) < SURFACE_MIN_POINTS:
        return None
    return float(np.percentile(pts[band, 2], SURFACE_PERCENTILE)) - float(p_obj[2])


def servo_depth_floor(height_m, width_m):
    """
    Return the nearest part depth [m] at which the pads close on it short of its surface.

    For a part `width_m` wide across the jaws whose near face stands
    `height_m` above its surface: the pads' deepest reach in the close
    (close_reach) stays VISUAL_SERVO_SURFACE_CLEARANCE_M short of the
    surface, so they overlap the part by `height_m` minus the clearance.
    """
    return close_reach(width_m) + VISUAL_SERVO_SURFACE_CLEARANCE_M - height_m


def approach_blocked(commanded_m, covered_m):
    """Return whether an inward range step `commanded_m` covering `covered_m` [m] hit something."""
    return (commanded_m >= BLOCKED_MIN_STEP_M
            and covered_m < BLOCKED_FRACTION * commanded_m)


def _stamp_tuple(stamp):
    """Return (sec, nanosec) of a builtin_interfaces/Time, for ordering stamps."""
    return (stamp.sec, stamp.nanosec)


def _step_color(dz):
    """
    Return the BGR color of the servo image's step arrow, coding its depth component.

    Red drives in (+Z, down onto the part at the top-down approach), green
    backs off, scaled over +/- VISUAL_SERVO_MAX_STEP_M.
    """
    t = (float(np.clip(dz / VISUAL_SERVO_MAX_STEP_M, -1.0, 1.0)) + 1.0) / 2.0
    return (0, int(round(255 * (1.0 - t))), int(round(255 * t)))


def _vec3_stamped(stamp, frame_id, v):
    """Build a geometry_msgs/Vector3Stamped from a 3-sequence (servo debug topics)."""
    msg = Vector3Stamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.vector.x = float(v[0])
    msg.vector.y = float(v[1])
    msg.vector.z = float(v[2])
    return msg
