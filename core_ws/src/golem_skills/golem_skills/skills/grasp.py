"""
Execute /skill/grasp: perception, a pluggable synthesis method, one executor.

The flow (_exec_grasp):

    prep -> detect_object (box) -> segment_object (cloud)
         -> get_grasp_candidates (the selected grasp_synthesis method)
         -> move_to_candidate_approach (take off in Z, transit to the pre-grasp)
         -> move_to_candidate (in along the grasp axis) + force close

Perception and execution are the same for every synthesis method; only the
candidate list differs (grasp_synthesis/__init__.py has the methods and how a
goal selects one). With goal.visual_servo the contact move is replaced by
servo(), a closed hand-camera loop that aligns on the object at the node's
`servo_depth` parameter and closes there; a servo that does not converge
still closes where it stopped, and a cancel or timeout aborts.

SAFETY: every arm move is refused once golem_safety's heartbeat halts the arms
or the frame_task server aborts a move (SkillsBase.arm_halt_reason); the skill
then aborts instead of trying more candidates or retreating.

Every target is boxed by Gemini and segmented by SAM while BATTERY_USE_GEMINI
is on. With it off, a battery-workcell part (BATTERY_OBJECTS) is boxed by the
head-camera YOLO detector and SAM is skipped (small parts are below its
reliable scale); detect_object holds that routing. The module constants are
the tuning knobs, grouped by pipeline stage.
"""

import copy
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from builtin_interfaces.msg import Time
from geometry_msgs.msg import (Pose, Point, Quaternion, TransformStamped,
                               Vector3Stamped)
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, DurabilityPolicy

from custom_ros_messages.action import SkillGrasp
from custom_ros_messages.msg import Detection

from ..base import (_Run, GRASP_FRAMES, SLOW_MODE_TIME_SCALE, WORLD_FRAME,
                    GEMINI_TIMEOUT_SEC)
from model_server.model_logging import _log_root
from ..perception_utils import pose_to_matrix, extract_json
from ..grasp_synthesis import METHODS, SynthesisInput, method_name
from ..grasp_synthesis.common import GRIPPER_BASE_TO_CONTACT_M, pose_at


# ---- perception routing -----------------------------------------------------
# The battery-workcell classes of the fine-tuned YOLO-World checkpoint, spelled
# as its labels are ('OrageCover' included); keep equal to
# yolo_server.DEFAULT_QUERIES. On the YOLO path these get a YOLO box and the
# raw box cloud instead of a SAM mask.
BATTERY_OBJECTS = ('Bolt', 'BusBar', 'InteriorScrew', 'Nut', 'OrageCover',
                   'Screw', 'ScrewHole')
_BATTERY_OBJECTS_LC = frozenset(o.lower() for o in BATTERY_OBJECTS)
# True: battery parts take the Gemini + SAM path on the head camera and in the
# visual servo like every other target, which is what the sim grasp tables
# measure (the YOLO checkpoint does not detect the sim's assets); a battery
# class label given as the target is sent to Gemini as its GEMINI_NAMES phrase.
# False: battery parts use the head and hand YOLO detectors. Either way the
# hand YOLO bundle supplies the servo's depth and intrinsics.
BATTERY_USE_GEMINI = True
GEMINI_NAMES = {'bolt': 'bolt', 'busbar': 'bus bar', 'interiorscrew': 'interior screw',
                'nut': 'nut', 'oragecover': 'orange cover', 'screw': 'screw',
                'screwhole': 'screw hole'}

# One box per VISIBLE instance (base.GEMINI_GRASP_PROMPT asks for exactly one),
# so look-alikes such as several grey screws can be told apart: the instance
# closest to the working arm is grasped (_gemini_box_closest_arm). Each box
# still fully encompasses its instance, so it stays a valid SAM prompt.
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

# The visual servo's hand-camera box. The servo only translates (it keeps the
# planned grasp's orientation), so no grasp axis is asked for.
GEMINI_SERVO_PROMPT = (
    'A robot gripper is about to pick up the {obj} in this hand-camera image. '
    'Return ONE JSON object and no prose: {{"box_2d": [y1, x1, y2, x2]}} where '
    'box_2d FULLY ENCOMPASSES the {obj}. Coordinates are integers normalized to '
    '0-1000 with y first. If the {obj} is not visible, return {{}}.'
)


# ---- failure retreat (_retreat_after_failure) -------------------------------
# m: before reporting a failure the hand lifts straight up so it is not left
# parked in the workspace, in slow mode because it starts wherever the failure
# left it, often next to the object. Skipped when 'prep' fails before
# detection (the arm has not moved) and when arm motion is halted.
FAILURE_RETRACT_M = 0.15
# s, full-speed budget, multiplied by SLOW_MODE_TIME_SCALE
FAILURE_RETRACT_SEC = 3.0


# ---- motion and execution ---------------------------------------------------
# Fraction of the gripper's range opened before the approach (1.0 fully open).
# The final close is force-based, so there is no closed fraction.
OPEN_PERCENT = 1.0
# m backed off along the grasp's +Z approach axis for the pre-grasp. A visual
# servo starts from there, so this is also its starting standoff; 10 cm starts
# it close to the object.
APPROACH_DIST = 0.10
# m every grasp is shifted along its own approach axis before execution
# (positive = deeper); the pre-grasp moves with it
GRASP_OFFSET = 0.0
# s before the close: frame_task reports done at IK convergence while the arm
# is still settling by several mm, and closing during that drags the part
PRE_CLOSE_SETTLE_SEC = 1.0
# s after the close (on top of close_gripper's settle), so the grip is at full
# force before e.g. pick_place's lift
POST_CLOSE_SETTLE_SEC = 4.0
# m the contact move stops short of the grasp pose; default of the node's
# `contact_offset` parameter, which the move reads
CONTACT_OFFSET = 0.25 * APPROACH_DIST
# Testing switch: hold at the pre-grasp until Enter is pressed on the skills
# node's terminal (needs interactive stdin; ignores cancel while it waits).
CONFIRM_BEFORE_CONTACT = False

# ---- approach snapshot (APPROACH_IMAGE_ONLY) --------------------------------
# Testing switch: END the skill at the pre-grasp. After the approach the skill
# settles, saves ONE frame from the grasping arm's hand-camera color stream (no
# head-camera fallback), returns the arms to 'prep' and succeeds. No visual
# servo (goal.visual_servo is ignored: it aligns a close that never happens,
# and a servo failure would abort before the image), no contact move, no close
# and no GraspOutcome, so pick_place's inner grasp fails while this is on.
# Images go to <golem_skills>/logs/approach_images/ (host-persistent through
# the core_ws bind mount).
APPROACH_IMAGE_ONLY = False
# s of robot time to wait for a hand-camera frame captured at the pre-grasp before
# saving the latest cached one instead
APPROACH_IMAGE_WAIT_SEC = 3.0

# ---- visual servo (goal.visual_servo) ---------------------------------------
# Closed-loop hand-camera alignment at the pre-grasp (servo()). The hand camera
# looks straight down the approach axis, 4.4 cm ahead of the gripper base, so
# "centered at depth D" means "on the approach ray, D ahead of the camera";
# the finger contact point is 0.0706 m ahead of the camera.
#
# m, camera -> object depth driven to; default of the node's `servo_depth`
# parameter. A real target near where the fingers sit: at 100 mm the object
# ends ~29 mm beyond the fingertips. The servo drives depth TO this value, so
# it is not a band centre: a one-sided band such as (0, 105) would centre the
# setpoint at 52.5 mm and make the servo plunge into the part whenever it
# starts beyond 105 mm. VISUAL_SERVO_RANGE_TOL_M makes the band (99, 101) mm.
VISUAL_SERVO_DEPTH_M = 0.10
# Per-axis convergence tolerances [m], the values real screw-picking runs
# converge with. The arm binds: frame_task settles to ~1.2 mm residual, so a
# correction under ~2 mm does not execute and < 4 mm lateral is unreachable.
#   X (across the jaws), what the grasp depends on: below the arm's ~1.2 mm
#     resolution, so the loop converges when it happens to land in the band;
#     if runs stall on X, 1-2 mm is the honest value against that residual.
#   Y (along the jaw span): the jaws open to 106 mm, so 5 mm still holds the
#     part near the middle of the grip.
#   range: decides whether the jaws close around the part or above it.
# Keep X <= Y: the reverse is backwards for a parallel gripper.
VISUAL_SERVO_X_TOL_M = 0.00025  # +/-0.25 mm -> X band (11.75, 12.25) at the 12 mm target
VISUAL_SERVO_Y_TOL_M = 0.005
VISUAL_SERVO_RANGE_TOL_M = 0.001
# s of robot time for the whole loop (~VISUAL_SERVO_MOVE_SEC per iteration); the
# goal's own timeout still wins
VISUAL_SERVO_TIMEOUT_SEC = 90.0
# s of robot time without a usable hand-camera detection before sight counts as lost
# (yolo_server publishes at ~5 Hz)
VISUAL_SERVO_DETECT_WAIT_SEC = 5.0
# Box source on the hand-camera frame while BATTERY_USE_GEMINI is off: a
# battery part uses the hand YOLO detector, any other object (not a YOLO class)
# Gemini; True sends battery parts to Gemini too. The hand YOLO bundle supplies
# the aligned depth and intrinsics either way (_servo_measure_gemini).
SERVO_USE_GEMINI = False
# Gemini servo only: segment the Gemini box with SAM on the hand frame and
# steer on the masked part (base.bundle_mask_point) instead of the median depth
# of the box's central quarter, which on a bent or elongated part is often
# empty space. A failed SAM call falls back to the box point.
SERVO_USE_SAM = True
# s per servo Gemini query. A Gemini call takes seconds to minutes, far longer
# than a frame_task iteration, so a Gemini servo is slow but general; servo()
# stretches the lost-sight wait to match.
SERVO_GEMINI_TIMEOUT_SEC = 60.0
# m: lost sight is usually geometry (out of frame, occluded by the fingers,
# inside the hand camera's minimum range), so the hand backs straight up
# (pelvis +Z) by this much and keeps servoing
VISUAL_SERVO_LOST_RETRACT_M = 0.05
# m: depth at or below which the object is too close to steer by. Keep clearly
# below 0.0706 m, where the fingers themselves are: a higher threshold reads the
# fingers as "too close" and backing off can never clear it.
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


# ---- reach budgets ----------------------------------------------------------
# servo_frame_to_world budgets, shared with pick_place: a long first IK move and
# many world-frame refinement passes; the iteration-0 unreachable check still
# drops dead candidates quickly
SERVO_DURATION_SEC = 15   # s, first (iteration-0) contact move
# s per approach move, a timeout rather than a trajectory time: generous so a
# long transit is never cut off; a rejected plan returns in well under a second
APPROACH_DURATION_SEC = 180.0
SERVO_MAX_ITER = 12       # world-frame refinement passes per pose
# Track the grasp targets as the pelvis moves: every pre-grasp and grasp is
# snapshotted once in WORLD_FRAME, each take-off, transit and axis-approach
# waypoint is re-resolved into the live pelvis frame before it is commanded,
# and a world-frame correction runs at contact. False drives the raw
# pelvis-frame poses (as when the world TF is down); use it when odometry is
# poor enough that the corrections add error. Needs WORLD_FRAME in the TF tree
# (navigation / FAST-LIO running).
GRASP_WORLD_ANCHOR = True
# Stop a frame_task move early once its streamed feedback stays unchanged for
# this many messages: the arm has settled and the server would only hold steady
# state until its ~5 s hold timeout. Passed to every move the grasp skill
# commands; None disables it.
FRAME_TASK_STABLE_STOP_MSGS = 20
# m, rad: servo_frame_to_world tolerances of the CONTACT move, shared with
# pick_place; real-robot IK plus pelvis drift rarely settles a 6-DOF grasp pose
# tighter than this
SERVO_LIN_TOL = 0.0125
SERVO_ANG_TOL = 0.05
# m, rad: looser, for the PRE-GRASP approach only (move_to_candidate_approach).
# The pre-grasp is a standoff the visual servo refines from, and a tighter
# tolerance rejects reachable candidates as unreachable and burns the pool.
APPROACH_LIN_TOL = 0.025
APPROACH_ANG_TOL = 0.10
# Every candidate is approached in three straight-line, unplanned, slow-mode
# stages (_straight_line_move):
#   1. take off straight up (pelvis +Z), holding the orientation, to
#      APPROACH_CLEARANCE_M above the higher of the hand and the pre-grasp;
#   2. transit to the pre-grasp, position and orientation interpolated;
#   3. in along the grasp's own +Z axis to the contact pose
#      (move_to_candidate), orientation held.
# The visual servo, when requested, replaces stage 3. A failed candidate is
# followed by a planned reset to 'prep' (a staged method: its staging config).
APPROACH_CLEARANCE_M = 0.05
# stage 1-2 segment limits [m, rad] per frame_task goal, so no single goal asks
# for a large reorientation
LINEAR_APPROACH_SEG_M = 0.10
LINEAR_APPROACH_SEG_RAD = 0.35
# s per stage 1-2 segment, already long enough for LINEAR_APPROACH_SEG_M at
# slow-mode speed
LINEAR_APPROACH_SEG_SEC = 10.0
# stage 3 step along the grasp axis [m] and its budget [s]
AXIS_APPROACH_STEP_M = 0.025
AXIS_APPROACH_STEP_SEC = 6.0


# ---- visualization ----------------------------------------------------------
# TF frame (RViz) of the target being driven to
TARGET_FRAME = 'graspgenx_target_frame'
# most candidates drawn on 'graspgen_markers'
KEPT_MARKER_MAX = 20
# candidate state -> marker (r, g, b, a); untried candidates are 'pending'
GRASP_MARKER_COLORS = {
    'pending': (0.5, 0.5, 0.0, 0.4),   # dim yellow
    'trying': (0.0, 0.6, 1.0, 1.0),    # blue: pre-grasp approach in progress
    'failed': (1.0, 0.0, 0.0, 0.8),    # red: pre-grasp unreachable
    'chosen': (0.0, 1.0, 0.0, 1.0),    # green: committed grasp
}


@dataclass
class GraspOutcome:
    """
    Carry what pick_place needs from an executed grasp.

    Handed back on the node's `_last_grasp_outcome` attribute, because the
    SkillGrasp result carries only success and message.
    """

    # arm that executed it (resolved, so an auto-select caller learns it)
    arm: str
    # executed contact pose of the GraspGenX gripper-base frame, in `frame`
    pose: Pose
    # WORLD_FRAME when the world TF was available, else 'pelvis'
    frame: str
    # confidence of the executed grasp
    score: float


@dataclass
class _Targets:
    """Hold every candidate's executable poses, frozen once by _snapshot_targets."""

    # the candidates' planning frame ('pelvis')
    frame: str
    # GRASP_OFFSET-shifted grasp poses, planning frame
    grasps_p: list
    # pre-grasp standoffs (APPROACH_DIST back), planning frame
    approaches_p: list
    # world-frame snapshots of grasps_p (entries None when TF was down)
    grasps_w: list
    # world-frame snapshots of approaches_p
    approaches_w: list
    # every snapshot resolved: drive with drift compensation
    have_world: bool
    # object centroid as a world Pose (None without TF)
    centroid_w: object


class GraspSkill:
    """Provide /skill/grasp's execute callback and its pipeline to SkillsNode."""

    def _exec_grasp(self, gh):
        """
        Grasp goal.target_object and leave it held (no lift); return the result.

        prep -> detect_object -> segment_object -> get_grasp_candidates ->
        move_to_candidate_approach -> contact (or visual servo) + close. The
        method is the `grasp_method` parameter, or `top_down` when the goal
        sets it (_resolve_grasp_method).

        Optional goal fields (SkillGrasp.action; zero / false = off):
          top_down: use the top_down synthesis method for this goal;
          min_downward_pitch_deg: keep only grasps approaching at least this
            far below horizontal;
          visual_servo: replace the contact move with servo().

        With APPROACH_IMAGE_ONLY on, the run ends at the pre-grasp (see the
        constant) and sets no GraspOutcome.

        `self._last_grasp_outcome` is cleared at the start and set only by a
        successful grasp, so a failed grasp never leaks a stale outcome to
        pick_place; one grasp goal runs at a time. TARGET_FRAME is broadcast
        while the goal runs and dropped when it ends, however it ends.
        """
        try:
            return self._grasp(gh)
        finally:
            self.drop_tf(TARGET_FRAME)

    def _grasp(self, gh):
        """Run the grasp goal's body (see _exec_grasp)."""
        goal = gh.request
        run = _Run(self, gh, SkillGrasp, 'grasp')
        self._last_grasp_outcome = None
        arm = self._validated_arm(goal)
        if arm is None and goal.arm.strip().lower() not in ('', 'none'):
            return run.abort(f'invalid arm {goal.arm!r}')
        method, err = self._resolve_grasp_method(goal)
        if err:
            return run.abort(err)
        obj = _perception_name(goal.target_object)
        self.get_logger().info(
            f'grasp: {obj!r} with synthesis method {method.name!r}'
            f'{" + visual servo" if goal.visual_servo else ""}')

        if not run.phase('detect_object', 0.0):
            return run.result
        # Arms to 'prep' first, out of the head camera's view (an arm across the
        # object corrupts the box, mask and cloud); planned, because the arms
        # start wherever the last skill left them. The only failure without a
        # retreat: the skill has not moved the arm yet.
        if not self.goto_named_config('prep', plan=True, slow_mode=True,
                                      duration_sec=30.0, result_timeout=90.0,
                                      outer_gh=gh):
            return run.abort("move to 'prep' before detection failed")

        def fail(message):
            """Lift the hand clear of the workspace, then abort with `message`."""
            self._retreat_after_failure(arm, gh)
            return run.abort(message)

        box, use_sam = self.detect_object(obj, run, gh, arm)
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return fail('detection canceled or timed out')
        obj_cloud, scene, err = self.segment_object(obj, box, use_sam, gh)
        if err:
            return fail(err)
        centroid_p = obj_cloud.mean(axis=0)
        # no arm requested: the arm whose gripper is closest to the object
        if arm is None:
            arm = self._closest_arm(centroid_p)
        self._note_run_arm(arm)

        # Opened after 'prep' and before staging, so the arm travels to the
        # staging pose and everything after it with the gripper open.
        if not self.open_gripper(arm, OPEN_PERCENT):
            return fail('gripper pre-open failed')

        # A staged method starts from the arm's joint-space staging config: a
        # repeatable posture above the workspace for a steep top-down approach,
        # reached without an IK solve, with the other arm at 'prep'. Other
        # methods start from 'prep'.
        stage_cfg = 'staging_right' if arm == 'right' else 'staging_left'
        if method.staged:
            if not self.goto_named_config(stage_cfg, plan=True, slow_mode=True,
                                          duration_sec=30.0, result_timeout=90.0,
                                          outer_gh=gh):
                return fail(f"move to {stage_cfg!r} staging config failed")

        if not run.phase('approach_grasp', 0.4):
            return run.result
        cands, err = self.get_grasp_candidates(method, goal, obj, arm,
                                               obj_cloud, scene)
        if err:
            return fail(err)

        targets = self._snapshot_targets(cands, centroid_p)
        # the run_telemetry target is the object centroid in the world frame
        if targets.centroid_w is not None:
            self._note_run_target(targets.centroid_w)
        reset_cfg = stage_cfg if method.staged else 'prep'
        idx, err = self.move_to_candidate_approach(run, gh, obj, arm, cands,
                                                   targets, reset_cfg)
        if err:
            # no retreat: a failed walk ends with the arms reset to reset_cfg,
            # and a halted arm must not be moved at all
            return run.abort(err)
        # The snapshot mode pre-empts the visual servo and the contact confirm:
        # both set up a close that never happens here, and a servo failure
        # would abort before the image is taken.
        if APPROACH_IMAGE_ONLY:
            if not run.phase('save_image', 0.75):
                return run.result
            # the arm is still moving by several mm when frame_task reports
            # done, which would blur the frame
            self.get_clock().sleep_for(
                RclpyDuration(seconds=PRE_CLOSE_SETTLE_SEC))
            path = self._save_approach_image(arm, obj)
            if path is None:
                return fail(f'no {arm} hand-camera frame to save at the '
                            f'pre-grasp for {obj!r}')
            if not run.phase('return_to_prep', 0.9):
                return run.result
            if not self.goto_named_config('prep', plan=True, slow_mode=True,
                                          duration_sec=30.0, result_timeout=90.0,
                                          outer_gh=gh):
                return fail(f"saved {path} but the return to 'prep' failed")
            return run.succeed(
                f'approached {obj!r} and saved {path}; no grasp executed '
                f'(APPROACH_IMAGE_ONLY)')

        # On success the servo replaces the contact move: the gripper closes
        # where the loop parked it.
        servo_pose_p = None
        if goal.visual_servo:
            servo_pose_p, err = self.servo(run, gh, arm, obj, targets, idx)
            if err:
                # a cancel or timeout aborts; a non-convergence closes anyway
                # where the servo left the arm (best-effort)
                if gh.is_cancel_requested or run.remaining() <= 0.0:
                    return fail(err)
                self.get_logger().warn(
                    f'grasp: visual servo did not converge ({err}) — closing '
                    'anyway at the current pose (best-effort)')
                servo_pose_p = self._frame_pose_in_pelvis(GRASP_FRAMES[arm])
        if CONFIRM_BEFORE_CONTACT:
            self.get_logger().info(
                f'grasp: holding at pre-grasp for {obj!r} — press Enter on the '
                f'skills node terminal to execute the grasp')
            input(f'at pre-grasp for {obj!r} — press Enter to execute the grasp ')

        if not run.phase('close_gripper', 0.75):
            return run.result
        if servo_pose_p is not None:
            contact_p = servo_pose_p
            contact_w = (self._transform_pose(contact_p, 'pelvis', WORLD_FRAME)
                         if targets.have_world else None)
        else:
            contact_p, contact_w = self.move_to_candidate(gh, arm, targets, idx)
            halt = self.arm_halt_reason()
            if halt is not None:
                return run.abort(f'arm halted during the contact approach: {halt}')
        self.get_clock().sleep_for(RclpyDuration(seconds=PRE_CLOSE_SETTLE_SEC))
        if not self.close_gripper(arm):
            return fail('gripper close failed')
        self.get_clock().sleep_for(RclpyDuration(seconds=POST_CLOSE_SETTLE_SEC))

        # the CONTACT pose, where the hand actually is, in the world snapshot's
        # frame when there is one
        if targets.have_world and targets.centroid_w is not None:
            out_pose, out_frame = contact_w, WORLD_FRAME
        else:
            out_pose, out_frame = contact_p, 'pelvis'
        self._last_grasp_outcome = GraspOutcome(
            arm=arm, pose=out_pose, frame=out_frame, score=cands.scores[idx])

        return run.succeed(
            f'grasped {obj!r} with {method.name} candidate '
            f'{cands.labels[idx]} (score {cands.scores[idx]:.2f})')

    def _resolve_grasp_method(self, goal):
        """Return (GraspMethod, None) for the goal, or (None, reason) for an unknown name."""
        name = method_name(goal.top_down,
                           self.get_parameter('grasp_method').value)
        method = METHODS.get(name)
        if method is None:
            return None, (f'unknown grasp_method {name!r}; expected one of '
                          f'{sorted(METHODS)}')
        return method, None

    def detect_object(self, obj, run, gh, arm=None):
        """
        Box `obj` in the head camera; return (pixel-xyxy box or None, use_sam).

        A Gemini box refined by SAM, or, for a battery part on the YOLO path
        (_yolo_routed), a YOLO box whose raw rectangle becomes the cloud.
        Among look-alike instances it picks the one closest to `arm`'s gripper
        (the robot when `arm` is None): _yolo_box on the YOLO path,
        _gemini_box_closest_arm on the Gemini path.
        """
        is_small = _yolo_routed(obj)
        box = (self._yolo_box(obj, arm) if is_small
               else self._gemini_box_closest_arm(obj, run, gh, arm))
        return box, not is_small

    def _gemini_box_closest_arm(self, obj, run, gh, arm):
        """
        Return the Gemini box (pixel xyxy) of the `obj` instance closest to `arm`'s gripper.

        With no arm, the instance closest to the pelvis origin. Falls back to
        base._gemini_box (the single-box query) when the multi-instance query
        returns zero or one box, or no box has usable depth to range.
        """
        boxes = self._gemini_boxes(obj, run, gh)
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return None
        if len(boxes) <= 1:
            return boxes[0] if boxes else self._gemini_box(obj, run, gh)
        ref = None
        if arm in GRASP_FRAMES:
            o = self._transform_pose(pose_at((0.0, 0.0, 0.0)),
                                     GRASP_FRAMES[arm], 'pelvis')
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
        data = extract_json(txt)
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = [data]
        else:
            return []
        info = self.latest_caminfo()
        if info is None or not info.width or not info.height:
            return []
        w, h = info.width, info.height
        boxes = []
        for entry in entries:
            if not isinstance(entry, dict) or 'box_2d' not in entry:
                continue
            try:
                y1, x1, y2, x2 = (float(v) for v in entry['box_2d'])
            except (ValueError, TypeError):
                continue
            px1, px2 = sorted((x1 / 1000.0 * w, x2 / 1000.0 * w))
            py1, py2 = sorted((y1 / 1000.0 * h, y2 / 1000.0 * h))
            boxes.append([px1, py1, px2, py2])
        return boxes

    def _box_point_pelvis(self, box):
        """Return a pixel-xyxy `box`'s pelvis-frame point (_detection_point), or None."""
        det = Detection()
        det.bbox_min = Point(x=float(box[0]), y=float(box[1]), z=0.0)
        det.bbox_max = Point(x=float(box[2]), y=float(box[3]), z=0.0)
        return self._detection_point(det, target_frame='pelvis')

    def segment_object(self, obj, box, use_sam, gh):
        """
        Lift a box to point clouds; return (obj_cloud, scene_cloud, None) or (None, None, reason).

        With `use_sam`, SAM gets the object name plus the box (when there is
        one) and only masked pixels are back-projected; without it the box
        RECTANGLE is back-projected and must be present. The whole-frame scene
        cloud is the obstacle context for graspgen's collision filter (None
        skips the filter).
        """
        if use_sam:
            mask = self.segment(text=obj, positive_boxes=box, outer_gh=gh)
            if mask is None:
                return None, None, f'no mask for {obj!r}'
            obj_cloud = self.mask_to_cloud(mask, target_frame='pelvis')
            reason = f'{obj!r} mask produced no usable cloud'
        else:
            if box is None:
                return None, None, f'no bounding box for {obj!r} (SAM disabled)'
            obj_cloud = self.box_to_cloud(box, target_frame='pelvis')
            reason = f'{obj!r} bounding box produced no usable cloud'
        if obj_cloud is None:
            return None, None, reason
        return obj_cloud, self.scene_to_cloud(target_frame='pelvis'), None

    def get_grasp_candidates(self, method, goal, obj, arm, obj_cloud, scene):
        """Run `method` on the segmented object; return (Candidates, None) or (None, reason)."""
        inp = SynthesisInput(
            obj=obj, arm=arm, cloud=obj_cloud, scene=scene,
            min_pitch_deg=float(goal.min_downward_pitch_deg or 0.0),
            plan_grasp=self.plan_grasp, log=self.get_logger())
        cands, err = method.synthesize(inp)
        if err is None and not cands.grasps:
            err = f'{method.name} produced no grasp for {obj!r}'
        return (None, err) if err else (cands, None)

    def _snapshot_targets(self, cands, centroid_p):
        """
        Freeze every candidate's grasp and pre-grasp (and the centroid) into the world frame.

        Done once, from the current pelvis frame, so world-anchored targets
        stay put as the pelvis drifts. When the world TF is unavailable or
        GRASP_WORLD_ANCHOR is off, `have_world` is False and the raw pelvis
        poses are driven without drift compensation.
        """
        grasps_p = [get_approach_pose(g.pose, approach_dist=GRASP_OFFSET)
                    for g in cands.grasps]
        approaches_p = [get_approach_pose(g, approach_dist=-APPROACH_DIST)
                        for g in grasps_p]
        if not GRASP_WORLD_ANCHOR:
            self.get_logger().info(
                'grasp: GRASP_WORLD_ANCHOR off; driving raw pelvis poses')
            return _Targets(frame=cands.grasps[0].header.frame_id,
                            grasps_p=grasps_p, approaches_p=approaches_p,
                            grasps_w=[None] * len(grasps_p),
                            approaches_w=[None] * len(approaches_p),
                            have_world=False, centroid_w=None)
        grasps_w = [self._transform_pose(g, 'pelvis', WORLD_FRAME)
                    for g in grasps_p]
        approaches_w = [self._transform_pose(a, 'pelvis', WORLD_FRAME)
                        for a in approaches_p]
        have_world = all(p is not None for p in grasps_w + approaches_w)
        if not have_world:
            self.get_logger().warn(
                f'grasp: {WORLD_FRAME} TF unavailable; pelvis-drift servoing OFF, '
                'driving raw pelvis poses (start navigation/FAST-LIO to enable)')
        centroid_w = (self._transform_pose(pose_at(centroid_p), 'pelvis',
                                           WORLD_FRAME) if have_world else None)
        return _Targets(frame=cands.grasps[0].header.frame_id,
                        grasps_p=grasps_p, approaches_p=approaches_p,
                        grasps_w=grasps_w, approaches_w=approaches_w,
                        have_world=have_world, centroid_w=centroid_w)

    def _straight_line_move(self, frame, target, run, gh, step_m, step_rad,
                            seg_sec, lin_tol, ang_tol, label, slow_mode=True,
                            target_w=None):
        """
        Move `frame` to `target` (pelvis frame) along a straight line; return reached.

        Direct (unplanned) frame_task goals of at most `step_m` and
        `step_rad` each, orientation slerped. With `target_w` (the same target
        in WORLD_FRAME) the line is laid out in the world frame and each
        waypoint is re-resolved into the live pelvis frame just before it is
        sent; without a world TF at the start the pelvis-frame line is driven.
        Returns False on the first missed segment, cancel, timeout or halt.
        """
        cur = self._frame_pose_in_pelvis(frame)
        if cur is None:
            return False
        if target_w is not None:
            cur_w = self._transform_pose(cur, 'pelvis', WORLD_FRAME)
            if cur_w is not None:
                cur, target = cur_w, target_w
            else:
                target_w = None
        p0 = _pose_xyz(cur)
        p1 = _pose_xyz(target)
        q0 = _pose_quat(cur)
        q1 = _pose_quat(target)
        dist = float(np.linalg.norm(p1 - p0))
        angle = _quat_angle(q0, q1)
        n = max(1, int(math.ceil(max(dist / step_m, angle / step_rad))))
        self.get_logger().info(
            f'grasp: {label}: {dist * 100:.1f}cm / {math.degrees(angle):.0f}deg '
            f'in {n} unplanned{" slow" if slow_mode else ""} step(s)'
            f'{f", tracking {WORLD_FRAME}" if target_w is not None else ""}')
        for k in range(1, n + 1):
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return False
            t = float(k) / n
            pk = p0 + t * (p1 - p0)
            qk = _quat_slerp(q0, q1, t)
            wp = Pose()
            wp.position = Point(x=float(pk[0]), y=float(pk[1]), z=float(pk[2]))
            wp.orientation = Quaternion(x=float(qk[0]), y=float(qk[1]),
                                        z=float(qk[2]), w=float(qk[3]))
            if target_w is not None:
                wp = self._live_pelvis_pose(wp)
                if wp is None:
                    return False
            if not self.move_frame_to(
                    frame, wp, outer_gh=gh, duration_sec=seg_sec,
                    do_plan=False, slow_mode=slow_mode,
                    lin_tol=lin_tol, ang_tol=ang_tol,
                    stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                    label=f'{label} {k}/{n}'):
                return False
        return True

    def _approach_pregrasp(self, frame, pregrasp, run, gh, pregrasp_w=None):
        """
        Run approach stages 1-2: take off straight up, then transit to the pre-grasp.

        The take-off holds the orientation and rises (pelvis +Z) to
        APPROACH_CLEARANCE_M above the higher of the hand and the pre-grasp.
        With `pregrasp_w` (world-anchored targets) both legs track the pelvis.
        Returns whether the pre-grasp was reached.
        """
        cur = self._frame_pose_in_pelvis(frame)
        if cur is None:
            return False
        lift_z = max(cur.position.z, pregrasp.position.z) + APPROACH_CLEARANCE_M
        if lift_z - cur.position.z > 1e-3:
            up = copy.deepcopy(cur)
            up.position.z = lift_z
            up_w = (self._transform_pose(up, 'pelvis', WORLD_FRAME)
                    if pregrasp_w is not None else None)
            if not self._straight_line_move(
                    frame, up, run, gh, LINEAR_APPROACH_SEG_M,
                    LINEAR_APPROACH_SEG_RAD, LINEAR_APPROACH_SEG_SEC,
                    APPROACH_LIN_TOL, APPROACH_ANG_TOL, 'take-off (pelvis +Z)',
                    target_w=up_w):
                return False
        return self._straight_line_move(
            frame, pregrasp, run, gh, LINEAR_APPROACH_SEG_M,
            LINEAR_APPROACH_SEG_RAD, LINEAR_APPROACH_SEG_SEC,
            APPROACH_LIN_TOL, APPROACH_ANG_TOL, 'transit to pre-grasp',
            target_w=pregrasp_w)

    def _live_pelvis_pose(self, pose_w):
        """Return a WORLD_FRAME pose re-resolved into the live pelvis frame, or None."""
        return self._transform_pose(pose_w, WORLD_FRAME, 'pelvis')

    def move_to_candidate_approach(self, run, gh, obj, arm, cands, targets,
                                   reset_cfg):
        """
        Reach the first reachable candidate's pre-grasp; return (index, None) or (None, reason).

        Walks the candidates in the method's order (_approach_pregrasp). After
        every failed candidate the arms return to the named config
        `reset_cfg` by a PLANNED joint-space move, because a failed Cartesian
        approach can leave a joint pinned at its limit, from which no further
        Cartesian move (the next take-off included) makes progress. Fails on
        cancel or timeout, an arm halt, a failed reset, or when no candidate is
        reachable. Each candidate's state (trying, failed, chosen) is
        republished on 'graspgen_markers'.
        """
        order = range(len(cands.grasps))
        width_mm = cands.gripper_width * 1000.0
        status = {}

        def mark(i, state):
            """Set candidate `i`'s marker state and republish the markers."""
            status[i] = state
            self._publish_grasp_markers(targets.frame, targets.grasps_p,
                                        cands.scores, status)

        for i in order:
            mark(i, 'trying')
            self._publish_target_tf(targets, targets.approaches_w[i],
                                    targets.approaches_p[i])
            self.get_logger().info(
                f'grasp {i} ({cands.labels[i]}) for {obj!r}: tier '
                f'{cands.tiers[i]}, score {cands.scores[i]:.2f}, '
                f'width {width_mm:.1f}mm')
            if self._approach_pregrasp(GRASP_FRAMES[arm], targets.approaches_p[i],
                                       run, gh, pregrasp_w=targets.approaches_w[i]):
                mark(i, 'chosen')
                return i, None
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return None, 'canceled or timed out during approach'
            halt = self.arm_halt_reason()
            if halt is not None:
                mark(i, 'failed')
                return None, f'arm halted during the approach: {halt}'
            mark(i, 'failed')
            self.get_logger().warn(
                f'grasp {i} pre-grasp unreachable; resetting to {reset_cfg!r}')
            if not self.goto_named_config(reset_cfg, plan=True, slow_mode=True,
                                          duration_sec=30.0, result_timeout=90.0,
                                          outer_gh=gh):
                halt = self.arm_halt_reason()
                return None, (f'arm halted during the reset to {reset_cfg!r}: {halt}'
                              if halt is not None else
                              f'reset to {reset_cfg!r} after grasp {i} failed')
        return None, (f'no reachable grasp for {obj!r} '
                      f'(tried {len(order)} of {cands.total})')

    def move_to_candidate(self, gh, arm, targets, idx):
        """
        Run approach stage 3: go in along the grasp axis to the contact target.

        The contact target is the committed grasp backed off by the node's
        `contact_offset` parameter, reached in AXIS_APPROACH_STEP_M straight
        steps with the grasp orientation held, slow and unplanned (planning
        would fight the approach into the object). Best-effort: the advance
        stops at the first step that misses its tolerance and the gripper
        closes there. With world anchoring each step is laid out along the
        world-frame grasp axis and re-resolved into the live pelvis frame, and
        a final world-frame correction removes the remaining pelvis drift.
        Returns (contact_p, contact_w) for the GraspOutcome.
        """
        frame = GRASP_FRAMES[arm]
        stop_short = float(self.get_parameter('contact_offset').value)
        contact_p = get_approach_pose(targets.grasps_p[idx],
                                      approach_dist=-stop_short)
        contact_w = (get_approach_pose(targets.grasps_w[idx],
                                       approach_dist=-stop_short)
                     if targets.have_world else None)
        self._publish_target_tf(targets, contact_w, contact_p)
        start = targets.approaches_p[idx]
        depth = float(np.linalg.norm(_pose_xyz(contact_p) - _pose_xyz(start)))
        if depth < 0.01:
            self.get_logger().warn(
                f'grasp: axis approach is only {depth * 100:.1f}cm; the '
                'gripper closes without approaching along the grasp axis')
        n = max(1, int(math.ceil(depth / AXIS_APPROACH_STEP_M)))
        for k in range(1, n + 1):
            if gh.is_cancel_requested:
                break
            step = get_approach_pose(start, approach_dist=depth * k / n)
            if targets.have_world:
                step = self._live_pelvis_pose(get_approach_pose(
                    targets.approaches_w[idx], approach_dist=depth * k / n)) or step
            if not self.move_frame_to(
                    frame, step, outer_gh=gh, duration_sec=AXIS_APPROACH_STEP_SEC,
                    do_plan=False, slow_mode=True,
                    lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL,
                    stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                    label=f'in along the grasp axis {k}/{n}'):
                self.get_logger().warn(
                    f'grasp: axis approach stopped at step {k}/{n} '
                    f'({depth * (k - 1) / n * 100:.1f} of {depth * 100:.1f}cm); '
                    'closing here')
                return contact_p, contact_w
        if targets.have_world and self.arm_halt_reason() is None:
            self.servo_frame_to_world(
                frame, contact_w, contact_p, outer_gh=gh,
                duration_sec=SERVO_DURATION_SEC * SLOW_MODE_TIME_SCALE,
                max_iter=SERVO_MAX_ITER,
                lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL, do_plan=False,
                slow_mode=True, stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                label='world-frame contact correction')
        return contact_p, contact_w

    def servo(self, run, gh, arm, obj, targets, idx):
        """
        Align the gripper on `obj` with the hand camera at the pre-grasp (goal.visual_servo).

        Drives GRASP_FRAMES[arm] in PURE TRANSLATION until `obj` sits at
        (VISUAL_SERVO_TARGET_XY_M, `servo_depth`) in the hand camera's optical
        frame; the orientation stays the planned grasp's.

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
        frame = GRASP_FRAMES[arm]
        planned_q = targets.grasps_p[idx].orientation
        target = np.array([VISUAL_SERVO_TARGET_XY_M[0],
                           VISUAL_SERVO_TARGET_XY_M[1],
                           float(self.get_parameter('servo_depth').value)])
        use_gemini = _servo_uses_gemini(obj)
        # one Gemini query outlasts the YOLO lost-sight window, so in Gemini
        # mode sight is lost only after ~1.5 queries
        detect_wait = (1.5 * SERVO_GEMINI_TIMEOUT_SEC if use_gemini
                       else VISUAL_SERVO_DETECT_WAIT_SEC)
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
            f'(tol {VISUAL_SERVO_X_TOL_M * 1000:.0f}mm across the jaws / '
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
                    f'{VISUAL_SERVO_X_TOL_M * 1000:.0f}/'
                    f'{VISUAL_SERVO_Y_TOL_M * 1000:.0f}mm X/Y)')
            if use_gemini:
                p_cam, cam_frame, stamp, bundle = \
                    self._servo_measure_gemini(arm, obj, after, gh)
            else:
                p_cam, cam_frame, stamp, bundle = \
                    self._servo_measure(arm, obj, after)
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
            # Lost (nothing usable for detect_wait) and too close (inside
            # MIN_RANGE for STRIKES frames in a row) are both answered by
            # backing straight up and looking again.
            if p_cam is not None and float(p_cam[2]) < VISUAL_SERVO_MIN_RANGE_M:
                too_close_hits += 1
            else:
                too_close_hits = 0
            lost = (p_cam is None
                    and self.robot_now() - last_seen > detect_wait)
            too_close = too_close_hits >= VISUAL_SERVO_TOO_CLOSE_STRIKES
            if lost or too_close:
                why = (f'lost sight of {obj!r} in the {arm} hand camera for '
                       f'{detect_wait:.0f}s' if lost else
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
                f'err X {err[0] * 1000:+.1f}mm (tol {VISUAL_SERVO_X_TOL_M * 1000:.0f}) '
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
                        f'{ax} {v * 1000:+.1f}mm (tol {t * 1000:.0f})'
                        for ax, v, t in (('X', err[0], VISUAL_SERVO_X_TOL_M),
                                         ('Y', err[1], VISUAL_SERVO_Y_TOL_M),
                                         ('range', err[2], VISUAL_SERVO_RANGE_TOL_M))
                        if abs(v) > t)
                    return None, (
                        f'visual servo stalled on {obj!r}: no improvement over '
                        f'{stalled} iterations, still out of tolerance on {out} — '
                        'the tolerance is below what this arm can resolve, or the '
                        'setpoint is off')
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
            self.publish_tf(_approach_target_tf(
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

    def _servo_measure(self, arm, obj, after):
        """
        Measure `obj` in `arm`'s hand camera via YOLO; return (point, frame_id, stamp, bundle).

        A tuple of Nones when there is no usable detection. Only a bundle
        stamped strictly after `after` counts: `after` is the stamp of the
        frame last USED, not a clock instant, so a frame's correction is never
        applied twice while frames from slow inference stay usable (requiring
        frames captured after the move starves the loop). Among detections of
        the class it takes the one closest to the boresight, so the loop stays
        on its instance.
        """
        bundle = self.latest_arm_detections(arm)
        if bundle is None or not bundle.detections:
            return None, None, None, None
        stamp = _stamp_tuple(bundle.rgb_image.header.stamp)
        if stamp <= after:
            return None, None, None, None
        key = obj.strip().lower()
        best_p, best_lat = None, None
        for d in bundle.detections:
            if d.cls.strip().lower() != key:
                continue
            p = self.bundle_detection_point(bundle, d,
                                            min_depth=VISUAL_SERVO_MIN_DEPTH_M)
            if p is None:
                continue
            lat = math.hypot(float(p[0]), float(p[1]))
            if best_lat is None or lat < best_lat:
                best_p, best_lat = p, lat
        if best_p is None:
            return None, None, None, None
        return best_p, self.bundle_camera_frame(bundle), stamp, bundle

    def _servo_measure_gemini(self, arm, obj, after, gh):
        """
        Measure `obj` in `arm`'s hand camera from a Gemini box: (point, frame, stamp, bundle).

        For any object Gemini can name, not only a YOLO class. The arm's YOLO
        DetectionBundle is still used, for the rgb frame, aligned depth and
        camera_info yolo_server attaches to every bundle. Gemini boxes that rgb
        frame; with SERVO_USE_SAM, SAM segments the box and bundle_mask_point
        lifts the mask to an on-object point, otherwise (or when SAM fails) the
        box goes through bundle_detection_point. A tuple of Nones when nothing
        is usable; only a bundle stamped strictly after `after` counts. A
        Gemini call takes seconds to minutes (servo() stretches its lost-sight
        wait to match).
        """
        bundle = self.latest_arm_detections(arm)
        if bundle is None or not bundle.rgb_image.data:
            return None, None, None, None
        stamp = _stamp_tuple(bundle.rgb_image.header.stamp)
        if stamp <= after:
            return None, None, None, None
        box = self._gemini_box_on_image(obj, bundle.rgb_image,
                                        bundle.camera_info, gh)
        if box is None:
            return None, None, None, None
        p = None
        if SERVO_USE_SAM:
            mask = self.segment(text=obj, positive_boxes=box,
                                image=bundle.rgb_image, outer_gh=gh)
            p = self.bundle_mask_point(bundle, mask,
                                       min_depth=VISUAL_SERVO_MIN_DEPTH_M)
            if p is None:
                self.get_logger().warn(
                    f'grasp: visual servo SAM mask for {obj!r} gave no point; '
                    'using the box centre')
        if p is None:
            det = Detection()
            det.cls = obj
            det.bbox_min = Point(x=float(box[0]), y=float(box[1]), z=0.0)
            det.bbox_max = Point(x=float(box[2]), y=float(box[3]), z=0.0)
            p = self.bundle_detection_point(bundle, det,
                                            min_depth=VISUAL_SERVO_MIN_DEPTH_M)
        if p is None:
            return None, None, None, None
        return p, self.bundle_camera_frame(bundle), stamp, bundle

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
        data = extract_json(txt)
        if isinstance(data, list) and data:
            entry = data[0]
        elif isinstance(data, dict):
            entry = data
        else:
            return None
        if not isinstance(entry, dict) or 'box_2d' not in entry:
            return None
        w, h = info.width, info.height
        try:
            y1, x1, y2, x2 = (float(v) for v in entry['box_2d'])
        except (ValueError, TypeError):
            return None
        px1, px2 = sorted((x1 / 1000.0 * w, x2 / 1000.0 * w))
        py1, py2 = sorted((y1 / 1000.0 * h, y2 / 1000.0 * h))
        return [px1, py1, px2, py2]

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
        self.servo_move_residual_pub.publish(
            _vec3_stamped(now, 'pelvis', residual))

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
                    cv2.arrowedLine(img, obj_px, nxt_px, cmd_color, 2,
                                    tipLength=0.3)
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
                        (10, 28 + 30 * 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cmd_color, 2)

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

    def _save_approach_image(self, arm, obj):
        """
        Save `arm`'s hand-camera view of `obj` at the pre-grasp (APPROACH_IMAGE_ONLY).

        Saves the color stream's next frame (only frames captured from now on,
        so a mid-move frame is never saved); without one within
        APPROACH_IMAGE_WAIT_SEC, the latest cached hand frame, with a warning.
        There is no head-camera fallback: the point is the hand view. Returns
        the absolute path written, or None when no hand frame could be decoded
        or written.
        """
        after = _stamp_tuple(self.get_clock().now().to_msg())
        deadline = self.robot_now() + APPROACH_IMAGE_WAIT_SEC
        msg = None
        while self.robot_now() < deadline:
            fresh = self.latest_arm_image(arm)
            if (fresh is not None
                    and _stamp_tuple(fresh.header.stamp) > after):
                msg = fresh
                break
            self.get_clock().sleep_for(RclpyDuration(seconds=0.1))
        if msg is None:
            msg = self.latest_arm_image(arm)
            if msg is not None:
                self.get_logger().warn(
                    f'grasp: no {arm} hand-camera frame newer than the '
                    f'pre-grasp arrival within {APPROACH_IMAGE_WAIT_SEC:.0f}s '
                    f'— saving the latest cached hand frame')
        if msg is None or not msg.data:
            return None
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            return None
        slug = re.sub(r'[^A-Za-z0-9]+', '_', obj).strip('_').lower() or 'object'
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        path = os.path.join(_log_root('golem_skills', __file__),
                            'approach_images', f'{stamp}_{slug}_{arm}.png')
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not cv2.imwrite(path, img):
                raise OSError('cv2.imwrite returned False')
        except OSError as e:
            self.get_logger().warn(f'grasp: could not write {path}: {e}')
            return None
        self.get_logger().info(
            f'grasp: saved {arm} hand-camera approach image for {obj!r} '
            f'to {path}')
        return path

    def _retreat_after_failure(self, arm, gh):
        """
        Lift `arm`'s hand FAILURE_RETRACT_M straight up (pelvis +Z) before a failure is reported.

        Shared with pick_place. Keeps the current orientation (from TF):
        re-orienting a stuck arm makes a bad situation worse. Best-effort: the
        skill is already failing and the retreat never replaces its reason.
        Skipped while arm motion is halted.
        """
        if arm is None:
            return
        halt = self.arm_halt_reason()
        if halt is not None:
            self.get_logger().warn(f'retreat: skipped, arm halted: {halt}')
            return
        frame = GRASP_FRAMES[arm]
        cur = self._frame_pose_in_pelvis(frame)
        if cur is None:
            self.get_logger().warn(
                f'retreat: TF {frame!r} -> pelvis unavailable; leaving the arm in place')
            return
        up = copy.deepcopy(cur)
        up.position.z += FAILURE_RETRACT_M
        self.get_logger().info(
            f'retreat: lifting {frame} {FAILURE_RETRACT_M * 100:.0f}cm (pelvis +Z) '
            'before reporting the failure')
        if not self.move_frame_to(frame, up, outer_gh=gh,
                                  duration_sec=FAILURE_RETRACT_SEC * SLOW_MODE_TIME_SCALE,
                                  do_plan=False, slow_mode=True,
                                  stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                                  label='failure retreat lift'):
            self.get_logger().warn('retreat: lift incomplete')

    def _frame_pose_in_pelvis(self, frame):
        """Return the current pose of URDF `frame` in the pelvis frame, or None."""
        return self._transform_pose(pose_at((0.0, 0.0, 0.0)), frame, 'pelvis')

    def _rotate_into_pelvis(self, vec, src_frame):
        """Rotate the 3-vector `vec` from `src_frame` into the pelvis frame, or return None."""
        o = self._transform_pose(pose_at((0.0, 0.0, 0.0)), src_frame, 'pelvis')
        t = self._transform_pose(pose_at(vec), src_frame, 'pelvis')
        if o is None or t is None:
            return None
        return np.array([t.position.x - o.position.x,
                         t.position.y - o.position.y,
                         t.position.z - o.position.z])

    def _publish_target_tf(self, targets, pose_w, pose_p):
        """Broadcast TARGET_FRAME at `pose_w` when drift-compensating, else at `pose_p`."""
        parent, pose = ((WORLD_FRAME, pose_w) if targets.have_world
                        else (targets.frame, pose_p))
        self.publish_tf(_approach_target_tf(
            parent, pose, self.get_clock().now().to_msg()))

    def _closest_arm(self, centroid_p):
        """
        Return the arm whose gripper is nearest the object centroid (pelvis xyz).

        Uses live TF; without both hand TFs, the side of the pelvis midline the
        object is on (+Y = left).
        """
        dists = {}
        for arm, frame in GRASP_FRAMES.items():
            p = self._transform_pose(pose_at(centroid_p), 'pelvis', frame)
            if p is not None:
                dists[arm] = math.sqrt(p.position.x ** 2 + p.position.y ** 2
                                       + p.position.z ** 2)
        if len(dists) == len(GRASP_FRAMES):
            arm = min(dists, key=dists.get)
            self.get_logger().info(
                'grasp: auto-selected %s arm (hand-to-object %s)' % (
                    arm, ', '.join(f'{a} {d:.3f}m' for a, d in sorted(dists.items()))))
        else:
            arm = 'left' if centroid_p[1] > 0.0 else 'right'
            self.get_logger().warn(
                f'grasp: hand TF unavailable for auto arm choice; picked {arm} '
                f'from centroid side (y={centroid_p[1]:+.3f}m)')
        return arm

    def _publish_grasp_markers(self, frame, poses, scores, status=None):
        """
        Publish the grasp candidates on 'graspgen_markers' (latched), at most KEPT_MARKER_MAX.

        Every candidate in `status` (index -> GRASP_MARKER_COLORS state)
        first, then the rest best-score-first as 'pending'. Per grasp: an
        ARROW from the gripper-base origin along +Z to the finger contact
        point, a SPHERE there, and a TEXT label with the candidate index (the
        'grasp {i}' of the log lines). Marker ids are candidate indices, so a
        state change replaces that grasp's markers. Best-effort: drawing never
        breaks the grasp.
        """
        status = status or {}
        try:
            if not hasattr(self, '_grasp_marker_pub'):
                self._grasp_marker_pub = self.create_publisher(
                    MarkerArray, 'graspgen_markers',
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            order = sorted(range(len(poses)),
                           key=lambda i: (i not in status, -scores[i]))
            # zero stamp = latest available TF; a now() stamp races the
            # pelvis -> map link and RViz drops the markers
            stamp = Time()
            arr = MarkerArray()
            clear = Marker()
            clear.header.frame_id = frame
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)
            for i in order[:KEPT_MARKER_MAX]:
                T = pose_to_matrix(poses[i])
                base = T[:3, 3]
                contact = base + GRIPPER_BASE_TO_CONTACT_M * T[:3, 2]
                r, g, b, a = GRASP_MARKER_COLORS[status.get(i, 'pending')]

                arrow = Marker()
                arrow.header.frame_id = frame
                arrow.header.stamp = stamp
                arrow.ns = 'graspgen_approach'
                arrow.id = i
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.points = [
                    Point(x=float(base[0]), y=float(base[1]), z=float(base[2])),
                    Point(x=float(contact[0]), y=float(contact[1]),
                          z=float(contact[2]))]
                arrow.scale.x = 0.008                  # m, shaft diameter
                arrow.scale.y = 0.018                  # m, head diameter
                arrow.scale.z = 0.03                   # m, head length
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = r, g, b, a
                arr.markers.append(arrow)

                dot = Marker()
                dot.header.frame_id = frame
                dot.header.stamp = stamp
                dot.ns = 'graspgen_contact'
                dot.id = i
                dot.type = Marker.SPHERE
                dot.action = Marker.ADD
                dot.pose.position = Point(
                    x=float(contact[0]), y=float(contact[1]), z=float(contact[2]))
                dot.pose.orientation.w = 1.0
                dot.scale.x = dot.scale.y = dot.scale.z = 0.02
                dot.color.r, dot.color.g, dot.color.b, dot.color.a = r, g, b, a
                arr.markers.append(dot)

                label = Marker()
                label.header.frame_id = frame
                label.header.stamp = stamp
                label.ns = 'graspgen_index'
                label.id = i
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position = Point(
                    x=float(base[0]), y=float(base[1]), z=float(base[2]) + 0.02)
                label.pose.orientation.w = 1.0
                label.scale.z = 0.025                  # m, text height
                label.text = str(i)
                label.color.r, label.color.g, label.color.b, label.color.a = r, g, b, 1.0
                arr.markers.append(label)
            self._grasp_marker_pub.publish(arr)
        except Exception as e:
            self.get_logger().warn(f'grasp marker publish failed: {e}')

    def _yolo_box(self, obj, arm=None):
        """
        Return the pixel-xyxy box of a head-camera YOLO detection of `obj`, or None.

        Among several instances, the one closest to `arm`'s gripper (ranked
        in that arm's frame), else to the pelvis origin.
        """
        target_frame = GRASP_FRAMES[arm] if arm in GRASP_FRAMES else 'pelvis'
        det, _ = self._closest_detection(obj, self.latest_detections(),
                                         target_frame=target_frame)
        if det is None:
            self.get_logger().warn(
                f'grasp: no head YOLO detection for {obj!r} with usable depth; '
                f'SAM will fall back to the text prompt alone')
            return None
        return [float(det.bbox_min.x), float(det.bbox_min.y),
                float(det.bbox_max.x), float(det.bbox_max.y)]

    def _closest_detection(self, obj, bundle, target_frame='pelvis'):
        """
        Return (detection, point) of the class-`obj` detection nearest `target_frame`'s origin.

        Each detection is back-projected to a 3-D point in `target_frame`; the
        physically nearest wins, not the most confident. Detections without
        depth are skipped; (None, None) when nothing matches. The class match
        ignores case.
        """
        if bundle is None or not bundle.detections:
            return None, None
        key = obj.strip().lower()
        best_det, best_point, best_dist = None, None, None
        for d in bundle.detections:
            if d.cls.strip().lower() != key:
                continue
            pt = self._detection_point(d, target_frame=target_frame)
            if pt is None:
                continue
            dist = float(np.linalg.norm(pt))
            if best_dist is None or dist < best_dist:
                best_det, best_point, best_dist = d, pt, dist
        return best_det, best_point

    def _detection_point(self, det, target_frame='pelvis'):
        """
        Back-project a Detection's box with the head depth to one point in `target_frame`.

        The median over the box's central quarter, which avoids background
        depth at the edges; None without depth.
        """
        info = self.latest_caminfo()
        if info is None or not info.width or not info.height:
            return None
        w, h = int(info.width), int(info.height)
        cx = (det.bbox_min.x + det.bbox_max.x) / 2.0
        cy = (det.bbox_min.y + det.bbox_max.y) / 2.0
        half_w = max(1.0, abs(det.bbox_max.x - det.bbox_min.x) * 0.25)
        half_h = max(1.0, abs(det.bbox_max.y - det.bbox_min.y) * 0.25)
        u0, u1 = max(0, int(round(cx - half_w))), min(w, int(round(cx + half_w)))
        v0, v1 = max(0, int(round(cy - half_h))), min(h, int(round(cy + half_h)))
        if u1 <= u0 or v1 <= v0:
            return None
        mask = np.zeros((h, w), dtype=bool)
        mask[v0:v1, u0:u1] = True
        pts = self._depth_to_cloud(mask, target_frame)
        if pts is None or len(pts) == 0:
            return None
        return np.median(pts, axis=0)


def _pose_xyz(pose):
    """Return `pose`'s position as a float array."""
    return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)


def _pose_quat(pose):
    """Return `pose`'s orientation as a unit quaternion array (x, y, z, w)."""
    q = np.array([pose.orientation.x, pose.orientation.y,
                  pose.orientation.z, pose.orientation.w], dtype=float)
    return q / np.linalg.norm(q)


def _quat_angle(q0, q1):
    """Return the rotation angle [rad] between unit quaternions q0 and q1."""
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))


def _quat_slerp(q0, q1, t):
    """Interpolate from q0 (t=0) to q1 (t=1) along the shortest arc."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    # nearly parallel: normalized lerp avoids dividing by sin(theta) ~ 0
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    return (math.sin((1.0 - t) * theta) * q0 + math.sin(t * theta) * q1) / math.sin(theta)


def get_approach_pose(pose, approach_dist):
    """
    Return a copy of `pose` moved `approach_dist` m along its own +Z (the approach axis).

    Positive is deeper toward the object, negative backs off (a pre-grasp
    standoff); the orientation is unchanged.
    """
    z_axis = pose_to_matrix(pose)[:3, 2]
    out = copy.deepcopy(pose)
    out.position.x += approach_dist * float(z_axis[0])
    out.position.y += approach_dist * float(z_axis[1])
    out.position.z += approach_dist * float(z_axis[2])
    return out


def _yolo_routed(obj):
    """Return True when `obj` is a battery part on the YOLO path (BATTERY_USE_GEMINI off)."""
    return not BATTERY_USE_GEMINI and obj.strip().lower() in _BATTERY_OBJECTS_LC


def _perception_name(obj):
    """Return the name perception queries use: a battery label's GEMINI_NAMES phrase on Gemini."""
    if BATTERY_USE_GEMINI:
        return GEMINI_NAMES.get(obj.strip().lower(), obj)
    return obj


def _servo_uses_gemini(obj):
    """Return True when the visual servo boxes `obj` with Gemini rather than hand YOLO."""
    return SERVO_USE_GEMINI or not _yolo_routed(obj)


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


def _approach_target_tf(parent_frame, pose, stamp, child=TARGET_FRAME):
    """Build a debug TransformStamped placing `child` (default TARGET_FRAME) at `pose`."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child
    t.transform.translation.x = pose.position.x
    t.transform.translation.y = pose.position.y
    t.transform.translation.z = pose.position.z
    t.transform.rotation = pose.orientation
    return t
