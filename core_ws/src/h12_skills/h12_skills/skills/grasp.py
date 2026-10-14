"""SkillGrasp: perception, a pluggable synthesis method, one executor.

Flow (_exec_grasp, the orchestrator):

    prep -> detect_object (box) -> segment_object (cloud)
         -> get_grasp_candidates (the selected grasp_synthesis method)
         -> move_to_candidate_approach (take off in Z, transit to pre-grasp)
         -> move_to_candidate (in along the grasp axis) + force close

Perception and execution are the same for every synthesis method; only the
candidate list differs (see grasp_synthesis/__init__.py for the methods and
how one is selected). Execution mode is a separate goal toggle: with
goal.visual_servo the contact move is replaced by servo() — a closed
hand-camera loop that aligns on the object at VISUAL_SERVO_DEPTH_M and closes
there. A servo that fails to converge still closes where it stopped; a cancel
or timeout aborts.

Testing switch: with APPROACH_IMAGE_ONLY on, the skill ends at the pre-grasp
— it saves one frame from the grasping arm's hand camera, returns the arms to
'prep', and succeeds without ever moving to contact or closing the gripper.

Every arm move refuses to send once the safety layer latches estop or the
frame_task server aborts a move (base.arm_halt_reason); the skill then aborts
instead of trying further candidates or retreating.

The box comes from Gemini for a generic object, or from the head-camera YOLO
detector for a battery-workcell part (BATTERY_OBJECTS; SAM is skipped there
too — small parts are below its reliable scale). That routing lives entirely
in detect_object. The module constants below are the tuning knobs, grouped by
pipeline stage.
"""

import copy
import math
import os
import re
import time
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
from ..model_logging import _log_root
from ..perception_utils import pose_to_matrix, extract_json
from ..grasp_synthesis import METHODS, SynthesisInput, method_name
from ..grasp_synthesis.common import GRIPPER_BASE_TO_CONTACT_M, pose_at


# ========================== perception routing ===============================
# Battery-workcell parts the fine-tuned YOLO-World checkpoint detects (mirrors
# yolo_server.DEFAULT_QUERIES; 'OrageCover' matches the checkpoint's label
# spelling). These get a YOLO box instead of a Gemini one, and the raw
# bounding-box cloud instead of a SAM mask.
BATTERY_OBJECTS = ('Bolt', 'BusBar', 'InteriorScrew', 'Nut', 'OrageCover',
                   'Screw', 'ScrewHole')
_BATTERY_OBJECTS_LC = frozenset(o.lower() for o in BATTERY_OBJECTS)

# Multi-instance variant of base.GEMINI_GRASP_PROMPT: one box per VISIBLE
# instance instead of exactly one. Used by the grasp skill to disambiguate
# look-alikes (e.g. several grey screws) — every instance is boxed, then the
# one closest to the working arm is grasped (_gemini_box_closest_arm). Each box
# still FULLY ENCOMPASSES its instance so it remains a valid SAM prompt.
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

# Gemini prompt for the hand-camera VISUAL SERVO: a single box to center the
# object on. The servo only translates (orientation is the planned grasp's),
# so no grasp axis is requested. Same 0-1000, y-first normalized
# convention as the other Gemini prompts.
GEMINI_SERVO_PROMPT = (
    'A robot gripper is about to pick up the {obj} in this hand-camera image. '
    'Return ONE JSON object and no prose: {{"box_2d": [y1, x1, y2, x2]}} where '
    'box_2d FULLY ENCOMPASSES the {obj}. Coordinates are integers normalized to '
    '0-1000 with y first. If the {obj} is not visible, return {{}}.'
)


# ===================== failure retreat (_retreat_after_failure) ==============
# Straight-up lift before a skill reports failure, so a failed grasp doesn't
# leave the hand parked in the workspace. Run in slow mode: it starts wherever
# the failure left the hand, often next to the object. Skipped for the
# prep-config failure (arm never moved) and when arm motion is halted.
FAILURE_RETRACT_M = 0.15
FAILURE_RETRACT_SEC = 3.0   # full-speed budget; scaled by SLOW_MODE_TIME_SCALE


# ========================= motion / execution knobs ==========================
# Fraction of the gripper's full range the jaws are set to BEFORE the approach
# (the final close is force-based, so there is no closed-fraction knob). 1.0 is
# fully open; 0.5 leaves the jaws half-open so they approach already half-closed.
OPEN_PERCENT = 1.0
# Metres to back off along the grasp's +Z approach axis for the pre-grasp.
# With a visual-servo goal the servo takes over FROM here (it drives the last
# stretch in to VISUAL_SERVO_DEPTH_M), so this doubles as the servo's start
# standoff — kept tight at 10 cm so the servo begins close to the object.
APPROACH_DIST = 0.10
# Metres to shift every grasp along its own approach axis before executing
# (positive = deeper). The pre-grasp standoff moves with it.
GRASP_OFFSET = 0.0
# Pause before the close [s]: frame_task reports done at IK convergence but
# the arm is still settling several mm; closing into that drags the part.
PRE_CLOSE_SETTLE_SEC = 1.0
# Pause after the close [s] (on top of close_gripper's own settle + hold loop)
# so the grip establishes at full force before e.g. pick_place's lift.
POST_CLOSE_SETTLE_SEC = 4.0
# Metres to STOP SHORT of the grasp pose on the contact move (0 = full pose).
CONTACT_OFFSET = 0.25 * APPROACH_DIST
# Testing switch: hold at the pre-grasp until Enter is pressed on the skills
# node's terminal (needs interactive stdin; ignores cancel while blocked).
CONFIRM_BEFORE_CONTACT = False

# ================== approach-snapshot mode (APPROACH_IMAGE_ONLY) =============
# Testing switch: END the skill at the pre-grasp. After the approach the
# skill settles, saves ONE frame from the grasping arm's HAND camera (raw
# color stream — no head-camera fallback), returns the arms to 'prep', and
# succeeds — no visual servo (goal.visual_servo is ignored: it aligns a close
# that never happens, and its failure would abort before the image), no
# contact move, no gripper close, no GraspOutcome (so pick_place's inner
# grasp fails on the missing outcome while this is on). Images land in
# <h12_skills>/logs/approach_images/ (host-persistent through the core_ws
# bind mount).
APPROACH_IMAGE_ONLY = False
# How long to wait for a hand-camera frame captured AT the pre-grasp before
# settling for the latest cached hand frame [s].
APPROACH_IMAGE_WAIT_SEC = 3.0

# ======================= visual servo (goal.visual_servo) ====================
# Closed-loop hand-camera alignment at the pre-grasp (servo()). The hand
# camera looks straight down the approach axis, 4.4 cm ahead of the gripper
# base, so "centered at depth D" = "on the approach ray, D ahead of the
# camera". The finger contact point sits 0.0706 m in front of the camera.
#
# Target camera->object depth [m] — a REAL drive target near where the fingers
# should sit (~100 mm; the finger contact point is 70.6 mm ahead of the
# camera, so the object ends ~29 mm beyond the fingertips). The servo drives
# depth TO this, so it must be a sensible value, not a band-center trick: a
# one-sided band such as (0,105) centers the setpoint at 52.5 mm and makes the
# servo plunge toward it whenever the object starts beyond 105 mm, overshooting
# into the part and disturbing X/Y. Paired with VISUAL_SERVO_RANGE_TOL_M for the
# accepted band (99, 101) mm.
VISUAL_SERVO_DEPTH_M = 0.10
# Per-axis convergence tolerances [m], measured from real screw-picking runs
# rather than derived — these are the values those runs converge with. The
# binding limit is the ARM: frame_task settles to ~1.2 mm residual, so
# corrections under ~2 mm don't execute and <4 mm lateral is unreachable.
#   X (across the jaws) — what a grasp depends on. BELOW the arm's own ~1.2 mm
#     resolution, so the loop cannot deliberately place within this band; it
#     converges when it happens to land there. If runs stall naming X, loosen
#     toward 1-2 mm (2 mm is the honest value against that residual).
#   Y (along the jaw span) — pre-opened to 106 mm, so far more forgiving; 5 mm
#     still holds the part near the middle of the grip rather than out at a jaw.
#   range — decides whether the jaws close around the part or above it.
# Keep X <= Y: inverting them is backwards for a parallel gripper.
VISUAL_SERVO_X_TOL_M = 0.00025  # +/-0.25 mm -> X band (11.75, 12.25) at 12 mm target
VISUAL_SERVO_Y_TOL_M = 0.005
VISUAL_SERVO_RANGE_TOL_M = 0.001
# Wall-clock budget for the whole loop [s] (~VISUAL_SERVO_MOVE_SEC per
# iteration). The goal's own timeout still wins.
VISUAL_SERVO_TIMEOUT_SEC = 90.0
# How long to wait for a usable hand-camera detection before LOST [s]
# (yolo_server publishes at ~5 Hz).
VISUAL_SERVO_DETECT_WAIT_SEC = 5.0
# Source the visual-servo box from GEMINI on the hand-camera frame instead of
# the hand YOLO detector — for EVERY object, INCLUDING battery parts that use
# YOLO on the HEAD camera (detect_object). A generic (non-battery) object is not
# even a hand-YOLO class, so the YOLO servo would go permanently LOST. The YOLO
# bundle is still used for its aligned depth + intrinsics; only the BOX comes
# from Gemini (see _servo_measure_gemini). Set False to go back to the YOLO
# hand-detector servo.
SERVO_USE_GEMINI = False
# Per-query timeout for the servo's Gemini box [s]. A Gemini call is
# seconds-to-minutes — far slower than a frame_task iteration — so a Gemini
# servo is deliberately slow-but-general. The LOST timer is stretched to match
# (see servo()), so a slow-but-successful query is not mistaken for lost sight.
SERVO_GEMINI_TIMEOUT_SEC = 60.0
# Lost sight is usually geometry (out of frame / occluded by the fingers /
# inside the D405 min range): back straight up (pelvis +Z) and keep servoing.
VISUAL_SERVO_LOST_RETRACT_M = 0.05
# Depth at/below which the object is TOO CLOSE to steer by [m]. Keep clearly
# BELOW 0.0706: the fingers themselves sit there, and a higher threshold reads
# finger material as "too close" and can never heal by backing off.
VISUAL_SERVO_MIN_RANGE_M = 0.055
# Consecutive too-close readings before acting (one frame must never command
# a 5 cm retreat).
VISUAL_SERVO_TOO_CLOSE_STRIKES = 2
# Plausibility band on a new measurement's depth [m] around the prediction
# from the last correction — rejects fingers/shadows/background without
# rejecting honest motion (MAX_STEP is inside the band by construction).
VISUAL_SERVO_RANGE_BAND_M = 0.02
# Cap on retracts per run, shared by lost-sight and too-close: uncapped, the
# loop would walk the arm out of the workspace 5 cm at a time.
VISUAL_SERVO_MAX_RETRACTS = 2
# Stall detection: an iteration counts as progress only if it beats the best
# error by EPS; STALL_ITERS non-improving iterations end the run with the
# residual in the message (both reset after a back-off).
VISUAL_SERVO_STALL_ITERS = 5
VISUAL_SERVO_STALL_EPS_M = 0.001
# Proportional gain (<1 so a bad depth sample can't fling the hand and the
# loop damps rather than oscillates).
VISUAL_SERVO_GAIN = 0.6
# Largest single correction [m] — bounds what one mis-detection can do.
VISUAL_SERVO_MAX_STEP_M = 0.05
# Per-iteration frame_task move duration [s]. Sized to INCLUDE the server's
# steady-state hold — cutting the hold short bakes in gravity droop
# (~5 mm/iteration observed).
VISUAL_SERVO_MOVE_SEC = 4.0
# Pause after each move before the next measurement [s], so the loop doesn't
# act on a frame captured mid-move.
VISUAL_SERVO_POST_MOVE_SLEEP_SEC = 0.25
# SETPOINT: the object is driven to (X, Y, DEPTH) in the camera optical frame.
# Fixed constants, nothing derived from TF or fitted per-run (a TF-derived
# finger-point target was tried and did not behave — the camera mount
# transform is suspect; recover any camera-to-finger offset by MEASURING the
# mount, not by re-introducing a trim here). Positive X sits the object
# further right in the image / hand further back in pelvis +X.
VISUAL_SERVO_TARGET_XY_M = (0.012, 0.0)
# Depth floor for the servo's back-projection [m] — base's DEPTH_MIN_M (0.1)
# sits ABOVE the target depth and would reject the needed measurements.
VISUAL_SERVO_MIN_DEPTH_M = 0.04

# The servo holds the planned grasp orientation — whatever the synthesis
# method produced — and only TRANSLATES. There is no in-loop orientation
# alignment.


# =============== reach budgets (move_to_candidate_approach etc.) =============
# Push past servo_frame_to_world's defaults: more time on the main IK move and
# more world-frame refinement passes. The iter-0 unreachable fast-fail still
# bails dead candidates quickly.
SERVO_DURATION_SEC = 15   # primary (iter-0) contact IK move budget [s]
# Timeout (not trajectory time) per approach move: generous so a long transit
# is never cut off; rejected plans return in well under a second.
APPROACH_DURATION_SEC = 180.0
SERVO_MAX_ITER = 12       # world-frame servo refinement passes per pose
# Anchor the contact move in WORLD_FRAME for pelvis-drift compensation. False
# drives the raw pelvis-frame poses (as when the world TF is down) — use it
# when odometry is poor enough that the world-frame corrections add error.
GRASP_WORLD_ANCHOR = False
# Stop a frame_task move early once the server's streamed feedback repeats
# UNCHANGED for this many messages — the arm has settled and the server is only
# holding steady state (which otherwise runs to its ~5 s hold timeout). Passed
# to every frame_task move the grasp skill commands. None disables it.
FRAME_TASK_STABLE_STOP_MSGS = 20
# Convergence tolerances, relaxed from base.py's 5 mm / ~1.15 deg: real-robot
# IK + pelvis drift rarely settle a 6-DOF grasp pose that tight. These gate the
# CONTACT move.
SERVO_LIN_TOL = 0.0125
SERVO_ANG_TOL = 0.05
# Looser tolerances for the PRE-GRASP approach only (move_to_candidate_approach).
# The pre-grasp is just a standoff the visual servo then refines from, so a
# loosely-reached approach is fine — and a tight approach tolerance was
# rejecting reachable candidates as "unreachable" and burning through the pool.
APPROACH_LIN_TOL = 0.025
APPROACH_ANG_TOL = 0.10
# Every candidate is approached in three straight-line, UNPLANNED, SLOW-mode
# stages (_straight_line_move):
#   1. take off: straight up (pelvis +Z), holding the current orientation, to
#      APPROACH_CLEARANCE_M above the higher of the hand and the pre-grasp;
#   2. transit to the pre-grasp, position and orientation interpolated;
#   3. in along the grasp's own +Z approach axis to the contact pose
#      (move_to_candidate), orientation held.
# The visual servo, when requested, replaces stage 3. A failed candidate is
# followed by a planned reset to 'prep' (a staged method: its staging config).
APPROACH_CLEARANCE_M = 0.05
# Stage 1-2 segment limits: at most this far / this much rotation per
# frame_task goal, so no single goal asks for a large reorientation.
LINEAR_APPROACH_SEG_M = 0.10
LINEAR_APPROACH_SEG_RAD = 0.35
# Per-segment frame_task budget [s] (already the slow-mode-scaled value the
# server needs to cover LINEAR_APPROACH_SEG_M at 1/4 speed).
LINEAR_APPROACH_SEG_SEC = 10.0
# Stage 3 step along the grasp axis [m] and its per-step budget [s].
AXIS_APPROACH_STEP_M = 0.025
AXIS_APPROACH_STEP_SEC = 6.0


# ============================== visualization ================================
# TF frame the currently-driven target is broadcast to (RViz).
TARGET_FRAME = 'graspgenx_target_frame'
# Cap on 'graspgen_markers' markers.
KEPT_MARKER_MAX = 20
# Candidate state -> marker (r, g, b, a). Untried candidates are 'pending'.
GRASP_MARKER_COLORS = {
    'pending': (0.5, 0.5, 0.0, 0.4),   # dim yellow
    'trying': (0.0, 0.6, 1.0, 1.0),    # blue: pre-grasp approach in progress
    'failed': (1.0, 0.0, 0.0, 0.8),    # red: pre-grasp unreachable
    'chosen': (0.0, 1.0, 0.0, 1.0),    # green: committed grasp
}


# ================================ data types =================================
@dataclass
class GraspOutcome:
    """What an in-process caller (pick_place) needs from an executed grasp,
    handed back through the node's `_last_grasp_outcome` attribute — the
    SkillGrasp result message only carries success/message."""
    arm: str                # arm that executed it (resolved, so auto callers learn it)
    pose: Pose              # executed grasp pose (GraspGenX gripper-base), in `frame`
    frame: str              # WORLD_FRAME when the world TF was available, else 'pelvis'
    centroid: np.ndarray    # object-cloud centroid [m], same frame
    rest_height: float      # centroid height above the object's own bottom [m]
    gripper_width: float    # method's planned opening [m]; 0.0 when it has none
    score: float            # confidence of the executed grasp
    method: str             # grasp_synthesis method that produced it


@dataclass
class _Targets:
    """Per-candidate executable poses, frozen once by _snapshot_targets."""
    frame: str              # the candidates' planning frame ('pelvis')
    grasps_p: list          # GRASP_OFFSET-shifted grasp poses, planning frame
    approaches_p: list      # pre-grasp standoffs (APPROACH_DIST back), same frame
    grasps_w: list          # world-frame snapshots (entries None if TF was down)
    approaches_w: list      # world-frame snapshots of the pre-grasps
    have_world: bool        # all snapshots resolved -> drift-compensated servoing
    centroid_w: object      # object centroid as a world Pose (None without TF)


class GraspSkill:
    # ------------------------------------------------------ action entry point
    def _exec_grasp(self, gh):
        """Orchestrate one grasp: prep -> detect_object -> segment_object ->
        get_grasp_candidates -> move_to_candidate_approach -> contact + close.
        No lift afterwards (by design).

        The synthesis method is the node's `grasp_method` parameter, or
        `top_down` when the goal sets it (_resolve_grasp_method).

        Optional goal fields (see SkillGrasp.action; zero/false = off):
          top_down — use the top_down synthesis method for this goal.
          min_downward_pitch_deg — every method keeps only grasps approaching
            at least this far below horizontal.
          visual_servo — replace the contact move with servo(): align on the
            hand camera at VISUAL_SERVO_DEPTH_M and close there.

        With APPROACH_IMAGE_ONLY on (module testing switch), the flow ends at
        the pre-grasp: save one hand-camera image, return to 'prep', succeed —
        goal.visual_servo is ignored and the contact/close stage below never
        runs, so no GraspOutcome is set.

        In-process RESULT side channel (for pick_place, same node):
        `self._last_grasp_outcome` is cleared here and only repopulated by a
        SUCCESSFUL grasp, so a failed grasp can never leak a stale outcome.
        Assumes one grasp goal in flight at a time (true in practice)."""
        goal = gh.request
        run = _Run(self, gh, SkillGrasp, 'grasp')
        self._last_grasp_outcome = None
        arm = self._validated_arm(goal)
        if arm is None and goal.arm.strip().lower() not in ('', 'none'):
            return run.abort(f'invalid arm {goal.arm!r}')
        method, err = self._resolve_grasp_method(goal)
        if err:
            return run.abort(err)
        obj = goal.target_object
        self.get_logger().info(
            f'grasp: {obj!r} with synthesis method {method.name!r}'
            f'{" + visual servo" if goal.visual_servo else ""}')

        # --- detect: prep the arms, box the object, lift it to a cloud --------
        if not run.phase('detect_object', 0.0):
            return run.result
        # Arms to 'prep' FIRST, clear of the head camera's view (an arm across
        # the object corrupts the box/mask/cloud). Planned, since the arms
        # start wherever the last skill left them. The ONLY failure that does
        # not retract — the skill has not moved the arm anywhere of its own.
        if not self.goto_named_config('prep', plan=True, slow_mode=True,
                                      duration_sec=30.0, result_timeout=90.0,
                                      outer_gh=gh):
            return run.abort("move to 'prep' before detection failed")

        def fail(message):
            """Abort, but lift the hand clear of the workspace first — from
            here on a failure leaves the arm somewhere it was put. The
            approach walk is the exception (plain run.abort below)."""
            self._retreat_after_failure(arm, gh)
            return run.abort(message)

        box, use_sam = self.detect_object(obj, run, gh, arm)
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return fail('detection canceled or timed out')
        obj_cloud, scene, err = self.segment_object(obj, box, use_sam, gh)
        if err:
            return fail(err)
        # Held-object geometry for the pick_place side channel, measured while
        # the object still sits undisturbed: centroid + its rest height.
        centroid_p = obj_cloud.mean(axis=0)
        rest_height = float(centroid_p[2] - np.percentile(obj_cloud[:, 2], 5))
        if arm is None:   # auto-select: the arm currently closest to the object
            arm = self._closest_arm(centroid_p)
        self._note_run_arm(arm)    # point the sway recorder's gripper column at it

        # Pre-open the gripper here — after 'prep', before staging — so the arm
        # travels to the staging pose (when used) and everything after open.
        if not self.open_gripper(arm, OPEN_PERCENT):
            return fail('gripper pre-open failed')

        # --- staging (staged methods only): named config for the arm we are
        # going to use. Joint-space (recorded on the robot), so it needs no
        # IK/reach solve and lands the grasping arm in a repeatable posture
        # above the workspace for a steep top-down approach, with the other
        # arm parked at 'prep'. Other methods start straight from 'prep'.
        stage_cfg = 'staging_right' if arm == 'right' else 'staging_left'
        if method.staged:
            if not self.goto_named_config(stage_cfg, plan=True, slow_mode=True,
                                          duration_sec=30.0, result_timeout=90.0,
                                          outer_gh=gh):
                return fail(f"move to {stage_cfg!r} staging config failed")

        # --- plan: ordered candidates from the synthesis method --------------
        if not run.phase('approach_grasp', 0.4):
            return run.result
        cands, err = self.get_grasp_candidates(method, goal, obj, arm,
                                               obj_cloud, scene)
        if err:
            return fail(err)

        # --- approach: servo to the first reachable pre-grasp -----------------
        # (the gripper was already pre-opened after 'prep', before staging)
        targets = self._snapshot_targets(cands, centroid_p)
        if targets.centroid_w is not None:   # sway target = object centroid in world
            self._note_run_target(targets.centroid_w)
        reset_cfg = stage_cfg if method.staged else 'prep'
        idx, err = self.move_to_candidate_approach(run, gh, obj, arm, cands,
                                                   targets, reset_cfg)
        if err:
            # No retreat lift: a failed walk ends with the arms reset to
            # reset_cfg, and a halted arm must not be moved at all.
            return run.abort(err)
        # --- snapshot mode: save an image at the pre-grasp, then withdraw ----
        # PREEMPTS the visual servo and the contact confirm: both exist to set
        # up a close that never happens in this mode, and a servo failure
        # (e.g. losing sight of the object) would abort the skill before the
        # image is taken — which is the one thing this mode is for.
        if APPROACH_IMAGE_ONLY:
            if not run.phase('save_image', 0.75):
                return run.result
            # Same settle as before a close: frame_task reports done while the
            # arm is still moving several mm, which would blur the frame.
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

        # Optional closed-loop alignment at the pre-grasp. On success this
        # REPLACES the contact move: the gripper closes where the loop parked.
        servo_pose_p = None
        if goal.visual_servo:
            servo_pose_p, err = self.servo(run, gh, arm, obj, targets, idx)
            if err:
                # A cancel or a timeout must still abort; but a genuine
                # non-convergence closes ANYWAY at wherever the servo left the
                # arm (best-effort) rather than giving up — the caller still
                # retracts/raises after.
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

        # --- grasp: move to contact + force close ----------------------------
        if not run.phase('close_gripper', 0.75):
            return run.result
        if servo_pose_p is not None:
            # Already parked by the visual servo — close where we stand.
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

        # Side channel for pick_place: pose and centroid share one frame (the
        # world snapshot when available). The pose is the CONTACT target —
        # where the hand actually is.
        if targets.have_world and targets.centroid_w is not None:
            out_pose, out_frame, out_c = contact_w, WORLD_FRAME, targets.centroid_w
        else:
            out_pose, out_frame, out_c = contact_p, 'pelvis', pose_at(centroid_p)
        self._last_grasp_outcome = GraspOutcome(
            arm=arm, pose=out_pose, frame=out_frame,
            centroid=np.array([out_c.position.x, out_c.position.y,
                               out_c.position.z]),
            rest_height=rest_height, gripper_width=cands.gripper_width,
            score=cands.scores[idx], method=method.name)

        return run.succeed(
            f'grasped {obj!r} with {method.name} candidate '
            f'{cands.labels[idx]} (score {cands.scores[idx]:.2f})')

    def _resolve_grasp_method(self, goal):
        """(GraspMethod, None) for this goal, or (None, abort-reason):
        goal.top_down forces `top_down`, else the `grasp_method` parameter."""
        name = method_name(goal.top_down,
                           self.get_parameter('grasp_method').value)
        method = METHODS.get(name)
        if method is None:
            return None, (f'unknown grasp_method {name!r}; expected one of '
                          f'{sorted(METHODS)}')
        return method, None

    # -------------------------------------------------------------- perception
    def detect_object(self, obj, run, gh, arm=None):
        """Locate `obj` in the head camera and return (box, use_sam): a pixel
        xyxy bounding box (or None) plus how segment_object should lift it to a
        cloud. Battery-workcell parts (BATTERY_OBJECTS) get a YOLO box and skip
        SAM (raw box cloud — small parts are below SAM's reliable scale);
        everything else gets a Gemini box refined by SAM. (The hand-camera
        visual servo uses Gemini for BOTH kinds — see SERVO_USE_GEMINI.)

        `arm` (the resolved grasping arm, or None for auto-select) picks, among
        multiple look-alike instances, the one closest to that arm's gripper —
        via the arm frame on the YOLO path (_yolo_box), and via
        _gemini_box_closest_arm on the Gemini path."""
        is_small = obj.strip().lower() in _BATTERY_OBJECTS_LC
        box = (self._yolo_box(obj, arm) if is_small
               else self._gemini_box_closest_arm(obj, run, gh, arm))
        return box, not is_small

    def _gemini_box_closest_arm(self, obj, run, gh, arm):
        """Gemini box (pixel xyxy) for `obj`, disambiguating multiple
        look-alike instances (e.g. grey screws) by picking the one whose
        back-projected head-camera point is CLOSEST to `arm`'s gripper — or,
        when no arm was provided, closest to the robot (pelvis origin).

        Falls back to base._gemini_box (the single-box, SAM-tuned query) when
        the multi-instance query returns zero or one box, or when no returned
        box has usable depth to range."""
        boxes = self._gemini_boxes(obj, run, gh)
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return None
        if len(boxes) <= 1:
            # Nothing to disambiguate — keep the well-tuned single-box path.
            return boxes[0] if boxes else self._gemini_box(obj, run, gh)
        # Reference point: the working arm's gripper origin in pelvis; with no
        # arm, the pelvis origin (nearest-to-robot).
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
        """Every Gemini bounding box (pixel xyxy) for `obj` in the head camera —
        one per visible instance — or []. Multi-instance analogue of
        base._gemini_box; the caller ranks them (see _gemini_box_closest_arm).
        Capped at the skill's remaining budget, `gh` threaded for prompt cancel."""
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
        """Back-project a pixel-xyxy `box` to a single (x, y, z) point in the
        pelvis frame via the head depth (central-quarter median), or None —
        reuses _detection_point by presenting the box as a Detection."""
        det = Detection()
        det.bbox_min = Point(x=float(box[0]), y=float(box[1]), z=0.0)
        det.bbox_max = Point(x=float(box[2]), y=float(box[3]), z=0.0)
        return self._detection_point(det, target_frame='pelvis')

    def segment_object(self, obj, box, use_sam, gh):
        """Lift a bounding box to point clouds: (obj_cloud, scene_cloud, None)
        or (None, None, reason). With `use_sam`, SAM is prompted with the
        object name plus the box as a positive exemplar (box optional — text
        alone is the fallback) and only masked pixels are back-projected;
        without it the box RECTANGLE is back-projected directly and must be
        present. The whole-frame scene cloud rides along as obstacle context
        for graspgen's collision filter (None just skips filtering)."""
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

    # ---------------------------------------------------- candidate generation
    def get_grasp_candidates(self, method, goal, obj, arm, obj_cloud, scene):
        """Run `method` (a grasp_synthesis.GraspMethod) on the segmented
        object: (Candidates in execution order, None) or (None, abort-reason).
        """
        inp = SynthesisInput(
            obj=obj, arm=arm, cloud=obj_cloud, scene=scene,
            min_pitch_deg=float(goal.min_downward_pitch_deg or 0.0),
            plan_grasp=self.plan_grasp, log=self.get_logger())
        cands, err = method.synthesize(inp)
        if err is None and not cands.grasps:
            err = f'{method.name} produced no grasp for {obj!r}'
        return (None, err) if err else (cands, None)

    # ---------------------------------------------------------------- execution
    def _snapshot_targets(self, cands, centroid_p):
        """Freeze every candidate's grasp + pre-grasp pose (and the centroid)
        from the CURRENT pelvis frame into the world frame, once, up front —
        world-anchored targets stay correct as the pelvis drifts
        (servo_frame_to_world re-resolves them each iteration). If the world
        TF is unavailable or GRASP_WORLD_ANCHOR is off, `have_world` is False
        and the raw pelvis poses are driven with no drift compensation."""
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
                            seg_sec, lin_tol, ang_tol, label, slow_mode=True):
        """Move `frame` from its current pose to `target` (pelvis frame) along a
        straight line, as DIRECT (unplanned) frame_task goals of at most
        `step_m` translation and `step_rad` rotation each; orientation is
        slerped. Returns True once every segment reached (lin_tol, ang_tol),
        False on the first miss, cancel, timeout or arm halt."""
        cur = self._frame_pose_in_pelvis(frame)
        if cur is None:
            return False
        p0 = _pose_xyz(cur)
        p1 = _pose_xyz(target)
        q0 = _pose_quat(cur)
        q1 = _pose_quat(target)
        dist = float(np.linalg.norm(p1 - p0))
        angle = _quat_angle(q0, q1)
        n = max(1, int(math.ceil(max(dist / step_m, angle / step_rad))))
        self.get_logger().info(
            f'grasp: {label}: {dist * 100:.1f}cm / {math.degrees(angle):.0f}deg '
            f'in {n} unplanned{" slow" if slow_mode else ""} step(s)')
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
            if not self.move_frame_to(
                    frame, wp, outer_gh=gh, duration_sec=seg_sec,
                    do_plan=False, slow_mode=slow_mode,
                    lin_tol=lin_tol, ang_tol=ang_tol,
                    stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                    label=f'{label} {k}/{n}'):
                return False
        return True

    def _approach_pregrasp(self, frame, pregrasp, run, gh):
        """Stages 1-2 of a candidate's approach: take off straight up (pelvis
        +Z, current orientation held) to APPROACH_CLEARANCE_M above the higher
        of the hand and the pre-grasp, then transit to the pre-grasp. Returns
        whether the pre-grasp was reached."""
        cur = self._frame_pose_in_pelvis(frame)
        if cur is None:
            return False
        lift_z = max(cur.position.z, pregrasp.position.z) + APPROACH_CLEARANCE_M
        if lift_z - cur.position.z > 1e-3:
            up = copy.deepcopy(cur)
            up.position.z = lift_z
            if not self._straight_line_move(
                    frame, up, run, gh, LINEAR_APPROACH_SEG_M,
                    LINEAR_APPROACH_SEG_RAD, LINEAR_APPROACH_SEG_SEC,
                    APPROACH_LIN_TOL, APPROACH_ANG_TOL, 'take-off (pelvis +Z)'):
                return False
        return self._straight_line_move(
            frame, pregrasp, run, gh, LINEAR_APPROACH_SEG_M,
            LINEAR_APPROACH_SEG_RAD, LINEAR_APPROACH_SEG_SEC,
            APPROACH_LIN_TOL, APPROACH_ANG_TOL, 'transit to pre-grasp')

    def move_to_candidate_approach(self, run, gh, obj, arm, cands, targets,
                                   reset_cfg):
        """Walk the candidates in the order the synthesis method gave them,
        reaching each pre-grasp by taking off in Z and then transiting to it
        (_approach_pregrasp) until one is reachable. After a
        failed candidate (the last one included) the arms return to the named
        config `reset_cfg` by a PLANNED joint-space move: a failed Cartesian
        approach can leave a joint pinned at its limit, from which no further
        Cartesian move (including the next take-off) makes progress.
        Returns (index, None) of the committed candidate, or
        (None, abort-reason) on cancel/timeout, arm halt, a failed reset or
        all-unreachable. Each
        candidate's state (trying / failed / chosen) is republished on
        'graspgen_markers' as it changes."""
        order = range(len(cands.grasps))
        width_mm = cands.gripper_width * 1000.0
        status = {}

        def mark(i, state):
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
            # Pelvis-frame pre-grasp (no world anchoring); the visual servo
            # or the in-along-the-axis contact move runs AFTER this.
            if self._approach_pregrasp(GRASP_FRAMES[arm], targets.approaches_p[i],
                                       run, gh):
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
        """Stage 3 of the approach: from the pre-grasp, go IN along the
        committed grasp's own +Z approach axis to the contact target (the grasp
        backed off CONTACT_OFFSET), in AXIS_APPROACH_STEP_M straight steps with
        the grasp orientation held, slow mode, unplanned (collision-aware
        planning would fight the intended approach into the object).
        Best-effort: advancing stops at the first step that misses tolerance,
        and the gripper still closes there. With world anchoring available, a
        final world-frame correction compensates pelvis drift. Returns
        (contact_p, contact_w) for the outcome record."""
        frame = GRASP_FRAMES[arm]
        contact_p = get_approach_pose(targets.grasps_p[idx],
                                      approach_dist=-CONTACT_OFFSET)
        contact_w = (get_approach_pose(targets.grasps_w[idx],
                                       approach_dist=-CONTACT_OFFSET)
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

    # ------------------------------------------------------------ visual servo
    def servo(self, run, gh, arm, obj, targets, idx):
        """Closed-loop hand-camera alignment at the pre-grasp
        (goal.visual_servo).

        Drives GRASP_FRAMES[arm] in PURE TRANSLATION until `obj` sits at
        (VISUAL_SERVO_TARGET_XY_M, VISUAL_SERVO_DEPTH_M) in the hand camera's
        optical frame. The orientation is held fixed at the planned grasp pose
        (as the synthesis method produced it), so the servo never changes it.

        One iteration: take a hand-camera DetectionBundle newer than the last
        frame consumed (fresh frames only — re-using one would apply the same
        correction twice), back-project the target's box to a 3-D point
        (bundle_detection_point), and step the gripper by GAIN x error,
        rotated into the pelvis frame and capped at MAX_STEP.

        Lost sight and too-close are RECOVERABLE: back straight up (pelvis
        +Z) by VISUAL_SERVO_LOST_RETRACT_M and carry on, up to
        VISUAL_SERVO_MAX_RETRACTS times.

        Returns (pose, None) with the pelvis-frame gripper Pose it converged
        at — the caller closes THERE, no contact descent — or (None, reason).
        A failure FAILS THE SKILL: quietly completing an unaligned open-loop
        grasp would hide exactly the misalignment the caller asked to
        eliminate."""
        frame = GRASP_FRAMES[arm]
        # Orientation is fixed for the whole servo — the planned grasp
        # orientation. The servo only translates.
        planned_q = targets.grasps_p[idx].orientation
        target = np.array([VISUAL_SERVO_TARGET_XY_M[0],
                           VISUAL_SERVO_TARGET_XY_M[1],
                           VISUAL_SERVO_DEPTH_M])
        # A single Gemini query already outlasts the YOLO LOST window, so in
        # Gemini mode wait ~1.5 queries before declaring lost sight.
        detect_wait = (1.5 * SERVO_GEMINI_TIMEOUT_SEC if SERVO_USE_GEMINI
                       else VISUAL_SERVO_DETECT_WAIT_SEC)
        deadline = time.monotonic() + VISUAL_SERVO_TIMEOUT_SEC
        last_seen = time.monotonic()
        retracts = 0
        best_err, stalled = None, 0        # stall detection
        too_close_hits = 0                 # consecutive sub-MIN_RANGE readings
        last_z = pred_z = None             # depth plausibility band
        # Only frames captured after this instant count, so the first
        # measurement describes the arm where it actually is.
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
            if time.monotonic() >= deadline:
                return None, (
                    f'visual servo did not converge on {obj!r} within '
                    f'{VISUAL_SERVO_TIMEOUT_SEC:.0f}s ({it} iteration(s), tol '
                    f'{VISUAL_SERVO_X_TOL_M * 1000:.0f}/'
                    f'{VISUAL_SERVO_Y_TOL_M * 1000:.0f}mm X/Y)')
            if SERVO_USE_GEMINI:
                p_cam, cam_frame, stamp, bundle = \
                    self._servo_measure_gemini(arm, obj, after, gh)
            else:
                p_cam, cam_frame, stamp, bundle = \
                    self._servo_measure(arm, obj, after)
            # Plausibility gate: a depth outside the band around what the last
            # correction predicted is not this object (finger, shadow,
            # background). Drop the frame; persistent implausibility ends up
            # in the lost-sight path, which is the right outcome.
            if p_cam is not None and pred_z is not None:
                lo = min(last_z, pred_z) - VISUAL_SERVO_RANGE_BAND_M
                hi = max(last_z, pred_z) + VISUAL_SERVO_RANGE_BAND_M
                if not lo <= float(p_cam[2]) <= hi:
                    self.get_logger().warn(
                        f'grasp: visual servo ignoring implausible depth '
                        f'{p_cam[2] * 1000:.0f}mm (expected '
                        f'{lo * 1000:.0f}-{hi * 1000:.0f}mm)',
                        throttle_duration_sec=2.0)
                    after = stamp          # consumed; do not re-test this frame
                    self.get_clock().sleep_for(RclpyDuration(seconds=0.1))
                    continue
            # LOST (nothing usable for DETECT_WAIT) and TOO CLOSE (inside
            # MIN_RANGE for STRIKES consecutive frames) are both answered by
            # backing straight up and looking again.
            if p_cam is not None and float(p_cam[2]) < VISUAL_SERVO_MIN_RANGE_M:
                too_close_hits += 1
            else:
                too_close_hits = 0
            lost = (p_cam is None
                    and time.monotonic() - last_seen > detect_wait)
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
                # After a back-off `after` becomes a CLOCK instant (pre-move
                # frames describe the previous vantage point outright), and the
                # detect wait, stall counters and depth prediction all reset —
                # the hand is somewhere genuinely new.
                after = _stamp_tuple(self.get_clock().now().to_msg())
                last_seen = time.monotonic()
                best_err, stalled = None, 0
                last_z = pred_z = None
                too_close_hits = 0
                continue
            if p_cam is None:
                self.get_clock().sleep_for(RclpyDuration(seconds=0.1))
                continue
            last_seen = time.monotonic()
            err = p_cam - target
            ex, ey = float(abs(err[0])), float(abs(err[1]))
            rng = float(abs(err[2]))
            # Move the camera BY the error (translating the camera by d shifts
            # the object's camera coordinates by -d), gain-scaled and capped.
            # All three axes (X, Y, Z) corrected together — including when depth
            # is already in band and only X/Y are off, which is the ordinary way
            # this loop finishes.
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
            # Stall detection: a run that stops improving will not start again on
            # its own — end it here with the residual, naming the axis out of
            # tolerance, rather than at the timeout.
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
            # Servo targets are pelvis-frame by construction; publish directly.
            self.publish_tf(_approach_target_tf(
                targets.frame, goal_pose, self.get_clock().now().to_msg()))
            # Undershoot is fine — the next measurement sees the error left.
            self.move_frame_to(
                frame, goal_pose, outer_gh=gh,
                duration_sec=VISUAL_SERVO_MOVE_SEC,
                do_plan=False, stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                label=f'visual servo correction iter {it}')
            self._log_servo_tracking(frame, cur, delta, it)
            # Depth prediction band for the next measurement: a fully-executed
            # move lands at z - step[2], an ignored one stays at z.
            last_z, pred_z = float(p_cam[2]), float(p_cam[2]) - float(step[2])
            after = stamp
            # Give the camera/detector a beat to publish from the new vantage.
            self.get_clock().sleep_for(
                RclpyDuration(seconds=VISUAL_SERVO_POST_MOVE_SLEEP_SEC))
            it += 1

    def _servo_measure(self, arm, obj, after):
        """One visual-servo measurement: the position of `obj` in the `arm`
        hand camera's optical frame, as (point, frame_id, stamp, bundle), or a
        tuple of Nones. Only accepts a bundle stamped strictly after `after` —
        the stamp of the frame last USED, not a wall-clock instant, so the
        same frame's correction is never applied twice while slow-inference
        frames stay usable (a stricter after-the-move rule starved the loop).
        Among detections of the class, picks the one CLOSEST TO THE BORESIGHT
        so the loop stays locked on its instance instead of hopping."""
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
            if p is None:                       # no usable depth under this box
                continue
            lat = math.hypot(float(p[0]), float(p[1]))
            if best_lat is None or lat < best_lat:
                best_p, best_lat = p, lat
        if best_p is None:
            return None, None, None, None
        return best_p, self.bundle_camera_frame(bundle), stamp, bundle

    def _servo_measure_gemini(self, arm, obj, after, gh):
        """Visual-servo measurement whose BOX comes from GEMINI instead of the
        hand-camera YOLO detector (SERVO_USE_GEMINI), so the servo can lock onto
        any object Gemini names rather than only a YOLO class.

        The arm's YOLO DetectionBundle is still consumed — but only for the
        aligned depth + camera_info + rgb frame yolo_server attaches every frame
        regardless of what it detects. Gemini boxes that rgb frame, the box is
        wrapped in a synthetic Detection, and the SAME bundle_detection_point
        back-projection as the YOLO path lifts it to a 3-D point in the hand
        camera's optical frame. Returns (point, frame_id, stamp, bundle) or a
        tuple of Nones. Only accepts a bundle stamped strictly after `after`, so
        one frame's correction is never applied twice.

        NOTE: a Gemini call is seconds-to-minutes — this is a deliberate
        slow-but-general servo (servo() stretches the LOST window to match)."""
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
        """Gemini bounding box (pixel xyxy) of `obj` in `image` (a
        CompressedImage), scaled by `info`'s width/height, or None. The
        hand-camera analogue of base._gemini_box (hard-wired to the head caches).
        `gh` threaded through so a cancel aborts the slow call."""
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
        """Publish the servo residual, stamped with the frame's CAPTURE time
        so plots line up with the camera stream it was measured from."""
        self.servo_error_pub.publish(_vec3_stamped(
            Time(sec=stamp[0], nanosec=stamp[1]), cam_frame, err))

    def _log_servo_tracking(self, frame, cur, delta, it):
        """Log + publish the executed move (TF after minus before) and its
        residual vs the commanded correction. Debug-only — a failed TF lookup
        just skips it."""
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
        """Draw one visual-servo frame for RViz: the measured object (red
        dot), the setpoint with its per-axis X/Y tolerance RECTANGLE (exactly
        the convergence test; an ellipse would under-report), and the
        commanded correction drawn as the object's PREDICTED next position.
        Range goes in the caption. Best-effort — viz must never break a
        grasp."""
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
                z = float(pt[2])
                if z <= 1e-6:
                    return None
                return (int(round(fx * float(pt[0]) / z + cx)),
                        int(round(fy * float(pt[1]) / z + cy)))

            # Project the setpoint's X/Y (and size its tolerance box) at the
            # OBJECT's CURRENT depth, not the depth setpoint. The convergence
            # test is on the 3-D X/Y error, so drawing both at the same depth is
            # what makes "dot inside box" match it — otherwise, while the object
            # depth differs from the target (e.g. the whole X/Y-first phase), the
            # box lands at the wrong pixel and looks offset.
            z_ref = float(p_cam[2]) if float(p_cam[2]) > 1e-6 else float(target[2])
            tgt_px = project(np.array([target[0], target[1], z_ref]))
            obj_px = project(p_cam)
            if tgt_px is not None:
                # Tolerances are metres at that depth; same pinhole scale -> px.
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
                # One line per axis: the current READING and the accepted
                # (min, max) band, green when the reading is inside it.
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

    # ------------------------------------------- approach snapshot (test mode)
    def _save_approach_image(self, arm, obj):
        """Save the `arm` HAND camera's view of `obj` from the pre-grasp
        (APPROACH_IMAGE_ONLY): the raw color stream's next frame — only
        frames captured from here on count, so a cached mid-move frame is
        never saved. If no fresh frame arrives within
        APPROACH_IMAGE_WAIT_SEC, the latest cached hand frame is saved with
        a warning; with no hand frame at all this returns None (there is
        deliberately no head-camera fallback — the point is the hand view).
        Returns the absolute path written, or None when no hand frame could
        be decoded or written."""
        after = _stamp_tuple(self.get_clock().now().to_msg())
        deadline = time.monotonic() + APPROACH_IMAGE_WAIT_SEC
        msg = None
        while time.monotonic() < deadline:
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
        path = os.path.join(_log_root('h12_skills', __file__),
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

    # -------------------------------------------------------- shared utilities
    def _retreat_after_failure(self, arm, gh):
        """Lift the hand FAILURE_RETRACT_M straight up (pelvis +Z) before a
        skill reports a failure. Shared with pick_place. Keeps the CURRENT
        orientation (from TF) — re-orienting a stuck arm is how a bad
        situation gets worse. Entirely best-effort: the skill is already
        failing, and the retreat must never replace the real reason."""
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
        """Current pose of URDF `frame` in the pelvis frame, or None."""
        return self._transform_pose(pose_at((0.0, 0.0, 0.0)), frame, 'pelvis')

    def _rotate_into_pelvis(self, vec, src_frame):
        """Rotate the 3-vector `vec` from `src_frame` into the pelvis frame
        (pure direction change), or None if TF is unavailable."""
        o = self._transform_pose(pose_at((0.0, 0.0, 0.0)), src_frame, 'pelvis')
        t = self._transform_pose(pose_at(vec), src_frame, 'pelvis')
        if o is None or t is None:
            return None
        return np.array([t.position.x - o.position.x,
                         t.position.y - o.position.y,
                         t.position.z - o.position.z])

    def _publish_target_tf(self, targets, pose_w, pose_p):
        """Broadcast TARGET_FRAME at the given target — the stable world
        snapshot when drift compensation is on, else the raw pelvis pose."""
        parent, pose = ((WORLD_FRAME, pose_w) if targets.have_world
                        else (targets.frame, pose_p))
        self.publish_tf(_approach_target_tf(
            parent, pose, self.get_clock().now().to_msg()))

    def _closest_arm(self, centroid_p):
        """Arm whose gripper is currently nearest the object centroid (pelvis
        xyz), via live TF; falls back to the side of the pelvis midline the
        object sits on (+Y = left) when a hand TF is unavailable."""
        dists = {}
        for arm, frame in GRASP_FRAMES.items():
            p = self._transform_pose(pose_at(centroid_p), 'pelvis', frame)
            if p is not None:
                dists[arm] = math.sqrt(p.position.x ** 2 + p.position.y ** 2
                                       + p.position.z ** 2)
        if len(dists) == len(GRASP_FRAMES):
            arm = min(dists, key=dists.get)
            self.get_logger().info(
                'grasp: auto-selected %s arm (hand-to-object %s)' % (arm,
                ', '.join(f'{a} {d:.3f}m' for a, d in sorted(dists.items()))))
        else:
            arm = 'left' if centroid_p[1] > 0.0 else 'right'
            self.get_logger().warn(
                f'grasp: hand TF unavailable for auto arm choice; picked {arm} '
                f'from centroid side (y={centroid_p[1]:+.3f}m)')
        return arm

    # ------------------------------------------------------------ visualization
    def _publish_grasp_markers(self, frame, poses, scores, status=None):
        """Publish grasp markers on 'graspgen_markers' (latched, pelvis frame),
        up to KEPT_MARKER_MAX: every candidate in `status` (index -> a
        GRASP_MARKER_COLORS state), then the rest best-score-first as
        'pending'. Per grasp: an ARROW from the gripper-base origin along +Z to
        the finger CONTACT point, a small SPHERE there, and a TEXT label with
        the candidate index (matches the 'grasp {i}' log lines). Marker ids are
        the candidate index, so a state change replaces that grasp's markers.
        Best-effort — viz must never break the grasp."""
        status = status or {}
        try:
            if not hasattr(self, '_grasp_marker_pub'):
                self._grasp_marker_pub = self.create_publisher(
                    MarkerArray, 'graspgen_markers',
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            order = sorted(range(len(poses)),
                           key=lambda i: (i not in status, -scores[i]))
            # Zero stamp = latest available TF; a now() stamp races the
            # pelvis->map link and RViz drops the markers.
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
                arrow.scale.x = 0.008                  # shaft diameter
                arrow.scale.y = 0.018                  # head diameter
                arrow.scale.z = 0.03                   # head length
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
                label.scale.z = 0.025                  # text height [m]
                label.text = str(i)
                label.color.r, label.color.g, label.color.b, label.color.a = r, g, b, 1.0
                arr.markers.append(label)
            self._grasp_marker_pub.publish(arr)
        except Exception as e:
            self.get_logger().warn(f'grasp marker publish failed: {e}')

    # ------------------------------------- battery parts: YOLO bounding box --
    def _yolo_box(self, obj, arm=None):
        """Bounding box (pixel xyxy) of a head-camera YOLO detection of `obj`,
        or None. Among several in view, picks the instance CLOSEST to the
        working arm's gripper (ranked in that arm's frame) when `arm` is given,
        else closest to the robot (pelvis origin)."""
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
        """Among detections of class `obj` in `bundle`, back-project each to a
        3-D point in `target_frame` and return the (detection, point) pair
        CLOSEST TO THE ORIGIN of that frame — the physically nearest instance,
        not the most confident. Unrangeable detections (no bbox depth) are
        skipped; (None, None) when nothing matches. Case-insensitive."""
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
        """Back-project a Detection's bounding box to a single (x, y, z) point
        in `target_frame`, or None: median over the central quarter of the box
        (avoids background depth at the edges). Head-camera aligned depth."""
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


# ============================ grasp-pose geometry ============================
def _pose_xyz(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)


def _pose_quat(pose):
    """Unit quaternion (x, y, z, w) of `pose`."""
    q = np.array([pose.orientation.x, pose.orientation.y,
                  pose.orientation.z, pose.orientation.w], dtype=float)
    return q / np.linalg.norm(q)


def _quat_angle(q0, q1):
    """Rotation angle [rad] between unit quaternions q0 and q1."""
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))


def _quat_slerp(q0, q1, t):
    """Spherical interpolation from q0 (t=0) to q1 (t=1), shortest arc."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    return (math.sin((1.0 - t) * theta) * q0 + math.sin(t * theta) * q1) / math.sin(theta)


def get_approach_pose(pose, approach_dist):
    """Translate `pose` along its OWN local +Z (the GraspGenX approach axis)
    by `approach_dist` metres, returning a NEW Pose with the orientation
    unchanged. Positive = deeper toward the object; negative = backed off
    (e.g. a pre-grasp standoff)."""
    z_axis = pose_to_matrix(pose)[:3, 2]
    out = copy.deepcopy(pose)
    out.position.x += approach_dist * float(z_axis[0])
    out.position.y += approach_dist * float(z_axis[1])
    out.position.z += approach_dist * float(z_axis[2])
    return out


def _stamp_tuple(stamp):
    """(sec, nanosec) of a builtin_interfaces/Time, for ordering stamps."""
    return (stamp.sec, stamp.nanosec)


def _step_color(dz):
    """BGR for the servo image's commanded-step arrow, coding its DEPTH
    component: RED driving in (+Z, down onto the part at the top-down
    approach), GREEN backing off, scaled over +/- VISUAL_SERVO_MAX_STEP_M."""
    t = (float(np.clip(dz / VISUAL_SERVO_MAX_STEP_M, -1.0, 1.0)) + 1.0) / 2.0
    return (0, int(round(255 * (1.0 - t))), int(round(255 * t)))


def _vec3_stamped(stamp, frame_id, v):
    """A geometry_msgs/Vector3Stamped from a 3-sequence (servo debug topics)."""
    msg = Vector3Stamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.vector.x = float(v[0])
    msg.vector.y = float(v[1])
    msg.vector.z = float(v[2])
    return msg


# ============================== debug TF helper ==============================
def _approach_target_tf(parent_frame, pose, stamp, child=TARGET_FRAME):
    """Build a debug TransformStamped for `pose` (in `parent_frame`) so the
    base node's broadcaster can publish it for RViz — by default the
    currently-targeted pre-grasp (TARGET_FRAME); pass `child` for others."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child
    t.transform.translation.x = pose.position.x
    t.transform.translation.y = pose.position.y
    t.transform.translation.z = pose.position.z
    t.transform.rotation = pose.orientation
    return t
