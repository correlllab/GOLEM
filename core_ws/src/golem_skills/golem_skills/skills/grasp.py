"""
Execute /skill/grasp: perception, a pluggable synthesis method, one executor.

The flow (_exec_grasp):

    prep -> detect_object (Gemini box) -> segment_object (SAM mask -> cloud)
         -> get_grasp_candidates (the selected grasp_synthesis method)
         -> per candidate, in the method's order: for a reach-checked method,
            skip it when the IK or the scene says the approach fails
            (grasp_reach.py); take off in Z, transit to the pre-grasp, then
            in along the grasp axis (or the visual servo); a candidate whose
            pre-grasp or contact is not reached is dropped and the next one
            tried
         -> close -> check the hand holds something

Perception and execution are the same for every synthesis method; only the
candidate list differs (grasp_synthesis/__init__.py has the methods and how a
goal selects one). The same path runs in the sim and on the robot, with no
switches: head-camera detection is grasp_detection.py, the visual servo
grasp_servo.py, the reach check grasp_reach.py, the RViz markers
grasp_markers.py.

SAFETY: every arm move is refused once golem_safety's heartbeat halts the arms
or the frame_task server aborts a move (SkillsBase.arm_halt_reason); the skill
then aborts instead of trying more candidates or retreating.
"""

import copy
import math
import time
from dataclasses import dataclass

import numpy as np

from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration as RclpyDuration

from custom_ros_messages.action import SkillGrasp

from ..base import (_Run, FRAME_TASK_STABLE_STOP_MSGS, SLOW_MODE_TIME_SCALE,
                    WORLD_FRAME)
from ..grasp_synthesis import METHODS, SynthesisInput, method_name
from ..grasp_synthesis.common import pose_at
from ..hands import (dex3_closing_joints, dex3_closure, dex3_closure_text,
                     dex3_grip_failure)
from ..perception_utils import (approach_pose, pose_quat, pose_xyz,
                                quat_geodesic, quat_slerp)
from .grasp_detection import GraspDetection, perception_name
from .grasp_markers import TARGET_FRAME, GraspMarkers
from .grasp_reach import GraspReach, scene_obstacles
from .grasp_servo import GraspServo


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
# Fraction of the in-along-the-axis move that must be covered before the
# gripper closes where the move stopped. In the battery sweep, a move stopped
# at its first step (nothing covered) left the part outside the fingers in 16
# of 17 trials, one stopped on its last step (5-5.6 of 7.5 cm) lifted the part
# in 17 of 22: the fingers had reached it. Less is a missed candidate.
AXIS_APPROACH_MIN_FRACTION = 0.25


# ---- grip check --------------------------------------------------------------
# mm: an aperture at or below this after the close means the pads met with
# nothing between them. An empty close reads 0.0-0.1 mm in the sim; unmeasured
# on the real H1-2.
EMPTY_GRIP_MAX_MM = 2.0
# mm: an aperture at or above this after the close means the fingers did not
# close: the jaws open to 106 mm on the real MAGPIE (108 in the sim), and
# every part the battery sweep held read under 60 mm. Its closes that stayed
# this open (fingertips blocked on the part's top) moved the part 0-4.9 cm on
# a 10 cm lift, dragged rather than held; held parts rose 5-10.5 cm. A part
# wider than this cannot be grasped.
OPEN_GRIP_MIN_MM = 90.0
# s of robot time to wait for a gripper state published after the close (the
# Magpie driver and the H1-2 sims publish at 10 Hz, dex_3's driver its finger
# joints at 50 Hz)
GRIP_STATE_WAIT_SEC = 1.0


# ---- reach budgets ----------------------------------------------------------
# servo_frame_to_world budgets, shared with pick_place: a long first IK move and
# many world-frame refinement passes; the iteration-0 unreachable check still
# drops dead candidates quickly
SERVO_DURATION_SEC = 15   # s, first (iteration-0) contact move
# s per approach move, a timeout rather than a trajectory time: generous so a
# long transit is never cut off; a rejected plan returns in well under a second
APPROACH_DURATION_SEC = 180.0
SERVO_MAX_ITER = 12       # world-frame refinement passes per pose
# m, rad: servo_frame_to_world tolerances of the CONTACT move, shared with
# pick_place; real-robot IK plus pelvis drift rarely settles a 6-DOF grasp pose
# tighter than this
SERVO_LIN_TOL = 0.0125
SERVO_ANG_TOL = 0.05
# m, rad: looser, for the PRE-GRASP approach only. The pre-grasp is a standoff
# the visual servo refines from, and a tighter tolerance rejects reachable
# candidates as unreachable and burns the pool.
APPROACH_LIN_TOL = 0.025
APPROACH_ANG_TOL = 0.10
# Every candidate is approached in three straight-line, unplanned, slow-mode
# stages (_straight_line_move):
#   1. take off straight up (pelvis +Z), holding the orientation, to
#      APPROACH_CLEARANCE_M above the higher of the hand and the pre-grasp;
#   2. transit to the pre-grasp, position and orientation interpolated;
#   3. in along the grasp's own +Z axis to the contact pose
#      (_move_in_along_axis), orientation held.
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


def axis_approach_closes(covered_m, depth_m):
    """Return whether an axis move that covered `covered_m` of its `depth_m` m may close there."""
    return depth_m <= 0.0 or covered_m >= AXIS_APPROACH_MIN_FRACTION * depth_m


def grip_failure(aperture_mm):
    """Return why the gripper holds nothing, given its aperture after the close (mm), else None."""
    if aperture_mm is None:
        return 'no gripper state arrived after the close'
    if aperture_mm <= EMPTY_GRIP_MAX_MM:
        return f'the gripper closed on nothing ({aperture_mm:.1f} mm)'
    if aperture_mm >= OPEN_GRIP_MIN_MM:
        return f'the fingers did not close ({aperture_mm:.1f} mm open)'
    return None


def line_waypoints(start, target, step_m, step_rad):
    """
    Return the waypoints of a straight line from Pose `start` to Pose `target`, `target` last.

    At most `step_m` [m] and `step_rad` apart, position interpolated and
    orientation slerped, in whatever frame the two poses share.
    """
    p0, p1 = pose_xyz(start), pose_xyz(target)
    q0, q1 = pose_quat(start), pose_quat(target)
    dist = float(np.linalg.norm(p1 - p0))
    angle = quat_geodesic(q0, q1)
    n = max(1, int(math.ceil(max(dist / step_m, angle / step_rad))))
    out = []
    for k in range(1, n + 1):
        t = float(k) / n
        pk = p0 + t * (p1 - p0)
        qk = quat_slerp(q0, q1, t)
        wp = Pose()
        wp.position = Point(x=float(pk[0]), y=float(pk[1]), z=float(pk[2]))
        wp.orientation = Quaternion(x=float(qk[0]), y=float(qk[1]),
                                    z=float(qk[2]), w=float(qk[3]))
        out.append(wp)
    return out


def approach_waypoints(cur, pregrasp, contact):
    """
    Return the executor's approach from hand pose `cur` as GraspReach.reach_check waypoints.

    The take-off, the transit to `pregrasp` and the axis steps to `contact`,
    laid out as _approach_pregrasp and _move_in_along_axis lay them out (all
    Poses in the pelvis frame), each with the tolerance its move must meet:
    the take-off and the pre-grasp the approach tolerances, the axis steps
    the contact tolerances until axis_approach_closes would close, and none
    after that. The pre-grasp and the axis steps are near the object, so
    they are the ones tested against the scene.
    """
    approach_tol = (APPROACH_LIN_TOL, APPROACH_ANG_TOL)
    out = []
    lift_z = max(cur.position.z, pregrasp.position.z) + APPROACH_CLEARANCE_M
    start = cur
    if lift_z - cur.position.z > 1e-3:
        start = copy.deepcopy(cur)
        start.position.z = lift_z
        up = line_waypoints(cur, start, LINEAR_APPROACH_SEG_M, LINEAR_APPROACH_SEG_RAD)
        for k, wp in enumerate(up, 1):
            out.append((wp, 'take-off', approach_tol if k == len(up) else None, False))
    transit = line_waypoints(start, pregrasp, LINEAR_APPROACH_SEG_M,
                             LINEAR_APPROACH_SEG_RAD)
    for k, wp in enumerate(transit, 1):
        last = k == len(transit)
        out.append((wp, 'pre-grasp' if last else f'transit {k}/{len(transit)}',
                    approach_tol if last else None, last))
    depth = float(np.linalg.norm(pose_xyz(contact) - pose_xyz(pregrasp)))
    n = max(1, int(math.ceil(depth / AXIS_APPROACH_STEP_M)))
    for k in range(1, n + 1):
        must = not axis_approach_closes(depth * (k - 1) / n, depth)
        out.append((approach_pose(pregrasp, depth * k / n), f'axis step {k}/{n}',
                    (SERVO_LIN_TOL, SERVO_ANG_TOL) if must else None, True))
    return out


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


@dataclass
class _Contact:
    """Hold where a candidate's approach left the gripper for the close."""

    # index of the candidate in Candidates
    idx: int
    # pelvis-frame contact pose (None when the servo's TF lookup failed)
    pose_p: object
    # the same in WORLD_FRAME, or None without world anchoring
    pose_w: object


class GraspSkill(GraspDetection, GraspServo, GraspMarkers, GraspReach):
    """Provide /skill/grasp's execute callback and its pipeline to SkillsNode."""

    def _exec_grasp(self, gh):
        """
        Grasp goal.target_object and leave it held (no lift); return the result.

        prep -> detect_object -> segment_object -> get_grasp_candidates ->
        _reach_candidate -> close -> grip check (grip_failure). The method is
        the `grasp_method` parameter, or `top_down` when the goal sets it
        (_resolve_grasp_method).

        Optional goal fields (SkillGrasp.action; zero / false = off):
          top_down: use the top_down synthesis method for this goal;
          min_downward_pitch_deg: keep only grasps approaching at least this
            far below horizontal;
          visual_servo: replace the contact move with servo().

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
        err = self._robot_refusal(method, goal)
        if err:
            return run.abort(err)
        obj = perception_name(goal.target_object)
        self.get_logger().info(
            f'grasp: {obj!r} with synthesis method {method.name!r}'
            f'{" + visual servo" if goal.visual_servo else ""}')
        if method.reach_check:
            # built in the background while perception runs
            self._reach_model_start()

        if not run.phase('detect_object', 0.0):
            return run.result
        # Arms to the robot's prep config ('prep' on the H1-2) first, out of
        # the head camera's view (an arm across the object corrupts the box,
        # mask and cloud); planned, because the arms start wherever the last
        # skill left them. The only failure without a retreat: the skill has
        # not moved the arm yet.
        prep = self.robot.prep_config
        if not self.goto_named_config(prep, plan=True, slow_mode=True,
                                      duration_sec=30.0, result_timeout=90.0,
                                      outer_gh=gh):
            return run.abort(f'move to {prep!r} before detection failed')

        def fail(message):
            """Lift the hand clear of the workspace, then abort with `message`."""
            self._retreat_after_failure(arm, gh)
            return run.abort(message)

        box = self.detect_object(obj, run, gh, arm)
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return fail('detection canceled or timed out')
        obj_cloud, scene, err = self.segment_object(obj, box, gh)
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
        # reached without an IK solve, with the other arm at prep. Other
        # methods start from prep.
        stage_cfg = self.robot.staging_configs[arm]
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
        reset_cfg = stage_cfg if method.staged else prep
        obstacles = scene_obstacles(scene, obj_cloud) if method.reach_check else None
        contact, err = self._reach_candidate(run, gh, goal, obj, arm, cands,
                                             targets, reset_cfg,
                                             method.reach_check, obstacles)
        if err:
            # A failed walk ends with the arms reset to reset_cfg and a halted
            # arm must not move, so only a servo cut short retreats.
            if contact is not None:
                return fail(err)
            return run.abort(err)

        if not run.phase('close_gripper', 0.75):
            return run.result
        self.get_clock().sleep_for(RclpyDuration(seconds=PRE_CLOSE_SETTLE_SEC))
        closed_at = self.robot_now()
        if not self.close_gripper(arm):
            return fail('gripper close failed')
        self.get_clock().sleep_for(RclpyDuration(seconds=POST_CLOSE_SETTLE_SEC))
        reason, held = self._grip_check(arm, closed_at)
        if reason is not None:
            return fail(f'grasp of {obj!r} with {method.name} candidate '
                        f'{cands.labels[contact.idx]} failed: {reason}')

        # the CONTACT pose, where the hand actually is, in the world snapshot's
        # frame when there is one
        if contact.pose_w is not None:
            out_pose, out_frame = contact.pose_w, WORLD_FRAME
        else:
            out_pose, out_frame = contact.pose_p, 'pelvis'
        self._last_grasp_outcome = GraspOutcome(
            arm=arm, pose=out_pose, frame=out_frame, score=cands.scores[contact.idx])

        return run.succeed(
            f'grasped {obj!r} with {method.name} candidate '
            f'{cands.labels[contact.idx]} (score {cands.scores[contact.idx]:.2f}, '
            f'{held})')

    def _grip_check(self, arm, closed_at):
        """
        Return (why `arm`'s hand holds nothing, or None; its state for the result message).

        `closed_at` is the robot time the close was sent; only hand state
        published after it counts. The Magpie's aperture (grip_failure), or
        how far the Dex3-1's fingers closed (hands.dex3_grip_failure).
        """
        if self.robot.hand == 'dex3':
            positions = self.joint_positions_since(
                dex3_closing_joints(arm), closed_at, GRIP_STATE_WAIT_SEC)
            reason = dex3_grip_failure(arm, positions)
            held = ('fingers closed ' + dex3_closure_text(dex3_closure(arm, positions))
                    if positions is not None else '')
            return reason, held
        aperture = self.gripper_aperture_since(arm, closed_at, GRIP_STATE_WAIT_SEC)
        reason = grip_failure(aperture)
        return reason, (f'gripper at {aperture:.1f} mm' if aperture is not None else '')

    def _resolve_grasp_method(self, goal):
        """Return (GraspMethod, None) for the goal, or (None, reason) for an unknown name."""
        name = method_name(goal.top_down,
                           self.get_parameter('grasp_method').value)
        method = METHODS.get(name)
        if method is None:
            return None, (f'unknown grasp_method {name!r}; expected one of '
                          f'{sorted(METHODS)}')
        return method, None

    def _robot_refusal(self, method, goal):
        """
        Return why this robot cannot run `method` or the goal's options, or None.

        Checked before the arm moves: a GraspGenX method needs a gripper
        GraspGenX has a model of, and goal.visual_servo needs hand cameras.
        """
        name = self.robot.name
        if method.graspgen and self.robot.graspgen_gripper is None:
            return (f'grasp_method {method.name!r} plans with GraspGenX, which has '
                    f'no model of the {name} hand; use one of '
                    f'{sorted(m for m, v in METHODS.items() if not v.graspgen)}')
        if goal.visual_servo and not self.robot.hand_cameras:
            return f'visual_servo needs hand cameras, and the {name} has none'
        return None

    def get_grasp_candidates(self, method, goal, obj, arm, obj_cloud, scene):
        """Run `method` on the segmented object; return (Candidates, None) or (None, reason)."""
        inp = SynthesisInput(
            obj=obj, arm=arm, cloud=obj_cloud, scene=scene,
            min_pitch_deg=float(goal.min_downward_pitch_deg or 0.0),
            plan_grasp=self.plan_grasp, log=self.get_logger(),
            gripper_name=self.robot.graspgen_gripper)
        cands, err = method.synthesize(inp)
        if err is None and not cands.grasps:
            err = f'{method.name} produced no grasp for {obj!r}'
        return (None, err) if err else (cands, None)

    def _snapshot_targets(self, cands, centroid_p):
        """
        Freeze every candidate's grasp and pre-grasp (and the centroid) into the world frame.

        Done once, from the current pelvis frame, so the targets stay put in
        the world as the pelvis drifts, and every waypoint is re-resolved into
        the live pelvis frame before it is commanded. Needs WORLD_FRAME in the
        TF tree (navigation / FAST-LIO running); without it `have_world` is
        False and the raw pelvis poses are driven without drift compensation.
        """
        grasps_p = [approach_pose(g.pose, GRASP_OFFSET) for g in cands.grasps]
        approaches_p = [approach_pose(g, -APPROACH_DIST) for g in grasps_p]
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

    def _reach_candidate(self, run, gh, goal, obj, arm, cands, targets, reset_cfg,
                         check_reach=False, obstacles=None):
        """
        Bring the gripper to the first candidate it can close on.

        Returns (_Contact, None), or (_Contact or None, reason) on failure.

        Walks the candidates in the method's order, moving the arm for at most
        cands.attempts of them. With `check_reach`, a candidate whose approach
        fails the reach check (grasp_reach.py; `obstacles` the scene points
        near the object) is skipped without moving the arm. Then the pre-grasp
        (_approach_pregrasp), then the visual servo with goal.visual_servo,
        else the move in along the grasp axis (_move_in_along_axis). A
        candidate is dropped when its pre-grasp is not reached or its axis
        move stops too early; the arms then return to the named config
        `reset_cfg` by a PLANNED joint-space move, because a failed Cartesian
        approach can leave a joint pinned at its limit, from which no further
        Cartesian move makes progress. A visual servo that does not converge
        closes where it stopped (best-effort). On failure the first element
        is a _Contact when the caller should retreat before reporting (a
        cancel or timeout during the servo), else None. Each candidate's
        state (trying, failed, chosen) is republished on 'graspgen_markers'.
        """
        width_mm = cands.gripper_width * 1000.0
        status = {}
        attempts = cands.attempts or len(cands.grasps)
        tried = skipped = 0

        def mark(i, state):
            """Set candidate `i`'s marker state and republish the markers."""
            status[i] = state
            self._publish_grasp_markers(targets.frame, targets.grasps_p,
                                        cands.scores, status)

        for i in range(len(cands.grasps)):
            if tried >= attempts:
                break
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return None, 'canceled or timed out during approach'
            if check_reach:
                why = self._check_candidate_reach(arm, targets, i, obstacles)
                if why is not None:
                    skipped += 1
                    status[i] = 'failed'
                    self.get_logger().info(
                        f'grasp {i} ({cands.labels[i]}) skipped by the reach check: {why}')
                    continue
            tried += 1
            mark(i, 'trying')
            self._publish_target_tf(targets, targets.approaches_w[i],
                                    targets.approaches_p[i])
            self.get_logger().info(
                f'grasp {i} ({cands.labels[i]}) for {obj!r}: tier '
                f'{cands.tiers[i]}, score {cands.scores[i]:.2f}, '
                f'width {width_mm:.1f}mm')
            contact, why = None, 'pre-grasp unreachable'
            if self._approach_pregrasp(self.robot.grasp_frames[arm], targets.approaches_p[i],
                                       run, gh, pregrasp_w=targets.approaches_w[i]):
                if goal.visual_servo:
                    contact, err = self._servo_contact(run, gh, arm, obj, targets, i)
                    if err:
                        mark(i, 'failed')
                        return contact, err
                else:
                    contact, why = self._move_in_along_axis(gh, arm, targets, i)
            if contact is not None:
                mark(i, 'chosen')
                return contact, None
            mark(i, 'failed')
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return None, 'canceled or timed out during approach'
            halt = self.arm_halt_reason()
            if halt is not None:
                return None, f'arm halted during the approach: {halt}'
            self.get_logger().warn(f'grasp {i}: {why}; resetting to {reset_cfg!r}')
            if not self.goto_named_config(reset_cfg, plan=True, slow_mode=True,
                                          duration_sec=30.0, result_timeout=90.0,
                                          outer_gh=gh):
                halt = self.arm_halt_reason()
                return None, (f'arm halted during the reset to {reset_cfg!r}: {halt}'
                              if halt is not None else
                              f'reset to {reset_cfg!r} after grasp {i} failed')
        checked = f', {skipped} failed the reach check' if check_reach else ''
        return None, (f'no reachable grasp for {obj!r} '
                      f'(tried {tried}{checked}, of {cands.total})')

    def _check_candidate_reach(self, arm, targets, idx, obstacles):
        """Return why candidate `idx`'s approach fails the reach check, or None."""
        frame = self.robot.grasp_frames[arm]
        cur = self._frame_pose_in_pelvis(frame)
        if cur is None:
            return None
        contact = approach_pose(targets.grasps_p[idx],
                                -float(self.get_parameter('contact_offset').value))
        t0 = time.monotonic()
        why = self.reach_check(
            frame, approach_waypoints(cur, targets.approaches_p[idx], contact),
            obstacles)
        self.get_logger().info(
            f'grasp {idx}: reach check {"failed" if why else "passed"} in '
            f'{time.monotonic() - t0:.1f}s')
        return why

    def _servo_contact(self, run, gh, arm, obj, targets, idx):
        """
        Run the visual servo from candidate `idx`'s pre-grasp; return (_Contact, reason or None).

        A servo that does not converge leaves the gripper where it stopped,
        which becomes the contact (best-effort). A cancel or timeout returns
        the reason with that contact, so the caller retreats.
        """
        pose_p, err = self.servo(run, gh, arm, obj, targets, idx)
        if err:
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return _Contact(idx, None, None), err
            self.get_logger().warn(
                f'grasp: visual servo did not converge ({err}) — closing '
                'anyway at the current pose (best-effort)')
            pose_p = self._frame_pose_in_pelvis(self.robot.grasp_frames[arm])
        pose_w = (self._transform_pose(pose_p, 'pelvis', WORLD_FRAME)
                  if targets.have_world and pose_p is not None else None)
        return _Contact(idx, pose_p, pose_w), None

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
        dist = float(np.linalg.norm(pose_xyz(target) - pose_xyz(cur)))
        angle = quat_geodesic(pose_quat(cur), pose_quat(target))
        waypoints = line_waypoints(cur, target, step_m, step_rad)
        n = len(waypoints)
        self.get_logger().info(
            f'grasp: {label}: {dist * 100:.1f}cm / {math.degrees(angle):.0f}deg '
            f'in {n} unplanned{" slow" if slow_mode else ""} step(s)'
            f'{f", tracking {WORLD_FRAME}" if target_w is not None else ""}')
        for k, wp in enumerate(waypoints, 1):
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return False
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

    def _move_in_along_axis(self, gh, arm, targets, idx):
        """
        Run approach stage 3 from candidate `idx`'s pre-grasp; return (_Contact or None, why).

        The contact target is the grasp backed off by the node's
        `contact_offset` parameter, reached in AXIS_APPROACH_STEP_M straight
        steps with the grasp orientation held, slow and unplanned (planning
        would fight the approach into the object). The move stops at the first
        step that misses its tolerance; the gripper closes there when
        axis_approach_closes allows it, else the candidate is missed. With
        world anchoring each step is laid out along the world-frame grasp axis
        and re-resolved into the live pelvis frame, and a final world-frame
        correction removes the remaining pelvis drift. A cancel or an arm halt
        misses the candidate.
        """
        frame = self.robot.grasp_frames[arm]
        stop_short = float(self.get_parameter('contact_offset').value)
        contact_p = approach_pose(targets.grasps_p[idx], -stop_short)
        contact_w = (approach_pose(targets.grasps_w[idx], -stop_short)
                     if targets.have_world else None)
        self._publish_target_tf(targets, contact_w, contact_p)
        start = targets.approaches_p[idx]
        depth = float(np.linalg.norm(pose_xyz(contact_p) - pose_xyz(start)))
        if depth < 0.01:
            self.get_logger().warn(
                f'grasp: axis approach is only {depth * 100:.1f}cm; the '
                'gripper closes without approaching along the grasp axis')
        n = max(1, int(math.ceil(depth / AXIS_APPROACH_STEP_M)))
        for k in range(1, n + 1):
            if gh.is_cancel_requested:
                return None, 'canceled during the axis approach'
            step = approach_pose(start, depth * k / n)
            if targets.have_world:
                step = self._live_pelvis_pose(approach_pose(
                    targets.approaches_w[idx], depth * k / n)) or step
            if self.move_frame_to(
                    frame, step, outer_gh=gh, duration_sec=AXIS_APPROACH_STEP_SEC,
                    do_plan=False, slow_mode=True,
                    lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL,
                    stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                    label=f'in along the grasp axis {k}/{n}'):
                continue
            halt = self.arm_halt_reason()
            if halt is not None:
                return None, f'arm halted during the contact approach: {halt}'
            covered = depth * (k - 1) / n
            where = f'{covered * 100:.1f} of {depth * 100:.1f}cm'
            if not axis_approach_closes(covered, depth):
                return None, f'axis approach stopped at step {k}/{n} ({where})'
            self.get_logger().warn(
                f'grasp: axis approach stopped at step {k}/{n} ({where}); closing here')
            return _Contact(idx, contact_p, contact_w), None
        if targets.have_world and self.arm_halt_reason() is None:
            self.servo_frame_to_world(
                frame, contact_w, contact_p, outer_gh=gh,
                duration_sec=SERVO_DURATION_SEC * SLOW_MODE_TIME_SCALE,
                max_iter=SERVO_MAX_ITER,
                lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL, do_plan=False,
                slow_mode=True, stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                label='world-frame contact correction')
        halt = self.arm_halt_reason()
        if halt is not None:
            return None, f'arm halted during the contact approach: {halt}'
        return _Contact(idx, contact_p, contact_w), None

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
        frame = self.robot.grasp_frames[arm]
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

    def _closest_arm(self, centroid_p):
        """
        Return the arm whose gripper is nearest the object centroid (pelvis xyz).

        Uses live TF; without both hand TFs, the side of the pelvis midline the
        object is on (+Y = left).
        """
        dists = {}
        for arm, frame in self.robot.grasp_frames.items():
            p = self._transform_pose(pose_at(centroid_p), 'pelvis', frame)
            if p is not None:
                dists[arm] = math.sqrt(p.position.x ** 2 + p.position.y ** 2
                                       + p.position.z ** 2)
        if len(dists) == len(self.robot.grasp_frames):
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
