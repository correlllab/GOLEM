"""
Execute /skill/pick_place: grasp an object, carry it above a place target and drop it.

prep -> detect the place target -> grasp the object (in-process /skill/grasp)
-> lift it clear -> park the graspgenx frame top-down above the place target
-> open the gripper.

The place has no orientation search and no descent to a rest height: the place
pose is the top_down grasp orientation (grasp_synthesis.top_down.top_down_pose:
forward azimuth, level wrist, pitched TOP_DOWN_PITCH_DEG down) with the finger
contact point PLACE_HEIGHT above the place target's top face, and the object
is dropped from there.

The place point is detected BEFORE the long grasp and anchored in WORLD_FRAME,
so pelvis drift during the grasp does not move it; at place time it is
re-resolved into the live pelvis frame and the pose is driven with
servo_frame_to_world's drift compensation. Without a world TF the skill uses
the detect-time pelvis point, uncompensated, as the grasp skill does without
odometry.

The post-grasp lift is world-anchored the same way (the LIFT_* constants).
"""

import copy

import numpy as np

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from tf2_ros import TransformException

from custom_ros_messages.action import SkillGrasp, SkillPickPlace

from ..base import _Run, GRASP_FRAMES, WORLD_FRAME
from ..perception_utils import (matrix_to_pose, pose_to_matrix,
                                transform_to_matrix)
from ..grasp_synthesis.common import pose_at
from ..grasp_synthesis.top_down import top_down_pose
from .grasp import (_approach_target_tf,
                    SERVO_MAX_ITER, SERVO_LIN_TOL, SERVO_ANG_TOL,
                    APPROACH_DURATION_SEC, FRAME_TASK_STABLE_STOP_MSGS)


# m, the drop height: the released object's contact point above the place
# target's top face
PLACE_HEIGHT = 0.10
# m below the place cloud's top percentile counted as its top face when
# averaging the drop point; excludes side-face points that would skew it
TOP_FACE_BAND = 0.015
# s of the skill budget kept for the place: the inner grasp's timeout is the
# remaining budget minus this, so a slow Gemini detection cannot starve the place
PLACE_RESERVE_SEC = 60.0
# m, total straight-up lift after the grasp closes
LIFT_HEIGHT = 0.10
# m per lift rung. The rise is a ladder of LIFT_STEP moves rather than one
# move, so the arm clears the pick site in short, interruptible hops. The
# ladder is anchored once at the executed grasp (outcome.pose) and every rung
# is an ABSOLUTE offset from that anchor, so rung errors do not accumulate and
# each rung pulls the hand back onto the grasp's own vertical.
#
# The anchor is in WORLD_FRAME whenever the grasp had a world TF, so +Z is
# gravity up and each rung is re-servoed against the measured world pose
# (_servo_pose). A pelvis +Z pull would follow the torso, which pitches and
# sways under the balancing lower body, and drag the held object sideways.
LIFT_STEP = 0.05
# s of commanded motion per rung
LIFT_STEP_DURATION_SEC = 4.0
# The lift drives the PINCH-POINT frame, not the GraspGenX gripper-base frame
# the targets are computed for. The two are GRIPPER_BASE_TO_CONTACT_M (0.1146 m)
# apart along the approach, so nulling the error at the base would let a wrist
# orientation residual swing the fingertips through ~0.1146 x ang_tol
# unnoticed; on a small part held in the fingertips that arc rolls it out of
# the jaws.
PINCH_FRAMES = {'left': 'left_grasp_frame', 'right': 'right_grasp_frame'}
# debug TF (RViz) of the detected place point, then the commanded place pose
PLACE_TARGET_FRAME = 'pick_place_target_frame'


class PickPlaceSkill:
    """Provide /skill/pick_place's execute callback to SkillsNode."""

    def _exec_pick_place(self, gh):
        """
        Pick goal.target_object and drop it on goal.place_target; return the result.

        Optional goal fields (SkillPickPlace.action; zero / empty = off):
          arm: '' or 'none' lets the grasp skill pick the nearest arm, which
            the GraspOutcome reports back;
          top_down, visual_servo: forwarded to /skill/grasp, which also reads
            this node's `grasp_method` parameter.

        PLACE_TARGET_FRAME is broadcast while the goal runs and dropped when
        it ends, however it ends.
        """
        try:
            return self._pick_place(gh)
        finally:
            self.drop_tf(PLACE_TARGET_FRAME)

    def _pick_place(self, gh):
        """Run the pick_place goal's body (see _exec_pick_place)."""
        goal = gh.request
        run = _Run(self, gh, SkillPickPlace, 'pick_place')
        arm = self._validated_arm(goal)
        if arm is None and goal.arm.strip().lower() not in ('', 'none'):
            return run.abort(f'invalid arm {goal.arm!r}')
        obj, place = goal.target_object, goal.place_target
        if not place.strip():
            return run.abort('empty place_target')

        def fail(message):
            """
            Lift the hand clear (mid-carry, over the target), then abort with `message`.

            `arm` is read at call time: it stays None until the inner grasp
            reports which hand it used.
            """
            self._retreat_after_failure(arm, gh)
            return run.abort(message)

        if not run.phase('detect_place_target', 0.0):
            return run.result
        # Arms out of the head camera's view before the detection image (the
        # grasp skill's own 'prep' is then a no-op). The skill has not moved
        # the arm yet, so there is no retreat.
        if not self.goto_named_config('prep', plan=True, slow_mode=True,
                                      duration_sec=30.0, result_timeout=90.0,
                                      outer_gh=gh):
            return run.abort("move to 'prep' before detection failed")
        place_cloud, _, err = self.detect_object_cloud(place, run, gh)
        if err:
            return fail(f'place target: {err}')
        # (x, y, z_top), pelvis frame
        top_p = _top_face_point(place_cloud)
        top_w = self._transform_pose(pose_at(top_p), 'pelvis', WORLD_FRAME)
        self.publish_tf(_approach_target_tf(
            WORLD_FRAME if top_w is not None else 'pelvis',
            top_w if top_w is not None else pose_at(top_p),
            self.get_clock().now().to_msg(), child=PLACE_TARGET_FRAME))
        self.get_logger().info(
            f'pick_place: {place!r} top face at pelvis '
            f'({top_p[0]:.3f}, {top_p[1]:.3f}, {top_p[2]:.3f}), '
            f'world anchor {"ok" if top_w is not None else "UNAVAILABLE"}')

        if not run.phase('grasp_object', 0.2):
            return run.result
        outcome, err = self._grasp_via_skill(goal, obj, arm, run, gh)
        if err:
            return fail(err)
        # the hand that grasped drives the rest
        arm = outcome.arm

        # The lift rungs are computed for the GraspGenX frame the grasp pose is
        # in, then restated for the pinch-point frame that holds the object
        # (to_pinch); see the LIFT_* constants.
        lift_frame = PINCH_FRAMES[arm]
        T_off = self._pinch_frame_offset(arm)
        if T_off is None:
            return fail(f'TF {GRASP_FRAMES[arm]} -> {lift_frame} unavailable, '
                        'so the post-grasp lift target is unknown')

        def to_pinch(pose_gx):
            """
            Restate a target computed for the GraspGenX frame as the pinch-point frame's target.

            The offset is body-fixed, so this holds in the world and pelvis
            frames alike.
            """
            return matrix_to_pose(pose_to_matrix(pose_gx) @ T_off)

        world_ok = outcome.frame == WORLD_FRAME
        n_rungs = max(1, int(round(LIFT_HEIGHT / LIFT_STEP)))
        self.get_logger().info(
            f'pick_place: lifting {lift_frame} {LIFT_HEIGHT * 100:.0f}cm up '
            f'({"gravity" if world_ok else "pelvis"} +Z) in {n_rungs} x '
            f'{LIFT_STEP * 100:.0f}cm rungs from the executed grasp pose'
            + ('' if world_ok else ' — NO world anchor, drift compensation OFF'))
        for k in range(1, n_rungs + 1):
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return fail('canceled or timed out during post-grasp lift')
            dz = min(k * LIFT_STEP, LIFT_HEIGHT)
            rung = copy.deepcopy(outcome.pose)
            rung.position.z += dz
            if not self._servo_pose(
                    lift_frame, to_pinch(rung), world_ok, gh, do_plan=False,
                    duration_sec=LIFT_STEP_DURATION_SEC,
                    stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                    label=f'lift rung {k}/{n_rungs} (+{dz * 100:.0f}cm Z)'):
                return fail(f'post-grasp lift failed at {dz * 100:.0f}cm of '
                            f'{LIFT_HEIGHT * 100:.0f}cm')

        if not run.phase('place', 0.7):
            return run.result
        # The world-anchored place point, re-resolved into the live pelvis
        # frame (the pelvis has drifted since detection); the place pose puts
        # the contact point PLACE_HEIGHT above it.
        top_now = top_p
        if top_w is not None:
            p = self._transform_pose(top_w, WORLD_FRAME, 'pelvis')
            if p is None:
                return fail(f'{WORLD_FRAME} -> pelvis TF lost before the place')
            top_now = np.array([p.position.x, p.position.y, p.position.z])
        else:
            self.get_logger().warn(
                'pick_place: no world anchor; placing in the pelvis frame '
                'with NO drift compensation')
        place_p = top_down_pose(top_now + [0.0, 0.0, PLACE_HEIGHT])
        place_w = (self._transform_pose(place_p, 'pelvis', WORLD_FRAME)
                   if top_w is not None else None)
        self.publish_tf(_approach_target_tf(
            WORLD_FRAME if place_w is not None else 'pelvis',
            place_w if place_w is not None else place_p,
            self.get_clock().now().to_msg(), child=PLACE_TARGET_FRAME))
        self.get_logger().info(
            f'pick_place: {GRASP_FRAMES[arm]} to the top-down release pose '
            f'{PLACE_HEIGHT * 100:.0f}cm above {place!r}')
        # planned: the hand carries the object across the workspace
        if not self.servo_frame_to_world(
                GRASP_FRAMES[arm], place_w, place_p, outer_gh=gh,
                duration_sec=APPROACH_DURATION_SEC, max_iter=SERVO_MAX_ITER,
                lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL,
                label='carry to release pose above place target'):
            return fail(f'release pose above {place!r} unreachable')
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return fail('canceled or timed out during place')

        if not run.phase('release', 0.9):
            return run.result
        if not self.open_gripper(arm):
            return fail('gripper open failed')
        return run.succeed(
            f'placed {obj!r} on {place!r} (released {PLACE_HEIGHT * 100:.0f}cm '
            f'above, grasp score {outcome.score:.2f})')

    def _grasp_via_skill(self, goal, obj, arm, run, gh):
        """
        Grasp `obj` through this node's own /skill/grasp: (GraspOutcome, None) or (None, reason).

        top_down and visual_servo are copied explicitly from the pick_place
        goal (an unset field defaults to false). `arm` None lets the grasp
        skill choose and report it on the outcome. The inner feedback is
        proxied into the 'grasp_object' phase, and canceling this goal cancels
        the inner one (_send_action's outer_gh).
        """
        inner_timeout = max(10.0, run.remaining() - PLACE_RESERVE_SEC)
        inner = SkillGrasp.Goal()
        inner.target_object = obj
        inner.arm = arm or ''
        inner.top_down = goal.top_down
        inner.visual_servo = goal.visual_servo
        if inner_timeout == float('inf'):
            # this goal has no deadline (timeout 0), and a zero inner Duration
            # means the same
            inner.timeout = Duration(sec=0, nanosec=0)
        else:
            whole = int(inner_timeout)
            inner.timeout = Duration(
                sec=whole, nanosec=int(round((inner_timeout - whole) * 1e9)))

        def _proxy(feedback_msg):
            """Republish the inner grasp's progress as this goal's 0.2-0.7 'grasp_object' phase."""
            run.feedback.phase = 'grasp_object'
            run.feedback.progress = (
                0.2 + 0.5 * float(feedback_msg.feedback.progress))
            gh.publish_feedback(run.feedback)

        resp = self._send_action(self.grasp_skill_cli, inner, feedback_cb=_proxy,
                                 result_timeout=inner_timeout + 15.0, outer_gh=gh)
        if (resp is None or resp.status != GoalStatus.STATUS_SUCCEEDED
                or not resp.result.success):
            msg = (resp.result.message if resp is not None
                   else 'goal failed or timed out')
            return None, f'grasp failed: {msg}'
        outcome = getattr(self, '_last_grasp_outcome', None)
        if outcome is None:
            return None, f'grasp of {obj!r} returned no outcome for placement'
        return outcome, None

    def _pinch_frame_offset(self, arm):
        """
        Return the 4x4 pose of `arm`'s pinch-point frame in its GraspGenX frame, or None.

        Both are fixed URDF children of the wrist yaw link, so the offset is
        constant; it is read from TF once per goal (None until the robot
        description is up). A target T for the GraspGenX frame becomes
        T @ offset for the pinch-point frame.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                GRASP_FRAMES[arm], PINCH_FRAMES[arm], Time(),
                timeout=RclpyDuration(seconds=1.0))
        except TransformException:
            return None
        return transform_to_matrix(tf.transform)

    def _servo_pose(self, frame, pose, world_ok, gh, do_plan,
                    duration_sec=LIFT_STEP_DURATION_SEC, slow_mode=False,
                    label='', stable_stop_msgs=None):
        """
        Drive URDF `frame` to `pose` with the grasp skill's servo tolerances; return reached.

        With `world_ok`, `pose` is in WORLD_FRAME and servo_frame_to_world
        compensates drift (the target is re-resolved into the live pelvis
        frame every pass and the frame's world pose measured back); otherwise
        `pose` is a pelvis-frame Pose driven directly.
        """
        if world_ok:
            world_pose = pose
            fallback = self._transform_pose(pose, WORLD_FRAME, 'pelvis')
            if fallback is None:
                self.get_logger().warn(
                    f'pick_place: {WORLD_FRAME} -> pelvis TF unavailable')
                return False
        else:
            world_pose, fallback = None, pose
        return self.servo_frame_to_world(
            frame, world_pose, fallback, outer_gh=gh,
            duration_sec=duration_sec, max_iter=SERVO_MAX_ITER,
            lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL, do_plan=do_plan,
            slow_mode=slow_mode, label=label,
            stable_stop_msgs=stable_stop_msgs)


def _top_face_point(cloud):
    """
    Return (x, y, z_top) of `cloud`'s top face, in the cloud's own (z-up) frame.

    z_top is the 95th-percentile z (robust to depth speckle); x and y average
    only the points within TOP_FACE_BAND below it, so side-face points do not
    skew the drop point. An empty band falls back to the whole cloud.
    """
    z = cloud[:, 2]
    z_top = float(np.percentile(z, 95))
    top = cloud[z > z_top - TOP_FACE_BAND]
    if len(top) == 0:
        top = cloud
    return np.array([float(np.mean(top[:, 0])), float(np.mean(top[:, 1])), z_top])
