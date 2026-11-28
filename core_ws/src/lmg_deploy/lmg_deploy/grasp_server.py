"""Serve /grasp_frame_target for marl_deploy_node and publish /{left,right}_grasp_frame_pose.

The action and the topics are the interface every arm controller of the robot serves alike (golem_upper_ik's
frame_task_server when it runs, this node in the loco-manipulation game, never both); GraspFrameTarget.action's
comments are the contract.

Threads: the action server runs on a node and a multi-threaded executor of its own, on threads of their own, so a
goal's execute callback (which resolves its targets through TF, then waits for its end) never blocks the control
loop, nor the cancels and goals that come in meanwhile.
marl_deploy_node's control tick owns the goals: step() starts the goal at the head of the queue, steps it, and ends
it; the execute callback publishes its feedback and ends its handle with the outcome. The queue and each goal's
outcome and errors are shared under one lock.

A goal's targets are each arm's grasp frame (grasp_goal.GRASP_FRAMES) in the pelvis frame, or in a world frame
resolved into the pelvis frame through TF once, when the goal is accepted; an arm whose header.frame_id is empty keeps
its command. The goals run one at a time, in the order they came. A running goal moves the targeted arms' wrist goals
to put the grasp frames on their targets (grasp_goal.GraspFrames), succeeds once every targeted grasp frame has stayed
within the tolerance for settle_s (robot time), and aborts at its timeout (robot time from its start; 0: the
default); a cancel or the timeout stops the targeted arms where they are (MarlController.hold_arms). abort_all()
aborts the running and the queued goals (a fault, a stale /lowstate, the episode's restart). ``plan`` is accepted and
ignored: the policy moves directly. The result's outcome is REACHED on success, NOT_REACHED at the timeout, CANCELED
on a cancel (running or queued) and FAILED for every other abort (a goal refused or that cannot run, abort_all()).
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from custom_ros_messages.action import GraspFrameTarget
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from lmg_deploy.grasp_goal import ARMS, GRASP_FRAMES, GoalJudge, GraspFrames, pose_error
from lmg_deploy.rotations import quat_apply, quat_mul, quat_unique, wxyz_to_xyzw, xyzw_to_wxyz

Result = GraspFrameTarget.Result  # every end sets its outcome: REACHED, NOT_REACHED, CANCELED or FAILED
GRASP_ACTION = "/grasp_frame_target"
GRASP_POSE_TOPICS = ("/left_grasp_frame_pose", "/right_grasp_frame_pose")
POSE_FRAME = "pelvis"      # the frame every target is held in and every pose is published in
FEEDBACK_PERIOD_S = 0.04   # s of steady time between a waiting goal's feedback messages (25 Hz)
TF_WAIT_S = 1.0            # s of steady time a goal with a world-frame target listens to TF for its transforms
TF_MAX_AGE_S = 0.5         # s of robot time: a world-frame target whose newest transform is older is refused
TF_POLL_S = 0.02           # s of steady time between TF lookups while waiting


class GoalError(Exception):
    """Signal a /grasp_frame_target goal that cannot run; its message says why."""


@dataclass
class _Goal:
    """Hold one accepted goal: its handle, its pelvis-frame grasp targets and what the control tick wrote back."""

    handle: object
    targets: dict                    # arm index -> (position m, quaternion wxyz), pelvis frame
    timeout_s: float                 # s of robot time from its start
    done: threading.Event = field(default_factory=threading.Event)
    outcome: tuple | None = None     # (GraspFrameTarget.Result outcome constant, message)
    errors: list | None = None       # per arm (m, rad); (0, 0) for an arm without a target
    deadline: float | None = None    # s of robot time, set when it starts
    judge: GoalJudge | None = None


class GraspGoalServer:
    """Run /grasp_frame_target's action server on a thread of its own and its goals on the control tick."""

    def __init__(self, node: Node, kin, tolerance_m: float, tolerance_rad: float, settle_s: float,
                 default_timeout_s: float, ready):
        """Start the action server on a node and executor of its own, beside node (the control node).

        kin: the control node's kinematics.ArmKinematics, GRASP_FRAMES among its extra frames, already updated.
        ready(): None when a goal may start, else why not (the node not engaged yet, latched, ...); called on the
        action server's thread, so it reads flags only. Raises ValueError when kin lacks the grasp frames.
        """
        self._node = node
        self._kin = kin
        self.frames = GraspFrames(kin)
        self._tolerance = (float(tolerance_m), float(tolerance_rad), float(settle_s))
        self._default_timeout_s = float(default_timeout_s)
        self._ready = ready
        self._lock = threading.Lock()
        self._queue: deque[_Goal] = deque()
        self._active: _Goal | None = None
        self._pose_pubs = tuple(node.create_publisher(PoseStamped, topic, 10) for topic in GRASP_POSE_TOPICS)
        self._server_node = Node(f"{node.get_name()}_grasp", context=node.context, use_global_arguments=False,
                                 start_parameter_services=False)
        # reentrant on several threads: a goal's execute callback waits for its end while cancels and other goals
        # come in
        self._server = ActionServer(self._server_node, GraspFrameTarget, GRASP_ACTION,
                                    execute_callback=self._execute, goal_callback=lambda _: GoalResponse.ACCEPT,
                                    cancel_callback=lambda _: CancelResponse.ACCEPT,
                                    callback_group=ReentrantCallbackGroup())
        self._executor = MultiThreadedExecutor(num_threads=4, context=node.context)
        self._executor.add_node(self._server_node)
        self._thread = threading.Thread(target=self._executor.spin, name="grasp_frame_target", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Abort the goals, stop the action server's executor and destroy its node."""
        self.abort_all("marl_deploy_node is shutting down")
        self._executor.shutdown()
        self._thread.join(timeout=2.0)
        self._server.destroy()
        self._server_node.destroy_node()

    # -- the action server's thread ---------------------------------------------

    def _execute(self, goal_handle):
        """Run one goal: refuse it or resolve its targets, queue it for the control tick, publish its feedback until
        the tick ends it, then end its handle; return its result."""
        try:
            refusal = self._ready()
            if refusal is not None:
                raise GoalError(refusal)
            targets = self._targets(goal_handle.request)
        except GoalError as error:
            return self._end(goal_handle, Result.FAILED, str(error))
        timeout = goal_handle.request.timeout.sec + 1e-9 * goal_handle.request.timeout.nanosec
        goal = _Goal(goal_handle, targets, timeout if timeout > 0.0 else self._default_timeout_s)
        self._node.get_logger().info("grasp frame target: " + "; ".join(
            f"{ARMS[arm]} to {np.round(pos, 3).tolist()} m" for arm, (pos, _) in targets.items())
            + f", timeout {goal.timeout_s:.1f} s")
        with self._lock:
            self._queue.append(goal)
        while not goal.done.wait(FEEDBACK_PERIOD_S):
            with self._lock:
                errors = goal.errors
            if errors is not None:
                feedback = GraspFrameTarget.Feedback()
                (feedback.left_error_m, feedback.left_error_rad), (feedback.right_error_m, feedback.right_error_rad) \
                    = errors
                goal_handle.publish_feedback(feedback)
        return self._end(goal_handle, *goal.outcome)

    def _end(self, goal_handle, outcome: int, message: str):
        """End goal_handle with outcome (a Result constant) and message; return its result.

        REACHED succeeds the goal, CANCELED marks it canceled, NOT_REACHED and FAILED abort it.
        """
        how = {Result.REACHED: "succeed", Result.CANCELED: "canceled"}.get(outcome, "abort")
        # one call site per severity: rclpy refuses a call site whose severity changes
        if outcome == Result.REACHED:
            self._node.get_logger().info(f"grasp frame target {how}: {message}")
        else:
            self._node.get_logger().warn(f"grasp frame target {how}: {message}")
        getattr(goal_handle, how)()
        result = Result()
        result.outcome = outcome
        result.success = outcome == Result.REACHED
        result.message = message
        return result

    def _targets(self, goal) -> dict:
        """Return {arm index: (position, quaternion wxyz)}, each targeted arm's grasp-frame target in the pelvis frame.

        A target in another frame is resolved through TF now (_pelvis_transforms). GoalError for a goal without a
        target, a non-finite value, a zero quaternion or a transform TF does not give.
        """
        stamped = {arm: getattr(goal, name) for arm, name in enumerate(ARMS)}
        targeted = {arm: pose for arm, pose in stamped.items() if pose.header.frame_id}
        if not targeted:
            raise GoalError("neither arm has a target: give left or right a header.frame_id "
                            f"({POSE_FRAME} or a world frame)")
        targets = {}
        for arm, pose in targeted.items():
            p, o = pose.pose.position, pose.pose.orientation
            pos, quat = np.array([p.x, p.y, p.z]), xyzw_to_wxyz(np.array([o.x, o.y, o.z, o.w]))
            if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
                raise GoalError(f"the {ARMS[arm]} target is not finite: {pos.tolist()} {quat.tolist()}")
            if np.linalg.norm(quat) < 1e-6:
                raise GoalError(f"the {ARMS[arm]} target has no orientation (a zero quaternion)")
            targets[arm] = (pos, quat / np.linalg.norm(quat))
        frames = sorted({pose.header.frame_id for pose in targeted.values()} - {POSE_FRAME})
        transforms = self._pelvis_transforms(frames) if frames else {}
        for arm, pose in targeted.items():
            if pose.header.frame_id != POSE_FRAME:
                t_pos, t_quat = transforms[pose.header.frame_id]
                pos, quat = targets[arm]
                targets[arm] = (t_pos + quat_apply(t_quat, pos), quat_unique(quat_mul(t_quat, quat)))
        return targets

    def _pelvis_transforms(self, frames: list[str]) -> dict:
        """Return {frame: (position, quaternion wxyz) of it in the pelvis frame}, from TF's newest transforms.

        Listens to TF (/tf, /tf_static) for up to TF_WAIT_S s of steady time, then stops. A transform older than
        TF_MAX_AGE_S s of robot time is not taken; one of static transforms only (stamp 0) is. GoalError, naming
        each frame, when a transform is missing or too old by then.

        The listener runs on a node and executor of its own, shut down before the node is destroyed: subscriptions
        destroyed on a node while its executor waits on them stop that executor (InvalidHandle).
        """
        context = self._node.context
        tf_node = Node(f"{self._node.get_name()}_grasp_tf", context=context, use_global_arguments=False,
                       start_parameter_services=False)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(tf_node)
        tf_buffer = Buffer()
        TransformListener(tf_buffer, tf_node)
        spinner = threading.Thread(target=executor.spin, name="grasp_tf", daemon=True)
        spinner.start()
        try:
            found, problems = {}, {}
            deadline = time.monotonic() + TF_WAIT_S
            while True:
                now = self._node.get_clock().now()
                for frame in frames:
                    if frame in found:
                        continue
                    try:
                        transform = tf_buffer.lookup_transform(POSE_FRAME, frame, Time())
                    except TransformException as error:
                        problems[frame] = f"no transform from {frame} to {POSE_FRAME} ({error})"
                        continue
                    stamp = Time.from_msg(transform.header.stamp, clock_type=now.clock_type)
                    age = (now - stamp).nanoseconds * 1e-9
                    if stamp.nanoseconds > 0 and age > TF_MAX_AGE_S:
                        problems[frame] = (f"the newest transform from {frame} to {POSE_FRAME} is {age:.2f} s old "
                                           f"(> {TF_MAX_AGE_S} s)")
                        continue
                    t, r = transform.transform.translation, transform.transform.rotation
                    found[frame] = (np.array([t.x, t.y, t.z]), xyzw_to_wxyz(np.array([r.x, r.y, r.z, r.w])))
                if len(found) == len(frames):
                    return found
                if time.monotonic() >= deadline:
                    raise GoalError("; ".join(problems[frame] for frame in frames if frame not in found)
                                    + f", after {TF_WAIT_S:.1f} s of listening to TF")
                time.sleep(TF_POLL_S)
        finally:
            executor.shutdown()
            spinner.join()
            tf_node.destroy_node()

    # -- the control tick ----------------------------------------------------------

    def step(self, now: float, controller, q: np.ndarray) -> None:
        """Run the goals for one control tick, after controller.step() at the measured joints q (rad).

        now: s of robot time. Ends a queued goal that was canceled, starts the next goal when none runs, and steps
        the running one: its errors, its cancel, its success, its timeout.
        """
        with self._lock:
            for goal in [g for g in self._queue if g.handle.is_cancel_requested]:
                self._queue.remove(goal)
                self._finish(goal, Result.CANCELED, "canceled before it started; the arms did not move for it")
            if self._active is None and self._queue:
                self._start(self._queue.popleft(), now, controller)
            goal = self._active
        if goal is None:
            return
        errors = self.errors(goal.targets)
        with self._lock:
            goal.errors = errors
        text = self._error_text(goal, errors)
        if goal.handle.is_cancel_requested:
            controller.hold_arms(q, list(goal.targets))
            self._end_active(Result.CANCELED, f"canceled; the arms hold where they are ({text})")
        elif goal.judge.reached(now, [errors[arm] for arm in goal.targets]):
            self._end_active(Result.REACHED, f"reached ({text})")
        elif now >= goal.deadline:
            controller.hold_arms(q, list(goal.targets))
            self._end_active(Result.NOT_REACHED,
                             f"timed out after {goal.timeout_s:.1f} s; the arms hold where they are ({text})")

    def _start(self, goal: _Goal, now: float, controller) -> None:
        """Start goal (the lock held): the targeted arms' wrist goals to put their grasp frames on the targets."""
        poses = controller.believed.copy()
        for arm, (pos, quat) in goal.targets.items():
            poses[arm] = self.frames.wrist_goal(arm, pos, quat)
        controller.request_arm_goal(poses, arms=list(goal.targets))
        goal.deadline = now + goal.timeout_s
        goal.judge = GoalJudge(*self._tolerance)
        self._active = goal

    def errors(self, targets: dict) -> list[tuple[float, float]]:
        """Return each arm's grasp-frame error (m, rad) from its target at the kinematics' last update (the measured
        joints after controller.step()); (0, 0) for an arm without a target."""
        out = []
        for arm in range(len(ARMS)):
            if arm not in targets:
                out.append((0.0, 0.0))
                continue
            pos, quat = self.frames.grasp_pose(arm, *self._kin.wrist_pose(arm))
            out.append(pose_error(pos, quat, *targets[arm]))
        return out

    @staticmethod
    def _error_text(goal: _Goal, errors) -> str:
        """Return the targeted arms' errors as text: mm and rad per arm."""
        return ", ".join(f"{ARMS[arm]} {errors[arm][0] * 1000.0:.1f} mm {errors[arm][1]:.3f} rad"
                         for arm in goal.targets)

    def _end_active(self, outcome: int, message: str) -> None:
        """End the running goal with outcome (a Result constant) and message."""
        with self._lock:
            goal, self._active = self._active, None
            if goal is not None:
                self._finish(goal, outcome, message)

    @staticmethod
    def _finish(goal: _Goal, outcome: int, message: str) -> None:
        """Give goal its outcome (a Result constant) and message and wake its execute callback (the lock held)."""
        goal.outcome = (outcome, message)
        goal.done.set()

    def abort_all(self, message: str) -> None:
        """Abort the running and the queued goals as FAILED with message; the arms keep their commands."""
        with self._lock:
            goals = ([self._active] if self._active is not None else []) + list(self._queue)
            self._active = None
            self._queue.clear()
            for goal in goals:
                self._finish(goal, Result.FAILED, message)

    @property
    def busy(self) -> bool:
        """Return whether a goal runs or waits."""
        with self._lock:
            return self._active is not None or bool(self._queue)

    def publish_poses(self, q: np.ndarray, stamp) -> None:
        """Publish each grasp frame's pose in the pelvis frame at the measured joints q (rad, policy order)."""
        for (pos, quat), publisher in zip(self._kin.frame_poses(q, list(GRASP_FRAMES)), self._pose_pubs):
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = POSE_FRAME
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in pos)
            x, y, z, w = (float(v) for v in wxyz_to_xyzw(quat))
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = x, y, z, w
            publisher.publish(msg)
