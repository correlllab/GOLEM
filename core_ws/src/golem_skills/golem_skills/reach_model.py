"""
Replay a grasp approach on golem_upper_ik's IK model and test the hand against the scene.

ReachModel.check() steps the same differential IK golem_upper_ik's frame_task_server steps,
on the frame it drives (the arm's grasp frame, the skills' pinch frame), from a measured
joint state, along the approach's waypoints, and tests the hand against the
head-camera scene: the points it would sit in, and the support surface extended toward
the robot where the camera does not see (support_edges). serve() runs it in a child
process for the grasp skill (skills/grasp_reach.py): pinocchio's URDF parse holds the GIL
for seconds at a time, which would stall every callback of the skills node, the safety
heartbeat's included.

This module imports no ROS package, so the child process starts without the node's.
"""

import time

import numpy as np
import pinocchio as pin

from golem_upper_ik.core.ik_solver import IKSolver
from golem_upper_ik.core.robot_model import FixedState, RobotModel
from golem_upper_ik.profile import load_profile

from .robots import arm_of, robot_profile


# m golem_upper_ik's self-collision barrier keeps between sphere pairs (controller.d_min of
# every H1-2 and G1 controller config)
REACH_D_MIN_M = 0.02
# s per simulated IK step; frame_task_server steps faster, but the path it follows
# is the same, and a coarser step keeps a check near a second
REACH_IK_DT_S = 0.05
# steps a waypoint may take before its error is judged
REACH_STEPS_MAX = 200
# a waypoint is judged once its position error improves less than this [m] over
# REACH_STABLE_STEPS steps, as the executor stops a move whose feedback is stable
REACH_STABLE_M = 0.0005
REACH_STABLE_STEPS = 20

# Each hand's volume in its grasp frame (the GraspGenX gripper-base convention: +Z the
# approach, +X the finger-closing axis), as axis-aligned boxes (x0, x1, y0, y1, z0, z1)
# [m], back to the contact plane (common.GRIPPER_BASE_TO_CONTACT_M). Beyond that plane
# the fingers straddle the part and a close stops on whatever they meet, so they are not
# tested.
HAND_BOXES_M = {
    # the base (base_top, base_bot) and the open fingers, from the magpie meshes in
    # CL_Assets/graspgenx/assets/meshes/magpie
    'magpie': (
        (-0.067, 0.067, -0.019, 0.015, 0.002, 0.085),
        (-0.108, -0.054, -0.021, 0.016, 0.066, 0.1146),
        (0.054, 0.108, -0.021, 0.016, 0.066, 0.1146),
    ),
    # the Dex3-1 open, in robots.py's grasp frame: the palm, the index and middle fingers
    # standing straight along +X, and the thumb standing out of the palm toward the part.
    # From the URDF's joint origins with links about 25 mm thick, not from the meshes;
    # unmeasured on the real G1.
    'dex3': (
        (-0.080, 0.000, -0.045, 0.045, 0.0496, 0.0846),
        (-0.010, 0.095, -0.042, 0.042, 0.0626, 0.0866),
        (-0.070, -0.040, -0.012, 0.012, 0.0846, 0.1146),
    ),
}
# m each box is shrunk by on every side, so depth noise on a surface the hand only
# touches does not count
HAND_MARGIN_M = 0.003
# scene points inside the hand that make a waypoint a collision
HAND_HIT_POINTS = 10

# The head camera does not see the space just in front of the robot at table height (on
# the H1-2 at the battery, nothing nearer than ~0.38 m ahead of the pelvis), where a low
# pre-grasp puts the hand. There the scene is taken to continue as first seen: in each
# pelvis-y slice, everything nearer the robot than the slice's nearest scene point is
# solid up to the height of the scene at that near edge. A hand reaching below a surface
# whose near edge the camera does see (a table edge) is rejected too.
# m: width of the pelvis-y slices
SUPPORT_SLICE_M = 0.02
# m: depth behind a slice's nearest point whose points give the slice's height
SUPPORT_BAND_M = 0.03
# percentile of those points' height taken as the slice's height (above the noise of the
# few highest points)
SUPPORT_TOP_PERCENTILE = 90
# scene points a slice needs to be extended
SUPPORT_MIN_POINTS = 5
# m: spacing of the points the hand's boxes are sampled at for the support test
HAND_SAMPLE_M = 0.005
# m the hand may sit below an extended support before a waypoint is a collision
SUPPORT_DEPTH_M = 0.005


def hand_hits(obstacles, T, hand):
    """Return how many `obstacles` points lie in `hand` (a HAND_BOXES_M key) at grasp pose `T`."""
    if obstacles is None or len(obstacles) == 0:
        return 0
    local = (obstacles - T[:3, 3]) @ T[:3, :3]
    m = HAND_MARGIN_M
    inside = np.zeros(len(local), dtype=bool)
    for x0, x1, y0, y1, z0, z1 in HAND_BOXES_M[hand]:
        inside |= ((local[:, 0] > x0 + m) & (local[:, 0] < x1 - m)
                   & (local[:, 1] > y0 + m) & (local[:, 1] < y1 - m)
                   & (local[:, 2] > z0 + m) & (local[:, 2] < z1 - m))
    return int(inside.sum())


def _hand_samples(boxes):
    """Return points filling `boxes`, each shrunk by HAND_MARGIN_M (grasp frame)."""
    m = HAND_MARGIN_M
    out = []
    for box in boxes:
        axes = [np.linspace(lo + m, hi - m, int(np.ceil((hi - lo - 2 * m) / HAND_SAMPLE_M)) + 1)
                for lo, hi in zip(box[0::2], box[1::2])]
        out.append(np.stack([a.ravel() for a in np.meshgrid(*axes, indexing='ij')], axis=-1))
    return np.concatenate(out)


HAND_SAMPLES = {hand: _hand_samples(boxes) for hand, boxes in HAND_BOXES_M.items()}


def support_edges(obstacles):
    """
    Return {pelvis-y slice index: (near x, height)} of the scene's near edge [m, pelvis].

    A slice's near x is its scene point nearest the robot, and its height the
    SUPPORT_TOP_PERCENTILE height of the points within SUPPORT_BAND_M behind it. Slices
    with fewer than SUPPORT_MIN_POINTS points are left out.
    """
    if obstacles is None or len(obstacles) == 0:
        return {}
    pts = np.asarray(obstacles, dtype=float)
    idx = np.floor(pts[:, 1] / SUPPORT_SLICE_M).astype(int)
    edges = {}
    for i in np.unique(idx):
        s = pts[idx == i]
        if len(s) < SUPPORT_MIN_POINTS:
            continue
        near = float(s[:, 0].min())
        band = s[s[:, 0] < near + SUPPORT_BAND_M, 2]
        edges[int(i)] = (near, float(np.percentile(band, SUPPORT_TOP_PERCENTILE)))
    return edges


def support_depth(edges, T, hand):
    """Return how far [m] `hand` at grasp pose `T` sits below the extended support."""
    if not edges:
        return 0.0
    p = HAND_SAMPLES[hand] @ T[:3, :3].T + T[:3, 3]
    idx = np.floor(p[:, 1] / SUPPORT_SLICE_M).astype(int)
    depth = 0.0
    for i in np.unique(idx):
        edge = edges.get(int(i))
        if edge is None:
            continue
        s = p[idx == i]
        ahead = s[s[:, 0] < edge[0], 2]
        if len(ahead):
            depth = max(depth, edge[1] - float(ahead.min()))
    return depth


class ReachModel:
    """Follow pelvis-frame waypoints with golem_upper_ik's IK from a measured pose."""

    def __init__(self, robot):
        """Build `robot`'s IK model (~15 s: the URDF and self-collision model)."""
        # the skills' profile (robots.py): its hand and tool frames
        self.robot = robot_profile(robot)
        self.profile = load_profile(self.robot.name)
        self.model = RobotModel(self.profile)
        self.joints = list(self.profile.body_joints)
        q = np.zeros(self.model.num_motor)
        self.model.state_subscriber = FixedState(q)
        self.ik = IKSolver(self.model, self.profile.named_configs, dt=REACH_IK_DT_S,
                           d_min=REACH_D_MIN_M)
        # body q the frozen (held) joints are locked at in the reduced model
        self._locked_q = q

    def body_q(self, names, positions):
        """Return body q (rad, profile order) from a JointState's names and positions."""
        q = np.zeros(self.model.num_motor)
        index = {n: i for i, n in enumerate(self.joints)}
        for name, value in zip(names, positions):
            if name in index:
                q[index[name]] = float(value)
        return q

    def start(self, q_body):
        """Set the IK at body q; re-lock the held joints where they are when they moved."""
        held = ~self.model.reduced_mask
        if np.any(np.abs(q_body[held] - self._locked_q[held]) > 0.01):
            self.model.state_subscriber = FixedState(q_body)
            self.model.rebuild_reduced_model(q_body)
            self.ik.rebuild_reduced()
            self._locked_q = q_body.copy()
        self.ik.update_configurations(q_body)

    def frame_pose(self, frame):
        """Return `frame`'s 4x4 pose (pelvis) at the IK's current configuration."""
        return np.asarray(self.model.get_frame_transformation_reduced(frame, self.ik.q_reduced))

    def pinch_target(self, frame, T):
        """
        Return (`frame`'s arm's pinch frame, its 4x4 target that puts `frame` at `T`).

        The pinch frame is the one the arm controller drives (/grasp_frame_target).
        A tool frame of the profile is fixed on a model frame; any other frame is a
        model frame. Both are rigid on the hand, so their transform to the pinch
        frame is read at the IK's current configuration.
        """
        pinch = self.robot.pinch_frames[arm_of(self.robot, frame)]
        if frame == pinch:
            return pinch, T
        tool = self.robot.tool_frames.get(frame)
        if tool is None:
            offset = np.linalg.inv(self.frame_pose(frame)) @ self.frame_pose(pinch)
        else:
            offset = (np.linalg.inv(tool.matrix())
                      @ np.linalg.inv(self.frame_pose(tool.parent)) @ self.frame_pose(pinch))
        return pinch, T @ offset

    def follow(self, frame, T):
        """Step the IK toward 4x4 pose `T` until it stops improving; return (m, rad) left."""
        self.ik.clear_frame_tasks()
        self.ik.add_frame_task('reach', frame, target=pin.SE3(T[:3, :3], T[:3, 3]))
        task = self.ik.frame_tasks['reach']
        history = []
        for _ in range(REACH_STEPS_MAX):
            err = task.compute_error(self.ik.configuration_reduced)
            lin = float(np.linalg.norm(err[:3]))
            history.append(lin)
            if (lin < REACH_STABLE_M and float(np.linalg.norm(err[3:])) < 0.002) or (
                    len(history) > REACH_STABLE_STEPS
                    and history[-REACH_STABLE_STEPS - 1] - lin < REACH_STABLE_M):
                break
            self.ik.integrate(self.ik.ik_step_reduced())
        err = task.compute_error(self.ik.configuration_reduced)
        return float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))

    def check(self, q_body, frame, waypoints, obstacles):
        """
        Return why the approach along `waypoints` fails from body q `q_body`, or None.

        `waypoints`: (4x4 pose, label, (lin_tol, ang_tol) or None, scene) in execution
        order, pelvis frame, for the grasp or pinch frame `frame`, which the IK follows
        through its arm's pinch frame, as the arm controller does (pinch_target); a
        waypoint with tolerances must be reached within them (the pinch frame's error,
        as the executor judges it), one without only
        steers the IK, and one with `scene` set must leave the hand out of `obstacles`
        ((N, 3) pelvis-frame points, or None) and above the support extended toward the
        robot (support_depth).
        """
        self.start(q_body)
        edges = support_edges(obstacles)
        hand = self.robot.hand
        for T, label, tol, scene in waypoints:
            if scene:
                hits = hand_hits(obstacles, T, hand)
                if hits >= HAND_HIT_POINTS:
                    return f'the hand would be in the scene at the {label} ({hits} points)'
                below = support_depth(edges, T, hand)
                if below > SUPPORT_DEPTH_M:
                    return (f'the hand would be {below * 1000:.0f} mm below the support seen '
                            f'ahead of it at the {label}')
            lin, ang = self.follow(*self.pinch_target(frame, T))
            if tol is not None and (lin > tol[0] or ang > tol[1]):
                return (f'the IK stops {lin * 1000:.0f} mm / {np.degrees(ang):.0f} deg '
                        f'short of the {label}')
        return None


def serve(conn, robot):
    """
    Build `robot`'s ReachModel and answer check requests on `conn` until the other end closes.

    Sends ('ready', build s) or ('error', reason) once, then for each request
    (joint names, positions, frame, waypoints, obstacles) one ('ok', ReachModel.check's
    answer) or ('error', reason).
    """
    try:
        t0 = time.monotonic()
        model = ReachModel(robot)
        conn.send(('ready', time.monotonic() - t0))
    except Exception as e:  # noqa: BLE001 - any failure is reported to the node
        conn.send(('error', f'{type(e).__name__}: {e}'))
        return
    while True:
        try:
            names, positions, frame, waypoints, obstacles = conn.recv()
        except (EOFError, OSError):
            return
        try:
            reply = ('ok', model.check(model.body_q(names, positions), frame, waypoints,
                                       obstacles))
        except Exception as e:  # noqa: BLE001 - one bad request leaves the model serving
            reply = ('error', f'{type(e).__name__}: {e}')
        conn.send(reply)
