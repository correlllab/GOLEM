"""
Replay a grasp approach on golem_upper_ik's IK model and test the hand against the scene.

ReachModel.check() steps the same differential IK the frame_task server steps, from a
measured joint state, along the approach's waypoints, and counts the head-camera scene
points the hand would sit in. serve() runs it in a child process for the grasp skill
(skills/grasp_reach.py): pinocchio's URDF parse holds the GIL for seconds at a time, which
would stall every callback of the skills node, the safety heartbeat's included.

This module imports no ROS package, so the child process starts without the node's.
"""

import time

import numpy as np
import pinocchio as pin

from golem_upper_ik.core.ik_solver import IKSolver
from golem_upper_ik.core.robot_model import FixedState, RobotModel
from golem_upper_ik.profile import load_profile


# robot whose IK model the check runs: the grasp skill drives the H1-2's Magpie frames
REACH_ROBOT = 'h12'
# m golem_upper_ik's self-collision barrier keeps between sphere pairs (controller.d_min of
# every H1-2 controller config)
REACH_D_MIN_M = 0.02
# s per simulated IK step; the frame_task server steps faster, but the path it follows
# is the same, and a coarser step keeps a check near a second
REACH_IK_DT_S = 0.05
# steps a waypoint may take before its error is judged
REACH_STEPS_MAX = 200
# a waypoint is judged once its position error improves less than this [m] over
# REACH_STABLE_STEPS steps, as the executor stops a move whose feedback is stable
REACH_STABLE_M = 0.0005
REACH_STABLE_STEPS = 20

# The hand's volume in the GraspGenX gripper-base frame (+Z the approach, +X the
# finger-closing axis), as axis-aligned boxes (x0, x1, y0, y1, z0, z1) [m], from the magpie
# meshes in CL_Assets/graspgenx/assets/meshes/magpie: the base (base_top, base_bot) and the
# open fingers back to the contact plane (common.GRIPPER_BASE_TO_CONTACT_M). Beyond that
# plane the pads straddle the part and a close stops on whatever they meet, so they are
# not tested.
HAND_BOXES_M = (
    (-0.067, 0.067, -0.019, 0.015, 0.002, 0.085),
    (-0.108, -0.054, -0.021, 0.016, 0.066, 0.1146),
    (0.054, 0.108, -0.021, 0.016, 0.066, 0.1146),
)
# m each box is shrunk by on every side, so depth noise on a surface the hand only
# touches does not count
HAND_MARGIN_M = 0.003
# scene points inside the hand that make a waypoint a collision
HAND_HIT_POINTS = 10


def hand_hits(obstacles, T):
    """Return how many `obstacles` points lie in the hand at gripper-base pose `T` (4x4)."""
    if obstacles is None or len(obstacles) == 0:
        return 0
    local = (obstacles - T[:3, 3]) @ T[:3, :3]
    m = HAND_MARGIN_M
    inside = np.zeros(len(local), dtype=bool)
    for x0, x1, y0, y1, z0, z1 in HAND_BOXES_M:
        inside |= ((local[:, 0] > x0 + m) & (local[:, 0] < x1 - m)
                   & (local[:, 1] > y0 + m) & (local[:, 1] < y1 - m)
                   & (local[:, 2] > z0 + m) & (local[:, 2] < z1 - m))
    return int(inside.sum())


class ReachModel:
    """Follow pelvis-frame waypoints with golem_upper_ik's IK from a measured pose."""

    def __init__(self):
        """Build the robot's IK model (~15 s: the URDF and self-collision model)."""
        self.profile = load_profile(REACH_ROBOT)
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
        order, pelvis frame, for URDF `frame`; a waypoint with tolerances must be reached
        within them, one without only steers the IK, and one with `scene` set must leave
        the hand out of `obstacles` ((N, 3) pelvis-frame points, or None).
        """
        self.start(q_body)
        for T, label, tol, scene in waypoints:
            hits = hand_hits(obstacles, T) if scene else 0
            if hits >= HAND_HIT_POINTS:
                return f'the hand would be in the scene at the {label} ({hits} points)'
            lin, ang = self.follow(frame, T)
            if tol is not None and (lin > tol[0] or ang > tol[1]):
                return (f'the IK stops {lin * 1000:.0f} mm / {np.degrees(ang):.0f} deg '
                        f'short of the {label}')
        return None


def serve(conn):
    """
    Build a ReachModel and answer check requests on `conn` until the other end closes.

    Sends ('ready', build s) or ('error', reason) once, then for each request
    (joint names, positions, frame, waypoints, obstacles) one ('ok', ReachModel.check's
    answer) or ('error', reason).
    """
    try:
        t0 = time.monotonic()
        model = ReachModel()
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
