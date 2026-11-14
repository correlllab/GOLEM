"""
Check a grasp candidate's approach before the arm moves: the IK along its path, and the scene.

GraspReach is a mixin of GraspSkill. reach_check() replays the executor's approach (take
off, straight-line transit to the pre-grasp, in along the grasp axis) on golem_upper_ik's
own IK model, from the arm's measured joint positions (/joint_states), stepping the same
differential IK the frame_task server steps, and counts the head-camera scene points the
hand would sit in at the pre-grasp and along the grasp axis. A candidate that fails either
is skipped without moving the arm, which costs about a second instead of a failed approach
and a reset to 'prep'.

The check is a filter, not a guarantee: the IK model has the self-collision spheres but
not the scene, and the scene test covers the hand only, from one head-camera view.
"""

import threading
import time

import numpy as np
import pinocchio as pin

from golem_upper_ik.core.ik_solver import IKSolver
from golem_upper_ik.core.robot_model import FixedState, RobotModel
from golem_upper_ik.profile import load_profile

from ..perception_utils import pose_to_matrix


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
# s the first check waits for the IK model (built once per node, ~17 s)
REACH_MODEL_WAIT_SEC = 60.0

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
# m: scene points farther than this from the object are left out of the test, which
# keeps out the arm itself (the head camera sees the hand at 'prep', ~0.4 m away)
SCENE_CROP_M = 0.3
# m: scene points this close to the object's cloud are the object itself
OBJECT_CLEAR_M = 0.015


def scene_obstacles(scene, obj_cloud):
    """Return the scene points (pelvis frame) near the object but not on it, or None."""
    if scene is None or len(scene) == 0 or obj_cloud is None or len(obj_cloud) == 0:
        return None
    from scipy.spatial import cKDTree
    scene = np.asarray(scene, dtype=float)
    scene = scene[np.all(np.isfinite(scene), axis=1)]
    centre = np.asarray(obj_cloud, dtype=float).mean(axis=0)
    near = scene[np.linalg.norm(scene - centre, axis=1) <= SCENE_CROP_M]
    if len(near) == 0:
        return near
    dist, _ = cKDTree(np.asarray(obj_cloud, dtype=float)).query(near)
    return near[dist > OBJECT_CLEAR_M]


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
        """Build the robot's IK model (several seconds: the self-collision model)."""
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


class GraspReach:
    """Skip grasp candidates whose approach the arm cannot follow (module doc)."""

    def _reach_model_start(self):
        """Start building the IK model in the background, once per node."""
        if getattr(self, '_reach_build', None) is None:
            self._reach_model = None
            self._reach_model_error = None
            self._reach_lock = threading.Lock()
            self._reach_build = threading.Thread(target=self._reach_model_build,
                                                 daemon=True)
            self._reach_build.start()

    def _reach_model_build(self):
        """Build the ReachModel; on failure keep the reason and leave the check off."""
        try:
            t0 = time.monotonic()
            self._reach_model = ReachModel()
            self.get_logger().info(
                f'grasp reach check: IK model built in {time.monotonic() - t0:.1f}s')
        except Exception as e:  # noqa: BLE001 - any failure turns the check off
            self._reach_model_error = f'{type(e).__name__}: {e}'
            self.get_logger().error(
                f'grasp reach check: IK model failed to build ({self._reach_model_error}); '
                'candidates go unchecked')

    def reach_check(self, frame, waypoints, obstacles):
        """
        Return why the approach along `waypoints` fails, or None when it passes.

        `waypoints`: (Pose, label, (lin_tol, ang_tol) or None, scene) in execution order,
        pelvis frame, for URDF `frame`; a waypoint with tolerances must be reached within
        them, one without only steers the IK, and one with `scene` set must leave the hand
        out of `obstacles` (scene points from scene_obstacles, or None).
        The IK starts at the latest /joint_states. Returns None without checking when the
        model or the joint state is unavailable (logged), so the executor tries the
        candidate as before.
        """
        self._reach_model_start()
        self._reach_build.join(REACH_MODEL_WAIT_SEC)
        model = self._reach_model
        js = self.latest_joint_state()
        if model is None or js is None:
            self.get_logger().warn(
                'grasp reach check skipped: '
                + ('no /joint_states' if model is not None else
                   self._reach_model_error or 'IK model not built yet'))
            return None
        with self._reach_lock:
            model.start(model.body_q(js.name, js.position))
            for pose, label, tol, scene in waypoints:
                T = pose_to_matrix(pose)
                hits = hand_hits(obstacles, T) if scene else 0
                if hits >= HAND_HIT_POINTS:
                    return f'the hand would be in the scene at the {label} ({hits} points)'
                lin, ang = model.follow(frame, T)
                if tol is not None and (lin > tol[0] or ang > tol[1]):
                    return (f'the IK stops {lin * 1000:.0f} mm / {np.degrees(ang):.0f} deg '
                            f'short of the {label}')
        return None
