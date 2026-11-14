"""
Check a grasp candidate's approach before the arm moves: the IK along its path, and the scene.

GraspReach is a mixin of GraspSkill. reach_check() replays the executor's approach (take
off, straight-line transit to the pre-grasp, in along the grasp axis) on golem_upper_ik's
own IK model, from the arm's measured joint positions (/joint_states), and counts the
head-camera scene points the hand would sit in at the pre-grasp and along the grasp axis
(reach_model.ReachModel.check). A candidate that fails either is skipped without moving
the arm, which costs about a second instead of a failed approach and a reset to 'prep'.

The model is built and the checks run in a child process (reach_model.serve): pinocchio's
URDF parse holds the GIL for seconds at a time, and in the node's own process that stalls
the safety heartbeat's callback until golem_skills halts the arm move in flight.

The check is a filter, not a guarantee: the IK model has the self-collision spheres but
not the scene, and the scene test covers the hand only, from one head-camera view.
"""

import multiprocessing
import threading
import time

import numpy as np

from .. import reach_model
from ..perception_utils import pose_to_matrix


# s from the model's start until a check stops waiting for it (it builds in ~15 s)
REACH_MODEL_WAIT_SEC = 60.0
# s one check may take before the child process is stopped and checks are turned off
REACH_CHECK_TIMEOUT_SEC = 20.0
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


class GraspReach:
    """Skip grasp candidates whose approach the arm cannot follow (module doc)."""

    def _reach_model_start(self):
        """Start the child process that builds the IK model and serves checks, once."""
        if getattr(self, '_reach_lock', None) is not None:
            return
        self._reach_lock = threading.Lock()
        self._reach_conn = None
        self._reach_proc = None
        self._reach_ready = False
        self._reach_error = None
        self._reach_started = time.monotonic()
        try:
            # spawn, not fork: a forked child would inherit the node's rclpy threads
            ctx = multiprocessing.get_context('spawn')
            self._reach_conn, child = ctx.Pipe()
            self._reach_proc = ctx.Process(target=reach_model.serve, args=(child,),
                                           name='grasp_reach', daemon=True)
            self._reach_proc.start()
            child.close()
        except Exception as e:  # noqa: BLE001 - any failure turns the check off
            self._reach_stop(f'could not start the IK process ({type(e).__name__}: {e})')

    def _reach_stop(self, reason):
        """Turn the check off for the node's life, stopping the child process; log why."""
        self._reach_error = reason
        self._reach_ready = False
        if self._reach_proc is not None and self._reach_proc.is_alive():
            self._reach_proc.kill()
        self.get_logger().error(f'grasp reach check off: {reason}; candidates go unchecked')

    def _reach_wait_ready(self):
        """Wait for the model until REACH_MODEL_WAIT_SEC after its start; return if ready."""
        if self._reach_ready or self._reach_error is not None:
            return self._reach_ready
        left = self._reach_started + REACH_MODEL_WAIT_SEC - time.monotonic()
        try:
            if not self._reach_conn.poll(max(0.0, left)):
                return False
            kind, value = self._reach_conn.recv()
        except (EOFError, OSError):
            self._reach_stop('the IK process exited while building the model')
            return False
        if kind == 'ready':
            self._reach_ready = True
            self.get_logger().info(f'grasp reach check: IK model built in {value:.1f}s')
        else:
            self._reach_stop(f'the IK model failed to build ({value})')
        return self._reach_ready

    def reach_check(self, frame, waypoints, obstacles):
        """
        Return why the approach along `waypoints` fails, or None when it passes.

        `waypoints`: (Pose, label, (lin_tol, ang_tol) or None, scene) in execution order,
        pelvis frame, for URDF `frame`; a waypoint with tolerances must be reached within
        them, one without only steers the IK, and one with `scene` set must leave the hand
        out of `obstacles` (scene points from scene_obstacles, or None).
        The IK starts at the latest /joint_states. Returns None without checking when the
        model or the joint state is unavailable (logged), so the executor tries the
        candidate as before. Blocks the calling thread only.
        """
        self._reach_model_start()
        with self._reach_lock:
            js = self.latest_joint_state()
            if not self._reach_wait_ready() or js is None:
                self.get_logger().warn(
                    'grasp reach check skipped: '
                    + (self._reach_error or ('IK model not built yet' if not self._reach_ready
                                             else 'no /joint_states')))
                return None
            request = (list(js.name), list(js.position), frame,
                       [(pose_to_matrix(p), label, tol, scene)
                        for p, label, tol, scene in waypoints],
                       obstacles)
            try:
                self._reach_conn.send(request)
                if not self._reach_conn.poll(REACH_CHECK_TIMEOUT_SEC):
                    self._reach_stop(f'a check took over {REACH_CHECK_TIMEOUT_SEC:.0f}s')
                    return None
                kind, value = self._reach_conn.recv()
            except (EOFError, OSError):
                self._reach_stop('the IK process exited')
                return None
            if kind == 'error':
                self.get_logger().warn(f'grasp reach check failed to run: {value}')
                return None
            return value
