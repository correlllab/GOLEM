"""Turn /grasp_frame_target goals into the policy's wrist goals and judge them, without ROS.

The action (custom_ros_messages/GraspFrameTarget) targets each arm's grasp frame, the pinch point rigid on its
wrist (the URDF's left_grasp_frame and right_grasp_frame on *_wrist_yaw_link). The policy reaches wrist poses, so
GraspFrames takes each arm's fixed wrist -> grasp-frame transform from the URDF the kinematics loaded and turns a
grasp-frame target into the wrist goal that puts the grasp frame there, and a measured wrist pose into the grasp
frame's. GoalJudge decides when a goal has been reached: every targeted arm within the tolerance for settle_s of
robot time. Poses are (position m, quaternion wxyz) in the pelvis frame.
"""

import numpy as np

from lmg_deploy.rotations import quat_apply, quat_conjugate, quat_mul, quat_unique

ARMS = ("left", "right")
# The H1-2's grasp frames, left then right (golem_upper_ik's config/h12/robot.yaml grasp_frames)
GRASP_FRAMES = ("left_grasp_frame", "right_grasp_frame")


class GraspFrames:
    """Hold each arm's wrist -> grasp-frame transform and map between wrist and grasp-frame poses."""

    def __init__(self, kin):
        """Read the transforms from kin (kinematics.ArmKinematics with GRASP_FRAMES among its extra frames).

        kin must have been updated at some joint angles; the transform is rigid, so any do. Raises ValueError when
        a grasp frame is not among kin's frames.
        """
        missing = [frame for frame in GRASP_FRAMES if frame not in kin.extra_frames]
        if missing:
            raise ValueError(f"the kinematics lack the grasp frames {missing}")
        self._wrist_grasp = []   # per arm: (position m, quaternion wxyz) of the grasp frame in the wrist frame
        for arm, frame in enumerate(GRASP_FRAMES):
            p_w, q_w = kin.wrist_pose(arm)
            p_g, q_g = kin.frame_pose(frame)
            q_wi = quat_conjugate(q_w)
            self._wrist_grasp.append((quat_apply(q_wi, p_g - p_w), quat_mul(q_wi, q_g)))

    def wrist_goal(self, arm: int, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Return the wrist pose (7,: x y z qw qx qy qz) that puts the arm's grasp frame at (pos, quat)."""
        p_wg, q_wg = self._wrist_grasp[arm]
        q = quat_mul(quat, quat_conjugate(q_wg))
        return np.concatenate([pos - quat_apply(q, p_wg), quat_unique(q)])

    def grasp_pose(self, arm: int, wrist_pos: np.ndarray, wrist_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the arm's grasp-frame pose (position, quaternion) for its wrist at (wrist_pos, wrist_quat)."""
        p_wg, q_wg = self._wrist_grasp[arm]
        return wrist_pos + quat_apply(wrist_quat, p_wg), quat_unique(quat_mul(wrist_quat, q_wg))


def pose_error(pos: np.ndarray, quat: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray) -> tuple[float, float]:
    """Return (m, rad): the distance between the two positions and the angle of the rotation between the poses."""
    dot = abs(float(np.dot(quat, target_quat))) / (np.linalg.norm(quat) * np.linalg.norm(target_quat))
    return float(np.linalg.norm(np.asarray(pos) - np.asarray(target_pos))), 2.0 * float(np.arccos(min(1.0, dot)))


class GoalJudge:
    """Decide a goal is reached once every targeted arm has stayed within the tolerance for settle_s."""

    def __init__(self, tolerance_m: float, tolerance_rad: float, settle_s: float):
        """Set the tolerance (m, rad) and the settle time (s of robot time)."""
        self.tolerance_m, self.tolerance_rad, self.settle_s = float(tolerance_m), float(tolerance_rad), float(settle_s)
        self._within_since: float | None = None

    def reached(self, now: float, errors: list[tuple[float, float]]) -> bool:
        """Return whether every (m, rad) error has been within the tolerance since settle_s before now (s)."""
        within = all(m < self.tolerance_m and rad < self.tolerance_rad for m, rad in errors)
        if not within:
            self._within_since = None
            return False
        if self._within_since is None:
            self._within_since = now
        return now - self._within_since >= self.settle_s - 1e-9
