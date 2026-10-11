"""Wrist forward kinematics and the arms' IK step, in the pelvis frame (Pinocchio).

Reproduces mdp.IKResidualArmAction in Locomanipulation_game: per arm, one
damped-least-squares step (Isaac Lab's DifferentialIKController, pose command,
absolute, dls) from the measured joint angles toward the commanded wrist pose,
with the wrist Jacobian expressed in the pelvis frame and restricted to the
arm's 7 joints, then each joint's step capped at max_ik_step.

The training Jacobian came from PhysX, whose linear rows are the velocity of
the link's centre of mass (the URDF <inertial> origin of the wrist link
alone), while the pose error is the link frame's. The IK step reproduces that
pairing: its linear rows are taken at the COM point, the poses at the frame.

The model is the H1-2 URDF with a fixed base at the pelvis: every pose and
Jacobian here is relative to the pelvis, as in training (PhysX world-frame
Jacobian rotated into the root frame).
"""

import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from locomotion_game_deploy.rotations import compute_pose_error, quat_from_matrix, quat_unique


class ArmKinematics:
    def __init__(self, urdf_path: str, joint_names: list[str], ee_frames: list[str],
                 arm_joint_names: list[list[str]], pelvis_frame: str = "pelvis"):
        """joint_names: the policy's joint order (Unitree motor order); q vectors use it."""
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._q_pin = pin.neutral(self.model)
        # policy index -> pinocchio q index (every policy joint is a 1-dof revolute joint)
        self._q_idx = np.array([self.model.joints[self.model.getJointId(n)].idx_q for n in joint_names])
        self._pelvis_id = self.model.getFrameId(pelvis_frame)
        self._ee_ids = [self.model.getFrameId(f) for f in ee_frames]
        for name, fid in zip([pelvis_frame] + ee_frames, [self._pelvis_id] + self._ee_ids):
            if fid >= len(self.model.frames):
                raise ValueError(f"frame {name!r} not in {urdf_path}")
        # per arm: columns in the policy's joint order, and in pinocchio's velocity vector
        self.arm_cols = [np.array([joint_names.index(j) for j in arm]) for arm in arm_joint_names]
        self._v_cols = [np.array([self.model.joints[self.model.getJointId(j)].idx_v for j in arm])
                        for arm in arm_joint_names]
        # each wrist link's own COM (link frame): where PhysX takes the Jacobian's linear rows
        self._jac_points = [_inertial_origin(urdf_path, f) for f in ee_frames]
        lower = self.model.lowerPositionLimit[self._q_idx]
        upper = self.model.upperPositionLimit[self._q_idx]
        self.joint_limits = np.stack([lower, upper], axis=1)  # (num joints, 2), policy order
        self._jacobians: list[np.ndarray] = []
        self._poses: list[tuple[np.ndarray, np.ndarray]] = []

    def update(self, q: np.ndarray) -> None:
        """Forward kinematics and wrist Jacobians at joint angles q (policy order)."""
        self._q_pin[self._q_idx] = q
        pin.computeJointJacobians(self.model, self.data, self._q_pin)
        pin.updateFramePlacements(self.model, self.data)
        o_m_pelvis = self.data.oMf[self._pelvis_id]
        r_pelvis_t = o_m_pelvis.rotation.T
        self._poses, self._jacobians = [], []
        for arm, fid in enumerate(self._ee_ids):
            o_m_ee = self.data.oMf[fid]
            pelvis_m_ee = o_m_pelvis.actInv(o_m_ee)
            self._poses.append((pelvis_m_ee.translation.copy(), quat_from_matrix(pelvis_m_ee.rotation)))
            # frame-origin linear and angular velocity, world axes; moved to the COM point
            jac = pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)[:, self._v_cols[arm]]
            r = o_m_ee.rotation @ self._jac_points[arm]
            lin = jac[:3] - _skew(r) @ jac[3:]
            self._jacobians.append(np.vstack([r_pelvis_t @ lin, r_pelvis_t @ jac[3:]]))

    def wrist_pose(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        """(position, quaternion wxyz with qw >= 0) of the arm's wrist frame in the pelvis frame."""
        pos, quat = self._poses[arm]
        return pos, quat_unique(quat)

    def ik_step(self, arm: int, target_pos: np.ndarray, target_quat: np.ndarray, damping: float,
                max_step: float) -> np.ndarray:
        """The capped DLS joint step for one arm toward a pelvis-frame wrist pose, in its arm_cols order."""
        pos, quat = self._poses[arm]
        pos_err, rot_err = compute_pose_error(pos, quat, target_pos, target_quat)
        err = np.concatenate([pos_err, rot_err])
        jac = self._jacobians[arm]
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(6), err)
        return np.clip(step, -max_step, max_step)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _inertial_origin(urdf_path: str, link: str) -> np.ndarray:
    """The <inertial><origin xyz> of a URDF link (its own COM, link frame)."""
    for element in ET.parse(urdf_path).getroot().iter("link"):
        if element.get("name") == link:
            origin = element.find("inertial/origin")
            if origin is None:
                return np.zeros(3)
            return np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
    raise ValueError(f"link {link!r} not in {urdf_path}")
