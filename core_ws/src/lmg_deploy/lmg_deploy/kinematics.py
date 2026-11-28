"""Compute the wrists' forward kinematics and Jacobians in the pelvis frame, with Pinocchio.

The model is the H1-2 URDF with a fixed base at the pelvis: every pose and
Jacobian here is relative to the pelvis, as in training (PhysX world-frame
Jacobian rotated into the root frame).

The training Jacobian came from PhysX, whose linear rows are the velocity of
the link's centre of mass (the URDF <inertial> origin of the wrist link
alone), while the pose error is the link frame's. jacobian() reproduces that
pairing: its linear rows are taken at the COM point, the poses at the frame.
mdp.IKPostureArmAction's integral (controller.py) steps on it.

For mdp.IKPostureArmAction (export.yaml arm_ik) it also gives the arm joints'
gravity torques (ArmKinematics.gravity_torques in Locomanipulation_game's
mdp/arm_kinematics.py) and the seed solve (its ArmKinematics.solve, the wrist
frame's origin, not the COM), on their own Pinocchio data so the measured
state's poses and Jacobians stay as update() left them.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from lmg_deploy.rotations import compute_pose_error, quat_from_matrix, quat_unique


class ArmKinematics:
    """Hold the H1-2 model and, after update(), the wrist poses, wrist Jacobians and extra frame poses."""

    def __init__(self, urdf_path: str, joint_names: list[str], ee_frames: list[str],
                 arm_joint_names: list[list[str]], pelvis_frame: str = "pelvis",
                 extra_frames: list[str] | None = None):
        """Build the model from the URDF; raise ValueError if a named frame is missing.

        joint_names: the policy's joint order (Unitree motor order), which every q vector uses.
        ee_frames: the wrist frame per arm. arm_joint_names: each arm's joints.
        extra_frames: more frames whose pelvis-frame pose update() computes, for frame_pose().
        """
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._q_pin = pin.neutral(self.model)
        # Policy index -> Pinocchio q index; every policy joint is a 1-dof revolute joint.
        self._q_idx = np.array([self.model.joints[self.model.getJointId(n)].idx_q for n in joint_names])
        self._pelvis_id = self.model.getFrameId(pelvis_frame)
        self._ee_ids = [self.model.getFrameId(f) for f in ee_frames]
        self._extra_ids = {f: self.model.getFrameId(f) for f in (extra_frames or [])}
        self._extra_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, fid in zip([pelvis_frame] + ee_frames + list(self._extra_ids),
                             [self._pelvis_id] + self._ee_ids + list(self._extra_ids.values())):
            if fid >= len(self.model.frames):
                raise ValueError(f"frame {name!r} not in {urdf_path}")
        # Per arm: its columns in the policy's joint order, and in Pinocchio's velocity vector.
        self.arm_cols = [np.array([joint_names.index(j) for j in arm]) for arm in arm_joint_names]
        self._v_cols = [np.array([self.model.joints[self.model.getJointId(j)].idx_v for j in arm])
                        for arm in arm_joint_names]
        # Each wrist link's own COM (link frame), where PhysX takes the Jacobian's linear rows.
        self._jac_points = [_inertial_origin(urdf_path, f) for f in ee_frames]
        lower = self.model.lowerPositionLimit[self._q_idx]
        upper = self.model.upperPositionLimit[self._q_idx]
        self.joint_limits = np.stack([lower, upper], axis=1)  # (num joints, 2), policy order
        self._jacobians: list[np.ndarray] = []
        self._poses: list[tuple[np.ndarray, np.ndarray]] = []
        self.extra_frames = list(self._extra_ids)
        # gravity_torques sets its model's gravity per call: a model of its own; solve_arms works on its own data
        self._gmodel = pin.buildModelFromUrdf(urdf_path)
        self._gdata = self._gmodel.createData()
        self._sdata = self.model.createData()
        self._pdata = self.model.createData()   # frame_poses' own: the measured state's update() stays as it is
        self._v_idx = np.array([self.model.joints[self.model.getJointId(n)].idx_v for n in joint_names])

    def update(self, q: np.ndarray) -> None:
        """Compute the poses and wrist Jacobians at joint angles q (rad, policy order)."""
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
            # Frame-origin linear and angular rows in world axes; the linear rows move to the COM point.
            jac = pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)[:, self._v_cols[arm]]
            r = o_m_ee.rotation @ self._jac_points[arm]
            lin = jac[:3] - _skew(r) @ jac[3:]
            self._jacobians.append(np.vstack([r_pelvis_t @ lin, r_pelvis_t @ jac[3:]]))
        for name, fid in self._extra_ids.items():
            pelvis_m_f = o_m_pelvis.actInv(self.data.oMf[fid])
            self._extra_poses[name] = (pelvis_m_f.translation.copy(), quat_from_matrix(pelvis_m_f.rotation))

    def wrist_pose(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the arm's wrist (position m, quaternion wxyz with qw >= 0) in the pelvis frame."""
        pos, quat = self._poses[arm]
        return pos, quat_unique(quat)

    def frame_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return an extra frame's (position m, quaternion wxyz) in the pelvis frame at the last update()."""
        return self._extra_poses[name]

    def frame_poses(self, q: np.ndarray, frames: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return each extra frame's (position m, quaternion wxyz with qw >= 0) in the pelvis frame at joint angles q
        (rad, policy order), on data of its own: what update() left is not changed."""
        q_pin = pin.neutral(self.model)
        q_pin[self._q_idx] = q
        pin.framesForwardKinematics(self.model, self._pdata, q_pin)
        o_m_pelvis = self._pdata.oMf[self._pelvis_id]
        out = []
        for name in frames:
            pelvis_m_f = o_m_pelvis.actInv(self._pdata.oMf[self._extra_ids[name]])
            out.append((pelvis_m_f.translation.copy(), quat_unique(quat_from_matrix(pelvis_m_f.rotation))))
        return out

    def jacobian(self, arm: int) -> np.ndarray:
        """Return the arm's (6, 7) wrist Jacobian at the last update(): pelvis frame, linear rows at the wrist COM."""
        return self._jacobians[arm]

    def gravity_torques(self, q: np.ndarray, gravity_b: np.ndarray) -> np.ndarray:
        """Return the torque gravity puts on each joint (Nm, policy order) at joint angles q (rad, policy order; the
        joints the policy leaves out, the grippers', at 0), gravity_b the gravity vector (m/s^2) in the pelvis frame.

        Training's ArmKinematics.gravity_torques for the arm joints (a PD holds q with its target at q - tau / kp).
        """
        q_pin = pin.neutral(self._gmodel)
        q_pin[self._q_idx] = q
        pin.framesForwardKinematics(self._gmodel, self._gdata, q_pin)
        r_root_pelvis = self._gdata.oMf[self._pelvis_id].rotation
        self._gmodel.gravity.linear = r_root_pelvis @ np.asarray(gravity_b, dtype=np.float64)
        # Pinocchio's generalized gravity is the torque that holds q: minus the torque gravity puts on each joint
        return -pin.computeGeneralizedGravity(self._gmodel, self._gdata, q_pin)[self._v_idx]

    def solve_arms(self, goals: np.ndarray, q0: np.ndarray, low: np.ndarray, high: np.ndarray, iters: int,
                   damping: float, rot_weight: float, max_step: float = 0.2) -> np.ndarray:
        """Return q0 (policy order) with each arm's joints solved for its wrist goal (arms, 7: x y z qw qx qy qz,
        pelvis frame), from q0's: training's ArmKinematics.solve, damped least squares on the pose error [position,
        rot_weight x axis-angle] at the wrist frame's origin, each step capped at max_step per joint and clamped into
        [low, high] (policy order); the other joints held at q0's."""
        q = np.asarray(q0, dtype=np.float64).copy()
        w = np.array([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight])
        damp = damping ** 2 * np.eye(6)
        q_pin = pin.neutral(self.model)
        data = self._sdata
        for _ in range(iters):
            q_pin[self._q_idx] = q
            pin.computeJointJacobians(self.model, data, q_pin)
            pin.updateFramePlacements(self.model, data)
            o_m_pelvis = data.oMf[self._pelvis_id]
            r_pelvis_t = o_m_pelvis.rotation.T
            for arm, fid in enumerate(self._ee_ids):
                pelvis_m_ee = o_m_pelvis.actInv(data.oMf[fid])
                pos_err, rot_err = compute_pose_error(pelvis_m_ee.translation, quat_from_matrix(pelvis_m_ee.rotation),
                                                      goals[arm, :3], goals[arm, 3:])
                jac = pin.getFrameJacobian(self.model, data, fid, pin.LOCAL_WORLD_ALIGNED)[:, self._v_cols[arm]]
                jac = np.vstack([r_pelvis_t @ jac[:3], r_pelvis_t @ jac[3:]]) * w[:, None]
                err = np.concatenate([pos_err, rot_err]) * w
                step = jac.T @ np.linalg.solve(jac @ jac.T + damp, err)
                cols = self.arm_cols[arm]
                q[cols] = np.clip(q[cols] + np.clip(step, -max_step, max_step), low[cols], high[cols])
        return q


def _skew(v: np.ndarray) -> np.ndarray:
    """Return the cross-product matrix of v."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _inertial_origin(urdf_path: str, link: str) -> np.ndarray:
    """Return a URDF link's <inertial><origin xyz> (its COM, link frame, m); raise ValueError if absent."""
    for element in ET.parse(urdf_path).getroot().iter("link"):
        if element.get("name") == link:
            origin = element.find("inertial/origin")
            if origin is None:
                return np.zeros(3)
            return np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
    raise ValueError(f"link {link!r} not in {urdf_path}")
