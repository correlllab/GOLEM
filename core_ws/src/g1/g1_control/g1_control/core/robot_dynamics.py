import time

import numpy as np
import pinocchio as pin

from g1_control.utility.joint_definition import (
    NUM_MOTOR, FREEFLYER_NV,
    LEFT_ARM_INDEX, RIGHT_ARM_INDEX
)


class RobotDynamics:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self._wrench_last_dq = None
        self._wrench_last_time = None
        self._wrench_ddq = np.zeros(NUM_MOTOR)
        self._wrench_filtered = {}

    def get_com(self, q: np.ndarray=None):
        robot = self.robot_model
        q = robot.state['q'] if q is None else q
        q_full = robot.full_q(q)
        com = pin.centerOfMass(robot.model, robot.data, q_full)
        return com

    def get_com_reduced(self, q_reduced: np.ndarray=None):
        '''Get center of mass for the reduced model (model_body_reduced)'''
        robot = self.robot_model
        q_reduced = robot.state_reduced['q'] if q_reduced is None else q_reduced
        com = pin.centerOfMass(robot.model_body_reduced, robot.data_body_reduced, q_reduced)
        return com

    def get_zmp(self, q: np.ndarray=None):
        robot = self.robot_model
        q = robot.state['q'] if q is None else q
        com = self.get_com(q)

        # find reference support plane
        left_foot_pos = robot.get_frame_position('left_ankle_roll_link', q)
        right_foot_pos = robot.get_frame_position('right_ankle_roll_link', q)
        ground_point = 0.5 * (left_foot_pos + right_foot_pos)
        ground_height = ground_point[2]

        # centroidal momentum time derivative
        q_full = robot.full_q(q)
        dq_full = robot.full_v(robot.state['dq'])
        ddq_full = robot.full_v(robot.state['ddq'])
        pin.computeCentroidalMomentumTimeVariation(
            robot.model, robot.data, q_full, dq_full, ddq_full
        )
        dhg = robot.data.dhg

        F_com = dhg.linear
        tau_com = dhg.angular

        # gravity force
        g = np.array([0, 0, -9.81])
        m = pin.computeTotalMass(robot.model)
        F_g = m * g

        # isolate contact resultant force
        F_contact = F_com - F_g
        # shift torque from CoM -> support_plane_point on the plane
        tau_contact = tau_com + np.cross(com - ground_point, F_contact)

        eps = 1e-6
        # TODO gate ZMP when contact is unreliable (low Fz / contact loss)
        if abs(F_contact[2]) < eps:
            return np.zeros(3)

        # ZMP on support plane
        zmp_x = ground_point[0] - tau_contact[1] / F_contact[2]
        zmp_y = ground_point[1] + tau_contact[0] / F_contact[2]
        zmp = np.array([zmp_x, zmp_y, ground_height])

        return zmp

    def get_gravity_compensation(self, q: np.ndarray=None, imu_quat=None):
        robot = self.robot_model
        q = robot.state['q'] if q is None else q
        q_full = robot.full_q(q, imu_quat)
        tau_full = pin.rnea(robot.model,
                            robot.data,
                            q_full,
                            np.zeros(robot.model.nv),
                            np.zeros(robot.model.nv))
        # return motor-only torques (skip free-flyer)
        return tau_full[FREEFLYER_NV:]

    def get_frame_jacobian(self, frame_name: str, q: np.ndarray=None, imu_quat=None):
        '''
        Get the frame jacobian in the world frame
        q is motor-only (27), returns motor-only jacobian (6 x 27)
        '''
        robot = self.robot_model
        if q is not None:
            q_full = robot.full_q(q, imu_quat)
            data = robot.model.createData()
        else:
            q_full = robot.full_q(robot.state['q'])
            data = robot.data
        # update kinematics
        pin.forwardKinematics(robot.model, data, q_full)
        pin.updateFramePlacements(robot.model, data)
        # compute jacobian
        frame_id = robot.model.getFrameId(frame_name)
        jacobian_full = pin.computeFrameJacobian(
            robot.model,
            data,
            q_full,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        # return motor-only columns (skip free-flyer columns)
        return jacobian_full[:, FREEFLYER_NV:]

    def get_joint_jacobian(self, joint_name: str, q: np.ndarray=None):
        '''
        Get the joint jacobian in the local frame of the joint
        q is motor-only (27), returns motor-only jacobian (6 x 27)
        '''
        robot = self.robot_model
        if q is not None:
            q_full = robot.full_q(q)
            data = robot.model.createData()
        else:
            q_full = robot.full_q(robot.state['q'])
            data = robot.data
        # update kinematics
        pin.forwardKinematics(robot.model, data, q_full)
        pin.updateFramePlacements(robot.model, data)
        # compute jacobian
        joint_id = robot.model.getJointId(joint_name)
        jacobian_full = pin.computeJointJacobian(
            robot.model,
            data,
            q_full,
            joint_id
        )
        # return motor-only columns (skip free-flyer columns)
        return jacobian_full[:, FREEFLYER_NV:]

    def get_frame_twist(self, frame_name: str):
        robot = self.robot_model
        frame_id = robot.model.getFrameId(frame_name)
        twist = pin.getFrameVelocity(
            robot.model,
            robot.data,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.concatenate([twist.linear, twist.angular])

    def get_frame_wrench(self, frame_name: str,
                         q: np.ndarray=None, tau: np.ndarray=None, imu_quat=None):
        '''
        Estimate frame wrench with arm-only dynamics compensation and SVD damping

        This keeps bad configurations quiet by damping weak Jacobian directions
        instead of amplifying torque noise into large force readings
        '''
        robot = self.robot_model
        q = robot.state['q'] if q is None else q
        tau = robot.state['tau'] if tau is None else tau
        dq = robot.state['dq']
        ddq = self._estimate_wrench_ddq(dq)

        joint_idx = self._get_frame_wrench_joint_indices(frame_name)
        tau_model = self.get_motion_compensation(q, dq, imu_quat=imu_quat, ddq=ddq)
        tau_residual = tau[joint_idx] - tau_model[joint_idx]
        jac = self.get_frame_jacobian(frame_name, q, imu_quat)[:, joint_idx]
        wrench = self._solve_damped_svd_wrench(jac, tau_residual)
        wrench *= self._motion_quality(dq[joint_idx], ddq[joint_idx])
        return self._filter_frame_wrench(frame_name, wrench)

    def get_frame_wrench_damped(self, frame_name: str,
                                q: np.ndarray=None, tau: np.ndarray=None, imu_quat=None):
        '''
        Estimate wrench from joint torques using a condition-aware least-squares solve

        Uses adaptive damped least-squares when the Jacobian is ill-conditioned to
        reduce sensitivity to torque noise near singular configurations
        '''
        robot = self.robot_model
        cond_threshold = 10
        base_damping = 1e-1
        max_damping = 10
        svd_tol = 1e-3

        q = robot.state['q'] if q is None else q
        tau = robot.state['tau'] if tau is None else tau
        tau_gravity = self.get_gravity_compensation(q, imu_quat)
        jac = self.get_frame_jacobian(frame_name, q, imu_quat)
        tau_residual = tau - tau_gravity

        singular_values = np.linalg.svd(jac.T, compute_uv=False)
        if singular_values.size == 0 or singular_values[0] < svd_tol:
            return np.zeros(6)

        cond_number = singular_values[0] / max(singular_values[-1], svd_tol)
        if cond_number <= cond_threshold:
            damping = 0.0
        else:
            # increase damping smoothly once condition number passes threshold
            damping = base_damping * (cond_number / cond_threshold)
            damping = np.clip(damping, base_damping, max_damping)

        jj_t = jac @ jac.T
        rhs = jac @ tau_residual
        damping_sq = damping * damping
        wrench = np.linalg.solve(jj_t + damping_sq * np.eye(6), rhs)
        return wrench

    def _get_frame_wrench_joint_indices(self, frame_name: str):
        '''Select the arm joints that should explain a wrist wrench'''
        if frame_name.startswith('left_'):
            return LEFT_ARM_INDEX
        if frame_name.startswith('right_'):
            return RIGHT_ARM_INDEX
        return np.arange(NUM_MOTOR)

    def get_motion_compensation(self, q: np.ndarray=None,
                                dq: np.ndarray=None, imu_quat=None,
                                ddq: np.ndarray=None):
        '''Estimate model torque for motion, excluding external contact'''
        robot = self.robot_model
        q = robot.state['q'] if q is None else q
        dq = robot.state['dq'] if dq is None else dq
        ddq = np.zeros(NUM_MOTOR) if ddq is None else ddq
        q_full = robot.full_q(q, imu_quat)
        dq_full = robot.full_v(dq)
        ddq_full = robot.full_v(ddq)
        tau_full = pin.rnea(
            robot.model,
            robot.data,
            q_full,
            dq_full,
            ddq_full,
        )
        return tau_full[FREEFLYER_NV:]

    def _estimate_wrench_ddq(self, dq: np.ndarray):
        '''Estimate filtered joint acceleration for wrench compensation'''
        now = time.time()
        dq = np.asarray(dq, dtype=float)
        if self._wrench_last_dq is None:
            self._wrench_last_dq = np.copy(dq)
            self._wrench_last_time = now
            return np.copy(self._wrench_ddq)

        dt = now - self._wrench_last_time
        if dt < 1e-3:
            return np.copy(self._wrench_ddq)
        if dt > 0.2:
            # avoid a large derivative after pauses or debugger stops
            self._wrench_last_dq = np.copy(dq)
            self._wrench_last_time = now
            self._wrench_ddq[:] = 0.0
            return np.copy(self._wrench_ddq)

        ddq_raw = (dq - self._wrench_last_dq) / dt
        ddq_raw = np.clip(ddq_raw, -40.0, 40.0)
        self._wrench_ddq = 0.9 * self._wrench_ddq + 0.1 * ddq_raw
        self._wrench_last_dq = np.copy(dq)
        self._wrench_last_time = now
        return np.copy(self._wrench_ddq)

    @staticmethod
    def _motion_quality(dq: np.ndarray, ddq: np.ndarray):
        '''Fade unreliable readings during faster arm motion'''
        velocity_norm = np.linalg.norm(dq)
        accel_norm = np.linalg.norm(ddq)
        velocity_quality = np.clip(1.0 - velocity_norm / 5.0, 0.0, 1.0)
        accel_quality = np.clip(1.0 - accel_norm / 20.0, 0.0, 1.0)
        return velocity_quality * accel_quality

    def _filter_frame_wrench(self, frame_name: str, wrench: np.ndarray):
        '''Low-pass filter wrench readings per frame'''
        beta = 0.75
        previous = self._wrench_filtered.get(frame_name)
        if previous is None:
            filtered = np.copy(wrench)
        else:
            filtered = beta * previous + (1.0 - beta) * wrench
        self._wrench_filtered[frame_name] = filtered
        return np.copy(filtered)

    @staticmethod
    def _solve_damped_svd_wrench(jac: np.ndarray, tau_residual: np.ndarray):
        '''Solve jac.T * wrench = tau while suppressing weak directions'''
        damping = 1e-1
        svd_floor = 5e-2
        a = jac.T
        u, singular_values, vt = np.linalg.svd(a, full_matrices=False)
        if singular_values.size == 0 or singular_values[0] < svd_floor:
            return np.zeros(6)

        gains = singular_values / (singular_values * singular_values + damping * damping)
        gains[singular_values < svd_floor] = 0.0
        wrench = vt.T @ (gains * (u.T @ tau_residual))

        # fade the whole reading near singular poses so bad estimates look ignorable
        quality = np.clip(singular_values[-1] / svd_floor, 0.0, 1.0)
        return quality * wrench

    def get_frame_wrench_raw(self, frame_name: str,
                             q: np.ndarray=None, tau: np.ndarray=None, imu_quat=None):
        '''
        Pseudo-inverse solution without damping for reference/debugging
        '''
        robot = self.robot_model
        q = robot.state['q'] if q is None else q
        tau = robot.state['tau'] if tau is None else tau
        tau_gravity = self.get_gravity_compensation(q, imu_quat)
        jac = self.get_frame_jacobian(frame_name, q, imu_quat)
        tau_residual = tau - tau_gravity
        wrench = np.linalg.pinv(jac.T) @ tau_residual
        return wrench

    def compute_frame_twist(self, frame_name: str, dq: np.ndarray):
        jac = self.get_frame_jacobian(frame_name)
        twist = jac @ dq
        return twist
