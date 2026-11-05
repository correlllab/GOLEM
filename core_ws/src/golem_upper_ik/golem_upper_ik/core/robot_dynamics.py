"""Compute dynamics on a RobotModel's free-flyer model.

RobotDynamics gives the gravity-compensation torques the controller sends as feed-forward
and the end-effector twists its speed limits use. Every q, dq and tau here is per body
joint, in /lowcmd slot order; RobotModel.full_q and full_v add the free-flyer's base
coordinates, and the results drop them.
"""
import numpy as np
import pinocchio as pin

# Pinocchio's free-flyer layout: q = [x, y, z, qx, qy, qz, qw, joints...] and
# v = [vx, vy, vz, wx, wy, wz, joints...]
FREEFLYER_NQ = 7
FREEFLYER_NV = 6
FREEFLYER_QUAT = slice(3, 7)  # x, y, z, w


class RobotDynamics:
    """Compute gravity torques, frame Jacobians and frame twists for one RobotModel."""

    def __init__(self, robot_model):
        """Compute on robot_model's free-flyer model and measured state."""
        self.robot_model = robot_model

    def get_gravity_compensation(self, q: np.ndarray=None, imu_quat=None):
        """Return the body joints' gravity torques at q (default: measured), N*m.

        The pelvis is oriented by imu_quat (w, x, y, z), default the measured IMU (full_q).
        """
        robot = self.robot_model
        q = robot.state['q'] if q is None else q
        q_full = robot.full_q(q, imu_quat)
        tau_full = pin.rnea(robot.model,
                            robot.data,
                            q_full,
                            np.zeros(robot.model.nv),
                            np.zeros(robot.model.nv))
        return tau_full[FREEFLYER_NV:]

    def get_frame_jacobian(self, frame_name: str, q: np.ndarray=None, imu_quat=None):
        """Return a frame's 6 x body-joint Jacobian, world-aligned at the frame (linear first).

        At q (on data of its own) or, by default, at the measured state on the model's data.
        """
        robot = self.robot_model
        if q is not None:
            q_full = robot.full_q(q, imu_quat)
            data = robot.model.createData()
        else:
            q_full = robot.full_q(robot.state['q'])
            data = robot.data
        pin.forwardKinematics(robot.model, data, q_full)
        pin.updateFramePlacements(robot.model, data)
        frame_id = robot.model.getFrameId(frame_name)
        jacobian_full = pin.computeFrameJacobian(
            robot.model,
            data,
            q_full,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return jacobian_full[:, FREEFLYER_NV:]

    def compute_frame_twist(self, frame_name: str, dq: np.ndarray):
        """Return a frame's twist (m/s, then rad/s) for body joint velocities dq, measured q."""
        jac = self.get_frame_jacobian(frame_name)
        twist = jac @ dq
        return twist
