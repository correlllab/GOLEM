'''
RobotDynamics: dynamics on a RobotModel's free-flyer model: the gravity-compensation
torques the controller sends as feed-forward, the end-effector twists its speed limits use,
and the centre of mass its recordings log.

Every q, dq and tau here is per body joint, in /lowcmd slot order; the free-flyer's base
coordinates are added and stripped by RobotModel.full_q and full_v.
'''
import numpy as np
import pinocchio as pin

# Pinocchio's free-flyer layout: q = [x, y, z, qx, qy, qz, qw, joints...] and
# v = [vx, vy, vz, wx, wy, wz, joints...]
FREEFLYER_NQ = 7
FREEFLYER_NV = 6
FREEFLYER_QUAT = slice(3, 7)  # x, y, z, w


class RobotDynamics:
    def __init__(self, robot_model):
        self.robot_model = robot_model

    def get_com(self, q: np.ndarray=None):
        robot = self.robot_model
        q = robot.state['q'] if q is None else q
        q_full = robot.full_q(q)
        com = pin.centerOfMass(robot.model, robot.data, q_full)
        return com

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
        q is per body joint; returns the body-joint columns (6 x body joints)
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

    def compute_frame_twist(self, frame_name: str, dq: np.ndarray):
        jac = self.get_frame_jacobian(frame_name)
        twist = jac @ dq
        return twist
