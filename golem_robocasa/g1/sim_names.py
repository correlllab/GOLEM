"""Map the G1's Unitree DDS motor slots, pelvis IMU and base to indices in its MuJoCo model.

RoboCasa builds a task around the ``G1Dex3`` robot with every name prefixed
``robot0_`` (no joint is renamed) and the actuators in MJCF order, which puts
each hand after its arm and orders the right hand's fingers differently. The
DDS side speaks Unitree's motor slots: slot i of rt/lowcmd / rt/lowstate is
BODY_JOINTS[i], and slot i of rt/dex3/<side>/cmd / state is
HAND_JOINTS[side][i]. Everything is resolved by name, so nothing depends on the
model's index order. ``robot_prefix=""`` resolves the standalone G1 MJCF in
CL_Assets/g1.

The tables are the sim's own copies of golem_safety's robots/g1.yaml slot
order and dex_3's HAND_JOINTS order: the robocasa container mounts neither
package.
"""
from typing import NamedTuple

import mujoco
import numpy as np

# rt/lowcmd / rt/lowstate slot order.
BODY_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# rt/dex3/<side>/cmd and state motor order.
HAND_ORDER = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
HAND_JOINTS = {
    side: tuple(f"{side}_hand_{finger}_joint" for finger in HAND_ORDER)
    for side in ("left", "right")
}

FREE_JOINT = "floating_base_joint"

# The pelvis IMU at site imu_in_pelvis: the MJCF's gyro and accelerometer, and
# the framequat g1_robosuite adds.
IMU_SENSORS = {
    "quat": "imu-pelvis-orientation",
    "gyro": "imu-pelvis-angular-velocity",
    "acc": "imu-pelvis-linear-acceleration",
}


def _id(model, objtype, name):
    """Return the id of the named object of objtype, or -1."""
    return mujoco.mj_name2id(model, objtype, name)


class Motors(NamedTuple):
    """Hold the model indices of a group of motors, in DDS slot order."""
    names: tuple        # unprefixed joint names
    qpos: np.ndarray    # qpos address of each joint
    qvel: np.ndarray    # dof address of each joint
    ctrl: np.ndarray    # actuator id of each joint's motor
    tau_min: np.ndarray  # N*m, the joint's actuatorfrcrange (-inf when unlimited)
    tau_max: np.ndarray  # N*m (+inf when unlimited)


class NameResolver:
    """Resolve the G1's body and hand motors, pelvis IMU and free base in an MjModel.

    body and hands[side] are Motors; sensor_adr maps quat, gyro and acc to
    (sensordata address, dimension) for the IMU sensors the model has (the
    standalone MJCF has no framequat); base_body, base_qpos and base_qvel locate
    the pelvis free joint.
    """

    def __init__(self, model, robot_prefix="robot0_"):
        """Resolve everything in model, raising KeyError for a missing motor or free joint.

        robot_prefix is 'robot0_' in a RoboCasa env, '' for the standalone G1 MJCF.
        """
        self.model = model
        self.robot_prefix = robot_prefix
        self.body = self._motors(BODY_JOINTS)
        self.hands = {side: self._motors(names) for side, names in HAND_JOINTS.items()}

        self.sensor_adr = {}
        for key, name in IMU_SENSORS.items():
            sid = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, robot_prefix + name)
            if sid >= 0:
                self.sensor_adr[key] = (int(model.sensor_adr[sid]), int(model.sensor_dim[sid]))

        fj = _id(model, mujoco.mjtObj.mjOBJ_JOINT, robot_prefix + FREE_JOINT)
        if fj < 0 or model.jnt_type[fj] != mujoco.mjtJoint.mjJNT_FREE:
            raise KeyError(f"free joint not found in model: {robot_prefix + FREE_JOINT}")
        self.base_body = int(model.jnt_bodyid[fj])
        self.base_qpos = int(model.jnt_qposadr[fj])
        self.base_qvel = int(model.jnt_dofadr[fj])

    def _motors(self, names):
        """Return the Motors of the joints names, each of which must carry its own motor."""
        model = self.model
        qpos, qvel, ctrl, tau_min, tau_max = [], [], [], [], []
        for name in names:
            sim_name = self.robot_prefix + name
            jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
            if jid < 0:
                raise KeyError(f"joint not found in model: {sim_name}")
            # Each actuator is named after its joint.
            aid = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, sim_name)
            if aid < 0 or model.actuator_trnid[aid, 0] != jid:
                raise KeyError(f"no motor on joint {sim_name}")
            qpos.append(model.jnt_qposadr[jid])
            qvel.append(model.jnt_dofadr[jid])
            ctrl.append(aid)
            low, high = model.jnt_actfrcrange[jid] if model.jnt_actfrclimited[jid] else (-np.inf, np.inf)
            tau_min.append(low)
            tau_max.append(high)
        return Motors(tuple(names), np.array(qpos), np.array(qvel), np.array(ctrl),
                      np.array(tau_min, dtype=float), np.array(tau_max, dtype=float))

    def site_id(self, name):
        """Return the id of the robot's site name (unprefixed), or -1."""
        return _id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.robot_prefix + name)

    def body_id(self, name):
        """Return the id of the robot's body name (unprefixed), or -1."""
        return _id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.robot_prefix + name)
