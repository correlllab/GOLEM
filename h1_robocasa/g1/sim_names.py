"""Name resolution between the Unitree DDS interfaces and the G1 MuJoCo model.

When RoboCasa builds a task around the ``G1Dex3`` robot, every joint, actuator,
sensor, body and site of the robot gets the ``robot0_`` prefix (no joint is
renamed), and the actuator order follows the MJCF, which interleaves each hand
after its arm and orders the right hand's fingers differently. The DDS side
speaks Unitree's motor slots instead: slot i of rt/lowcmd / rt/lowstate is
BODY_JOINTS[i], and slot i of rt/dex3/<side>/cmd / state is HAND_JOINTS[side][i].
This module maps one to the other by name, so nothing depends on the model's
index order.

The tables match core_ws/src/g1/g1_description/g1_description/joints.py (generated
from the leprechaun_g1 URDF), which the robocasa container does not mount.

Works on the standalone leprechaun_g1 MJCF with ``robot_prefix=""``.
"""
from typing import NamedTuple

import mujoco
import numpy as np

# rt/lowcmd / rt/lowstate slot order (G1JointIndex in Unitree's SDK examples)
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

# rt/dex3/<side>/cmd HandCmd_ motor order
HAND_ORDER = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
HAND_JOINTS = {
    side: tuple(f"{side}_hand_{finger}_joint" for finger in HAND_ORDER)
    for side in ("left", "right")
}

FREE_JOINT = "floating_base_joint"

# the pelvis IMU (site imu_in_pelvis): the file's gyro and accelerometer, plus
# the framequat g1_robosuite adds
IMU_SENSORS = {
    "quat": "imu-pelvis-orientation",
    "gyro": "imu-pelvis-angular-velocity",
    "acc": "imu-pelvis-linear-acceleration",
}


def _id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


class Motors(NamedTuple):
    """Model indices of a group of motors, in DDS slot order."""
    names: tuple        # unprefixed joint names
    qpos: np.ndarray    # qpos address of each joint
    qvel: np.ndarray    # dof address of each joint
    ctrl: np.ndarray    # the motor actuating each joint
    tau_min: np.ndarray  # the joint's actuatorfrcrange
    tau_max: np.ndarray


class NameResolver:
    """Resolves the G1's DDS motors, pelvis IMU and base to indices in a
    (possibly robosuite-merged) MjModel. ``robot_prefix`` is e.g. 'robot0_'; use
    '' for the standalone leprechaun_g1 MJCF."""

    def __init__(self, model, robot_prefix="robot0_"):
        self.model = model
        self.robot_prefix = robot_prefix
        self.body = self._motors(BODY_JOINTS)
        self.hands = {side: self._motors(names) for side, names in HAND_JOINTS.items()}

        # IMU sensors -> sensordata address + dim (the standalone MJCF has no framequat)
        self.sensor_adr = {}
        for key, name in IMU_SENSORS.items():
            sid = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, robot_prefix + name)
            if sid >= 0:
                self.sensor_adr[key] = (int(model.sensor_adr[sid]), int(model.sensor_dim[sid]))

        # the pelvis free joint: its body, and where its pose and twist live
        fj = _id(model, mujoco.mjtObj.mjOBJ_JOINT, robot_prefix + FREE_JOINT)
        if fj < 0 or model.jnt_type[fj] != mujoco.mjtJoint.mjJNT_FREE:
            raise KeyError(f"free joint not found in model: {robot_prefix + FREE_JOINT}")
        self.base_body = int(model.jnt_bodyid[fj])
        self.base_qpos = int(model.jnt_qposadr[fj])
        self.base_qvel = int(model.jnt_dofadr[fj])

    def _motors(self, names):
        model = self.model
        qpos, qvel, ctrl, tau_min, tau_max = [], [], [], [], []
        for name in names:
            sim_name = self.robot_prefix + name
            jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
            if jid < 0:
                raise KeyError(f"joint not found in model: {sim_name}")
            # actuator name == joint name in the source MJCF
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
        return _id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.robot_prefix + name)

    def body_id(self, name):
        return _id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.robot_prefix + name)

    def strip(self, name):
        """Strip the robot prefix from a sim name for ROS-facing TF frame_ids."""
        if name and name.startswith(self.robot_prefix):
            return name[len(self.robot_prefix):]
        return name
