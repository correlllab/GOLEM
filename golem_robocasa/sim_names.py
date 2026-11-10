"""Map the H1-2's Unitree DDS motor slots and IMU to indices in its MuJoCo model.

RoboCasa builds a task around the ``H1_2`` robot with every name prefixed
(``robot0_``) and the leg joints renamed (``left_hip_yaw_joint`` ->
``left_leg_hip_yaw_joint``), in robosuite's index order. The DDS side speaks
the original Unitree names in golem_safety's 27-slot H1-2 order
(core_ws/src/golem_safety/robots/h12.yaml), so the DDS interface and the sim
loop resolve every index by name here.
"""
import mujoco
import numpy as np

# rt/lowcmd / rt/lowstate slot order.
ROS_MOTOR_ORDER = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# The leg renaming CL_Assets/robosuite_assets/build_assets.py applies.
LEG_JOINT_RENAME = {
    f"{side}_{j}_joint": f"{side}_leg_{j}_joint"
    for side in ("left", "right")
    for j in ("hip_yaw", "hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")
}


# RoboCasa's prefix of every robot name.
ROBOT_PREFIX = "robot0_"


def _id(model, objtype, name):
    """Return the id of the named object of objtype, or -1."""
    return mujoco.mj_name2id(model, objtype, name)


class NameResolver:
    """Resolve the 27 body motors, the pelvis IMU and bodies of an H1-2 MjModel.

    motor_ctrl, motor_qpos, motor_qvel and motor_tau hold, per DDS slot, the
    actuator, qpos address, dof address and torque sensordata address.
    sensor_adr maps imu_quat, imu_gyro and imu_acc to (sensordata address,
    dimension) when the model has them.
    """

    def __init__(self, model):
        """Resolve every motor in model, raising KeyError for a missing one."""
        self.model = model

        self.motor_ctrl = np.empty(len(ROS_MOTOR_ORDER), dtype=int)
        self.motor_qpos = np.empty(len(ROS_MOTOR_ORDER), dtype=int)
        self.motor_qvel = np.empty(len(ROS_MOTOR_ORDER), dtype=int)
        self.motor_tau = np.empty(len(ROS_MOTOR_ORDER), dtype=int)
        for i, ros_name in enumerate(ROS_MOTOR_ORDER):
            sim_joint = ROBOT_PREFIX + LEG_JOINT_RENAME.get(ros_name, ros_name)
            jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_joint)
            if jid < 0:
                raise KeyError(f"joint not found in model: {sim_joint}")
            self.motor_qpos[i] = model.jnt_qposadr[jid]
            self.motor_qvel[i] = model.jnt_dofadr[jid]
            # Each actuator is named after its joint.
            aid = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, sim_joint)
            if aid < 0:
                raise KeyError(f"actuator not found in model: {sim_joint}")
            self.motor_ctrl[i] = aid
            # The <jointactuatorfrc> sensor "<ros_name>_torque" (not leg-renamed)
            # reports the torque applied after the joint's actuatorfrcrange
            # clamp, as the real motor's measured torque does, not the demand.
            tau_sensor = ROBOT_PREFIX + ros_name + "_torque"
            sid = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, tau_sensor)
            if sid < 0:
                raise KeyError(f"torque sensor not found in model: {tau_sensor}")
            self.motor_tau[i] = model.sensor_adr[sid]

        self.sensor_adr = {}
        for base in ("imu_quat", "imu_gyro", "imu_acc"):
            sid = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, ROBOT_PREFIX + base)
            if sid >= 0:
                self.sensor_adr[base] = (model.sensor_adr[sid], model.sensor_dim[sid])

    def body_id(self, name):
        """Return the id of the robot's body name (unprefixed), or -1."""
        return _id(self.model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_PREFIX + name)
