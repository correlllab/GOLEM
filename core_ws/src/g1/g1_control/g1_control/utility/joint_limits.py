import numpy as np

from g1_description.joints import BODY

from g1_control.utility.joint_definition import BODY_JOINTS, UPPER_BODY_INDEX

# Joint limits use the G1 URDF values from g1_description, in rt/lowcmd slot order
JOINT_POSITION_LIMITS = [{'low': joint.lower, 'high': joint.upper} for joint in BODY]

JOINT_VELOCITY_LIMITS = [joint.velocity for joint in BODY]

JOINT_TORQUE_LIMITS = [joint.effort for joint in BODY]

# fallback gains when a config gives none: Unitree's rt/arm_sdk example
# (unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py)
DEFAULT_UPPER_KP = 60.0
DEFAULT_UPPER_KD = 1.5


def get_default_gains():
    kp = np.zeros(len(BODY_JOINTS), dtype=np.float32)
    kd = np.zeros(len(BODY_JOINTS), dtype=np.float32)
    kp[UPPER_BODY_INDEX] = DEFAULT_UPPER_KP
    kd[UPPER_BODY_INDEX] = DEFAULT_UPPER_KD

    return kp, kd


def setup_gains(command_publisher, gains_config=None):
    '''Configure motor gains for all joints'''
    if (gains_config is not None and
        gains_config.get('kp') is not None and
        gains_config.get('kd') is not None):
        command_publisher.kp[:] = np.asarray(gains_config['kp'], dtype=np.float32)
        command_publisher.kd[:] = np.asarray(gains_config['kd'], dtype=np.float32)
        return

    default_kp, default_kd = get_default_gains()
    command_publisher.kp[:] = default_kp
    command_publisher.kd[:] = default_kd
