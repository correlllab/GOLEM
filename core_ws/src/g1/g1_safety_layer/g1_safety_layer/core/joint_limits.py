'''Per-joint baseline limits used to derive safety bounds'''

from g1_description.joints import BODY
from g1_description.layout import (
    ARM_SDK_SLOTS,
    ARM_SDK_WEIGHT_SLOT,
    LEG_SLOTS,
    MODE_PR,
    NUM_BODY_MOTORS,
    WAIST_SLOTS,
)

# G1 URDF limits in rt/lowcmd slot order, from the shared g1_description table
MOTOR_COUNT = NUM_BODY_MOTORS

JOINT_NAMES = [joint.name for joint in BODY]

URDF_POSITION_LIMITS = [{'low': joint.lower, 'high': joint.upper} for joint in BODY]

URDF_VELOCITY_LIMITS = [joint.velocity for joint in BODY]

URDF_TORQUE_LIMITS = [joint.effort for joint in BODY]

# split_mode takes legs from the lower channel and waist + arms from the upper
SPLIT_UPPER_START = LEG_SLOTS.stop

__all__ = [
    'ARM_SDK_SLOTS',
    'ARM_SDK_WEIGHT_SLOT',
    'JOINT_NAMES',
    'LEG_SLOTS',
    'MODE_PR',
    'MOTOR_COUNT',
    'SPLIT_UPPER_START',
    'URDF_POSITION_LIMITS',
    'URDF_TORQUE_LIMITS',
    'URDF_VELOCITY_LIMITS',
    'WAIST_SLOTS',
]
