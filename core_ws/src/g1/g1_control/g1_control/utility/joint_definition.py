'''G1 joint names and index groups, from the shared g1_description table'''

from g1_description.joints import BODY, LEFT_HAND, RIGHT_HAND
from g1_description.layout import (
    LEFT_ARM_SLOTS,
    LEG_SLOTS,
    NUM_BODY_MOTORS,
    RIGHT_ARM_SLOTS,
    WAIST_SLOTS,
)

# free-flyer base offsets
# q vector: [x, y, z, qx, qy, qz, qw] then motor joints
# v vector: [vx, vy, vz, wx, wy, wz] then motor joints
FREEFLYER_NQ = 7  # position (3) + quaternion xyzw (4)
FREEFLYER_NV = 6  # linear vel (3) + angular vel (3)

# slice indices for free-flyer q vector (Pinocchio xyzw convention)
FREEFLYER_POS = slice(0, 3)   # [x, y, z]
FREEFLYER_QUAT = slice(3, 7)  # [qx, qy, qz, qw]

# motor degrees of freedom on the Unitree DDS rt/lowstate / rt/lowcmd path.
# the Dex3-1 hands are on their own rt/dex3/* topics, not on this path
NUM_MOTOR = NUM_BODY_MOTORS

# body joints in rt/lowcmd slot order; with every other URDF joint frozen,
# Pinocchio orders the model's joints the same way (tested)
BODY_JOINTS = [joint.name for joint in BODY]

# Dex3-1 finger joints in the with-hand URDF; frozen out of the controller models
HAND_JOINTS = [joint.name for joint in LEFT_HAND + RIGHT_HAND]

LOWER_BODY_JOINTS = [BODY_JOINTS[i] for i in LEG_SLOTS]

WAIST_JOINTS = [BODY_JOINTS[i] for i in WAIST_SLOTS]

LEFT_ARM_JOINTS = [BODY_JOINTS[i] for i in LEFT_ARM_SLOTS]

RIGHT_ARM_JOINTS = [BODY_JOINTS[i] for i in RIGHT_ARM_SLOTS]

# waist + arms: everything rt/arm_sdk drives
UPPER_BODY_JOINTS = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

# joints the IK solves for; the waist is held at controller.waist_target
ENABLED_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

# end-effector frames used for velocity limiting and the ee pose topics
LEFT_EE_FRAME = 'left_hand_palm_link'
RIGHT_EE_FRAME = 'right_hand_palm_link'

UPPER_BODY_INDEX = [BODY_JOINTS.index(joint) for joint in UPPER_BODY_JOINTS]
LOWER_BODY_INDEX = [BODY_JOINTS.index(joint) for joint in LOWER_BODY_JOINTS]
WAIST_INDEX = list(WAIST_SLOTS)
LEFT_ARM_INDEX = list(LEFT_ARM_SLOTS)
RIGHT_ARM_INDEX = list(RIGHT_ARM_SLOTS)
