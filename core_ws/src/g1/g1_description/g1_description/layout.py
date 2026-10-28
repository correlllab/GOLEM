"""Which DDS motor slot is which on the G1, for rt/lowcmd, rt/lowstate and Dex3.

Slot numbers are Unitree's (G1JointIndex in unitree_sdk2_python/example/g1/).
Two LowCmd_ header fields also matter on the G1:

- mode_machine: echo the value read from rt/lowstate (our unit reports 5).
- mode_pr: MODE_PR drives the ankle pitch/roll joints this table names;
  MODE_AB would drive the two ankle motors directly.
"""

from .joints import BODY, LEFT_HAND, RIGHT_HAND

NUM_BODY_MOTORS = 29
LOWCMD_MOTOR_SLOTS = 35  # array length of unitree_hg LowCmd_/LowState_
NUM_HAND_MOTORS = 7

MODE_PR = 0
MODE_AB = 1

LEG_SLOTS = range(0, 12)
WAIST_SLOTS = range(12, 15)
LEFT_ARM_SLOTS = range(15, 22)
RIGHT_ARM_SLOTS = range(22, 29)

BODY_JOINT_NAMES = tuple(j.name for j in BODY)
SLOT_BY_NAME = {j.name: j.slot for j in BODY}
HAND_JOINT_NAMES = {
    'left': tuple(j.name for j in LEFT_HAND),
    'right': tuple(j.name for j in RIGHT_HAND),
}
