'''The G1's joint limits, the source of robots/g1.yaml (tools/generate_robot_data.py).

29 body joints in slot order (Unitree's G1JointIndex): position in rad, velocity in rad/s,
torque in N*m. They are the limits of Unitree's model (g1_29dof_with_hand_rev_1_0.urdf in
CL_Assets/g1/g1_description), not measurements of our unit. That URDF gives the ankles and
waist roll/pitch 35 N*m where the MJCF gives 50 N*m; confirm on hardware before relying on
either. The Dex3-1 hands are not here: dex_3 keeps their joints.
'''

MOTOR_COUNT = 29
LOWCMD_MOTOR_SLOTS = 35  # array length of unitree_hg LowCmd_/LowState_

# LowCmd_.mode_pr: MODE_PR drives the ankle pitch/roll joints this table names; MODE_AB (1)
# would drive the two ankle motors directly
MODE_PR = 0

LEG_SLOTS = range(0, 12)
WAIST_SLOTS = range(12, 15)
LEFT_ARM_SLOTS = range(15, 22)
RIGHT_ARM_SLOTS = range(22, 29)

JOINT_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]

URDF_POSITION_LIMITS = [
    {'low': -2.5307, 'high': 2.8798},             # left_hip_pitch_joint
    {'low': -0.5236, 'high': 2.9671},             # left_hip_roll_joint
    {'low': -2.7576, 'high': 2.7576},             # left_hip_yaw_joint
    {'low': -0.087267, 'high': 2.8798},           # left_knee_joint
    {'low': -0.87267, 'high': 0.5236},            # left_ankle_pitch_joint
    {'low': -0.2618, 'high': 0.2618},             # left_ankle_roll_joint
    {'low': -2.5307, 'high': 2.8798},             # right_hip_pitch_joint
    {'low': -2.9671, 'high': 0.5236},             # right_hip_roll_joint
    {'low': -2.7576, 'high': 2.7576},             # right_hip_yaw_joint
    {'low': -0.087267, 'high': 2.8798},           # right_knee_joint
    {'low': -0.87267, 'high': 0.5236},            # right_ankle_pitch_joint
    {'low': -0.2618, 'high': 0.2618},             # right_ankle_roll_joint
    {'low': -2.618, 'high': 2.618},               # waist_yaw_joint
    {'low': -0.52, 'high': 0.52},                 # waist_roll_joint
    {'low': -0.52, 'high': 0.52},                 # waist_pitch_joint
    {'low': -3.0892, 'high': 2.6704},             # left_shoulder_pitch_joint
    {'low': -1.5882, 'high': 2.2515},             # left_shoulder_roll_joint
    {'low': -2.618, 'high': 2.618},               # left_shoulder_yaw_joint
    {'low': -1.0472, 'high': 2.0944},             # left_elbow_joint
    {'low': -1.972222054, 'high': 1.972222054},   # left_wrist_roll_joint
    {'low': -1.614429558, 'high': 1.614429558},   # left_wrist_pitch_joint
    {'low': -1.614429558, 'high': 1.614429558},   # left_wrist_yaw_joint
    {'low': -3.0892, 'high': 2.6704},             # right_shoulder_pitch_joint
    {'low': -2.2515, 'high': 1.5882},             # right_shoulder_roll_joint
    {'low': -2.618, 'high': 2.618},               # right_shoulder_yaw_joint
    {'low': -1.0472, 'high': 2.0944},             # right_elbow_joint
    {'low': -1.972222054, 'high': 1.972222054},   # right_wrist_roll_joint
    {'low': -1.614429558, 'high': 1.614429558},   # right_wrist_pitch_joint
    {'low': -1.614429558, 'high': 1.614429558},   # right_wrist_yaw_joint
]

URDF_VELOCITY_LIMITS = [
    32.0, # left_hip_pitch_joint
    20.0, # left_hip_roll_joint
    32.0, # left_hip_yaw_joint
    20.0, # left_knee_joint
    30.0, # left_ankle_pitch_joint
    30.0, # left_ankle_roll_joint
    32.0, # right_hip_pitch_joint
    20.0, # right_hip_roll_joint
    32.0, # right_hip_yaw_joint
    20.0, # right_knee_joint
    30.0, # right_ankle_pitch_joint
    30.0, # right_ankle_roll_joint
    32.0, # waist_yaw_joint
    30.0, # waist_roll_joint
    30.0, # waist_pitch_joint
    37.0, # left_shoulder_pitch_joint
    37.0, # left_shoulder_roll_joint
    37.0, # left_shoulder_yaw_joint
    37.0, # left_elbow_joint
    37.0, # left_wrist_roll_joint
    22.0, # left_wrist_pitch_joint
    22.0, # left_wrist_yaw_joint
    37.0, # right_shoulder_pitch_joint
    37.0, # right_shoulder_roll_joint
    37.0, # right_shoulder_yaw_joint
    37.0, # right_elbow_joint
    37.0, # right_wrist_roll_joint
    22.0, # right_wrist_pitch_joint
    22.0, # right_wrist_yaw_joint
]

URDF_TORQUE_LIMITS = [
    88.0,  # left_hip_pitch_joint
    139.0, # left_hip_roll_joint
    88.0,  # left_hip_yaw_joint
    139.0, # left_knee_joint
    35.0,  # left_ankle_pitch_joint
    35.0,  # left_ankle_roll_joint
    88.0,  # right_hip_pitch_joint
    139.0, # right_hip_roll_joint
    88.0,  # right_hip_yaw_joint
    139.0, # right_knee_joint
    35.0,  # right_ankle_pitch_joint
    35.0,  # right_ankle_roll_joint
    88.0,  # waist_yaw_joint
    35.0,  # waist_roll_joint
    35.0,  # waist_pitch_joint
    25.0,  # left_shoulder_pitch_joint
    25.0,  # left_shoulder_roll_joint
    25.0,  # left_shoulder_yaw_joint
    25.0,  # left_elbow_joint
    25.0,  # left_wrist_roll_joint
    5.0,   # left_wrist_pitch_joint
    5.0,   # left_wrist_yaw_joint
    25.0,  # right_shoulder_pitch_joint
    25.0,  # right_shoulder_roll_joint
    25.0,  # right_shoulder_yaw_joint
    25.0,  # right_elbow_joint
    25.0,  # right_wrist_roll_joint
    5.0,   # right_wrist_pitch_joint
    5.0,   # right_wrist_yaw_joint
]

assert len(JOINT_NAMES) == MOTOR_COUNT
assert len(URDF_POSITION_LIMITS) == MOTOR_COUNT
assert len(URDF_VELOCITY_LIMITS) == MOTOR_COUNT
assert len(URDF_TORQUE_LIMITS) == MOTOR_COUNT
