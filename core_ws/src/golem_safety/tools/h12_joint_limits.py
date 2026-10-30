'''The H1-2's joint limits, the source of robots/h12.yaml (tools/generate_robot_data.py).

27 body joints in slot order: position in rad, velocity in rad/s, torque in N*m. These are
the limits the robot runs, not the URDF's: shoulder roll is +-0.38 here and +-0.19 in
h1_2_magpie.urdf.
'''

MOTOR_COUNT = 27

JOINT_NAMES = [
    'left_hip_yaw_joint',
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_yaw_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'torso_joint',
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
    {'low': -0.43, 'high': 0.43},         # left_hip_yaw_joint
    {'low': -3.14, 'high': 2.5},          # left_hip_pitch_joint
    {'low': -0.43, 'high': 3.14},         # left_hip_roll_joint
    {'low': -0.12, 'high': 2.19},         # left_knee_joint
    {'low': -0.897334, 'high': 0.523598}, # left_ankle_pitch_joint
    {'low': -0.261799, 'high': 0.261799}, # left_ankle_roll_joint
    {'low': -0.43, 'high': 0.43},         # right_hip_yaw_joint
    {'low': -3.14, 'high': 2.5},          # right_hip_pitch_joint
    {'low': -3.14, 'high': 0.43},         # right_hip_roll_joint
    {'low': -0.12, 'high': 2.19},         # right_knee_joint
    {'low': -0.897334, 'high': 0.523598}, # right_ankle_pitch_joint
    {'low': -0.261799, 'high': 0.261799}, # right_ankle_roll_joint
    {'low': -2.35, 'high': 2.35},         # torso_joint
    {'low': -3.14, 'high': 1.57},         # left_shoulder_pitch_joint
    {'low': -0.38, 'high': 3.4},          # left_shoulder_roll_joint
    {'low': -2.66, 'high': 3.01},         # left_shoulder_yaw_joint
    {'low': -0.95, 'high': 3.18},         # left_elbow_joint
    {'low': -3.01, 'high': 2.75},         # left_wrist_roll_joint
    {'low': -0.4625, 'high': 0.4625},     # left_wrist_pitch_joint
    {'low': -1.27, 'high': 1.27},         # left_wrist_yaw_joint
    {'low': -3.14, 'high': 1.57},         # right_shoulder_pitch_joint
    {'low': -3.4, 'high': 0.38},          # right_shoulder_roll_joint
    {'low': -3.01, 'high': 2.66},         # right_shoulder_yaw_joint
    {'low': -0.95, 'high': 3.18},         # right_elbow_joint
    {'low': -2.75, 'high': 3.01},         # right_wrist_roll_joint
    {'low': -0.4625, 'high': 0.4625},     # right_wrist_pitch_joint
    {'low': -1.27, 'high': 1.27},         # right_wrist_yaw_joint
]

URDF_VELOCITY_LIMITS = [
    23.0, # left_hip_yaw_joint
    23.0, # left_hip_pitch_joint
    23.0, # left_hip_roll_joint
    14.0, # left_knee_joint
    9.0,  # left_ankle_pitch_joint
    9.0,  # left_ankle_roll_joint
    23.0, # right_hip_yaw_joint
    23.0, # right_hip_pitch_joint
    23.0, # right_hip_roll_joint
    14.0, # right_knee_joint
    9.0,  # right_ankle_pitch_joint
    9.0,  # right_ankle_roll_joint
    23.0, # torso_joint
    9.0,  # left_shoulder_pitch_joint
    9.0,  # left_shoulder_roll_joint
    20.0, # left_shoulder_yaw_joint
    20.0, # left_elbow_joint
    31.4, # left_wrist_roll_joint
    31.4, # left_wrist_pitch_joint
    31.4, # left_wrist_yaw_joint
    9.0,  # right_shoulder_pitch_joint
    9.0,  # right_shoulder_roll_joint
    20.0, # right_shoulder_yaw_joint
    20.0, # right_elbow_joint
    31.4, # right_wrist_roll_joint
    31.4, # right_wrist_pitch_joint
    31.4, # right_wrist_yaw_joint
]

URDF_TORQUE_LIMITS = [
    200.0, # left_hip_yaw_joint
    200.0, # left_hip_pitch_joint
    200.0, # left_hip_roll_joint
    300.0, # left_knee_joint
    60.0,  # left_ankle_pitch_joint
    40.0,  # left_ankle_roll_joint
    200.0, # right_hip_yaw_joint
    200.0, # right_hip_pitch_joint
    200.0, # right_hip_roll_joint
    300.0, # right_knee_joint
    60.0,  # right_ankle_pitch_joint
    40.0,  # right_ankle_roll_joint
    200.0, # torso_joint
    40.0,  # left_shoulder_pitch_joint
    40.0,  # left_shoulder_roll_joint
    18.0,  # left_shoulder_yaw_joint
    18.0,  # left_elbow_joint
    19.0,  # left_wrist_roll_joint
    19.0,  # left_wrist_pitch_joint
    19.0,  # left_wrist_yaw_joint
    40.0,  # right_shoulder_pitch_joint
    40.0,  # right_shoulder_roll_joint
    18.0,  # right_shoulder_yaw_joint
    18.0,  # right_elbow_joint
    19.0,  # right_wrist_roll_joint
    19.0,  # right_wrist_pitch_joint
    19.0,  # right_wrist_yaw_joint
]

assert len(JOINT_NAMES) == MOTOR_COUNT
assert len(URDF_POSITION_LIMITS) == MOTOR_COUNT
assert len(URDF_VELOCITY_LIMITS) == MOTOR_COUNT
assert len(URDF_TORQUE_LIMITS) == MOTOR_COUNT
