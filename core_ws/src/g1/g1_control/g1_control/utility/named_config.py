import numpy as np

from g1_control.utility.joint_definition import ENABLED_JOINTS

# Named arm postures for the 14 IK joints (ENABLED_JOINTS); the waist is held
# separately at controller.waist_target. Values are rad and inside the URDF limits.
#
# index: 0 L_sh_pitch 1 L_sh_roll 2 L_sh_yaw 3 L_elbow 4 L_wr_roll 5 L_wr_pitch 6 L_wr_yaw
#        7 R_sh_pitch 8 R_sh_roll 9 R_sh_yaw 10 R_elbow 11 R_wr_roll 12 R_wr_pitch 13 R_wr_yaw
#
# G1 conventions, checked with forward kinematics on the with-hand URDF:
# - all zeros is upper arms down, elbows at 90 deg, forearms forward (palms
#   ~0.24 m in front of the pelvis at waist height)
# - elbow +pi/2 straightens the arm downward
# - shoulder pitch < 0 raises the arm forward
# - shoulder roll > 0 (left) / < 0 (right) raises the arms out to the sides
HALF_PI = np.pi / 2

NAMED_CONFIGS = {
    'home':  # zero posture, forearms forward; the SDK example's first stage
        np.zeros(len(ENABLED_JOINTS)),
    'arms_down':  # straight arms hanging below the shoulders
        np.array([0.0, 0.0, 0.0, HALF_PI, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, HALF_PI, 0.0, 0.0, 0.0]),
    't_pose':  # straight arms out to the sides, horizontal; the SDK example's lift target
        np.array([0.0, HALF_PI, 0.0, HALF_PI, 0.0, 0.0, 0.0,
                  0.0, -HALF_PI, 0.0, HALF_PI, 0.0, 0.0, 0.0]),
    'arms_front_45':  # straight arms, 45 deg forward and down
        np.array([-np.pi / 4, 0.0, 0.0, HALF_PI, 0.0, 0.0, 0.0,
                  -np.pi / 4, 0.0, 0.0, HALF_PI, 0.0, 0.0, 0.0]),
}

# postures the controller moves through when it starts
STARTUP_SEQUENCE = ('home',)
