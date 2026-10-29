'''The G1's controller data: the 29-DoF model with both Dex3-1 hands, its frames and postures'''
from golem_ros2_models import paths
import numpy as np

# the robot's name in messages
DISPLAY_NAME = 'G1'

# joints the IK solves for, in the order of every named configuration
ENABLED_JOINTS = (
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
)
# upper-body joints the IK does not move (its reduced model locks them at their measured
# angles at each goal's start); every command holds them at controller.hold_targets
HELD_JOINTS = ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint')
# golem_safety writes /lowcmd in every mode, and a walking policy may balance the legs,
# trained with a stiff waist: the controller's shutdown command keeps the waist at its hold,
# and only the arms go limp.
SHUTDOWN_KEEPS_HELD_JOINTS_ON_LOWCMD = True
# the Dex3-1 palms: end effectors for velocity limiting and the ee pose topics, and the
# planner's frames
EE_FRAMES = ('left_hand_palm_link', 'right_hand_palm_link')
PLANNER_FRAMES = ('left_hand_palm_link', 'right_hand_palm_link')
# frame of the /lowstate IMU. To verify on the robot: if turning the waist changes the IMU
# yaw while the pelvis stays still, the IMU is imu_in_torso (controller.imu_frame overrides)
IMU_FRAME = 'imu_in_pelvis'

HALF_PI = np.pi / 2
# Named arm postures for the 14 IK joints, rad, inside the URDF limits:
#   0 L_sh_pitch 1 L_sh_roll 2 L_sh_yaw 3 L_elbow 4 L_wr_roll 5 L_wr_pitch 6 L_wr_yaw
#   7 R_sh_pitch 8 R_sh_roll 9 R_sh_yaw 10 R_elbow 11 R_wr_roll 12 R_wr_pitch 13 R_wr_yaw
# G1 conventions, checked with forward kinematics on the with-hand URDF:
# - all zeros is upper arms down, elbows at 90 deg, forearms forward (palms about 0.24 m in
#   front of the pelvis at waist height)
# - elbow +pi/2 straightens the arm downward
# - shoulder pitch < 0 raises the arm forward
# - shoulder roll > 0 (left) / < 0 (right) raises the arms out to the sides
# "The example" is Unitree's G1 arm example, unitree_sdk2_python's g1_arm7_sdk_dds_example.py.
NAMED_CONFIGS = {
    'home':  # zero posture, forearms forward; the example's first stage
        np.zeros(len(ENABLED_JOINTS)),
    # straight arms hanging below the shoulders. The one named configuration in
    # self-collision: the real wrists and hands rest on the hips (g1_description's meshes
    # touch), and each hand and wrist sphere sits 2 to 5 cm inside its hip-pitch sphere, so
    # a goal to it stops short with the barrier on, and a planned goal to it fails
    'arms_down':
        np.array([0.0, 0.0, 0.0, HALF_PI, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, HALF_PI, 0.0, 0.0, 0.0]),
    't_pose':  # straight arms out to the sides, horizontal; the example's lift target
        np.array([0.0, HALF_PI, 0.0, HALF_PI, 0.0, 0.0, 0.0,
                  0.0, -HALF_PI, 0.0, HALF_PI, 0.0, 0.0, 0.0]),
    'arms_front_45':  # straight arms, 45 deg forward and down
        np.array([-np.pi / 4, 0.0, 0.0, HALF_PI, 0.0, 0.0, 0.0,
                  -np.pi / 4, 0.0, 0.0, HALF_PI, 0.0, 0.0, 0.0]),
}
# postures the controller moves through when it starts, then it moves the waist to its hold
# targets
STARTUP_SEQUENCE = ('home',)

# Controller-config defaults the two robots do not share (golem_upper_ik.config). The
# follow gate reads golem_safety's output, /lowcmd in both G1 modes. The /lowstate tick-lag
# check is off: the real G1's tick has not been measured.
CONFIG_DEFAULTS = {
    'topics.safety_out': '/lowcmd',
    'controller.max_state_lag_ms': 0.0,
}
# the controller config frame_task_server and joint_state_publisher load when their config
# parameter is not given
DEFAULT_CONFIG = 'g1_split'


def model_paths() -> dict:
    '''
    The model files, installed by golem_ros2_models. Pinocchio resolves the relative mesh
    paths of the plain URDF; the ROS copy has package:// URIs for robot_state_publisher and
    RViz. The self-collision model is the sphere-packed G1 and its SRDF (golem_ros2_models'
    paths); its spheres were packed on the rubber-hand G1, two per hand at the Dex3-1 palm
    and none on the fingers, so the barrier and the planner do not see the Dex3-1 fingers.
    The G1's configs keep pairs 20 mm apart (controller.d_min), as the H1-2's do; at home the
    closest pairs, the upper arms against the torso, are 25 mm apart. The IK can start a few
    millimetres inside that keep-out, and the barrier pushes the pair back out, but not much
    further: from a pose with a pair about 10 mm apart or closer, the start-up's first step
    has no solution and frame_task_server stops. On the real G1 in debug mode the arms hang
    limp until its first command, so they must hang clear (in the sim they hang with the
    wrists resting on the hips, their spheres 20 mm from the hip spheres; arms_down is inside
    them)
    '''
    return paths.model_paths('g1')
