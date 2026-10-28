'''The H1-2's controller data: the Magpie-gripper model, its frames and named postures'''
import os

import numpy as np

# the robot's name in messages
DISPLAY_NAME = 'H1-2'

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
# upper-body joints the IK does not move; every command holds them at controller.hold_targets
HELD_JOINTS = ('torso_joint',)
# The controller's shutdown command turns the torso off with the arms, in both modes: gravity
# barely loads the torso yaw
SHUTDOWN_KEEPS_HELD_JOINTS_ON_LOWCMD = False
# end effectors for velocity limiting and the ee pose topics
EE_FRAMES = ('left_wrist_yaw_link', 'right_wrist_yaw_link')
# frames the OMPL planner plans for and bounds by planner.frame_* limits
PLANNER_FRAMES = ('left_grasp_frame', 'right_grasp_frame')
# frame of the /lowstate IMU, from which the model recovers the pelvis orientation
IMU_FRAME = 'torso_link'

# Named arm postures for the 14 IK joints, rad, all inside the joint limits:
#   0 L_sh_pitch 1 L_sh_roll 2 L_sh_yaw 3 L_elbow 4 L_wr_roll 5 L_wr_pitch 6 L_wr_yaw
#   7 R_sh_pitch 8 R_sh_roll 9 R_sh_yaw 10 R_elbow 11 R_wr_roll 12 R_wr_pitch 13 R_wr_yaw
# Arms horizontal load the shoulders and elbows most, which the system-identification runs
# cycle through with the robot on a gantry.
NAMED_CONFIGS = {
    'home':  # arms down, the low-load baseline
        np.zeros(len(ENABLED_JOINTS)),
    't_pose':  # arms out to the sides, horizontal
        np.array([0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'arms_front_45':  # forward 45 deg, mid load
        np.array([-0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  -0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'arms_overhead':  # arms overhead, gravity along the arm
        np.array([0.0, 2.9, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, -2.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'arms_asym':  # left arm forward, right arm to the side
        np.array([-1.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'arms_front_yaw':  # forward with a shoulder-yaw twist, loading the upper arm axially
        np.array([-1.45, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  -1.45, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
    'init_1':  # arms sideways, elbows bent: the first start-up stage
        np.array([0.0, 0.35, 0.0, 1.3, 0.0, 0.0, 0.0,
                  0.0, -0.35, 0.0, 1.3, 0.0, 0.0, 0.0]),
    'init_2':
        np.array([0.7, 0.35, 0.0, 1.3, 0.0, 0.0, 0.0,
                  0.7, -0.35, 0.0, 1.3, 0.0, 0.0, 0.0]),
    'init_3':
        np.array([0.7, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.7, -0.35, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'prep':  # init_3 with the elbows bent, for level, raised grasp frames
        np.array([0.7, 0.35, 0.0, -0.7, 0.0, 0.0, 0.0,
                  0.7, -0.35, 0.0, -0.7, 0.0, 0.0, 0.0]),
    # right arm as recorded from the real robot's /lowstate (slots 20-26) at the grasp
    # staging pose, left arm at 'prep'. Its wrist_yaw (-1.2632) is about 0.007 rad inside
    # the -1.27 limit: leave headroom when changing it
    'staging_right':
        np.array([0.7, 0.35, 0.0, -0.7, 0.0, 0.0, 0.0,
                  -1.0597, -0.8747, 1.5802, -0.5551, 2.4575, -0.0036, -1.2632]),
    # 'staging_right' mirrored across the sagittal plane, right arm at 'prep' (itself
    # symmetric). The arm joint axes are the same on both sides, so the mirror keeps pitch,
    # elbow and wrist pitch and negates roll, yaw, wrist roll and wrist yaw. Derived, not
    # recorded: check it against the left arm's joint limits before relying on it
    'staging_left':
        np.array([-1.0597, 0.8747, -1.5802, -0.5551, -2.4575, -0.0036, 1.2632,
                  0.7, -0.35, 0.0, -0.7, 0.0, 0.0, 0.0]),
}
# postures the controller moves through when it starts, then it moves the torso to its
# hold target; the staged moves keep the hands from scraping the hips
STARTUP_SEQUENCE = ('init_1', 'init_2', 'init_3', 'home')

# Controller-config defaults the two robots do not share (golem_upper_ik.config). The
# follow gate reads golem_safety's output, /lowcmd in both H1-2 modes. A goal stops when
# /lowstate's tick lags by more than max_state_lag_ms (ms; <= 0 turns the check off).
CONFIG_DEFAULTS = {
    'topics.safety_out': '/lowcmd',
    'controller.max_state_lag_ms': 50.0,
}
# the controller config frame_task_server and joint_state_publisher load when their config
# parameter is not given
DEFAULT_CONFIG = 'debug'


def model_paths() -> dict:
    '''
    The model files, installed by h12_ros2_model. Pinocchio resolves the relative mesh paths
    of h1_2_magpie.urdf; the _ros copy has package:// URIs for robot_state_publisher and RViz
    '''
    from ament_index_python.packages import get_package_share_directory

    model = os.path.join(get_package_share_directory('h12_ros2_model'), 'assets', 'h1_2')
    return {
        'urdf_path': os.path.join(model, 'h1_2_magpie.urdf'),
        'urdf_ros_path': os.path.join(model, 'h1_2_magpie_ros.urdf'),
        'collision': (os.path.join(model, 'h1_2_magpie_sphere.urdf'),
                      os.path.join(model, 'h1_2_magpie_sphere_collision.srdf')),
    }
