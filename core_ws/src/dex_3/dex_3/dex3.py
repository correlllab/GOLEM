'''
Dex3-1 hand logic with no ROS or DDS: joint limits, named poses, motor mode,
interpolation, the driver's DDS domain and shutdown choices
'''

import math

import numpy as np

SIDES = ('left', 'right')
# motors per hand: the HandCmd_/HandState_ motor entries the driver writes and reads
NUM_HAND_MOTORS = 7
# each hand's joints in HandCmd_/HandState_ motor order: (name, lower, upper), rad.
# The limits are the Dex3-1's in Unitree's G1 URDF
# (CL_Assets/g1/g1_description/g1_29dof_with_hand_rev_1_0.urdf). Six of the seven are
# mirrored between the hands; thumb_0's range is symmetric, so it is the same on both
HAND_JOINTS = {
    'left': (
        ('left_hand_thumb_0_joint', -1.04719755, 1.04719755),
        ('left_hand_thumb_1_joint', -0.61086523, 1.04719755),
        ('left_hand_thumb_2_joint', 0.0, 1.74532925),
        ('left_hand_middle_0_joint', -1.57079632, 0.0),
        ('left_hand_middle_1_joint', -1.74532925, 0.0),
        ('left_hand_index_0_joint', -1.57079632, 0.0),
        ('left_hand_index_1_joint', -1.74532925, 0.0),
    ),
    'right': (
        ('right_hand_thumb_0_joint', -1.04719755, 1.04719755),
        ('right_hand_thumb_1_joint', -1.04719755, 0.61086523),
        ('right_hand_thumb_2_joint', -1.74532925, 0.0),
        ('right_hand_middle_0_joint', 0.0, 1.57079632),
        ('right_hand_middle_1_joint', 0.0, 1.74532925),
        ('right_hand_index_0_joint', 0.0, 1.57079632),
        ('right_hand_index_1_joint', 0.0, 1.74532925),
    ),
}
JOINT_NAMES = {side: [name for name, _, _ in HAND_JOINTS[side]] for side in SIDES}
LOWER = {side: np.array([lower for _, lower, _ in HAND_JOINTS[side]]) for side in SIDES}
UPPER = {side: np.array([upper for _, _, upper in HAND_JOINTS[side]]) for side in SIDES}

# gains from Unitree's Dex3 grip example
# (unitree_ros2/example/src/src/g1/dex3/g1_dex3_example.cpp); the hand's real
# PD gains are not in any model, so these start deliberately soft
DEFAULT_KP = 1.5
DEFAULT_KD = 0.1

# cap on any finger joint's commanded speed, rad/s: a move, including a 0 s
# "go directly", takes at least its largest joint distance over this. A
# first-session choice, not a measured limit: below every URDF velocity limit
# (12 rad/s, thumb_0 3.14), and fast enough that open and close over the
# default 1 s are never stretched (from any pose inside the limits, no joint
# is more than 1.75 rad from either)
DEFAULT_MAX_SPEED = 2.0

# HandCmd_ motor mode byte, as in Unitree's example: bits 0-3 motor index,
# bits 4-6 status (1 = enabled), bit 7 timeout. Commands use timeout 0; the
# example's stop sends timeout 1 with zero gains, which releases the finger
MODE_STATUS_ENABLED = 0x01

# closing direction per joint (thumb_0, thumb_1, thumb_2, middle_0, middle_1,
# index_0, index_1), checked with forward kinematics on the with-hand URDF:
# the left hand closes toward the upper limit on the thumb and the lower limit
# on the fingers; the right hand is mirrored
CLOSE_TOWARD_UPPER = {
    'left': np.array([False, True, True, False, False, False, False]),
    'right': np.array([False, False, False, True, True, True, True]),
}
# fraction of each joint's closing range used by the `close` pose; thumb_0
# (thumb rotation) stays at 0. Brings the thumb ~1 cm from the finger tips
CLOSE_FRACTION = np.array([0.0, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7])


def motor_mode(index: int, timeout: bool = False) -> int:
    '''HandCmd_ mode byte for motor `index`'''
    if not 0 <= index < NUM_HAND_MOTORS:
        raise ValueError(f'motor index must be in [0, {NUM_HAND_MOTORS}), got {index}')
    return (index & 0x0F) | ((MODE_STATUS_ENABLED & 0x07) << 4) | ((1 if timeout else 0) << 7)


def named_pose(side: str, name: str) -> np.ndarray:
    '''Joint targets for `open` (all zeros, fingers straight) or `close` (power grip)'''
    if name == 'open':
        return np.zeros(NUM_HAND_MOTORS)
    if name == 'close':
        end = np.where(CLOSE_TOWARD_UPPER[side], UPPER[side], LOWER[side])
        return CLOSE_FRACTION * end
    raise ValueError(f'unknown hand pose {name!r} (open, close)')


def validate_positions(side: str, positions) -> np.ndarray:
    '''Return positions as a 7-vector clipped to the URDF limits; raise on bad input'''
    if side not in SIDES:
        raise ValueError(f'side must be one of {SIDES}, got {side!r}')
    q = np.asarray(positions, dtype=np.float64).reshape(-1)
    if q.shape != (NUM_HAND_MOTORS,):
        raise ValueError(f'expected {NUM_HAND_MOTORS} joint positions, got {q.size}')
    if not np.all(np.isfinite(q)):
        raise ValueError('joint positions must be finite')
    return np.clip(q, LOWER[side], UPPER[side])


def move_duration(q_start, q_goal, duration, max_speed: float) -> float:
    '''
    Seconds to interpolate from q_start to q_goal: `duration`, stretched so no
    joint moves faster than `max_speed` rad/s. Raise on a negative or
    non-finite duration (a NaN one would command NaN positions)
    '''
    duration = float(duration)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f'duration must be finite and >= 0 s, got {duration}')
    distance = float(np.max(np.abs(np.asarray(q_goal, dtype=np.float64)
                                   - np.asarray(q_start, dtype=np.float64))))
    return max(duration, distance / max_speed)


class PoseInterpolator:
    '''Linear joint-space interpolation from one pose to another over a duration'''

    def __init__(self, q_start, q_goal=None, duration=0.0, t_start=0.0):
        self.q_start = np.asarray(q_start, dtype=np.float64)
        self.q_goal = self.q_start.copy() if q_goal is None else np.asarray(q_goal, dtype=np.float64)
        self.duration = max(float(duration), 0.0)
        self.t_start = float(t_start)

    def value(self, now: float) -> np.ndarray:
        if self.duration <= 0.0:
            return self.q_goal.copy()
        alpha = min(max((now - self.t_start) / self.duration, 0.0), 1.0)
        return (1.0 - alpha) * self.q_start + alpha * self.q_goal

    def done(self, now: float) -> bool:
        return now - self.t_start >= self.duration


# dex3_hands' on_shutdown parameter: what a hand the node was commanding gets on
# exit (Unitree's stop, so the fingers go limp; or no message at all)
SHUTDOWN_ACTIONS = ('release', 'silent')


def dds_domain(ros_domain_id, domain_id_parameter, ros_domain=None):
    '''
    Unitree DDS domain from ROS_DOMAIN_ID (unset or empty: None), else from the
    domain_id parameter (< 0: not set). There is no default: domain 0 is the
    real robot, so a missing domain is an error, and so is a mismatch. With
    ros_domain, the domain the node's own ROS side runs on, the result must
    also equal it: with ROS_DOMAIN_ID unset ROS runs on domain 0, and rcl reads
    a leading 0 as octal ('010' is 8)
    '''
    from_env = None
    if ros_domain_id:
        if not ros_domain_id.isdigit():
            raise ValueError(f'ROS_DOMAIN_ID must be an integer >= 0, got {ros_domain_id!r}')
        from_env = int(ros_domain_id)
    parameter = int(domain_id_parameter)
    from_parameter = parameter if parameter >= 0 else None
    if from_env is None and from_parameter is None:
        raise ValueError('no DDS domain: set ROS_DOMAIN_ID (0 is the real robot, 1 the sim) '
                         'or the domain_id parameter')
    if from_env is not None and from_parameter is not None and from_env != from_parameter:
        raise ValueError(f'ROS_DOMAIN_ID={from_env} and domain_id:={from_parameter} disagree')
    domain = from_env if from_env is not None else from_parameter
    if ros_domain is not None and int(ros_domain) != domain:
        raise ValueError(f'the node runs ROS on domain {ros_domain} but would run Unitree DDS on '
                         f'domain {domain}: set ROS_DOMAIN_ID (decimal) to one domain for both')
    return domain
