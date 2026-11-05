"""Hold the Dex3-1 hand logic that needs no ROS or DDS.

Joint limits, named poses, the HandCmd_ motor mode byte, interpolation, and
the driver's DDS domain and shutdown choices.
"""

import math

import numpy as np

SIDES = ('left', 'right')
# HandCmd_/HandState_ motor entries per hand that the driver writes and reads.
NUM_HAND_MOTORS = 7
# (name, lower rad, upper rad) per joint, in HandCmd_/HandState_ motor order.
# The limits are the Dex3-1's in Unitree's G1 URDF
# (CL_Assets/g1/g1_description/g1_29dof_with_hand_rev_1_0.urdf).
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

# PD gains from Unitree's Dex3 grip example
# (unitree_ros2/example/src/src/g1/dex3/g1_dex3_example.cpp). No model gives
# the hand's real gains, so these stay soft.
DEFAULT_KP = 1.5
DEFAULT_KD = 0.1

# rad/s, cap on any finger joint's commanded speed: a move, a 0 s one included,
# takes at least its largest joint distance over this. A chosen value, not a
# measured limit (unmeasured on the real G1): below every URDF velocity limit,
# and fast enough that open and close over the default 1 s are never stretched
# (no joint inside the limits is more than 1.75 rad from either pose).
DEFAULT_MAX_SPEED = 2.0

# Status field of the HandCmd_ mode byte (1 = enabled). The byte, as in
# Unitree's example: bits 0-3 motor index, bits 4-6 status, bit 7 timeout.
# Commands use timeout 0; Unitree's stop sends timeout 1 with zero gains, which
# releases the finger.
MODE_STATUS_ENABLED = 0x01

# True where a joint closes toward its upper limit, in HAND_JOINTS order,
# checked with forward kinematics on the with-hand URDF. The right hand
# mirrors the left.
CLOSE_TOWARD_UPPER = {
    'left': np.array([False, True, True, False, False, False, False]),
    'right': np.array([False, False, False, True, True, True, True]),
}
# Fraction of each joint's closing range the `close` pose uses, in HAND_JOINTS
# order; thumb_0 (thumb rotation) stays at 0. It brings the thumb about 1 cm
# from the finger tips.
CLOSE_FRACTION = np.array([0.0, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7])


def motor_mode(index: int, timeout: bool = False) -> int:
    """Return the HandCmd_ mode byte that enables motor `index`, timeout bit as given."""
    if not 0 <= index < NUM_HAND_MOTORS:
        raise ValueError(f'motor index must be in [0, {NUM_HAND_MOTORS}), got {index}')
    return (index & 0x0F) | ((MODE_STATUS_ENABLED & 0x07) << 4) | ((1 if timeout else 0) << 7)


def named_pose(side: str, name: str) -> np.ndarray:
    """Return the joint targets (rad) of `open` (fingers straight) or `close` (power grip)."""
    if name == 'open':
        return np.zeros(NUM_HAND_MOTORS)
    if name == 'close':
        end = np.where(CLOSE_TOWARD_UPPER[side], UPPER[side], LOWER[side])
        return CLOSE_FRACTION * end
    raise ValueError(f'unknown hand pose {name!r} (open, close)')


def validate_positions(side: str, positions) -> np.ndarray:
    """Return `positions` as a 7-vector clipped to the URDF limits.

    Raises ValueError on an unknown side, a wrong length or a non-finite value.
    """
    if side not in SIDES:
        raise ValueError(f'side must be one of {SIDES}, got {side!r}')
    q = np.asarray(positions, dtype=np.float64).reshape(-1)
    if q.shape != (NUM_HAND_MOTORS,):
        raise ValueError(f'expected {NUM_HAND_MOTORS} joint positions, got {q.size}')
    if not np.all(np.isfinite(q)):
        raise ValueError('joint positions must be finite')
    return np.clip(q, LOWER[side], UPPER[side])


def move_duration(q_start, q_goal, duration, max_speed: float) -> float:
    """Return the seconds to move from q_start to q_goal.

    That is `duration`, stretched so no joint moves faster than `max_speed`
    rad/s. Raises ValueError on a negative or non-finite duration, which would
    otherwise reach PoseInterpolator and command NaN positions.
    """
    duration = float(duration)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f'duration must be finite and >= 0 s, got {duration}')
    distance = float(np.max(np.abs(np.asarray(q_goal, dtype=np.float64)
                                   - np.asarray(q_start, dtype=np.float64))))
    return max(duration, distance / max_speed)


class PoseInterpolator:
    """Interpolate linearly in joint space from one pose to another over a duration."""

    def __init__(self, q_start, q_goal=None, duration=0.0, t_start=0.0):
        """Start at t_start (s); with no q_goal, hold q_start."""
        self.q_start = np.asarray(q_start, dtype=np.float64)
        self.q_goal = self.q_start.copy() if q_goal is None else np.asarray(q_goal, dtype=np.float64)
        self.duration = max(float(duration), 0.0)
        self.t_start = float(t_start)

    def value(self, now: float) -> np.ndarray:
        """Return the pose at time `now` (s, the clock of t_start)."""
        if self.duration <= 0.0:
            return self.q_goal.copy()
        alpha = min(max((now - self.t_start) / self.duration, 0.0), 1.0)
        return (1.0 - alpha) * self.q_start + alpha * self.q_goal


# Values of dex3_hands' on_shutdown parameter: on exit a hand the node was
# commanding gets Unitree's stop, so the fingers go limp ('release'), or no
# message at all ('silent').
SHUTDOWN_ACTIONS = ('release', 'silent')


def dds_domain(ros_domain_id, domain_id_parameter, ros_domain=None):
    """Return the Unitree DDS domain from ROS_DOMAIN_ID or the domain_id parameter.

    `ros_domain_id` is the environment string (None or empty: unset);
    `domain_id_parameter` < 0 means unset. Raises ValueError when neither is
    set, when they disagree, and, given `ros_domain` (the domain the node's own
    ROS side runs on), when the result differs from it.

    SAFETY: there is no default domain, because domain 0 is the real robot.
    The ros_domain check catches ROS running on domain 0 with ROS_DOMAIN_ID
    unset, and rcl reading a leading 0 as octal ('010' is 8).
    """
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
