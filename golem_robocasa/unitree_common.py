"""Validate and latch Unitree DDS motor commands for both robots' SimInterface.

unitree_interface.py (H1-2) and g1/unitree_interface.py (G1) share the command
timeout knob, rt/lowcmd validation, the latched MotorCommand, rate-limited
warnings, the CRC, the IMU's rpy and the DDS domain from here.

A DDS handler validates a frame and latches it as one MotorCommand, swapped in
by a single assignment, and never touches MjData; the main loop turns the
latched command into torques every physics step.
"""
import ctypes
import math
import os
import sys
import threading
import time
from typing import NamedTuple

import numpy as np

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmd_default
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.utils.crc import CRC

# rt/lowcmd and rt/lowstate mode_pr of both robots: the ankles in pitch/roll mode.
MODE_PR = 0
WARN_PERIOD_S = 5.0   # wall s between repeats of one warning


def env_seconds(name, default, env=os.environ):
    """Return the env timeout knob name in sim seconds, or default when unset or empty.

    Raises ValueError, naming the knob, for a value that is not finite and
    positive: a NaN timeout would never expire (silence > NaN is never true).
    """
    raw = env.get(name, '').strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        raise ValueError(f'{name} must be a number of sim seconds, got {raw!r}') from None
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f'{name} must be finite and positive (sim seconds), got {raw!r}')
    return value


class MotorCommand(NamedTuple):
    """Hold one topic's latched motor commands, as arrays, and the sim time they arrived."""
    on: np.ndarray  # bool per motor: apply the PD; the others get zero torque
    q: np.ndarray   # rad
    dq: np.ndarray  # rad/s
    tau: np.ndarray  # N*m feed-forward
    kp: np.ndarray
    kd: np.ndarray
    time: float     # s of sim time

    def torques(self, q, dq):
        """Return the PD torque (N*m) per motor at measured q (rad) and dq (rad/s), 0 where off."""
        return np.where(self.on, self.kp * (self.q - q) + self.kd * (self.dq - dq) + self.tau, 0.0)


def parse_motor_cmds(motor_cmds, slots, sim_time, motor_on):
    """Return the MotorCommand of motor_cmds' slots, or None if a value is not finite.

    motor_on(mode) decides which motors apply their PD.
    """
    cmds = [motor_cmds[i] for i in slots]
    values = np.array([[c.q, c.dq, c.tau, c.kp, c.kd] for c in cmds], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return None
    return MotorCommand(np.array([motor_on(int(c.mode)) for c in cmds]), *values.T, time=sim_time)


def log(text):
    """Print a [unitree_interface] line in one write, as the DDS handler threads print too."""
    sys.stdout.write(f'[unitree_interface] {text}\n')
    sys.stdout.flush()


class RateLimitedWarnings:
    """Print each kind of warning at most once per WARN_PERIOD_S, with how many it stands for.

    Thread-safe: the DDS handler threads call it.
    """

    def __init__(self):
        """Start with no warning printed."""
        self._lock = threading.Lock()
        self._warned = {}   # key -> (wall time of the last print, warnings since)

    def warn(self, key, text):
        """Print text unless a warning with this key was printed within WARN_PERIOD_S."""
        now = time.monotonic()
        with self._lock:
            last, count = self._warned.get(key, (None, 0))
            count += 1
            if last is not None and now - last < WARN_PERIOD_S:
                self._warned[key] = (last, count)
                return
            self._warned[key] = (now, 0)
        repeats = f' ({count} since the last warning)' if count > 1 else ''
        log(f'{text}{repeats}')


def fast_crc():
    """Return unitree_sdk2py's CRC object with its per-word Python loop replaced by numpy.

    The replacement packs the same fields and calls the same C routine, so the
    CRC is the same at about a quarter of the cost (33 instead of 145 us a
    frame); rt/lowstate is signed every physics step and every rt/lowcmd frame
    is checked. The SDK's CRC object is one per process, so the patch applies
    once. It is checked against the SDK's own result on sample frames and
    dropped, with a warning, if they differ.
    """
    crc = CRC()
    lib = getattr(crc, 'crc_lib', None)
    if getattr(crc, '_sim_fast', False) or lib is None:   # done, or not Linux
        return crc
    samples = [LowState_default(), LowCmd_default()]
    rng = np.random.default_rng(0)
    for msg in samples:
        for motor in getattr(msg, 'motor_state', None) or msg.motor_cmd:
            motor.q, motor.dq = (float(v) for v in rng.normal(size=2))
    expected = [crc.Crc(msg) for msg in samples]

    def words(packed):
        """Return the packed message's little-endian uint32 words but the last (the crc's)."""
        return np.frombuffer(packed, '<u4', count=len(packed) // 4 - 1)

    def crc32(data):
        """Return the SDK C routine's CRC of the words."""
        return lib.crc32_core(data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(data))

    crc._CRC__Trans, crc._crc_ctypes = words, crc32
    if [crc.Crc(msg) for msg in samples] == expected:
        crc._sim_fast = True
    else:
        del crc._CRC__Trans, crc._crc_ctypes
        log("WARNING: the fast CRC does not match unitree_sdk2py's; using the SDK's")
    return crc


def lowcmd_error(msg, crc, mode_machine, robot):
    """Return why an rt/lowcmd frame must be dropped, or None to accept it.

    robot is the robot's name for the message, mode_machine the value it accepts.
    """
    if crc.Crc(msg) != msg.crc:
        return 'bad CRC'
    if msg.mode_machine != mode_machine:
        return f'mode_machine {msg.mode_machine}, the {robot} reports {mode_machine}'
    if msg.mode_pr != MODE_PR:
        return f'mode_pr {msg.mode_pr}, not MODE_PR ({MODE_PR})'
    return None


def rpy_from_quat(w, x, y, z):
    """Return [roll, pitch, yaw] (rad) of a wxyz quaternion, as the IMU's rpy."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [roll, pitch, yaw]


def domain_id():
    """Return ROS_DOMAIN_ID for the Unitree SDK, raising RuntimeError for unset, empty or 0.

    SAFETY: domain 0 is the real robots' command bus; a sim never joins it.
    """
    raw = os.environ.get('ROS_DOMAIN_ID', '')
    try:
        domain = int(raw)
    except ValueError:
        domain = 0
    if domain <= 0:
        raise RuntimeError(
            f'ROS_DOMAIN_ID must be a positive integer, got {raw!r}; domain 0 is '
            'reserved for the real robot, e.g. export ROS_DOMAIN_ID=1')
    return domain
