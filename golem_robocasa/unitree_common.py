"""The Unitree DDS command path both RoboCasa robots' SimInterface share
(unitree_interface.py for the H1-2, g1/unitree_interface.py for the G1): the
command timeout knob, rt/lowcmd validation and latching, rate-limited warnings,
the CRC, the IMU's rpy and the DDS domain.

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

# rt/lowcmd mode_pr both robots' stacks write and accept: the ankles' pitch/roll joints
MODE_PR = 0
# wall seconds between repeats of one warning
WARN_PERIOD_S = 5.0


def env_seconds(name, default, env=os.environ):
    """A timeout knob in sim seconds: unset or empty for the default, else finite
    and positive. A NaN would never time out (silence > NaN is never true), so
    a bad value stops the sim at startup, naming the knob."""
    raw = env.get(name, '').strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        raise ValueError(f'{name} must be a number of sim seconds, got {raw!r}') from None
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f'{name} must be finite and positive (sim seconds), got {raw!r}')
    return value


class MotorCommand(NamedTuple):
    """Latched commands of one topic's motors, as arrays, and the sim time they arrived"""
    on: np.ndarray  # motors that apply their PD; the others are off (zero torque)
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    time: float

    def torques(self, q, dq):
        return np.where(self.on, self.kp * (self.q - q) + self.kd * (self.dq - dq) + self.tau, 0.0)


def parse_motor_cmds(motor_cmds, slots, sim_time, motor_on):
    """MotorCommand for the given slots of a motor_cmd array, with motor_on(mode)
    deciding which motors are on; None if a value is not finite"""
    cmds = [motor_cmds[i] for i in slots]
    values = np.array([[c.q, c.dq, c.tau, c.kp, c.kd] for c in cmds], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return None
    return MotorCommand(np.array([motor_on(int(c.mode)) for c in cmds]), *values.T, time=sim_time)


def log(text):
    """One write per line: the DDS handler threads print too"""
    sys.stdout.write(f'[unitree_interface] {text}\n')
    sys.stdout.flush()


class RateLimitedWarnings:
    """Prints a warning at most once per WARN_PERIOD_S for each key, with how
    many it stands for; the DDS handler threads call it"""

    def __init__(self):
        self._lock = threading.Lock()
        self._warned = {}   # key -> (wall time of the last print, warnings since)

    def warn(self, key, text):
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
    """unitree_sdk2py's CRC, with its Python loop over the packed words (__Trans)
    replaced by a numpy view of them: the same field packing and the same C
    routine, so the same CRC, at about a quarter of the cost (33 instead of 145
    us a frame, measured over rt/lowstate and rt/lowcmd frames; rt/lowstate is
    signed every physics step, and every rt/lowcmd frame is checked). The SDK's
    CRC object is one per process. The result is checked against the SDK's own
    on sample frames; if it differs, the SDK's loop is kept."""
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

    def words(packed):   # as __Trans: every little-endian uint32 but the last (the crc's)
        return np.frombuffer(packed, '<u4', count=len(packed) // 4 - 1)

    def crc32(data):
        return lib.crc32_core(data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(data))

    crc._CRC__Trans, crc._crc_ctypes = words, crc32
    if [crc.Crc(msg) for msg in samples] == expected:
        crc._sim_fast = True
    else:
        del crc._CRC__Trans, crc._crc_ctypes
        log("WARNING: the fast CRC does not match unitree_sdk2py's; using the SDK's")
    return crc


def lowcmd_error(msg, crc, mode_machine, robot):
    """Why an rt/lowcmd frame must be dropped, or None to accept it: robot
    (its name) reports mode_machine"""
    if crc.Crc(msg) != msg.crc:
        return 'bad CRC'
    if msg.mode_machine != mode_machine:
        return f'mode_machine {msg.mode_machine}, the {robot} reports {mode_machine}'
    if msg.mode_pr != MODE_PR:
        return f'mode_pr {msg.mode_pr}, not MODE_PR ({MODE_PR})'
    return None


def rpy_from_quat(w, x, y, z):
    """Roll, pitch, yaw (rad) of a wxyz quaternion, as the IMU reports them"""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [roll, pitch, yaw]


def domain_id():
    """ROS_DOMAIN_ID, which the Unitree SDK uses too; refuses unset, empty and 0"""
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
