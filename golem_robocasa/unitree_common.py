"""Validate and latch Unitree DDS motor commands and write rt/lowstate for both robots' SimInterface.

unitree_interface.py (H1-2) and g1/unitree_interface.py (G1) share the command
timeout knob, the rt/lowcmd command path (LowCmdLatch: validation, the latched
MotorCommand, the timeout), rate-limited warnings, the CRC, the rt/lowstate
writer, the IMU's rpy and the DDS domain from here.

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

from cyclonedds._clayer import ddspy_write
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


# s of sim time without an rt/lowcmd frame, once frames have driven the motors,
# before every body motor turns off (GOLEM_CMD_TIMEOUT; unmeasured on the real
# G1). A bad value stops the sim when it imports its interface.
LOWCMD_TIMEOUT = env_seconds('GOLEM_CMD_TIMEOUT', 0.5)


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
    if getattr(crc, '_sim_fast', False):
        return crc
    lib = crc.crc_lib
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


class _LayoutError(Exception):
    """Report that a LowState_ field was not found where a probe expected it."""


def _pad4(ser):
    """Return serialized CDR bytes padded with zeros to a multiple of 4, as DataWriter.write pads them."""
    return ser.ljust((len(ser) + 3) & ~3, b'\0')


# A float32 and a uint32 whose four little-endian bytes are all distinct and non-zero.
_PROBE_F32 = float(np.frombuffer(bytes.fromhex('11223344'), '<f4')[0])
_PROBE_U32 = 0x44332211


def _sdk_crc():
    """Return a CRC object running unitree_sdk2py's own code, whatever fast_crc() did to CRC()."""
    crc = object.__new__(CRC)   # CRC() is a per-process singleton, which fast_crc() patches
    crc.__init__()
    return crc


def _word_index(encode, msg, base, set_value, value):
    """Return the uint32 index of the 4-byte field set_value(msg, v) writes in encode(msg).

    base is encode(msg) as msg is. Raises _LayoutError unless setting the field to value (whose
    little-endian bytes are 11 22 33 44) changes exactly one aligned word, to those bytes.
    """
    set_value(msg, value)
    changed = encode(msg)
    set_value(msg, 0)
    diff = np.flatnonzero(np.frombuffer(changed, np.uint8) != np.frombuffer(base, np.uint8))
    pattern = bytes.fromhex('11223344')
    if len(diff) == 0 or diff[0] % 4 or diff[-1] >= diff[0] + 4 \
            or changed[diff[0]:diff[0] + 4] != pattern:
        raise _LayoutError(f'field not found as a little-endian word (bytes {diff[:8]})')
    return int(diff[0]) // 4


def _field_words(encode, msg, motors):
    """Return the uint32 indices of tick, then of every float field LowStateWriter fills, in encode(msg).

    The floats are in LowStateWriter's value order: q, dq, tau_est of each motor slot, then the
    IMU's quaternion, gyroscope, accelerometer and rpy.
    """
    base = encode(msg)

    def element(get_list, i):
        """Return a setter of element i of the list get_list(msg) returns."""
        def set_value(m, v):
            """Set the element."""
            get_list(m)[i] = v
        return set_value

    def attribute(get_obj, name):
        """Return a setter of attribute name of the object get_obj(msg) returns."""
        return lambda m, v: setattr(get_obj(m), name, v)

    tick = _word_index(encode, msg, base, attribute(lambda m: m, 'tick'), _PROBE_U32)
    setters = [attribute(lambda m, i=i: m.motor_state[i], name)
               for name in ('q', 'dq', 'tau_est') for i in range(motors)]
    for name, size in (('quaternion', 4), ('gyroscope', 3), ('accelerometer', 3), ('rpy', 3)):
        setters += [element(lambda m, name=name: getattr(m.imu_state, name), i) for i in range(size)]
    floats = [_word_index(encode, msg, base, set_value, _PROBE_F32) for set_value in setters]
    return tick, np.array(floats, dtype=np.intp)


class LowStateWriter:
    """Write rt/lowstate samples built with numpy instead of unitree_sdk2py's Python serializer.

    The sample is the LowState_ that unitree_sdk2py serializes for: mode_pr MODE_PR, the robot's
    mode_machine, the tick, q, dq and tau_est of the first `motors` slots, the IMU's quaternion,
    gyroscope, accelerometer and rpy (rpy_from_quat), every other field the IDL's zero, and the
    SDK's CRC. Its CDR bytes and the CRC's packed struct are kept in two preallocated buffers;
    each write fills their fields through numpy indices found once, at construction, by
    serializing and packing probe samples with the SDK, computes the CRC with the SDK's C routine
    over the same words as fast_crc(), and hands the bytes to cyclonedds' ddspy_write, the call
    DataWriter.write makes after serializing (padded to 4 bytes as it pads). Like the SDK's
    ChannelPublisher.Write without a timeout, it writes whether or not a reader is matched.

    If the publisher has no cyclonedds DataWriter, the SDK's LowState_ packer is missing, or the
    bytes of a probe state differ from the SDK's (or building or checking them raises), it logs
    one warning and takes the SDK's
    path instead: it fills one LowState_, signs it with fast_crc() and hands it to
    publisher.Write. Not thread-safe: the physics thread owns it.
    """

    def __init__(self, publisher, mode_machine, motors):
        """Find the publisher's DataWriter and the sample's field offsets, and check them against the SDK.

        publisher is an initialised unitree_sdk2py ChannelPublisher of LowState_; motors is the
        number of motor slots filled (27 on the H1-2, 29 on the G1).
        """
        self._publisher = publisher
        self.mode_machine = mode_machine
        self.motors = motors
        self._crc = fast_crc()
        self._msg = LowState_default()
        self._msg.mode_pr = MODE_PR
        self._msg.mode_machine = mode_machine
        self._warnings = RateLimitedWarnings()
        # The DataWriter of the SDK's Channel; only its entity handle is kept, so that the
        # publisher's Close still deletes the writer.
        channel = getattr(publisher, '_ChannelPublisher__channel', None)
        writer = getattr(getattr(channel, '_Channel__writer', None), '_Writer__writer', None)
        self._ref = getattr(writer, '_ref', None)
        self._use_version_2 = getattr(writer, '_use_version_2', None)
        reason = None
        if self._ref is None:
            reason = 'the publisher has no cyclonedds DataWriter'
        else:
            try:
                self._build()
                if not self._matches_sdk():
                    reason = "the numpy bytes differ from the SDK's"
            except _LayoutError as e:
                reason = f'the numpy path cannot be built: {e}'
            except Exception as e:   # an SDK or cyclonedds change the probes did not foresee
                reason = f'the numpy path failed its check: {e!r}'
        self.fast = reason is None
        if reason:
            log(f"WARNING: rt/lowstate: {reason}; using the SDK's serializer")

    def _build(self):
        """Make the CDR and packed-struct buffers, their numpy views and the field indices.

        Raises _LayoutError if a field is not a 4-byte little-endian word in either encoding
        or the SDK's LowState_ packer is missing.
        """
        sdk = _sdk_crc()
        lib = sdk.crc_lib
        pack = getattr(sdk, '_CRC__PackHGLowState', None)   # the SDK CRC's packed struct, as words
        if pack is None:
            raise _LayoutError("unitree_sdk2py's LowState_ packer is missing")

        def cdr(msg):
            """Return msg as DataWriter.write hands it to ddspy_write."""
            return _pad4(msg.serialize(use_version_2=self._use_version_2))

        def packed(msg):
            """Return the little-endian words the SDK's CRC runs over (the packed struct but its crc)."""
            return np.asarray(pack(msg), dtype='<u4').tobytes()

        probe = LowState_default()
        probe.mode_pr = MODE_PR
        probe.mode_machine = self.mode_machine
        self._cdr = bytearray(cdr(probe))
        self._packed = bytearray(packed(probe))
        self._cdr_tick, self._cdr_floats = _field_words(cdr, probe, self.motors)
        self._packed_tick, self._packed_floats = _field_words(packed, probe, self.motors)
        self._cdr_crc = _word_index(cdr, probe, bytes(self._cdr), lambda m, v: setattr(m, 'crc', v),
                                    _PROBE_U32)
        self._cdr_f32 = np.frombuffer(self._cdr, '<f4')
        self._cdr_u32 = np.frombuffer(self._cdr, '<u4')
        self._packed_f32 = np.frombuffer(self._packed, '<f4')
        self._packed_u32 = np.frombuffer(self._packed, '<u4')
        self._values = np.zeros(len(self._cdr_floats))   # q, dq, tau, quat, gyro, acc, rpy
        self._crc32 = lib.crc32_core
        self._packed_ptr = self._packed_u32.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        self._packed_words = len(self._packed_u32)

    def _matches_sdk(self):
        """Return whether the bytes of random probe states equal the SDK's serialized, signed LowState_.

        Leaves both paths' IMU fields at the IDL's zeros, which a caller without an IMU keeps.
        """
        sdk = _sdk_crc()
        rng = np.random.default_rng(0)
        try:
            for _ in range(3):
                quat = rng.normal(size=4)
                state = (int(rng.integers(0, 2**32)), rng.normal(size=self.motors),
                         rng.normal(size=self.motors), rng.normal(size=self.motors),
                         quat / np.linalg.norm(quat), rng.normal(size=3), rng.normal(size=3))
                msg = self._fill_msg(*state)
                msg.crc = sdk.Crc(msg)
                if bytes(self._fill(*state)) != _pad4(msg.serialize(use_version_2=self._use_version_2)):
                    return False
            return True
        finally:
            zero = (0, np.zeros(self.motors), np.zeros(self.motors), np.zeros(self.motors),
                    np.zeros(4), np.zeros(3), np.zeros(3))
            self._fill(*zero)
            self._fill_msg(*zero)

    def _fill(self, tick, q, dq, tau, quat_wxyz, gyro, acc):
        """Fill the CDR buffer with the state and its CRC, and return it (the buffer, not a copy)."""
        n = self.motors
        values = self._values
        values[:n] = q
        values[n:2 * n] = dq
        values[2 * n:3 * n] = tau
        if quat_wxyz is not None:
            values[3 * n:3 * n + 4] = quat_wxyz
            values[3 * n + 4:3 * n + 13] = (*gyro, *acc, *rpy_from_quat(*quat_wxyz))
        self._cdr_f32[self._cdr_floats] = values
        self._packed_f32[self._packed_floats] = values
        self._cdr_u32[self._cdr_tick] = tick
        self._packed_u32[self._packed_tick] = tick
        self._cdr_u32[self._cdr_crc] = self._crc32(self._packed_ptr, self._packed_words)
        return self._cdr

    def _fill_msg(self, tick, q, dq, tau, quat_wxyz, gyro, acc):
        """Fill the SDK path's LowState_ with the state and return it, its crc not yet set."""
        msg = self._msg
        msg.tick = tick
        for i, motor in enumerate(msg.motor_state[:self.motors]):
            motor.q = float(q[i])
            motor.dq = float(dq[i])
            motor.tau_est = float(tau[i])
        if quat_wxyz is not None:
            imu = msg.imu_state
            quat = [float(v) for v in quat_wxyz]
            imu.quaternion = quat
            imu.gyroscope = [float(v) for v in gyro]
            imu.accelerometer = [float(v) for v in acc]
            imu.rpy = rpy_from_quat(*quat)
        return msg

    def bytes_for(self, tick, q, dq, tau, quat_wxyz, gyro, acc):
        """Return the bytes write() puts on the wire for this state.

        tick is the uint32 tick (ms); q (rad), dq (rad/s) and tau (N*m) hold one value per motor
        slot filled; quat_wxyz, gyro (rad/s) and acc (m/s^2) are the IMU's, or all None to
        leave the IMU fields as they are (zero unless an earlier write set them). The fields
        are float32, as in the IDL: a value beyond float32's range is written as inf on the
        numpy path (numpy warns), while the SDK's serializer raises.
        """
        if self.fast:
            return bytes(self._fill(tick, q, dq, tau, quat_wxyz, gyro, acc))
        msg = self._fill_msg(tick, q, dq, tau, quat_wxyz, gyro, acc)
        msg.crc = self._crc.Crc(msg)
        return _pad4(msg.serialize(use_version_2=self._use_version_2))

    def write(self, tick, q, dq, tau, quat_wxyz, gyro, acc):
        """Publish the state on rt/lowstate; the arguments are bytes_for's.

        A value beyond float32's range goes out as bytes_for says (inf on the numpy path, a
        raise on the SDK's). A write cyclonedds refuses is reported in a rate-limited warning, as the SDK's
        ChannelPublisher.Write prints and returns.
        """
        if not self.fast:
            msg = self._fill_msg(tick, q, dq, tau, quat_wxyz, gyro, acc)
            msg.crc = self._crc.Crc(msg)
            self._publisher.Write(msg)
            return
        ret = ddspy_write(self._ref, self._fill(tick, q, dq, tau, quat_wxyz, gyro, acc))
        if ret < 0:
            self._warnings.warn('lowstate write', f'rt/lowstate write failed: DDS return code {ret}')


class LowCmdLatch:
    """Latch each robot's valid rt/lowcmd frames and turn the latest into body torques: SimInterface's command path.

    A subclass sets MODE_MACHINE (the value accepted), ROBOT (its name in the
    messages) and lowcmd_motor_on(mode) (which motor modes apply the PD), and
    calls _init_lowcmd before it subscribes. Every body motor is off until the
    first valid frame (commanded). _on_lowcmd runs on the DDS thread and only
    validates and swaps in one MotorCommand; _lowcmd_torques runs on the
    physics thread.
    """

    MODE_MACHINE = None
    ROBOT = ''

    @staticmethod
    def lowcmd_motor_on(mode):
        """Return whether an rt/lowcmd motor mode applies the PD."""
        raise NotImplementedError

    def _init_lowcmd(self, motors, sim_time):
        """Start with no command, for `motors` body slots, at sim_time (s): every motor off."""
        self._motors = motors
        self.lowcmd_timeout = LOWCMD_TIMEOUT
        self._crc = fast_crc()
        # s, sim time of the last step: the handlers stamp commands with it
        # instead of reading MjData.
        self.sim_time = sim_time
        self._lowcmd = None
        self.commanded = False
        self._lowcmd_timed_out = False
        self._warnings = RateLimitedWarnings()

    def _on_lowcmd(self, msg):
        """Latch a valid rt/lowcmd frame, or drop it with a rate-limited warning (DDS thread)."""
        error = lowcmd_error(msg, self._crc, self.MODE_MACHINE, self.ROBOT)
        cmd = None if error else parse_motor_cmds(msg.motor_cmd, range(self._motors),
                                                  self.sim_time, self.lowcmd_motor_on)
        if cmd is None:
            error = error or 'a value is not finite'
            self._warnings.warn(f'lowcmd {error}', f'dropped rt/lowcmd: {error}')
            return
        self._lowcmd = cmd
        if not self.commanded:
            self.commanded = True
            log(f't={cmd.time:.3f}s: first valid rt/lowcmd; it drives all {self._motors} body motors')

    def _lowcmd_torques(self, q, dq, now):
        """Return the latched command's torques at q, dq, or zeros once it is older than the timeout."""
        cmd = self._lowcmd
        if now - cmd.time > self.lowcmd_timeout:
            if not self._lowcmd_timed_out:
                self._lowcmd_timed_out = True
                log(f't={now:.3f}s: rt/lowcmd silent for {self.lowcmd_timeout:.2f}s; '
                    'every motor off (zero torque)')
            return np.zeros_like(q)
        if self._lowcmd_timed_out:
            self._lowcmd_timed_out = False
            log(f't={now:.3f}s: rt/lowcmd back')
        return cmd.torques(q, dq)


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
