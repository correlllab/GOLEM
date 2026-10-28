"""Unitree DDS for the simulated G1: the topics the robot's own computers serve
in Unitree's debug mode, where nothing but rt/lowcmd drives the body.

  rt/lowstate           published after every physics step (500 Hz sim time)
  rt/lowcmd             subscribed; PD on the 29 body motors
  rt/dex3/<side>/cmd    subscribed; PD on the hand's 7 motors
  rt/dex3/<side>/state  published at 100 Hz sim time

Every motor is off (zero torque) until the first rt/lowcmd frame with a valid
CRC, mode_machine 5 and mode_pr 0; from then on rt/lowcmd drives every body
motor, and motor mode 0 turns one off. A hand's motors are off until its first
rt/dex3/<side>/cmd. Until a command drives it the robot hangs from the sim's
elastic band (g1_mujoco), as the H1-2 does in its sim.

Every torque is recomputed every physics step from the latched commands and the
current state (write_ctrl), which emulates the motors' own servo loop, and is
limited to the joint's actuatorfrcrange. rt/lowcmd silent for GOLEM_CMD_TIMEOUT
(0.5 s of sim time, so a sim running slower than real time does not trip it)
turns every body motor off (zero torque) until frames return. This is the H1-2
sim's behaviour, kept as an assumption; what the real G1's motors do when
rt/lowcmd stops has not been measured.
The slot layout and header values match
core_ws/src/g1/g1_description/g1_description/layout.py.
"""
import ctypes
import math
import os
import sys
import threading
import time
from typing import NamedTuple

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_ as HandState_default
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmd_default
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

TOPIC_LOWSTATE = 'rt/lowstate'
TOPIC_LOWCMD = 'rt/lowcmd'
TOPIC_HAND_CMD = 'rt/dex3/{side}/cmd'
TOPIC_HAND_STATE = 'rt/dex3/{side}/state'

# rt/lowcmd header values the G1 accepts: our unit (g1_29dof_rev_1_0) reports
# mode_machine 5, and mode_pr 0 (MODE_PR) drives the ankles' pitch/roll joints
MODE_MACHINE = 5
MODE_PR = 0
# Dex3 HandCmd_ mode byte (g1_dex3/dex3.py): bits 0-3 motor index, bits 4-6
# status (1 = enabled), bit 7 timeout, which the driver sets to release a finger
HAND_STATUS_ENABLED = 0x01
HAND_STATE_HZ = 100.0


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


# Sim seconds without a frame before rt/lowcmd, once it has driven the motors,
# turns every motor off (an assumption, see the module docstring)
LOWCMD_TIMEOUT = env_seconds('GOLEM_CMD_TIMEOUT', 0.5)
# wall seconds between repeats of one warning
WARN_PERIOD_S = 5.0


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


def _log(text):
    """One write per line: the DDS handler threads print too"""
    sys.stdout.write(f'[unitree_interface] {text}\n')
    sys.stdout.flush()


def fast_crc():
    """unitree_sdk2py's CRC, with its Python loop over the packed words (__Trans)
    replaced by a numpy view of them: the same field packing and the same C
    routine, so the same CRC, at about a quarter of the cost (33 instead of 145
    us a frame, measured over rt/lowstate and rt/lowcmd frames; rt/lowstate is
    signed every physics step, and every rt/lowcmd frame is checked). The SDK's CRC object is one per process. The result is checked
    against the SDK's own on sample frames; if it differs, the SDK's loop is
    kept."""
    crc = CRC()
    lib = getattr(crc, 'crc_lib', None)
    if getattr(crc, '_g1_sim_fast', False) or lib is None:   # done, or not Linux
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
        crc._g1_sim_fast = True
    else:
        del crc._CRC__Trans, crc._crc_ctypes
        _log("WARNING: the fast CRC does not match unitree_sdk2py's; using the SDK's")
    return crc


def lowcmd_error(msg, crc):
    """Why an rt/lowcmd frame must be dropped, or None to accept it"""
    if crc.Crc(msg) != msg.crc:
        return 'bad CRC'
    if msg.mode_machine != MODE_MACHINE:
        return f'mode_machine {msg.mode_machine}, the G1 reports {MODE_MACHINE}'
    if msg.mode_pr != MODE_PR:
        return f'mode_pr {msg.mode_pr}, not MODE_PR ({MODE_PR})'
    return None


def lowcmd_motor_on(mode):
    """rt/lowcmd: motor mode 0 is off"""
    return mode != 0


def hand_motor_on(mode):
    """A Dex3 motor applies its PD only when enabled and without the timeout bit"""
    status = (mode >> 4) & 0x07
    timeout = (mode >> 7) & 0x01
    return status == HAND_STATUS_ENABLED and not timeout


def rpy_from_quat(w, x, y, z):
    """Roll, pitch, yaw (rad) of a wxyz quaternion, as the IMU reports them"""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [roll, pitch, yaw]


def _domain_id():
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


class SimInterface:
    """The G1's Unitree DDS topics on a MuJoCo model.

    The main loop calls write_ctrl() before and after_step() after every
    mj_step, holding the sim lock. The DDS handlers only validate and latch
    commands (one reference swap each) and never touch MjData.
    """

    def __init__(self, model, data, resolver):
        self.model = model
        self.data = data
        self.resolver = resolver
        missing = {'quat', 'gyro', 'acc'} - set(resolver.sensor_adr)
        if missing:
            raise KeyError(f'pelvis IMU sensors missing from the model: {sorted(missing)}')
        self.lowcmd_timeout = LOWCMD_TIMEOUT
        self._crc = fast_crc()
        self._hand_state_every = max(1, round(1.0 / (HAND_STATE_HZ * model.opt.timestep)))
        self._steps = 0
        # sim time of the last step, for the handlers (a float swap, no MjData access)
        self.sim_time = float(data.time)

        # latched commands (None until the first valid frame)
        self._lowcmd = None
        self._hand_cmd = {side: None for side in resolver.hands}
        # True from the first valid rt/lowcmd frame on; until then every body motor is off
        self.commanded = False
        self._lowcmd_timed_out = False
        # The sim's elastic band (mujoco_env.ElasticBand), which sim_loop sets once
        # it has built it, for the tests that record the sim; this class never uses it.
        self.band = None
        self._warn_lock = threading.Lock()
        self._warned = {}

        # Unitree DDS on ROS_DOMAIN_ID. The SDK ignores CYCLONEDDS_URI, so the
        # interface comes from G1_DDS_INTERFACE, as for the rest of the stack.
        interface = os.environ.get('G1_DDS_INTERFACE', '')
        if interface:
            ChannelFactoryInitialize(_domain_id(), interface)
        else:
            ChannelFactoryInitialize(_domain_id())

        self.low_state = LowState_default()
        self.low_state.mode_pr = MODE_PR
        self.low_state.mode_machine = MODE_MACHINE
        self._low_state_pub = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self._low_state_pub.Init()
        self._hand_state = {side: HandState_default() for side in resolver.hands}
        self._hand_state_pub = {}
        for side in resolver.hands:
            self._hand_state_pub[side] = ChannelPublisher(TOPIC_HAND_STATE.format(side=side), HandState_)
            self._hand_state_pub[side].Init()

        # the subscribed topics and their subscribers
        self._subscribers = {TOPIC_LOWCMD: ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)}
        self._subscribers[TOPIC_LOWCMD].Init(self._on_lowcmd, 10)
        for side in resolver.hands:
            topic = TOPIC_HAND_CMD.format(side=side)
            self._subscribers[topic] = ChannelSubscriber(topic, HandCmd_)
            self._subscribers[topic].Init(lambda msg, side=side: self._on_hand_cmd(side, msg), 10)

    # ------------------------------------------------------------------ #
    # DDS handlers: validate and latch
    # ------------------------------------------------------------------ #
    def _warn(self, key, text):
        """Print a warning at most once per WARN_PERIOD_S, with how many it stands for"""
        now = time.monotonic()
        with self._warn_lock:
            last, count = self._warned.get(key, (None, 0))
            count += 1
            if last is not None and now - last < WARN_PERIOD_S:
                self._warned[key] = (last, count)
                return
            self._warned[key] = (now, 0)
        repeats = f' ({count} since the last warning)' if count > 1 else ''
        _log(f'{text}{repeats}')

    def _on_lowcmd(self, msg: LowCmd_):
        error = lowcmd_error(msg, self._crc)
        cmd = None if error else parse_motor_cmds(msg.motor_cmd, range(len(self.resolver.body.names)),
                                                  self.sim_time, lowcmd_motor_on)
        if cmd is None:
            error = error or 'a value is not finite'
            self._warn(f'lowcmd {error}', f'dropped rt/lowcmd: {error}')
            return
        self._lowcmd = cmd
        if not self.commanded:
            self.commanded = True
            _log(f't={cmd.time:.3f}s: first valid rt/lowcmd; it drives all 29 body motors')

    def _on_hand_cmd(self, side, msg: HandCmd_):
        count = len(self.resolver.hands[side].names)
        cmd = None
        if len(msg.motor_cmd) >= count:
            cmd = parse_motor_cmds(msg.motor_cmd, range(count), self.sim_time, hand_motor_on)
        if cmd is None:
            self._warn(f'hand_{side}', f'dropped rt/dex3/{side}/cmd: '
                       f'{len(msg.motor_cmd)} motors or a value that is not finite')
            return
        self._hand_cmd[side] = cmd

    # ------------------------------------------------------------------ #
    # main loop (the caller holds the sim lock)
    # ------------------------------------------------------------------ #
    def physics_reset(self):
        """MuJoCo has reset the whole scene and its time: restamp the latched
        rt/lowcmd at the new sim time, so its timeout still measures the silence
        since the last frame instead of waiting for the old sim time to come
        round again"""
        now = float(self.data.time)
        self.sim_time = now
        if self._lowcmd is not None:
            self._lowcmd = self._lowcmd._replace(time=now)

    def _lowcmd_torques(self, q, dq, now):
        cmd = self._lowcmd
        if now - cmd.time > self.lowcmd_timeout:
            if not self._lowcmd_timed_out:
                self._lowcmd_timed_out = True
                _log(f't={now:.3f}s: rt/lowcmd silent for {self.lowcmd_timeout:.2f}s; '
                     'every motor off (zero torque)')
            return np.zeros_like(q)
        if self._lowcmd_timed_out:
            self._lowcmd_timed_out = False
            _log(f't={now:.3f}s: rt/lowcmd back')
        return cmd.torques(q, dq)

    def write_ctrl(self):
        """The torque of every motor for the next physics step"""
        d = self.data
        body = self.resolver.body
        if self.commanded:
            tau = self._lowcmd_torques(d.qpos[body.qpos], d.qvel[body.qvel], float(d.time))
            d.ctrl[body.ctrl] = np.clip(tau, body.tau_min, body.tau_max)
        else:   # never commanded: every motor off
            d.ctrl[body.ctrl] = 0.0

        for side, hand in self.resolver.hands.items():
            cmd = self._hand_cmd[side]
            if cmd is None:  # never commanded: no torque, as after the driver's release
                d.ctrl[hand.ctrl] = 0.0
                continue
            tau = cmd.torques(d.qpos[hand.qpos], d.qvel[hand.qvel])
            d.ctrl[hand.ctrl] = np.clip(tau, hand.tau_min, hand.tau_max)

    def after_step(self):
        """Publish the state after a physics step"""
        d = self.data
        r = self.resolver
        self.sim_time = float(d.time)
        self._steps += 1
        self._publish_low_state(d.qpos[r.body.qpos], d.qvel[r.body.qvel])
        if self._steps % self._hand_state_every == 0:
            for side in r.hands:
                self._publish_hand_state(side)

    def _publish_low_state(self, q, dq):
        d = self.data
        r = self.resolver
        sd = d.sensordata
        msg = self.low_state
        msg.tick = int(round(d.time * 1000.0)) & 0xFFFFFFFF  # ms of sim time
        # tau_est: the torque the joint got, i.e. the motor's clamped to actuatorfrcrange
        tau = d.qfrc_actuator[r.body.qvel]
        for i, motor in enumerate(msg.motor_state[:len(q)]):
            motor.q = float(q[i])
            motor.dq = float(dq[i])
            motor.tau_est = float(tau[i])
        imu = msg.imu_state
        adr = r.sensor_adr
        quat = [float(v) for v in sd[adr['quat'][0]:adr['quat'][0] + 4]]  # w, x, y, z
        imu.quaternion = quat
        imu.gyroscope = [float(v) for v in sd[adr['gyro'][0]:adr['gyro'][0] + 3]]
        imu.accelerometer = [float(v) for v in sd[adr['acc'][0]:adr['acc'][0] + 3]]
        imu.rpy = rpy_from_quat(*quat)
        msg.crc = self._crc.Crc(msg)
        self._low_state_pub.Write(msg)

    def _publish_hand_state(self, side):
        d = self.data
        hand = self.resolver.hands[side]
        q = d.qpos[hand.qpos]
        dq = d.qvel[hand.qvel]
        tau = d.qfrc_actuator[hand.qvel]
        msg = self._hand_state[side]  # pressure stays 0: the sim has no tactile sensors
        for i, motor in enumerate(msg.motor_state[:len(q)]):
            motor.q = float(q[i])
            motor.dq = float(dq[i])
            motor.tau_est = float(tau[i])
        self._hand_state_pub[side].Write(msg)

    def shutdown(self):
        for sub in self._subscribers.values():
            sub.Close()
        self._low_state_pub.Close()
        for pub in self._hand_state_pub.values():
            pub.Close()
