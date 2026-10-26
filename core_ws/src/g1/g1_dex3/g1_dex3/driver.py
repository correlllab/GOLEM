'''DDS driver for one Dex3-1 hand (rt/dex3/<side>/cmd and state), with no ROS'''

import math
import time
import threading

import numpy as np

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_ as HandCmdDefault

from g1_description.layout import NUM_HAND_MOTORS
from g1_dex3.dex3 import (
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_MAX_SPEED,
    PoseInterpolator,
    motor_mode,
    move_duration,
    validate_positions,
)

# stop commands sent on release (timeout bit set, zero gains), like the stop in
# Unitree's Dex3 example; then the driver goes silent
RELEASE_MESSAGES = 10
# how long shutdown() waits for those stop commands to go out (it returns as
# soon as they have; 10 take 0.1 s at 100 Hz, longer on a starved CPU)
RELEASE_TIMEOUT_S = 1.0
# a first command needs hand state newer than this
MAX_STATE_AGE_S = 0.5


def _check_value(name, value, positive=False):
    '''value as a float; raise unless finite and >= 0 (> 0 if positive)'''
    value = float(value)
    bound = '> 0' if positive else '>= 0'
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        raise ValueError(f'{name} must be finite and {bound}, got {value}')
    return value


class Dex3Hand:
    '''
    One Dex3-1 hand. Silent until the first command, so a hand driven by another
    program (or by nobody) is left alone; after that it streams HandCmd_ at
    rate_hz, holding the current target. No joint is commanded faster than
    max_speed rad/s. release() sends Unitree's stop and goes silent again.
    Timing uses time.monotonic(), so a wall-clock step cannot skip a move
    '''

    def __init__(self, side, kp=DEFAULT_KP, kd=DEFAULT_KD, rate_hz=100.0,
                 cmd_topic=None, state_topic=None, max_speed=DEFAULT_MAX_SPEED):
        self.side = side
        self.kp = _check_value('kp', kp)
        self.kd = _check_value('kd', kd)
        self.max_speed = _check_value('max_speed', max_speed, positive=True)
        self._dt = 1.0 / _check_value('rate_hz', rate_hz, positive=True)
        self._lock = threading.Lock()
        self._interp = None  # None: not commanding
        self._release_left = 0
        self._state = None
        self._cmd = HandCmdDefault()
        self._pub = ChannelPublisher(cmd_topic or f'rt/dex3/{side}/cmd', HandCmd_)
        self._pub.Init()
        self._sub = ChannelSubscriber(state_topic or f'rt/dex3/{side}/state', HandState_)
        self._sub.Init(self._on_state, 10)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f'dex3_{side}', daemon=True)
        self._thread.start()

    def _on_state(self, msg: HandState_) -> None:
        if len(msg.motor_state) < NUM_HAND_MOTORS:
            return
        motors = msg.motor_state[:NUM_HAND_MOTORS]
        state = {
            'q': np.array([m.q for m in motors], dtype=np.float64),
            'dq': np.array([m.dq for m in motors], dtype=np.float64),
            'tau': np.array([m.tau_est for m in motors], dtype=np.float64),
            'pressure': np.array([list(p.pressure) for p in msg.press_sensor_state],
                                 dtype=np.float32).reshape(-1, 12),
            'time': time.monotonic(),
        }
        with self._lock:
            self._state = state

    @property
    def state(self):
        '''Latest hand state (q, dq, tau, pressure, time from time.monotonic()) or None'''
        with self._lock:
            return None if self._state is None else dict(self._state)

    @property
    def commanding(self) -> bool:
        with self._lock:
            return self._interp is not None

    def set_pose(self, positions, duration: float = 0.0) -> np.ndarray:
        '''
        Move to `positions` (7 rad, clipped to the URDF limits) over `duration`
        s, or longer if a joint would otherwise move faster than max_speed
        '''
        goal = validate_positions(self.side, positions)
        now = time.monotonic()
        with self._lock:
            if self._interp is not None:
                start = self._interp.value(now)
            else:
                # first command: start from where the hand is
                if self._state is None or now - self._state['time'] > MAX_STATE_AGE_S:
                    raise RuntimeError(
                        f'no recent {self.side} hand state on rt/dex3/{self.side}/state')
                start = self._state['q'].copy()
                if not np.all(np.isfinite(start)):
                    raise RuntimeError(
                        f'{self.side} hand state has non-finite joint positions: {start}')
            seconds = move_duration(start, goal, duration, self.max_speed)
            self._interp = PoseInterpolator(start, goal, seconds, now)
            self._release_left = 0
        return goal

    def release(self) -> bool:
        '''
        Send Unitree's stop (fingers go limp) and stop commanding. Does nothing,
        and returns False, if this driver is not commanding the hand
        '''
        with self._lock:
            if self._interp is None:
                return False
            self._interp = None
            self._release_left = RELEASE_MESSAGES
            return True

    def _fill(self, q, kp, kd, timeout):
        for i in range(NUM_HAND_MOTORS):
            motor = self._cmd.motor_cmd[i]
            motor.mode = motor_mode(i, timeout=timeout)
            motor.q = float(q[i])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(kp)
            motor.kd = float(kd)

    def _loop(self):
        while self._running:
            start_time = time.monotonic()
            send = False
            with self._lock:
                if self._interp is not None:
                    self._fill(self._interp.value(start_time), self.kp, self.kd, timeout=False)
                    send = True
                elif self._release_left > 0:
                    self._fill(np.zeros(NUM_HAND_MOTORS), 0.0, 0.0, timeout=True)
                    self._release_left -= 1
                    send = True
            if send:
                self._pub.Write(self._cmd)
            time.sleep(max(0.0, self._dt - (time.monotonic() - start_time)))

    def shutdown(self, release: bool = True) -> None:
        '''
        Stop the driver. With release (the default), a hand this driver was
        commanding first gets Unitree's stop, so its fingers go limp. With
        release=False the driver just stops publishing, and the hand is left
        with the last command it got (what the Dex3 then does is not documented
        by Unitree). Stop messages already queued by release() are sent either way
        '''
        if release:
            self.release()
        deadline = time.monotonic() + RELEASE_TIMEOUT_S
        while time.monotonic() < deadline:
            with self._lock:
                if self._release_left == 0:
                    break
            time.sleep(self._dt)
        self._running = False
        self._thread.join(timeout=1.0)
        self._pub.Close()
        self._sub.Close()
