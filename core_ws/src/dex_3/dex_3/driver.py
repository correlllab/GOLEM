"""Drive one Dex3-1 hand over Unitree DDS (rt/dex3/<side>/cmd and state), with no ROS."""

import math
import time
import threading

import numpy as np

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_ as HandCmdDefault

from dex_3.dex3 import (
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_MAX_SPEED,
    NUM_HAND_MOTORS,
    PoseInterpolator,
    motor_mode,
    move_duration,
    validate_positions,
)

# Stop commands (timeout bit set, zero gains, as in Unitree's Dex3 example) that
# release() sends before the driver goes silent.
RELEASE_MESSAGES = 10
# s, the longest shutdown() waits for the stop commands to go out. It returns as
# soon as they have; 10 take 0.1 s at 100 Hz, longer on a starved CPU.
RELEASE_TIMEOUT_S = 1.0
# s, the oldest hand state a first command may start from.
MAX_STATE_AGE_S = 0.5


def _check_value(name, value, positive=False):
    """Return `value` as a float; raise ValueError unless finite and >= 0 (> 0 if positive)."""
    value = float(value)
    bound = '> 0' if positive else '>= 0'
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        raise ValueError(f'{name} must be finite and {bound}, got {value}')
    return value


class Dex3Hand:
    """Command one Dex3-1 hand from a background thread.

    The driver is silent until the first set_pose(), so a hand driven by another
    program, or by nobody, is left alone. From then on it streams HandCmd_ at
    rate_hz, holding the current target, and no joint is commanded faster than
    max_speed rad/s. release() sends Unitree's stop and goes silent again.
    Timing uses time.monotonic(), so a wall-clock step cannot skip a move.
    """

    def __init__(self, side, kp=DEFAULT_KP, kd=DEFAULT_KD, rate_hz=100.0,
                 cmd_topic=None, state_topic=None, max_speed=DEFAULT_MAX_SPEED):
        """Open the hand's DDS channels and start the command thread.

        kp, kd: PD gains; rate_hz: command rate while commanding; max_speed:
        rad/s. Raises ValueError on a non-finite or negative setting, or a zero
        rate_hz or max_speed. ChannelFactoryInitialize must have run first.
        """
        self.side = side
        self.kp = _check_value('kp', kp)
        self.kd = _check_value('kd', kd)
        self.max_speed = _check_value('max_speed', max_speed, positive=True)
        self._dt = 1.0 / _check_value('rate_hz', rate_hz, positive=True)
        self._lock = threading.Lock()
        self._interp = None  # PoseInterpolator while commanding, else None
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
        """Store a HandState_ sample, stamped with time.monotonic(); runs on a DDS thread."""
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
        """Return the latest hand state, or None before the first.

        A dict of q (rad), dq (rad/s), tau (N m), pressure (sensors x 12) and
        time (s, time.monotonic()).
        """
        with self._lock:
            return None if self._state is None else dict(self._state)

    def set_pose(self, positions, duration: float = 0.0) -> np.ndarray:
        """Move to `positions` and return the clipped goal.

        `positions` is 7 joint angles (rad), clipped to the URDF limits; the
        move takes `duration` s, or longer if a joint would otherwise move
        faster than max_speed. A first command starts from the measured pose
        and raises RuntimeError without a finite hand state newer than
        MAX_STATE_AGE_S; later ones start from the current target. Raises
        ValueError on bad positions or duration.
        """
        goal = validate_positions(self.side, positions)
        now = time.monotonic()
        with self._lock:
            if self._interp is not None:
                start = self._interp.value(now)
            else:
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
        """Send Unitree's stop, so the fingers go limp, then stop commanding.

        Returns False, and sends nothing, if this driver is not commanding the hand.
        """
        with self._lock:
            if self._interp is None:
                return False
            self._interp = None
            self._release_left = RELEASE_MESSAGES
            return True

    def _fill(self, q, kp, kd, timeout):
        """Write targets `q` (rad) and the gains into every motor of the command message."""
        for i in range(NUM_HAND_MOTORS):
            motor = self._cmd.motor_cmd[i]
            motor.mode = motor_mode(i, timeout=timeout)
            motor.q = float(q[i])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(kp)
            motor.kd = float(kd)

    def _loop(self):
        """Publish the interpolated target, or a queued stop, every 1 / rate_hz s."""
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
        """Stop the driver and close its DDS channels.

        With `release`, a hand this driver was commanding first gets Unitree's
        stop, so its fingers go limp. Without it the driver just stops
        publishing and the hand keeps the last command it got; Unitree does not
        document what the Dex3 then does. Stop messages already queued by
        release() are sent either way, waiting up to RELEASE_TIMEOUT_S.
        """
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
