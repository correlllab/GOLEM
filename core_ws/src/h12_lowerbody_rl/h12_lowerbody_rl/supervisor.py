"""
Start-up and fault logic of the H1-2's leg node, with no ROS.

The checks and thresholds are ``g1_homie``'s (its ``locomotion.py``). The node
gives ``LegSupervisor`` every /lowstate sample, every /safety/heartbeat beat,
/cmd_vel and the squat command, the operator's confirm and each policy step,
and does what ``phase`` says on its control tick:

- waiting: silent until golem_safety's first true heartbeat and a finite,
  upright /lowstate (gravity z in the base frame below UPRIGHT_GRAVITY_Z)
- ramping: the node pre-poses the legs, then calls ``posed``
- holding (``wait_for_confirm``): the node holds the pre-pose until the
  operator's ``confirm``, accepted only now
- running: the policy drives the legs (``step_policy``), from zero /cmd_vel
  and the default squat height
- faulted: the node publishes nothing, for good. Once the legs are commanded
  (ramping, holding, running) a /lowstate gap over ``max_state_gap``, a
  non-finite value or a non-unit IMU quaternion in /lowstate, a fall (gravity
  z above FALLEN_GRAVITY_Z) or a policy step that fails or gives non-finite
  targets latches it. In every phase, a false heartbeat (golem_safety's estop
  has latched), or none for ``heartbeat_timeout`` once the first has come,
  latches it too. golem_safety then keeps relaying the last leg command
  (CLAUDE.md, quiet inputs are held), which does not balance the robot.

Not thread-safe: the node calls it from one executor thread. The ``now``
arguments are one monotonic clock, in s.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .policy import NUM_LEG_JOINTS, RobotState

WAITING = 'waiting'
RAMPING = 'ramping'
HOLDING = 'holding'
RUNNING = 'running'
FAULTED = 'faulted'

# gravity's z in the base frame is -1 upright. Starting needs it below -0.9
# (tilted less than about 25 deg); above -0.5 (more than 60 deg) is a fall
UPRIGHT_GRAVITY_Z = -0.9
FALLEN_GRAVITY_Z = -0.5
# how far the IMU quaternion's norm may be from 1
QUAT_NORM_TOLERANCE = 0.05
# a reason to wait is logged when it changes, each kind at most this often (s)
WAIT_LOG_SECONDS = 5.0


def gravity_z(quat_wxyz: np.ndarray) -> float:
    """Return the z of gravity in the base frame, -1 upright."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return -(w * w - x * x - y * y + z * z)


class LegSupervisor:
    """
    The phases, checks and commands of the H1-2's leg node (module docstring).

    ``log(level, message)`` reports phase changes, why it waits, and faults;
    level is 'info', 'warn' or 'error'.
    """

    def __init__(self, *, heartbeat_timeout: float, max_state_gap: float,
                 cmd_vel_timeout: float, wait_for_confirm: bool, default_height: float,
                 log: Callable[[str, str], None]):
        self._heartbeat_timeout = heartbeat_timeout
        self._max_state_gap = max_state_gap
        self._cmd_vel_timeout = cmd_vel_timeout
        self._wait_for_confirm = wait_for_confirm
        self._default_height = default_height
        self._log = log
        self._phase = WAITING
        self._fault: str | None = None
        self._beat_time: float | None = None   # when the last true beat came
        self._last_sample: float | None = None  # when the last /lowstate came
        self._cmd = np.zeros(3, dtype=np.float32)
        self._cmd_time: float | None = None
        self._height = default_height
        self._wait_reason: str | None = None
        self._wait_logged: dict[str, float] = {}

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def fault(self) -> str | None:
        """Why the node faulted, or None."""
        return self._fault

    @property
    def height_cmd(self) -> float:
        """Return the squat (base-height) command the policy gets."""
        return self._height

    def on_heartbeat(self, alive: bool, now: float) -> None:
        """Take golem_safety's /safety/heartbeat: true while relaying, false once latched."""
        if self._phase == FAULTED:
            return
        if not alive:
            self._latch_heartbeat(
                "golem_safety's /safety/heartbeat is false: its estop has latched")
            return
        self._beat_time = now

    def on_lowstate(self, state: RobotState, now: float) -> None:
        """Take a /lowstate sample received at now: start the pre-pose, or check it."""
        gap = None if self._last_sample is None else now - self._last_sample
        self._last_sample = now
        if self._phase == FAULTED:
            return
        problem = self._state_problem(state)
        if self._phase == WAITING:
            self._wait(problem, state, now)
            return
        if gap is not None and gap > self._max_state_gap:
            self._latch(f'/lowstate stale: {gap * 1000:.0f} ms between samples '
                        f'(> {self._max_state_gap * 1000:.0f} ms)')
            return
        if problem is not None:
            self._latch(problem)
            return
        gz = gravity_z(state.quat)
        if gz > FALLEN_GRAVITY_Z:
            self._latch(f'fell: gravity z {gz:.2f} in the base frame (> {FALLEN_GRAVITY_Z})')

    def check_stale(self, now: float) -> None:
        """Latch when the heartbeat, or /lowstate while the legs are commanded, has gone quiet."""
        if self._phase == FAULTED:
            return
        if self._beat_time is not None and now - self._beat_time > self._heartbeat_timeout:
            self._latch_heartbeat(
                f'no /safety/heartbeat for {(now - self._beat_time) * 1000:.0f} ms '
                f'(> {self._heartbeat_timeout * 1000:.0f} ms): golem_safety has stopped or '
                'cannot keep up')
            return
        if self._phase == WAITING or self._last_sample is None:
            return
        age = now - self._last_sample
        if age > self._max_state_gap:
            self._latch(f'/lowstate stale: no sample for {age * 1000:.0f} ms '
                        f'(> {self._max_state_gap * 1000:.0f} ms)')

    def posed(self) -> None:
        """End the pre-pose: hold for the operator's confirm, or run the policy."""
        if self._phase != RAMPING:
            return
        if self._wait_for_confirm:
            self._phase = HOLDING
            self._log('info', 'legs pre-posed: holding until the operator confirms '
                              '(ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger)')
        else:
            self._run()

    def confirm(self) -> tuple[bool, str]:
        """Take the operator's go-ahead (only while holding); return (accepted, message)."""
        if self._phase != HOLDING:
            return False, (f'refused: the node is {self._phase}; the policy engages on a '
                           f'confirm only while the legs hold the pre-pose ({HOLDING})')
        self._run()
        return True, 'confirmed: the policy engages on the next control tick'

    def set_cmd_vel(self, vx: float, vy: float, wz: float, now: float) -> None:
        """Take a /cmd_vel (vx, vy, wz) received at now; a non-finite one is zero velocity."""
        cmd = np.array([vx, vy, wz], dtype=np.float32)
        if not np.all(np.isfinite(cmd)):
            self._log('warn', f'/cmd_vel is not finite ({cmd.tolist()}): zero velocity')
            cmd[:] = 0.0
        self._cmd, self._cmd_time = cmd, now

    def cmd_vel(self, now: float) -> np.ndarray:
        """Return the velocity command for the policy: zero once older than cmd_vel_timeout."""
        if self._cmd_time is None or now - self._cmd_time > self._cmd_vel_timeout:
            return np.zeros(3, dtype=np.float32)
        return self._cmd.copy()

    def set_height_cmd(self, height: float) -> None:
        """Take a squat (base-height) command; a non-finite one is ignored."""
        if not np.isfinite(height):
            self._log('warn', f'ignoring a squat command that is not finite ({height})')
            return
        self._height = float(height)

    def step_policy(self, fn: Callable[[], np.ndarray]) -> np.ndarray | None:
        """
        Run one policy step: fn() returns the 12 leg targets.

        Return them, or None, with a latched fault, when fn raises or returns
        anything but 12 finite targets. Only while running.
        """
        if self._phase != RUNNING:
            return None
        try:
            targets = np.asarray(fn(), dtype=np.float64)
        except Exception as e:  # noqa: BLE001 - any policy failure stops the legs
            self._latch(f'policy step failed: {e!r}')
            return None
        if targets.shape != (NUM_LEG_JOINTS,) or not np.all(np.isfinite(targets)):
            self._latch(f'policy output not finite: expected {NUM_LEG_JOINTS} finite targets, '
                        f'got {targets.tolist()}')
            return None
        return targets

    def latch(self, reason: str) -> None:
        """Stop commanding the legs for good."""
        if self._phase != FAULTED:
            self._latch(reason)

    def _state_problem(self, state: RobotState) -> str | None:
        """Why a sample cannot be used (a value not finite, a bad IMU quaternion), or None."""
        for name in ('q', 'dq', 'quat', 'gyro'):
            bad = np.flatnonzero(~np.isfinite(np.asarray(getattr(state, name))))
            if bad.size:
                return f'/lowstate {name} is not finite (at {bad.tolist()})'
        norm = float(np.linalg.norm(state.quat))
        if abs(norm - 1.0) > QUAT_NORM_TOLERANCE:
            return f'the /lowstate IMU quaternion is not a unit quaternion (norm {norm:.3f})'
        return None

    def _wait(self, problem: str | None, state: RobotState, now: float) -> None:
        """Start the pre-pose on the first usable, upright sample after the first true beat."""
        reason = problem
        if reason is None:
            gz = gravity_z(state.quat)
            if gz >= UPRIGHT_GRAVITY_Z:
                reason = (f'the robot is not upright (gravity z {gz:.2f} in the base frame, '
                          f'needs < {UPRIGHT_GRAVITY_Z})')
        if reason is None and self._beat_time is None:
            reason = 'no /safety/heartbeat yet (golem_safety beats once it can relay)'
        if reason is not None:
            self._log_wait(reason, now)
            return
        self._phase = RAMPING
        self._log('info', 'pre-posing the legs')

    def _log_wait(self, reason: str, now: float) -> None:
        """Log a reason to wait when its kind changes, each kind at most every WAIT_LOG_SECONDS."""
        kind = reason.split(' (')[0]
        last = self._wait_logged.get(kind)
        if kind != self._wait_reason and (last is None or now - last >= WAIT_LOG_SECONDS):
            self._wait_logged[kind] = now
            self._log('info', f'waiting to pre-pose the legs: {reason}')
        self._wait_reason = kind

    def _run(self) -> None:
        """Run the policy from zero velocity and the default squat height."""
        self._phase = RUNNING
        self._cmd, self._cmd_time = np.zeros(3, dtype=np.float32), None
        self._height = self._default_height
        self._log('info', 'engaging the policy at zero velocity and the default squat height')

    def _latch(self, reason: str, after: str | None = None) -> None:
        self._phase = FAULTED
        self._fault = reason
        if after is None:
            after = ('Stopped commanding the legs for good; golem_safety keeps relaying their '
                     'last command, which does not balance the robot. Restart the node to clear')
        self._log('error', f'fault: {reason}. {after}')

    def _latch_heartbeat(self, reason: str) -> None:
        self._latch(reason, 'Stopped commanding the legs for good. Restart the bringup to clear')
