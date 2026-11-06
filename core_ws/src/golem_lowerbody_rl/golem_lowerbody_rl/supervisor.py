"""
Decide when the leg node commands the legs, and with what, for every robot, with no ROS.

The node hands LegSupervisor every /lowstate sample (a StateSample with its tick), every
/safety/heartbeat beat, /cmd_vel, the height command, the arm controller's commands, the
first /left_ee_pose, the operator's confirm and the band toggle's reply. on_state returns
the leg targets to publish for a sample, or None to stay silent. What differs between
robots, and between a sim and a real robot, comes only from the run config, the robot
profile and the policy's spec.

Phases (Phase):

- WAITING: silent until golem_safety's first true heartbeat and a finite, upright
  /lowstate (gravity z in the IMU frame below UPRIGHT_GRAVITY_Z, a unit quaternion).
- RAMPING: the legs blend from their measured pose to the pre-pose along a cosine over
  startup.ramp_seconds, then hold it until they settle (every leg within
  startup.settle_tolerance for startup.settle_ticks steps) or startup.settle_max_ticks
  steps have passed (logged NOT SETTLED; held all the same).
- HOLDING: the pre-pose every step until the policy may start. It may start once ready:
  /lowstate's tick advancing at the node clock's rate (within startup.tick_rate) over the
  last full window of startup.steady_seconds of ticks (each window starts where the last
  ended, and its verdict stands while the next fills), no gap over faults.max_state_gap,
  the arm controller commanding within startup.upper_fresh_seconds and, with band.release,
  a first /left_ee_pose or band.max_wait since HOLDING began. With
  startup.wait_for_confirm it then also waits for the operator's confirm, accepted only
  while holding and ready; without, it starts.
- RUNNING: the policy drives the legs from zero velocity and its default height, reset
  as its spec's reset_on says. With band.release the band toggle is due once, once
  /left_ee_pose has come or band.max_wait after the policy started.
- FAULTED: silent for good. Once the legs are commanded (RAMPING, HOLDING, RUNNING): a
  /lowstate gap over faults.max_state_gap, a tick that goes back, a non-finite value, a
  quaternion norm outside 1 +- QUAT_NORM_TOLERANCE, a fall (gravity z above
  FALLEN_GRAVITY_Z), a policy step that fails or gives anything but finite targets, and
  while running a tick lag over faults.max_latency when it is set. In every phase: a
  false heartbeat, or none for heartbeat.timeout after the first (checked by check() and
  on every sample).

Clocks: every mono argument is one monotonic clock (s), for gaps, the heartbeat, the
commands and the band's waits. node_t is the node's clock (sim time in the sims, wall
time on the robots), against which the tick's rate is checked and which the policy gets
as its time. Steps follow /lowstate's tick on a grid one policy period apart (TickGrid).

Not thread-safe: the caller holds one lock around every call, so a band reply's policy
reset never runs inside a policy step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Callable, Protocol

import numpy as np

from golem_lowerbody_rl.band import BandRelease
from golem_lowerbody_rl.imu import gravity_z
from golem_lowerbody_rl.policy import LegTargets, PolicyError, PolicyInput
from golem_lowerbody_rl.profile import RobotProfile
from golem_lowerbody_rl.run_config import RunConfig
from golem_lowerbody_rl.timing import TICK_MS, TICK_SECONDS, tick_delta, TickGrid, TickLag

__all__ = ['LegPolicy', 'LegSupervisor', 'Phase', 'StateSample']

# gravity's z in the IMU frame is -1 upright. Starting needs it below -0.9 (tilted less
# than about 25 deg); above -0.5 (more than 60 deg) is a fall
UPRIGHT_GRAVITY_Z = -0.9
FALLEN_GRAVITY_Z = -0.5
# how far the IMU quaternion's norm may be from 1
QUAT_NORM_TOLERANCE = 0.05
# a repeated message (a reason to wait, a command that is not finite) is logged at most
# this often per kind, and again this often while it lasts (s)
WAIT_LOG_SECONDS = 5.0
# how often a ready, holding node reminds the operator to confirm (s)
CONFIRM_REMINDER_SECONDS = 5.0
# how often the step spacing and tick lag are logged (s)
STATS_SECONDS = 10.0
# /elastic_band/toggle's message when the call left the band on (golem_robocasa's
# MeasurementBridge answers "enabled" or "disabled")
BAND_ON = 'enabled'
CONFIRM_CALL = 'ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger'
EE_POSE_MISSING = 'frame_task_server is not up yet (no /left_ee_pose)'
UPPER_KIND = 'no recent command from the arm controller'
# the steady window is filling: a reason that is not logged
STEADY_FILLING = '/lowstate has not been steady for long enough yet'
LATCH_TAIL = ('Stopped commanding the legs for good; golem_safety keeps relaying their last '
              'command, which does not balance the robot (it estops instead when /lowstate '
              "has a non-finite joint value, a joint past limits.estop or another robot's "
              'mode_machine). Restart the node to clear')
HEARTBEAT_TAIL = 'Stopped commanding the legs for good. Restart the bringup to clear'


class Phase(Enum):
    """Name the leg node's phases (the module docstring)."""

    WAITING = 'waiting'
    RAMPING = 'ramping'
    HOLDING = 'holding'
    RUNNING = 'running'
    FAULTED = 'faulted'


# the phases in which the legs are commanded
COMMANDED = (Phase.RAMPING, Phase.HOLDING, Phase.RUNNING)


@dataclass(frozen=True)
class StateSample:
    """Hold one /lowstate sample, the IMU already trimmed."""

    tick: int  # the robot's ms counter (uint32, wraps)
    q: np.ndarray  # rad, every body slot, in slot order
    dq: np.ndarray  # rad/s, every body slot
    quat_wxyz: np.ndarray  # the IMU orientation, w first
    gyro: np.ndarray  # rad/s, the IMU's angular velocity in its frame


class LegPolicy(Protocol):
    """Name what LegSupervisor uses of a PolicyRunner."""

    spec: Any  # rate_hz, reset_on and command.height.default
    kp: np.ndarray
    kd: np.ndarray

    def step(self, inp: PolicyInput) -> LegTargets:
        """Return one step's leg targets; raise on any failure."""

    def reset(self, seed_cmd=None) -> None:
        """Reset the policy; seed None re-seeds from the last command."""


class LegSupervisor:
    """
    Run the leg node's phases, checks and commands (the module docstring).

    log(level, message) reports phase changes, why it waits, the timing and faults;
    level is 'info', 'warn' or 'error'.
    """

    def __init__(self, run: RunConfig, profile: RobotProfile, prepose: np.ndarray,
                 runner: LegPolicy, log: Callable[[str, str], None]):
        """
        Supervise runner's policy on profile's legs under run.

        prepose (rad, profile.legs order) is the pose the legs ramp to and hold before
        the policy runs. Raises ValueError for a pre-pose or gains that are not one finite
        value per leg.
        """
        legs = len(profile.legs)
        self._prepose = np.asarray(prepose, dtype=np.float64)
        # SAFETY: the legs are driven to the pre-pose at the policy's gains before the
        # policy runs: a missing or non-finite angle or gain would command a leg to nowhere
        if self._prepose.shape != (legs,) or not np.all(np.isfinite(self._prepose)):
            raise ValueError(f'the pre-pose must be {legs} finite angles, not '
                             f'{self._prepose.tolist()}')
        self._kp = np.array(runner.kp, dtype=np.float64)
        self._kd = np.array(runner.kd, dtype=np.float64)
        for name, gains in (('kp', self._kp), ('kd', self._kd)):
            if gains.shape != (legs,) or not np.all(np.isfinite(gains)):
                raise ValueError(f'the policy {name} must be {legs} finite gains, not '
                                 f'{gains.tolist()}')
        self._run = run
        self._profile = profile
        self._runner = runner
        self._log = log
        self._legs = legs
        self._leg_slots = profile.slots(profile.legs)
        self._body = len(profile.body_joints)
        rate_hz = float(runner.spec.rate_hz)
        self._grid = TickGrid(1.0 / rate_hz)
        self._ramp_steps = max(1, int(round(run.startup.ramp_seconds * rate_hz)))
        self._steady_ticks = int(round(run.startup.steady_seconds / TICK_SECONDS))
        self._reset_on = tuple(runner.spec.reset_on)
        self._default_height = float(runner.spec.command.height.default)
        self._band = BandRelease(run.band.release, run.band.max_wait)

        self._phase = Phase.WAITING
        self._fault: str | None = None
        self._beat_mono: float | None = None  # when the last true beat came
        self._last_tick: int | None = None  # the newest tick taken
        self._last_mono: float | None = None  # when it was taken
        self._lag = TickLag()
        self._wait_reason = 'no /lowstate sample yet'
        self._logged: dict[str, float] = {}  # when each kind of repeated message was last logged
        # pre-pose
        self._ramp_start = np.zeros(legs)
        self._ramp_step = 0
        self._settle_count = 0
        self._holding_since: float | None = None
        self._posed_note = ''
        self._reminded: float | None = None
        # readiness: the steady window's first tick and node time, why the policy may not
        # start (None: it may), the last full window's verdict on the tick (None: on the
        # node's clock; STEADY_FILLING: no full window yet) and its rate
        self._window: tuple[int, float] | None = None
        self._ready_reason: str | None = STEADY_FILLING
        self._rate_verdict: str | None = STEADY_FILLING
        self._rate = 0.0
        self._upper_mono: float | None = None
        self._ee_seen = False
        self._ee_wait_over = False
        # running
        self._running_since: float | None = None
        self._cmd = np.zeros(3)
        self._cmd_mono: float | None = None
        self._height = self._default_height
        self._band_missing_logged = False
        # timing log: step spacing (ticks) and tick lag (ms) per step
        self._last_step_tick: int | None = None
        self._step_gaps: list[int] = []
        self._lags: list[float] = []
        self._stats_start: float | None = None

    @property
    def phase(self) -> Phase:
        """Return the current phase."""
        return self._phase

    @property
    def fault(self) -> str | None:
        """Return why the node faulted, or None."""
        return self._fault

    @property
    def wait_reason(self) -> str | None:
        """Return why the legs or the policy wait, or None while running or faulted."""
        if self._phase is Phase.WAITING:
            return self._wait_reason
        if self._phase not in (Phase.RAMPING, Phase.HOLDING):
            return None
        if self._ready_reason is not None:
            return self._ready_reason
        if self._phase is Phase.RAMPING:
            return 'the legs are still pre-posing'
        return 'the operator has not confirmed yet'

    # -- inputs ------------------------------------------------------------------------

    def on_heartbeat(self, value: bool, mono: float) -> None:
        """Take golem_safety's /safety/heartbeat: true while it relays, false once latched."""
        if self._phase is Phase.FAULTED:
            return
        # SAFETY: a false beat means golem_safety's estop has latched; whatever the phase,
        # the legs are not commanded again until the bringup restarts
        if not value:
            self._latch("golem_safety's /safety/heartbeat is false: its estop has latched",
                        HEARTBEAT_TAIL)
            return
        self._beat_mono = mono

    def on_cmd(self, vx: float, vy: float, yaw: float, mono: float) -> None:
        """Take a /cmd_vel (m/s, m/s, rad/s) received at mono; a non-finite one is zero."""
        cmd = np.array([vx, vy, yaw], dtype=np.float64)
        # SAFETY: the policy never observes a non-finite command: one means stop
        if not np.all(np.isfinite(cmd)):
            self._log_every('warn', f'/cmd_vel is not finite ({cmd.tolist()}): zero velocity',
                            mono)
            cmd = np.zeros(3)
        self._cmd, self._cmd_mono = cmd, mono

    def on_height(self, h: float, mono: float) -> None:
        """Take a height command (the policy's own units); a non-finite one is ignored."""
        if not math.isfinite(h):
            self._log_every('warn', f'ignoring a height command that is not finite ({h})',
                            mono)
            return
        self._height = float(h)

    def on_upper_cmd(self, mono: float) -> None:
        """Take a command of the arm controller's, received at mono (readiness only)."""
        self._upper_mono = mono

    def on_ee_pose(self, mono: float) -> None:
        """Take a /left_ee_pose: frame_task_server has finished its start-up moves."""
        self._ee_seen = True

    def check(self, mono: float) -> None:
        """Latch when the heartbeat, or /lowstate while the legs are commanded, went quiet."""
        if self._phase is Phase.FAULTED or self._heartbeat_silent(mono):
            return
        if self._phase not in COMMANDED or self._last_mono is None:
            return
        age = mono - self._last_mono
        max_gap = self._run.faults.max_state_gap
        if age > max_gap:
            self._latch(f'/lowstate stale: no sample for {age * 1000:.0f} ms '
                        f'(> {max_gap * 1000:.0f} ms)')

    def _heartbeat_silent(self, mono: float) -> bool:
        """Latch and return True when no beat came for heartbeat.timeout after the first."""
        timeout = self._run.heartbeat.timeout
        # SAFETY: the watchdog arms on the first true beat, and then holds in every phase
        if self._beat_mono is not None and mono - self._beat_mono > timeout:
            self._latch(f'no /safety/heartbeat for {(mono - self._beat_mono) * 1000:.0f} ms '
                        f'(> {timeout * 1000:.0f} ms): golem_safety has stopped or cannot '
                        'keep up', HEARTBEAT_TAIL)
            return True
        return False

    def confirm(self, mono: float) -> tuple[bool, str]:
        """Take the operator's go-ahead; return (accepted, message)."""
        if self._phase is not Phase.HOLDING:
            return False, (f'refused: the node is {self._phase.value}; the policy starts on '
                           'a confirm only while the legs hold the pre-pose '
                           f'({Phase.HOLDING.value})')
        # SAFETY: the policy takes the legs only once the timing and the arm controller
        # are ready, whoever asks
        if self._ready_reason is not None:
            return False, f'refused: the policy cannot start yet: {self._ready_reason}'
        self._start(mono)
        if self._phase is not Phase.RUNNING:
            return False, f'refused: the policy could not start: {self._fault}'
        return True, 'confirmed: the policy starts on the next policy step'

    def band_due(self, mono: float, service_ready: bool = True) -> str | None:
        """
        Return the reason to call /elastic_band/toggle now, once; otherwise None.

        service_ready says whether the toggle service is there; while it is not, nothing
        is due and a warning is logged once. The caller sends the call at once and hands
        its reply to on_band_reply.
        """
        # SAFETY: only a running policy releases the band; a faulted node's legs no longer
        # balance
        if self._phase is not Phase.RUNNING:
            return None
        reason = self._band.poll(mono, self._running_since, self._ee_seen, service_ready)
        if self._band.service_missing and not self._band_missing_logged:
            self._band_missing_logged = True
            self._log('warn', 'the elastic band is due for release, but /elastic_band/toggle '
                              'is not there yet: waiting for it')
        if reason is not None:
            self._log('info', f'releasing the elastic band: {reason}')
        return reason

    def on_band_reply(self, success: bool, message: str) -> None:
        """
        Take how the one /elastic_band/toggle call ended; final, whatever it says.

        success false: message says why it failed. A success with "enabled" put the band
        back on. A release resets the policy at once when its spec's reset_on says so.
        """
        if not self._band.result(success and message != BAND_ON):
            return
        if not success:
            self._log('error', f'elastic band release failed: {message}. The band stays on; '
                               'the node does not call /elastic_band/toggle again')
            return
        if message == BAND_ON:
            # the band was already off (SPACE in the viewer), and the toggle hung the robot
            # from it again. Another call would take it off again, under a robot someone
            # may have hung up on purpose, so none is sent
            self._log('warn', f'/elastic_band/toggle answered "{BAND_ON}": the elastic band '
                              'was already off and is now ON, and the robot hangs from it '
                              'again. The node does not call the toggle again')
            return
        released = f'elastic band released (/elastic_band/toggle: {message})'
        if self._phase is not Phase.RUNNING or 'band_release' not in self._reset_on:
            self._log('info', released)
            return
        # the policy's history filled with band-held states: it restarts on the free robot
        if self._reset(None):
            self._log('info', f'{released}: the policy restarts from an empty history')

    def latch(self, reason: str) -> None:
        """Stop commanding the legs for good."""
        if self._phase is not Phase.FAULTED:
            self._latch(reason)

    def on_state(self, sample: StateSample, mono: float, node_t: float) -> LegTargets | None:
        """
        Take a /lowstate sample received at mono (node clock node_t).

        Return the leg targets to publish when a step is due on its tick, else None. A
        sample whose tick did not advance is ignored.
        """
        try:
            return self._on_state(sample, mono, node_t)
        except Exception as e:  # noqa: BLE001 - never leave the legs to a broken check
            self.latch(f'unexpected error: {e!r}')
            return None

    # -- the state machine -------------------------------------------------------------

    def _on_state(self, sample: StateSample, mono: float,
                  node_t: float) -> LegTargets | None:
        """Run on_state."""
        tick = int(sample.tick)
        delta = None if self._last_tick is None else tick_delta(self._last_tick, tick)
        if delta == 0:
            return None
        gap = None if self._last_mono is None else mono - self._last_mono
        previous = self._last_tick
        self._last_tick, self._last_mono = tick, mono
        went_back = delta is not None and delta < 0
        if went_back and self._phase is Phase.WAITING:
            # a sim restarted before the legs were commanded: measure from the new tick
            self._lag = TickLag()
        lag_ms = self._lag.update(tick, mono)
        # SAFETY: a sample never moves the legs after the heartbeat went quiet, whether or
        # not check() has run since
        if self._phase is Phase.FAULTED or self._heartbeat_silent(mono):
            return None
        if self._phase is Phase.WAITING:
            return self._wait(sample, tick, mono, node_t, gap)

        # SAFETY: the legs are commanded: every problem with /lowstate stops them for good
        max_gap = self._run.faults.max_state_gap
        if went_back:
            return self._stop(f'/lowstate tick went back from {previous} to {tick}')
        if gap is not None and gap > max_gap:
            return self._stop(f'/lowstate stale: {gap * 1000:.0f} ms between samples '
                              f'(> {max_gap * 1000:.0f} ms)')
        problem = self._state_problem(sample)
        if problem is not None:
            return self._stop(problem)
        gz = gravity_z(sample.quat_wxyz)
        if gz > FALLEN_GRAVITY_Z:
            return self._stop(f'fell: gravity z {gz:.2f} in the IMU frame '
                              f'(> {FALLEN_GRAVITY_Z})')
        max_latency = self._run.faults.max_latency
        # the pre-pose and the hold are PD targets, which a late sample does not upset
        if (self._phase is Phase.RUNNING and max_latency is not None
                and lag_ms > max_latency * 1000.0):
            return self._stop(f'/lowstate late: tick lag {lag_ms:.1f} ms '
                              f'(> {max_latency * 1000:.0f} ms)')
        if self._phase is not Phase.RUNNING:
            self._track_ready(tick, mono, node_t, gap)
        if not self._grid.due(tick):
            return None
        self._note_step(tick, lag_ms)
        out: LegTargets | None
        if self._phase is Phase.RAMPING:
            out = self._ramp(sample, mono)
        elif self._phase is Phase.HOLDING:
            out = self._hold_step(sample, mono, node_t)
        else:
            out = self._run_step(sample, mono, node_t)
        if out is not None:
            self._log_timing(mono)
        return out

    def _state_problem(self, sample: StateSample) -> str | None:
        """Return why a sample cannot be used (a value not finite, a bad quaternion), or None."""
        for name, size in (('q', self._body), ('dq', self._body), ('quat_wxyz', 4),
                           ('gyro', 3)):
            values = np.asarray(getattr(sample, name), dtype=np.float64)
            if values.shape != (size,):
                return f'/lowstate {name} has shape {values.shape}, not ({size},)'
            bad = np.flatnonzero(~np.isfinite(values))
            if bad.size:
                where: list = bad.tolist()
                if name in ('q', 'dq'):
                    where = [self._profile.body_joints[i] for i in bad]
                label = 'quat' if name == 'quat_wxyz' else name
                return f'/lowstate {label} is not finite (at {where})'
        norm = float(np.linalg.norm(np.asarray(sample.quat_wxyz, dtype=np.float64)))
        if abs(norm - 1.0) > QUAT_NORM_TOLERANCE:
            return f'the /lowstate IMU quaternion is not a unit quaternion (norm {norm:.3f})'
        return None

    def _wait(self, sample: StateSample, tick: int, mono: float, node_t: float,
              gap: float | None) -> LegTargets | None:
        """Start the pre-pose on the first usable, upright sample after the first true beat."""
        reason = self._state_problem(sample)
        if reason is None:
            gz = gravity_z(sample.quat_wxyz)
            if gz >= UPRIGHT_GRAVITY_Z:
                reason = (f'the robot is not upright (gravity z {gz:.2f} in the IMU frame, '
                          f'needs < {UPRIGHT_GRAVITY_Z})')
        if reason is None and self._beat_mono is None:
            reason = 'no /safety/heartbeat yet (golem_safety beats once it can relay)'
        if reason is not None:
            self._wait_reason = reason
            self._log_every('info', f'waiting to pre-pose the legs: {reason}', mono)
            return None
        self._phase = Phase.RAMPING
        self._grid.due(tick)  # the first step sets the grid
        self._note_step(tick, 0.0)
        self._ramp_start = np.asarray(sample.q, dtype=np.float64)[self._leg_slots].copy()
        self._log('info', 'pre-posing the legs: a cosine ramp from where they are to the '
                          f'pre-pose over {self._ramp_steps / self._run_rate():.1f} s')
        self._track_ready(tick, mono, node_t, gap)
        return self._ramp(sample, mono)

    def _run_rate(self) -> float:
        """Return the policy's steps per second."""
        return float(self._runner.spec.rate_hz)

    def _targets(self, q: np.ndarray) -> LegTargets:
        """Return leg targets q at the policy's gains."""
        return LegTargets(q=np.asarray(q, dtype=np.float32), kp=self._kp.copy(),
                          kd=self._kd.copy())

    def _ramp(self, sample: StateSample, mono: float) -> LegTargets:
        """Return the next pre-pose target; after the ramp, check whether the legs settled."""
        step = self._ramp_step
        self._ramp_step += 1
        n = self._ramp_steps
        if step < n:
            # SAFETY: a cosine blend from the measured pose, never a step at full gains: a
            # step from a hanging leg to the pre-pose at the policy spec's gains can drive a
            # leg past golem_safety's velocity estop (its config's limits.estop) the moment
            # the relay opens
            alpha = 0.5 * (1.0 - math.cos(math.pi * step / n))
            return self._targets((1.0 - alpha) * self._ramp_start + alpha * self._prepose)
        legs = np.asarray(sample.q, dtype=np.float64)[self._leg_slots]
        err = float(np.max(np.abs(legs - self._prepose)))
        startup = self._run.startup
        self._settle_count = self._settle_count + 1 if err < startup.settle_tolerance else 0
        after = step - n + 1
        if self._settle_count >= startup.settle_ticks:
            self._enter_hold(f'settled, max error {err:.3f} rad', 'info', mono)
        elif after >= startup.settle_max_ticks:
            # held all the same: with a confirm the operator decides, in the sim the
            # policy starts from where the legs are
            self._enter_hold(f'NOT SETTLED: max error {err:.3f} rad after {after} steps',
                             'warn', mono)
        return self._targets(self._prepose)

    def _enter_hold(self, note: str, level: str, mono: float) -> None:
        """Hold the pre-pose until the policy may start."""
        self._phase = Phase.HOLDING
        self._holding_since = mono
        self._reminded = mono
        self._posed_note = note
        until = (f'until the operator confirms ({CONFIRM_CALL})'
                 if self._run.startup.wait_for_confirm else 'until the policy may start')
        self._log(level, f'legs pre-posed ({note}) — holding {until}')

    def _hold_step(self, sample: StateSample, mono: float,
                   node_t: float) -> LegTargets | None:
        """Hold the pre-pose, or start the policy once ready when no confirm is needed."""
        ready = self._ready_reason is None
        if ready and not self._run.startup.wait_for_confirm:
            self._start(mono)
            if self._phase is not Phase.RUNNING:
                return None
            return self._run_step(sample, mono, node_t)
        if ready and self._run.startup.wait_for_confirm and (
                self._reminded is None or mono - self._reminded >= CONFIRM_REMINDER_SECONDS):
            self._reminded = mono
            self._log('info', f'legs pre-posed ({self._posed_note}) — holding; the policy may '
                              f'start: {CONFIRM_CALL}')
        return self._targets(self._prepose)

    def _track_ready(self, tick: int, mono: float, node_t: float, gap: float | None) -> None:
        """Set why the policy may not start yet, or None once it may (the module docstring)."""
        max_gap = self._run.faults.max_state_gap
        reason = None
        if gap is not None and gap > max_gap:
            reason = (f'/lowstate is not steady ({gap * 1000:.0f} ms between samples, '
                      f'max {max_gap * 1000:.0f} ms)')
            self._window = None
            self._rate_verdict = STEADY_FILLING
        elif self._window is None:
            self._window = (tick, node_t)
        elif tick_delta(self._window[0], tick) >= self._steady_ticks:
            self._judge_window(self._window, tick, node_t)
        if reason is None:
            reason = self._upper_reason(mono)
        if reason is None:
            reason = self._ee_pose_reason(mono)
        if reason is None:
            reason = self._rate_verdict
            if reason == STEADY_FILLING:
                self._ready_reason = reason
                return
        if reason is not None:
            self._log_every('info', f'the policy waits: {reason}', mono)
        elif self._ready_reason is not None:
            self._log('info', f"the policy may start (/lowstate's tick ran at {self._rate:.3f} "
                              "s per s of the node's clock)")
        self._ready_reason = reason

    def _judge_window(self, window: tuple[int, float], tick: int, node_t: float) -> None:
        """Judge the full steady window from window to tick; start the next one at tick."""
        window_tick, window_t = window
        ticks = tick_delta(window_tick, tick)
        elapsed = node_t - window_t
        low, high = self._run.startup.tick_rate
        # SAFETY: the next window starts here, so the verdict covers about the last
        # steady_seconds, not everything since the hold began; it stands while the next
        # window fills
        self._window = (tick, node_t)
        if elapsed <= 0.0:
            self._rate_verdict = (
                f"the node's clock is not advancing (it stood still over {ticks * TICK_MS:.0f} "
                'ms of /lowstate tick: with use_sim_time, nothing publishes /clock?)')
            return
        # SAFETY: s of tick per s of the node's clock over the window. The policy steps on
        # the tick's grid, so a tick that is not ms of the node's clock (Isaac counts
        # publishes) would run it at the wrong rate: never ready
        self._rate = ticks * TICK_SECONDS / elapsed
        if low <= self._rate <= high:
            self._rate_verdict = None
        else:
            self._rate_verdict = (
                f"/lowstate's tick is off the node's clock ({self._rate:.3f} s of tick per s, "
                f"needs {low:.3f}-{high:.3f}: the tick must count ms of the node's clock)")

    def _upper_reason(self, mono: float) -> str | None:
        """Return why the arm controller does not count as commanding, or None."""
        # SAFETY: in split mode an arm with no command is limp; the legs' policy does not
        # take over a robot whose arms are not held
        fresh = self._run.startup.upper_fresh_seconds
        if self._upper_mono is None:
            return f'{UPPER_KIND} (none yet)'
        age = mono - self._upper_mono
        if age > fresh:
            return f'{UPPER_KIND} (the last {age:.2f} s ago, max {fresh:.2f} s)'
        return None

    def _ee_pose_reason(self, mono: float) -> str | None:
        """Return why the policy waits for frame_task_server under the band, or None."""
        band = self._run.band
        if not band.release or self._ee_seen or self._ee_wait_over:
            return None
        held = None if self._holding_since is None else mono - self._holding_since
        if held is not None and held >= band.max_wait:
            self._ee_wait_over = True
            self._log('warn', f'{EE_POSE_MISSING} {held:.0f} s after the legs were pre-posed: '
                              'the policy no longer waits for it (band.max_wait)')
            return None
        # a policy run long under the band settles into a stance the robot stumbles from
        # at the release
        return EE_POSE_MISSING

    def _log_every(self, level: str, message: str, mono: float) -> None:
        """
        Log message at most every WAIT_LOG_SECONDS per kind.

        The kind is the text before any ' ('. Two kinds that alternate are each logged at
        most that often, and one that lasts is logged again that often.
        """
        kind = message.split(' (')[0]
        last = self._logged.get(kind)
        if last is None or mono - last >= WAIT_LOG_SECONDS:
            self._logged[kind] = mono
            self._log(level, message)

    def _start(self, mono: float) -> None:
        """Start the policy at zero velocity and its default height, reset as its spec says."""
        self._phase = Phase.RUNNING
        self._running_since = mono
        # a /cmd_vel or height request from before the start is dropped
        self._cmd, self._cmd_mono = np.zeros(3), None
        self._height = self._default_height
        if 'engage' in self._reset_on and not self._reset(np.zeros(3)):
            return
        self._log('info', 'policy started at zero velocity and the default height '
                          f'{self._height:.2f}')

    def _reset(self, seed_cmd) -> bool:
        """Reset the policy; latch and return False if it fails."""
        try:
            self._runner.reset(seed_cmd)
        except Exception as e:  # noqa: BLE001 - a policy that cannot reset cannot run
            self._latch(f'policy reset failed: {e!r}')
            return False
        return True

    def _command(self, mono: float) -> np.ndarray:
        """Return the velocity command: zero once older than commands.timeout."""
        if self._cmd_mono is None or mono - self._cmd_mono > self._run.commands.timeout:
            return np.zeros(3)
        return self._cmd.copy()

    def _run_step(self, sample: StateSample, mono: float,
                  node_t: float) -> LegTargets | None:
        """Return one policy step's targets; latch on any failure."""
        inp = PolicyInput(t=node_t, q=sample.q, dq=sample.dq, quat_wxyz=sample.quat_wxyz,
                          gyro=sample.gyro, cmd=self._command(mono), height=self._height)
        try:
            out = self._runner.step(inp)
        except PolicyError as e:
            return self._stop(f'policy step failed: {e}')
        except Exception as e:  # noqa: BLE001 - any policy failure stops the legs
            return self._stop(f'policy step failed: {e!r}')
        q = np.asarray(out.q, dtype=np.float64)
        # SAFETY: only finite targets for every leg are published
        if q.shape != (self._legs,) or not np.all(np.isfinite(q)):
            return self._stop(f'policy output is not {self._legs} finite targets: '
                              f'{q.tolist()}')
        return out

    def _note_step(self, tick: int, lag_ms: float) -> None:
        """Record a step's spacing and tick lag for the timing log."""
        if self._last_step_tick is not None:
            self._step_gaps.append(tick_delta(self._last_step_tick, tick))
        self._last_step_tick = tick
        self._lags.append(lag_ms)

    def _log_timing(self, mono: float) -> None:
        """Log the step spacing and the tick lag once every STATS_SECONDS."""
        if self._stats_start is None:
            self._stats_start = mono
            return
        if mono - self._stats_start < STATS_SECONDS or not self._step_gaps:
            return
        gaps = np.array(self._step_gaps, dtype=np.float64) * TICK_MS
        lags = np.array(self._lags, dtype=np.float64)
        max_latency = self._run.faults.max_latency
        limit = 'unchecked' if max_latency is None else f'limit {max_latency * 1000:.0f} ms'
        self._log('info', f'timing over the last {mono - self._stats_start:.0f} s: '
                          f'{lags.size} steps, {gaps.min():.0f}-{gaps.max():.0f} ms of '
                          f'/lowstate tick apart (mean {gaps.mean():.1f} ms); tick lag median '
                          f'{np.median(lags):.1f} ms, max {lags.max():.1f} ms ({limit})')
        self._step_gaps, self._lags, self._stats_start = [], [], mono

    def _latch(self, reason: str, tail: str = LATCH_TAIL) -> None:
        """Fault for good: log reason once and publish nothing more."""
        self._phase = Phase.FAULTED
        self._fault = reason
        self._log('error', f'fault: {reason}. {tail}')

    def _stop(self, reason: str) -> LegTargets | None:
        """Latch reason and return None: no targets for this sample or any later one."""
        self._latch(reason)
        return None
