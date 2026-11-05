"""
The H1-2 leg node's start-up and faults with no ROS.

LegSupervisor on a fake clock, fed /lowstate at 500 Hz and the heartbeat at 10 Hz.
"""

import math

import numpy as np
import pytest

from golem_lowerbody_rl.policy import NUM_LEG_JOINTS, NUM_POLICY_JOINTS, RobotState
from golem_lowerbody_rl.supervisor import (
    FAULTED,
    HOLDING,
    RAMPING,
    RUNNING,
    WAITING,
    LegSupervisor,
)

SAMPLE_DT = 0.002  # /lowstate at 500 Hz
BEAT_DT = 0.1  # golem_safety's /safety/heartbeat at 10 Hz
DEFAULT_HEIGHT = 1.0


def _state(roll_deg=0.0, quat=None, q=None, gyro=(0.0, 0.0, 0.0)):
    """Return a still robot's state, rolled by roll_deg about x unless quat is given."""
    if quat is None:
        half = math.radians(roll_deg) / 2
        quat = (math.cos(half), math.sin(half), 0.0, 0.0)
    return RobotState(
        q=np.zeros(NUM_POLICY_JOINTS, dtype=np.float32) if q is None else np.asarray(q),
        dq=np.zeros(NUM_POLICY_JOINTS, dtype=np.float32),
        quat=np.asarray(quat, dtype=np.float32),
        gyro=np.asarray(gyro, dtype=np.float32),
        cmd=np.zeros(3, dtype=np.float32), height_cmd=DEFAULT_HEIGHT, t=0.0)


UPRIGHT = _state()


class Rig:
    """A LegSupervisor on a fake clock with the real robot's limits."""

    def __init__(self, wait_for_confirm=True, max_state_gap=0.05):
        self.t = 100.0
        self.log = []
        self.last_beat = None
        self.sup = LegSupervisor(
            heartbeat_timeout=0.5, max_state_gap=max_state_gap, cmd_vel_timeout=0.5,
            wait_for_confirm=wait_for_confirm, default_height=DEFAULT_HEIGHT,
            log=lambda level, message: self.log.append((level, message)))

    def run(self, seconds, state=UPRIGHT, beating=True):
        """Feed samples (and beats while beating) for `seconds`, checking for silence each time."""
        for _ in range(round(seconds / SAMPLE_DT)):
            self.t += SAMPLE_DT
            if beating and (self.last_beat is None or self.t - self.last_beat >= BEAT_DT - 1e-9):
                self.sup.on_heartbeat(True, self.t)
                self.last_beat = self.t
            self.sup.check_stale(self.t)
            self.sup.on_lowstate(state, self.t)

    def to(self, phase):
        """Drive the supervisor to `phase` (RAMPING, HOLDING or RUNNING)."""
        self.run(0.01)
        assert self.sup.phase == RAMPING, self.log
        if phase == RAMPING:
            return
        self.sup.posed()
        if phase == RUNNING and self.sup.phase == HOLDING:
            assert self.sup.confirm()[0]
        assert self.sup.phase == phase, self.log

    def messages(self, level=None):
        return [m for lvl, m in self.log if level is None or lvl == level]


def test_waits_for_the_first_beat():
    rig = Rig()
    rig.run(2.0, beating=False)
    assert rig.sup.phase == WAITING
    waits = [m for m in rig.messages() if 'no /safety/heartbeat yet' in m]
    assert len(waits) == 1, rig.log  # a steady reason is logged once
    rig.run(SAMPLE_DT)
    assert rig.sup.phase == RAMPING


def test_waits_while_not_upright():
    rig = Rig()
    rig.run(1.0, state=_state(roll_deg=40.0))
    assert rig.sup.phase == WAITING
    assert any('not upright' in m for m in rig.messages()), rig.log
    # an unusable sample is a reason to wait too, not a fault, before the legs are commanded
    rig.run(0.1, state=_state(gyro=(math.nan, 0.0, 0.0)))
    rig.run(0.1, state=_state(quat=(0.5, 0.0, 0.0, 0.0)))
    assert rig.sup.phase == WAITING and rig.sup.fault is None
    rig.run(SAMPLE_DT)
    assert rig.sup.phase == RAMPING


def test_a_false_first_beat_latches():
    rig = Rig()
    rig.run(0.5, beating=False)
    rig.sup.on_heartbeat(False, rig.t)
    assert rig.sup.phase == FAULTED and 'estop has latched' in rig.sup.fault
    # a true beat after it changes nothing
    rig.run(1.0)
    assert rig.sup.phase == FAULTED
    assert len(rig.messages('error')) == 1


def test_silence_after_the_first_beat_latches():
    rig = Rig()
    rig.run(0.05)
    assert rig.sup.phase == RAMPING
    rig.run(0.4, beating=False)
    assert rig.sup.phase == RAMPING  # the last beat, at the first sample, is under 0.5 s old
    rig.run(0.1, beating=False)
    assert rig.sup.phase == FAULTED and 'no /safety/heartbeat' in rig.sup.fault


@pytest.mark.parametrize('phase', [RAMPING, HOLDING, RUNNING])
def test_a_fall_latches_once_the_legs_are_commanded(phase):
    rig = Rig()
    rig.to(phase)
    rig.run(0.1, state=_state(roll_deg=45.0))  # tilted, not fallen
    assert rig.sup.phase == phase
    rig.run(SAMPLE_DT, state=_state(roll_deg=70.0))
    assert rig.sup.phase == FAULTED and rig.sup.fault.startswith('fell')


@pytest.mark.parametrize('phase', [RAMPING, HOLDING, RUNNING])
def test_a_stale_lowstate_latches(phase):
    rig = Rig()
    rig.to(phase)
    rig.sup.check_stale(rig.t + 0.049)
    assert rig.sup.phase == phase
    rig.sup.check_stale(rig.t + 0.051)
    assert rig.sup.phase == FAULTED and '/lowstate stale' in rig.sup.fault


def test_a_lowstate_gap_latches():
    rig = Rig()
    rig.to(RAMPING)
    rig.t += 0.06
    rig.sup.on_lowstate(UPRIGHT, rig.t)
    assert rig.sup.phase == FAULTED and 'between samples' in rig.sup.fault


def test_a_stale_lowstate_is_no_fault_while_waiting():
    rig = Rig()
    rig.run(0.1, beating=False)
    rig.sup.check_stale(rig.t + 0.4)
    assert rig.sup.phase == WAITING


@pytest.mark.parametrize('state, message', [
    (_state(gyro=(0.0, math.inf, 0.0)), 'gyro is not finite'),
    (_state(q=[math.nan] + [0.0] * (NUM_POLICY_JOINTS - 1)), 'q is not finite'),
    (_state(quat=(1.1, 0.0, 0.0, 0.0)), 'not a unit quaternion'),
])
def test_an_unusable_state_latches_once_the_legs_are_commanded(state, message):
    rig = Rig()
    rig.to(RAMPING)
    rig.run(SAMPLE_DT, state=state)
    assert rig.sup.phase == FAULTED and message in rig.sup.fault


def test_confirm_is_accepted_only_while_holding():
    rig = Rig(wait_for_confirm=True)
    accepted, message = rig.sup.confirm()
    assert not accepted and WAITING in message
    rig.to(RAMPING)
    accepted, message = rig.sup.confirm()
    assert not accepted and RAMPING in message
    rig.sup.posed()
    assert rig.sup.phase == HOLDING
    accepted, message = rig.sup.confirm()
    assert accepted and rig.sup.phase == RUNNING
    accepted, message = rig.sup.confirm()
    assert not accepted and RUNNING in message


def test_without_confirm_the_policy_runs_once_posed():
    rig = Rig(wait_for_confirm=False)
    rig.sup.posed()  # not ramping yet: nothing
    assert rig.sup.phase == WAITING
    rig.to(RAMPING)
    rig.sup.posed()
    assert rig.sup.phase == RUNNING
    assert not rig.sup.confirm()[0]


def test_the_policy_starts_from_zero_commands():
    rig = Rig()
    rig.to(HOLDING)
    rig.sup.set_cmd_vel(0.3, 0.0, 0.1, rig.t)
    rig.sup.set_height_cmd(0.8)
    assert rig.sup.confirm()[0]
    assert np.all(rig.sup.cmd_vel(rig.t) == 0.0)
    assert rig.sup.height_cmd == DEFAULT_HEIGHT


def test_cmd_vel_is_zero_once_older_than_the_timeout():
    rig = Rig()
    assert np.all(rig.sup.cmd_vel(rig.t) == 0.0)
    rig.sup.set_cmd_vel(0.3, -0.1, 0.2, rig.t)
    assert rig.sup.cmd_vel(rig.t + 0.5) == pytest.approx([0.3, -0.1, 0.2])
    assert np.all(rig.sup.cmd_vel(rig.t + 0.501) == 0.0)


def test_a_non_finite_cmd_vel_is_zero_velocity():
    rig = Rig()
    rig.sup.set_cmd_vel(0.3, 0.0, 0.0, rig.t)
    rig.sup.set_cmd_vel(math.nan, 0.0, 0.0, rig.t)
    assert np.all(rig.sup.cmd_vel(rig.t) == 0.0)
    assert any('/cmd_vel is not finite' in m for m in rig.messages('warn'))


def test_a_non_finite_squat_command_is_ignored():
    rig = Rig()
    rig.sup.set_height_cmd(0.8)
    rig.sup.set_height_cmd(math.nan)
    assert rig.sup.height_cmd == 0.8
    assert any('squat' in m for m in rig.messages('warn'))


def test_a_policy_step_runs_only_while_running():
    rig = Rig()
    calls = []
    rig.to(HOLDING)
    assert rig.sup.step_policy(lambda: calls.append(1) or np.zeros(NUM_LEG_JOINTS)) is None
    assert not calls
    assert rig.sup.confirm()[0]
    targets = rig.sup.step_policy(lambda: np.full(NUM_LEG_JOINTS, 0.1))
    assert targets == pytest.approx(np.full(NUM_LEG_JOINTS, 0.1))


def test_a_policy_that_raises_latches():
    rig = Rig()
    rig.to(RUNNING)

    def broken():
        raise RuntimeError('boom')

    assert rig.sup.step_policy(broken) is None
    assert rig.sup.phase == FAULTED and rig.sup.fault.startswith('policy step failed')
    # faulted for good: no further step, no further error
    assert rig.sup.step_policy(lambda: np.zeros(NUM_LEG_JOINTS)) is None
    rig.sup.latch('another')
    assert len(rig.messages('error')) == 1


@pytest.mark.parametrize('output', [
    np.array([math.nan] + [0.0] * (NUM_LEG_JOINTS - 1)),
    np.zeros(NUM_LEG_JOINTS - 1),
])
def test_a_policy_output_that_is_not_12_finite_targets_latches(output):
    rig = Rig()
    rig.to(RUNNING)
    assert rig.sup.step_policy(lambda: output) is None
    assert rig.sup.phase == FAULTED and rig.sup.fault.startswith('policy output not finite')
