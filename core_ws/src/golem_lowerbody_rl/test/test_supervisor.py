"""
Check the leg node's start-up, readiness, band release and faults, with no ROS.

LegSupervisor runs on fake clocks with each robot's shipped run configs and a fake policy
runner: /lowstate at 500 Hz, its tick in ms of the node's clock unless a test says
otherwise, golem_safety's heartbeat at 10 Hz and the arm controller's commands with every
sample.
"""

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from golem_lowerbody_rl.policy import LegTargets, PolicyError
from golem_lowerbody_rl.profile import load_profile
from golem_lowerbody_rl.run_config import load_run_config
from golem_lowerbody_rl.supervisor import LegSupervisor, Phase, StateSample
from golem_lowerbody_rl.timing import TICK_MODULO

SAMPLE_DT = 0.002  # s: /lowstate at 500 Hz
TICK_S = 0.001  # s of the node's clock per tick
BEAT_DT = 0.1  # s: golem_safety's /safety/heartbeat at 10 Hz
SAMPLES_PER_STEP = 10  # samples per 20 ms policy step
RAMP_STEPS = 100  # 2 s at 50 Hz
DEFAULT_HEIGHT = 0.7
ROBOTS = ('h12', 'g1')
CONFIRM = 'ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger'
LATCH_TAIL = ('Stopped commanding the legs for good; golem_safety keeps relaying their last '
              'command, which does not balance the robot')


def _rolled(degrees):
    """Return the quaternion (w, x, y, z) of a roll of degrees about x."""
    half = math.radians(degrees) / 2
    return (math.cos(half), math.sin(half), 0.0, 0.0)


class FakeRunner:
    """Stand in for PolicyRunner: record each input and reset, return output(inp) as targets."""

    def __init__(self, legs, reset_on=('engage', 'band_release')):
        """Hold a 50 Hz spec with DEFAULT_HEIGHT and fixed gains for legs joints."""
        self.spec = SimpleNamespace(
            rate_hz=50.0, reset_on=tuple(reset_on),
            command=SimpleNamespace(height=SimpleNamespace(default=DEFAULT_HEIGHT)))
        self.kp = np.linspace(100.0, 200.0, legs)
        self.kd = np.linspace(2.0, 4.0, legs)
        self.inputs = []
        self.resets = []  # (inputs before the reset, its seed command or None)
        self.output = lambda inp: np.full(legs, 0.1)

    def step(self, inp):
        """Record inp and return output(inp) with the gains."""
        self.inputs.append(inp)
        q = np.asarray(self.output(inp), dtype=np.float32)
        return LegTargets(q=q, kp=self.kp.copy(), kd=self.kd.copy())

    def reset(self, seed_cmd=None):
        """Record a reset and its seed."""
        seed = None if seed_cmd is None else np.asarray(seed_cmd).tolist()
        self.resets.append((len(self.inputs), seed))


class Rig:
    """
    A LegSupervisor on fake clocks with one of a robot's shipped run configs.

    mono is robot time, steady the steady clock (the heartbeat's), node_t the node's clock;
    each advances by a sample's dt unless feed() says otherwise.
    """

    def __init__(self, robot='h12', run='sim', tick=1000, reset_on=('engage', 'band_release'),
                 **sections):
        """Load robot's run config, its sections' given fields replaced (startup={...})."""
        config = load_run_config(robot, run)
        for section, values in sections.items():
            config = replace(config, **{section: replace(getattr(config, section), **values)})
        self.config = config
        self.profile = load_profile(robot)
        self.leg_slots = self.profile.slots(self.profile.legs)
        legs = len(self.profile.legs)
        self.prepose = np.linspace(-0.4, 0.4, legs).astype(np.float32)
        self.legs = self.prepose.astype(np.float64)  # the measured legs
        self.runner = FakeRunner(legs, reset_on)
        self.mono, self.node_t, self.tick = 100.0, 50.0, tick
        self.steady = 200.0
        self.log = []
        self.log_times = []
        self.steps = []  # the ticks of the returned targets
        self.beating = True
        self.last_beat = None
        self.sup = LegSupervisor(config, self.profile, self.prepose, self.runner,
                                 log=self._log)

    def _log(self, level, message):
        """Record a log line and when it came."""
        self.log.append((level, message))
        self.log_times.append(self.mono)

    def sample(self, legs=None, quat=(1.0, 0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0), q=None):
        """Return a still robot's sample at the current tick, legs at legs (or self.legs)."""
        if q is None:
            q = np.zeros(len(self.profile.body_joints))
            q[self.leg_slots] = self.legs if legs is None else legs
        return StateSample(tick=self.tick, q=np.asarray(q, dtype=np.float64),
                           dq=np.zeros(len(self.profile.body_joints)),
                           quat_wxyz=np.asarray(quat, dtype=np.float64),
                           gyro=np.asarray(gyro, dtype=np.float64))

    def feed(self, n=1, dt=SAMPLE_DT, ticks=None, node_dt=None, steady_dt=None, upper=True,
             check=False, **sample):
        """
        Feed n samples dt s of robot time apart and return what each gave.

        Their ticks advance ticks (dt in ms by default), the node's clock node_dt and the
        steady clock steady_dt (dt by default). Beats come every BEAT_DT of the steady clock
        while beating, the arm controller commands with each sample while upper, and check()
        runs before each sample with check.
        """
        out = []
        for _ in range(n):
            self.mono += dt
            self.node_t += dt if node_dt is None else node_dt
            self.steady += dt if steady_dt is None else steady_dt
            step = round(dt / TICK_S) if ticks is None else ticks
            self.tick = (self.tick + step) % TICK_MODULO
            if self.beating and (self.last_beat is None
                                 or self.steady - self.last_beat >= BEAT_DT - 1e-9):
                self.sup.on_heartbeat(True, self.steady)
                self.last_beat = self.steady
            if upper:
                self.sup.on_upper_cmd(self.mono)
            if check:
                self.sup.check(self.mono, self.steady)
            targets = self.sup.on_state(self.sample(**sample), self.mono, self.node_t,
                                        self.steady)
            if targets is not None:
                self.steps.append(self.tick)
            out.append(targets)
        return out

    def until(self, phase, limit=20000, **feed):
        """Feed samples one at a time until the supervisor is in phase."""
        for _ in range(limit):
            if self.sup.phase is phase:
                return
            self.feed(1, **feed)
        raise AssertionError(f'never {phase}: {self.sup.phase}, {self.log[-5:]}')

    def to(self, phase):
        """
        Drive the supervisor to phase, confirming when the run config needs it.

        /left_ee_pose comes first, except for holding without a confirm (the sims), where
        its absence is what keeps the policy from starting.
        """
        if phase is not Phase.HOLDING or self.config.startup.wait_for_confirm:
            self.sup.on_ee_pose(self.mono)
        if phase is Phase.WAITING:
            return
        self.until(Phase.RAMPING)
        if phase is Phase.RAMPING:
            return
        if phase is Phase.HOLDING or self.config.startup.wait_for_confirm:
            self.until(Phase.HOLDING)
            self.feed(SAMPLES_PER_STEP)
        if phase is Phase.RUNNING:
            if self.config.startup.wait_for_confirm:
                assert self.sup.confirm(self.mono)[0], self.log[-3:]
            self.until(Phase.RUNNING)
            self.feed(SAMPLES_PER_STEP)
        assert self.sup.phase is phase, self.log[-3:]

    def messages(self, level=None):
        """Return the logged messages, of level only when given."""
        return [m for lvl, m in self.log if level is None or lvl == level]


def _phases(*names):
    """Return pytest parameters of the named phases."""
    return pytest.mark.parametrize('phase', [Phase[name] for name in names],
                                   ids=[name.lower() for name in names])


# -- the pre-pose -------------------------------------------------------------------------

@pytest.mark.parametrize('robot', ROBOTS)
def test_cosine_ramp_reaches_the_prepose_in_2_s(robot):
    """Blend from the measured legs to the pre-pose along a cosine over 100 steps (2 s)."""
    rig = Rig(robot, 'real')
    start = rig.prepose + np.linspace(-0.3, 0.3, rig.prepose.size)
    rig.legs = start
    out = [t for t in rig.feed((RAMP_STEPS + 1) * SAMPLES_PER_STEP) if t is not None]
    targets = np.array([t.q for t in out])
    alpha = 0.5 * (1.0 - np.cos(np.pi * np.arange(RAMP_STEPS) / RAMP_STEPS))
    expected = (1.0 - alpha)[:, None] * start + alpha[:, None] * rig.prepose
    np.testing.assert_allclose(targets[0], start, atol=1e-6)
    np.testing.assert_allclose(targets[:RAMP_STEPS], expected, atol=1e-6)
    np.testing.assert_array_equal(targets[RAMP_STEPS], rig.prepose)
    assert rig.steps[RAMP_STEPS] - rig.steps[0] == 2000
    assert set(np.diff(rig.steps).tolist()) == {20}
    np.testing.assert_array_equal(out[0].kp, rig.runner.kp)
    np.testing.assert_array_equal(out[0].kd, rig.runner.kd)
    assert not rig.runner.inputs, 'the policy must not run during the pre-pose'
    assert any(m.endswith('over 2.0 s') for m in rig.messages())


def test_settled_holds_after_10_ticks_within_tolerance():
    """Hold once the legs stay within settle_tolerance for settle_ticks steps after the ramp."""
    rig = Rig('h12', 'real')
    rig.legs = rig.prepose + 0.1
    rig.until(Phase.HOLDING)
    assert len(rig.steps) == RAMP_STEPS + 10
    assert rig.messages()[-1].startswith('legs pre-posed (settled, max error 0.100 rad) '
                                         '— holding until the operator confirms')


def test_a_step_out_of_tolerance_restarts_the_settle_count():
    """Count only consecutive steps within tolerance."""
    rig = Rig('g1', 'real')
    rig.until(Phase.RAMPING)
    rig.feed((RAMP_STEPS - 1 + 5) * SAMPLES_PER_STEP)
    rig.feed(SAMPLES_PER_STEP, legs=rig.prepose + 0.2)
    rig.feed(9 * SAMPLES_PER_STEP)
    assert rig.sup.phase is Phase.RAMPING
    rig.feed(SAMPLES_PER_STEP)
    assert rig.sup.phase is Phase.HOLDING


def test_not_settled_holds_after_400_ticks():
    """Hold the pre-pose anyway 400 steps after the ramp, logging NOT SETTLED."""
    rig = Rig('h12', 'real')
    rig.legs = rig.prepose + 0.3
    rig.until(Phase.RAMPING)
    rig.feed((RAMP_STEPS + 399) * SAMPLES_PER_STEP - 1)
    assert rig.sup.phase is Phase.RAMPING and len(rig.steps) == RAMP_STEPS + 399
    rig.feed(SAMPLES_PER_STEP)
    assert rig.sup.phase is Phase.HOLDING and len(rig.steps) == RAMP_STEPS + 400
    (warning,) = rig.messages('warn')
    assert warning.startswith('legs pre-posed (NOT SETTLED: max error 0.300 rad after 400 '
                              'steps) — holding until the operator confirms')
    held = [t for t in rig.feed(5 * SAMPLES_PER_STEP) if t is not None]
    assert len(held) == 5
    for targets in held:
        np.testing.assert_array_equal(targets.q, rig.prepose)


def test_holding_sends_the_prepose_and_reminds_the_operator():
    """Hold the pre-pose every step until the confirm, and remind every 5 s once ready."""
    rig = Rig('g1', 'real')
    rig.to(Phase.HOLDING)
    rig.steps.clear()
    held = [t for t in rig.feed(round(11.0 / SAMPLE_DT)) if t is not None]
    assert len(held) == 550 and set(np.diff(rig.steps).tolist()) == {20}
    for targets in held:
        np.testing.assert_array_equal(targets.q, rig.prepose)
    assert not rig.runner.inputs
    reminders = [m for m in rig.messages() if 'holding; the policy may start' in m]
    assert len(reminders) == 2 and all(m.endswith(CONFIRM) for m in reminders)
    assert rig.sup.wait_reason == 'the operator has not confirmed yet'


# -- waiting --------------------------------------------------------------------------------

def test_waits_for_the_first_beat():
    """Stay silent until the first true beat, and log the steady reason once."""
    rig = Rig()
    rig.beating = False
    assert rig.feed(1000) == [None] * 1000
    rig.sup.check(rig.mono + 10.0, rig.steady + 10.0)  # silence before the first beat is no fault
    assert rig.sup.phase is Phase.WAITING and rig.sup.fault is None
    assert rig.messages() == ['waiting to pre-pose the legs: no /safety/heartbeat yet '
                              '(golem_safety beats once it can relay)']
    assert rig.sup.wait_reason.startswith('no /safety/heartbeat yet')
    rig.beating = True
    assert rig.feed(1)[0] is not None
    assert rig.sup.phase is Phase.RAMPING


def test_waits_for_an_upright_finite_state():
    """Wait, not fault, on a tilted or unusable sample, then ramp from the measured legs."""
    rig = Rig('g1', 'real')
    rig.feed(50, quat=_rolled(40.0))
    rig.feed(50, gyro=(math.nan, 0.0, 0.0))
    rig.feed(50, quat=(0.5, 0.0, 0.0, 0.0))
    assert rig.sup.phase is Phase.WAITING and rig.sup.fault is None
    assert rig.messages() == [
        'waiting to pre-pose the legs: the robot is not upright (gravity z -0.77 in the IMU '
        'frame, needs < -0.9)',
        'waiting to pre-pose the legs: /lowstate gyro is not finite (at [0])',
        'waiting to pre-pose the legs: the /lowstate IMU quaternion is not a unit quaternion '
        '(norm 0.500)',
    ]
    legs = rig.prepose + 0.2
    out = rig.feed(1, legs=legs, upper=False)
    np.testing.assert_allclose(out[0].q, legs, atol=1e-6)
    assert rig.sup.phase is Phase.RAMPING


def test_a_tick_that_goes_back_while_waiting_is_no_fault():
    """Take a tick that went back before the legs are commanded (a sim restarted)."""
    rig = Rig()
    rig.beating = False
    rig.feed(10)
    rig.feed(1, ticks=-500)
    assert rig.sup.phase is Phase.WAITING and rig.sup.fault is None


def test_alternating_reasons_to_wait_are_logged_at_most_every_5_s():
    """Log each kind of reason at most every 5 s, so two that alternate do not flood."""
    rig = Rig('h12', 'sim')
    rig.until(Phase.HOLDING, upper=False)
    start = len(rig.log)
    # no arm controller for 0.6 s, then 10 ms of it with no /left_ee_pose, for 7.3 s
    for _ in range(12):
        rig.feed(300, upper=False)
        rig.feed(5)
    lines = [(m.split(' (')[0], t) for (_, m), t in zip(rig.log[start:], rig.log_times[start:])
             if m.startswith('the policy waits')]
    kinds = {kind for kind, _ in lines}
    assert kinds == {'the policy waits: no recent command from the arm controller',
                     'the policy waits: frame_task_server is not up yet'}, lines
    for kind in kinds:
        assert np.all(np.diff([t for k, t in lines if k == kind]) >= 5.0)
    assert len(lines) <= 4, lines


# -- readiness ------------------------------------------------------------------------------

@pytest.mark.parametrize('ticks', [1, 4], ids=['half', 'double'])
def test_tick_not_in_ms_never_becomes_ready(ticks):
    """Never start the policy on a tick that is not ms of the node's clock, and say why."""
    rig = Rig('h12', 'sim')
    rig.to(Phase.WAITING)
    rig.feed(round(20.0 / SAMPLE_DT), ticks=ticks)
    assert rig.sup.phase is Phase.HOLDING and rig.sup.fault is None
    assert not rig.runner.inputs
    rate = ticks / 2
    reason = (f"/lowstate's tick is off the node's clock ({rate:.3f} s of tick per s, needs "
              "0.900-1.100: the tick must count ms of the node's clock)")
    times = [t for (_, m), t in zip(rig.log, rig.log_times) if m == f'the policy waits: {reason}']
    assert 3 <= len(times) <= 4, rig.messages()
    assert np.all(np.diff(times) >= 5.0 - 1e-9) and np.all(np.diff(times) < 5.0 + 0.1)
    assert rig.sup.wait_reason == reason
    # once the tick counts ms, the policy starts within two steady windows
    rig.feed(round(2.5 / SAMPLE_DT))
    assert rig.sup.phase is Phase.RUNNING


def test_tick_not_in_ms_refuses_the_confirm():
    """Refuse the operator's confirm while the tick is off the node's clock."""
    rig = Rig('g1', 'real')
    rig.feed(round(8.0 / SAMPLE_DT), ticks=1)
    assert rig.sup.phase is Phase.HOLDING
    ok, message = rig.sup.confirm(rig.mono)
    assert not ok and message.startswith(
        "refused: the policy cannot start yet: /lowstate's tick is off the node's clock")


def test_a_sim_at_a_fifth_of_real_time_starts_the_policy():
    """Start on a tick in ms of sim time however slowly the samples come in wall time."""
    rig = Rig('g1', 'sim')
    rig.to(Phase.WAITING)
    for _ in range(4000):
        rig.feed(1, dt=0.009, ticks=2, node_dt=SAMPLE_DT)
        rig.feed(1, dt=0.018, ticks=2, node_dt=SAMPLE_DT)
        if rig.sup.phase is Phase.RUNNING:
            break
    assert rig.sup.phase is Phase.RUNNING, rig.messages()[-3:]


def test_ready_needs_fresh_upper_command():
    """Start only while the arm controller's last command is under 0.5 s old."""
    rig = Rig('g1', 'sim')
    rig.to(Phase.WAITING)
    rig.feed(round(4.0 / SAMPLE_DT), upper=False)
    assert rig.sup.phase is Phase.HOLDING and not rig.runner.inputs
    assert rig.messages('info').count(
        'the policy waits: no recent command from the arm controller (none yet)') == 1
    rig.sup.on_upper_cmd(rig.mono - 0.6)
    rig.feed(5 * SAMPLES_PER_STEP, upper=False)
    assert rig.sup.phase is Phase.HOLDING and not rig.runner.inputs
    assert rig.sup.wait_reason.startswith('no recent command from the arm controller')
    rig.sup.on_upper_cmd(rig.mono - 0.4)
    rig.feed(SAMPLES_PER_STEP, upper=False)
    assert rig.sup.phase is Phase.RUNNING and len(rig.runner.inputs) == 1
    assert any(m.startswith("the policy may start (/lowstate's tick ran at 1.000 s per s")
               for m in rig.messages())


def test_ready_needs_steady_seconds_from_the_ramp_start():
    """Start no sooner than steady_seconds of ticks after the ramp began."""
    rig = Rig('h12', 'sim', startup={'ramp_seconds': 0.1})
    rig.to(Phase.WAITING)
    rig.until(Phase.RAMPING)
    ramp_tick = rig.tick
    rig.until(Phase.RUNNING)
    assert 1000 <= rig.tick - ramp_tick < 1000 + 20


def test_confirm_refused_until_ready():
    """Accept the confirm only while holding and ready; start from zero commands."""
    rig = Rig('h12', 'real')
    ok, message = rig.sup.confirm(rig.mono)
    assert not ok and message == ('refused: the node is waiting; the policy starts on a '
                                  'confirm only while the legs hold the pre-pose (holding)')
    rig.until(Phase.RAMPING, upper=False)
    ok, message = rig.sup.confirm(rig.mono)
    assert not ok and 'the node is ramping' in message
    rig.until(Phase.HOLDING, upper=False)
    ok, message = rig.sup.confirm(rig.mono)
    assert not ok and message.startswith(
        'refused: the policy cannot start yet: no recent command from the arm controller')
    rig.feed(1)
    rig.sup.on_cmd(0.3, 0.0, 0.1, rig.mono)
    rig.sup.on_height(0.5, rig.mono)
    ok, message = rig.sup.confirm(rig.mono)
    assert ok and message == 'confirmed: the policy starts on the next policy step'
    assert rig.sup.phase is Phase.RUNNING and rig.sup.wait_reason is None
    assert rig.runner.resets == [(0, [0.0, 0.0, 0.0])]
    assert rig.messages()[-1] == 'policy started at zero velocity and the default height 0.70'
    rig.feed(SAMPLES_PER_STEP)
    (inp,) = rig.runner.inputs
    assert inp.cmd.tolist() == [0.0, 0.0, 0.0] and inp.height == DEFAULT_HEIGHT
    ok, message = rig.sup.confirm(rig.mono)
    assert not ok and 'the node is running' in message


def test_engage_resets_only_when_the_spec_says():
    """Reset the policy at engage only for a spec whose reset_on lists engage."""
    rig = Rig('h12', 'sim', reset_on=('band_release',))
    rig.to(Phase.RUNNING)
    assert rig.runner.resets == []


def test_sim_policy_waits_for_ee_pose_or_max_wait():
    """In the sim, start on the first /left_ee_pose, or max_wait after holding began."""
    rig = Rig('h12', 'sim')
    rig.feed(round(6.0 / SAMPLE_DT))
    assert rig.sup.phase is Phase.HOLDING and not rig.runner.inputs
    assert 'the policy waits: frame_task_server is not up yet (no /left_ee_pose)' in \
        rig.messages()
    rig.sup.on_ee_pose(rig.mono)
    rig.feed(SAMPLES_PER_STEP)
    assert rig.sup.phase is Phase.RUNNING and len(rig.runner.inputs) == 1
    assert not rig.messages('warn')

    rig = Rig('g1', 'sim', band={'max_wait': 3.0})
    rig.until(Phase.HOLDING)
    holding_at = rig.mono
    rig.feed(round(2.9 / SAMPLE_DT))
    assert rig.sup.phase is Phase.HOLDING
    rig.until(Phase.RUNNING)
    assert rig.mono - holding_at == pytest.approx(3.0, abs=0.021)
    (warning,) = rig.messages('warn')
    assert warning.startswith('frame_task_server is not up yet (no /left_ee_pose) 3 s after '
                              'the legs were pre-posed: the policy no longer waits for it')


def test_real_run_has_no_ee_pose_gate():
    """Accept the confirm on the real robot without any /left_ee_pose."""
    rig = Rig('g1', 'real')
    rig.until(Phase.HOLDING)
    rig.feed(SAMPLES_PER_STEP)
    assert rig.sup.confirm(rig.mono)[0]


# -- commands ------------------------------------------------------------------------------

def test_non_finite_cmd_is_zero():
    """Pass a finite /cmd_vel on raw; make a non-finite one zero velocity, with a warning."""
    rig = Rig('g1', 'sim')
    rig.to(Phase.RUNNING)
    rig.sup.on_cmd(0.3, -0.1, 2.0, rig.mono)
    rig.feed(SAMPLES_PER_STEP)
    assert rig.runner.inputs[-1].cmd.tolist() == pytest.approx([0.3, -0.1, 2.0])
    for bad in (math.nan, math.inf, -math.inf):
        rig.sup.on_cmd(0.3, bad, 0.0, rig.mono)
        rig.feed(SAMPLES_PER_STEP)
        cmd = rig.runner.inputs[-1].cmd
        assert np.all(np.isfinite(cmd)) and cmd.tolist() == [0.0, 0.0, 0.0]
    # one warning per WAIT_LOG_SECONDS, however many come
    warnings = rig.messages('warn')
    assert warnings == ['/cmd_vel is not finite ([0.3, nan, 0.0]): zero velocity']
    rig.feed(round(5.0 / SAMPLE_DT))
    rig.sup.on_cmd(math.nan, 0.0, 0.0, rig.mono)
    assert rig.messages('warn')[1:] == ['/cmd_vel is not finite ([nan, 0.0, 0.0]): zero velocity']


def test_stale_cmd_is_zero_after_0_5_s():
    """Hand on zero velocity once the last /cmd_vel is older than commands.timeout."""
    rig = Rig('h12', 'sim')
    rig.to(Phase.RUNNING)
    sent = rig.mono
    rig.sup.on_cmd(0.4, 0.0, 0.0, sent)
    first = len(rig.runner.inputs)
    rig.feed(round(1.0 / SAMPLE_DT))
    offset = rig.mono - rig.node_t
    for inp in rig.runner.inputs[first:]:
        age = inp.t + offset - sent
        if age < 0.499:
            assert inp.cmd.tolist() == pytest.approx([0.4, 0.0, 0.0])
        elif age > 0.501:
            assert inp.cmd.tolist() == [0.0, 0.0, 0.0]


def test_commands_before_the_start_are_dropped():
    """Start the policy at zero velocity and the default height whatever came before."""
    rig = Rig('h12', 'sim')
    rig.sup.on_ee_pose(rig.mono)
    rig.until(Phase.HOLDING, upper=False)
    rig.sup.on_cmd(0.5, 0.0, 0.0, rig.mono)
    rig.sup.on_height(0.4, rig.mono)
    rig.feed(SAMPLES_PER_STEP)
    assert rig.sup.phase is Phase.RUNNING
    (inp,) = rig.runner.inputs
    assert inp.cmd.tolist() == [0.0, 0.0, 0.0] and inp.height == DEFAULT_HEIGHT


def test_non_finite_height_is_ignored():
    """Hand on the height command; ignore one that is not finite, with a warning."""
    rig = Rig('g1', 'sim')
    rig.to(Phase.RUNNING)
    rig.sup.on_height(0.5, rig.mono)
    rig.sup.on_height(math.nan, rig.mono)
    rig.feed(SAMPLES_PER_STEP)
    assert rig.runner.inputs[-1].height == 0.5
    assert rig.messages('warn') == ['ignoring a height command that is not finite (nan)']
    for _ in range(3):
        rig.sup.on_height(math.inf, rig.mono)
    assert len(rig.messages('warn')) == 1, 'logged at most every WAIT_LOG_SECONDS'


# -- the elastic band ----------------------------------------------------------------------

def test_band_called_once_and_never_retried():
    """Ask for the toggle once; take a failure as final and never ask again."""
    rig = Rig('h12', 'sim')
    assert rig.sup.band_due(rig.mono) is None
    rig.to(Phase.RUNNING)
    resets = list(rig.runner.resets)
    assert rig.sup.band_due(rig.mono, service_ready=False) is None
    assert rig.sup.band_due(rig.mono, service_ready=False) is None
    assert rig.messages('warn') == ['the elastic band is due for release, but '
                                    '/elastic_band/toggle is not there yet: waiting for it']
    reason = rig.sup.band_due(rig.mono)
    assert reason == 'the policy runs and frame_task_server is up (/left_ee_pose)'
    assert rig.messages()[-1] == f'releasing the elastic band: {reason}'
    for later in (0.1, 10.0, 100.0):
        assert rig.sup.band_due(rig.mono + later) is None
    rig.sup.on_band_reply(False, 'the service answered "nope"')
    errors = rig.messages('error')
    assert errors == ['elastic band release failed: the service answered "nope". The band '
                      'stays on; the node does not call /elastic_band/toggle again']
    rig.sup.on_band_reply(True, 'disabled')
    assert rig.sup.band_due(rig.mono + 100.0) is None
    rig.feed(5 * SAMPLES_PER_STEP)
    assert rig.runner.resets == resets
    assert rig.sup.phase is Phase.RUNNING and rig.sup.fault is None
    assert len(rig.messages('error')) == 1


def test_band_success_resets_the_policy_immediately():
    """Reset the policy on the successful reply, before the next step."""
    rig = Rig('g1', 'sim')
    rig.to(Phase.RUNNING)
    rig.feed(5 * SAMPLES_PER_STEP)
    steps = len(rig.runner.inputs)
    assert rig.sup.band_due(rig.mono) is not None
    rig.sup.on_band_reply(True, 'disabled')
    assert rig.runner.resets[-1] == (steps, None)
    assert rig.messages()[-1] == ('elastic band released (/elastic_band/toggle: disabled): '
                                  'the policy restarts from an empty history')
    rig.sup.on_band_reply(True, 'disabled')
    assert len(rig.runner.resets) == 2


def test_band_reply_enabled_is_no_release():
    """Take a reply that turned the band on as final, with a warning and no reset."""
    rig = Rig('h12', 'sim')
    rig.to(Phase.RUNNING)
    assert rig.sup.band_due(rig.mono) is not None
    rig.sup.on_band_reply(True, 'enabled')
    assert len(rig.runner.resets) == 1
    (warning,) = rig.messages('warn')
    assert 'now ON' in warning
    assert rig.sup.band_due(rig.mono + 100.0) is None


def test_band_without_band_release_reset():
    """Release without a reset for a spec whose reset_on omits band_release."""
    rig = Rig('h12', 'sim', reset_on=('engage',))
    rig.to(Phase.RUNNING)
    assert rig.sup.band_due(rig.mono) is not None
    rig.sup.on_band_reply(True, 'disabled')
    assert len(rig.runner.resets) == 1
    assert rig.messages()[-1] == 'elastic band released (/elastic_band/toggle: disabled)'


def test_band_due_max_wait_after_running_without_ee_pose():
    """Ask for the toggle max_wait after the policy started when no /left_ee_pose came."""
    rig = Rig('g1', 'sim', band={'max_wait': 2.0})
    rig.until(Phase.RUNNING)
    started = rig.mono
    while rig.mono - started < 2.0 - 0.01:
        assert rig.sup.band_due(rig.mono) is None
        rig.feed(1)
    rig.feed(10)
    assert 'maximum wait' in rig.sup.band_due(rig.mono)


def test_real_run_never_asks_for_the_band():
    """Never ask for the toggle on the real robots, which have no band."""
    rig = Rig('h12', 'real')
    rig.to(Phase.RUNNING)
    assert rig.sup.band_due(rig.mono + 100.0) is None


# -- faults ---------------------------------------------------------------------------------

def _waiting_after_a_beat(rig):
    """Bring rig past its first beat but keep it waiting, tilted."""
    rig.feed(50, quat=_rolled(40.0))


@_phases('WAITING', 'RAMPING', 'HOLDING', 'RUNNING')
def test_false_heartbeat_latches_in_every_phase(phase):
    """Stop for good on a false heartbeat, whatever the phase."""
    rig = Rig('g1', 'real')
    if phase is Phase.WAITING:
        _waiting_after_a_beat(rig)
    else:
        rig.to(phase)
    steps = len(rig.runner.inputs)
    rig.sup.on_heartbeat(False, rig.steady)
    reason = "golem_safety's /safety/heartbeat is false: its estop has latched"
    assert rig.sup.phase is Phase.FAULTED and rig.sup.fault == reason
    assert rig.feed(200) == [None] * 200
    assert len(rig.runner.inputs) == steps
    (error,) = rig.messages('error')
    assert error == (f'fault: {reason}. Stopped commanding the legs for good. Restart the '
                     'bringup to clear')


def test_a_false_first_beat_latches():
    """Stop for good on a false beat before any true one; later beats change nothing."""
    rig = Rig()
    rig.beating = False
    rig.feed(50)
    rig.sup.on_heartbeat(False, rig.steady)
    rig.beating = True
    assert rig.feed(1000) == [None] * 1000
    assert rig.sup.phase is Phase.FAULTED and len(rig.messages('error')) == 1


@_phases('WAITING', 'RAMPING', 'HOLDING', 'RUNNING')
def test_silent_heartbeat_latches_after_0_5_s(phase):
    """Stop for good when no beat came for heartbeat.timeout after the first."""
    rig = Rig('h12', 'sim')
    if phase is Phase.WAITING:
        _waiting_after_a_beat(rig)
    else:
        rig.to(phase)
    rig.beating = False
    last = rig.last_beat
    quat = _rolled(40.0) if phase is Phase.WAITING else (1.0, 0.0, 0.0, 0.0)
    while rig.steady + SAMPLE_DT - last < 0.5 - 1e-6:
        rig.feed(1, check=True, quat=quat)
    assert rig.sup.phase is phase and rig.sup.fault is None
    rig.sup.check(rig.mono, last + 0.51)
    assert rig.sup.phase is Phase.FAULTED
    assert rig.sup.fault.startswith('no /safety/heartbeat for 510 ms (> 500 ms)')
    assert rig.feed(100, quat=quat) == [None] * 100


@_phases('WAITING', 'RAMPING', 'HOLDING', 'RUNNING')
def test_the_heartbeat_watchdog_counts_steady_time_while_robot_time_stands_still(phase):
    """SAFETY: with robot time stopped (a sim's /clock), a silent heartbeat still latches."""
    rig = Rig('h12', 'sim')
    if phase is Phase.WAITING:
        _waiting_after_a_beat(rig)
    else:
        rig.to(phase)
    rig.beating = False
    rig.sup.check(rig.mono, rig.last_beat + 0.49)
    assert rig.sup.phase is phase and rig.sup.fault is None
    rig.sup.check(rig.mono, rig.last_beat + 0.51)
    assert rig.sup.fault.startswith('no /safety/heartbeat for 510 ms (> 500 ms)')


@_phases('RAMPING', 'HOLDING', 'RUNNING')
def test_the_state_gap_counts_robot_time_not_steady_time(phase):
    """A sim much slower than the wall clock is no gap; a gap of robot time is."""
    rig = Rig('h12', 'sim')
    rig.to(phase)
    # 2 s of steady time between samples 4 ms of robot time apart: a sim at 1/500 speed
    rig.feed(5, steady_dt=0.4)
    assert rig.sup.phase is phase and rig.sup.fault is None
    max_gap = rig.config.faults.max_state_gap
    rig.sup.check(rig.mono + max_gap + 0.01, rig.steady)
    assert rig.sup.fault == (f'/lowstate stale: no sample for {(max_gap + 0.01) * 1000:.0f} '
                             f'ms (> {max_gap * 1000:.0f} ms)')


@pytest.mark.parametrize('robot', ROBOTS)
@pytest.mark.parametrize('how', ['between samples', 'silence'])
@_phases('RAMPING', 'HOLDING', 'RUNNING')
def test_gap_over_limit_faults_once_commanded(robot, how, phase):
    """Stop for good on a /lowstate gap over max_state_gap once the legs are commanded."""
    rig = Rig(robot, 'real')
    rig.beating = False
    rig.feed(1, dt=1.0)  # waiting: a gap is no fault
    rig.beating = True
    rig.to(phase)
    if how == 'between samples':
        rig.feed(1, dt=0.24)
        assert rig.sup.phase is phase
        rig.feed(1, dt=0.26)
        assert rig.sup.fault == '/lowstate stale: 260 ms between samples (> 250 ms)'
    else:
        rig.sup.check(rig.mono + 0.249, rig.steady)
        assert rig.sup.phase is phase
        rig.sup.check(rig.mono + 0.251, rig.steady)
        assert rig.sup.fault == '/lowstate stale: no sample for 251 ms (> 250 ms)'
    assert rig.sup.phase is Phase.FAULTED
    (error,) = rig.messages('error')
    assert LATCH_TAIL in error and error.endswith('Restart the node to clear')
    assert rig.feed(100) == [None] * 100


def test_a_tick_that_did_not_advance_is_ignored_and_not_fresh():
    """Ignore samples whose tick did not advance; they do not keep /lowstate fresh."""
    rig = Rig('g1', 'real')
    rig.to(Phase.RUNNING)
    steps = len(rig.runner.inputs)
    assert rig.feed(130, ticks=0) == [None] * 130
    assert len(rig.runner.inputs) == steps
    rig.sup.check(rig.mono, rig.steady)
    assert rig.sup.fault == '/lowstate stale: no sample for 260 ms (> 250 ms)'


@_phases('RAMPING', 'HOLDING', 'RUNNING')
def test_tick_back_faults(phase):
    """Stop for good on a tick that goes back once the legs are commanded."""
    rig = Rig('h12', 'sim')
    rig.to(phase)
    before = rig.tick
    rig.feed(1, ticks=-10)
    assert rig.sup.phase is Phase.FAULTED
    assert rig.sup.fault == f'/lowstate tick went back from {before} to {before - 10}'


@pytest.mark.parametrize('robot', ROBOTS)
@pytest.mark.parametrize('run', ['sim', 'real'])
def test_latency_off_when_null(robot, run):
    """Never fault on tick lag where the run config's max_latency is null."""
    rig = Rig(robot, run)
    assert rig.config.faults.max_latency is None
    rig.to(Phase.RUNNING)
    rig.feed(1, dt=0.2, ticks=2)  # 198 ms late
    rig.feed(5 * SAMPLES_PER_STEP)
    assert rig.sup.phase is Phase.RUNNING and rig.sup.fault is None


def test_latency_over_max_latency_faults_only_while_running():
    """Stop for good on a tick lag over max_latency while running, not before."""
    rig = Rig('g1', 'real', faults={'max_latency': 0.025})
    rig.to(Phase.HOLDING)
    rig.feed(1, dt=0.032, ticks=2)  # 30 ms late
    rig.feed(1)
    assert rig.sup.phase is Phase.HOLDING
    rig = Rig('g1', 'real', faults={'max_latency': 0.025})
    rig.to(Phase.RUNNING)
    rig.feed(1, dt=0.022, ticks=2)  # 20 ms late
    assert rig.sup.phase is Phase.RUNNING
    rig.feed(1, dt=0.012, ticks=2)  # 30 ms late
    assert rig.sup.phase is Phase.FAULTED
    assert rig.sup.fault.startswith('/lowstate late: tick lag 30.0 ms (> 25 ms)')


@_phases('RAMPING', 'HOLDING', 'RUNNING')
def test_fall_faults(phase):
    """Stop for good when gravity z rises above -0.5 once the legs are commanded."""
    rig = Rig('h12', 'real')
    rig.to(phase)
    rig.feed(5, quat=_rolled(45.0))
    assert rig.sup.phase is phase
    rig.feed(1, quat=_rolled(70.0))
    assert rig.sup.phase is Phase.FAULTED
    assert rig.sup.fault == 'fell: gravity z -0.34 in the IMU frame (> -0.5)'


@pytest.mark.parametrize('robot', ROBOTS)
@pytest.mark.parametrize('field, value, message', [
    ('gyro', (0.0, math.inf, 0.0), '/lowstate gyro is not finite (at [1])'),
    ('knee', math.nan, "/lowstate q is not finite (at ['left_knee_joint'])"),
    ('quat', (1.1, 0.0, 0.0, 0.0),
     'the /lowstate IMU quaternion is not a unit quaternion (norm 1.100)'),
], ids=['gyro', 'q', 'quaternion'])
def test_an_unusable_state_faults_once_commanded(robot, field, value, message):
    """Stop for good on a non-finite value or a bad quaternion once commanded."""
    rig = Rig(robot, 'real')
    rig.to(Phase.RAMPING)
    if field == 'knee':
        q = np.zeros(len(rig.profile.body_joints))
        q[rig.profile.slots(['left_knee_joint'])] = value
        rig.feed(1, q=q)
    elif field == 'gyro':
        rig.feed(1, gyro=value)
    else:
        rig.feed(1, quat=value)
    assert rig.sup.phase is Phase.FAULTED and rig.sup.fault == message


@pytest.mark.parametrize('output, message', [
    (PolicyError('policy p: the network output is not 12 finite values'),
     'policy step failed: policy p: the network output is not 12 finite values'),
    (RuntimeError('boom'), "policy step failed: RuntimeError('boom')"),
    (np.full(12, math.nan), 'policy output is not 12 finite targets'),
    (np.zeros(11), 'policy output is not 12 finite targets'),
], ids=['policy error', 'other error', 'not finite', 'wrong size'])
def test_policy_error_faults_and_stops_output(output, message):
    """Stop for good when a policy step fails or gives anything but 12 finite targets."""
    rig = Rig('g1', 'sim')
    rig.to(Phase.RUNNING)

    def step(inp):
        if isinstance(output, Exception):
            raise output
        return output

    rig.runner.output = step
    steps = len(rig.runner.inputs)
    assert rig.feed(SAMPLES_PER_STEP) == [None] * SAMPLES_PER_STEP
    assert rig.sup.phase is Phase.FAULTED and rig.sup.fault.startswith(message)
    assert rig.feed(20 * SAMPLES_PER_STEP) == [None] * (20 * SAMPLES_PER_STEP)
    assert len(rig.runner.inputs) == steps + 1
    (error,) = rig.messages('error')
    assert LATCH_TAIL in error
    rig.sup.latch('another')
    assert len(rig.messages('error')) == 1


def test_faulted_refuses_everything():
    """Refuse the confirm and the band once faulted."""
    rig = Rig('h12', 'sim')
    rig.to(Phase.RUNNING)
    rig.sup.latch('test')
    assert rig.sup.phase is Phase.FAULTED and rig.sup.fault == 'test'
    ok, message = rig.sup.confirm(rig.mono)
    assert not ok and 'the node is faulted' in message
    assert rig.sup.band_due(rig.mono) is None
    assert rig.sup.wait_reason is None


def test_prepose_must_be_one_finite_angle_per_leg():
    """Refuse a pre-pose that is not one finite angle per leg."""
    profile = load_profile('h12')
    config = load_run_config('h12', 'sim')
    runner = FakeRunner(12)
    for prepose in (np.zeros(11), np.full(12, math.nan)):
        with pytest.raises(ValueError, match='pre-pose'):
            LegSupervisor(config, profile, prepose, runner, log=lambda *_: None)


# -- the tick grid -------------------------------------------------------------------------

@pytest.mark.parametrize('spacing', [
    [2],
    [6],
    [2, 4, 2, 8, 6, 2, 12, 2, 16, 4],
], ids=['every sample', 'every third', 'irregular'])
def test_policy_steps_on_the_tick_grid_whatever_samples_arrive(spacing):
    """Step on a 20 ms grid of ticks however the samples are spaced."""
    rig = Rig('g1', 'real')
    rig.to(Phase.RUNNING)
    rig.steps.clear()
    for i in range(3000):
        ticks = spacing[i % len(spacing)]
        rig.feed(1, dt=ticks * TICK_S, ticks=ticks)
    gaps = np.diff(rig.steps)
    assert gaps.min() >= 10, gaps.min()
    if max(spacing) <= 10:
        assert abs(gaps.mean() - 20.0) < 0.05, gaps.mean()
    if spacing == [2]:
        assert set(gaps.tolist()) == {20}
    assert rig.sup.fault is None


def test_after_a_backlog_the_newest_sample_steps_once():
    """Step once on the newest sample after a backlog, then return to the grid."""
    rig = Rig('g1', 'real')
    rig.to(Phase.RUNNING)
    rig.feed(SAMPLES_PER_STEP)
    last = rig.steps[-1]
    out = rig.feed(1, dt=0.046)
    assert out[0] is not None and rig.steps[-1] == last + 46
    rig.feed(3 * SAMPLES_PER_STEP)
    assert rig.steps[-3:] == [last + 60, last + 80, last + 100]
    assert rig.sup.fault is None


def test_the_tick_wraps_while_running():
    """Keep the 20 ms grid across the uint32 tick's wrap."""
    rig = Rig('h12', 'sim', tick=TICK_MODULO - 6000)
    rig.to(Phase.RUNNING)
    rig.steps.clear()
    rig.feed(500 * SAMPLES_PER_STEP)
    assert rig.steps[0] > TICK_MODULO // 2 and rig.steps[-1] < TICK_MODULO // 2, 'it wrapped'
    assert {(b - a) % TICK_MODULO for a, b in zip(rig.steps, rig.steps[1:])} == {20}
    assert rig.sup.fault is None


def test_the_policy_input_carries_the_sample_and_the_node_clock():
    """Hand the policy the sample's state and the node clock's time."""
    rig = Rig('g1', 'sim')
    rig.to(Phase.RUNNING)
    inp = rig.runner.inputs[-1]
    assert rig.node_t - 0.021 < inp.t <= rig.node_t
    np.testing.assert_array_equal(inp.q[rig.leg_slots], rig.legs)
    assert inp.quat_wxyz.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_timing_is_logged_every_10_s():
    """Log the step spacing and the tick lag every 10 s."""
    rig = Rig('h12', 'sim')
    rig.to(Phase.RUNNING)
    rig.log.clear()
    rig.feed(round(10.5 / SAMPLE_DT))
    timing = [m for m in rig.messages('info') if m.startswith('timing')]
    assert len(timing) == 1, timing
    assert timing[0].startswith('timing over the last 10 s: ')
    assert '20-20 ms of /lowstate tick apart (mean 20.0 ms)' in timing[0]
    assert timing[0].endswith('tick lag median 0.0 ms, max 0.0 ms (unchecked)')


# -- readiness over the last steady window --------------------------------------------------

@pytest.mark.parametrize('robot', ROBOTS)
def test_a_recent_stall_after_a_long_hold_is_not_ready(robot):
    """
    SAFETY: judge the tick over about the last steady_seconds, not since the hold began.

    After a minute of steady hold, 2 s in which /lowstate's tick advances 1 ms per 2 ms of
    the node's clock make the policy not ready, and the confirm is refused.
    """
    rig = Rig(robot, 'real')
    rig.until(Phase.HOLDING)
    rig.feed(round(62.0 / SAMPLE_DT))
    assert rig.sup.wait_reason == 'the operator has not confirmed yet'
    rig.feed(round(2.0 / SAMPLE_DT), ticks=1)
    assert rig.sup.wait_reason.startswith("/lowstate's tick is off the node's clock"), \
        rig.sup.wait_reason
    ok, message = rig.sup.confirm(rig.mono)
    assert not ok and "tick is off the node's clock" in message
    assert rig.sup.phase is Phase.HOLDING and not rig.runner.inputs


@pytest.mark.parametrize('robot', ROBOTS)
@pytest.mark.parametrize('rate', (0.5, 0.05, 0.01))
def test_a_slowed_tick_is_not_ready_within_steady_seconds_of_the_node_clock(robot, rate):
    """
    SAFETY: judge a slowed tick within steady_seconds of the node's clock.

    The steady window closes after steady_seconds of the node's clock or of tick. After a
    minute of steady hold, /lowstate samples whose tick advances at rate s per s of the
    node's clock, every sample within max_state_gap, make the policy not ready within
    steady_seconds (1 s) of the node's clock and one sample.
    """
    rig = Rig(robot, 'real')
    rig.until(Phase.HOLDING)
    rig.feed(round(62.0 / SAMPLE_DT))
    assert rig.sup.wait_reason == 'the operator has not confirmed yet'
    dt = max(0.02, TICK_S / rate)  # s between samples: at least one tick each
    start = rig.node_t
    while rig.sup.wait_reason == 'the operator has not confirmed yet':
        rig.feed(1, dt=dt, ticks=round(rate * dt / TICK_S))
        assert rig.node_t - start <= 1.0 + dt + 1e-9, 'still ready after steady_seconds'
    assert rig.sup.wait_reason.startswith("/lowstate's tick is off the node's clock"), \
        rig.sup.wait_reason
    assert not rig.sup.confirm(rig.mono)[0]


@pytest.mark.parametrize('robot', ROBOTS)
def test_a_ready_node_stays_ready_while_the_next_window_fills(robot):
    """Keep the last window's verdict while the next one fills: ready stays ready."""
    rig = Rig(robot, 'real')
    rig.until(Phase.HOLDING)
    rig.feed(SAMPLES_PER_STEP)
    while rig.sup.wait_reason != 'the operator has not confirmed yet':
        rig.feed(1)
    for _ in range(round(3.0 / SAMPLE_DT)):
        rig.feed(1)
        assert rig.sup.wait_reason == 'the operator has not confirmed yet'
    assert rig.messages().count("the policy may start (/lowstate's tick ran at 1.000 s per s "
                                "of the node's clock)") == 1
    assert rig.sup.confirm(rig.mono)[0]


def test_the_rate_is_logged_to_three_decimals():
    """Print the tick's rate to 0.001 s per s, so a rate just under the limit reads so."""
    rig = Rig('h12', 'real')
    rig.feed(round(4.0 / SAMPLE_DT), node_dt=SAMPLE_DT / 0.899)
    assert rig.sup.phase is Phase.HOLDING
    assert rig.sup.wait_reason == ("/lowstate's tick is off the node's clock (0.899 s of tick "
                                   "per s, needs 0.900-1.100: the tick must count ms of the "
                                   "node's clock)")


def test_a_node_clock_that_stands_still_is_named():
    """Say that the node's clock does not advance (no /clock) rather than blame the tick."""
    rig = Rig('g1', 'sim')
    rig.to(Phase.WAITING)
    rig.feed(round(4.0 / SAMPLE_DT), node_dt=0.0)
    assert rig.sup.phase is Phase.HOLDING and not rig.runner.inputs
    reason = ("the node's clock is not advancing (it stood still over 1000 ms of /lowstate "
              'tick: with use_sim_time, nothing publishes /clock?)')
    assert rig.sup.wait_reason == reason
    assert f'the policy waits: {reason}' in rig.messages()


# -- the heartbeat on every sample ----------------------------------------------------------

@_phases('RAMPING', 'HOLDING', 'RUNNING')
def test_a_silent_heartbeat_stops_the_next_sample_without_check(phase):
    """SAFETY: a sample after heartbeat.timeout of silence latches, whether check() ran or not."""
    rig = Rig('h12', 'sim')
    rig.to(phase)
    rig.beating = False
    last = rig.last_beat
    out = []
    while rig.sup.phase is not Phase.FAULTED and rig.steady - last < 1.0:
        out += rig.feed(1)
    assert rig.sup.phase is Phase.FAULTED and out[-1] is None
    assert 0.5 < rig.steady - last <= 0.5 + SAMPLE_DT + 1e-9
    assert rig.sup.fault.startswith('no /safety/heartbeat for ')


def test_a_silent_heartbeat_never_starts_the_ramp():
    """SAFETY: an upright sample after heartbeat.timeout of silence latches; no ramp starts."""
    rig = Rig('g1', 'real')
    _waiting_after_a_beat(rig)
    rig.beating = False
    assert rig.feed(1, dt=0.6) == [None]
    assert rig.sup.phase is Phase.FAULTED
    assert rig.feed(10) == [None] * 10


# -- failures inside the supervisor ---------------------------------------------------------

def _raise_on_reset(rig):
    """Make rig's runner raise on every reset."""
    def reset(seed_cmd=None):
        raise RuntimeError('reset broke')
    rig.runner.reset = reset


def test_an_engage_reset_that_fails_refuses_the_confirm_and_latches():
    """SAFETY: a policy that cannot reset at engage never runs; the confirm says why."""
    rig = Rig('h12', 'real')
    rig.to(Phase.HOLDING)
    _raise_on_reset(rig)
    ok, message = rig.sup.confirm(rig.mono)
    assert not ok
    assert message == "refused: the policy could not start: policy reset failed: " \
        "RuntimeError('reset broke')"
    assert rig.sup.phase is Phase.FAULTED
    assert rig.feed(5 * SAMPLES_PER_STEP) == [None] * (5 * SAMPLES_PER_STEP)
    assert not rig.runner.inputs


def test_a_band_reset_that_fails_latches():
    """SAFETY: a policy that cannot reset at the band's release stops the legs for good."""
    rig = Rig('g1', 'sim')
    rig.to(Phase.RUNNING)
    assert rig.sup.band_due(rig.mono) is not None
    _raise_on_reset(rig)
    steps = len(rig.runner.inputs)
    rig.sup.on_band_reply(True, 'disabled')
    assert rig.sup.phase is Phase.FAULTED
    assert rig.sup.fault == "policy reset failed: RuntimeError('reset broke')"
    assert rig.feed(5 * SAMPLES_PER_STEP) == [None] * (5 * SAMPLES_PER_STEP)
    assert len(rig.runner.inputs) == steps


@_phases('WAITING', 'RUNNING')
def test_an_unexpected_error_in_on_state_latches(phase):
    """SAFETY: an exception inside on_state latches "unexpected error" and returns None."""
    rig = Rig('h12', 'sim')
    rig.to(phase)
    sample = replace(rig.sample(), tick=None)
    assert rig.sup.on_state(sample, rig.mono + SAMPLE_DT, rig.node_t + SAMPLE_DT,
                            rig.steady + SAMPLE_DT) is None
    assert rig.sup.phase is Phase.FAULTED
    assert rig.sup.fault.startswith('unexpected error: TypeError(')
    assert rig.feed(10) == [None] * 10


@pytest.mark.parametrize('name, value', [
    ('kp', np.full(12, math.nan)), ('kd', np.zeros(11)), ('kp', np.full(12, math.inf)),
], ids=['kp nan', 'kd size', 'kp inf'])
def test_gains_must_be_one_finite_value_per_leg(name, value):
    """SAFETY: refuse policy gains that are not one finite value per leg."""
    profile = load_profile('h12')
    config = load_run_config('h12', 'sim')
    runner = FakeRunner(12)
    setattr(runner, name, value)
    with pytest.raises(ValueError, match=f'the policy {name} must be 12 finite gains'):
        LegSupervisor(config, profile, np.zeros(12), runner, log=lambda *_: None)


# -- the band once faulted ------------------------------------------------------------------

@pytest.mark.parametrize('robot', ROBOTS)
def test_a_faulted_node_never_releases_the_band(robot):
    """SAFETY: a node that faulted while running, /left_ee_pose seen, never asks for the band."""
    rig = Rig(robot, 'sim')
    rig.to(Phase.RUNNING)
    rig.feed(1, quat=_rolled(70.0))
    assert rig.sup.phase is Phase.FAULTED
    for later in (0.0, 1.0, 100.0):
        assert rig.sup.band_due(rig.mono + later) is None
    assert not any(m.startswith('releasing the elastic band') for m in rig.messages())
