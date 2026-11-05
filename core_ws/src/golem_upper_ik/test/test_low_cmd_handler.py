"""Check LowCmdHandler, FollowGate and relays_commands, on both robots.

LowCmdHandler runs on domain 92 against a fake robot publishing /lowstate: the held joints,
the follow gate, command clipping, no controller-side estop, no gating without a measured
state, and the shutdown command. It reads the measured state through a shared RobotModel
whose state source is a StateSubscriber on the fake /lowstate; every test puts the model
back at rest. FollowGate and relays_commands are then checked on their own.
"""
import time

import numpy as np
import pytest
from golem_upper_ik.profile import known_robots

from ros_harness import Echo, FakeRobot, Recorder, upper_q, wait_for

ROBOTS = known_robots()
CONFIGS = {'h12': 'safety_split', 'g1': 'g1_split'}
SAG = -0.08  # rad: where the fake robot's held joints (torso, waist) sit
ARM_Q = 0.1  # rad: the fake robot's arm joints
FLOAT32 = 1e-6  # rad: what a command loses as float32


def profile_of(robot):
    """Return robot's RobotProfile."""
    from golem_upper_ik.profile import load_profile
    return load_profile(robot)


class Rig:
    """Hold a fake robot, a StateSubscriber feeding the shared model, and a command recorder."""

    def __init__(self, ros, models, robot, config=None, **controller):
        """Start the fake robot (arms at ARM_Q, held joints at SAG) and the subscriber."""
        from golem_upper_ik.config import load_controller_config
        from golem_upper_ik.core.ros_io import StateSubscriber
        self.ros = ros
        self.profile = profile_of(robot)
        self.config = load_controller_config(robot, config or CONFIGS[robot])
        self.config['controller'].update(controller)
        self.upper = list(self.profile.upper_slots)
        self.held = [self.profile.body_joints.index(j) for j in self.profile.held_joints]
        self.arms = [slot for slot in self.upper if slot not in self.held]

        self.robot = FakeRobot(ros, self.profile)
        self.robot.q[self.arms] = ARM_Q
        self.robot.q[self.held] = SAG
        self.measured = self.robot.q.copy()
        self.robot.start()
        self.node = ros.node('controller')
        self.subscriber = StateSubscriber(self.node, self.profile)
        ros.on_close(self.subscriber.shutdown)
        assert wait_for(lambda: self.subscriber.latest() is not None, 2.0)
        self.model = models(robot)
        self.model.state_subscriber = self.subscriber
        ros.on_close(self._release_model)
        self.recorder = Recorder(ros, self.config['topics']['low_cmd'])

    def _release_model(self):
        """Detach the shared model's state source."""
        self.model.state_subscriber = None

    def handler(self):
        """Return a LowCmdHandler on the rig, shut down at the test's end."""
        from golem_upper_ik.core.low_cmd_handler import LowCmdHandler
        handler = LowCmdHandler(self.node, self.profile, self.config, self.model)
        self.ros.on_close(handler.shutdown)
        return handler

    def clip(self, q):
        """Return q clipped to the config's position clip limits."""
        limits = self.config['limits']['q_clip_limits']
        return np.clip(q, limits[:, 0], limits[:, 1])

    def echo(self, **kwargs):
        """Return an Echo relaying the commands on golem_safety's output (/lowcmd)."""
        topics = self.config['topics']
        return Echo(self.ros, topics['low_cmd'], '/lowcmd', **kwargs)


def command(rig, handler, q):
    """Send one control step: the body q, dq 0 and tau 0, as the controller does."""
    count = len(rig.profile.body_joints)
    handler.set_joint_commands(q=q, dq=np.zeros(count), tau=np.zeros(count))


@pytest.mark.parametrize('robot', ROBOTS)
def test_held_joints_carry_hold_target(ros, models, robot):
    """Every command carries the hold, which only set_hold moves; dq and tau set as any joint's."""
    rig = Rig(ros, models, robot, follow_tolerance=0.0)
    targets = {'h12': [0.05], 'g1': [0.03, -0.04, 0.06]}[robot]
    handler = rig.handler()

    # enabled where the robot is: the hold starts at the measured, sagging angles
    handler.start()
    np.testing.assert_allclose(handler.hold, SAG, atol=FLOAT32)
    assert wait_for(lambda: rig.recorder.count > 0, 2.0)

    held_in_upper = [rig.upper.index(slot) for slot in rig.held]

    def carries_targets(msg):
        """Return whether msg's held joints carry the targets."""
        return np.allclose(upper_q(msg, rig.profile)[held_in_upper], targets, atol=FLOAT32)

    handler.set_hold(targets)
    # a message already on its way may still carry the old hold; from the first that carries
    # the targets on, every one does
    assert wait_for(lambda: carries_targets(rig.recorder.last), 2.0)
    start = time.monotonic()
    # the IK seeds from the measured state, so its held joints carry the sag
    for _ in range(10):
        command(rig, handler, rig.measured)
        time.sleep(0.01)
    q_cmd = handler.q_cmd
    np.testing.assert_allclose(q_cmd[rig.held], targets, atol=FLOAT32)
    np.testing.assert_allclose(q_cmd[rig.arms], ARM_Q, atol=FLOAT32)
    sent = rig.recorder.since(start)
    assert len(sent) > 10
    for msg in sent:
        np.testing.assert_allclose(upper_q(msg, rig.profile)[held_in_upper], targets,
                                   atol=FLOAT32)
    # while the robot still reports the sag
    np.testing.assert_allclose(rig.subscriber.state['q'][rig.held], SAG, atol=FLOAT32)

    # a held joint's q moves only through set_hold; its dq and tau are set as any joint's
    with pytest.raises(ValueError, match='held'):
        handler.set_joint_commands(q=np.zeros(1), joint_ids=[rig.held[0]])
    handler.set_joint_commands(dq=[0.01], tau=[1.5], joint_ids=[rig.held[0]])
    np.testing.assert_allclose(handler.q_cmd[rig.held], targets, atol=FLOAT32)
    assert wait_for(lambda: rig.recorder.last.motor_cmd[rig.held[0]].tau == np.float32(1.5),
                    2.0)
    assert rig.recorder.last.motor_cmd[rig.held[0]].dq == np.float32(0.01)


@pytest.mark.parametrize('robot', ROBOTS)
def test_follow_gate_holds_until_relayed(ros, models, robot):
    """Until relayed, every command stays within the tolerance of the measured pose; then free."""
    rig = Rig(ros, models, robot)
    tolerance = rig.config['controller']['follow_tolerance']
    assert tolerance == 0.05
    handler = rig.handler()
    handler.start()
    assert not handler.following

    target = rig.measured.copy()
    target[rig.arms] += 0.3
    start = time.monotonic()
    # 0.5 s of control steps toward a target 0.3 rad away, with nothing relaying them
    while time.monotonic() - start < 0.5:
        command(rig, handler, target)
        gap = np.abs(handler.q_cmd[rig.upper] - rig.measured[rig.upper])
        assert np.max(gap) <= tolerance + FLOAT32
        time.sleep(1.0 / 30.0)
    assert not handler.following
    sent = rig.recorder.since(start)
    assert len(sent) > 100
    for msg in sent:
        gap = np.abs(upper_q(msg, rig.profile) - rig.measured[rig.upper])
        assert np.max(gap) <= tolerance + FLOAT32

    Echo(ros, rig.config['topics']['low_cmd'], '/lowcmd')
    assert wait_for(lambda: handler.following, 3.0)

    # open: the commands go where the controller sends them
    command(rig, handler, target)
    expected = rig.clip(target).astype(np.float32)
    np.testing.assert_array_equal(handler.q_cmd[rig.upper], expected[rig.upper])
    assert np.max(np.abs(expected[rig.arms] - rig.measured[rig.arms])) > 0.2
    assert wait_for(lambda: np.array_equal(upper_q(rig.recorder.last, rig.profile),
                                           expected[rig.upper]), 2.0)
    # and stay there although the robot does not follow
    time.sleep(0.1)
    np.testing.assert_array_equal(handler.q_cmd[rig.upper], expected[rig.upper])


@pytest.mark.parametrize('robot', ROBOTS)
def test_no_controller_side_estop(ros, models, robot):
    """Measured velocities past the limits stop nothing: commands go on with the motors on."""
    rig = Rig(ros, models, robot, follow_tolerance=0.0)
    # every upper joint measured at twice its velocity limit
    rig.robot.dq[rig.upper] = 2.0 * rig.profile.dq_max[rig.upper]
    assert wait_for(
        lambda: np.all(np.abs(rig.subscriber.state['dq'][rig.upper])
                       > rig.profile.dq_max[rig.upper]), 2.0)
    handler = rig.handler()
    assert not hasattr(handler, 'estop')
    handler.start()
    start = time.monotonic()
    while time.monotonic() - start < 0.5:
        command(rig, handler, rig.measured)
        time.sleep(1.0 / 30.0)
    end = time.monotonic()
    # still publishing at its rate, with the motors on
    sent = [msg for t, msg in rig.recorder.samples if start + 0.1 <= t <= end]
    assert len(sent) >= 0.8 * 500 * (end - start - 0.1)
    assert all(msg.motor_cmd[slot].mode == 1 and msg.motor_cmd[slot].kp > 0
               for msg in sent for slot in rig.upper if rig.config['gains']['kp'][slot] > 0)


def test_commands_are_clipped(ros, models):
    """q, dq and tau are clipped to the config's limits, for every joint or a subset."""
    rig = Rig(ros, models, 'h12', follow_tolerance=0.0)
    handler = rig.handler()
    handler.start()
    count = len(rig.profile.body_joints)
    limits = rig.config['limits']
    q = rig.measured.copy()
    q[rig.arms] = rig.profile.q_high[rig.arms] + 1.0
    dq = np.full(count, 1e3)
    tau = np.full(count, -1e3)
    handler.set_joint_commands(q=q, dq=dq, tau=tau)
    np.testing.assert_allclose(handler.q_cmd[rig.arms], limits['q_clip_limits'][rig.arms, 1],
                               atol=FLOAT32)
    assert wait_for(lambda: rig.recorder.last is not None
                    and rig.recorder.last.motor_cmd[rig.arms[0]].dq > 0.0, 2.0)
    msg = rig.recorder.last
    for slot in rig.upper:
        assert msg.motor_cmd[slot].dq == pytest.approx(limits['dq_clip_limits'][slot], rel=1e-6)
        assert msg.motor_cmd[slot].tau == pytest.approx(-limits['tau_clip_limits'][slot],
                                                        rel=1e-6)
    # a subset of joints
    handler.set_joint_commands(q=np.array([-10.0]), joint_ids=[rig.arms[0]])
    assert handler.q_cmd[rig.arms[0]] == pytest.approx(limits['q_clip_limits'][rig.arms[0], 0],
                                                       abs=FLOAT32)


def test_non_finite_command_is_refused(ros, models):
    """A command before start() or with a non-finite value is refused; the last good one stands."""
    rig = Rig(ros, models, 'h12', follow_tolerance=0.0)
    handler = rig.handler()
    with pytest.raises(RuntimeError, match='start'):
        command(rig, handler, rig.measured)
    handler.start()
    good = rig.measured.copy()
    good[rig.arms[0]] = 0.2
    command(rig, handler, good)
    before = handler.q_cmd
    for field in ('q', 'dq', 'tau'):
        bad = {key: np.zeros(len(good)) for key in ('q', 'dq', 'tau')}
        bad['q'] = good.copy()
        bad[field][rig.arms[1]] = np.nan
        with pytest.raises(ValueError, match=field):
            handler.set_joint_commands(**bad)
    np.testing.assert_array_equal(handler.q_cmd, before)
    # the last good command keeps going out
    start = time.monotonic()
    time.sleep(0.2)
    sent = rig.recorder.since(start)
    assert len(sent) > 50
    assert all(np.isfinite(upper_q(msg, rig.profile)).all() for msg in sent)
    assert sent[-1].motor_cmd[rig.arms[0]].q == pytest.approx(0.2, abs=FLOAT32)


def test_start_refuses_a_non_finite_pose(ros, models):
    """start() with a non-finite pose is refused and publishes nothing."""
    rig = Rig(ros, models, 'h12', follow_tolerance=0.0)
    handler = rig.handler()
    q = rig.measured.copy()
    q[rig.arms[2]] = np.nan
    with pytest.raises(ValueError, match='finite'):
        handler.start(q)
    time.sleep(0.2)
    assert rig.recorder.count == 0


# (robot, controller config, whether the shutdown keeps the held joints)
SHUTDOWNS = [('h12', 'safety_split', False), ('h12', 'safety_full', False),
             ('g1', 'g1_split', True)]


@pytest.mark.parametrize('robot, config, keeps_held', SHUTDOWNS,
                         ids=[f'{r}-{c}' for r, c, _ in SHUTDOWNS])
def test_shutdown_command_then_silence(ros, models, robot, config, keeps_held):
    """The shutdown command goes out once, then nothing more.

    The IK joints go limp: mode 0, kp, kd and tau 0, q as they were. The held joints too,
    except on the G1, where a walking policy may be on the legs: there they keep their hold
    angle, gains and torque.
    """
    rig = Rig(ros, models, robot, config=config, follow_tolerance=0.0)
    handler = rig.handler()
    handler.start()
    command(rig, handler, rig.measured)
    tau = 1.5  # N*m on every joint, gravity feed-forward as the controller sends it
    handler.set_joint_commands(tau=np.full(len(rig.profile.body_joints), tau))
    assert wait_for(lambda: rig.recorder.last is not None
                    and rig.recorder.last.motor_cmd[rig.arms[0]].tau == np.float32(tau), 2.0)
    held_q = handler.hold
    handler.shutdown()
    assert wait_for(lambda: rig.recorder.last.motor_cmd[rig.arms[0]].mode == 0, 2.0)
    last = rig.recorder.last
    kp, kd = rig.config['gains']['kp'], rig.config['gains']['kd']
    for slot in rig.arms:
        motor = last.motor_cmd[slot]
        assert (motor.mode, motor.kp, motor.kd, motor.tau) == (0, 0.0, 0.0, 0.0), slot
        assert motor.q == pytest.approx(rig.measured[slot], abs=FLOAT32)
    for k, slot in enumerate(rig.held):
        motor = last.motor_cmd[slot]
        assert motor.q == pytest.approx(held_q[k], abs=FLOAT32)
        if keeps_held:
            assert (motor.mode, motor.kp, motor.kd) == (1, np.float32(kp[slot]),
                                                         np.float32(kd[slot])), slot
            assert motor.tau == np.float32(tau)
        else:
            assert (motor.mode, motor.kp, motor.kd, motor.tau) == (0, 0.0, 0.0, 0.0), slot
    count = rig.recorder.count
    time.sleep(0.2)
    assert rig.recorder.count == count
    handler.shutdown()  # again: nothing more


@pytest.mark.parametrize('robot', ROBOTS)
def test_set_hold_while_the_gate_is_shut(ros, models, robot):
    """While the gate is shut a hold stays within the tolerance of the measured angle."""
    rig = Rig(ros, models, robot)
    tolerance = rig.config['controller']['follow_tolerance']
    handler = rig.handler()
    handler.start()
    np.testing.assert_allclose(handler.hold, SAG, atol=FLOAT32)
    # a hold 0.3 rad away goes no further than the tolerance from the measured angle, and
    # the hold restarts from there
    for _ in range(3):
        handler.set_hold(np.full(len(rig.held), SAG + 0.3))
        np.testing.assert_allclose(handler.hold, SAG + tolerance, atol=FLOAT32)
        np.testing.assert_allclose(handler.q_cmd[rig.held], SAG + tolerance, atol=FLOAT32)
    assert not handler.following
    rig.echo()
    assert wait_for(lambda: handler.following, 3.0)
    handler.set_hold(np.full(len(rig.held), SAG + 0.3))
    np.testing.assert_allclose(handler.hold, SAG + 0.3, atol=FLOAT32)
    np.testing.assert_allclose(handler.q_cmd[rig.held], SAG + 0.3, atol=FLOAT32)


def test_zero_gain_output_keeps_the_gate_shut(ros, models):
    """A relay with kp 0 (golem_safety before it engages) never opens the gate."""
    rig = Rig(ros, models, 'h12')
    handler = rig.handler()
    handler.start()
    # golem_safety before it engages the input: our q, but kp 0 on every slot
    echo = rig.echo(zero_kp=True)
    start = time.monotonic()
    while time.monotonic() - start < 0.5:
        command(rig, handler, rig.measured + 0.3)
        time.sleep(1.0 / 30.0)
    assert not handler.following
    echo.zero_kp = False
    assert wait_for(lambda: handler.following, 3.0)


def test_g1_split_gate_opens_on_lowcmd(ros, models):
    """The G1 split config's gate opens on its relayed commands on /lowcmd."""
    rig = Rig(ros, models, 'g1', config='g1_split')
    handler = rig.handler()
    handler.start()
    time.sleep(0.3)
    assert not handler.following
    rig.echo()
    assert wait_for(lambda: handler.following, 3.0)


def test_gated_command_stays_within_the_clip_limits(ros, models):
    """A gated command near a joint past its clip limit stays clipped, and still opens the gate."""
    rig = Rig(ros, models, 'h12')
    limits = rig.config['limits']['q_clip_limits']
    elbow = rig.profile.body_joints.index('left_elbow_joint')
    # measured 0.15 rad past its clip limit, three tolerances
    rig.robot.q[elbow] = limits[elbow, 1] + 0.15
    rig.measured = rig.robot.q.copy()
    assert wait_for(lambda: rig.subscriber.state['q'][elbow] > limits[elbow, 1] + 0.1, 2.0)
    handler = rig.handler()
    handler.start()
    start = time.monotonic()
    for _ in range(10):
        command(rig, handler, rig.measured)
        time.sleep(1.0 / 30.0)
    assert handler.q_cmd[elbow] == pytest.approx(limits[elbow, 1], abs=FLOAT32)
    for msg in rig.recorder.since(start):
        q = upper_q(msg, rig.profile)
        assert np.all(q >= limits[rig.upper, 0] - FLOAT32)
        assert np.all(q <= limits[rig.upper, 1] + FLOAT32)
    # golem_safety clips what it relays: the relayed command is the stored one, so the gate
    # opens
    rig.echo(clip=(limits[:, 0], limits[:, 1]))
    assert wait_for(lambda: handler.following, 3.0)


def test_start_needs_a_measured_state(ros, models):
    """start() without a state source, or before its first sample, raises and publishes nothing."""
    from golem_upper_ik.config import load_controller_config
    from golem_upper_ik.core.low_cmd_handler import LowCmdHandler
    from golem_upper_ik.core.ros_io import StaleStateError, StateSubscriber
    profile = profile_of('h12')
    config = load_controller_config('h12', CONFIGS['h12'])
    model = models('h12')
    node = ros.node('controller')
    recorder = Recorder(ros, config['topics']['low_cmd'])
    handler = LowCmdHandler(node, profile, config, model)
    ros.on_close(handler.shutdown)
    # a state source that has no sample yet: start() refuses and nothing is published
    model.state_subscriber = StateSubscriber(node, profile, topic='/lowstate_unpublished')
    ros.on_close(lambda: setattr(model, 'state_subscriber', None))
    with pytest.raises(StaleStateError, match='no sample'):
        handler.start()
    # an initial pose given does not help: the gate has no measured pose to hold it near
    q_init = np.zeros(len(profile.body_joints))
    with pytest.raises(StaleStateError, match='no sample'):
        handler.start(q_init=q_init)
    # and none at all
    model.state_subscriber = None
    with pytest.raises(StaleStateError, match='no state source'):
        handler.start()
    with pytest.raises(StaleStateError, match='no state source'):
        handler.start(q_init=q_init)
    time.sleep(0.2)
    assert recorder.count == 0


@pytest.mark.parametrize('robot', ROBOTS)
def test_start_holds_the_initial_pose_near_the_measured_one(ros, models, robot):
    """start()'s initial pose is gated like any command, and the hold starts from it."""
    rig = Rig(ros, models, robot)
    handler = rig.handler()
    tolerance = rig.config['controller']['follow_tolerance']
    q_init = rig.measured.copy()
    q_init[rig.upper] += 0.3
    handler.start(q_init=q_init)
    assert not handler.following
    # the first command already leads the measured pose by the tolerance at most, and the
    # hold starts from it
    assert wait_for(lambda: rig.recorder.count > 0, 2.0)
    first = upper_q(rig.recorder.samples[0][1], rig.profile)
    np.testing.assert_allclose(first, rig.measured[rig.upper] + tolerance, atol=FLOAT32)
    np.testing.assert_allclose(handler.hold, rig.measured[rig.held] + tolerance, atol=FLOAT32)


def test_detached_state_source_leaves_the_last_command(ros, models):
    """A detached state source stops the gate check, leaving the last command; gating raises."""
    from golem_upper_ik.core.ros_io import StaleStateError
    rig = Rig(ros, models, 'h12')
    handler = rig.handler()
    handler.start()
    target = rig.measured.copy()
    target[rig.arms] += 0.3
    for _ in range(5):
        command(rig, handler, target)
        time.sleep(1.0 / 30.0)
    assert not handler.following
    last = handler.q_cmd
    # the arms are commanded the tolerance past the measured 0.1 rad, away from 0
    assert np.all(last[rig.arms] > ARM_Q)
    assert wait_for(lambda: np.array_equal(upper_q(rig.recorder.last, rig.profile),
                                           last[rig.upper]), 2.0)

    # the controller's shutdown in the wrong order: the state source goes first, and the
    # model falls back to its all-zero rest pose
    rig.model.shutdown()
    assert rig.model.state_subscriber is None
    start = time.monotonic()
    time.sleep(0.3)
    sent = rig.recorder.since(start)
    assert len(sent) > 100
    # no command moved toward q = 0: every one is the last command
    for msg in sent:
        np.testing.assert_array_equal(upper_q(msg, rig.profile), last[rig.upper])
    np.testing.assert_array_equal(handler.q_cmd, last)
    # nothing is gated against a pose the robot does not report
    with pytest.raises(RuntimeError, match='no measured state'):
        command(rig, handler, target)
    with pytest.raises(RuntimeError, match='no measured state'):
        handler.set_hold(np.zeros(len(rig.held)))
    np.testing.assert_array_equal(handler.q_cmd, last)
    assert issubclass(StaleStateError, RuntimeError)


def _creep(gate, measured, target, steps, step=0.01):
    """Step an IK command toward target from the gated command; return it and the gaps."""
    q_cmd = np.asarray(measured(0, None), dtype=float).copy()
    gaps = []
    for k in range(steps):
        m = np.asarray(measured(k, q_cmd), dtype=float)
        q_cmd = gate.limit(q_cmd + np.clip(target - q_cmd, -step, step), m)
        gaps.append(np.max(np.abs(q_cmd[:2] - m[:2])))
    return q_cmd, gaps


def test_follow_gate_keeps_commands_near_a_robot_that_does_not_follow():
    """A shut gate keeps commands within tolerance of a robot at rest, swaying or pushed."""
    from golem_upper_ik.core.follow_gate import FollowGate
    # slot 0 commanded 0.8 rad away, slot 1 at its target; slot 2 is not gated
    target = np.array([1.0, 0.1, 1.0])
    gate = FollowGate([0, 1], 0.05)
    # at rest: the command waits 0.05 rad ahead of the robot
    q_cmd, gaps = _creep(gate, lambda k, q: [0.2, 0.1, 0.0], target, 200)
    assert max(gaps) <= 0.05 + 1e-12 and not gate.following
    assert np.allclose(q_cmd, [0.25, 0.1, 1.0])

    # swaying 0.1 rad: the command is dragged along
    def sway(k, q):
        """Return a pose swaying 0.1 rad on slot 0 at step k."""
        return [0.2 + 0.1 * np.sin(k / 10.0), 0.1, 0.0]
    _, gaps = _creep(gate, sway, target, 300)
    assert max(gaps) <= 0.05 + 1e-12 and not gate.following

    # pushed 0.15 rad away from the target, let go, falling back: still shut
    path = np.concatenate([np.linspace(0.0, -0.15, 10), np.linspace(-0.15, 0.0, 5),
                           np.zeros(100)])
    gate = FollowGate([0, 1], 0.05)
    _, gaps = _creep(gate, lambda k, q: [path[k], 0.1, 0.0], target, len(path))
    assert max(gaps) <= 0.05 + 1e-12 and not gate.following


def test_follow_gate_opens_only_when_told():
    """Following motion does not open the gate; once open it never moves a command."""
    from golem_upper_ik.core.follow_gate import FollowGate
    target = np.array([1.0, -1.0, 0.0])
    gate = FollowGate([0, 1], 0.05)
    lagged = {'q': np.zeros(3)}

    def follows(k, q_cmd):
        # the robot reaches each command one step later
        """Return the measured pose, which reaches each command one step later."""
        measured = lagged['q'].copy()
        if q_cmd is not None:
            lagged['q'] = q_cmd.copy()
        return measured
    # a robot keeping up over 1 rad of travel does not open it: the motion is no evidence
    # that the safety layer relays the commands
    q_cmd, gaps = _creep(gate, follows, target, 150)
    assert max(gaps) <= 0.05 + 1e-12 and not gate.following
    assert np.allclose(q_cmd[:2], target[:2])

    # once open (the safety layer relays the commands), a held command never follows the
    # measured pose: a pushed or tipping joint is resisted
    gate.open()
    assert gate.following
    for measured in ([1.3, -1.2, 0.0], [0.0, 0.0, 0.0], [np.nan, 5.0, 0.0]):
        assert np.array_equal(gate.limit(target, measured), target)


def test_relays_commands_needs_every_joint_close_with_kp():
    """A relay counts only with every gated joint close and kp > 0."""
    from golem_upper_ik.core.follow_gate import relays_commands
    ids = [12, 13, 14, 15]
    q_cmd = np.zeros(29)
    q_cmd[ids] = [0.0, 0.1, -0.1, 0.8]
    sample = q_cmd.copy()
    sample[ids] += [0.01, -0.02, 0.0, 0.04]
    sample[:12] = 0.7  # the legs (the walking node's in split mode) are not compared
    kp = np.zeros(29)
    kp[ids] = [200.0, 200.0, 200.0, 60.0]
    assert relays_commands(sample, kp, q_cmd, ids, 0.05)
    # one joint off: someone else's command, or not ours yet
    off = sample.copy()
    off[15] = 0.7
    assert not relays_commands(off, kp, q_cmd, ids, 0.05)
    off[15] = np.nan
    assert not relays_commands(off, kp, q_cmd, ids, 0.05)
    # kp 0 on a slot: the safety layer's estop command (q 0, kp 0) or a default LowCmd, even
    # where our commands are near 0
    q_zero = np.zeros(29)
    for zero_kp in (np.zeros(29), np.where(np.arange(29) == 15, 0.0, kp)):
        assert not relays_commands(np.zeros(29), zero_kp, q_zero, ids, 0.05)
    assert relays_commands(np.zeros(29), kp, q_zero, ids, 0.05)


def test_follow_gate_tolerance_zero_is_off():
    """Tolerance 0 leaves the gate open; a negative one is refused."""
    from golem_upper_ik.core.follow_gate import FollowGate
    gate = FollowGate([0, 1], 0.0)
    assert gate.following
    assert np.array_equal(gate.limit([1.0, 2.0], [0.0, 0.0]), [1.0, 2.0])
    with pytest.raises(ValueError):
        FollowGate([0], -0.1)
