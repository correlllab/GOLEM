'''
The controller's closed loop over ROS on domain 92, on both robots, against kinematic fake
robots in this process, each controller sharing the session's robot model.

A FollowingRobot executes every command at once and relays it on golem_safety's output, as
an engaged golem_safety would: the start-up, the held joints' hold, an end-effector target,
the zero-gain shutdown, and a start-up move cut short by controller.startup_move_timeout.
Without /lowstate the controller commands nothing and stops with an error. With nothing
relaying its commands, they stay near the swaying robot, and the start-up stops with an
error at controller.startup_timeout, well before controller.startup_move_timeout.

The others sit behind golem_safety's safety_node, run as a subprocess (RelayedRobot). The
controller waits for /lowstate and, unless follow_tolerance turns the gate off, never
commands far from the measured pose before and at engagement: on the G1 until its engage
gate passes, on the H1-2 until its zero stream gives way to the upper input. The G1's gate
opens only once the walking node commands too, the H1-2's without it; when the relay starts
late, the start-up moves still finish, though the start-up outlasts
controller.startup_move_timeout. Once engaged, the controller holds against a push instead
of following it.
'''
import re
import threading
import time

import numpy as np
import pytest

from ros_harness import (
    CommandRecorder, FollowingRobot, RelayedRobot, SafetyNode, WalkingNode, body_pose,
    profile_of, shared_model, slots_of, wait_for,
)

ROBOTS = ('h12', 'g1')
# the controller config, and the golem_safety config it pairs with, of each test setting
CONFIGS = {'h12': 'safety_split', 'g1': 'g1_split'}
SPLIT = {'h12': ('safety_split', 'relax_safety_split'), 'g1': ('g1_split', 'split')}
LATE = {'h12': ('safety_split', 'relax_safety_split'), 'g1': ('sim_split', 'sim_split')}
# the ROS delivery, and the robot's motion since the controller's sample (rad)
SLACK = 0.025
# a robot standing under its own controller: arms hanging forward, elbows bent, the first
# held joint turned a little (one value per IK joint, then per held joint)
STANDING = {
    'h12': ([0.35, 0.175, 0.0, 0.65, 0.0, 0.0, 0.0, 0.35, -0.175, 0.0, 0.65, 0.0, 0.0, 0.0],
            [0.04]),
    'g1': ([0.3, 0.25, 0.0, 0.8, 0.0, 0.0, 0.0, 0.3, -0.25, 0.0, 0.8, 0.0, 0.0, 0.0],
           [0.04, 0.0, 0.0]),
}
# where a push moves joints past the follow tolerance once engaged, rad
PUSH = {
    'h12': {'torso_joint': 0.2, 'left_elbow_joint': 0.15},
    'g1': {'waist_roll_joint': 0.2, 'waist_pitch_joint': -0.1, 'left_elbow_joint': 0.15},
}


def config_of(robot, name, **controller):
    from golem_upper_ik.config import load_controller_config
    config = load_controller_config(robot, name)
    config['controller'].update(controller)
    return config


def build(ros, model, config, init=True):
    from golem_upper_ik.core.controller.frame_controller import FrameController
    controller = FrameController(ros.node('controller'), model.profile, config,
                                 robot_model=model, init=init)
    ros.on_close(controller.shutdown)
    return controller


def build_in_thread(ros, model, config):
    '''
    Start building a controller (init=True) in a thread; the thread and its result dict. At
    the test's end the build is waited for and the controller shut down
    '''
    built = {}
    node = ros.node('controller')

    def run():
        from golem_upper_ik.core.controller.frame_controller import FrameController
        try:
            built['controller'] = FrameController(node, model.profile, config,
                                                  robot_model=model)
        except BaseException as error:  # reported by the test's thread
            built['error'] = error
    builder = threading.Thread(target=run, daemon=True)

    def close():
        builder.join(45.0)
        if 'controller' in built:
            built['controller'].shutdown()
    ros.on_close(close)
    builder.start()
    return builder, built


def watch_gate_opening(monkeypatch):
    '''The time of each golem_safety output sample that opens a follow gate'''
    from golem_upper_ik.core import low_cmd_handler
    opened = []
    relays_commands = low_cmd_handler.relays_commands

    def relays_and_note(*args):
        relays = relays_commands(*args)
        if relays:
            opened.append(time.monotonic())
        return relays
    monkeypatch.setattr(low_cmd_handler, 'relays_commands', relays_and_note)
    return opened


def watch_startup(monkeypatch):
    '''The start-up routine's start and end times, and an event set once it starts'''
    from golem_upper_ik.core.controller.upper_controller import UpperController
    times = {'started': threading.Event()}
    startup_routine = UpperController._startup_routine

    def startup_and_time(self, *args, **kwargs):
        times['start'] = time.monotonic()
        times['started'].set()
        try:
            startup_routine(self, *args, **kwargs)
        finally:
            times['end'] = time.monotonic()
    monkeypatch.setattr(UpperController, '_startup_routine', startup_and_time)
    return times


def final_pose(profile, config):
    '''The upper-slot q the start-up ends at: the last start-up posture, the hold targets'''
    targets = config['controller']['hold_targets']
    q = body_pose(profile, profile.startup_sequence[-1],
                  [targets[joint] for joint in profile.held_joints])
    return q[list(profile.upper_slots)]


def settles(robot, upper_q, tolerance=0.01, timeout=5.0):
    '''True once the robot's upper-slot q is within tolerance of upper_q'''
    upper = list(robot.profile.upper_slots)
    return wait_for(lambda: np.max(np.abs(robot.measured()[upper] - upper_q)) < tolerance,
                    timeout, period=0.02)


def ee_position(controller, side):
    return controller.ee_transformations()[side][:3, 3].copy()


def push_and_check_the_hold(controller, robot, recorders):
    '''
    Once engaged, push held and arm joints well past the follow tolerance: every command
    stays where it was, and the joints move by the push and no further
    '''
    profile = robot.profile
    upper = list(profile.upper_slots)
    push = np.zeros(len(profile.body_joints))
    for joint, rad in PUSH[profile.robot].items():
        push[profile.body_joints.index(joint)] = rad
    held = controller.low_cmd_handler.q_cmd[upper].astype(np.float64)
    # the held joints hold their targets themselves (as float32 on the wire)
    held_in_upper = [upper.index(slot) for slot in slots_of(profile, profile.held_joints)]
    np.testing.assert_allclose(held[held_in_upper], controller.low_cmd_handler.hold_targets,
                               atol=1e-6)
    # every recorder has caught up with the held command
    assert wait_for(lambda: all(np.array_equal(r.samples[-1].q, held) for r in recorders), 2.0)
    start = time.monotonic()
    robot.set_push(push)
    # the joints move by the push and no further, and stay there
    assert settles(robot, held + push[upper])
    time.sleep(1.0)
    pushed = robot.measured()
    robot.set_push(np.zeros(len(profile.body_joints)))
    end = time.monotonic()
    assert np.max(np.abs(pushed[upper] - (held + push[upper]))) < 0.01
    # and back to the hold once let go
    assert settles(robot, held)
    for recorder in recorders:
        during = np.array([s.q for s in recorder.samples if start <= s.t <= end])
        assert len(during) > 100
        assert np.max(np.abs(during - held)) < 1e-6, 'a held command followed the push'


@pytest.mark.parametrize('robot', ROBOTS)
def test_startup_home_ee_target_and_zero_gain_shutdown(ros, models, robot):
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot])
    upper = list(profile.upper_slots)
    held = slots_of(profile, profile.held_joints)
    # a standing robot: arms hanging, the first held joint turned a little
    arms = 'arms_down' if robot == 'g1' else 'home'
    q0 = body_pose(profile, arms, [0.05] + [0.0] * (len(held) - 1))
    model = shared_model(ros, models, robot)
    robot_ = FollowingRobot(ros, profile, q0, config['topics']['safety_out'])
    controller = build(ros, model, config)

    # start-up: the arms through the start-up postures to the last, the held joints eased to
    # their targets, never a zero-gain command
    np.testing.assert_allclose(robot_.measured()[upper], final_pose(profile, config), atol=0.02)
    assert robot_.commands > 0 and robot_.zero_gain_commands == 0
    # the commands were relayed: the gate is open
    assert controller.low_cmd_handler.following

    # a target 5 cm forward of the left end effector is reached; the right one stays put
    controller.update_ik_solver()
    left = controller.ee_transformations()[0]
    right_start = ee_position(controller, 1)
    target = left.copy()
    target[0, 3] += 0.05
    controller.add_frame_task('left_task', profile.ee_frames[0], target)
    for _ in range(150):
        controller.control_step_reduced()
        if np.linalg.norm(ee_position(controller, 0) - target[:3, 3]) < 0.002:
            break
        time.sleep(controller.dt)
    assert np.linalg.norm(ee_position(controller, 0) - target[:3, 3]) < 0.005
    assert np.linalg.norm(ee_position(controller, 1) - right_start) < 0.005

    # shutdown: one command, then silence. Mode 0 and zero gains and torque on the upper
    # slots, except the held joints of a profile that keeps them on /lowcmd (the G1's waist),
    # which stay at their hold
    controller.shutdown()
    stopped = time.monotonic()
    time.sleep(0.3)
    received, last = robot_.last_command
    assert received < stopped + 0.05
    keeps_held = profile.shutdown_keeps_held_joints_on_lowcmd
    if not keeps_held:
        assert robot_.zero_gain_commands >= 1
    for slot in upper:
        motor = last.motor_cmd[slot]
        if keeps_held and slot in held:
            assert motor.mode == 1 and motor.kp > 0.0, slot
        else:
            assert (motor.mode, motor.kp, motor.kd, motor.tau) == (0, 0.0, 0.0, 0.0), slot


@pytest.mark.parametrize('raising', ['handler', 'planner'])
def test_shutdown_stops_the_planner_and_detaches_the_state_whatever_raises(raising):
    '''
    A command handler or planner whose shutdown raises leaves nothing behind: the planner
    subprocess is still stopped and the state source detached, and the error propagates
    '''
    from golem_upper_ik.core.controller.upper_controller import UpperController
    calls = []

    def part(name):
        def shutdown():
            calls.append(name)
            if name == raising:
                raise RuntimeError(f'the {name} failed')
        return type(name, (), {'shutdown': staticmethod(shutdown)})()

    controller = UpperController.__new__(UpperController)
    controller.low_cmd_handler = part('handler')
    controller.planner = part('planner')
    controller.robot_model = part('state')
    with pytest.raises(RuntimeError, match=f'the {raising} failed'):
        controller.shutdown()
    assert calls == ['handler', 'planner', 'state']


@pytest.mark.parametrize('robot', ROBOTS)
def test_no_lowstate_means_no_command_and_a_clear_error(ros, models, robot):
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot], state_timeout=0.5)
    recorder = CommandRecorder(ros, config['topics']['low_cmd'], profile)
    model = shared_model(ros, models, robot)
    with pytest.raises(RuntimeError, match=r'No fresh /lowstate with finite joint positions '
                                           r'within 0.5 s .*not enabling any motor'):
        build(ros, model, config)
    time.sleep(0.2)
    assert recorder.count == 0, 'nothing may be commanded without /lowstate'
    assert model.state_subscriber is None


@pytest.mark.parametrize('robot', ROBOTS)
def test_startup_stops_with_an_error_when_the_safety_layer_never_engages(
        ros, models, robot, monkeypatch):
    from golem_upper_ik.core.low_cmd_handler import LowCmdHandler
    from golem_upper_ik.core.robot_model import RobotModel
    # the publisher stops before the state source goes: the robot model's all-zero rest
    # pose must never reach the follow gate while commands go out
    order = []
    for cls, name in ((LowCmdHandler, 'publisher'), (RobotModel, 'state')):
        def shutdown_and_note(self, _shutdown=cls.shutdown, _name=name):
            order.append(_name)
            _shutdown(self)
        monkeypatch.setattr(cls, 'shutdown', shutdown_and_note)
    profile = profile_of(robot)
    # nothing relays the commands: the gate stays shut, the commands stay near the swaying
    # robot, and the start-up gives up after its own 2 s wait, not after the moves' 10 s
    config = config_of(robot, CONFIGS[robot], startup_timeout=2.0, startup_move_timeout=10.0)
    safety_out = config['topics']['safety_out']
    q0 = np.zeros(len(profile.body_joints))
    q0[slots_of(profile, ('left_elbow_joint', 'right_elbow_joint'))] = 0.8
    robot_ = RelayedRobot(ros, profile, q0, safety_out)
    robot_.start()
    recorder = CommandRecorder(ros, config['topics']['low_cmd'], profile, robot_)
    startup = watch_startup(monkeypatch)
    model = shared_model(ros, models, robot)
    with pytest.raises(RuntimeError, match=re.escape(
            'golem_safety did not relay the commands within controller.startup_timeout '
            f'(2 s): no {safety_out} sample carried them')):
        build(ros, model, config)
    stopped = time.monotonic()
    time.sleep(0.3)
    assert 2.0 <= startup['end'] - startup['start'] < 3.0
    tolerance = config['controller']['follow_tolerance']
    samples = recorder.samples
    assert len(samples) > 100
    assert max(np.max(np.abs(s.q - s.measured)) for s in samples) <= tolerance + SLACK
    assert all(s.t < stopped + 0.05 for s in samples), 'still publishing'
    assert order == ['publisher', 'state']


@pytest.mark.parametrize('robot', ROBOTS)
def test_startup_move_stops_with_an_error_after_its_own_timeout(ros, models, robot):
    profile = profile_of(robot)
    upper = list(profile.upper_slots)
    # relayed at once, so the gate opens at the first command. The G1 moves its elbows pi/2
    # from hanging to home (at least 5.2 s at 0.3 rad/s); the H1-2 starts at home and moves
    # through its start-up postures back to it (at least 4 s at 1 rad/s)
    q0 = body_pose(profile, 'arms_down' if robot == 'g1' else 'home')
    config = config_of(robot, CONFIGS[robot], startup_move_timeout=2.0)
    robot_ = FollowingRobot(ros, profile, q0, config['topics']['safety_out'])
    recorder = CommandRecorder(ros, config['topics']['low_cmd'], profile)
    model = shared_model(ros, models, robot)
    with pytest.raises(RuntimeError, match=r'Moving \w+: not done within '
                                           r'controller.startup_move_timeout \(2 s\) of the '
                                           r'follow gate opening'):
        build(ros, model, config)
    stopped = time.monotonic()
    time.sleep(0.3)
    # stopped mid-move, and silent since
    assert np.max(np.abs(robot_.measured()[upper] - final_pose(profile, config))) > 0.1
    if robot == 'g1':
        elbow = robot_.measured()[profile.body_joints.index('left_elbow_joint')]
        assert 0.1 < elbow < profile.named_configs['arms_down'][3] - 0.1, elbow
    assert recorder.count > 50
    assert all(s.t < stopped + 0.05 for s in recorder.samples), 'still publishing'


@pytest.mark.parametrize('robot', ROBOTS)
def test_every_command_holds_the_hold_targets_while_the_held_joints_sag(ros, models, robot):
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot])
    upper = list(profile.upper_slots)
    held = slots_of(profile, profile.held_joints)
    held_in_upper = [upper.index(slot) for slot in held]
    targets = [config['controller']['hold_targets'][joint] for joint in profile.held_joints]
    # the last held joint sags 0.02 rad below every command; each goal re-seeds the IK from
    # the measured pose, which must not drag the held joint's command down
    sag = np.zeros(len(profile.body_joints))
    sag[held[-1]] = 0.02
    q0 = body_pose(profile, 'home', targets)
    robot_ = FollowingRobot(ros, profile, q0, config['topics']['safety_out'], sag=sag)
    model = shared_model(ros, models, robot)
    controller = build(ros, model, config, init=False)
    recorder = CommandRecorder(ros, config['topics']['low_cmd'], profile)
    for _ in range(3):
        controller.clear_frame_tasks()
        controller.update_ik_solver()
        target = controller.ee_transformations()[0]
        target[0, 3] += 0.02
        controller.add_frame_task('left_task', profile.ee_frames[0], target)
        for _ in range(30):
            controller.control_step_reduced()
            time.sleep(controller.dt)
    commanded = np.array([s.q[held_in_upper] for s in recorder.samples])
    assert len(commanded) > 100
    assert np.max(np.abs(commanded - targets)) < 1e-6
    # the measured held joint sags once, it does not drift goal after goal
    assert abs(robot_.measured()[held[-1]] - (targets[-1] - 0.02)) < 1e-3


@pytest.mark.parametrize('robot, follow_tolerance', [
    ('h12', 0.05), ('h12', 0.0), ('g1', 0.05), ('g1', 0.0)],
    ids=['h12-gate', 'h12-no_gate', 'g1-gate', 'g1-no_gate'])
def test_waits_for_state_and_tracks_the_robot_through_engagement(
        ros, models, robot, follow_tolerance, monkeypatch, tmp_path):
    profile = profile_of(robot)
    controller_config, safety_config = SPLIT[robot]
    config = config_of(robot, controller_config, follow_tolerance=follow_tolerance)
    safety_out = config['topics']['safety_out']
    arms, held = STANDING[robot]
    q0 = body_pose(profile, arms, held)
    robot_ = RelayedRobot(ros, profile, q0, safety_out)
    upper_in = CommandRecorder(ros, config['topics']['low_cmd'], profile, robot_)
    output = CommandRecorder(ros, safety_out, profile, robot_)
    layer = SafetyNode(robot, safety_config, tmp_path)
    ros.on_close(layer.stop)
    # the G1's engage gate waits for the legs too
    legs = WalkingNode(ros, profile, q0) if robot == 'g1' else None
    opened = watch_gate_opening(monkeypatch)
    model = shared_model(ros, models, robot)
    probe = ros.node('probe')

    builder, built = build_in_thread(ros, model, config)
    # know when the controller listens, so /lowstate starts only after that
    assert wait_for(lambda: probe.count_subscribers('/lowstate') > 0, 30.0)
    time.sleep(0.5)
    assert upper_in.count == 0, 'nothing may be commanded before /lowstate'
    robot_.start()
    # nothing relays the commands yet: the robot does not follow for 2 s
    time.sleep(2.0)
    layer.start()
    if legs is not None:
        legs.start()
    # before golem_safety runs, from the controller's commands alone
    before = [s for s in upper_in.samples if s.t < layer.started]
    assert before[0].t >= robot_.started
    assert len(before) > 30
    gap_before = max(np.max(np.abs(s.q - s.measured)) for s in before)
    if follow_tolerance == 0.0:
        # without the gate the start-up runs open loop, far from the robot that is not
        # following: the jump the gate prevents
        assert gap_before > 0.3
        return

    builder.join(45.0)
    assert not builder.is_alive() and 'error' not in built, built.get('error')
    controller = built['controller']
    bound = follow_tolerance + SLACK
    # the measured pose swayed more than a command standing still could cover
    sway = [list(profile.upper_slots).index(slot) for slot in robot_.sway_slots]
    assert np.all(np.ptp(np.array([s.measured[sway] for s in before]), axis=0) > 2 * bound)
    assert gap_before <= bound
    # until golem_safety relays the upper input (the G1's engage gate passes, the H1-2's zero
    # stream gives way to it), and with no jump at engagement
    relaying = [s for s in output.samples if s.kp > 0.0]
    through = [s for s in upper_in.samples if s.t <= relaying[0].t]
    assert max(np.max(np.abs(s.q - s.measured)) for s in through) <= bound
    assert np.max(np.abs(relaying[0].q - relaying[0].measured)) <= bound, 'jump at engagement'

    # the gate opened on one golem_safety output sample
    assert controller.low_cmd_handler.following
    assert len(opened) == 1
    # then the start-up moves: the arms to their last start-up posture, the held joints at
    # their targets
    assert settles(robot_, final_pose(profile, config), 0.02)
    assert not layer.estopped(), layer.output()

    # engaged: the hold resists a push instead of following it
    push_and_check_the_hold(controller, robot_, [upper_in, output])


@pytest.mark.parametrize('robot', ROBOTS)
def test_split_mode_engages_then_holds_against_a_push(ros, models, robot, monkeypatch,
                                                      tmp_path):
    profile = profile_of(robot)
    controller_config, safety_config = SPLIT[robot]
    config = config_of(robot, controller_config)
    # hanging on the gantry at the start-up's end pose: on the G1 the start-up has nothing
    # to move, and only golem_safety's /lowcmd can tell the controller it engaged
    q0 = np.zeros(len(profile.body_joints))
    q0[list(profile.upper_slots)] = final_pose(profile, config)
    robot_ = RelayedRobot(ros, profile, q0, '/lowcmd', sway=0.0)
    robot_.start()
    upper_in = CommandRecorder(ros, config['topics']['low_cmd'], profile, robot_)
    lowcmd = CommandRecorder(ros, '/lowcmd', profile, robot_)
    layer = SafetyNode(robot, safety_config, tmp_path)
    ros.on_close(layer.stop)
    legs = WalkingNode(ros, profile, q0)
    opened = watch_gate_opening(monkeypatch)
    model = shared_model(ros, models, robot)

    layer.start()
    builder, built = build_in_thread(ros, model, config)
    assert wait_for(lambda: upper_in.count >= 100, 30.0)
    if robot == 'g1':
        # the upper input alone: the G1's engage gate waits for the legs, silent, the gate shut
        time.sleep(1.0)
        assert upper_in.count > 100 and lowcmd.count == 0 and opened == []
        assert builder.is_alive(), 'the start-up must wait for golem_safety'
        legs.start()
    else:
        # the H1-2's zero stream engages each input on its own: the arms without the legs
        assert wait_for(lambda: opened, 5.0)
    builder.join(30.0)
    assert not builder.is_alive() and 'error' not in built, built.get('error')
    controller = built['controller']
    first = next(s for s in lowcmd.samples if s.kp > 0.0)
    # opened on one /lowcmd sample
    assert len(opened) == 1
    assert controller.low_cmd_handler.following
    assert np.max(np.abs(first.q - first.measured)) <= config['controller']['follow_tolerance']
    assert settles(robot_, final_pose(profile, config), 0.02)

    # engaged at its pose, the held joints pushed past the tolerance must not drag their
    # commands along
    push_and_check_the_hold(controller, robot_, [upper_in, lowcmd])
    assert not layer.estopped(), layer.output()


def test_start_up_steps_sleep_a_whole_period_after_their_work(ros, models, monkeypatch):
    from golem_upper_ik.core.controller.upper_controller import UpperController
    from golem_upper_ik.core.low_cmd_handler import LowCmdHandler
    profile = profile_of('g1')
    config = config_of('g1', 'sim_split')
    # every step of both start-up moves takes 10 ms of work more
    steps = {'posture': [], 'hold': []}
    for cls, name, kind in ((UpperController, 'goto_reduced_configuration', 'posture'),
                            (LowCmdHandler, 'set_hold', 'hold')):
        def slow(self, *args, _call=getattr(cls, name), _kind=kind, **kwargs):
            steps[_kind].append(time.monotonic())
            time.sleep(0.01)
            return _call(self, *args, **kwargs)
        monkeypatch.setattr(cls, name, slow)
    # the elbows 0.5 rad from home and the waist yaw 0.1 rad from its target
    q0 = body_pose(profile, 'home', [0.1, 0.0, 0.0])
    q0[slots_of(profile, ('left_elbow_joint', 'right_elbow_joint'))] = 0.5
    FollowingRobot(ros, profile, q0, config['topics']['safety_out'])
    model = shared_model(ros, models, 'g1')
    controller = build(ros, model, config)
    dt = controller.dt
    for kind in ('posture', 'hold'):
        times = steps[kind]
        assert len(times) > 8, (kind, len(times))
        # a step's period is its work and then a whole period, not the period less the work
        periods = np.diff(times)
        assert np.min(periods) >= dt + 0.01 - 1e-3, (kind, np.min(periods))
        assert np.median(periods) < dt + 0.02, (kind, np.median(periods))


@pytest.mark.parametrize('robot', ROBOTS)
def test_relay_starting_late_still_gets_the_whole_start_up_move(ros, models, robot,
                                                               monkeypatch, tmp_path):
    profile = profile_of(robot)
    controller_config, safety_config = LATE[robot]
    q0 = np.zeros(len(profile.body_joints))
    # Each start-up step moves a joint at most dq_lim x dt and then sleeps a whole control
    # period dt, so a move of d rad takes at least d / dq_lim s, plus the steps' work
    if robot == 'g1':
        # the walking node commands 5 s into a 7 s wait for the engagement. The elbows hang
        # 1.5 rad from home: at sim_split's 1 rad/s the move takes at least 1.5 s (about 2 s
        # here), so the start-up ends over 6.5 s after it began, and a 6 s move budget
        # suffices only if it counts from the gate opening
        q0[slots_of(profile, ('left_elbow_joint', 'right_elbow_joint'))] = 1.5
        late, timeouts = 5.0, {'startup_timeout': 7.0, 'startup_move_timeout': 6.0}
    else:
        # golem_safety starts 5 s into a 7 s wait. From home the start-up postures move
        # 1.3 + 0.7 + 1.3 + 0.7 rad at 1 rad/s: at least 4 s (about 6 s here, the IK's work
        # added to each step), so the start-up ends over 9 s after it began; an 8.5 s budget
        late, timeouts = 5.0, {'startup_timeout': 7.0, 'startup_move_timeout': 8.5}
    config = config_of(robot, controller_config, **timeouts)
    robot_ = RelayedRobot(ros, profile, q0, '/lowcmd', sway=0.0)
    robot_.start()
    layer = SafetyNode(robot, safety_config, tmp_path, sim=robot == 'g1')
    ros.on_close(layer.stop)
    legs = WalkingNode(ros, profile, q0)
    startup = watch_startup(monkeypatch)
    opened = watch_gate_opening(monkeypatch)
    model = shared_model(ros, models, robot)

    if robot == 'g1':
        layer.start()
    builder, built = build_in_thread(ros, model, config)
    assert startup['started'].wait(30.0)
    time.sleep(late)
    if robot == 'g1':
        legs.start()
    else:
        layer.start()
    builder.join(45.0)
    assert not builder.is_alive() and 'error' not in built, built.get('error')
    # engaged late, then moved to the end pose past the end of the wait for it and past the
    # move budget counted from the start of the start-up
    assert len(opened) == 1 and opened[0] - startup['start'] >= late
    assert startup['end'] - startup['start'] > config['controller']['startup_move_timeout']
    assert settles(robot_, final_pose(profile, config), 0.02)
    # the controller kept commanding: golem_safety never latched its estop
    assert not layer.estopped(), layer.output()
