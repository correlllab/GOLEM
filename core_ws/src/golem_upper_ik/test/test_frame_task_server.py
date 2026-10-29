'''
frame_task_server and joint_state_publisher on domain 92.

In this process, frame_task_server against a robot that executes and relays its commands
(FollowingRobot), sharing the session's robot model: relative and slow-mode frame goals,
slow-mode named configurations, the fresh-state wait, the stale-state abort, planned goals
and planning turned off, the goal start from the last command, the hold pinned at a
limit, and execute_path's pacing.

Then the installed console scripts as processes: joint_state_publisher's /joint_states,
the parameters both nodes refuse, frame_task_server's shutdown on SIGINT, and its command
rate with the IK running, measured from another process.
'''
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
import numpy as np
import pytest
import yaml

from ros_harness import (
    TEST_DOMAIN, CommandRecorder, FakeRobot, FollowingRobot, RelayedRobot, body_pose,
    profile_of, send_goal, shared_model, slots_of, wait_for,
)

ROBOTS = ('h12', 'g1')
CONFIGS = {'h12': 'safety_split', 'g1': 'g1_split'}
# where the arms start: a working pose with room to move the end effectors forward
START = {'h12': 'prep', 'g1': 'home'}
RATE_TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'frame_task_server_rate.py'


def config_of(robot, name):
    from golem_upper_ik.config import load_controller_config
    return load_controller_config(robot, name)


def edited_config(tmp_path, robot, name, planner=None, **controller):
    '''
    A copy of the installed config with controller keys (and the planner keys of planner)
    replaced, under tmp_path/<robot>/
    '''
    from ament_index_python.packages import get_package_share_directory
    share = Path(get_package_share_directory('golem_upper_ik'))
    raw = yaml.safe_load((share / 'config' / robot / f'{name}.yaml').read_text())
    raw['controller'] = dict(raw.get('controller') or {}, **controller)
    raw['planner'] = dict(raw.get('planner') or {}, **(planner or {}))
    path = tmp_path / robot / f'{name}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw))
    return path


def server_of(ros, model, config, init=False):
    '''A started frame_task_server in this process: model's robot, config a name or a path'''
    from rclpy.parameter import Parameter
    from golem_upper_ik.ros2.frame_task_server import FrameTaskServer
    server = FrameTaskServer(
        context=ros.context, robot_model=model, init=init,
        parameter_overrides=[Parameter('robot', value=model.profile.robot),
                             Parameter('config', value=str(config))])
    ros.add(server)
    ros.on_close(server.shutdown)
    server.start()
    return server


class Log:
    '''What this process has written to stderr, where the nodes' loggers write'''

    def __init__(self, capfd):
        self._capfd = capfd
        self._text = ''

    def lines(self, text):
        self._text += self._capfd.readouterr().err
        return [line for line in self._text.splitlines() if text in line]


class PoseRecorder:
    def __init__(self, ros, topic, msg_type=PoseStamped):
        self.node = ros.node('pose_recorder')
        self.poses = []
        self.node.create_subscription(msg_type, topic, self.poses.append, 10)

    def position(self):
        pose = self.poses[-1].pose
        return np.array([pose.position.x, pose.position.y, pose.position.z])

    def quaternion(self):
        q = self.poses[-1].pose.orientation
        return np.array([q.w, q.x, q.y, q.z])


def offset_pose(x=0.0):
    pose = Pose()
    pose.position.x = x
    pose.orientation.w = 1.0
    return pose


def frame_goal(frame, target, relative=False, slow_mode=False, plan=False, duration=0.0):
    from custom_ros_messages.action import FrameTask
    goal = FrameTask.Goal()
    goal.frame_names = [frame]
    goal.frame_targets = [target]
    goal.relative = relative
    goal.slow_mode = slow_mode
    goal.plan = plan
    goal.duration.sec = int(duration)
    goal.duration.nanosec = int((duration - int(duration)) * 1e9)
    return goal


def named_goal(name, slow_mode=False, plan=False):
    from custom_ros_messages.action import NamedConfig
    goal = NamedConfig.Goal()
    goal.config_name = name
    goal.slow_mode = slow_mode
    goal.plan = plan
    return goal


def ik_slots(profile):
    return slots_of(profile, profile.enabled_joints)


def ee_speeds(model, samples, frame_index, ctrl_hz):
    '''The commanded end-effector speed of each control step, m/s, from the commands'''
    profile = model.profile
    upper = list(profile.upper_slots)
    arm = [upper.index(slot) for slot in ik_slots(profile)]
    steps = [samples[0].q]
    for sample in samples[1:]:
        if not np.array_equal(sample.q, steps[-1]):
            steps.append(sample.q)
    positions = np.array([
        model.get_frame_transformation_reduced(profile.ee_frames[frame_index], q[arm])[:3, 3]
        for q in steps])
    return np.linalg.norm(np.diff(positions, axis=0), axis=1) * ctrl_hz


def joint_speeds(profile, samples, ctrl_hz):
    '''The commanded IK-joint speed of each control step, rad/s, the largest over the joints'''
    upper = list(profile.upper_slots)
    arm = [upper.index(slot) for slot in ik_slots(profile)]
    steps = [samples[0].q[arm]]
    for sample in samples[1:]:
        if not np.array_equal(sample.q[arm], steps[-1]):
            steps.append(sample.q[arm])
    return np.max(np.abs(np.diff(np.array(steps), axis=0)), axis=1) * ctrl_hz


@pytest.mark.parametrize('robot', ROBOTS)
def test_relative_frame_goal(ros, models, robot):
    from custom_ros_messages.action import FrameTask
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot])
    FollowingRobot(ros, profile, body_pose(profile, START[robot]),
                   config['topics']['safety_out'])
    model = shared_model(ros, models, robot)
    left = PoseRecorder(ros, '/left_ee_pose')
    server_of(ros, model, CONFIGS[robot])
    assert wait_for(lambda: left.poses, 5.0)
    start, start_quaternion = left.position(), left.quaternion()

    status, result = send_goal(ros, FrameTask, '/frame_task',
                               frame_goal(profile.ee_frames[0], offset_pose(0.05), relative=True))
    assert status == GoalStatus.STATUS_SUCCEEDED and result.success
    time.sleep(0.1)
    threshold = config['controller']['threshold_linear']
    assert np.linalg.norm(left.position() - (start + [0.05, 0.0, 0.0])) <= threshold + 1e-4
    # the orientation is kept: the target's is the frame's own
    assert abs(abs(np.dot(left.quaternion(), start_quaternion)) - 1.0) < 1e-4


@pytest.mark.parametrize('robot', ROBOTS)
def test_slow_mode_scales_speed(ros, models, robot, tmp_path):
    '''
    slow_mode scales each IK step's joint velocity by SLOW_MODE_SCALE, after the speed limits.
    With the carried configs' limits a move's speed is set by the IK's own acceleration limit
    and damping, not by v_lim, and a slow move steps through more of that speed curve than a
    full-speed one; so here v_lim is low enough to bind, and the two moves' peaks compare
    '''
    from custom_ros_messages.action import FrameTask
    from golem_upper_ik.ros2.frame_task_server import SLOW_MODE_SCALE
    assert SLOW_MODE_SCALE == 0.25
    profile = profile_of(robot)
    v_lim = 0.04  # m/s
    config_path = edited_config(tmp_path, robot, CONFIGS[robot], v_lim=v_lim)
    config = config_of(robot, str(config_path))
    FollowingRobot(ros, profile, body_pose(profile, START[robot]),
                   config['topics']['safety_out'])
    model = shared_model(ros, models, robot)
    recorder = CommandRecorder(ros, config['topics']['low_cmd'], profile)
    controller = server_of(ros, model, config_path).controller
    frame = profile.ee_frames[0]

    # one step: the slow velocity is the full-speed one, scaled
    vel = np.zeros(model.num_motor)
    vel[ik_slots(profile)] = np.linspace(-0.8, 0.8, len(profile.enabled_joints))
    full = controller._limit_joint_vel(vel)
    controller.speed_scale = SLOW_MODE_SCALE
    np.testing.assert_allclose(controller._limit_joint_vel(vel), SLOW_MODE_SCALE * full,
                               rtol=1e-12)
    controller.speed_scale = 1.0

    # the same 2 cm move at full speed and in slow mode, with a move back between them
    peaks = {}
    for slow_mode, x, measured in ((False, 0.02, True), (False, -0.02, False),
                                   (True, 0.02, True)):
        start = time.monotonic()
        status, result = send_goal(
            ros, FrameTask, '/frame_task',
            frame_goal(frame, offset_pose(x), relative=True, slow_mode=slow_mode,
                       duration=15.0))
        assert status == GoalStatus.STATUS_SUCCEEDED and result.success
        if measured:
            samples = [s for s in recorder.samples if s.t >= start]
            peaks[slow_mode] = np.max(
                ee_speeds(model, samples, 0, config['frequency']['ctrl_hz']))
    # the full-speed move is held at the speed limit
    assert peaks[False] == pytest.approx(v_lim, rel=0.02), peaks
    assert peaks[True] <= SLOW_MODE_SCALE * peaks[False] + 1e-3, peaks


@pytest.mark.parametrize('robot', ROBOTS)
def test_named_config_slow_mode(ros, models, robot):
    from custom_ros_messages.action import NamedConfig
    from golem_upper_ik.ros2.frame_task_server import SLOW_MODE_SCALE
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot])
    name = START[robot]
    # the left shoulder pitch starts 0.15 rad from the named configuration
    q0 = body_pose(profile, name)
    q0[profile.body_joints.index('left_shoulder_pitch_joint')] += 0.15
    robot_ = FollowingRobot(ros, profile, q0, config['topics']['safety_out'])
    model = shared_model(ros, models, robot)
    recorder = CommandRecorder(ros, config['topics']['low_cmd'], profile)
    server_of(ros, model, CONFIGS[robot])

    start = time.monotonic()
    status, result = send_goal(ros, NamedConfig, '/named_config',
                               named_goal(name, slow_mode=True))
    assert status == GoalStatus.STATUS_SUCCEEDED and result.success
    reached = robot_.measured()[ik_slots(profile)]
    assert np.max(np.abs(reached - profile.named_configs[name])) < \
        config['controller']['threshold_joint']
    # at a quarter of the joint speed limit
    samples = [s for s in recorder.samples if s.t >= start]
    speeds = joint_speeds(profile, samples, config['frequency']['ctrl_hz'])
    dq_lim = config['controller']['dq_lim']
    assert np.max(speeds) <= SLOW_MODE_SCALE * dq_lim * 1.001 + 1e-4
    assert np.max(speeds) > 0.5 * SLOW_MODE_SCALE * dq_lim


def test_startup_waits_for_fresh_state(ros, models, tmp_path):
    profile = profile_of('h12')
    config = edited_config(tmp_path, 'h12', CONFIGS['h12'], state_timeout=0.5)
    recorder = CommandRecorder(ros, '/safety/lowcmd_upper_in', profile)
    model = shared_model(ros, models, 'h12')
    probe = ros.node('probe')
    start = time.monotonic()
    with pytest.raises(RuntimeError, match=r'No fresh /lowstate .* within 0.5 s .*not '
                                           r'enabling any motor'):
        server_of(ros, model, config, init=True)
    assert time.monotonic() - start < 2.0
    time.sleep(0.2)
    assert recorder.count == 0
    # nothing tells the robot's other nodes that the controller is up
    assert probe.count_publishers('/left_ee_pose') == 0
    assert probe.count_publishers('/safety/lowcmd_upper_in') == 0


def test_stale_state_aborts_goal(ros, models, monkeypatch, capfd):
    from custom_ros_messages.action import FrameTask
    from golem_upper_ik.core.controller import upper_controller
    profile = profile_of('h12')
    config = config_of('h12', CONFIGS['h12'])
    robot_ = FollowingRobot(ros, profile, body_pose(profile, START['h12']),
                            config['topics']['safety_out'])
    model = shared_model(ros, models, 'h12')
    recorder = CommandRecorder(ros, config['topics']['low_cmd'], profile)
    log = Log(capfd)
    # off the real robot's domain the tick-lag check is off: the sims stamp their tick in
    # sim steps
    assert upper_controller.ros_domain_id(ros.node('probe')) == TEST_DOMAIN
    monkeypatch.setattr(upper_controller, 'ros_domain_id', lambda node: 0)
    server = server_of(ros, model, CONFIGS['h12'])
    assert server.controller.max_state_lag_ms == config['controller']['max_state_lag_ms'] == 50.0

    # a target out of reach keeps the goal stepping the IK until /lowstate stops
    stop_at = []

    def stop_state_mid_goal():
        time.sleep(1.0)
        stop_at.append(time.monotonic())
        robot_.stop_state()
    threading.Thread(target=stop_state_mid_goal, daemon=True).start()
    status, result = send_goal(
        ros, FrameTask, '/frame_task',
        frame_goal(profile.ee_frames[0], offset_pose(1.0), relative=True, duration=8.0))
    ended = time.monotonic()
    assert status == GoalStatus.STATUS_ABORTED and not result.success
    assert ended - stop_at[0] < 1.0
    assert wait_for(lambda: log.lines('[frame_task_server]: Aborting goal: /lowstate is '), 2.0)
    line = log.lines('Aborting goal')[0]
    assert 'ms stale (> 50 ms); holding the last command' in line, line
    # the last command holds
    time.sleep(0.2)
    held = [s.q for s in recorder.samples if s.t > ended]
    assert len(held) > 50
    assert all(np.array_equal(q, held[0]) for q in held)


def test_stale_state_does_not_abort_a_goal_off_the_real_robots_domain(ros, models, capfd):
    from custom_ros_messages.action import FrameTask
    profile = profile_of('h12')
    config = config_of('h12', CONFIGS['h12'])
    robot_ = FollowingRobot(ros, profile, body_pose(profile, START['h12']),
                            config['topics']['safety_out'])
    model = shared_model(ros, models, 'h12')
    log = Log(capfd)
    # on the test domain the tick-lag check is off: its deadline is 0
    server = server_of(ros, model, CONFIGS['h12'])
    assert server.controller.max_state_lag_ms == 0.0
    # a slow 30 cm move keeps the IK stepping past the goal's 2 s; /lowstate stops at 0.5 s
    threading.Timer(0.5, robot_.stop_state).start()
    start = time.monotonic()
    status, result = send_goal(
        ros, FrameTask, '/frame_task',
        frame_goal(profile.ee_frames[0], offset_pose(0.3), relative=True, slow_mode=True,
                   duration=2.0))
    # the goal ran to its timeout on the stale state, and was not aborted
    assert status == GoalStatus.STATUS_SUCCEEDED
    assert time.monotonic() - start >= 2.0
    assert log.lines('Aborting goal') == []


def test_frame_task_target_never_raises_while_tasks_are_added(ros, models):
    from golem_upper_ik.core.controller.frame_controller import FrameController
    profile = profile_of('g1')
    config = config_of('g1', CONFIGS['g1'])
    FollowingRobot(ros, profile, body_pose(profile, START['g1']), config['topics']['safety_out'])
    model = shared_model(ros, models, 'g1')
    controller = FrameController(ros.node('controller'), profile, config, robot_model=model,
                                 init=False)
    ros.on_close(controller.shutdown)
    frame = profile.ee_frames[0]
    target = controller.ee_transformations()[0]
    # another thread reads the target in a tight loop, with the GIL handed over as often as
    # the interpreter allows, while tasks come and go
    errors, reads = [], [0]
    done = threading.Event()

    def read():
        while not done.is_set():
            try:
                controller.frame_task_target(f'{frame}_task')
                reads[0] += 1
            except Exception as error:  # the outcome under test
                errors.append(error)
                return
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    reader = threading.Thread(target=read, daemon=True)
    try:
        reader.start()
        for _ in range(3000):
            controller.clear_frame_tasks()
            controller.add_frame_task(f'{frame}_task', frame, target)
            if errors:
                break
    finally:
        done.set()
        reader.join(timeout=5.0)
        sys.setswitchinterval(switch_interval)
    assert errors == []
    assert reads[0] > 1000
    np.testing.assert_allclose(controller.frame_task_target(f'{frame}_task'), target)


def test_unknown_frame_is_refused_and_the_poses_keep_going(ros, models, capfd):
    from custom_ros_messages.action import FrameTask
    from rclpy.action import ActionClient
    profile = profile_of('h12')
    config = config_of('h12', CONFIGS['h12'])
    FollowingRobot(ros, profile, body_pose(profile, START['h12']),
                   config['topics']['safety_out'])
    model = shared_model(ros, models, 'h12')
    log = Log(capfd)
    left = PoseRecorder(ros, '/left_ee_pose')
    poses = PoseRecorder(ros, '/frame_poses', PoseArray)
    server_of(ros, model, CONFIGS['h12'])
    assert wait_for(lambda: left.poses, 5.0)

    goal = FrameTask.Goal()
    goal.frame_names = [profile.ee_frames[0], 'no_such_frame']
    goal.frame_targets = [offset_pose(0.0), offset_pose(0.0)]
    goal.relative = True
    client = ActionClient(ros.node('action_client'), FrameTask, '/frame_task')
    assert client.wait_for_server(timeout_sec=10.0)
    sent = client.send_goal_async(goal)
    assert wait_for(sent.done, 10.0)
    assert not sent.result().accepted
    line = log.lines('no_such_frame')
    assert line and 'Refusing' in line[0], line
    # the poses keep going out, the refused goal's frames never among them
    count = len(left.poses)
    assert wait_for(lambda: len(left.poses) > count + 50, 2.0)
    assert all(len(msg.poses) == 0 for msg in poses.poses)


@pytest.mark.parametrize('robot', ROBOTS)
def test_planned_goals(ros, models, robot):
    '''
    Planned named configurations out to another posture and back, then a planned frame goal.
    The G1 goes out to t_pose; the H1-2, whose t_pose is past its planner frames' y limit, to
    arms_front_45
    '''
    from custom_ros_messages.action import FrameTask, NamedConfig
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot])
    robot_ = FollowingRobot(ros, profile, body_pose(profile, START[robot]),
                            config['topics']['safety_out'])
    model = shared_model(ros, models, robot)
    server_of(ros, model, CONFIGS[robot])
    threshold = config['controller']['threshold_joint']

    for name in ({'h12': 'arms_front_45', 'g1': 't_pose'}[robot], START[robot]):
        status, result = send_goal(ros, NamedConfig, '/named_config',
                                   named_goal(name, plan=True))
        assert status == GoalStatus.STATUS_SUCCEEDED and result.success, (name, result)
        reached = robot_.measured()[ik_slots(profile)]
        assert np.max(np.abs(reached - profile.named_configs[name])) < threshold, name

    left = profile.ee_frames[0]
    before = model.get_frame_position(left, robot_.measured()).copy()
    status, result = send_goal(ros, FrameTask, '/frame_task',
                               frame_goal(left, offset_pose(0.05), relative=True, plan=True))
    assert status == GoalStatus.STATUS_SUCCEEDED and result.success, result
    moved = model.get_frame_position(left, robot_.measured()) - before
    assert np.linalg.norm(moved) == pytest.approx(0.05, abs=5e-3), moved


def test_planning_turned_off(ros, models, capfd, tmp_path):
    '''planner.enabled: false refuses every planned goal before any IK solve'''
    from custom_ros_messages.action import FrameTask, NamedConfig
    profile = profile_of('g1')
    config_path = edited_config(tmp_path, 'g1', CONFIGS['g1'], planner={'enabled': False})
    config = config_of('g1', str(config_path))
    FollowingRobot(ros, profile, body_pose(profile, START['g1']), config['topics']['safety_out'])
    model = shared_model(ros, models, 'g1')
    log = Log(capfd)
    recorder = CommandRecorder(ros, config['topics']['low_cmd'], profile)
    server_of(ros, model, config_path)
    assert wait_for(lambda: recorder.count > 0, 2.0)

    start = time.monotonic()
    status, result = send_goal(ros, FrameTask, '/frame_task',
                               frame_goal(profile.ee_frames[0], offset_pose(0.05),
                                          relative=True, plan=True))
    assert status == GoalStatus.STATUS_ABORTED and not result.success
    status, result = send_goal(ros, NamedConfig, '/named_config', named_goal('home', plan=True))
    assert status == GoalStatus.STATUS_ABORTED and not result.success
    assert wait_for(lambda: len(log.lines('Planning is turned off')) == 2, 2.0)
    line = log.lines('Planning is turned off')[0]
    assert '[frame_task_server]: Planning or execution failed: ' in line, line
    assert f'the controller config {config_path} sets planner.enabled: false' in line, line
    assert 'send the goal with plan=false' in line, line
    # nothing moved
    sent = [s.q for s in recorder.samples if s.t >= start]
    assert all(np.array_equal(q, sent[0]) for q in sent)


@pytest.mark.parametrize('robot', ROBOTS)
def test_goals_start_from_the_last_command(ros, models, robot):
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot])
    arms = ik_slots(profile)
    # the left elbow sits 0.05 rad below every command (a gravity sag inside the 0.1 rad
    # the seed accepts), the right elbow 0.2 rad below (an arm held off its command)
    sag = np.zeros(len(profile.body_joints))
    left_elbow = profile.body_joints.index('left_elbow_joint')
    sag[left_elbow] = 0.05
    robot_ = FollowingRobot(ros, profile, body_pose(profile, START[robot]),
                            config['topics']['safety_out'], sag=sag)
    model = shared_model(ros, models, robot)
    controller = server_of(ros, model, CONFIGS[robot]).controller
    assert wait_for(lambda: controller.low_cmd_handler.following, 2.0)
    controller.goto_reduced_configuration(profile.named_configs[START[robot]])
    time.sleep(0.1)
    commanded = controller.low_cmd_handler.q_cmd.astype(np.float64)
    assert abs(robot_.measured()[left_elbow] - (commanded[left_elbow] - 0.05)) < 1e-6

    # within the gap the IK starts from the command, not the sagging measured pose
    controller.update_ik_solver()
    np.testing.assert_allclose(controller.ik_solver.q[arms], commanded[arms], atol=1e-6)

    # past it, from the measured pose, and the held torque bias is dropped
    right_elbow = profile.body_joints.index('right_elbow_joint')
    sag[right_elbow] = 0.2
    robot_.set_sag(sag)
    controller.goto_reduced_configuration(profile.named_configs[START[robot]])
    time.sleep(0.1)
    controller._tau_bias[:] = 1.0
    controller.update_ik_solver()
    measured = robot_.measured()
    np.testing.assert_allclose(controller.ik_solver.q[arms], measured[arms], atol=1e-6)
    np.testing.assert_array_equal(controller._tau_bias, 0.0)


@pytest.mark.parametrize('robot', ROBOTS)
def test_hold_pinned_at_a_limit_is_not_converged(ros, models, robot):
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot])
    # the G1's configs carry no integral gain; the hold's I-term needs one
    config['gains']['ki'][list(profile.upper_slots)] = 100.0
    limits = config['limits']['q_clip_limits']
    elbow = profile.body_joints.index('left_elbow_joint')
    # the left elbow is commanded at its clip limit and stops 0.5 mrad short of it
    sag = np.zeros(len(profile.body_joints))
    sag[elbow] = 0.0005
    q0 = body_pose(profile, START[robot])
    robot_ = FollowingRobot(ros, profile, q0, config['topics']['safety_out'], sag=sag)
    model = shared_model(ros, models, robot)
    from golem_upper_ik.core.controller.frame_controller import FrameController
    controller = FrameController(ros.node('controller'), profile, config, robot_model=model,
                                 init=False)
    ros.on_close(controller.shutdown)
    assert wait_for(lambda: controller.low_cmd_handler.following, 2.0)
    q = q0.copy()
    q[elbow] = limits[elbow, 1]
    controller._apply_joint_position(q)
    assert wait_for(lambda: abs(robot_.measured()[elbow] - (limits[elbow, 1] - 0.0005)) < 1e-6,
                    2.0)
    controller.init_steady_state()
    assert controller.steady_state_step() is False
    limited = controller.steady_state_limited_joints
    assert [name for name, _, _ in limited] == ['left_elbow_joint']
    assert limited[0][2] == pytest.approx(limits[elbow, 1])


def test_execute_path_waits_for_the_follow_gate(ros, models):
    from golem_upper_ik.core.controller.frame_controller import FrameController
    profile = profile_of('h12')
    config = config_of('h12', CONFIGS['h12'])
    model = shared_model(ros, models, 'h12')
    # nothing relays the commands: a path's absolute waypoints would run ahead of a robot
    # that is not following, so it is refused
    robot_ = RelayedRobot(ros, profile, body_pose(profile, START['h12']), '/lowcmd')
    robot_.start()
    controller = FrameController(ros.node('controller'), profile, config, robot_model=model,
                                 init=False)
    ros.on_close(controller.shutdown)
    path = np.linspace(profile.named_configs['prep'], profile.named_configs['prep'] + 0.05, 30)
    with pytest.raises(RuntimeError, match='follow gate'):
        controller.execute_path(path)


def test_execute_path_paces_at_dt_minus_work(ros, models, monkeypatch):
    profile = profile_of('h12')
    config = config_of('h12', CONFIGS['h12'])
    model = shared_model(ros, models, 'h12')
    q0 = body_pose(profile, START['h12'])
    FollowingRobot(ros, profile, q0, config['topics']['safety_out'])
    from golem_upper_ik.core.controller.frame_controller import FrameController
    controller = FrameController(ros.node('controller'), profile, config, robot_model=model,
                                 init=False)
    ros.on_close(controller.shutdown)
    assert wait_for(lambda: controller.low_cmd_handler.following, 2.0)
    # each waypoint's checks take 10 ms: a step that slept the whole period after them would
    # stretch a 30-waypoint path by 0.3 s
    check = model.check_collision_free_reduced

    def slow_check(q_reduced):
        time.sleep(0.01)
        return check(q_reduced)
    monkeypatch.setattr(model, 'check_collision_free_reduced', slow_check)
    path = np.linspace(profile.named_configs['prep'], profile.named_configs['prep'] + 0.05, 30)
    start = time.monotonic()
    controller.execute_path(path)
    elapsed = time.monotonic() - start
    assert len(path) * controller.dt - 0.05 < elapsed < len(path) * controller.dt + 0.15
    np.testing.assert_allclose(controller.low_cmd_handler.q_cmd[ik_slots(profile)], path[-1],
                               atol=1e-6)


# the installed console scripts as processes

def executable(name):
    from ament_index_python.packages import get_package_prefix
    return str(Path(get_package_prefix('golem_upper_ik')) / 'lib' / 'golem_upper_ik' / name)


def node_env():
    return dict(os.environ, ROS_DOMAIN_ID=str(TEST_DOMAIN), RCUTILS_COLORIZED_OUTPUT='0')


def run_node(name, *params, timeout=30.0):
    '''Run a console script with -p params until it exits: (exit code, output)'''
    args = [executable(name), '--ros-args']
    for param in params:
        args += ['-p', param]
    done = subprocess.run(args, env=node_env(), capture_output=True, text=True,
                          timeout=timeout)
    return done.returncode, done.stdout + done.stderr


class NodeProcess:
    def __init__(self, name, *params):
        args = [executable(name), '--ros-args']
        for param in params:
            args += ['-p', param]
        self.proc = subprocess.Popen(args, env=node_env(), text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
        self.lines = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip('\n'))

    def output(self):
        return '\n'.join(self.lines)

    def interrupt(self, signals=(signal.SIGINT,), timeout=15.0):
        '''Send signals one after another; the exit code'''
        for signum in signals:
            self.proc.send_signal(signum)
        return self.proc.wait(timeout=timeout)

    def wait(self, timeout):
        return self.proc.wait(timeout=timeout)

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5.0)


@pytest.mark.parametrize('robot', ROBOTS)
def test_joint_state_publisher(ros, robot):
    from sensor_msgs.msg import JointState
    profile = profile_of(robot)
    fake = FakeRobot(ros, profile)
    fake.q[:] = np.clip(0.01 * np.arange(len(profile.body_joints)) - 0.1, profile.q_low,
                        profile.q_high)
    fake.start(rate_hz=200.0)
    node = ros.node('joint_states_recorder')
    received = []
    node.create_subscription(JointState, '/joint_states',
                             lambda msg: received.append((time.monotonic(), msg)), 100)
    process = NodeProcess('joint_state_publisher', f'robot:={robot}')
    try:
        # the joints carry /lowstate's positions once its first sample is in
        expected = fake.q.astype(np.float32)
        assert wait_for(lambda: received and np.array_equal(
            np.float32(received[-1][1].position), expected), 20.0), process.output()
        start = time.monotonic()
        time.sleep(3.0)
        window = [(t, msg) for t, msg in received if t >= start]
    finally:
        assert process.interrupt() == 0, process.output()
    assert tuple(window[-1][1].name) == profile.body_joints
    np.testing.assert_array_equal(np.float32(window[-1][1].position), expected)
    rate = (len(window) - 1) / (window[-1][0] - window[0][0])
    assert 95.0 <= rate <= 105.0, rate


@pytest.mark.parametrize('name, params, words', [
    ('frame_task_server', [], ['robot', 'h12', 'g1']),
    ('frame_task_server', ['robot:=h1'], ["unknown robot 'h1'"]),
    ('frame_task_server', ['robot:=h12', 'config:=g1_split'], ['robot h12', "'g1_split'"]),
    ('frame_task_server', ['robot:=h12', 'config:=/nonexistent/g1/g1.yaml'],
     ['robot h12', 'config directory of robot g1']),
    ('joint_state_publisher', [], ['robot', 'h12', 'g1']),
    ('joint_state_publisher', ['robot:=h12', 'config:=no_such_config'],
     ['robot h12', 'no_such_config']),
])
def test_nodes_refuse_bad_parameters(name, params, words):
    code, output = run_node(name, *params)
    assert code != 0, output
    for word in words:
        assert word in output, output


def assert_zero_gain_last(robot, profile, before):
    '''
    The robot's last command arrived before `before` and turns the IK joints off; the G1's
    held joints (its waist) stay at their hold on /lowcmd
    '''
    received, last = robot.last_command
    assert received < before
    held = slots_of(profile, profile.held_joints)
    for slot in profile.upper_slots:
        motor = last.motor_cmd[slot]
        if slot in held:
            assert motor.mode == 1 and motor.kp > 0.0, slot
        else:
            assert (motor.mode, motor.kp, motor.kd, motor.tau) == (0, 0.0, 0.0, 0.0), slot


@pytest.mark.parametrize('signals', [
    (signal.SIGINT,), (signal.SIGTERM,), (signal.SIGINT, signal.SIGINT),
    (signal.SIGTERM, signal.SIGINT)], ids=['SIGINT', 'SIGTERM', 'SIGINT-twice', 'SIGTERM-SIGINT'])
def test_frame_task_server_shuts_down_with_zero_gains_on_a_signal(ros, signals):
    profile = profile_of('g1')
    config = config_of('g1', CONFIGS['g1'])
    robot_ = FollowingRobot(ros, profile, body_pose(profile, 'home'),
                            config['topics']['safety_out'])
    left = PoseRecorder(ros, '/left_ee_pose')
    process = NodeProcess('frame_task_server', 'robot:=g1', f'config:={CONFIGS["g1"]}')
    try:
        # its end-effector poses go out once the start-up is done
        assert wait_for(lambda: left.poses, 60.0), process.output()
        assert robot_.zero_gain_commands == 0
        # a second signal, as from an impatient operator, arrives during the shutdown
        code = process.interrupt(signals)
        stopped = time.monotonic()
    finally:
        process.kill()
    assert code == 0, process.output()
    assert 'Traceback' not in process.output(), process.output()
    time.sleep(0.3)
    assert_zero_gain_last(robot_, profile, stopped)


def test_frame_task_server_start_up_failure_exits_1_after_a_zero_gain_command(ros, tmp_path):
    # /lowstate comes, but nothing relays the commands: the start-up stops at its 2 s
    # startup_timeout after it has begun publishing
    profile = profile_of('g1')
    config_path = edited_config(tmp_path, 'g1', CONFIGS['g1'], startup_timeout=2.0)
    robot_ = RelayedRobot(ros, profile, body_pose(profile, 'home'), '/lowcmd')
    robot_.start()
    commands = FollowingCommands(ros, profile)
    process = NodeProcess('frame_task_server', 'robot:=g1', f'config:={config_path}')
    try:
        code = process.wait(timeout=60.0)
        stopped = time.monotonic()
    finally:
        process.kill()
    assert code == 1, process.output()
    assert 'golem_safety did not relay the commands within controller.startup_timeout' in \
        process.output(), process.output()
    assert commands.count > 100, process.output()
    time.sleep(0.3)
    assert_zero_gain_last(commands, profile, stopped)


class FollowingCommands:
    '''Every command on /safety/lowcmd_upper_in: their count and the last, with its time'''

    def __init__(self, ros, profile):
        from unitree_hg.msg import LowCmd
        from ros_harness import RECORD_QOS
        self.node = ros.node('commands')
        self.count = 0
        self.last_command = None
        self.node.create_subscription(LowCmd, '/safety/lowcmd_upper_in', self._on_command,
                                      RECORD_QOS)

    def _on_command(self, msg):
        self.count += 1
        self.last_command = (time.monotonic(), msg)


# Threads frame_task_server may have that carry its process's name: the main thread, the
# executor's, its pool's, the command publisher's and the follow gate's. OpenBLAS's workers
# carry it too, one per core but one, unless the node pins BLAS to one thread
MAX_THREADS_NAMED_AS_PROCESS = 16


def test_frame_task_server_rate():
    '''
    The H1-2's commands through its start-up and its goals, received in another process: at
    500 Hz, with at most 3 gaps over 20 ms and none over 50 ms. The 99th percentile of the
    period is printed, not bounded: the IK, rclpy's handling of /lowstate at 500 Hz and the
    100 Hz pose topics share the GIL with the publisher thread. BLAS runs one thread
    '''
    env = dict(os.environ, ROS_DOMAIN_ID=str(TEST_DOMAIN))
    done = subprocess.run([sys.executable, str(RATE_TOOL), '--robot', 'h12', '--seconds', '6'],
                          env=env, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stdout + done.stderr
    stats = json.loads(done.stdout.strip().splitlines()[-1])
    print(json.dumps(stats))
    for window in ('startup', 'goals'):
        assert stats[window]['mean_hz'] >= 495, stats
        assert stats[window]['gaps_over_20ms'] <= 3, stats
        assert stats[window]['max_gap_ms'] <= 50.0, stats
    # the IK stepped at ctrl_hz throughout the goals
    assert stats['goals']['control_steps_hz'] >= 25.0, stats
    assert stats['server_threads_named_as_process'] <= MAX_THREADS_NAMED_AS_PROCESS, stats
    assert stats['server_exit_code'] == 0, stats
    assert stats['shutdown_zero_gain'] is True, stats


def test_launch_file_starts_both_nodes():
    import importlib.util
    from ament_index_python.packages import get_package_share_directory
    from launch_ros.actions import Node
    path = Path(get_package_share_directory('golem_upper_ik')) / 'launch' / 'upper_ik.launch.py'
    spec = importlib.util.spec_from_file_location('upper_ik_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    nodes = [entity for entity in description.entities if isinstance(entity, Node)]
    assert sorted(node.node_executable for node in nodes) == [
        'frame_task_server', 'joint_state_publisher']
    arguments = {entity.name for entity in description.entities
                 if type(entity).__name__ == 'DeclareLaunchArgument'}
    assert arguments == {'robot', 'config', 'use_sim_time'}
