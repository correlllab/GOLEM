"""Check frame_task_server and joint_state_publisher on domain 92, and the launch files.

In this process, frame_task_server against a robot that executes and relays its commands
(FollowingRobot), sharing the session's robot model: pelvis-frame grasp-frame goals whatever
the IMU says, a goal's timeout in sim time, the fresh-state wait, the stale-state abort,
planned goals and planning turned off, the goal start from the last command, the hold pinned
at a limit, and execute_path's pacing; /grasp_frame_target's goals in the pelvis frame and
through TF, planned, refused, timed out and canceled, and the grasp pose topics.

Then the installed console scripts as processes: joint_state_publisher's /joint_states,
the parameters both nodes refuse, frame_task_server's shutdown on a signal, and its command
rate with the IK running, measured from another process. Last, the package's launch files:
upper_ik.launch.py's pairing check and controller, and robot_state.launch.py's state nodes.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseArray, PoseStamped
import numpy as np
import pytest
import yaml
from golem_upper_ik.profile import known_robots

from ros_harness import (
    TEST_DOMAIN, CommandRecorder, FakeRobot, FollowingRobot, RelayedRobot, SimClock, Spring,
    body_pose, profile_of, shared_model, slots_of, wait_for,
)

ROBOTS = known_robots()
CONFIGS = {'h12': 'safety_split', 'g1': 'g1_split'}
# where the arms start: a working pose with room to move the end effectors forward
START = {'h12': 'prep', 'g1': 'home'}
RATE_TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'frame_task_server_rate.py'


def config_of(robot, name):
    """Return robot's controller config name."""
    from golem_upper_ik.config import load_controller_config
    return load_controller_config(robot, name)


def edited_config(tmp_path, robot, name, planner=None, **controller):
    """Return a copy of the installed config, keys replaced, written under tmp_path/<robot>/."""
    from ament_index_python.packages import get_package_share_directory
    share = Path(get_package_share_directory('golem_upper_ik'))
    raw = yaml.safe_load((share / 'config' / robot / f'{name}.yaml').read_text())
    raw['controller'] = dict(raw.get('controller') or {}, **controller)
    raw['planner'] = dict(raw.get('planner') or {}, **(planner or {}))
    path = tmp_path / robot / f'{name}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw))
    return path


def server_of(ros, model, config, init=False, sim_time=False):
    """Return a started frame_task_server in this process on model's robot and config.

    sim_time: the node runs with use_sim_time.
    """
    from rclpy.parameter import Parameter
    from golem_upper_ik.ros2.frame_task_server import FrameTaskServer
    server = FrameTaskServer(
        context=ros.context, robot_model=model, init=init,
        parameter_overrides=[Parameter('robot', value=model.profile.robot),
                             Parameter('config', value=str(config)),
                             Parameter('use_sim_time', value=sim_time)])
    ros.add(server)
    ros.on_close(server.shutdown)
    server.start()
    return server


class Log:
    """Read what this process has written to stderr, where the nodes' loggers write."""

    def __init__(self, capfd):
        """Read through capfd."""
        self._capfd = capfd
        self._text = ''

    def lines(self, text):
        """Return the stderr lines so far that contain text."""
        self._text += self._capfd.readouterr().err
        return [line for line in self._text.splitlines() if text in line]


class PoseRecorder:
    """Record every pose message on a topic."""
    def __init__(self, ros, topic, msg_type=PoseStamped):
        """Subscribe to topic for msg_type."""
        self.node = ros.node('pose_recorder')
        self.poses = []
        self.node.create_subscription(msg_type, topic, self.poses.append, 10)

    def position(self):
        """Return the newest pose's position, m."""
        pose = self.poses[-1].pose
        return np.array([pose.position.x, pose.position.y, pose.position.z])

    def quaternion(self):
        """Return the newest pose's orientation (w, x, y, z)."""
        q = self.poses[-1].pose.orientation
        return np.array([q.w, q.x, q.y, q.z])


def ik_slots(profile):
    """Return the body slots of the IK joints."""
    return slots_of(profile, profile.enabled_joints)


def _rotation(axis, angle):
    """Return the 3x3 rotation by angle (rad) about the axis x, y or z."""
    c, s = np.cos(angle), np.sin(angle)
    i, j = {'x': (1, 2), 'y': (2, 0), 'z': (0, 1)}[axis]
    rotation = np.eye(3)
    rotation[i, i], rotation[i, j], rotation[j, i], rotation[j, j] = c, -s, s, c
    return rotation


def _quaternion_wxyz(rotation):
    """Return a rotation matrix as a quaternion (w, x, y, z)."""
    import pinocchio as pin
    q = pin.Quaternion(rotation)
    return np.array([q.w, q.x, q.y, q.z])


# IMU orientations (w, x, y, z): level and facing the world's x, and the G1 sim's yaw of
# -90 deg with a tilt, where the pelvis frame and the IMU-oriented world differ
IMU_ORIENTATIONS = {
    'identity': np.array([1.0, 0.0, 0.0, 0.0]),
    'yawed_tilted': _quaternion_wxyz(
        _rotation('z', -np.pi / 2) @ _rotation('y', 0.1) @ _rotation('x', 0.15)),
}


def _matrix(stamped):
    """Return a PoseStamped's pose as a 4x4 transformation."""
    from golem_upper_ik.ros2.utility import pose_to_matrix
    return pose_to_matrix(stamped.pose)


@pytest.mark.parametrize('imu', IMU_ORIENTATIONS)
@pytest.mark.parametrize('robot', ROBOTS)
def test_grasp_frame_goal_is_a_pelvis_frame_move(ros, models, robot, imu):
    """A pelvis-frame target is reached in the pelvis frame whatever the IMU says.

    Each target is the grasp frame's measured pose (/left_grasp_frame_pose) moved in pelvis
    axes; /frame_targets carries it and /frame_poses the measured pose, both in the pelvis
    frame too.
    """
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot])
    FollowingRobot(ros, profile, body_pose(profile, START[robot]),
                   '/lowcmd', imu_quaternion=IMU_ORIENTATIONS[imu])
    model = shared_model(ros, models, robot)
    left = PoseRecorder(ros, '/left_grasp_frame_pose')
    frame_targets = PoseRecorder(ros, '/frame_targets', PoseArray)
    frame_poses = PoseRecorder(ros, '/frame_poses', PoseArray)
    server_of(ros, model, CONFIGS[robot])
    assert wait_for(lambda: left.poses, 5.0)
    threshold_linear = config['controller']['threshold_linear']
    threshold_angular = config['controller']['threshold_angular']
    client = GraspClient(ros)

    # down, not up: from the H1-2's prep, 5 cm up and 5 cm out yawed needs the left elbow at
    # its joint-table limit, outside the IK's range
    for position, rotation in (((0.0, 0.0, -0.05), np.eye(3)),
                               ((0.05, 0.0, 0.0), np.eye(3)),
                               ((0.0, 0.0, 0.0), _rotation('z', 0.2))):
        time.sleep(0.1)
        expected = _shifted(_matrix(left.poses[-1]), position, rotation)
        status, result = client.run(grasp_goal(left=expected))
        assert status == GoalStatus.STATUS_SUCCEEDED and result.success, result.message
        time.sleep(0.1)
        resolved = frame_targets.poses[-1].poses[0]
        np.testing.assert_allclose(
            [resolved.position.x, resolved.position.y, resolved.position.z],
            expected[:3, 3], atol=1e-5)
        reached = _matrix(left.poses[-1])
        assert np.linalg.norm(reached[:3, 3] - expected[:3, 3]) <= threshold_linear + 1e-4
        assert _angle(reached, expected) <= threshold_angular + 1e-4
        measured = frame_poses.poses[-1].poses[0]
        np.testing.assert_allclose(
            [measured.position.x, measured.position.y, measured.position.z],
            reached[:3, 3], atol=1e-4)


@pytest.mark.parametrize('rate', [0.5, 2.0])
def test_goal_duration_counts_sim_time(ros, models, rate):
    """Under use_sim_time a goal's timeout is s of sim time at any sim speed."""
    profile = profile_of('h12')
    name = START['h12']
    # the left elbow settles 0.05 rad short of every command, so the grasp frame never
    # reaches its target and the goal runs for its whole timeout
    sag = np.zeros(len(profile.body_joints))
    sag[profile.body_joints.index('left_elbow_joint')] = 0.05
    FollowingRobot(ros, profile, body_pose(profile, name), '/lowcmd', sag=sag)
    sim = SimClock(ros, rate=rate)
    sim.start()
    model = shared_model(ros, models, 'h12')
    left = PoseRecorder(ros, '/left_grasp_frame_pose')
    server = server_of(ros, model, CONFIGS['h12'], sim_time=True)
    assert server.controller.clock._ros_clock is not None
    assert wait_for(lambda: server.controller.low_cmd_handler.following, 5.0)
    assert wait_for(lambda: left.poses, 5.0)
    target = _shifted(_matrix(left.poses[-1]), (0.02, 0.0, 0.0))
    start, sim_start = time.monotonic(), sim.now
    status, result = GraspClient(ros).run(grasp_goal(left=target, timeout=1.0))
    wall, sim_seconds = time.monotonic() - start, sim.now - sim_start
    assert status == GoalStatus.STATUS_ABORTED, result.message
    assert result.outcome == grasp_outcome('NOT_REACHED')
    # 1 s of sim time: 2 s of wall time at half speed, 0.5 s at twice
    assert 1.0 <= sim_seconds < 1.3, sim_seconds
    assert 1.0 / rate - 0.1 <= wall < 1.0 / rate + 0.4, wall


def test_startup_waits_for_fresh_state(ros, models, tmp_path):
    """Without /lowstate the server stops at state_timeout, commanding and publishing nothing."""
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
    assert probe.count_publishers('/left_grasp_frame_pose') == 0
    assert probe.count_publishers('/safety/lowcmd_upper_in') == 0


def test_stale_state_aborts_goal(ros, models, monkeypatch, capfd):
    """On domain 0, /lowstate going stale aborts a goal, and the last command holds."""
    from golem_upper_ik.core.controller import upper_controller
    profile = profile_of('h12')
    config = config_of('h12', CONFIGS['h12'])
    robot_ = FollowingRobot(ros, profile, body_pose(profile, START['h12']), '/lowcmd')
    model = shared_model(ros, models, 'h12')
    left = PoseRecorder(ros, '/left_grasp_frame_pose')
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
        """Stop /lowstate 1 s into the goal."""
        time.sleep(1.0)
        stop_at.append(time.monotonic())
        robot_.stop_state()
    threading.Thread(target=stop_state_mid_goal, daemon=True).start()
    assert wait_for(lambda: left.poses, 5.0)
    target = _shifted(_matrix(left.poses[-1]), (1.0, 0.0, 0.0))
    status, result = GraspClient(ros).run(grasp_goal(left=target, timeout=8.0))
    ended = time.monotonic()
    assert status == GoalStatus.STATUS_ABORTED and not result.success
    assert result.outcome == grasp_outcome('FAILED')
    assert ended - stop_at[0] < 1.0
    assert wait_for(
        lambda: log.lines('[frame_task_server]: Grasp frame target abort: /lowstate is '), 2.0)
    line = log.lines('Grasp frame target abort')[0]
    assert 'ms stale (> 50 ms); holding the last command' in line, line
    # the last command holds: once it arrives it never changes. Samples carry their receive
    # time, so commands published before the abort can still arrive after it under load.

    def held():
        """Return the samples since the last command first arrived after the goal ended."""
        received = [s.q for s in recorder.samples if s.t > ended]
        if not received:
            return []
        return received[next(i for i, q in enumerate(received)
                             if np.array_equal(q, received[-1])):]
    assert wait_for(lambda: len(held()) > 50, 2.0)
    samples = held()
    assert all(np.array_equal(q, samples[0]) for q in samples)


def test_stale_state_does_not_abort_a_goal_off_the_real_robots_domain(ros, models, capfd):
    """Off domain 0 the staleness check is off: a goal runs to its timeout."""
    profile = profile_of('h12')
    robot_ = FollowingRobot(ros, profile, body_pose(profile, START['h12']), '/lowcmd')
    model = shared_model(ros, models, 'h12')
    left = PoseRecorder(ros, '/left_grasp_frame_pose')
    log = Log(capfd)
    # on the test domain the tick-lag check is off: its deadline is 0
    server = server_of(ros, model, CONFIGS['h12'])
    assert server.controller.max_state_lag_ms == 0.0
    assert wait_for(lambda: left.poses, 5.0)
    # a target out of reach keeps the IK stepping past the goal's 2 s; /lowstate stops at 0.5 s
    target = _shifted(_matrix(left.poses[-1]), (1.0, 0.0, 0.0))
    threading.Timer(0.5, robot_.stop_state).start()
    start = time.monotonic()
    status, result = GraspClient(ros).run(grasp_goal(left=target, timeout=2.0))
    # the goal ran to its timeout on the stale state: not reached, and no fault
    assert status == GoalStatus.STATUS_ABORTED
    assert result.outcome == grasp_outcome('NOT_REACHED'), result.message
    assert time.monotonic() - start >= 2.0
    assert log.lines('stale') == []


@pytest.mark.parametrize('robot', ROBOTS)
def test_planned_goals(ros, models, robot):
    """Planned goals to a posture's grasp-frame poses, back, then 5 cm forward, are reached.

    The targets are the grasp frames at the named configurations (the model's forward
    kinematics, every other joint as measured). The G1 goes out to t_pose; the H1-2, whose
    t_pose is past its planner frames' y limit, to arms_front_45.
    """
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot])
    threshold_linear = config['controller']['threshold_linear']
    robot_ = FollowingRobot(ros, profile, body_pose(profile, START[robot]), '/lowcmd')
    model = shared_model(ros, models, robot)
    poses = {arm: PoseRecorder(ros, f'/{arm}_grasp_frame_pose') for arm in ('left', 'right')}
    server_of(ros, model, CONFIGS[robot])
    assert wait_for(lambda: all(recorder.poses for recorder in poses.values()), 5.0)
    frames = [frame.name for frame in profile.grasp_frames]
    client = GraspClient(ros)

    def reached(targets):
        """Return whether both grasp frames are within threshold_linear of targets."""
        time.sleep(0.2)
        return all(np.linalg.norm(_matrix(poses[arm].poses[-1])[:3, 3] - targets[arm][:3, 3])
                   <= threshold_linear + 1e-4 for arm in targets)

    for name in ({'h12': 'arms_front_45', 'g1': 't_pose'}[robot], START[robot]):
        q = body_pose(profile, name)
        targets = {arm: model.get_frame_transformation_body(frame, q)
                   for arm, frame in zip(('left', 'right'), frames)}
        status, result = client.run(grasp_goal(plan=True, timeout=30.0, **targets), 40.0)
        assert status == GoalStatus.STATUS_SUCCEEDED and result.success, (name, result)
        assert reached(targets), name

    target = _shifted(_matrix(poses['left'].poses[-1]), (0.05, 0.0, 0.0))
    status, result = client.run(grasp_goal(left=target, plan=True, timeout=20.0))
    assert status == GoalStatus.STATUS_SUCCEEDED and result.success, result
    assert reached({'left': target})
    assert robot_.commands > 0


def test_planning_turned_off(ros, models, capfd, tmp_path):
    """planner.enabled: false refuses every planned goal before any IK solve."""
    profile = profile_of('g1')
    config_path = edited_config(tmp_path, 'g1', CONFIGS['g1'], planner={'enabled': False})
    config = config_of('g1', str(config_path))
    FollowingRobot(ros, profile, body_pose(profile, START['g1']), '/lowcmd')
    model = shared_model(ros, models, 'g1')
    left = PoseRecorder(ros, '/left_grasp_frame_pose')
    log = Log(capfd)
    recorder = CommandRecorder(ros, config['topics']['low_cmd'], profile)
    server_of(ros, model, config_path)
    assert wait_for(lambda: recorder.count > 0 and left.poses, 2.0)

    start = time.monotonic()
    target = _shifted(_matrix(left.poses[-1]), (0.05, 0.0, 0.0))
    status, result = GraspClient(ros).run(grasp_goal(left=target, plan=True))
    assert status == GoalStatus.STATUS_ABORTED and not result.success
    assert result.outcome == grasp_outcome('FAILED')
    assert result.message.startswith('planning failed: Planning is turned off'), result.message
    assert f'the controller config {config_path} sets planner.enabled: false' in result.message
    assert 'send the goal with plan=false' in result.message
    assert wait_for(lambda: log.lines('Planning is turned off'), 2.0)
    # nothing moved
    sent = [s.q for s in recorder.samples if s.t >= start]
    assert all(np.array_equal(q, sent[0]) for q in sent)


@pytest.mark.parametrize('robot', ROBOTS)
def test_goals_start_from_the_last_command(ros, models, robot):
    """A goal starts from the last command within the gap, past it from the measured pose."""
    profile = profile_of(robot)
    arms = ik_slots(profile)
    # the left elbow sits 0.05 rad below every command (a gravity sag inside the 0.1 rad
    # the seed accepts), the right elbow 0.2 rad below (an arm held off its command)
    sag = np.zeros(len(profile.body_joints))
    left_elbow = profile.body_joints.index('left_elbow_joint')
    sag[left_elbow] = 0.05
    robot_ = FollowingRobot(ros, profile, body_pose(profile, START[robot]), '/lowcmd', sag=sag)
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
    """A hold pinned at a position clip limit is not converged, naming the joint."""
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot])
    # one integral gain on every upper joint, the G1's waist included (its configs' is 0),
    # so the I-term pushes the left elbow whatever the configs say
    config['gains']['ki'][list(profile.upper_slots)] = 100.0
    limits = config['limits']['q_clip_limits']
    elbow = profile.body_joints.index('left_elbow_joint')
    # the left elbow is commanded at its clip limit and stops 0.5 mrad short of it
    sag = np.zeros(len(profile.body_joints))
    sag[elbow] = 0.0005
    q0 = body_pose(profile, START[robot])
    robot_ = FollowingRobot(ros, profile, q0, '/lowcmd', sag=sag)
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


@pytest.mark.parametrize('robot', ROBOTS)
def test_controller_keeps_the_ik_in_the_clip_range(ros, models, robot):
    """The controller narrows its model's and its planner's range to the config's clip range."""
    from golem_upper_ik.core.controller.frame_controller import FrameController
    profile = profile_of(robot)
    config = config_of(robot, CONFIGS[robot])
    clip = config['limits']['q_clip_limits']
    model = shared_model(ros, models, robot)
    mask = model.reduced_mask
    low = np.maximum(model.model_body.lowerPositionLimit, clip[:, 0])[mask]
    high = np.minimum(model.model_body.upperPositionLimit, clip[:, 1])[mask]
    RelayedRobot(ros, profile, body_pose(profile, START[robot]), '/lowcmd').start()
    controller = FrameController(ros.node('controller'), profile, config, robot_model=model,
                                 init=False)
    ros.on_close(controller.shutdown)
    np.testing.assert_array_equal(model.model_body_reduced.lowerPositionLimit, low)
    np.testing.assert_array_equal(model.model_body_reduced.upperPositionLimit, high)
    assert controller.ik_solver.configuration_reduced.model is model.model_body_reduced
    np.testing.assert_array_equal(controller.planner.q_limits, clip)


def test_execute_path_waits_for_the_follow_gate(ros, models):
    """execute_path is refused while nothing relays the commands."""
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
    """A path plays back at the control period, the checks' time inside it."""
    profile = profile_of('h12')
    config = config_of('h12', CONFIGS['h12'])
    model = shared_model(ros, models, 'h12')
    q0 = body_pose(profile, START['h12'])
    FollowingRobot(ros, profile, q0, '/lowcmd')
    from golem_upper_ik.core.controller.frame_controller import FrameController
    controller = FrameController(ros.node('controller'), profile, config, robot_model=model,
                                 init=False)
    ros.on_close(controller.shutdown)
    assert wait_for(lambda: controller.low_cmd_handler.following, 2.0)
    # each waypoint's checks take 10 ms: a step that slept the whole period after them would
    # stretch a 30-waypoint path by 0.3 s
    check = model.check_collision_free_reduced

    def slow_check(q_reduced):
        """Check collisions after 10 ms of work."""
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


def grasp_outcome(name):
    """Return the GraspFrameTarget result's outcome constant called name."""
    from custom_ros_messages.action import GraspFrameTarget
    return getattr(GraspFrameTarget.Result, name)


def grasp_goal(left=None, right=None, frame_id='pelvis', plan=False, timeout=0.0):
    """Return a GraspFrameTarget goal: each arm's 4x4 target in frame_id, None for none.

    timeout in s, 0 for controller.timeout.
    """
    from custom_ros_messages.action import GraspFrameTarget
    from golem_upper_ik.ros2.utility import matrix_to_pose
    goal = GraspFrameTarget.Goal()
    for arm, target in (('left', left), ('right', right)):
        if target is not None:
            stamped = getattr(goal, arm)
            stamped.header.frame_id = frame_id
            stamped.pose = matrix_to_pose(target)
    goal.plan = plan
    goal.timeout.sec = int(timeout)
    goal.timeout.nanosec = int((timeout - int(timeout)) * 1e9)
    return goal


class GraspClient:
    """Send /grasp_frame_target goals, keeping every feedback message."""

    def __init__(self, ros):
        """Wait for the action server."""
        from custom_ros_messages.action import GraspFrameTarget
        from rclpy.action import ActionClient
        self.client = ActionClient(ros.node('grasp_client'), GraspFrameTarget,
                                   '/grasp_frame_target')
        assert self.client.wait_for_server(timeout_sec=10.0), 'no /grasp_frame_target server'
        self.feedback = []

    def send(self, goal):
        """Send goal; return its handle, once accepted."""
        sent = self.client.send_goal_async(
            goal, feedback_callback=lambda message: self.feedback.append(message.feedback))
        assert wait_for(sent.done, 10.0), '/grasp_frame_target did not answer the goal'
        handle = sent.result()
        assert handle.accepted, '/grasp_frame_target refused the goal'
        return handle

    @staticmethod
    def outcome(handle, timeout=30.0):
        """Wait for a sent goal's outcome; return (status, result)."""
        done = handle.get_result_async()
        assert wait_for(done.done, timeout), f'no result within {timeout} s'
        return done.result().status, done.result().result

    def run(self, goal, timeout=30.0):
        """Send goal and return (status, result) once it ends."""
        return self.outcome(self.send(goal), timeout)


def _angle(first, second):
    """Return the angle between two 4x4 transformations' rotations, rad."""
    cosine = (np.trace(first[:3, :3].T @ second[:3, :3]) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _shifted(pose, position=(0.0, 0.0, 0.0), rotation=np.eye(3)):
    """Return 4x4 pose moved by position (m, pelvis axes) and turned by rotation about itself."""
    moved = np.array(pose, dtype=float)
    moved[:3, :3] = rotation @ moved[:3, :3]
    moved[:3, 3] += position
    return moved


def grasp_server(ros, models, robot, config=None):
    """Start frame_task_server on robot at START, behind a FollowingRobot; return its parts.

    Returns (robot, model, server, {arm: PoseRecorder of /<arm>_grasp_frame_pose}), once
    both grasp poses have come.
    """
    profile = profile_of(robot)
    following = FollowingRobot(ros, profile, body_pose(profile, START[robot]), '/lowcmd')
    model = shared_model(ros, models, robot)
    poses = {arm: PoseRecorder(ros, f'/{arm}_grasp_frame_pose') for arm in ('left', 'right')}
    server = server_of(ros, model, CONFIGS[robot] if config is None else config)
    assert wait_for(lambda: all(recorder.poses for recorder in poses.values()), 5.0)
    return following, model, server, poses


@pytest.mark.parametrize('robot', ROBOTS)
def test_grasp_frame_target_moves_one_arm(ros, models, robot):
    """A pelvis-frame target moves its arm's grasp frame there; the other arm stays put.

    The goal succeeds with its errors under the thresholds, the arm without a target reports
    0, the grasp pose topics give the measured grasp frames, and the arms hold afterwards.
    """
    config = config_of(robot, CONFIGS[robot])
    threshold_linear = config['controller']['threshold_linear']
    threshold_angular = config['controller']['threshold_angular']
    following, model, _, poses = grasp_server(ros, models, robot)
    frames = [frame.name for frame in model.profile.grasp_frames]
    # the topics give the grasp frames' poses at the measured body joints
    measured = following.measured()
    for arm, frame in zip(('left', 'right'), frames):
        np.testing.assert_allclose(_matrix(poses[arm].poses[-1]),
                                   model.get_frame_transformation_body(frame, measured),
                                   atol=1e-6)
        assert poses[arm].poses[-1].header.frame_id == 'pelvis'
    start = {arm: _matrix(poses[arm].poses[-1]) for arm in ('left', 'right')}
    target = _shifted(start['left'], (0.03, 0.0, 0.02), _rotation('z', 0.1))

    client = GraspClient(ros)
    status, result = client.run(grasp_goal(left=target))
    assert status == GoalStatus.STATUS_SUCCEEDED and result.success, result.message
    assert result.outcome == grasp_outcome('REACHED')
    assert result.message.startswith('reached (left '), result.message
    last = client.feedback[-1]
    assert last.left_error_m < threshold_linear and last.left_error_rad < threshold_angular
    assert last.right_error_m == 0.0 and last.right_error_rad == 0.0
    time.sleep(0.2)
    reached = _matrix(poses['left'].poses[-1])
    assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) <= threshold_linear + 1e-4
    assert _angle(reached, target) <= threshold_angular + 1e-4
    assert np.linalg.norm(_matrix(poses['right'].poses[-1])[:3, 3] - start['right'][:3, 3]) < 1e-3
    # the arms hold the goal's last command
    held = following.measured()
    time.sleep(0.3)
    np.testing.assert_array_equal(following.measured(), held)


# the named postures whose grasp-frame poses a loaded hold must still reach: home, the
# H1-2's prep and its staging postures (the wrist yaw 6 mrad inside its clip), the G1's
# arms_front_45
REACHED_POSTURES = {'h12': ('home', 'prep', 'staging_left', 'staging_right'),
                    'g1': ('home', 'arms_front_45')}


@pytest.mark.parametrize('robot', ROBOTS)
def test_loaded_holds_reach_the_named_postures(ros, models, robot):
    """Goals to the named postures' poses end REACHED with a load the hold must cancel.

    Every IK joint of the targeted arm settles 0.3 N*m over kp below its command, less the
    torque bias over kp (ros_harness.Spring): the hold integrates the bias that brings each
    joint onto its command, near a clip (the H1-2's staging wrist yaw, 6 mrad inside its
    clip) as away from one, within the bias bound, and each goal ends REACHED.
    """
    config = config_of(robot, CONFIGS[robot])
    kp = np.asarray(config['gains']['kp'], dtype=np.float64)
    following, model, server, _ = grasp_server(ros, models, robot)
    controller = server.controller
    profile = model.profile
    client = GraspClient(ros)
    for arm in ('left', 'right'):
        joints = controller.arm_joint_ids([arm])
        load = np.zeros(len(profile.body_joints))
        load[joints] = 0.3
        spring = Spring(ros, following, controller, joints, load, kp)
        for name in REACHED_POSTURES[robot]:
            q = body_pose(profile, name)
            frame = [f.name for f in profile.grasp_frames][('left', 'right').index(arm)]
            target = model.get_frame_transformation_body(frame, q)
            # the real G1's config moves the arms at 0.15 m/s: 6 to 7 s to arms_front_45
            status, result = client.run(grasp_goal(**{arm: target}, timeout=20.0))
            assert status == GoalStatus.STATUS_SUCCEEDED and result.success, (arm, name,
                                                                                result.message)
        spring.stop()
        following.set_sag(np.zeros(len(profile.body_joints)))


def _slot_commands(following, slots):
    """Return the newest command's q and tau on slots, as two arrays."""
    motors = following.last_command[1].motor_cmd
    return (np.array([motors[slot].q for slot in slots]),
            np.array([motors[slot].tau for slot in slots]))


def test_grasp_frame_target_leaves_the_other_arm_and_the_torso_alone(ros, models):
    """A one-arm goal leaves the other arm's and the torso's command q and torque bias alone.

    The H1-2's hold integrates a torque bias (gains.ki) toward the held command; the other arm
    and the torso sag 0.01 rad below theirs here, and a bias integrated on them would lift
    them. The server starts without a bias, so their torque stays the gravity torque.
    """
    profile = profile_of('h12')
    following, model, _, poses = grasp_server(ros, models, 'h12')
    others = [profile.body_joints.index(joint) for joint in profile.enabled_joints
              if joint.startswith('right_')] + [profile.body_joints.index('torso_joint')]
    sag = np.zeros(len(profile.body_joints))
    sag[others] = 0.01
    following.set_sag(sag)
    time.sleep(0.3)
    q_before, _ = _slot_commands(following, others)
    target = _shifted(_matrix(poses['left'].poses[-1]), (0.03, 0.0, 0.02))
    status, result = GraspClient(ros).run(grasp_goal(left=target))
    assert status == GoalStatus.STATUS_SUCCEEDED and result.success, result.message
    time.sleep(0.1)
    q_after, tau_after = _slot_commands(following, others)
    gravity = model.dynamics.get_gravity_compensation(following.measured())[others]
    np.testing.assert_array_equal(q_after, q_before)
    # a bias integrated over the 0.2 s settling alone would be more than 1 N*m; float32 on
    # the wire
    np.testing.assert_allclose(tau_after, gravity, atol=1e-3)


@pytest.mark.parametrize('robot', ROBOTS)
def test_stop_joints_commands_the_measured_pose_plus_the_resting_offset(ros, models, robot):
    """stop_joints commands its joints at their measured positions plus their resting offset.

    The left arm rests 0.02 rad below its command here. Without a reference the stop command
    is the measured pose; with stop_reference taken at rest (no bias fading since) it is the
    measured pose plus those 0.02 rad, where the arm rests. The other joints keep their
    command.
    """
    profile = profile_of(robot)
    following, model, server, _ = grasp_server(ros, models, robot)
    controller = server.controller
    left = controller.arm_joint_ids(['left'])
    others = [slot for slot in profile.upper_slots if slot not in left]
    sag = np.zeros(len(profile.body_joints))
    sag[left] = 0.02
    following.set_sag(sag)
    command = controller.low_cmd_handler.q_cmd
    assert wait_for(lambda: np.allclose(controller.robot_model.state['q'][left],
                                        command[left] - 0.02, atol=1e-5), 2.0)
    reference = controller.stop_reference(left)
    np.testing.assert_allclose(reference[0], 0.02, atol=1e-5)
    measured = np.array(controller.robot_model.state['q'])

    controller.stop_joints(left, reference)
    after = controller.low_cmd_handler.q_cmd
    np.testing.assert_allclose(after[left], command[left], atol=1e-5)
    np.testing.assert_array_equal(after[others], command[others])

    controller.stop_joints(left)
    after = controller.low_cmd_handler.q_cmd
    np.testing.assert_allclose(after[left], measured[left], atol=1e-6)
    np.testing.assert_array_equal(after[others], command[others])


@pytest.mark.parametrize('robot', ROBOTS)
def test_planned_grasp_frame_target(ros, models, robot):
    """A planned goal moves both grasp frames to their targets along a plan."""
    config = config_of(robot, CONFIGS[robot])
    threshold_linear = config['controller']['threshold_linear']
    _, _, _, poses = grasp_server(ros, models, robot)
    targets = {arm: _shifted(_matrix(poses[arm].poses[-1]), (0.04, 0.0, 0.0))
               for arm in ('left', 'right')}
    status, result = GraspClient(ros).run(grasp_goal(plan=True, timeout=20.0, **targets))
    assert status == GoalStatus.STATUS_SUCCEEDED and result.success, result.message
    time.sleep(0.2)
    for arm in ('left', 'right'):
        reached = _matrix(poses[arm].poses[-1])
        assert np.linalg.norm(reached[:3, 3] - targets[arm][:3, 3]) <= threshold_linear + 1e-4


def _world(translation, yaw):
    """Return a 4x4 pose of the pelvis in a world frame: translation (m), yaw (rad)."""
    pose = np.eye(4)
    pose[:3, :3] = _rotation('z', yaw)
    pose[:3, 3] = translation
    return pose


def _transform(parent, child, pose, stamp):
    """Return a TransformStamped of child's 4x4 pose in parent, stamped stamp."""
    from geometry_msgs.msg import TransformStamped
    transform = TransformStamped()
    transform.header.frame_id = parent
    transform.header.stamp = stamp
    transform.child_frame_id = child
    transform.transform.translation.x, transform.transform.translation.y, \
        transform.transform.translation.z = (float(value) for value in pose[:3, 3])
    w, x, y, z = _quaternion_wxyz(pose[:3, :3])
    transform.transform.rotation.w, transform.transform.rotation.x = float(w), float(x)
    transform.transform.rotation.y, transform.transform.rotation.z = float(y), float(z)
    return transform


def test_grasp_frame_target_in_a_world_frame(ros, models):
    """A world-frame target is resolved through TF into the pelvis frame when the goal starts."""
    from tf2_ros import StaticTransformBroadcaster
    config = config_of('h12', CONFIGS['h12'])
    _, _, _, poses = grasp_server(ros, models, 'h12')
    pelvis_in_world = _world((1.0, 2.0, 0.5), 0.7)
    node = ros.node('world')
    broadcaster = StaticTransformBroadcaster(node)
    broadcaster.sendTransform(_transform('test_world', 'pelvis', pelvis_in_world,
                                         node.get_clock().now().to_msg()))
    target = _shifted(_matrix(poses['right'].poses[-1]), (0.03, -0.02, 0.0))
    status, result = GraspClient(ros).run(
        grasp_goal(right=pelvis_in_world @ target, frame_id='test_world'))
    assert status == GoalStatus.STATUS_SUCCEEDED and result.success, result.message
    time.sleep(0.2)
    reached = _matrix(poses['right'].poses[-1])
    assert (np.linalg.norm(reached[:3, 3] - target[:3, 3])
            <= config['controller']['threshold_linear'] + 1e-4)


def test_grasp_frame_target_refuses_a_missing_or_stale_transform(ros, models):
    """A target in a frame TF does not give, or gives only old, aborts saying which."""
    from golem_upper_ik.ros2.frame_task_server import TF_MAX_AGE_S, TF_WAIT_S
    from tf2_ros import TransformBroadcaster
    import rclpy.duration
    following, _, _, poses = grasp_server(ros, models, 'h12')
    held = following.measured()
    client = GraspClient(ros)
    target = _matrix(poses['left'].poses[-1])

    start = time.monotonic()
    status, result = client.run(grasp_goal(left=target, frame_id='nowhere'))
    assert status == GoalStatus.STATUS_ABORTED and not result.success
    assert result.outcome == grasp_outcome('FAILED')
    assert result.message.startswith('no transform from nowhere to pelvis'), result.message
    assert TF_WAIT_S <= time.monotonic() - start < TF_WAIT_S + 1.0

    # /tf two seconds behind, as from a stopped odometry
    node = ros.node('old_world')
    broadcaster = TransformBroadcaster(node)
    stop = threading.Event()

    def publish():
        """Publish old_world -> pelvis stamped 2 s ago until stopped."""
        while not stop.is_set():
            stamp = node.get_clock().now() - rclpy.duration.Duration(seconds=2.0)
            broadcaster.sendTransform(_transform('old_world', 'pelvis', np.eye(4),
                                                 stamp.to_msg()))
            time.sleep(0.02)
    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    try:
        status, result = client.run(grasp_goal(left=target, frame_id='old_world'))
    finally:
        stop.set()
        thread.join(timeout=1.0)
    assert status == GoalStatus.STATUS_ABORTED and not result.success
    assert result.message.startswith(
        'the newest transform from old_world to pelvis is 2.0'), result.message
    assert f'(> {TF_MAX_AGE_S} s)' in result.message
    np.testing.assert_array_equal(following.measured(), held)


def test_world_frame_goals_keep_the_server_serving(ros, models):
    """World-frame goals under a busy /tf leave the server serving every later goal.

    Each world-frame goal listens to TF and then stops listening; that must not stop the
    executor that serves the controller's callbacks (it did once, InvalidHandle, live on the
    H1-2 sim, whose /tf runs at the joint-state rate).
    """
    from tf2_ros import TransformBroadcaster
    _, _, _, poses = grasp_server(ros, models, 'h12')
    pelvis_in_world = _world((1.0, 2.0, 0.5), 0.7)
    node = ros.node('busy_world')
    broadcaster = TransformBroadcaster(node)
    stop = threading.Event()

    def publish():
        """Publish busy_world -> pelvis, stamped now, at about 500 Hz until stopped."""
        while not stop.is_set():
            broadcaster.sendTransform(_transform('busy_world', 'pelvis', pelvis_in_world,
                                                 node.get_clock().now().to_msg()))
            time.sleep(0.002)
    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    client = GraspClient(ros)
    try:
        for _ in range(8):
            target = _matrix(poses['left'].poses[-1])
            status, result = client.run(grasp_goal(left=pelvis_in_world @ target,
                                                   frame_id='busy_world'))
            assert status == GoalStatus.STATUS_SUCCEEDED, result.message
    finally:
        stop.set()
        thread.join(timeout=1.0)
    status, result = client.run(grasp_goal(left=_matrix(poses['left'].poses[-1])))
    assert status == GoalStatus.STATUS_SUCCEEDED, result.message


def test_grasp_frame_target_needs_a_target(ros, models):
    """A goal with no target aborts at once, and nothing moves."""
    following, _, _, _ = grasp_server(ros, models, 'g1')
    held = following.measured()
    status, result = GraspClient(ros).run(grasp_goal())
    assert status == GoalStatus.STATUS_ABORTED and not result.success
    assert result.outcome == grasp_outcome('FAILED')
    assert result.message.startswith('neither arm has a target'), result.message
    np.testing.assert_array_equal(following.measured(), held)


def test_grasp_frame_target_times_out(ros, models):
    """A target out of reach aborts at the goal's timeout, s of robot time, saying so."""
    _, _, _, poses = grasp_server(ros, models, 'g1')
    target = _shifted(_matrix(poses['left'].poses[-1]), (1.0, 0.0, 0.0))
    start = time.monotonic()
    status, result = GraspClient(ros).run(grasp_goal(left=target, timeout=1.0))
    assert status == GoalStatus.STATUS_ABORTED and not result.success
    assert result.outcome == grasp_outcome('NOT_REACHED')
    assert result.message.startswith('timed out after 1.0 s'), result.message
    assert 1.0 <= time.monotonic() - start < 2.0


@pytest.mark.parametrize('plan', [False, True])
def test_canceled_grasp_frame_target_holds_where_it_is(ros, models, tmp_path, plan):
    """A cancel ends a direct move or a planned path where it is, and the arms hold there."""
    # slow enough that the move or the path is still under way at the cancel
    planner = {'moving_speed': 0.02}
    config = edited_config(tmp_path, 'h12', CONFIGS['h12'], planner=planner, v_lim=0.02)
    following, _, _, poses = grasp_server(ros, models, 'h12', config=config)
    start = _matrix(poses['left'].poses[-1])
    target = _shifted(start, (0.05, 0.0, 0.0))
    client = GraspClient(ros)
    handle = client.send(grasp_goal(left=target, plan=plan, timeout=30.0))
    begun = following.measured()
    # a first plan waits for the planner process to build its robot model
    assert wait_for(lambda: np.max(np.abs(following.measured() - begun)) > 0.005, 60.0)
    time.sleep(0.5)
    handle.cancel_goal_async()
    status, result = client.outcome(handle)
    assert status == GoalStatus.STATUS_CANCELED and not result.success
    assert result.outcome == grasp_outcome('CANCELED')
    expected = 'canceled during the path' if plan else 'canceled; the arms hold where they are'
    assert result.message.startswith(expected), result.message
    # the last step's command may still be on its way at the result
    time.sleep(0.1)
    held = following.measured()
    time.sleep(0.3)
    np.testing.assert_array_equal(following.measured(), held)
    moved = np.linalg.norm(_matrix(poses['left'].poses[-1])[:3, 3] - start[:3, 3])
    assert 0.002 < moved < 0.045, moved


def executable(name):
    """Return the installed console script name."""
    from ament_index_python.packages import get_package_prefix
    return str(Path(get_package_prefix('golem_upper_ik')) / 'lib' / 'golem_upper_ik' / name)


def node_env():
    """Return the environment for a node process on the test domain."""
    return dict(os.environ, ROS_DOMAIN_ID=str(TEST_DOMAIN), RCUTILS_COLORIZED_OUTPUT='0')


def run_node(name, *params, timeout=30.0):
    """Run a console script with -p params until it exits; return (exit code, output)."""
    args = [executable(name), '--ros-args']
    for param in params:
        args += ['-p', param]
    done = subprocess.run(args, env=node_env(), capture_output=True, text=True,
                          timeout=timeout)
    return done.returncode, done.stdout + done.stderr


class NodeProcess:
    """Run an installed console script as a process, collecting its output."""
    def __init__(self, name, *params):
        """Start the script name with -p params."""
        args = [executable(name), '--ros-args']
        for param in params:
            args += ['-p', param]
        self.proc = subprocess.Popen(args, env=node_env(), text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
        self.lines = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        """Collect the output lines."""
        for line in self.proc.stdout:
            self.lines.append(line.rstrip('\n'))

    def output(self):
        """Return the output so far."""
        return '\n'.join(self.lines)

    def interrupt(self, signals=(signal.SIGINT,), timeout=15.0):
        """Send signals one after another; return the exit code."""
        for signum in signals:
            self.proc.send_signal(signum)
        return self.proc.wait(timeout=timeout)

    def wait(self, timeout):
        """Wait for the exit; return the code."""
        return self.proc.wait(timeout=timeout)

    def kill(self):
        """Kill the process if it still runs."""
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5.0)


@pytest.mark.parametrize('robot', ROBOTS)
def test_joint_state_publisher(ros, robot):
    """joint_state_publisher publishes the body joints' /lowstate positions at 100 Hz."""
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


def test_joint_state_publisher_reads_its_low_state_topic(ros):
    """joint_state_publisher takes the positions from low_state_topic when it is given."""
    from sensor_msgs.msg import JointState
    profile = profile_of('g1')
    fake = FakeRobot(ros, profile, topic='/other/lowstate')
    fake.q[:] = np.clip(0.02, profile.q_low, profile.q_high)
    fake.start(rate_hz=200.0)
    received = []
    ros.node('joint_states_recorder').create_subscription(
        JointState, '/joint_states', received.append, 100)
    process = NodeProcess('joint_state_publisher', 'robot:=g1',
                          'low_state_topic:=/other/lowstate')
    try:
        expected = fake.q.astype(np.float32)
        assert wait_for(lambda: received and np.array_equal(
            np.float32(received[-1].position), expected), 20.0), process.output()
    finally:
        assert process.interrupt() == 0, process.output()


@pytest.mark.parametrize('name, params, words', [
    ('frame_task_server', [], ['robot', 'h12', 'g1']),
    ('frame_task_server', ['robot:=h1'], ["unknown robot 'h1'"]),
    ('frame_task_server', ['robot:=h12'], ['robot h12', 'config parameter is required']),
    ('frame_task_server', ['robot:=h12', 'config:=g1_split'], ['robot h12', "'g1_split'"]),
    ('frame_task_server', ['robot:=h12', 'config:=/nonexistent/g1/g1.yaml'],
     ['robot h12', 'config directory of robot g1']),
    ('joint_state_publisher', [], ['robot', 'h12', 'g1']),
    ('joint_state_publisher', ['robot:=h1'], ["unknown robot 'h1'"]),
    ('joint_state_publisher', ['robot:=h12', 'low_state_topic:=no such topic'],
     ['no such topic']),
])
def test_nodes_refuse_bad_parameters(name, params, words):
    """Both nodes exit non-zero, saying why, on a bad robot, config or state topic."""
    code, output = run_node(name, *params)
    assert code != 0, output
    for word in words:
        assert word in output, output


def assert_zero_gain_last(robot, profile, before):
    """Check the last command came before `before`, the IK joints off and the G1's waist held."""
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
    """On SIGINT or SIGTERM, once or twice, the server exits 0 after its zero-gain command."""
    profile = profile_of('g1')
    robot_ = FollowingRobot(ros, profile, body_pose(profile, 'home'), '/lowcmd')
    left = PoseRecorder(ros, '/left_grasp_frame_pose')
    process = NodeProcess('frame_task_server', 'robot:=g1', f'config:={CONFIGS["g1"]}')
    try:
        # its grasp-frame poses go out once the start-up is done
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
    """A start-up never relayed exits 1 after the zero-gain command."""
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
    """Count the commands on /safety/lowcmd_upper_in and keep the last, with its time."""

    def __init__(self, ros, profile):
        """Subscribe, keeping every sample."""
        from unitree_hg.msg import LowCmd
        from ros_harness import RECORD_QOS
        self.node = ros.node('commands')
        self.count = 0
        self.last_command = None
        self.node.create_subscription(LowCmd, '/safety/lowcmd_upper_in', self._on_command,
                                      RECORD_QOS)

    def _on_command(self, msg):
        """Count msg and keep it as the last."""
        self.count += 1
        self.last_command = (time.monotonic(), msg)


# Threads frame_task_server may have that carry its process's name: the main thread, the
# executor's, its pool's, the command publisher's and the follow gate's. OpenBLAS's workers
# carry it too, one per core but one, unless the node pins BLAS to one thread
MAX_THREADS_NAMED_AS_PROCESS = 16


# runs of the rate tool test_frame_task_server_rate may make: its timing bounds are wall time
RATE_ATTEMPTS = 2


def _rate_misses(stats):
    """Return the wall-time bounds the rate tool's stats miss, as text (empty when none)."""
    misses = []
    for window in ('startup', 'goals'):
        if stats[window]['mean_hz'] < 495:
            misses.append(f"{window}: mean {stats[window]['mean_hz']} Hz < 495")
        if stats[window]['gaps_over_20ms'] > 3:
            misses.append(f"{window}: {stats[window]['gaps_over_20ms']} gaps over 20 ms > 3")
        if stats[window]['max_gap_ms'] > 50.0:
            misses.append(f"{window}: largest gap {stats[window]['max_gap_ms']} ms > 50")
    # the IK stepped at ctrl_hz throughout the goals
    if stats['goals']['control_steps_hz'] < 25.0:
        misses.append(f"goals: {stats['goals']['control_steps_hz']} control steps/s < 25")
    return misses


def test_frame_task_server_rate():
    """The H1-2's commands reach another process at 500 Hz through its start-up and goals.

    At most 3 gaps over 20 ms and none over 50 ms. The 99th percentile of the period is
    printed, not bounded: the IK, rclpy's handling of /lowstate at 500 Hz and the 25 Hz pose
    topics share the GIL with the publisher thread. BLAS runs one thread. The bounds are wall
    time on the steady clock, which a loaded host can miss once, so a second run may pass in
    the first's place (RATE_ATTEMPTS); the exit checks hold on every run.
    """
    env = dict(os.environ, ROS_DOMAIN_ID=str(TEST_DOMAIN))
    runs = []
    for _ in range(RATE_ATTEMPTS):
        done = subprocess.run([sys.executable, str(RATE_TOOL), '--robot', 'h12', '--seconds', '6'],
                              env=env, capture_output=True, text=True, timeout=180)
        assert done.returncode == 0, done.stdout + done.stderr
        stats = json.loads(done.stdout.strip().splitlines()[-1])
        print(json.dumps(stats))
        assert stats['server_threads_named_as_process'] <= MAX_THREADS_NAMED_AS_PROCESS, stats
        assert stats['server_exit_code'] == 0, stats
        assert stats['shutdown_zero_gain'] is True, stats
        misses = _rate_misses(stats)
        runs.append(misses)
        if not misses:
            break
    assert not runs[-1], runs


def _launch_description(launch_file):
    """Return the installed launch file's LaunchDescription."""
    import importlib.util
    from ament_index_python.packages import get_package_share_directory
    path = Path(get_package_share_directory('golem_upper_ik')) / 'launch' / launch_file
    spec = importlib.util.spec_from_file_location(launch_file.split('.')[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def _launched(launch_file, **args):
    """Return launch_file's declared argument names, its [(delay, Node)] and the context.

    Its entities run in order with args, as ros2 launch runs them, so a refusal raises before
    any node.
    """
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, TimerAction
    from launch_ros.actions import Node
    description = _launch_description(launch_file)
    context = LaunchContext()
    context.launch_configurations.update(args)
    started = []

    def walk(entities, delay):
        """Collect the Nodes among entities, adding timer periods to their delay."""
        for entity in entities:
            if isinstance(entity, Node):
                started.append((delay, entity))
            elif isinstance(entity, TimerAction):
                walk(entity.actions, delay + float(entity.period))
            else:
                walk(entity.visit(context) or [], delay)

    walk(description.entities, 0.0)
    declared = {entity.name for entity in description.entities
                if isinstance(entity, DeclareLaunchArgument)}
    return declared, started, context


def _parameters(node, context):
    """Return a launched Node's parameters, evaluated in context."""
    from launch_ros.utilities import evaluate_parameters
    parameters = {}
    for params in evaluate_parameters(context, node._Node__parameters):
        parameters.update(params)
    return parameters


def test_upper_ik_launch_starts_only_frame_task_server():
    """upper_ik.launch.py checks the pairing, then starts frame_task_server after start_delay."""
    from launch.actions import DeclareLaunchArgument, OpaqueFunction
    declared, started, context = _launched(
        'upper_ik.launch.py', robot='h12', config='safety_split',
        safety_config='relax_safety_split.yaml', sim='true', start_delay='1.5')
    assert declared == {'robot', 'config', 'safety_config', 'sim', 'use_sim_time',
                        'start_delay'}
    ((delay, node),) = started
    assert (node.node_package, node.node_executable, node._Node__node_name, delay) == (
        'golem_upper_ik', 'frame_task_server', 'frame_task_server', 1.5)
    assert _parameters(node, context) == {
        'robot': 'h12', 'config': 'safety_split', 'use_sim_time': False}
    # the pairing check runs first, before the controller starts
    entities = [entity for entity in _launch_description('upper_ik.launch.py').entities
                if not isinstance(entity, DeclareLaunchArgument)]
    assert isinstance(entities[0], OpaqueFunction)


@pytest.mark.parametrize('args, reason', [
    (dict(robot='h12', config='safety_full', safety_config='relax_safety_split', sim='true'),
     'safety config "relax_safety_split" runs with controller config "safety_split", not '
     '"safety_full"'),
    (dict(robot='g1', config='sim_split', safety_config='sim_split', sim='false'),
     'safety config "sim_split" is a sim config'),
    (dict(robot='g1', config='sim_split', safety_config='full', sim='true'),
     'safety config "full" is not one of the G1 pairs'),
])
def test_upper_ik_launch_refuses_a_pairing(args, reason):
    """upper_ik.launch.py refuses a bad pairing before any node starts."""
    import re
    with pytest.raises(RuntimeError, match='^upper_ik: not starting: ' + re.escape(reason)):
        _launched('upper_ik.launch.py', **args)


@pytest.mark.parametrize('robot', ROBOTS)
def test_robot_state_launch_starts_both_state_nodes(robot):
    """robot_state.launch.py starts both state nodes, on the robot's ROS URDF by default."""
    from golem_ros2_models.paths import urdf_path
    declared, started, context = _launched('robot_state.launch.py', robot=robot,
                                           use_sim_time='true')
    assert declared == {'robot', 'use_sim_time', 'urdf_file'}
    nodes = {node._Node__node_name: (delay, node) for delay, node in started}
    assert sorted(nodes) == ['joint_state_publisher', 'robot_state_publisher']
    delay, publisher = nodes['joint_state_publisher']
    assert (publisher.node_package, publisher.node_executable, delay) == (
        'golem_upper_ik', 'joint_state_publisher', 0.0)
    assert _parameters(publisher, context) == {'robot': robot, 'use_sim_time': True}
    delay, state = nodes['robot_state_publisher']
    assert (state.node_package, state.node_executable, delay) == (
        'robot_state_publisher', 'robot_state_publisher', 0.0)
    # the robot's ROS URDF by default
    assert _parameters(state, context) == {
        'robot_description': Path(urdf_path(robot, ros=True)).read_text(),
        'use_sim_time': True}


def test_robot_state_launch_reads_urdf_file(tmp_path):
    """robot_state.launch.py publishes urdf_file when given."""
    urdf = tmp_path / 'robot.urdf'
    urdf.write_text('<robot name="test"/>')
    _, started, context = _launched('robot_state.launch.py', robot='h12', urdf_file=str(urdf))
    (state,) = [node for _, node in started if node._Node__node_name == 'robot_state_publisher']
    assert _parameters(state, context) == {'robot_description': '<robot name="test"/>',
                                           'use_sim_time': False}
