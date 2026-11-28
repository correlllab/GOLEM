"""Check arm_planner_server: its request parsing, and the node serving /plan_arm_trajectory.

parse_request turns a PlanArmTrajectory request into ArmPlanner's arguments, in the robot's
joint order, and refuses a malformed one. The node, in this process on domain 93, serves a
pose goal's timed trajectory, refuses a malformed request and a robot without a
self-collision model (ERROR), and reports a goal it cannot reach. Last, the installed console
script and the package's launch file start the node on a robot and config.
"""
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import pytest

from custom_ros_messages.srv import PlanArmTrajectory
from golem_ompl.arm_planner import ArmPlanner, PlanRefused
from golem_ompl.planner_server import PLAN_SERVICE, parse_request
from golem_upper_ik.config import load_controller_config
from golem_upper_ik.profile import load_profile
from golem_upper_ik.ros2.utility import matrix_to_pose

TEST_DOMAIN = 93
CONFIGS = {'h12': 'safety_split', 'g1': 'sim_split'}
START = {'h12': 'prep', 'g1': 'home'}


@pytest.fixture(scope='module')
def planners():
    """Return planner(robot): the module's ArmPlanner of robot, built on first use."""
    built = {}

    def planner(robot):
        """Return robot's ArmPlanner on its sim config."""
        if robot not in built:
            built[robot] = ArmPlanner(load_profile(robot),
                                      load_controller_config(robot, CONFIGS[robot]))
        return built[robot]
    return planner


def body_q(planner, arms):
    """Return a body q: 0, the IK joints at the named config arms."""
    q = np.zeros(len(planner.profile.body_joints))
    q[planner.ik_model.reduced_mask] = planner.profile.named_configs[arms]
    return q


def make_request(planner, measured_q, targets=None, goal=None, limits=None, dt=1.0 / 30.0,
                 reverse=False):
    """Return a request: measured and start at measured_q, the goal, the range and dt.

    targets: {arm: 4x4 in the pelvis frame}; goal: {joint name: rad}; limits: (body joints,
    2), None for none; reverse: name the joints in reverse order.
    """
    request = PlanArmTrajectory.Request()
    body = list(planner.profile.body_joints)
    order = list(reversed(range(len(body)))) if reverse else list(range(len(body)))
    request.measured.name = [body[i] for i in order]
    request.measured.position = [float(measured_q[i]) for i in order]
    arm = list(planner.arm_joints)
    start = measured_q[planner.ik_model.reduced_mask]
    arm_order = list(reversed(range(len(arm)))) if reverse else list(range(len(arm)))
    request.start.name = [arm[i] for i in arm_order]
    request.start.position = [float(start[i]) for i in arm_order]
    if limits is not None:
        request.position_lower = [float(limits[i, 0]) for i in order]
        request.position_upper = [float(limits[i, 1]) for i in order]
    for side, target in (targets or {}).items():
        stamped = getattr(request, f'{side}_pose')
        stamped.header.frame_id = 'pelvis'
        stamped.pose = matrix_to_pose(target)
    if goal:
        request.goal_joints.name = list(goal)
        request.goal_joints.position = [float(value) for value in goal.values()]
    request.dt = dt
    return request


def targets_of(planner, q, shift=(0.05, 0.0, 0.0), arms=('left',)):
    """Return {arm: 4x4}: the arms' grasp frames at body q, shifted (m)."""
    out = {}
    for arm in arms:
        target = planner.ik_model.get_frame_transformation_body(
            planner.grasp_frames[arm], q).copy()
        target[:3, 3] += shift
        out[arm] = target
    return out


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_a_request_is_read_in_the_robots_joint_order(planners, robot):
    """Joints named in any order come back in body and IK order, the range too."""
    planner = planners(robot)
    q = body_q(planner, START[robot])
    q[0] = 0.01
    limits = np.asarray(planner.config['limits']['q_clip_limits'], dtype=float)
    parsed = parse_request(make_request(planner, q, targets=targets_of(planner, q),
                                        limits=limits, reverse=True), planner)
    np.testing.assert_array_equal(parsed['measured_q'], q)
    np.testing.assert_array_equal(parsed['start'], q[planner.ik_model.reduced_mask])
    np.testing.assert_array_equal(parsed['limits'], limits)
    assert list(parsed['targets']) == ['left']
    assert parsed['timing'] == (1.0 / 30.0, None, None, None)
    assert parsed['ik_timeout'] is None
    # no range: the URDF's limits
    parsed = parse_request(make_request(planner, q, targets=targets_of(planner, q)), planner)
    np.testing.assert_array_equal(parsed['limits'], planner.urdf_limits)


def test_a_joint_goal_moves_the_arms_it_names(planners):
    """A joint goal naming every right-arm joint moves the right arm; the left keeps its start."""
    planner = planners('h12')
    q = body_q(planner, 'prep')
    goal = {name: 0.1 for name in planner.arm_joint_names['right']}
    parsed = parse_request(make_request(planner, q, goal=goal), planner)
    assert parsed['arms'] == ['right']
    start = q[planner.ik_model.reduced_mask]
    for index, name in enumerate(planner.arm_joints):
        expected = 0.1 if name in goal else start[index]
        assert parsed['goal'][index] == expected


@pytest.mark.parametrize('edit, message', [
    (lambda r: (r.measured.name.pop(), r.measured.position.pop()),
     'measured is missing joints'),
    (lambda r: r.start.position.pop(), 'start names 14 joints but gives 13 positions'),
    (lambda r: r.measured.name.__setitem__(0, 'tail_joint'), 'not measured joints'),
    (lambda r: r.start.position.__setitem__(0, float('nan')), 'start has a position that is '
                                                              'not finite'),
    (lambda r: setattr(r, 'dt', 0.0), 'dt must be positive'),
    (lambda r: setattr(r, 'frame_speed', -1.0), 'frame_speed must be finite and not '
                                                'negative'),
    (lambda r: setattr(r, 'position_lower', [0.0]), 'one value per joint of measured.name'),
    (lambda r: setattr(r.left_pose.header, 'frame_id', 'map'), "in 'map', not in 'pelvis'"),
    (lambda r: [setattr(r.left_pose.pose.orientation, axis, 0.0) for axis in 'xyzw'],
     'zero quaternion'),
    (lambda r: setattr(r.left_pose.header, 'frame_id', ''), 'gives no goal'),
    (lambda r: (setattr(r.goal_joints, 'name', ['left_elbow_joint']),
                setattr(r.goal_joints, 'position', [0.1])), 'both a pose goal and a joint'),
])
def test_a_malformed_request_is_refused(planners, edit, message):
    """A malformed request is refused, saying why."""
    planner = planners('h12')
    q = body_q(planner, 'prep')
    request = make_request(planner, q, targets=targets_of(planner, q))
    edit(request)
    with pytest.raises(PlanRefused, match=message):
        parse_request(request, planner)


def test_a_joint_goal_must_name_every_joint_of_its_arms(planners):
    """A joint goal naming some of an arm's joints is refused."""
    planner = planners('h12')
    q = body_q(planner, 'prep')
    goal = {name: 0.1 for name in planner.arm_joint_names['right'][:3]}
    with pytest.raises(PlanRefused, match='goal_joints is missing joints'):
        parse_request(make_request(planner, q, goal=goal), planner)


class _Ros:
    """Hold an rclpy context on TEST_DOMAIN with an executor spinning its nodes."""

    def __init__(self):
        """Start the context and the executor's thread."""
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        self.context = rclpy.Context()
        rclpy.init(context=self.context, domain_id=TEST_DOMAIN)
        self.executor = MultiThreadedExecutor(context=self.context)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def add(self, node):
        """Spin node."""
        self.executor.add_node(node)
        return node

    def close(self):
        """Stop the executor and the context."""
        import rclpy
        self.executor.shutdown(timeout_sec=1.0)
        rclpy.try_shutdown(context=self.context)


@pytest.fixture
def ros():
    """Yield a ROS context on domain 93; close it after."""
    harness = _Ros()
    try:
        yield harness
    finally:
        harness.close()


def server_of(ros, robot, monkeypatch=None, profile=None):
    """Return a started PlannerServer of robot on its sim config in ros."""
    from rclpy.parameter import Parameter
    from golem_ompl import planner_server
    if profile is not None:
        monkeypatch.setattr(planner_server, 'load_profile', lambda name: profile)
    server = planner_server.PlannerServer(
        context=ros.context,
        parameter_overrides=[Parameter('robot', value=robot),
                             Parameter('config', value=CONFIGS[robot])])
    server.start()
    ros.add(server)
    return server


def call(ros, request, timeout=60.0):
    """Call /plan_arm_trajectory with request and return the response."""
    from rclpy.node import Node
    node = ros.add(Node('plan_client', context=ros.context))
    client = node.create_client(PlanArmTrajectory, PLAN_SERVICE)
    assert client.wait_for_service(timeout_sec=10.0), 'no /plan_arm_trajectory'
    future = client.call_async(request)
    deadline = time.monotonic() + timeout
    while not future.done():
        assert time.monotonic() < deadline, 'no response'
        time.sleep(0.01)
    return future.result()


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_the_node_serves_a_timed_trajectory(ros, planners, robot):
    """A pose goal comes back as a trajectory in the IK joints, one point per dt."""
    server = server_of(ros, robot)
    planner = planners(robot)
    q = body_q(planner, START[robot])
    dt = 1.0 / 30.0
    response = call(ros, make_request(planner, q, targets=targets_of(planner, q), dt=dt))
    assert response.outcome == PlanArmTrajectory.Response.PLANNED, response.message
    assert response.success and response.message == 'direct path'
    assert response.ik_seed == 'current'
    trajectory = response.trajectory
    assert tuple(trajectory.joint_names) == planner.arm_joints
    assert len(trajectory.points) >= 2
    for index, point in enumerate(trajectory.points):
        seconds = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
        assert seconds == pytest.approx(index * dt, abs=1e-9)
    np.testing.assert_array_equal(trajectory.points[0].positions,
                                  q[planner.ik_model.reduced_mask])
    np.testing.assert_array_equal(response.goal.position, trajectory.points[-1].positions)
    assert server.planner is not planner


def test_the_node_refuses_a_malformed_request(ros, planners):
    """A malformed request comes back as ERROR, saying why, and the node goes on serving."""
    server_of(ros, 'g1')
    planner = planners('g1')
    q = body_q(planner, 'home')
    request = make_request(planner, q, targets=targets_of(planner, q))
    request.dt = 0.0
    response = call(ros, request)
    assert response.outcome == PlanArmTrajectory.Response.ERROR and not response.success
    assert response.message.startswith('dt must be positive'), response.message
    assert response.trajectory.points == []
    request.dt = 1.0 / 30.0
    assert call(ros, request).success


def test_the_node_reports_a_target_out_of_reach(ros, planners):
    """A pose goal no IK seed reaches comes back as NO_IK_SOLUTION."""
    server_of(ros, 'g1')
    planner = planners('g1')
    q = body_q(planner, 'home')
    request = make_request(planner, q, targets=targets_of(planner, q, shift=(1.0, 0.0, 0.0)))
    request.ik_timeout = 0.2
    response = call(ros, request)
    assert response.outcome == PlanArmTrajectory.Response.NO_IK_SOLUTION, response.message
    assert response.message == 'IK failed to resolve current targets'


def test_a_robot_without_a_self_collision_model_refuses_every_request(ros, planners,
                                                                      monkeypatch):
    """Without a self-collision model the node serves, and refuses every request."""
    server = server_of(ros, 'g1', monkeypatch,
                       profile=replace(load_profile('g1'), collision=None))
    assert server.planner is None
    planner = planners('g1')
    q = body_q(planner, 'home')
    response = call(ros, make_request(planner, q, targets=targets_of(planner, q)))
    assert response.outcome == PlanArmTrajectory.Response.ERROR
    assert response.message.startswith('Planning is unavailable'), response.message


def _env():
    """Return the environment for a node process on the test domain."""
    return dict(os.environ, ROS_DOMAIN_ID=str(TEST_DOMAIN), RCUTILS_COLORIZED_OUTPUT='0',
                PYTHONUNBUFFERED='1')


def _wait_for_line(process, text, timeout):
    """Return the output of process up to the first line holding text; fail at timeout."""
    lines = []
    deadline = time.monotonic() + timeout
    os.set_blocking(process.stdout.fileno(), False)
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if line:
            lines.append(line)
            if text in line:
                return ''.join(lines)
        elif process.poll() is not None:
            break
        else:
            time.sleep(0.05)
    raise AssertionError(f'no {text!r} within {timeout} s:\n' + ''.join(lines))


@pytest.mark.parametrize('how', ['script', 'launch'])
def test_the_installed_node_starts_and_stops_on_a_signal(how):
    """The console script and planner.launch.py start the node, ready; SIGINT stops it."""
    import signal
    from ament_index_python.packages import get_package_prefix, get_package_share_directory
    if how == 'script':
        command = [str(Path(get_package_prefix('golem_ompl')) / 'lib' / 'golem_ompl'
                       / 'arm_planner_server'),
                   '--ros-args', '-p', 'robot:=h12', '-p', 'config:=safety_split']
    else:
        command = ['ros2', 'launch',
                   str(Path(get_package_share_directory('golem_ompl')) / 'launch'
                       / 'planner.launch.py'), 'robot:=h12', 'config:=safety_split']
    process = subprocess.Popen(command, env=_env(), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output = _wait_for_line(process, 'Planner server ready', 60.0)
        assert 'robot h12, config safety_split' in output
    finally:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    assert process.returncode == 0 or how == 'launch', process.returncode
