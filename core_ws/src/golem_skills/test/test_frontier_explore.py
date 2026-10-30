"""Frontier exploration skill tests on synthetic OccupancyGrids and over loopback DDS."""

import math
import os
import threading
import time

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from custom_ros_messages.action import SkillFrontierExplore
from geometry_msgs.msg import TransformStamped
from golem_skills.skills import frontier_explore as fe
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy import ndimage
from tf2_ros import TransformBroadcaster
import yaml

RES = 0.05
FOOTPRINT = fe.parse_footprint(fe.G1_FOOTPRINT)
TEST_DOMAIN = 97
DDS_CONFIG = """<CycloneDDS><Domain Id="any">
  <General>
    <Interfaces><NetworkInterface name="lo"/></Interfaces>
    <AllowMulticast>false</AllowMulticast>
  </General>
  <Discovery>
    <ParticipantIndex>auto</ParticipantIndex>
    <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
    <Peers><Peer address="127.0.0.1"/></Peers>
  </Discovery>
</Domain></CycloneDDS>
"""


def _use_loopback_dds(tmp_dir):
    path = os.path.join(tmp_dir, 'cyclonedds.xml')
    with open(path, 'w') as f:
        f.write(DDS_CONFIG)
    os.environ['CYCLONEDDS_URI'] = 'file://' + path
    os.environ['ROS_DOMAIN_ID'] = str(TEST_DOMAIN)
    os.environ['ROS_LOCALHOST_ONLY'] = '0'
    os.environ['RCUTILS_LOGGING_BUFFERED_STREAM'] = '1'


def _grid(data, origin=(0.0, 0.0), stamp=0.0):
    return fe.Grid(np.array(data, dtype=np.int16), RES, origin, stamp)


def _costmap(grid, inscribed=0.20):
    occupied = grid.data >= fe.OCCUPIED_FROM
    data = np.zeros_like(grid.data)
    data[ndimage.distance_transform_edt(~occupied) * RES <= inscribed + 1e-9] = (
        fe.COSTMAP_INSCRIBED
    )
    data[occupied] = fe.COSTMAP_LETHAL
    data[grid.data < 0] = -1
    return grid._replace(data=data)


def _empty_costmap(grid):
    return grid._replace(data=np.zeros_like(grid.data))


def _room(height, width, openings=()):
    data = np.zeros((height, width), np.int16)
    data[[0, -1], :] = 100
    data[:, [0, -1]] = 100
    for rows, cols in openings:
        data[rows, cols] = -1
    return _grid(data)


def _centre(row, col):
    return (col + 0.5) * RES, (row + 0.5) * RES


def test_grid_of_an_occupancy_grid():
    msg = OccupancyGrid()
    msg.header.stamp.sec, msg.header.stamp.nanosec = 12, 500_000_000
    msg.info.width, msg.info.height, msg.info.resolution = 3, 2, 0.05
    msg.info.origin.position.x, msg.info.origin.position.y = -1.0, 2.0
    msg.data = [0, 100, -1, 0, 0, 100]
    grid = fe.grid_of(msg)
    np.testing.assert_array_equal(grid.data, [[0, 100, -1], [0, 0, 100]])
    assert (grid.res, grid.origin, grid.stamp) == (pytest.approx(0.05), (-1.0, 2.0), 12.5)
    np.testing.assert_array_equal(
        fe.sample(grid, [-0.99, -0.94, -0.87, -1.01], [2.01, 2.01, 2.06, 2.01]),
        [0, 100, 100, -1],
    )


def test_frontiers_are_free_cells_next_to_unknown_but_not_next_to_a_wall():
    data = np.full((20, 30), -1)
    data[:, :15] = 0
    data[[0, -1], :15] = 100
    data[:, 0] = 100
    expected = np.zeros(data.shape, bool)
    expected[2:18, 14] = True
    np.testing.assert_array_equal(fe.frontier_mask(_grid(data)), expected)


def test_off_the_map_counts_as_unknown():
    expected = np.ones((5, 6), bool)
    expected[1:-1, 1:-1] = False
    np.testing.assert_array_equal(fe.frontier_mask(_grid(np.zeros((5, 6)))), expected)


def test_clusters_smaller_than_min_cells_are_left_out():
    room = _room(40, 60, openings=[(slice(5, 10), 59), (slice(20, 28), 59)])
    robot = (1.0, 1.0)
    found = fe.find_frontiers(room, _empty_costmap(room), robot, 6, FOOTPRINT)
    assert [f.size for f in found] == [8]
    found = fe.find_frontiers(room, _empty_costmap(room), robot, 5, FOOTPRINT)
    assert sorted(f.size for f in found) == [5, 8]


def test_the_goal_is_the_frontier_cell_nearest_the_centroid():
    room = _room(60, 80, openings=[(slice(15, 45), 79)])
    robot = (1.0, 1.475)
    [goal] = fe.find_frontiers(room, _costmap(room), robot, 6, FOOTPRINT)
    assert goal.size == 24
    assert (goal.x, goal.y) == pytest.approx(_centre(29, 78))
    assert goal.yaw == pytest.approx(0.0)


def test_the_goal_moves_where_the_footprint_fits():
    room = _room(60, 80, openings=[(slice(15, 45), 79)])
    costmap = _costmap(room)
    costmap.data[25, 76] = fe.COSTMAP_LETHAL
    robot = (1.0, 1.475)
    assert not fe.footprint_fits(costmap, *_centre(29, 78), 0.0, FOOTPRINT)
    [goal] = fe.find_frontiers(room, costmap, robot, 6, FOOTPRINT)
    assert (goal.x, goal.y) == pytest.approx(_centre(30, 78))
    assert fe.footprint_fits(costmap, goal.x, goal.y, goal.yaw, FOOTPRINT)


def test_a_gap_the_robot_does_not_fit_through_is_no_frontier():
    robot = (1.5, 1.5)
    room = _room(60, 60, openings=[(slice(27, 33), 59)])
    assert fe.find_frontiers(room, _costmap(room), robot, 1, FOOTPRINT) == []
    room = _room(60, 60, openings=[(slice(27, 34), 59)])
    assert fe.walkable_mask(room, _costmap(room))[30, 58]
    assert fe.find_frontiers(room, _costmap(room), robot, 1, FOOTPRINT) == []


def test_frontiers_the_robot_cannot_walk_to_are_left_out():
    room = _room(60, 80, openings=[(slice(15, 45), 0), (slice(15, 45), 79)])
    room.data[:, 40] = 100
    robot = (1.0, 1.5)
    found = fe.find_frontiers(room, _costmap(room), robot, 6, FOOTPRINT)
    assert len(found) == 1
    assert (found[0].x, found[0].y) == pytest.approx(_centre(29, 1))
    assert abs(found[0].yaw) == pytest.approx(math.pi, abs=0.05)
    room.data[20:40, 40] = 0
    found = fe.find_frontiers(room, _costmap(room), robot, 6, FOOTPRINT)
    assert sorted(round(f.x, 3) for f in found) == [0.075, 3.925]


def test_the_robot_snaps_to_walkable_space_or_there_are_no_frontiers():
    room = _room(60, 80, openings=[(slice(15, 45), 79)])
    costmap = _costmap(room)
    assert len(fe.find_frontiers(room, costmap, (0.1, 1.5), 6, FOOTPRINT)) == 1
    assert fe.find_frontiers(room, costmap, (-0.6, 1.5), 6, FOOTPRINT) is None
    lethal = costmap._replace(data=np.full_like(costmap.data, fe.COSTMAP_LETHAL))
    assert fe.find_frontiers(room, lethal, (1.0, 1.5), 6, FOOTPRINT) is None


def test_the_costmap_can_differ_from_the_map_in_extent():
    room = _room(60, 80, openings=[(slice(15, 45), 79)])
    costmap = _costmap(room)
    short = costmap._replace(data=costmap.data[:, :70])
    assert fe.find_frontiers(room, short, (1.0, 1.475), 6, FOOTPRINT) == []
    shifted = costmap._replace(
        data=np.pad(costmap.data, ((1, 0), (1, 0)), constant_values=-1),
        origin=(-RES, -RES),
    )
    [goal] = fe.find_frontiers(room, shifted, (1.0, 1.475), 6, FOOTPRINT)
    assert (goal.x, goal.y) == pytest.approx(_centre(29, 78))


def test_choose_the_nearest_not_blacklisted_and_not_too_near():
    frontiers = [fe.Frontier(x, 0.0, 0.0, 10) for x in (1.0, 2.0, 3.0, 6.0)]
    robot = (0.0, 0.0)
    goal, candidates = fe.choose(frontiers, robot, [], 0.6, 0.7)
    assert goal.x == 1.0 and len(candidates) == 4
    goal, candidates = fe.choose(frontiers, robot, [(1.4, 0.3)], 0.6, 2.5)
    assert goal.x == 3.0 and [f.x for f in candidates] == [3.0, 6.0]
    assert fe.choose(frontiers, robot, [(3.0, 0.0), (6.0, 0.5)], 0.6, 2.5) == (None, [])


def test_done_when_no_frontier_is_left():
    room = _room(60, 80)
    assert fe.find_frontiers(room, _costmap(room), (1.0, 1.5), 6, FOOTPRINT) == []
    room = _room(60, 80, openings=[(slice(20, 31), 79)])
    assert fe.find_frontiers(room, _costmap(room), (1.0, 1.5), 6, FOOTPRINT) == []
    found = fe.find_frontiers(room, _costmap(room), (3.6, 1.25), 5, FOOTPRINT)
    assert [f.size for f in found] == [5]
    assert fe.choose(found, (3.6, 1.25), [], 0.6, 0.7) == (None, [])


def test_a_goal_on_its_way_is_outdated_once_seen_or_blocked():
    room = _room(60, 80, openings=[(slice(15, 45), 79)])
    costmap = _costmap(room)
    [goal] = fe.find_frontiers(room, costmap, (1.0, 1.475), 6, FOOTPRINT)
    assert fe.outdated(room, costmap, goal, FOOTPRINT, 6) is None
    seen = room._replace(data=np.pad(room.data, ((0, 0), (0, 20))))
    seen.data[15:45, 79] = 0
    seen.data[[0, -1], 80:] = 100
    seen.data[:, -1] = 100
    assert fe.outdated(seen, _costmap(seen), goal, FOOTPRINT, 6) == 'seen'
    assert fe.outdated(room, costmap, goal, FOOTPRINT, 25) == 'seen'
    blocked = costmap._replace(data=costmap.data.copy())
    blocked.data[29, 77] = fe.COSTMAP_LETHAL
    assert fe.outdated(room, blocked, goal, FOOTPRINT, 6) == 'blocked'


def test_parse_footprint():
    np.testing.assert_allclose(
        fe.parse_footprint('[[1, 2], [3, 4.5], [0, 0]]'),
        [[1, 2], [3, 4.5], [0, 0]],
    )
    for bad in ('[[1, 2], [3, 4]]', '[1, 2, 3]', '[[1, 2, 3], [1, 2, 3], [1, 2, 3]]'):
        with pytest.raises(ValueError):
            fe.parse_footprint(bad)


def test_the_default_footprint_is_g1_nav2_yaml_s():
    nav2_config = os.path.join(
        get_package_share_directory('golem_slam'), 'config', 'g1_nav2.yaml')
    with open(nav2_config) as f:
        config = yaml.safe_load(f)
    for costmap in ('global_costmap', 'local_costmap'):
        footprint = config[costmap][costmap]['ros__parameters']['footprint']
        np.testing.assert_array_equal(fe.parse_footprint(footprint), FOOTPRINT)


class FakeStack(Node):
    """Fake slam_toolbox and Nav2 for DDS integration tests."""

    SEE = 1.5

    def __init__(self, world, start, trap=None, hold=False, slow=False, block=None):
        super().__init__('fake_stack')
        self.world, self.robot, self.trap, self.hold = world, start, trap, hold
        self.slow, self.block = slow, block
        self.known = np.zeros(world.data.shape, bool)
        self.goals, self.results = [], []
        self.lock = threading.Lock()
        latched = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', latched)
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, '/global_costmap/costmap', latched)
        self.tf = TransformBroadcaster(self)
        self._see()
        self.published = self.map()
        self.create_timer(0.05, self._send_tf)
        self.create_timer(0.5, self._publish_map)
        self.create_timer(0.2, self._publish_costmap)
        self.server = ActionServer(
            self, NavigateToPose, '/navigate_to_pose',
            execute_callback=self._navigate,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

    def _see(self, xy=None):
        x0, y0 = self.robot if xy is None else xy
        rows, cols = np.indices(self.known.shape)
        x, y = fe.cell_centres(self.world, rows, cols)
        self.known |= np.hypot(x - x0, y - y0) <= self.SEE

    def map(self):
        with self.lock:
            return self.world._replace(data=np.where(self.known, self.world.data, -1))

    def _msg(self, grid):
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = grid.res
        msg.info.height, msg.info.width = grid.data.shape
        msg.info.origin.position.x, msg.info.origin.position.y = grid.origin
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.data.astype(np.int8).ravel().tolist()
        return msg

    def _publish_map(self):
        grid = self.map()
        self.map_pub.publish(self._msg(grid))
        self.published = grid

    def _publish_costmap(self):
        self.costmap_pub.publish(self._msg(_costmap(self.published)))

    def _send_tf(self):
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = 'map', 'pelvis'
        t.header.stamp = self.get_clock().now().to_msg()
        with self.lock:
            t.transform.translation.x, t.transform.translation.y = self.robot
        t.transform.translation.z = 0.74
        t.transform.rotation.w = 1.0
        self.tf.sendTransform(t)

    def _navigate(self, gh):
        p = gh.request.pose.pose.position
        with self.lock:
            self.goals.append((p.x, p.y))
        if self.slow:
            with self.lock:
                self._see((p.x, p.y))
        blocked = self.block is not None and self.block(p.x, p.y)
        if blocked:
            with self.lock:
                row, col = int(p.y / RES), int(p.x / RES)
                self.world.data[row, col] = 100
                self.known[row, col] = True
        if self.hold or self.slow or blocked:
            end = time.time() + (3.0 if self.slow else 20.0)
            while not gh.is_cancel_requested and time.time() < end:
                time.sleep(0.02)
        if gh.is_cancel_requested:
            gh.canceled()
            status = GoalStatus.STATUS_CANCELED
        elif self.trap is not None and self.trap(p.x, p.y):
            time.sleep(0.1)
            gh.abort()
            status = GoalStatus.STATUS_ABORTED
        else:
            time.sleep(0.1)
            with self.lock:
                self.robot = (p.x, p.y)
                self._see()
            gh.succeed()
            status = GoalStatus.STATUS_SUCCEEDED
        with self.lock:
            self.results.append((p.x, p.y, status))
        return NavigateToPose.Result()


def _two_rooms():
    world = _room(80, 120)
    world.data[:, 60] = 100
    world.data[50:70, 60] = 0
    return world


@pytest.fixture
def ros(tmp_path):
    _use_loopback_dds(str(tmp_path))
    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=8)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    nodes = []

    def spin(*new_nodes):
        for node in new_nodes:
            executor.add_node(node)
            nodes.append(node)
        if not spinner.is_alive():
            spinner.start()

    yield spin
    executor.shutdown()
    for node in nodes:
        node.destroy_node()
    rclpy.shutdown()


def _until(condition, timeout):
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        time.sleep(0.05)
    return condition()


def _done(future, timeout):
    return _until(future.done, timeout)


def _explore(ros, fake, goal=None):
    client = Node('explore_client')
    action = ActionClient(client, SkillFrontierExplore, '/skill/frontier_explore')
    ros(fe.FrontierExplore(), fake, client)
    assert action.wait_for_server(timeout_sec=10.0)
    feedbacks = []
    sent = action.send_goal_async(
        goal or SkillFrontierExplore.Goal(),
        feedback_callback=lambda m: feedbacks.append(m.feedback),
    )
    assert _done(sent, 10.0) and sent.result().accepted
    return action, sent.result(), feedbacks


def _reachable(world, start):
    labels, _ = ndimage.label(world.data == 0, structure=fe.EIGHT)
    col, row = (int(v / RES) for v in start)
    return labels == labels[row, col]


def test_node_explores_both_rooms_and_says_when_it_is_done(ros):
    world = _two_rooms()
    fake = FakeStack(world, (1.0, 1.0))
    _, handle, feedbacks = _explore(ros, fake)
    result = handle.get_result_async()
    assert _done(result, 60.0), 'no result in 60 s'
    assert result.result().status == GoalStatus.STATUS_SUCCEEDED, result.result().result.message
    reached = result.result().result.goals_reached
    assert reached == len(fake.goals) >= 3
    assert (
        f'map explored: {reached} of {reached} frontier goal(s) reached ({reached} by Nav2, '
        '0 seen on the way), 0 failed' == result.result().result.message
    )
    reachable = _reachable(world, (1.0, 1.0))
    seen = fake.known[reachable].mean()
    assert seen > 0.97, seen
    assert any(x > 3.05 for x, _ in fake.goals)
    assert _until(lambda: feedbacks and feedbacks[-1].phase == 'done', 2.0)
    assert 'navigating' in [f.phase for f in feedbacks]
    assert feedbacks[-1].frontiers_remaining == 0


def test_node_blacklists_a_goal_that_fails(ros):
    world = _two_rooms()
    fake = FakeStack(world, (1.0, 1.0), trap=lambda x, y: x > 2.7)
    _, handle, _ = _explore(ros, fake, SkillFrontierExplore.Goal(blacklist_radius=0.6))
    result = handle.get_result_async()
    assert _done(result, 60.0), 'no result in 60 s'
    assert result.result().status == GoalStatus.STATUS_SUCCEEDED, result.result().result.message
    results = list(fake.results)
    failed = [(x, y) for x, y, status in results if status == GoalStatus.STATUS_ABORTED]
    assert failed, results
    for i, (x, y, _) in enumerate(results):
        assert not any(
            math.hypot(x - fx, y - fy) < 0.6 for fx, fy, status in results[:i]
            if status == GoalStatus.STATUS_ABORTED
        ), (x, y, results)
    succeeded = [r for r in results if r[2] == GoalStatus.STATUS_SUCCEEDED]
    assert len(succeeded) + len(failed) == len(results), results
    message = result.result().result.message
    assert (
        f'{len(succeeded)} of {len(results)} frontier goal(s) reached '
        f'({len(succeeded)} by Nav2, 0 seen on the way), {len(failed)} failed' in message
    ), message
    assert result.result().result.goals_reached == len(succeeded)
    assert fake.robot[0] < 2.7


def test_node_cancels_a_goal_seen_on_the_way(ros):
    world = _two_rooms()
    fake = FakeStack(world, (1.0, 1.0), slow=True)
    _, handle, _ = _explore(ros, fake)
    result = handle.get_result_async()
    assert _done(result, 90.0), 'no result in 90 s'
    assert result.result().status == GoalStatus.STATUS_SUCCEEDED, result.result().result.message
    results = list(fake.results)
    canceled = [r for r in results if r[2] == GoalStatus.STATUS_CANCELED]
    assert canceled, results
    succeeded = [r for r in results if r[2] == GoalStatus.STATUS_SUCCEEDED]
    assert len(succeeded) + len(canceled) == len(results), results
    message = result.result().result.message
    assert (
        f'{len(results)} of {len(results)} frontier goal(s) reached '
        f'({len(succeeded)} by Nav2, {len(canceled)} seen on the way), 0 failed' in message
    ), (message, results)
    assert result.result().result.goals_reached == len(results)
    assert fake.known[_reachable(world, (1.0, 1.0))].mean() > 0.97


def test_node_blacklists_a_goal_blocked_on_the_way_and_does_not_count_it(ros):
    world = _two_rooms()
    fake = FakeStack(world, (1.0, 1.0), block=lambda x, y: x > 2.7)
    _, handle, _ = _explore(ros, fake, SkillFrontierExplore.Goal(blacklist_radius=0.6))
    result = handle.get_result_async()
    assert _done(result, 90.0), 'no result in 90 s'
    assert result.result().status == GoalStatus.STATUS_SUCCEEDED, result.result().result.message
    results = list(fake.results)
    blocked = [(x, y) for x, y, status in results if status == GoalStatus.STATUS_CANCELED]
    succeeded = [r for r in results if r[2] == GoalStatus.STATUS_SUCCEEDED]
    assert blocked and all(x > 2.7 for x, _ in blocked), results
    assert len(succeeded) + len(blocked) == len(results), results
    for i, (x, y, _) in enumerate(results):
        assert not any(
            math.hypot(x - bx, y - by) < 0.6 for bx, by, status in results[:i]
            if status == GoalStatus.STATUS_CANCELED
        ), (x, y, results)
    message = result.result().result.message
    assert (
        f'{len(succeeded)} of {len(results)} frontier goal(s) reached '
        f'({len(succeeded)} by Nav2, 0 seen on the way), {len(blocked)} failed' in message
    ), (message, results)
    assert result.result().result.goals_reached == len(succeeded)


def test_node_cancels_nav2_on_a_cancel_and_refuses_a_second_goal(ros):
    fake = FakeStack(_two_rooms(), (1.0, 1.0), hold=True)
    action, handle, _ = _explore(ros, fake)
    deadline = time.time() + 10.0
    while not fake.goals and time.time() < deadline:
        time.sleep(0.05)
    assert fake.goals, 'no Nav2 goal'
    second = action.send_goal_async(SkillFrontierExplore.Goal())
    assert _done(second, 10.0) and not second.result().accepted
    assert _done(handle.cancel_goal_async(), 10.0)
    result = handle.get_result_async()
    assert _done(result, 10.0)
    assert result.result().status == GoalStatus.STATUS_CANCELED
    assert result.result().result.message == 'canceled'
    deadline = time.time() + 5.0
    while not fake.results and time.time() < deadline:
        time.sleep(0.05)
    assert [r[2] for r in fake.results] == [GoalStatus.STATUS_CANCELED]


def test_node_gives_up_at_its_timeout(ros):
    fake = FakeStack(_two_rooms(), (1.0, 1.0), hold=True)
    goal = SkillFrontierExplore.Goal()
    goal.timeout.sec = 2
    _, handle, _ = _explore(ros, fake, goal)
    result = handle.get_result_async()
    assert _done(result, 15.0)
    assert result.result().status == GoalStatus.STATUS_ABORTED
    assert result.result().result.message == 'skill timeout'
    assert [r[2] for r in fake.results] == [GoalStatus.STATUS_CANCELED]


def test_node_refuses_negative_fields(ros):
    fake = FakeStack(_two_rooms(), (1.0, 1.0))
    client = Node('explore_client')
    action = ActionClient(client, SkillFrontierExplore, '/skill/frontier_explore')
    ros(fe.FrontierExplore(), fake, client)
    assert action.wait_for_server(timeout_sec=10.0)
    sent = action.send_goal_async(SkillFrontierExplore.Goal(min_goal_distance=-1.0))
    assert _done(sent, 10.0) and not sent.result().accepted
