'''Frontier exploration: walks the G1 to the frontiers of slam_toolbox's /map until none is left.

    ros2 launch g1_navigation g1_navigation.launch.py frontier_explore:=true ...
    ros2 run g1_navigation frontier_explore --ros-args -p use_sim_time:=true   # beside the stack
    ros2 action send_goal /skill/frontier_explore custom_ros_messages/action/SkillFrontierExplore \
        "{timeout: {sec: 300}}" --feedback

Serves /skill/frontier_explore with custom_ros_messages' SkillFrontierExplore,
as the H1-2 skills node (h12_skills) does, so its callers do not change. Each
round it takes /map, finds its frontiers (free cells next to unknown space and
not next to a wall), groups them into 8-connected clusters and sends Nav2's
/navigate_to_pose to the nearest cluster of at least min_frontier_cells cells
that is farther than min_goal_distance from the robot and not within
blacklist_radius of a goal that failed. A goal that Nav2 does not reach within
goal_timeout is canceled and blacklisted. The exploration succeeds when no
frontier is left, and aborts when the goal's timeout (0: default_timeout) runs
out. That is the H1-2 skill's policy, with its defaults. On the G1 three things
are added:

- Only frontiers the robot can walk to count. A cell is walkable where /map
  has it free and the global costmap (/map, the live scan and the inflation)
  is below Nav2's inscribed cost, the test Smac 2D plans with; a frontier
  counts if it is walkable and 8-connected through walkable cells to the
  robot. The scan, cut 0.15 to 1.2 m above the floor, sees over and between
  what stands on the kitchen's counters, and in the G1 sim's kitchen a map
  that held all of the floor the robot can reach still had sizeable
  frontiers on and behind the counters (README "Frontier exploration"). The
  H1-2 policy would send Nav2 to each in turn and wait for it to fail.
- The goal is the cluster's cell nearest its centroid where Nav2's footprint
  (the footprint parameter, config/g1_nav2.yaml's), facing from the robot to
  the cell, covers no lethal costmap cell; a cluster without one is skipped.
  The H1-2 skill sent the centroid, which for a curved frontier can lie in a
  wall or in unknown space.
- A goal on its way is checked against each new /map and costmap. If fewer
  than min_frontier_cells frontier cells are left within GOAL_FRONTIER of it,
  the robot has seen that space on the way: the goal is canceled, not
  blacklisted, and the next one chosen. If its footprint now covers a lethal
  cell, it is canceled and blacklisted. The H1-2 skill waited for Nav2's
  result; in the G1 sim's kitchen a goal at the edge of the unknown turned
  out to lie 0.2 m from a cabinet, long after the space around it had been
  seen, and Nav2 tried for 53 s before it gave up (README "Frontier
  exploration").

goals_reached in the result counts the goals Nav2 reached and the goals seen on
the way: either way the frontier is resolved. The 360-degree MID-360 sees most
frontiers before the robot gets there, so in the G1 sim's kitchen Nav2 reached
none of them, and the H1-2 count (Nav2's successes only) read 0 for a finished
exploration. Failed and blocked goals, which are blacklisted, do not count.
The result's message gives each count.

Everything is in the map frame, the robot's position is map -> base_frame in
TF. In the G1 stack the odometry localises and slam_toolbox draws /map at its
poses, with map -> odom the identity, so map -> pelvis is the odometry's
odom -> pelvis (README "slam_toolbox"). The deadlines run on the node's clock: sim time in
the sim, so a slow sim stretches the exploration instead of cutting it short.
Nav2 not taking a goal (it is not active, or gone) aborts the exploration
rather than blacklisting the frontier.
'''

import ast
import math
import threading
from collections import namedtuple

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from custom_ros_messages.action import SkillFrontierExplore
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy import ndimage
from tf2_ros import Buffer, TransformException, TransformListener

# a goal field left at 0 takes these (SkillFrontierExplore.action; the H1-2 skill's)
DEFAULT_MIN_CELLS = 6
DEFAULT_BLACKLIST_RADIUS = 0.6   # m
DEFAULT_MIN_GOAL_DISTANCE = 0.7  # m
DEFAULT_GOAL_TIMEOUT = 120.0     # s
# /map: slam_toolbox writes 0 (free), 100 (occupied) and -1 (unknown)
FREE_BELOW = 25
OCCUPIED_FROM = 65
# the global costmap as an OccupancyGrid (nav2_costmap_2d's translation): 99 is
# INSCRIBED_INFLATED_OBSTACLE, 100 LETHAL_OBSTACLE, -1 NO_INFORMATION
COSTMAP_INSCRIBED = 99
COSTMAP_LETHAL = 100
EIGHT = np.ones((3, 3), bool)
ROBOT_SNAP = 0.5    # m: the robot's cell if walkable, else the nearest walkable cell this close
GOAL_FRONTIER = 0.5  # m: a goal is wanted while min_cells frontier cells are this close to it
MAP_WAIT = 3.0      # s: each round, how long to wait for a /map and a costmap that are current
# s: a costmap this much newer than a /map holds it (the global costmap updates at 5 Hz)
COSTMAP_LAG = 0.3
POLL = 0.05         # s
# config/g1_nav2.yaml's footprint (test_frontier_explore.py checks that they agree)
G1_FOOTPRINT = '[[0.44, 0.24], [0.44, -0.24], [-0.20, -0.24], [-0.20, 0.24]]'

# data (rows y, cols x) int16; res m; origin (x, y) of cell (0, 0)'s corner; stamp s.
# Both slam_toolbox's map and the costmap are axis-aligned in map.
Grid = namedtuple('Grid', 'data res origin stamp')
# a goal: a frontier cell (x, y) in map, the heading from the robot, the cluster's size in cells
Frontier = namedtuple('Frontier', 'x y yaw size')


def grid_of(msg):
    '''Grid of a nav_msgs/OccupancyGrid'''
    info = msg.info
    data = np.array(msg.data, dtype=np.int16).reshape(info.height, info.width)
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return Grid(data, float(info.resolution), (info.origin.position.x, info.origin.position.y),
                stamp)


def cell_centres(grid, rows, cols):
    return grid.origin[0] + (cols + 0.5) * grid.res, grid.origin[1] + (rows + 0.5) * grid.res


def sample(grid, x, y):
    '''The grid's values at points (x, y arrays), -1 off the grid'''
    col = np.floor((np.asarray(x) - grid.origin[0]) / grid.res).astype(int)
    row = np.floor((np.asarray(y) - grid.origin[1]) / grid.res).astype(int)
    height, width = grid.data.shape
    inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    values = np.full(np.shape(col), -1, dtype=np.int16)
    values[inside] = grid.data[row[inside], col[inside]]
    return values


def frontier_mask(grid):
    '''Free cells 4-adjacent to unknown (off the map counts as unknown) and not 4-adjacent
    to an occupied cell: next to a wall the unknown is the wall's far side'''
    g = grid.data
    free = (g >= 0) & (g < FREE_BELOW)
    near_unknown = ndimage.binary_dilation(g < 0, border_value=1)
    near_wall = ndimage.binary_dilation(g >= OCCUPIED_FROM)
    return free & near_unknown & ~near_wall


def walkable_mask(grid, costmap):
    '''Cells of grid the robot's centre may be on: free in /map, and known and below the
    inscribed cost in the costmap (sampled at the cell's centre)'''
    rows, cols = np.indices(grid.data.shape)
    cost = sample(costmap, *cell_centres(grid, rows, cols))
    free = (grid.data >= 0) & (grid.data < FREE_BELOW)
    return free & (cost >= 0) & (cost < COSTMAP_INSCRIBED)


def reachable_mask(walkable, grid, robot_xy, snap=ROBOT_SNAP):
    '''The walkable cells 8-connected to the robot's cell (or, if it is not walkable, to the
    nearest walkable cell within snap m), or None without one'''
    rows, cols = np.nonzero(walkable)
    if not len(rows):
        return None
    x, y = cell_centres(grid, rows, cols)
    distance = np.hypot(x - robot_xy[0], y - robot_xy[1])
    k = int(np.argmin(distance))
    if distance[k] > snap:
        return None
    labels, _ = ndimage.label(walkable, structure=EIGHT)
    return labels == labels[rows[k], cols[k]]


def parse_footprint(text):
    '''Nav2's footprint string, '[[x, y], ...]' in the robot's frame, as an (n, 2) array'''
    points = np.array(ast.literal_eval(text), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f'not a footprint polygon: {text!r}')
    return points


def _inside(px, py, polygon):
    '''Points (px, py) inside the polygon, by the even-odd rule'''
    inside = np.zeros(np.shape(px), bool)
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if y1 == y2:
            continue
        crosses = (y1 > py) != (y2 > py)
        inside ^= crosses & (px < x1 + (py - y1) * (x2 - x1) / (y2 - y1))
    return inside


def footprint_fits(costmap, x, y, yaw, footprint):
    '''No lethal costmap cell has its centre inside the footprint at (x, y, yaw)'''
    reach = float(np.hypot(footprint[:, 0], footprint[:, 1]).max())
    col0 = int(np.floor((x - reach - costmap.origin[0]) / costmap.res))
    row0 = int(np.floor((y - reach - costmap.origin[1]) / costmap.res))
    n = int(np.ceil(2 * reach / costmap.res)) + 2
    rows, cols = np.mgrid[row0:row0 + n, col0:col0 + n]
    cx, cy = cell_centres(costmap, rows, cols)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = cx - x, cy - y
    covered = _inside(c * dx + s * dy, -s * dx + c * dy, footprint)
    return not np.any(sample(costmap, cx[covered], cy[covered]) >= COSTMAP_LETHAL)


def find_frontiers(grid, costmap, robot_xy, min_cells, footprint, counts=None):
    '''The frontier clusters of at least min_cells cells that the robot can walk to, as
    Frontiers (their goals), or None if the robot is not on walkable space. counts, a dict,
    gets the frontier cells, those the robot can walk to, and their clusters: all, too
    small, and without a cell where the footprint fits'''
    counts = {} if counts is None else counts
    reachable = reachable_mask(walkable_mask(grid, costmap), grid, robot_xy)
    if reachable is None:
        return None
    frontier = frontier_mask(grid)
    labels, n = ndimage.label(frontier & reachable, structure=EIGHT)
    counts.update(cells=int(frontier.sum()), reachable=int((labels > 0).sum()), clusters=n,
                  small=0, no_fit=0)
    frontiers = []
    for i in range(1, n + 1):
        rows, cols = np.nonzero(labels == i)
        if len(rows) < min_cells:
            counts['small'] += 1
            continue
        x, y = cell_centres(grid, rows, cols)
        for k in np.argsort(np.hypot(x - x.mean(), y - y.mean()), kind='stable'):
            yaw = math.atan2(y[k] - robot_xy[1], x[k] - robot_xy[0])
            if footprint_fits(costmap, x[k], y[k], yaw, footprint):
                frontiers.append(Frontier(float(x[k]), float(y[k]), yaw, len(rows)))
                break
        else:
            counts['no_fit'] += 1
    return frontiers


def outdated(grid, costmap, goal, footprint, min_cells, radius=GOAL_FRONTIER):
    '''Why a goal on its way is no longer wanted, or None: 'seen' when fewer than min_cells
    frontier cells are left within radius of it (the robot saw the space on the way),
    'blocked' when its footprint now covers a lethal costmap cell'''
    if not footprint_fits(costmap, goal.x, goal.y, goal.yaw, footprint):
        return 'blocked'
    rows, cols = np.nonzero(frontier_mask(grid))
    x, y = cell_centres(grid, rows, cols)
    if np.count_nonzero(np.hypot(x - goal.x, y - goal.y) <= radius) < min_cells:
        return 'seen'
    return None


def choose(frontiers, robot_xy, blacklist, blacklist_radius, min_goal_distance):
    '''(the nearest candidate or None, the candidates): the frontiers farther than
    min_goal_distance from the robot and not within blacklist_radius of a blacklisted goal'''
    def distance(f):
        return math.hypot(f.x - robot_xy[0], f.y - robot_xy[1])
    candidates = [f for f in frontiers if distance(f) > min_goal_distance and not any(
        math.hypot(f.x - bx, f.y - by) < blacklist_radius for bx, by in blacklist)]
    return min(candidates, key=distance, default=None), candidates


class _Run:
    '''One exploration goal: feedback, result, cancel and the deadline (the H1-2 skills' _Run)'''

    def __init__(self, node, goal_handle, timeout):
        self.node, self.gh = node, goal_handle
        self.feedback = SkillFrontierExplore.Feedback()
        self.result = SkillFrontierExplore.Result()
        self.deadline = node.now() + timeout

    def remaining(self):
        return max(0.0, self.deadline - self.node.now())

    def _canceled(self):
        self.result.success, self.result.message = False, 'canceled'
        self.gh.canceled()
        self.node.get_logger().info('canceled')
        return self.result

    def check(self):
        '''False, with the goal ended, on a cancel or past the deadline'''
        if self.gh.is_cancel_requested:
            self._canceled()
            return False
        if self.node.now() > self.deadline:
            self.abort('skill timeout')
            return False
        return True

    def phase(self, name, progress):
        '''Publish the phase; False (the goal ended) on a cancel or past the deadline'''
        if not self.check():
            return False
        self.feedback.phase, self.feedback.progress = name, float(progress)
        self.gh.publish_feedback(self.feedback)
        return True

    def abort(self, message):
        if self.gh.is_cancel_requested:
            return self._canceled()
        self.result.success, self.result.message = False, message
        self.node.get_logger().error(message)
        self.gh.abort()
        return self.result

    def succeed(self, message):
        if self.gh.is_cancel_requested:
            return self._canceled()
        self.feedback.phase, self.feedback.progress = 'done', 1.0
        self.gh.publish_feedback(self.feedback)
        self.result.success, self.result.message = True, message
        self.node.get_logger().info(message)
        self.gh.succeed()
        return self.result


class FrontierExplore(Node):

    def __init__(self, **kwargs):
        super().__init__('frontier_explore', **kwargs)
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'pelvis').value
        self.footprint = parse_footprint(self.declare_parameter('footprint', G1_FOOTPRINT).value)
        self.default_timeout = float(self.declare_parameter('default_timeout', 300.0).value)
        map_topic = self.declare_parameter('map_topic', '/map').value
        costmap_topic = self.declare_parameter('costmap_topic', '/global_costmap/costmap').value
        self.group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.grids = {'map': None, 'costmap': None}
        self.busy = False
        # both are latched, so a late subscriber gets the last one
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for key, topic in (('map', map_topic), ('costmap', costmap_topic)):
            self.create_subscription(OccupancyGrid, topic,
                                     lambda msg, key=key: self._on_grid(key, msg), latched,
                                     callback_group=self.group)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.navigate = ActionClient(self, NavigateToPose, '/navigate_to_pose',
                                     callback_group=self.group)
        self.server = ActionServer(
            self, SkillFrontierExplore, '/skill/frontier_explore',
            execute_callback=self._execute, goal_callback=self._on_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self.group)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_grid(self, key, msg):
        grid = grid_of(msg)
        with self.lock:
            self.grids[key] = grid

    def _on_goal(self, request):
        values = (request.min_frontier_cells, request.blacklist_radius,
                  request.min_goal_distance, request.goal_timeout,
                  request.timeout.sec + request.timeout.nanosec * 1e-9)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            self.get_logger().warn(f'refused a goal with negative or non-finite fields: {values}')
            return GoalResponse.REJECT
        with self.lock:
            if self.busy:
                self.get_logger().warn('refused a goal: already exploring')
                return GoalResponse.REJECT
            self.busy = True
        return GoalResponse.ACCEPT

    def _robot_xy(self):
        try:
            t = self.tf.lookup_transform(self.map_frame, self.base_frame, Time())
        except TransformException:
            return None
        return t.transform.translation.x, t.transform.translation.y

    def _wait(self, future, timeout, gh=None):
        '''The future's result, or None after timeout s (node clock) or on gh's cancel'''
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        end = self.now() + timeout
        while not done.is_set():
            if (gh is not None and gh.is_cancel_requested) or self.now() >= end or (
                    not self.context.ok()):
                return None
            done.wait(POLL)
        return future.result()

    def _sleep(self, seconds, gh):
        end = self.now() + seconds
        while self.now() < end and not gh.is_cancel_requested and self.context.ok():
            threading.Event().wait(POLL)

    def _navigate(self, goal, gh, timeout, min_cells):
        '''(Nav2's status for the goal (a GoalStatus), or None if Nav2 did not accept it; why
        it was canceled early: outdated()'s reason, or None). With each new /map or costmap
        on the way the goal is checked (outdated()); on that, a timeout or gh's cancel, the
        Nav2 goal is canceled on the server'''
        msg = NavigateToPose.Goal()
        msg.pose.header.frame_id = self.map_frame
        msg.pose.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = goal.x, goal.y
        msg.pose.pose.orientation.z = math.sin(goal.yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(goal.yaw / 2.0)
        # not cut short by a cancel: an accepted goal must not be left running
        handle = self._wait(self.navigate.send_goal_async(msg), 10.0)
        if handle is None or not handle.accepted:
            return None, None
        result = handle.get_result_async()
        end, why, checked = self.now() + timeout, None, None
        while True:
            response = self._wait(result, min(1.0, max(0.0, end - self.now())), gh)
            if response is not None:
                return response.status, None
            if gh.is_cancel_requested or self.now() >= end or not self.context.ok():
                break
            with self.lock:
                grid, costmap = self.grids['map'], self.grids['costmap']
            if (grid.stamp, costmap.stamp) != checked:
                checked = (grid.stamp, costmap.stamp)
                why = outdated(grid, costmap, goal, self.footprint, min_cells)
                if why is not None:
                    break
        self._wait(handle.cancel_goal_async(), 2.0)
        return GoalStatus.STATUS_CANCELED, why

    def _execute(self, gh):
        request = gh.request
        timeout = request.timeout.sec + request.timeout.nanosec * 1e-9
        run = _Run(self, gh, timeout or self.default_timeout)
        try:
            return self._explore(gh, run, request)
        except Exception as e:  # end the goal instead of leaving it executing
            self.get_logger().error(f'internal error: {e!r}')
            return run.abort(f'internal error: {e}') if gh.is_active else run.result
        finally:
            with self.lock:
                self.busy = False

    def _explore(self, gh, run, request):
        min_cells = request.min_frontier_cells or DEFAULT_MIN_CELLS
        blacklist_radius = request.blacklist_radius or DEFAULT_BLACKLIST_RADIUS
        min_goal_distance = request.min_goal_distance or DEFAULT_MIN_GOAL_DISTANCE
        goal_timeout = request.goal_timeout or DEFAULT_GOAL_TIMEOUT
        log = self.get_logger()
        if not self.navigate.wait_for_server(timeout_sec=10.0):
            return run.abort('Nav2 /navigate_to_pose is not available')
        log.info(f'exploring: min_frontier_cells {min_cells}, blacklist_radius '
                 f'{blacklist_radius:.2f} m, min_goal_distance {min_goal_distance:.2f} m, '
                 f'goal_timeout {goal_timeout:.0f} s, timeout {run.remaining():.0f} s')
        blacklist, sent, succeeded, failed, seen = [], 0, 0, 0, 0
        last_end = None   # node clock at the end of the last goal
        round_start, grid = self.now(), None
        while True:
            if not run.check():
                return run.result
            with self.lock:
                latest, costmap = self.grids['map'], self.grids['costmap']
            robot = self._robot_xy()
            if latest is None or costmap is None or robot is None:
                self._sleep(0.2, gh)
                continue
            # a /map from after the last goal, and a costmap made after that map: the
            # costmap lags the map, and the frontier cells are the map's newest, so
            # with an older costmap they are unknown there and nothing looks reachable.
            # Unless they are late
            late = self.now() > round_start + MAP_WAIT
            if grid is None or late or (last_end is not None and grid.stamp < last_end):
                grid = latest
            if not late and ((last_end is not None and grid.stamp < last_end)
                             or costmap.stamp < grid.stamp + COSTMAP_LAG):
                self._sleep(0.1, gh)
                continue
            if not run.phase('exploring', 0.0):
                return run.result
            counts = {}
            frontiers = find_frontiers(grid, costmap, robot, min_cells, self.footprint, counts)
            if frontiers is None:
                return run.abort(f'the robot at ({robot[0]:.2f}, {robot[1]:.2f}) is not on '
                                 'walkable space of the map and the costmap')
            goal, candidates = choose(frontiers, robot, blacklist, blacklist_radius,
                                      min_goal_distance)
            run.feedback.frontiers_remaining = len(candidates)
            log.info(f'frontiers: {counts["cells"]} cells, {counts["reachable"]} reachable in '
                     f'{counts["clusters"]} clusters ({counts["small"]} under {min_cells} cells, '
                     f'{counts["no_fit"]} where the footprint does not fit); {len(candidates)} '
                     f'candidates; robot at ({robot[0]:.2f}, {robot[1]:.2f}), /map and costmap '
                     f'{self.now() - grid.stamp:.1f} and {self.now() - costmap.stamp:.1f} s old')
            if goal is None:
                return run.succeed(f'map explored: {run.result.goals_reached} of {sent} frontier '
                                   f'goal(s) reached ({succeeded} by Nav2, {seen} seen on the '
                                   f'way), {failed} failed')
            if not run.phase('navigating', 0.5):
                return run.result
            sent += 1
            log.info(f'goal {sent}: ({goal.x:.2f}, {goal.y:.2f}) yaw {math.degrees(goal.yaw):.0f} '
                     f'deg, {goal.size} frontier cells, '
                     f'{math.hypot(goal.x - robot[0], goal.y - robot[1]):.2f} m from the robot at '
                     f'({robot[0]:.2f}, {robot[1]:.2f}); {len(candidates)} candidates, '
                     f'{len(blacklist)} blacklisted')
            start = self.now()
            status, why = self._navigate(goal, gh, min(goal_timeout, run.remaining()), min_cells)
            last_end = round_start = self.now()
            grid = None
            if status is None and not gh.is_cancel_requested:
                # not the frontier's fault (Nav2 not active, or gone): blacklisting would
                # strike out every frontier in turn and end in a false 'explored'
                return run.abort(f'goal {sent}: Nav2 did not accept it')
            # reached: Nav2 got there, or the robot saw the frontier on the way
            if status == GoalStatus.STATUS_SUCCEEDED:
                succeeded += 1
                run.result.goals_reached += 1
                log.info(f'goal {sent}: reached after {last_end - start:.1f} s')
            elif why == 'seen':
                seen += 1
                run.result.goals_reached += 1
                log.info(f'goal {sent}: seen on the way after {last_end - start:.1f} s (its '
                         'frontier is gone); canceled')
            elif not gh.is_cancel_requested:
                blacklist.append((goal.x, goal.y))
                failed += 1
                reason = ('its footprint covers an obstacle now' if why == 'blocked'
                          else f'Nav2 status {status}')
                log.warn(f'goal {sent}: failed ({reason}) after {last_end - start:.1f} s; '
                         'blacklisted')


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplore()
    # the exploration waits inside its execute callback, while the other threads
    # take /map, the costmap, TF and Nav2's responses
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
