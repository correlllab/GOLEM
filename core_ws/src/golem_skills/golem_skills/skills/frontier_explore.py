"""Frontier exploration skill (/skill/frontier_explore).

Walks the robot to the frontiers of slam_toolbox's /map via Nav2's
/navigate_to_pose until no reachable frontier remains or the goal timeout
expires:

- Only frontiers the robot can walk to count. A cell is walkable where /map
  has it free and the global costmap (/global_costmap/costmap) is below Nav2's
  inscribed cost; a frontier counts if it is walkable and 8-connected through
  walkable cells to the robot.
- The goal for each frontier cluster is the cluster cell nearest its centroid
  where Nav2's footprint, facing from the robot to the cell, covers no lethal
  costmap cell; a cluster without one is skipped.
- A goal in flight is checked against each new /map and costmap. If fewer than
  min_frontier_cells frontier cells remain within GOAL_FRONTIER of it, the
  robot has seen that space on the way: the goal is canceled, not blacklisted,
  counted in goals_reached, and the next frontier is chosen. If its footprint
  now covers a lethal cell, it is canceled and blacklisted.
- All deadlines run on the node's ROS clock (sim time when use_sim_time is
  true). Candidate frontiers and the active goal are published on
  /frontier_explore/frontiers (visualization_msgs/MarkerArray) for RViz.
"""

import ast
from collections import namedtuple
import math
import threading

from action_msgs.msg import GoalStatus
from custom_ros_messages.action import SkillFrontierExplore
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
import numpy as np
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy import ndimage
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# Defaults applied when a goal field is left at 0 (SkillFrontierExplore.action).
DEFAULT_MIN_CELLS = 6
DEFAULT_BLACKLIST_RADIUS = 0.6   # m
DEFAULT_MIN_GOAL_DISTANCE = 0.7  # m
DEFAULT_GOAL_TIMEOUT = 120.0     # s
# /map: slam_toolbox writes 0 (free), 100 (occupied) and -1 (unknown).
FREE_BELOW = 25
OCCUPIED_FROM = 65
# Global costmap as OccupancyGrid (nav2_costmap_2d translation):
# 99 is INSCRIBED_INFLATED_OBSTACLE, 100 LETHAL_OBSTACLE, -1 NO_INFORMATION.
COSTMAP_INSCRIBED = 99
COSTMAP_LETHAL = 100
EIGHT = np.ones((3, 3), bool)
ROBOT_SNAP = 0.5    # m: snap to nearest walkable cell within this distance
GOAL_FRONTIER = 0.5  # m: a goal stays wanted while min_cells frontier cells remain this close
MAP_WAIT = 3.0      # s: how long each round waits for a current /map and costmap
COSTMAP_LAG = 0.3   # s: costmap must be at least this much newer than /map
POLL = 0.05         # s
MARKER_TOPIC = '/frontier_explore/frontiers'
# Default footprint (matches golem_slam/config/g1_nav2.yaml; override via 'footprint' param).
G1_FOOTPRINT = '[[0.44, 0.24], [0.44, -0.24], [-0.20, -0.24], [-0.20, 0.24]]'

# data (rows y, cols x) int16; res m; origin (x, y) of cell (0, 0)'s corner; stamp s.
Grid = namedtuple('Grid', 'data res origin stamp')
# A goal: frontier cell (x, y) in map, heading from the robot, cluster size in cells.
Frontier = namedtuple('Frontier', 'x y yaw size')


def grid_of(msg):
    """Return the Grid representation of a nav_msgs/OccupancyGrid."""
    info = msg.info
    data = np.array(msg.data, dtype=np.int16).reshape(info.height, info.width)
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return Grid(
        data, float(info.resolution),
        (info.origin.position.x, info.origin.position.y), stamp,
    )


def cell_centres(grid, rows, cols):
    """Return world (x, y) coordinates of cell centres at (rows, cols)."""
    return (
        grid.origin[0] + (cols + 0.5) * grid.res,
        grid.origin[1] + (rows + 0.5) * grid.res,
    )


def sample(grid, x, y):
    """Return the grid's values at points (x, y arrays), or -1 off the grid."""
    col = np.floor((np.asarray(x) - grid.origin[0]) / grid.res).astype(int)
    row = np.floor((np.asarray(y) - grid.origin[1]) / grid.res).astype(int)
    height, width = grid.data.shape
    inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    values = np.full(np.shape(col), -1, dtype=np.int16)
    values[inside] = grid.data[row[inside], col[inside]]
    return values


def frontier_mask(grid):
    """Return mask of free cells 4-adjacent to unknown and not 4-adjacent to occupied."""
    g = grid.data
    free = (g >= 0) & (g < FREE_BELOW)
    near_unknown = ndimage.binary_dilation(g < 0, border_value=1)
    near_wall = ndimage.binary_dilation(g >= OCCUPIED_FROM)
    return free & near_unknown & ~near_wall


def walkable_mask(grid, costmap):
    """Return mask of grid cells free in /map and known below inscribed cost in costmap."""
    rows, cols = np.indices(grid.data.shape)
    cost = sample(costmap, *cell_centres(grid, rows, cols))
    free = (grid.data >= 0) & (grid.data < FREE_BELOW)
    return free & (cost >= 0) & (cost < COSTMAP_INSCRIBED)


def reachable_mask(walkable, grid, robot_xy, snap=ROBOT_SNAP):
    """Return walkable cells 8-connected to the robot (within snap m), or None."""
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
    """Parse Nav2's footprint string '[[x, y], ...]' into an (n, 2) float array."""
    points = np.array(ast.literal_eval(text), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f'not a footprint polygon: {text!r}')
    return points


def _inside(px, py, polygon):
    """Return boolean mask of points (px, py) inside polygon by the even-odd rule."""
    inside = np.zeros(np.shape(px), bool)
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if y1 == y2:
            continue
        crosses = (y1 > py) != (y2 > py)
        inside ^= crosses & (px < x1 + (py - y1) * (x2 - x1) / (y2 - y1))
    return inside


def footprint_fits(costmap, x, y, yaw, footprint):
    """Return True if no lethal costmap cell centre lies inside the footprint at (x, y, yaw)."""
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
    """Return reachable frontier clusters of at least min_cells cells as Frontiers, or None."""
    counts = {} if counts is None else counts
    reachable = reachable_mask(walkable_mask(grid, costmap), grid, robot_xy)
    if reachable is None:
        return None
    frontier = frontier_mask(grid)
    labels, n = ndimage.label(frontier & reachable, structure=EIGHT)
    counts.update(
        cells=int(frontier.sum()), reachable=int((labels > 0).sum()),
        clusters=n, small=0, no_fit=0,
    )
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
    """Return why an in-flight goal is no longer wanted ('seen' or 'blocked'), or None."""
    if not footprint_fits(costmap, goal.x, goal.y, goal.yaw, footprint):
        return 'blocked'
    rows, cols = np.nonzero(frontier_mask(grid))
    x, y = cell_centres(grid, rows, cols)
    if np.count_nonzero(np.hypot(x - goal.x, y - goal.y) <= radius) < min_cells:
        return 'seen'
    return None


def choose(frontiers, robot_xy, blacklist, blacklist_radius, min_goal_distance):
    """Return (nearest candidate or None, candidates) outside min_goal_distance and blacklist."""
    def distance(f):
        return math.hypot(f.x - robot_xy[0], f.y - robot_xy[1])

    candidates = [
        f for f in frontiers
        if distance(f) > min_goal_distance and not any(
            math.hypot(f.x - bx, f.y - by) < blacklist_radius for bx, by in blacklist
        )
    ]
    return min(candidates, key=distance, default=None), candidates


class _Run:
    """One exploration goal: feedback, result, cancel and ROS-clock deadline."""

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
        """Return False (with the goal ended) on cancel or past the deadline."""
        if self.gh.is_cancel_requested:
            self._canceled()
            return False
        if self.node.now() > self.deadline:
            self.abort('skill timeout')
            return False
        return True

    def phase(self, name, progress):
        """Publish the phase; return False (goal ended) on cancel or past the deadline."""
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


class FrontierExploreSkill:
    """Mixin providing /skill/frontier_explore on SkillsNode (or standalone FrontierExplore)."""

    def _frontier_init(self):
        """Initialize parameters, /map + costmap subscriptions, Nav2 client, and markers."""
        if getattr(self, '_frontier_ready', False):
            return
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'pelvis').value
        self.footprint = parse_footprint(
            self.declare_parameter('footprint', G1_FOOTPRINT).value)
        self.default_timeout = float(
            self.declare_parameter('default_timeout', 300.0).value)
        map_topic = self.declare_parameter('map_topic', '/map').value
        costmap_topic = self.declare_parameter(
            'costmap_topic', '/global_costmap/costmap').value
        cb_group = getattr(self, '_cb_group', None)
        if cb_group is None:
            cb_group = ReentrantCallbackGroup()
            self._cb_group = cb_group
        self._frontier_lock = threading.Lock()
        self._frontier_grids = {'map': None, 'costmap': None}
        self._frontier_busy = False
        latched = QoSProfile(
            depth=1, history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        for key, topic in (('map', map_topic), ('costmap', costmap_topic)):
            self.create_subscription(
                OccupancyGrid, topic,
                lambda msg, key=key: self._on_frontier_grid(key, msg),
                latched, callback_group=cb_group,
            )
        if not hasattr(self, 'tf_buffer'):
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        self._frontier_nav = ActionClient(
            self, NavigateToPose, '/navigate_to_pose', callback_group=cb_group)
        self._frontier_markers = self.create_publisher(
            MarkerArray, MARKER_TOPIC, 10)
        self._frontier_ready = True

    def now(self):
        """Return current node clock time in seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_frontier_grid(self, key, msg):
        grid = grid_of(msg)
        with self._frontier_lock:
            self._frontier_grids[key] = grid

    def _on_frontier_goal(self, request):
        self._frontier_init()
        values = (
            request.min_frontier_cells, request.blacklist_radius,
            request.min_goal_distance, request.goal_timeout,
            request.timeout.sec + request.timeout.nanosec * 1e-9,
        )
        if not all(math.isfinite(v) and v >= 0 for v in values):
            self.get_logger().warn(
                f'refused a goal with negative or non-finite fields: {values}')
            return GoalResponse.REJECT
        with self._frontier_lock:
            if self._frontier_busy:
                self.get_logger().warn('refused a goal: already exploring')
                return GoalResponse.REJECT
            self._frontier_busy = True
        return GoalResponse.ACCEPT

    def _frontier_robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
        except TransformException:
            return None
        return t.transform.translation.x, t.transform.translation.y

    def _publish_frontier_markers(self, frontiers, selected=None):
        arr = MarkerArray()
        wipe = Marker()
        wipe.header.frame_id = self.map_frame
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        stamp = self.get_clock().now().to_msg()
        for i, f in enumerate(frontiers):
            is_sel = (
                selected is not None
                and abs(f.x - selected.x) < 1e-6
                and abs(f.y - selected.y) < 1e-6
            )
            mk = Marker()
            mk.header.frame_id = self.map_frame
            mk.header.stamp = stamp
            mk.ns = 'frontiers'
            mk.id = i
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = float(f.x)
            mk.pose.position.y = float(f.y)
            mk.pose.position.z = 0.15
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 0.35 if is_sel else 0.22
            mk.color.a = 0.9
            mk.color.r, mk.color.g, mk.color.b = (
                (0.1, 1.0, 0.2) if is_sel else (0.1, 0.6, 1.0))
            arr.markers.append(mk)
        self._frontier_markers.publish(arr)

    def _clear_frontier_markers(self):
        arr = MarkerArray()
        wipe = Marker()
        wipe.header.frame_id = self.map_frame
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        self._frontier_markers.publish(arr)

    def _frontier_wait(self, future, timeout, gh=None):
        """Return future result, or None after timeout s (node clock) or on gh cancel."""
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        end = self.now() + timeout
        while not done.is_set():
            if (
                (gh is not None and gh.is_cancel_requested)
                or self.now() >= end
                or not self.context.ok()
            ):
                return None
            done.wait(POLL)
        return future.result()

    def _frontier_sleep(self, seconds, gh):
        end = self.now() + seconds
        while self.now() < end and not gh.is_cancel_requested and self.context.ok():
            threading.Event().wait(POLL)

    def _frontier_navigate(self, goal, gh, timeout, min_cells):
        """Send goal to Nav2 and monitor map/costmap for early completion or blockage."""
        msg = NavigateToPose.Goal()
        msg.pose.header.frame_id = self.map_frame
        msg.pose.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = goal.x, goal.y
        msg.pose.pose.orientation.z = math.sin(goal.yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(goal.yaw / 2.0)
        handle = self._frontier_wait(self._frontier_nav.send_goal_async(msg), 10.0)
        if handle is None or not handle.accepted:
            return None, None
        result = handle.get_result_async()
        end, why, checked = self.now() + timeout, None, None
        while True:
            response = self._frontier_wait(
                result, min(1.0, max(0.0, end - self.now())), gh)
            if response is not None:
                return response.status, None
            if gh.is_cancel_requested or self.now() >= end or not self.context.ok():
                break
            with self._frontier_lock:
                grid, costmap = self._frontier_grids['map'], self._frontier_grids['costmap']
            if (grid.stamp, costmap.stamp) != checked:
                checked = (grid.stamp, costmap.stamp)
                why = outdated(grid, costmap, goal, self.footprint, min_cells)
                if why is not None:
                    break
        self._frontier_wait(handle.cancel_goal_async(), 2.0)
        return GoalStatus.STATUS_CANCELED, why

    def _exec_frontier_explore(self, gh):
        self._frontier_init()
        request = gh.request
        timeout = request.timeout.sec + request.timeout.nanosec * 1e-9
        run = _Run(self, gh, timeout or self.default_timeout)
        try:
            return self._explore_loop(gh, run, request)
        except Exception as e:
            self.get_logger().error(f'internal error: {e!r}')
            return run.abort(f'internal error: {e}') if gh.is_active else run.result
        finally:
            self._clear_frontier_markers()
            with self._frontier_lock:
                self._frontier_busy = False

    def _explore_loop(self, gh, run, request):
        min_cells = request.min_frontier_cells or DEFAULT_MIN_CELLS
        blacklist_radius = request.blacklist_radius or DEFAULT_BLACKLIST_RADIUS
        min_goal_distance = request.min_goal_distance or DEFAULT_MIN_GOAL_DISTANCE
        goal_timeout = request.goal_timeout or DEFAULT_GOAL_TIMEOUT
        log = self.get_logger()
        if not self._frontier_nav.wait_for_server(timeout_sec=10.0):
            return run.abort('Nav2 /navigate_to_pose is not available')
        log.info(
            f'exploring: min_frontier_cells {min_cells}, blacklist_radius '
            f'{blacklist_radius:.2f} m, min_goal_distance {min_goal_distance:.2f} m, '
            f'goal_timeout {goal_timeout:.0f} s, timeout {run.remaining():.0f} s'
        )
        blacklist, sent, succeeded, failed, seen = [], 0, 0, 0, 0
        last_end = None
        round_start, grid = self.now(), None
        while True:
            if not run.check():
                return run.result
            with self._frontier_lock:
                latest = self._frontier_grids['map']
                costmap = self._frontier_grids['costmap']
            robot = self._frontier_robot_xy()
            if latest is None or costmap is None or robot is None:
                self._frontier_sleep(0.2, gh)
                continue
            late = self.now() > round_start + MAP_WAIT
            if grid is None or late or (last_end is not None and grid.stamp < last_end):
                grid = latest
            if not late and (
                (last_end is not None and grid.stamp < last_end)
                or costmap.stamp < grid.stamp + COSTMAP_LAG
            ):
                self._frontier_sleep(0.1, gh)
                continue
            if not run.phase('exploring', 0.0):
                return run.result
            counts = {}
            frontiers = find_frontiers(
                grid, costmap, robot, min_cells, self.footprint, counts)
            if frontiers is None:
                return run.abort(
                    f'the robot at ({robot[0]:.2f}, {robot[1]:.2f}) is not on '
                    'walkable space of the map and the costmap'
                )
            goal, candidates = choose(
                frontiers, robot, blacklist, blacklist_radius, min_goal_distance)
            self._publish_frontier_markers(candidates, selected=goal)
            run.feedback.frontiers_remaining = len(candidates)
            log.info(
                f'frontiers: {counts["cells"]} cells, {counts["reachable"]} reachable in '
                f'{counts["clusters"]} clusters ({counts["small"]} under {min_cells} cells, '
                f'{counts["no_fit"]} where the footprint does not fit); {len(candidates)} '
                f'candidates; robot at ({robot[0]:.2f}, {robot[1]:.2f}), /map and costmap '
                f'{self.now() - grid.stamp:.1f} and {self.now() - costmap.stamp:.1f} s old'
            )
            if goal is None:
                return run.succeed(
                    f'map explored: {run.result.goals_reached} of {sent} frontier '
                    f'goal(s) reached ({succeeded} by Nav2, {seen} seen on the '
                    f'way), {failed} failed'
                )
            if not run.phase('navigating', 0.5):
                return run.result
            sent += 1
            log.info(
                f'goal {sent}: ({goal.x:.2f}, {goal.y:.2f}) yaw {math.degrees(goal.yaw):.0f} '
                f'deg, {goal.size} frontier cells, '
                f'{math.hypot(goal.x - robot[0], goal.y - robot[1]):.2f} m from the robot at '
                f'({robot[0]:.2f}, {robot[1]:.2f}); {len(candidates)} candidates, '
                f'{len(blacklist)} blacklisted'
            )
            start = self.now()
            status, why = self._frontier_navigate(
                goal, gh, min(goal_timeout, run.remaining()), min_cells)
            last_end = round_start = self.now()
            grid = None
            if status is None and not gh.is_cancel_requested:
                return run.abort(f'goal {sent}: Nav2 did not accept it')
            if status == GoalStatus.STATUS_SUCCEEDED:
                succeeded += 1
                run.result.goals_reached += 1
                log.info(f'goal {sent}: reached after {last_end - start:.1f} s')
            elif why == 'seen':
                seen += 1
                run.result.goals_reached += 1
                log.info(
                    f'goal {sent}: seen on the way after {last_end - start:.1f} s (its '
                    'frontier is gone); canceled'
                )
            elif not gh.is_cancel_requested:
                blacklist.append((goal.x, goal.y))
                failed += 1
                reason = (
                    'its footprint covers an obstacle now' if why == 'blocked'
                    else f'Nav2 status {status}'
                )
                log.warn(
                    f'goal {sent}: failed ({reason}) after {last_end - start:.1f} s; '
                    'blacklisted'
                )


class FrontierExplore(Node, FrontierExploreSkill):
    """Standalone node wrapper around FrontierExploreSkill."""

    def __init__(self, **kwargs):
        super().__init__('frontier_explore', **kwargs)
        self._frontier_init()
        self.server = ActionServer(
            self, SkillFrontierExplore, '/skill/frontier_explore',
            execute_callback=self._exec_frontier_explore,
            goal_callback=self._on_frontier_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )

