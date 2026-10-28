"""Closed-loop navigation test of the G1 sim: Nav2 walks the G1 to a goal in the
RoboCasa kitchen. The parts that run in the containers; test_sim_nav.sh runs them
and has the usage.

  sim     in the robocasa container: h12_mujoco.py --g1 as __main__, recording
          the robot's state as test_sim_bringup.py's sim role does (<out>/sim.npz),
          and also, after every physics step, any penetrating contact between a
          robot geom and anything but the floor (<out>/contacts.json). At startup,
          with the robot standing at the spawn, it maps the kitchen from the sim's
          own geometry (<out>/scene.npz) and picks the goal, or takes NAV_GOAL's
          (<out>/goal.json). It stops on <out>/stop and saves sim.npz, then
          contacts.json
  drive   in the ros container, beside 'ros2 launch g1_bringup g1_sim_bringup.launch.py
          use_locomotion:=true use_nav:=true': waits until the walking node walks
          and Nav2's lifecycle nodes are active, sends one NavigateToPose goal,
          and records /cmd_vel, the TF the stack computes, the scan path's
          arrivals (/livox/lidar, /lidar_points, every /scan) and the final /map
          (<out>/drive.npz, map.npz; with --keep-lidar the /livox/lidar frames
          from the goal on, lidar_frames.npz). If /scan stops while /livox/lidar
          still arrives, or slam_toolbox's map -> odom stops advancing, it writes
          the scan path's threads and sockets to <out>/stall.txt and gives up the goal
  report  anywhere with numpy and scipy: checks the run, prints the numbers and
          writes <out>/report.json; exit 1 on a failure

The goal. The sim maps the kitchen around the spawn from the world bounding box
of every geom that is not the robot's or the floor's, visual (what the lidar
sees) or colliding, between GOAL_Z (the robot's height), on a 5 cm grid, and
takes the cell 2-3 m from the pelvis (GOAL_DISTANCE) with the most room: at
least GOAL_CLEARANCE m from every occupied cell, reachable from the spawn through
cells PATH_CLEARANCE m clear, as near GOAL_DISTANCE[1] as possible (ties: the
most room). Its heading is the direction from the spawn to it. NAV_GOAL="x,y[,yaw_deg]"
(world frame) replaces the pick, its heading the direction from the spawn unless
given; the sim refuses it (goal.json's "error", and the test fails) unless its
cell is reachable from the spawn through cells PATH_CLEARANCE m clear, and
goal.json records it ("goal_override"). The drive sends the goal in the map
frame: through odom as the odometry defines it (the world in sim_gt; FAST_LIO's
start pose, the spawn, in fastlio), then slam_toolbox's map -> odom.

Pass if: the goal succeeds within GOAL_TIMEOUT_S of sim time; the true pelvis
ends within FINAL_TOL of the goal; every /cmd_vel is inside the speed caps and
the velocity smoother is its only publisher; /scan never stops for more than
MAX_SCAN_GAP_S while navigating; every occupied cell of the final /map is within
MAP_TOL of something the sim has in the scan's height band (MAP_BAND_Z), and at
most MAX_SCAN_PHANTOM_SHARE of the /scan returns within SCAN_PHANTOM_RANGE, placed
with the sim's true pelvis, are farther than SCAN_PHANTOM_TOL from it; the sim
publishes /odom exactly when the test runs sim_gt; while navigating, map -> pelvis
stays within MAX_LOCALISATION_ERR of the sim's truth in x, y (the odometry
localises, and slam_toolbox only draws the map: with scan matching it drifted
past that); and over the whole recording the pelvis stays above MIN_PELVIS_Z and
tilts at most MAX_TILT_DEG (the sim's own state), no robot geom penetrates
anything but the floor, and the sim's fall logger (sim.log) logs no fall.
test_sim_nav.sh stops the sim before the bringup, so the recording ends with the
robot standing under the walking node. The report also gives the odometry's
drift against the sim's ground truth, and the localisation's (map -> pelvis)
over the whole drive, for the record.

Waits for the stack run on wall time with limits; the goal's time and every
check of the motion are in sim time, so a slow sim stretches the test instead
of failing it.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_bringup as bringup_test  # noqa: E402  (the recorder and the report's helpers)

# the goal, from the sim's geometry
GRID_RES = 0.05            # m
GRID_HALF = 5.0            # m around the spawn
GOAL_Z = (0.02, 1.40)      # m: what the walking robot (1.32 m tall) can touch
GOAL_DISTANCE = (2.0, 2.5, 3.0)   # m from the spawn's pelvis: min, preferred, max
GOAL_CLEARANCE = 1.0       # m to the nearest occupied cell (footprint 0.5 m + inflation 0.55)
PATH_CLEARANCE = 0.35      # m: the robot fits through (half width 0.24 m)
# the scan's height band in the world: pelvis_level -0.59 to +0.45 m, with the
# pelvis at 0.72-0.75 m standing and walking, so 0.13-1.20 m; a little wider
MAP_BAND_Z = (0.10, 1.25)
FLOOR_TOP_Z = 0.005        # m: an env geom whose top is this low is the floor
# drive
NAV2_NODES = ("controller_server", "planner_server", "behavior_server", "bt_navigator",
              "velocity_smoother")
STAND_S = 3.0              # sim s standing before the goal, Nav2 up
SETTLE_S = 2.0             # sim s after the result
GOAL_TIMEOUT_S = 90.0      # sim s from the goal to its result
TF_PERIOD_S = 0.1          # wall s between TF samples
SCAN_STALL_S = 3.0         # wall s without /scan, /livox/lidar still arriving: dump the scan path's threads
# the scan path and its consumers
SCAN_PATH = ("livox_to_pointcloud", "pointcloud_to_laserscan", "level_frame", "body_to_pelvis", "slam_toolbox",
             "controller_server", "planner_server")
# pass limits
FINAL_TOL = 0.4            # m: the true pelvis at the result to the goal (Nav2's xy tolerance 0.25)
MIN_PELVIS_Z = bringup_test.MIN_PELVIS_Z   # 0.70 m
MAX_TILT_DEG = bringup_test.MAX_TILT_DEG   # 15 deg
CAPS = {"vx": (-0.3, 0.5), "vy": (0.0, 0.0), "wz": (-0.6, 0.6)}   # the user's speed caps
CAP_TOL = 1e-6
MAP_TOL = 0.15             # m: an occupied /map cell to the nearest band cell of the sim
MAP_OCCUPIED = 65          # OccupancyGrid value counted occupied (map_server's default threshold)
MAX_SCAN_GAP_S = 0.5       # sim s between /scan stamps: the costmaps' expected_update_rate
SCAN_PHANTOM_RANGE = 4.0   # m: /scan returns this near are checked against the sim's geometry
SCAN_PHANTOM_TOL = 0.25    # m from the nearest band cell; more than SLAM's error and the 5 cm cells
# the largest share of those returns that may lie farther: without livox_to_pointcloud's
# deskew, the floor in the scan made it 1.2-1.7%; with it, 0 in 32 runs
MAX_SCAN_PHANTOM_SHARE = 0.001
# map -> pelvis against the sim's truth in x, y while navigating (the map frame placed in the world at
# the drive's first sample). The odometry localising (slam_toolbox without scan matching, its base frame on
# the floor, so map -> odom is the identity): at most 0.71 cm with fastlio and 0.06 cm with sim_gt in 8 runs.
# With slam_toolbox's base frame on the tilted pelvis it was up to 0.69 cm with fastlio and 1.60 cm with
# sim_gt in 12 runs, the largest 6 m from sim_gt's odom origin (TF interpolating the tilted map -> odom between
# scans). Scan matching: 3.33-5.49 cm on the default goal in the 36 runs since the scan was deskewed
# (2.50-7.26 cm in 11 earlier runs with the floor still in the scan), 13.1-18.1 cm on the corridor goal (6
# runs). Halfway between 1.6 and 3.3 cm
MAX_LOCALISATION_ERR = 0.025   # m


# --------------------------------------------------------------------------- #
# geometry, numpy only
# --------------------------------------------------------------------------- #
_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def world_boxes(geom_xpos, geom_xmat, geom_aabb):
    """(low corners, high corners) (n, 3) of each geom's bounding box in the world"""
    corners = geom_aabb[:, None, :3] + _SIGNS[None] * geom_aabb[:, None, 3:]
    world = np.einsum("nij,nkj->nki", geom_xmat.reshape(-1, 3, 3), corners) + geom_xpos[:, None, :]
    return world.min(axis=1), world.max(axis=1)


def raster(low, high, select, z_range, origin, shape, res=GRID_RES):
    """Occupancy (rows y, cols x) of the selected boxes that reach into z_range"""
    grid = np.zeros(shape, bool)
    sel = select & (high[:, 2] >= z_range[0]) & (low[:, 2] <= z_range[1])
    for i in np.nonzero(sel)[0]:
        a = np.clip(np.floor((low[i, :2] - origin) / res).astype(int), 0, shape[::-1])
        b = np.clip(np.ceil((high[i, :2] - origin) / res).astype(int), 0, shape[::-1])
        grid[a[1]:b[1], a[0]:b[0]] = True
    return grid


def yaw_of(quat_wxyz):
    w, x, y, z = np.asarray(quat_wxyz, float).T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def se2(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def se2_parts(t):
    return float(t[0, 2]), float(t[1, 2]), float(math.atan2(t[1, 0], t[0, 0]))


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pick_goal(clearance, reachable, origin, spawn_xy, res=GRID_RES):
    """(x, y, clearance, distance) of the goal cell, or None (see the module docstring)"""
    ny, nx = clearance.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = origin[0] + (xx + 0.5) * res, origin[1] + (yy + 0.5) * res
    dist = np.hypot(cx - spawn_xy[0], cy - spawn_xy[1])
    lo, want, hi = GOAL_DISTANCE
    ok = reachable & (clearance >= GOAL_CLEARANCE) & (dist >= lo) & (dist <= hi)
    if not ok.any():
        return None
    # nearest the preferred distance, to 5 cm; then the most room
    score = np.where(ok, -np.round(np.abs(dist - want) / res) * 1e3 + clearance, -np.inf)
    k = int(np.argmax(score))
    return float(cx.flat[k]), float(cy.flat[k]), float(clearance.flat[k]), float(dist.flat[k])


def override_goal(text, clearance, reachable, origin, spawn_xy, res=GRID_RES):
    """NAV_GOAL "x,y[,yaw_deg]" (world frame): ((x, y, clearance, distance), yaw rad
    or None, None), or (None, None, the reason the sim refuses it)"""
    try:
        parts = [float(v) for v in text.split(",")]
    except ValueError:
        parts = []
    if len(parts) not in (2, 3) or not all(math.isfinite(v) for v in parts):
        return None, None, f"NAV_GOAL '{text}': not x,y[,yaw_deg] in the world frame"
    x, y = parts[:2]
    cell = tuple(np.floor((np.array([y, x]) - origin[::-1]) / res).astype(int))
    if not all(0 <= c < n for c, n in zip(cell, reachable.shape)) or not reachable[cell]:
        return None, None, (f"NAV_GOAL '{text}': not reachable from the spawn through cells "
                            f"{PATH_CLEARANCE} m clear")
    picked = (x, y, float(clearance[cell]), float(math.hypot(x - spawn_xy[0], y - spawn_xy[1])))
    return picked, (math.radians(parts[2]) if len(parts) == 3 else None), None


# --------------------------------------------------------------------------- #
# sim: h12_mujoco.py --g1 with the bringup test's recorder, a contact check and the goal
# --------------------------------------------------------------------------- #
def _geom_names(model, ids):
    import mujoco
    out = []
    for i in ids:
        geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or f"geom {int(i)}"
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or ""
        out.append(f"{geom} ({body})")
    return out


def robot_and_floor(model, data, base_body):
    """Masks over the geoms: the robot's (the pelvis's tree), and the floor's
    (not the robot's, top at most FLOOR_TOP_Z), and the world boxes"""
    robot = model.body_rootid[model.geom_bodyid] == model.body_rootid[base_body]
    low, high = world_boxes(data.geom_xpos, data.geom_xmat, model.geom_aabb)
    floor = ~robot & (high[:, 2] <= FLOOR_TOP_Z)
    return robot, floor, low, high


def map_scene(out, model, data, base_body):
    """At the spawn: the kitchen's occupancy (scene.npz) and the goal (goal.json)"""
    from scipy import ndimage
    robot, floor, low, high = robot_and_floor(model, data, base_body)
    alpha = model.geom_rgba[:, 3].copy()
    mat = model.geom_matid
    alpha[mat >= 0] = np.minimum(alpha[mat >= 0], model.mat_rgba[mat[mat >= 0], 3])
    # what the lidar can hit (g1_lidar casts against visual geoms with alpha > 0)
    visual = (model.geom_group == 1) & (alpha > 0)
    colliding = (model.geom_contype > 0) | (model.geom_conaffinity > 0)
    env = ~robot & ~floor
    pelvis = data.xpos[base_body].copy()
    yaw = float(yaw_of(data.xquat[base_body]))
    n = int(round(2 * GRID_HALF / GRID_RES))
    origin = np.round((pelvis[:2] - GRID_HALF) / GRID_RES) * GRID_RES
    solid = raster(low, high, env & (visual | colliding), GOAL_Z, origin, (n, n))
    band = raster(low, high, env & visual, MAP_BAND_Z, origin, (n, n))
    clearance = ndimage.distance_transform_edt(~solid) * GRID_RES
    spawn_cell = tuple(np.floor((pelvis[1::-1] - origin[::-1]) / GRID_RES).astype(int))
    labels, _ = ndimage.label(clearance >= PATH_CLEARANCE)
    reachable = (labels == labels[spawn_cell]) & (labels[spawn_cell] > 0)
    override = os.environ.get("NAV_GOAL", "").strip()
    if override:
        picked, goal_yaw, error = override_goal(override, clearance, reachable, origin, pelvis[:2])
    else:
        picked, goal_yaw, error = pick_goal(clearance, reachable, origin, pelvis[:2]), None, (
            f"no cell {GOAL_DISTANCE[0]}-{GOAL_DISTANCE[2]} m from the spawn with {GOAL_CLEARANCE} m of room "
            f"reachable from it")
    np.savez(os.path.join(out, "scene.npz"), origin=origin, res=GRID_RES, solid=solid, band=band,
             clearance=clearance, reachable=reachable, spawn=pelvis, spawn_yaw=yaw)
    goal = {"spawn": [float(v) for v in pelvis], "spawn_yaw": yaw,
            "spawn_clearance": float(clearance[spawn_cell]), "grid_origin": origin.tolist(),
            "res": GRID_RES, "goal_z": list(GOAL_Z), "band_z": list(MAP_BAND_Z),
            "env_geoms": {"visual": int((env & visual).sum()), "colliding": int((env & colliding).sum())},
            "floor_geoms": _geom_names(model, np.nonzero(floor)[0])}
    if override:
        goal["goal_override"] = override
    if picked is None:
        goal["error"] = error
    else:
        x, y, room, dist = picked
        if goal_yaw is None:
            goal_yaw = math.atan2(y - pelvis[1], x - pelvis[0])
        goal.update(goal=[x, y, goal_yaw], goal_clearance=room, goal_distance=dist)
    with open(os.path.join(out, "goal.json"), "w") as f:
        json.dump(goal, f, indent=1)
    print(f"[test_sim_nav] spawn pelvis ({pelvis[0]:.3f}, {pelvis[1]:.3f}), yaw {math.degrees(yaw):.1f} deg, "
          f"{goal['spawn_clearance']:.2f} m of room; goal {goal.get('goal', goal.get('error'))}", flush=True)


class NavRecorder(bringup_test.Recorder):
    """test_sim_bringup's recorder, plus robot-environment contacts after every step"""

    MAX_LISTED = 50

    def __init__(self, out):
        super().__init__(out)
        self._masks = None
        self.contact_steps = 0          # physics steps with a penetrating robot-env contact
        self.contacts = []              # the first MAX_LISTED: [sim t, robot geom, env geom, dist]
        self.deepest = 0.0

    def step(self, iface):
        m, d = iface.model, iface.data
        if self._masks is None:
            robot, floor, _, _ = robot_and_floor(m, d, iface.resolver.base_body)
            self._masks = (robot, ~robot & ~floor)
        n = d.ncon
        if n:
            robot, env = self._masks
            geom = d.contact.geom[:n]
            dist = d.contact.dist[:n]
            hit = (dist < 0) & ((robot[geom[:, 0]] & env[geom[:, 1]]) | (robot[geom[:, 1]] & env[geom[:, 0]]))
            if hit.any():
                self.contact_steps += 1
                self.deepest = min(self.deepest, float(dist[hit].min()))
                for g, gap in zip(geom[hit], dist[hit]):
                    if len(self.contacts) < self.MAX_LISTED:
                        r, e = (g[0], g[1]) if robot[g[0]] else (g[1], g[0])
                        self.contacts.append([float(d.time)] + _geom_names(m, (r, e)) + [float(gap)])
        super().step(iface)

    def save(self):
        super().save()
        with open(os.path.join(self.out, "contacts.json"), "w") as f:
            json.dump({"steps": self.steps, "contact_steps": self.contact_steps, "deepest": self.deepest,
                       "first": self.contacts}, f, indent=1)


def run_sim(out, sim_args):
    import runpy
    import signal

    sys.path.insert(0, bringup_test.SIM)
    # the G1's own classes: golem_robocasa's top-level unitree_interface is the H1-2's
    from g1 import unitree_interface
    from g1.g1_sensors import SensorConfig

    with open(os.path.join(out, "sensors.json"), "w") as f:
        json.dump(SensorConfig.from_env()._asdict(), f, indent=1)
    recorder = NavRecorder(out)
    after_step = unitree_interface.SimInterface.after_step
    init = unitree_interface.SimInterface.__init__

    def recorded_after_step(self):
        after_step(self)
        recorder.step(self)

    def mapped_init(self, model, data, resolver, *args, **kwargs):
        # the robot stands at the spawn, settled, before the first step and the bridges
        init(self, model, data, resolver, *args, **kwargs)
        map_scene(out, model, data, resolver.base_body)

    def interrupt(signum, frame):
        raise KeyboardInterrupt()

    unitree_interface.SimInterface.after_step = recorded_after_step
    unitree_interface.SimInterface.__init__ = mapped_init
    signal.signal(signal.SIGTERM, interrupt)
    script = os.path.join(bringup_test.SIM, "h12_mujoco.py")
    sys.argv = [script, "--g1"] + sim_args
    try:
        runpy.run_path(script, run_name="__main__")
    except (bringup_test.StopRecording, KeyboardInterrupt):
        print("[test_sim_nav] sim stopped", flush=True)
    finally:
        recorder.save()


# --------------------------------------------------------------------------- #
# drive: the goal against the running bringup
# --------------------------------------------------------------------------- #
def _tf_parts(tf):
    """(stamp s, x, y, z, qx, qy, qz, qw) of a geometry_msgs/TransformStamped, or None"""
    if tf is None:
        return None
    t, r = tf.transform.translation, tf.transform.rotation
    return [tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9, t.x, t.y, t.z, r.x, r.y, r.z, r.w]


def _yaw_xyzw(x, y, z, w):
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _process_cpu(names):
    """{name: CPU cores used on average since its process started}, for the nodes named"""
    hz = os.sysconf("SC_CLK_TCK")
    with open("/proc/uptime") as f:
        uptime = float(f.read().split()[0])
    out = {}
    for pid in (p for p in os.listdir("/proc") if p.isdigit()):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read().rsplit(")", 1)[-1].split()
        except OSError:
            continue
        for name in names:
            if f"__node:={name}" in cmd or (name == "fastlio_mapping" and "fastlio_mapping" in cmd):
                age = uptime - int(stat[19]) / hz
                out[name] = round((int(stat[11]) + int(stat[12])) / hz / age, 3) if age > 0 else None
    return out


class Driver:
    def __init__(self, out, odom_source, keep_lidar=False):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import OccupancyGrid
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from tf2_ros import Buffer, TransformListener
        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("test_sim_nav")
        self.out = out
        self.odom_source = odom_source
        self.events, self.info = [], {"odom_source": odom_source}
        self.clock, self.clocks = None, 0
        self.node.create_subscription(Clock, "/clock", self._on_clock,
                                      QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT), raw=True)
        self.cmds = []   # wall, sim, vx, vy, vz, wx, wy, wz
        self.node.create_subscription(Twist, "/cmd_vel", self._on_cmd, 100)
        self.map = None
        self.node.create_subscription(OccupancyGrid, "/map", self._on_map, QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # the scan path, as its consumers hear it: the stamps of /livox/lidar and
        # /lidar_points (raw, not decoded), and every /scan
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan, PointCloud2
        from livox_ros_driver2.msg import CustomMsg
        self.arrivals = {"/livox/lidar": [], "/lidar_points": []}   # wall, sim, stamp
        # with keep_lidar, the /livox/lidar frames themselves from the goal on (lidar_frames.npz)
        self.keep_lidar, self.lidar_frames = keep_lidar, None
        for topic, kind in (("/livox/lidar", CustomMsg), ("/lidar_points", PointCloud2)):
            self.node.create_subscription(kind, topic, lambda raw, topic=topic: self._on_arrival(topic, raw),
                                          qos_profile_sensor_data, raw=True)
        self.scans = []   # wall, sim, stamp, angle_min, angle_increment, then the ranges
        self.node.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self._stall_dumped = False
        # TF on a node of its own, spun on the listener's thread: /tf runs at hundreds of Hz
        self.tf_node = Node("test_sim_nav_tf")
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self.tf_node, spin_thread=True)
        self.samples = {"map_pelvis": [], "odom_pelvis": []}   # wall, sim, then _tf_parts
        self._next_sample = 0.0
        self.feedback = None
        self.launch_log = os.path.join(out, "launch.log")

    def _on_clock(self, raw):
        self.clock = bringup_test._cdr_stamp(raw)
        self.clocks += 1

    def _on_cmd(self, msg):
        self.cmds.append([time.time(), self._sim(), msg.linear.x, msg.linear.y, msg.linear.z,
                          msg.angular.x, msg.angular.y, msg.angular.z])

    def _on_map(self, msg):
        self.map = msg

    def _on_arrival(self, topic, raw):
        self.arrivals[topic].append([time.time(), self._sim(), bringup_test._cdr_stamp(raw)])
        if topic == "/livox/lidar" and self.lidar_frames is not None:
            self.lidar_frames.append(bytes(raw))

    def _on_scan(self, msg):
        self.scans.append([time.time(), self._sim(), msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                           msg.angle_min, msg.angle_increment] + list(msg.ranges))

    def _check_scan_path(self, now):
        """Once: if /scan stopped while /livox/lidar still arrives, or slam_toolbox's
        map -> odom stopped advancing while /scan arrives, record the scan path to
        <out>/stall.txt: the ROS graph of its topics, and for each of its processes
        (and their consumers), per thread, the state, kernel wait channel and what
        changed in 1 s (CPU time, context switches), and its UDP sockets' queues and drops"""
        points, lidar = self.arrivals["/lidar_points"], self.arrivals["/livox/lidar"]
        if self._stall_dumped or not self.scans or not points or not lidar or self.clock is None:
            return
        scan_stalled = now - self.scans[-1][0] >= SCAN_STALL_S and now - lidar[-1][0] <= 1.0
        map_odom = _tf_parts(self.lookup("map", "odom"))
        slam_stalled = (map_odom is not None and self.clock - map_odom[0] >= SCAN_STALL_S
                        and now - self.scans[-1][0] <= 1.0)
        if not (scan_stalled or slam_stalled):
            return
        self._stall_dumped = True
        lines = [f"{'/scan' if scan_stalled else 'map -> odom'} stalled. /scan last heard "
                 f"{now - self.scans[-1][0]:.1f} s ago (stamp {self.scans[-1][2]:.2f}), /lidar_points "
                 f"{now - points[-1][0]:.1f} s ago (stamp "
                 f"{points[-1][2]:.2f}), /livox/lidar {now - self.arrivals['/livox/lidar'][-1][0]:.1f} s ago; map -> "
                 f"odom stamp {map_odom[0] if map_odom else None}; sim t {self.clock}"]
        for topic in ("/livox/lidar", "/lidar_points", "/scan"):
            subs = [f"{i.node_name}({i.qos_profile.reliability.name})"
                    for i in self.node.get_subscriptions_info_by_topic(topic)]
            pubs = [i.node_name for i in self.node.get_publishers_info_by_topic(topic)]
            lines.append(f"graph {topic}: publishers {pubs}, subscribers {subs}")
        pids = {}
        for pid in sorted(int(p) for p in os.listdir("/proc") if p.isdigit()):
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if any(f"__node:={name}" in cmd for name in SCAN_PATH):
                pids[pid] = cmd

        def threads(pid):
            out = {}
            for tid in os.listdir(f"/proc/{pid}/task"):
                try:
                    with open(f"/proc/{pid}/task/{tid}/stat") as f:
                        stat = f.read().rsplit(")", 1)[-1].split()
                    with open(f"/proc/{pid}/task/{tid}/status") as f:
                        status = dict(line.split(":", 1) for line in f if ":" in line)
                    with open(f"/proc/{pid}/task/{tid}/comm") as f:
                        comm = f.read().strip()
                    with open(f"/proc/{pid}/task/{tid}/wchan") as f:
                        wchan = f.read().strip()
                except OSError:
                    continue
                out[int(tid)] = (comm, stat[0], wchan, (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK"),
                                 int(status["voluntary_ctxt_switches"]) + int(status["nonvoluntary_ctxt_switches"]))
            return out
        before = {pid: threads(pid) for pid in pids}
        time.sleep(1.0)
        udp = {}
        for path in ("/proc/net/udp", "/proc/net/udp6"):
            try:
                with open(path) as f:
                    for row in f.readlines()[1:]:
                        c = row.split()
                        tx, rx = (int(v, 16) for v in c[4].split(":"))
                        udp[c[9]] = (c[1], tx, rx, int(c[-1]))   # inode: local address, tx, rx queue, drops
            except OSError:
                pass
        for pid, cmd in pids.items():
            lines.append(f"pid {pid}: {cmd[:150]}")
            after = threads(pid)
            for tid, (comm, state, wchan, cpu, switches) in sorted(after.items()):
                b = before[pid].get(tid)
                lines.append(f"  tid {tid} {comm:16s} {state} {wchan:34s} cpu {cpu:7.2f} s "
                             f"(+{cpu - b[3] if b else 0:.2f}) switches +{switches - b[4] if b else 0}")
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    target = os.readlink(f"/proc/{pid}/fd/{fd}")
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in udp:
                    addr, tx, rx, drops = udp[target[8:-1]]
                    lines.append(f"  udp {addr}: tx queue {tx}, rx queue {rx}, drops {drops}")
        with open(os.path.join(self.out, "stall.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        self.mark("scan_stall" if scan_stalled else "slam_stall", since_wall_s=now - self.scans[-1][0])

    def _sim(self):
        return math.nan if self.clock is None else self.clock

    def mark(self, name, **extra):
        self.events.append({"name": name, "wall": time.time(), "sim": self.clock, **extra})
        print(f"[test_sim_nav] {name} {extra if extra else ''}", flush=True)
        with open(os.path.join(self.out, "events.json"), "w") as f:
            json.dump({"events": self.events, "info": self.info}, f, indent=1)

    def lookup(self, target, source):
        from rclpy.time import Time
        try:
            return self.tf.lookup_transform(target, source, Time())
        except Exception:
            return None

    def spin(self, timeout=0.05):
        """One callback, and the TF samples when they are due"""
        self.rclpy.spin_once(self.node, timeout_sec=timeout)
        now = time.time()
        if now >= self._next_sample:
            self._next_sample = now + TF_PERIOD_S
            self._check_scan_path(now)
            for key, frames in (("map_pelvis", ("map", "pelvis")), ("odom_pelvis", ("odom", "pelvis"))):
                tf = self.lookup(*frames)
                if tf is not None:
                    parts = _tf_parts(tf)
                    rows = self.samples[key]
                    if not rows or rows[-1][2] != parts[0]:   # a new stamp
                        rows.append([now, self._sim()] + parts)

    def spin_for(self, seconds, wall_limit=120.0):
        """Spin for `seconds` of sim time; fail if the sim stalls (see test_sim_bringup)"""
        give_up = time.time() + wall_limit
        heard = self.clocks
        while self.clocks == heard and time.time() < give_up:
            self.spin()
        if self.clocks == heard:
            raise RuntimeError("no /clock from the sim")
        start = self.clock
        while self.clock < start + seconds:
            if time.time() > give_up:
                self.mark("sim_stalled", sim_seconds=self.clock - start, wanted=seconds)
                raise RuntimeError(f"the sim's clock advanced {self.clock - start:.2f} of {seconds} s "
                                   f"in {wall_limit:.0f} s wall")
            self.spin()

    def wait_future(self, future, timeout):
        end = time.time() + timeout
        while not future.done() and time.time() < end:
            self.spin()
        return future.result() if future.done() else None

    def _log_has(self, text):
        try:
            with open(self.launch_log, errors="replace") as f:
                return text in f.read()
        except OSError:
            return False

    def wait_walking(self, timeout=240.0):
        """The walking node logs 'walking: policy started' once it has the legs"""
        end = time.time() + timeout
        while time.time() < end:
            if self._log_has("walking: policy started"):
                self.mark("walking")
                return True
            if self._log_has("g1_core: not starting"):
                self.mark("launch_refused")
                return False
            for _ in range(20):
                self.spin()
        self.mark("walking_timeout")
        return False

    def wait_nav(self, timeout=180.0):
        """Nav2's lifecycle nodes active, /navigate_to_pose served, map -> pelvis in TF"""
        from lifecycle_msgs.msg import State
        from lifecycle_msgs.srv import GetState
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        self.navigate = ActionClient(self.node, NavigateToPose, "navigate_to_pose")
        clients = {name: self.node.create_client(GetState, f"/{name}/get_state") for name in NAV2_NODES}
        end = time.time() + timeout
        states = {}
        while time.time() < end:
            for name, client in clients.items():
                if states.get(name) == "active" or not client.service_is_ready():
                    continue
                response = self.wait_future(client.call_async(GetState.Request()), 5.0)
                if response is not None:
                    states[name] = ("active" if response.current_state.id == State.PRIMARY_STATE_ACTIVE
                                    else response.current_state.label)
            if (all(states.get(n) == "active" for n in NAV2_NODES) and self.navigate.server_is_ready()
                    and self.lookup("map", "pelvis") is not None):
                self.mark("nav_active", states=states)
                return True
            for _ in range(20):
                self.spin()
        self.mark("nav_timeout", states=states, action_server=self.navigate.server_is_ready(),
                  map_pelvis=self.lookup("map", "pelvis") is not None)
        return False

    def goal_in_map(self, goal):
        """The goal (world x, y, yaw) in map: through odom as the odom source defines
        it (the world in sim_gt, the spawn's pelvis on the floor in fastlio), then
        slam_toolbox's map -> odom"""
        if self.odom_source == "sim_gt":
            world_odom = np.eye(3)
        else:
            world_odom = se2(goal["spawn"][0], goal["spawn"][1], goal["spawn_yaw"])
        map_odom = self.lookup("map", "odom")
        if map_odom is None:
            raise RuntimeError("no map -> odom in TF")
        _, x, y, _, qx, qy, qz, qw = _tf_parts(map_odom)
        t = se2(x, y, _yaw_xyzw(qx, qy, qz, qw)) @ np.linalg.inv(world_odom) @ se2(*goal["goal"])
        return se2_parts(t)

    def navigate_to(self, x, y, yaw):
        from action_msgs.msg import GoalStatus
        from nav2_msgs.action import NavigateToPose
        msg = NavigateToPose.Goal()
        msg.pose.header.frame_id = "map"   # stamp 0: the latest transform
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        map_pelvis, map_odom = self.lookup("map", "pelvis"), self.lookup("map", "odom")
        self.mark("goal_sent", goal_map=[x, y, yaw], map_pelvis=_tf_parts(map_pelvis),
                  map_odom=_tf_parts(map_odom))
        if self.keep_lidar:
            self.lidar_frames = []
        handle = self.wait_future(self.navigate.send_goal_async(
            msg, feedback_callback=self._on_feedback), 10.0)
        if handle is None or not handle.accepted:
            self.mark("goal_result", accepted=False, status=None, success=False)
            return False
        result = handle.get_result_async()
        start, give_up = self.clock, time.time() + 4 * GOAL_TIMEOUT_S
        while not result.done():
            if self.clock - start > GOAL_TIMEOUT_S or time.time() > give_up or self._stall_dumped:
                self.mark("goal_timeout" if not self._stall_dumped else "goal_abandoned_on_stall",
                          sim_seconds=self.clock - start)
                self.wait_future(handle.cancel_goal_async(), 10.0)
                self.wait_future(result, 10.0)
                break
            self.spin()
        status = result.result().status if result.done() else None
        self.mark("goal_result", accepted=True, status=status,
                  success=status == GoalStatus.STATUS_SUCCEEDED, sim_seconds=self.clock - start,
                  feedback=self.feedback, map_pelvis=_tf_parts(self.lookup("map", "pelvis")))
        return status == GoalStatus.STATUS_SUCCEEDED

    def _on_feedback(self, msg):
        f = msg.feedback
        self.feedback = {"distance_remaining": f.distance_remaining,
                         "navigation_time": f.navigation_time.sec + f.navigation_time.nanosec * 1e-9,
                         "number_of_recoveries": f.number_of_recoveries}

    def finish(self):
        """The final map, what publishes /cmd_vel and /odom, the scan path's CPU, the samples"""
        self.info["cpu_cores"] = _process_cpu(SCAN_PATH + ("fastlio_mapping", "locomotion_node"))
        self.info["cmd_vel_publishers"] = sorted(
            i.node_name for i in self.node.get_publishers_info_by_topic("/cmd_vel"))
        self.info["odom_publishers"] = sorted(
            i.node_name for i in self.node.get_publishers_info_by_topic("/odom"))
        m = self.map
        if m is not None:
            o = m.info.origin
            q = o.orientation
            np.savez(os.path.join(self.out, "map.npz"),
                     data=np.array(m.data, np.int8).reshape(m.info.height, m.info.width),
                     res=m.info.resolution, frame_id=m.header.frame_id,
                     origin=[o.position.x, o.position.y, _yaw_xyzw(q.x, q.y, q.z, q.w)],
                     stamp=m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        self.info["map"] = None if m is None else {"width": m.info.width, "height": m.info.height,
                                                   "res": m.info.resolution}
        if self.lidar_frames:
            frames, self.lidar_frames = self.lidar_frames, None
            np.savez(os.path.join(self.out, "lidar_frames.npz"),
                     data=np.frombuffer(b"".join(frames), np.uint8),
                     sizes=np.array([len(f) for f in frames], np.int64))
        width = max((len(r) for r in self.scans), default=5)
        scans = np.full((len(self.scans), width), np.nan)
        for i, row in enumerate(self.scans):
            scans[i, :len(row)] = row
        np.savez(os.path.join(self.out, "drive.npz"), cmds=np.array(self.cmds, float).reshape(-1, 8),
                 scans=scans, lidar=np.array(self.arrivals["/livox/lidar"], float).reshape(-1, 3),
                 points=np.array(self.arrivals["/lidar_points"], float).reshape(-1, 3),
                 **{k: np.array(v, float).reshape(-1, 10) for k, v in self.samples.items()})
        self.mark("end")

    def close(self):
        self.tf_node.destroy_node()
        self.node.destroy_node()
        self.rclpy.shutdown()


def run_drive(out, odom_source, keep_lidar=False):
    path = os.path.join(out, "goal.json")
    with open(path) as f:
        goal = json.load(f)
    d = Driver(out, odom_source, keep_lidar)
    d.info["goal"] = goal
    ok = False
    try:
        d.mark("drive_start")
        if "error" in goal:
            d.mark("no_goal", error=goal["error"])
            return 1
        if d.wait_walking() and d.wait_nav():
            d.spin_for(STAND_S)
            x, y, yaw = d.goal_in_map(goal)
            ok = d.navigate_to(x, y, yaw)
            d.spin_for(SETTLE_S)
        return 0 if ok else 1
    finally:
        try:
            d.finish()   # what was recorded, whatever happened
        finally:
            d.close()


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _interp_pose(s, t):
    """The sim's pelvis (x, y, z, yaw, upz, quat wxyz) at sim times t, linearly
    between the 50 Hz samples (the yaw unwrapped)"""
    st = s["t"]
    yaw = np.unwrap(yaw_of(s["quat"]))
    xyz = np.stack([np.interp(t, st, s["pelvis"][:, i]) for i in range(3)], axis=-1)
    k = np.clip(np.searchsorted(st, t), 1, len(st) - 1)
    k = np.where(np.abs(st[k - 1] - t) < np.abs(st[k] - t), k - 1, k)   # the nearest sample
    return xyz, np.interp(t, st, yaw), s["quat"][k]


def _quat_xyzw_to_matrix(q):
    x, y, z, w = np.asarray(q, float).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _quat_wxyz_to_matrix(q):
    q = np.asarray(q, float)
    return _quat_xyzw_to_matrix(np.concatenate([q[..., 1:], q[..., :1]], axis=-1))


class Report(bringup_test.Report):
    def __init__(self, out):
        super().__init__(out)
        with open(os.path.join(out, "goal.json")) as f:
            self.goal = json.load(f)
        drive = os.path.join(out, "drive.npz")
        self.drive = dict(np.load(drive)) if os.path.exists(drive) else None

    def pelvis_at(self, t):
        return _interp_pose(self.s, np.asarray(t, float))

    def world_from(self, frame_row):
        """SE(2) world <- frame at a TF sample (stamp, x, y, z, qx, qy, qz, qw) of frame -> pelvis"""
        stamp, x, y, _, qx, qy, qz, qw = frame_row
        xyz, yaw, _ = self.pelvis_at(stamp)
        return se2(xyz[0], xyz[1], yaw) @ np.linalg.inv(se2(x, y, _yaw_xyzw(qx, qy, qz, qw)))

    def nav_test(self):
        s, ev = self.s, self.events
        g = self.goal
        self.numbers["spawn"] = {"pelvis": g["spawn"], "yaw_deg": math.degrees(g["spawn_yaw"]),
                                 "clearance": g["spawn_clearance"]}
        if "goal" not in g:
            self.check("goal picked", False, g.get("error", "no goal"))
            return
        gx, gy, gyaw = g["goal"]
        self.numbers["goal_world"] = {"xy": [gx, gy], "yaw_deg": math.degrees(gyaw),
                                      "clearance": g["goal_clearance"], "distance": g["goal_distance"],
                                      "override": g.get("goal_override")}
        print(f"     goal ({gx:.3f}, {gy:.3f}) heading {math.degrees(gyaw):.1f} deg, {g['goal_distance']:.2f} m "
              f"from the spawn ({g['spawn'][0]:.3f}, {g['spawn'][1]:.3f}, yaw {math.degrees(g['spawn_yaw']):.1f} "
              f"deg), {g['goal_clearance']:.2f} m of room (the spawn {g['spawn_clearance']:.2f} m)"
              + (f"; NAV_GOAL {g['goal_override']}" if "goal_override" in g else "; the sim's pick"))
        # the stack: the odometry the test asked for, the walking node, Nav2
        odom_pubs = self.info.get("odom_publishers")
        want_odom = ["mujoco_sensors"] if self.info["odom_source"] == "sim_gt" else []
        self.check("odom source", odom_pubs == want_odom,
                   f"odom_source {self.info['odom_source']}: /odom publishers {odom_pubs} (expected {want_odom})")
        released = np.nonzero(s["released"])[0]
        if not released.size:
            self.check("hand-over", False, "the stand-in was never released (no valid rt/lowcmd)")
            return
        r = int(released[0])
        self.numbers["handover_t"] = float(s["t"][r])
        for name in ("walking", "nav_active", "goal_sent"):
            if name not in ev:
                self.check("stack up", False, f"no '{name}' mark: see drive.log and launch.log")
                return
        first = ev["drive_start"]["wall"]
        self.numbers["stack_up_wall_s"] = {n: ev[n]["wall"] - first for n in ("walking", "nav_active")}
        self.check("stack up", True,
                   f"stand-in released at sim t {s['t'][r]:.2f} s; the drive saw the walking node walk at sim t "
                   f"{ev['walking']['sim']:.1f} s and Nav2 active at {ev['nav_active']['sim']:.1f} s "
                   f"({ev['walking']['wall'] - first:.0f} and {ev['nav_active']['wall'] - first:.0f} s wall after "
                   f"it started, beside the launch)")
        # the goal
        sent, result = ev["goal_sent"], ev.get("goal_result", {})
        took = result.get("sim_seconds", math.nan)
        wall = result["wall"] - sent["wall"] if result else math.nan
        fb = result.get("feedback") or {}
        self.numbers["goal"] = {"status": result.get("status"), "sim_s": took, "wall_s": wall,
                                "recoveries": fb.get("number_of_recoveries")}
        self.check("goal reached", bool(result.get("success")) and took <= GOAL_TIMEOUT_S,
                   f"NavigateToPose status {result.get('status')} (4 = succeeded) after {took:.1f} s of sim time "
                   f"({wall:.1f} s wall; limit {GOAL_TIMEOUT_S:.0f} s sim), {fb.get('number_of_recoveries')} "
                   f"recoveries")
        # where the goal is in the world: the map frame as it was when it was sent
        world_map = self.world_from(sent["map_pelvis"])
        goal_map = sent["goal_map"]
        goal_world = world_map @ se2(*goal_map)
        gwx, gwy, gwyaw = se2_parts(goal_world)
        self.numbers["goal_sent_world_vs_picked_m"] = float(math.hypot(gwx - gx, gwy - gy))
        # the drive's last mark: the sim stops soon after, with the robot standing
        last = max(ev.values(), key=lambda e: e["wall"])["name"]
        i_end = self.at("goal_result") if "goal_result" in ev else self.at(last)
        end_xyz, end_yaw, _ = self.pelvis_at(s["t"][i_end])
        err = float(math.hypot(end_xyz[0] - gx, end_xyz[1] - gy))
        yaw_err = float(math.degrees(_wrap(end_yaw - gyaw)))
        nav = result.get("map_pelvis")
        nav_err = (float(math.hypot(nav[1] - goal_map[0], nav[2] - goal_map[1])) if nav else math.nan)
        nav_yaw = (float(math.degrees(_wrap(_yaw_xyzw(*nav[4:]) - goal_map[2]))) if nav else math.nan)
        start_xyz, _, _ = self.pelvis_at(sent["sim"])
        # and once the robot has stood SETTLE_S
        i_last = self.at(last)
        last_xyz, last_yaw, _ = self.pelvis_at(s["t"][i_last])
        settled = float(math.hypot(last_xyz[0] - gx, last_xyz[1] - gy))
        self.numbers["final"] = {"true_xy_err": err, "true_yaw_err_deg": yaw_err, "nav_xy_err": nav_err,
                                 "nav_yaw_err_deg": nav_yaw, "true_end": [float(v) for v in end_xyz],
                                 "travelled_straight": float(math.hypot(*(end_xyz[:2] - start_xyz[:2]))),
                                 "settled_xy_err": settled,
                                 "settled_yaw_err_deg": float(math.degrees(_wrap(last_yaw - gyaw)))}
        self.check("final position", max(err, settled) <= FINAL_TOL,
                   f"true pelvis at the result ({end_xyz[0]:.3f}, {end_xyz[1]:.3f}): {err:.3f} m from the goal, "
                   f"heading off by {yaw_err:+.1f} deg; {s['t'][i_last] - s['t'][i_end]:.1f} s later {settled:.3f} m, "
                   f"{self.numbers['final']['settled_yaw_err_deg']:+.1f} deg (limit {FINAL_TOL} m); Nav2's own view "
                   f"at the result {nav_err:.3f} m, {nav_yaw:+.1f} deg; the goal as sent lands "
                   f"{self.numbers['goal_sent_world_vs_picked_m']:.3f} m from the picked one")
        # the pelvis per phase, for the record (run() checks the whole recording)
        a, b = self.at("goal_sent"), i_end
        self.phases([("stand-in", 0, r), ("standing, walking node", r, a), ("navigating", a, b + 1),
                     ("after the result", b + 1, len(s["t"]))])
        self.cmd_vel()
        self.scan_path(sent["sim"], s["t"][i_end])
        self.map_check(world_map)
        self.scan_phantoms()
        self.drift(sent["sim"], s["t"][i_end])

    def scan_path(self, t0, t1):
        """/scan kept coming while navigating; and where the scan path stopped if it did"""
        if self.drive is None or "scans" not in self.drive or not len(self.drive["scans"]):
            self.check("scan path", False, "no /scan heard")
            return
        numbers, text = {}, []
        for key, label in (("lidar", "/livox/lidar"), ("points", "/lidar_points"), ("scans", "/scan")):
            rows = self.drive[key]
            stamps = rows[:, 2]
            inside = stamps[(stamps >= t0) & (stamps <= t1)]
            # the gaps over the window, its edges included
            edges = np.concatenate([[t0], inside, [t1]])
            gap = float(np.diff(edges).max()) if len(edges) > 1 else math.inf
            numbers[label] = {"count": int(len(rows)), "in_window": int(len(inside)), "longest_gap_s": gap,
                              "last_stamp": float(stamps.max()) if len(stamps) else None}
            text.append(f"{label} {len(inside)} (longest gap {gap:.2f} s)")
        self.numbers["scan_path"] = numbers
        gap = numbers["/scan"]["longest_gap_s"]
        stall = os.path.exists(os.path.join(self.out, "stall.txt"))
        self.check("scan path", gap <= MAX_SCAN_GAP_S,
                   f"while navigating (sim t {t0:.1f}-{t1:.1f} s): " + ", ".join(text)
                   + f"; limit {MAX_SCAN_GAP_S} s for /scan" + ("; see stall.txt" if stall else ""))

    def scan_phantoms(self):
        """Every /scan return within SCAN_PHANTOM_RANGE, placed in the world with the
        sim's true pelvis at the scan's stamp (the scan is in pelvis_level), against
        what the sim has in the band"""
        if self.drive is None or "scans" not in self.drive or not len(self.drive["scans"]):
            self.check("scan has no phantoms", False, "no /scan heard")
            return
        from scipy import ndimage
        scene = np.load(os.path.join(self.out, "scene.npz"))
        band, origin, gres = scene["band"], scene["origin"], float(scene["res"])
        to_band = ndimage.distance_transform_edt(~band) * gres
        rows = self.drive["scans"]
        rows = rows[(rows[:, 2] >= self.s["t"][0]) & (rows[:, 2] <= self.s["t"][-1])]
        ranges = rows[:, 5:]
        angles = rows[:, 3:4] + rows[:, 4:5] * np.arange(ranges.shape[1])
        xyz, yaw, _ = self.pelvis_at(rows[:, 2])
        ok = np.isfinite(ranges) & (ranges <= SCAN_PHANTOM_RANGE)
        placed = np.where(ok, ranges, 0.0)   # (no inf or NaN into the integer cells below)
        wx = xyz[:, 0:1] + placed * np.cos(angles + yaw[:, None])
        wy = xyz[:, 1:2] + placed * np.sin(angles + yaw[:, None])
        ix = np.floor((wx - origin[0]) / gres).astype(int)
        iy = np.floor((wy - origin[1]) / gres).astype(int)
        ok &= (ix >= 0) & (iy >= 0) & (ix < band.shape[1]) & (iy < band.shape[0])
        dist = np.full(ranges.shape, np.nan)
        dist[ok] = to_band[iy[ok], ix[ok]]
        far = ok & (dist > SCAN_PHANTOM_TOL + 1e-9)
        per_scan = far.sum(axis=1)
        k, j = np.nonzero(far)
        examples = [[round(float(rows[a, 2]), 2), round(float(wx[a, b]), 2), round(float(wy[a, b]), 2),
                     round(float(ranges[a, b]), 2), round(float(dist[a, b]), 2)] for a, b in zip(k[:5], j[:5])]
        self.numbers["scan_phantoms"] = {"scans": int(len(rows)), "returns": int(ok.sum()), "far": int(far.sum()),
                                         "scans_with_far": int((per_scan > 0).sum()),
                                         "dist_p99": float(np.nanpercentile(dist[ok], 99)) if ok.any() else None,
                                         "examples_t_x_y_range_dist": examples}
        share = far.sum() / max(1, ok.sum())
        self.check("scan has no phantoms", ok.sum() > 0 and share <= MAX_SCAN_PHANTOM_SHARE,
                   f"{len(rows)} scans, {ok.sum()} returns within {SCAN_PHANTOM_RANGE} m placed with the sim's "
                   f"pelvis: {far.sum()} ({share:.2%}, limit {MAX_SCAN_PHANTOM_SHARE:.1%}) in {(per_scan > 0).sum()} "
                   f"scans more than {SCAN_PHANTOM_TOL} m from anything the sim has in the band; 99th percentile "
                   f"{self.numbers['scan_phantoms']['dist_p99'] or 0:.3f} m"
                   + (f"; e.g. (stamp, x, y, range, distance) {examples}" if examples else ""))

    def upright(self):
        """The pelvis's height and tilt from the sim, over the whole recording"""
        z, tilt, t = self.s["pelvis"][:, 2], bringup_test._tilt_deg(self.s["upz"]), self.s["t"]
        self.numbers["upright"] = {"z_min": float(z.min()), "z_max": float(z.max()),
                                   "tilt_max_deg": float(tilt.max()), "sim_t": [float(t[0]), float(t[-1])]}
        self.check("never fell", z.min() >= MIN_PELVIS_Z and tilt.max() <= MAX_TILT_DEG,
                   f"the whole recording (sim t {t[0]:.1f}-{t[-1]:.1f} s): pelvis z {z.min():.3f}-"
                   f"{z.max():.3f} m, tilt max {tilt.max():.1f} deg (limits z >= {MIN_PELVIS_Z}, tilt <= "
                   f"{MAX_TILT_DEG})")

    def contacts(self):
        """Robot-environment contacts over the whole recording"""
        path = os.path.join(self.out, "contacts.json")
        if not os.path.exists(path):
            self.check("no contact", False, "no contacts.json from the sim")
            return
        with open(path) as f:
            c = json.load(f)
        # the sim lists the first MAX_LISTED contacts, in time order
        self.numbers["contacts"] = {"steps": c["steps"], "contact_steps": c["contact_steps"],
                                    "deepest": c["deepest"], "first": c["first"][:5]}
        if c["contact_steps"]:
            text = (f"{c['contact_steps']} physics steps with contact, deepest {c['deepest'] * 1000:.1f} mm, from "
                    f"sim t {c['first'][0][0]:.2f} s: {c['first'][:3]}")
        else:
            text = "no robot geom penetrated anything but the floor"
        self.check("no contact", not c["contact_steps"], f"{c['steps']} physics steps: {text}")

    def fall_log(self):
        """The sim's fall logger (golem_robocasa/g1/g1_mujoco.py), which looks at every physics step"""
        path = os.path.join(self.out, "sim.log")
        if not os.path.exists(path):
            self.check("no fall logged", False, "no sim.log")
            return
        with open(path, errors="replace") as f:
            falls = [line.strip() for line in f if "ROBOT FELL" in line]
        self.check("no fall logged", not falls, falls[0] if falls else "sim.log logs no fall")

    def cmd_vel(self):
        pubs = self.info.get("cmd_vel_publishers")
        if self.drive is None:
            self.check("/cmd_vel caps", False, "no drive.npz")
            return
        c = self.drive["cmds"]
        vx, vy, wz = c[:, 2], c[:, 3], c[:, 7]
        other = np.abs(c[:, [4, 5, 6]]).max() if len(c) else 0.0
        ok = (len(c) > 0 and pubs == ["velocity_smoother"]
              and np.all((vx >= CAPS["vx"][0] - CAP_TOL) & (vx <= CAPS["vx"][1] + CAP_TOL))
              and np.all(np.abs(vy) <= CAP_TOL)
              and np.all((wz >= CAPS["wz"][0] - CAP_TOL) & (wz <= CAPS["wz"][1] + CAP_TOL)) and other <= CAP_TOL)
        rng = (lambda v: f"{v.min():+.3f}..{v.max():+.3f}") if len(c) else (lambda v: "none")
        self.numbers["cmd_vel"] = {"count": int(len(c)), "publishers": pubs,
                                   "vx": [float(vx.min()), float(vx.max())] if len(c) else None,
                                   "vy_abs_max": float(np.abs(vy).max()) if len(c) else None,
                                   "wz": [float(wz.min()), float(wz.max())] if len(c) else None}
        self.check("/cmd_vel caps", ok,
                   f"{len(c)} messages from {pubs}: vx {rng(vx)} m/s (caps {CAPS['vx']}), vy {rng(vy)}, wz "
                   f"{rng(wz)} rad/s (caps {CAPS['wz']}), largest |vz|, |wx|, |wy| {other:.2g}")

    def map_check(self, world_map):
        path = os.path.join(self.out, "map.npz")
        if not os.path.exists(path):
            self.check("map", False, "no /map")
            return
        from scipy import ndimage
        m = np.load(path)
        scene = np.load(os.path.join(self.out, "scene.npz"))
        rows, cols = np.nonzero(m["data"] >= MAP_OCCUPIED)
        res = float(m["res"])
        ox, oy, oyaw = m["origin"]
        local = np.stack([(cols + 0.5) * res, (rows + 0.5) * res, np.ones(len(rows))])
        world = world_map @ se2(ox, oy, oyaw) @ local
        # distance from each cell of the sim's grid to the nearest band cell
        band, origin, gres = scene["band"], scene["origin"], float(scene["res"])
        to_band = ndimage.distance_transform_edt(~band) * gres
        ix = np.floor((world[0] - origin[0]) / gres).astype(int)
        iy = np.floor((world[1] - origin[1]) / gres).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < band.shape[1]) & (iy < band.shape[0])
        dist = np.full(len(rows), np.inf)
        dist[inside] = to_band[iy[inside], ix[inside]]
        far = dist > MAP_TOL + 1e-9   # the grid's distances are multiples of 5 cm, in floats
        worst = world[:2, far][:, np.argsort(-dist[far])[:5]].T.round(2).tolist() if far.any() else []
        self.numbers["map"] = {"occupied": int(len(rows)), "far": int(far.sum()), "outside_grid": int((~inside).sum()),
                               "dist_p99": float(np.percentile(dist[inside], 99)) if inside.any() else None,
                               "dist_max": float(dist.max()) if len(dist) else None, "far_examples": worst,
                               "unknown_share": float((m["data"] < 0).mean()),
                               "free_cells": int(((m["data"] >= 0) & (m["data"] < MAP_OCCUPIED)).sum())}
        self.check("map has no phantoms", len(rows) > 50 and not far.any(),
                   f"{len(rows)} occupied cells in the final /map, {far.sum()} more than {MAP_TOL} m from anything "
                   f"the sim has at {MAP_BAND_Z[0]}-{MAP_BAND_Z[1]} m" + (f", e.g. {worst}" if worst else "")
                   + (f"; 99th percentile {self.numbers['map']['dist_p99']:.3f} m" if inside.any() else ""))

    def drift(self, t0, t1):
        """The odometry (odom -> pelvis) and the localisation (map -> pelvis) against the sim, for the
        record; and the localisation's x, y while navigating (sim t0 to t1), checked"""
        if self.drive is None:
            self.check("localisation", False, "no drive.npz")
            return
        for key, label in (("odom_pelvis", "odometry"), ("map_pelvis", "localisation")):
            rows = self.drive[key]
            rows = rows[np.isfinite(rows[:, 2]) & (rows[:, 2] >= self.s["t"][0]) & (rows[:, 2] <= self.s["t"][-1])]
            if len(rows) < 2:
                self.numbers[label] = None
                continue
            stamp = rows[:, 2]
            xyz, yaw, quat = self.pelvis_at(stamp)
            # the frame in the world from the first sample (odom is level, the floor z = 0)
            frame = self.world_from(rows[0, 2:])
            local = np.linalg.inv(frame) @ np.stack([xyz[:, 0], xyz[:, 1], np.ones(len(rows))])
            dyaw = math.atan2(frame[1, 0], frame[0, 0])
            xy_err = np.hypot(rows[:, 3] - local[0], rows[:, 4] - local[1])
            z_err = rows[:, 5] - xyz[:, 2]
            yaw_est = np.array([_yaw_xyzw(*q) for q in rows[:, 6:10]])
            yaw_err = np.degrees(_wrap(yaw_est - (yaw - dyaw)))
            # tilt: the estimated pelvis z axis against the true one, turned into the frame
            up_est = _quat_xyzw_to_matrix(rows[:, 6:10])[:, :, 2]
            up_true = _quat_wxyz_to_matrix(quat)[:, :, 2]
            c, si = math.cos(dyaw), math.sin(dyaw)
            up_true = up_true @ np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])   # R(-dyaw) u
            tilt_err = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", up_est, up_true), -1.0, 1.0)))
            path = float(np.sum(np.hypot(np.diff(xyz[:, 0]), np.diff(xyz[:, 1]))))
            numbers = {"samples": int(len(rows)), "sim_t": [float(stamp[0]), float(stamp[-1])],
                       "path_m": path, "xy_err_max": float(xy_err.max()), "xy_err_end": float(xy_err[-1]),
                       "yaw_err_deg_max_abs": float(np.abs(yaw_err).max()), "yaw_err_deg_end": float(yaw_err[-1])}
            text = (f"xy error max {xy_err.max() * 100:.1f} cm, at the end {xy_err[-1] * 100:.1f} cm; yaw error max "
                    f"{np.abs(yaw_err).max():.2f} deg, at the end {yaw_err[-1]:+.2f} deg")
            if key == "odom_pelvis":
                # odom is level with the floor at z = 0: its z and tilt against the sim's too.
                # map -> pelvis adds slam_toolbox's map -> odom, the identity: x, y, yaw there
                numbers.update(z_err_range=[float(z_err.min()), float(z_err.max())],
                               tilt_err_deg_max_abs=float(np.abs(tilt_err).max()))
                text += (f"; z error {z_err.min() * 100:+.1f} to {z_err.max() * 100:+.1f} cm; tilt error max "
                         f"{np.abs(tilt_err).max():.2f} deg")
            else:
                navigating = (stamp >= t0) & (stamp <= t1)
                numbers["xy_err_max_navigating"] = float(xy_err[navigating].max()) if navigating.any() else None
            self.numbers[label] = numbers
            print(f"     {label} ({key.replace('_', ' -> ')}) vs the sim over {stamp[-1] - stamp[0]:.0f} s of sim "
                  f"time and {path:.2f} m walked: {text}")
        # the odometry localises (slam_toolbox only draws the map): a scan matcher's drift shows here
        worst = (self.numbers.get("localisation") or {}).get("xy_err_max_navigating")
        window = f"while navigating (sim t {t0:.1f}-{t1:.1f} s)"
        self.check("localisation", worst is not None and worst <= MAX_LOCALISATION_ERR,
                   f"map -> pelvis against the sim's truth {window}: x, y error max {worst * 100:.2f} cm (limit "
                   f"{MAX_LOCALISATION_ERR * 100:.1f} cm)" if worst is not None
                   else f"no map -> pelvis sample {window}")

    def run(self, out):
        t, wall = self.s["t"], self.s["wall"]
        self.numbers["real_time_factor"] = float((t[-1] - t[0]) / (wall[-1] - wall[0]))
        edges = np.searchsorted(t, np.arange(t[0], t[-1], 1.0))
        per_s = np.diff(t[edges]) / np.diff(wall[edges]) if len(edges) > 2 else np.array([math.nan])
        self.numbers["real_time_factor_per_s"] = {"min": float(per_s.min()), "p5": float(np.percentile(per_s, 5)),
                                                  "median": float(np.median(per_s))}
        steps = ""
        if "goal_sent" in self.events and "goal_result" in self.events:
            a, b = self.at("goal_sent"), self.at("goal_result")
            if b > a:
                self.numbers["real_time_factor_goal"] = float((t[b] - t[a]) / (wall[b] - wall[a]))
                steps = f"; {self.numbers['real_time_factor_goal']:.2f} from the goal to its result"
        load = os.path.join(out, "load.log")
        if os.path.exists(load):
            with open(load) as f:
                loads = [float(line.rsplit("load average:", 1)[1].split(",")[0])
                         for line in f if "load average" in line]
            if loads:
                self.numbers["host_load_1min"] = [min(loads), max(loads)]
                steps += f"; host load (1 min) {min(loads):.1f}-{max(loads):.1f}"
        print(f"     {len(t)} samples, sim t {t[0]:.1f}-{t[-1]:.1f} s, real-time factor "
              f"{self.numbers['real_time_factor']:.2f} (per sim second: min {per_s.min():.2f}, 5th percentile "
              f"{np.percentile(per_s, 5):.2f}, median {np.median(per_s):.2f}){steps}")
        try:
            self.nav_test()
        except KeyError as e:
            self.check("test steps", False, f"the run has no {e} mark, so the test's steps did not finish "
                                            "(see drive.log)")
        # however far the drive got: the sim stops before the bringup, so the robot stands to the end
        self.upright()
        self.contacts()
        self.fall_log()
        ok = all(c["pass"] for c in self.checks)
        with open(os.path.join(out, "report.json"), "w") as f:
            json.dump({"pass": ok, "checks": self.checks, "numbers": self.numbers}, f, indent=1)
        print(f"{'PASSED' if ok else 'FAILED'}: {sum(c['pass'] for c in self.checks)}/{len(self.checks)} checks")
        return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=("sim", "drive", "report"))
    parser.add_argument("--out", required=True, help="the shared output directory")
    parser.add_argument("--odom-source", choices=("fastlio", "sim_gt"), default="fastlio",
                        help="drive: the bringup's odom_source")
    parser.add_argument("--keep-lidar", action="store_true",
                        help="drive: keep the /livox/lidar frames from the goal on (lidar_frames.npz)")
    parser.epilog = "sim: the G1 sim's arguments (h12_mujoco.py --g1) follow a --"
    argv = sys.argv[1:]
    sim_args = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:len(argv) - len(sim_args) - ("--" in argv)])
    if args.role == "sim":
        run_sim(args.out, sim_args)
        return 0
    if args.role == "drive":
        return run_drive(args.out, args.odom_source, args.keep_lidar)
    return Report(args.out).run(args.out)


if __name__ == "__main__":
    sys.exit(main())
