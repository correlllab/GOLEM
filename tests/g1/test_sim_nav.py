"""Test the G1's navigation in the sim, closed loop: the roles that run in the containers.

Nav2 walks the G1 to a goal in the RoboCasa kitchen. test_sim_nav.sh runs the roles and has
the usage.

  sim     in the robocasa container: h12_mujoco.py --g1 as __main__, recording
          the robot's state as test_sim_bringup.py's sim role does (<out>/sim.npz),
          and also, after every physics step, any penetrating contact between a
          robot geom and anything but the floor (<out>/contacts.json). At startup,
          with the robot at the spawn, it maps the kitchen from the sim's
          own geometry (<out>/scene.npz) and picks the goal, or takes NAV_GOAL's
          (<out>/goal.json). It stops on <out>/stop and saves sim.npz, then
          contacts.json.
  drive   in the ros container, beside 'ros2 launch golem_bringup g1_sim.launch.py
          use_nav:=true': waits until the walking node walks, the sim's elastic band
          is released and Nav2's lifecycle nodes are active, sends one NavigateToPose goal,
          and records /cmd_vel, the TF the stack computes, the scan path's
          arrivals (/livox/lidar, /cloud_body_filtered, every /converted_scan) and the final /map
          (<out>/drive.npz, map.npz; with --keep-lidar the /livox/lidar frames
          from the goal on, lidar_frames.npz). If /converted_scan stops while /livox/lidar
          still arrives, or slam_toolbox's map -> odom stops advancing, it writes
          the scan path's threads and sockets to <out>/stall.txt and gives up the goal.
  report  anywhere with numpy and scipy: checks the run, prints the numbers and
          writes <out>/report.json; exit 1 on a failure.

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
frame: through odom as FAST_LIO's odometry defines it (its start pose, the
spawn), then slam_toolbox's map -> odom.

Pass if: the goal succeeds within GOAL_TIMEOUT_S of sim time; the true pelvis
ends within FINAL_TOL of the goal; every /cmd_vel is inside the speed caps and
the velocity smoother is its only publisher; /converted_scan never stops for
more than MAX_SCAN_GAP_S while navigating; every occupied cell of the final /map
is within MAP_TOL of something the sim has in the scan's height band
(MAP_BAND_Z), and at most MAX_SCAN_PHANTOM_SHARE of the /converted_scan returns
within SCAN_PHANTOM_RANGE, placed with the sim's true pelvis, are farther than
SCAN_PHANTOM_TOL from it; the sim publishes no /odom; while navigating,
map -> pelvis stays within MAX_LOCALISATION_ERR of the sim's truth in x, y; and
over the whole recording the pelvis stays above MIN_PELVIS_Z and
tilts at most MAX_TILT_DEG (the sim's own state: hanging from the band, then
standing on the walking node), no robot geom penetrates anything but the floor,
the sim's fall logger (sim.log) logs no fall, and golem_safety (launch.log)
latches no estop before the drive's end. The stack has to come up: the walking
node's policy started and the band released (the sim's record) before the goal.
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

# The goal, from the sim's geometry.
GRID_RES = 0.05  # m
GRID_HALF = 5.0  # m around the spawn
GOAL_Z = (0.02, 1.40)  # m: what the walking robot (1.32 m tall) can touch
GOAL_DISTANCE = (2.0, 2.5, 3.0)  # m from the spawn's pelvis: min, preferred, max
GOAL_CLEARANCE = 1.0  # m to the nearest occupied cell (footprint 0.5 m + inflation 0.55)
PATH_CLEARANCE = 0.35  # m: the robot fits through (half width 0.24 m)
# m: the scan's height band in the world: pelvis_level -0.59 to +0.45 m, with the pelvis at
# 0.72-0.75 m standing and walking, so 0.13-1.20 m; a little wider.
MAP_BAND_Z = (0.10, 1.35)
FLOOR_TOP_Z = 0.005  # m: an env geom whose top is this low is the floor
# The drive.
NAV2_NODES = ("controller_server", "planner_server", "behavior_server", "bt_navigator",
              "velocity_smoother")
STAND_S = 3.0  # s of sim time standing before the goal, Nav2 up
SETTLE_S = 2.0  # s of sim time after the result
GOAL_TIMEOUT_S = 90.0  # s of sim time from the goal to its result
TF_PERIOD_S = 0.1  # s of wall time between TF samples
SCAN_STALL_S = 3.0  # s of wall time without /converted_scan, /livox/lidar still arriving: dump the scan path's threads
# The scan path's nodes and their consumers.
SCAN_PATH = ("self_filter", "pointcloud_to_laserscan", "level_odom", "octomap_server", "body_to_pelvis", "slam_toolbox",
             "controller_server", "planner_server")
# Pass limits.
FINAL_TOL = 0.4  # m: the true pelvis at the result to the goal (Nav2's xy tolerance 0.25)
MIN_PELVIS_Z = bringup_test.MIN_PELVIS_Z  # 0.70 m
MAX_TILT_DEG = bringup_test.MAX_TILT_DEG  # 15 deg
CAPS = {"vx": (-0.2, 0.3), "vy": (0.0, 0.0), "wz": (-0.5, 0.5)}  # m/s and rad/s: the user's speed caps
CAP_TOL = 1e-6
MAP_TOL = 0.15  # m: an occupied /map cell to the nearest band cell of the sim
MAP_OCCUPIED = 65  # OccupancyGrid value counted occupied (map_server's default threshold)
MAX_SCAN_GAP_S = 0.5  # s of sim time between /converted_scan stamps: the costmaps' expected_update_rate
SCAN_PHANTOM_RANGE = 4.0  # m: /converted_scan returns this near are checked against the sim's geometry
SCAN_PHANTOM_TOL = 0.25  # m from the nearest band cell; more than SLAM's error and the 5 cm cells
# The largest share of those returns that may lie farther. The deskewed scan measures 0 (32
# runs); a scan with the floor in it gives 1.2-1.7 %, which this fails.
MAX_SCAN_PHANTOM_SHARE = 0.001
# m: map -> pelvis against the sim's truth in x, y while navigating (the map frame placed in the
# world at the drive's first sample). Measured: at most 0.71 cm with the odometry localising and
# slam_toolbox's base frame on the floor (8 runs); 3.3-5.5 cm on the default goal and
# 13-18 cm on the corridor goal with slam_toolbox's scan matching correcting map -> odom.
MAX_LOCALISATION_ERR = 0.025


_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def world_boxes(geom_xpos, geom_xmat, geom_aabb):
    """Return (low corners, high corners), each (n, 3), of each geom's bounding box in the world."""
    corners = geom_aabb[:, None, :3] + _SIGNS[None] * geom_aabb[:, None, 3:]
    world = np.einsum("nij,nkj->nki", geom_xmat.reshape(-1, 3, 3), corners) + geom_xpos[:, None, :]
    return world.min(axis=1), world.max(axis=1)


def raster(low, high, select, z_range, origin, shape, res=GRID_RES):
    """Return the occupancy grid (rows y, columns x) of the selected boxes that reach into `z_range`."""
    grid = np.zeros(shape, bool)
    sel = select & (high[:, 2] >= z_range[0]) & (low[:, 2] <= z_range[1])
    for i in np.nonzero(sel)[0]:
        a = np.clip(np.floor((low[i, :2] - origin) / res).astype(int), 0, shape[::-1])
        b = np.clip(np.ceil((high[i, :2] - origin) / res).astype(int), 0, shape[::-1])
        grid[a[1]:b[1], a[0]:b[0]] = True
    return grid


def yaw_of(quat_wxyz):
    """Return the yaw, rad, of WXYZ quaternions."""
    w, x, y, z = np.asarray(quat_wxyz, float).T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def se2(x, y, yaw):
    """Return the 3x3 homogeneous SE(2) transform of (x, y, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def se2_parts(t):
    """Return (x, y, yaw) of a 3x3 SE(2) transform."""
    return float(t[0, 2]), float(t[1, 2]), float(math.atan2(t[1, 0], t[0, 0]))


def _wrap(angle):
    """Return `angle` wrapped to [-pi, pi), rad."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pick_goal(clearance, reachable, origin, spawn_xy, res=GRID_RES):
    """Return (x, y, clearance, distance) of the goal cell, or None (see the module docstring)."""
    ny, nx = clearance.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = origin[0] + (xx + 0.5) * res, origin[1] + (yy + 0.5) * res
    dist = np.hypot(cx - spawn_xy[0], cy - spawn_xy[1])
    lo, want, hi = GOAL_DISTANCE
    ok = reachable & (clearance >= GOAL_CLEARANCE) & (dist >= lo) & (dist <= hi)
    if not ok.any():
        return None
    # Nearest the preferred distance, to 5 cm; then the most room.
    score = np.where(ok, -np.round(np.abs(dist - want) / res) * 1e3 + clearance, -np.inf)
    k = int(np.argmax(score))
    return float(cx.flat[k]), float(cy.flat[k]), float(clearance.flat[k]), float(dist.flat[k])


def override_goal(text, clearance, reachable, origin, spawn_xy, res=GRID_RES):
    """Parse NAV_GOAL "x,y[,yaw_deg]" (world frame) and check it is reachable.

    Returns ((x, y, clearance, distance), yaw in rad or None, None), or (None, None, the reason
    the sim refuses it).
    """
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


def _geom_names(model, ids):
    """Return "geom (body)" names of the given geom ids."""
    import mujoco
    out = []
    for i in ids:
        geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or f"geom {int(i)}"
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or ""
        out.append(f"{geom} ({body})")
    return out


def robot_and_floor(model, data, base_body):
    """Return masks over the geoms, the robot's (the pelvis' tree) and the floor's, and their world boxes.

    The floor's geoms are not the robot's and have their top at most FLOOR_TOP_Z.
    """
    robot = model.body_rootid[model.geom_bodyid] == model.body_rootid[base_body]
    low, high = world_boxes(data.geom_xpos, data.geom_xmat, model.geom_aabb)
    floor = ~robot & (high[:, 2] <= FLOOR_TOP_Z)
    return robot, floor, low, high


def map_scene(out, model, data, base_body):
    """Map the kitchen around the spawn (scene.npz) and pick the goal, or take NAV_GOAL's (goal.json)."""
    from scipy import ndimage
    robot, floor, low, high = robot_and_floor(model, data, base_body)
    alpha = model.geom_rgba[:, 3].copy()
    mat = model.geom_matid
    alpha[mat >= 0] = np.minimum(alpha[mat >= 0], model.mat_rgba[mat[mat >= 0], 3])
    # What the lidar can hit: g1_lidar casts against visual geoms with alpha > 0.
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


def write_live_pose(out, data, base):
    """Write <out>/live_pose.json: the sim's pelvis (x, y, yaw) at its stamp.

    The drive uses it to place a world goal in the map frame. It is written whole, then renamed,
    and never published.
    """
    pos = data.xpos[base]
    rot = data.xmat[base].reshape(3, 3)
    snapshot = {"stamp": float(data.time),
                "pose": [float(pos[0]), float(pos[1]), float(math.atan2(rot[1, 0], rot[0, 0]))]}
    temporary = os.path.join(out, "live_pose.tmp")
    with open(temporary, "w") as target:
        json.dump(snapshot, target)
    os.replace(temporary, os.path.join(out, "live_pose.json"))


class ContactLog:
    """Log penetrating contacts between a robot geom and anything but the floor, after every physics step.

    The robot is the pelvis' tree. Shared with tests/sim/test_sim_stack.py.
    """

    MAX_LISTED = 50

    def __init__(self):
        """Start with no contacts."""
        self._masks = None
        self.contact_steps = 0  # physics steps with a penetrating robot-env contact
        self.spans = []  # [first sim t, last sim t, steps] of consecutive such steps
        self.contacts = []  # the first MAX_LISTED: [sim t, robot geom, env geom, dist]
        self.deepest = 0.0  # m, the most negative contact distance

    def step(self, model, data, base_body):
        """Log this physics step's robot-environment contacts."""
        if self._masks is None:
            robot, floor, _, _ = robot_and_floor(model, data, base_body)
            self._masks = (robot, ~robot & ~floor)
        n = data.ncon
        if not n:
            return
        robot, env = self._masks
        geom = data.contact.geom[:n]
        dist = data.contact.dist[:n]
        hit = (dist < 0) & ((robot[geom[:, 0]] & env[geom[:, 1]]) | (robot[geom[:, 1]] & env[geom[:, 0]]))
        if not hit.any():
            return
        self.contact_steps += 1
        now = float(data.time)
        if self.spans and now - self.spans[-1][1] <= 1.5 * model.opt.timestep:
            self.spans[-1][1:] = [now, self.spans[-1][2] + 1]
        else:
            self.spans.append([now, now, 1])
        self.deepest = min(self.deepest, float(dist[hit].min()))
        for g, gap in zip(geom[hit], dist[hit]):
            if len(self.contacts) < self.MAX_LISTED:
                r, e = (g[0], g[1]) if robot[g[0]] else (g[1], g[0])
                self.contacts.append([now] + _geom_names(model, (r, e)) + [float(gap)])

    def save(self, path, steps):
        """Write contacts.json: the physics steps, those with contact, the deepest, the spans and the first contacts."""
        with open(path, "w") as f:
            json.dump({"steps": steps, "contact_steps": self.contact_steps, "deepest": self.deepest,
                       "spans": self.spans, "first": self.contacts}, f, indent=1)


class NavRecorder(bringup_test.Recorder):
    """Record as test_sim_bringup's recorder does, plus contacts and the live pose."""

    def __init__(self, out):
        """Record into `out`."""
        super().__init__(out)
        self.contact_log = ContactLog()

    def step(self, iface):
        """Log contacts every step and the live pose every 100 steps, then record as the base does."""
        base = iface.resolver.base_body
        self.contact_log.step(iface.model, iface.data, base)
        if self.steps % 100 == 0:
            write_live_pose(self.out, iface.data, base)
        super().step(iface)

    def save(self):
        """Write sim.npz, then contacts.json, whose presence means the recording is complete."""
        super().save()
        self.contact_log.save(os.path.join(self.out, "contacts.json"), self.steps)


def run_sim(out, sim_args):
    """Run h12_mujoco.py --g1 with the recorder, the scene map and the goal hooked into SimInterface."""
    import runpy
    import signal

    sys.path.insert(0, bringup_test.SIM)
    # The G1's own classes: golem_robocasa's top-level unitree_interface is the H1-2's.
    from g1 import unitree_interface
    from g1.g1_sensors import SensorConfig

    with open(os.path.join(out, "sensors.json"), "w") as f:
        json.dump(SensorConfig.from_env()._asdict(), f, indent=1)
    recorder = NavRecorder(out)
    after_step = unitree_interface.SimInterface.after_step
    init = unitree_interface.SimInterface.__init__

    def recorded_after_step(self):
        """Publish rt/lowstate as the sim does, then record the step."""
        after_step(self)
        recorder.step(self)

    def mapped_init(self, model, data, resolver, *args, **kwargs):
        """Make the interface, then map the scene and pick the goal."""
        # The robot is at the spawn, in the sim's stance, before the first step and the bridges.
        init(self, model, data, resolver, *args, **kwargs)
        map_scene(out, model, data, resolver.base_body)

    def interrupt(signum, frame):
        """Turn SIGTERM into KeyboardInterrupt, so that the recording is saved."""
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


def _tf_parts(tf):
    """Return [stamp s, x, y, z, qx, qy, qz, qw] of a geometry_msgs/TransformStamped, or None."""
    if tf is None:
        return None
    t, r = tf.transform.translation, tf.transform.rotation
    return [tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9, t.x, t.y, t.z, r.x, r.y, r.z, r.w]


def _yaw_xyzw(x, y, z, w):
    """Return the yaw, rad, of an XYZW quaternion."""
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _process_cpu(names):
    """Return {name: CPU cores used on average since its process started} for the named nodes."""
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


class SimDriver:
    """Run a drive's ROS side; shared with tests/sim/test_sim_stack.py.

    A node named `name` with /clock (the waits run in sim time), a TF buffer filled on a node of
    its own, marks (<out>/events.json, printed as [name]) and the waits both tests use. A
    subclass adds its subscriptions and samples(now), which spin() calls after each callback.
    """

    def __init__(self, out, name):
        """Start the node, the /clock subscription and the TF listener."""
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from tf2_ros import Buffer, TransformListener
        rclpy.init()
        self.rclpy = rclpy
        self.name = name
        self.node = Node(name)
        self.out = out
        self.events, self.info = [], {}
        self.clock, self.clocks = None, 0
        self.node.create_subscription(Clock, "/clock", self._on_clock,
                                      QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT), raw=True)
        # TF on a node of its own, spun on the listener's thread: /tf runs at hundreds of Hz.
        self.tf_node = Node(f"{name}_tf")
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self.tf_node, spin_thread=True)
        self.launch_log = os.path.join(out, "launch.log")

    def _on_clock(self, raw):
        """Keep the sim time from a serialised /clock."""
        self.clock = bringup_test._cdr_stamp(raw)
        self.clocks += 1

    def _sim(self):
        """Return the sim time, s, or NaN before the first /clock."""
        return math.nan if self.clock is None else self.clock

    def mark(self, name, **extra):
        """Record a named event with its wall and sim times, and rewrite events.json."""
        self.events.append({"name": name, "wall": time.time(), "sim": self.clock, **extra})
        print(f"[{self.name}] {name} {extra if extra else ''}", flush=True)
        with open(os.path.join(self.out, "events.json"), "w") as f:
            json.dump({"events": self.events, "info": self.info}, f, indent=1)

    def lookup(self, target, source):
        """Return the latest target <- source transform, or None."""
        from rclpy.time import Time
        try:
            return self.tf.lookup_transform(target, source, Time())
        except Exception:
            return None

    def samples(self, now):
        """Take the subclass's periodic samples at wall time `now`; none here."""

    def spin(self, timeout=0.05):
        """Run one callback, then samples()."""
        self.rclpy.spin_once(self.node, timeout_sec=timeout)
        self.samples(time.time())

    def spin_for(self, seconds, wall_limit=120.0):
        """Spin for `seconds` of sim time; raise RuntimeError if the sim stalls (see test_sim_bringup)."""
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
        """Spin until `future` is done or `timeout` s of wall time pass; return its result or None."""
        end = time.time() + timeout
        while not future.done() and time.time() < end:
            self.spin()
        return future.result() if future.done() else None

    def _log_has(self, text):
        """Return whether launch.log contains `text`."""
        try:
            with open(self.launch_log, errors="replace") as f:
                return text in f.read()
        except OSError:
            return False

    def wait_band_released(self, timeout=120.0, refusals=()):
        """Wait for the sim to log the band's release; return False on a timeout or at once on a refusal.

        The robot's leg node asks for the release once its policy runs and frame_task_server is
        up; `refusals` are the launch's refusal lines in launch.log.
        """
        end = time.time() + timeout
        sim_log = os.path.join(self.out, "sim.log")
        while time.time() < end:
            if bringup_test.file_has(sim_log, bringup_test.BAND_RELEASED_LINE):
                self.mark("band_released")
                return True
            if any(self._log_has(refusal) for refusal in refusals):
                self.mark("launch_refused")
                return False
            for _ in range(20):
                self.spin()
        self.mark("band_timeout")
        return False

    def goal_in_map(self, goal):
        """Return the world goal (goal["goal"]: x, y, yaw) in the map frame.

        The sim's pelvis at one stamp (live_pose.json, written by the sim, never published)
        against odom -> pelvis at that stamp places odom in the world; map -> odom does the rest.
        """
        from rclpy.time import Time
        from tf2_ros import TransformException
        with open(os.path.join(self.out, "live_pose.json")) as source:
            truth = json.load(source)
        end = time.time() + 5.0
        while True:
            try:
                estimate = self.tf.lookup_transform("odom", "pelvis", Time(seconds=truth["stamp"]))
                break
            except TransformException:
                if time.time() >= end:
                    raise RuntimeError("no odometry at the recorded ground-truth pose stamp")
                self.spin()
        _, ex, ey, _, qx, qy, qz, qw = _tf_parts(estimate)
        world_odom = se2(*truth["pose"]) @ np.linalg.inv(se2(ex, ey, _yaw_xyzw(qx, qy, qz, qw)))
        map_odom = self.lookup("map", "odom")
        if map_odom is None:
            raise RuntimeError("no map -> odom in TF")
        _, x, y, _, qx, qy, qz, qw = _tf_parts(map_odom)
        t = se2(x, y, _yaw_xyzw(qx, qy, qz, qw)) @ np.linalg.inv(world_odom) @ se2(*goal["goal"])
        return se2_parts(t)

    def close(self):
        """Destroy the nodes and shut ROS down."""
        self.tf_node.destroy_node()
        self.node.destroy_node()
        self.rclpy.shutdown()


class Driver(SimDriver):
    """Drive the nav test: one Nav2 goal, recording /cmd_vel, TF, the scan path and the final /map."""

    def __init__(self, out, keep_lidar=False):
        """Subscribe to /cmd_vel, /map, /projected_map and the scan path; keep lidar frames with `keep_lidar`."""
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        super().__init__(out, "test_sim_nav")
        self.cmds = []  # wall, sim, vx, vy, vz, wx, wy, wz
        self.node.create_subscription(Twist, "/cmd_vel", self._on_cmd, 100)
        self.map = None
        self.node.create_subscription(OccupancyGrid, "/map", self._on_map, QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # The scan path, as its consumers hear it: the stamps of /livox/lidar and
        # /cloud_body_filtered (raw, not decoded), and every /converted_scan.
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan, PointCloud2
        from livox_ros_driver2.msg import CustomMsg
        self.projected_map = None
        self.node.create_subscription(OccupancyGrid, "/projected_map",
                                      lambda msg: setattr(self, "projected_map", msg), 1)
        self.arrivals = {"/livox/lidar": [], "/cloud_body_filtered": []}  # wall, sim, stamp
        # With keep_lidar, the /livox/lidar frames themselves from the goal on (lidar_frames.npz).
        self.keep_lidar, self.lidar_frames = keep_lidar, None
        for topic, kind in (("/livox/lidar", CustomMsg), ("/cloud_body_filtered", PointCloud2)):
            self.node.create_subscription(kind, topic, lambda raw, topic=topic: self._on_arrival(topic, raw),
                                          qos_profile_sensor_data, raw=True)
        self.scans = []  # wall, sim, stamp, angle_min, angle_increment, then the ranges
        self.node.create_subscription(LaserScan, "/converted_scan", self._on_scan, qos_profile_sensor_data)
        self._stall_dumped = False
        self.tf_samples = {"map_pelvis": [], "odom_pelvis": []}  # wall, sim, then _tf_parts
        self._next_sample = 0.0
        self.feedback = None

    def _on_cmd(self, msg):
        """Record a /cmd_vel."""
        self.cmds.append([time.time(), self._sim(), msg.linear.x, msg.linear.y, msg.linear.z,
                          msg.angular.x, msg.angular.y, msg.angular.z])

    def _on_map(self, msg):
        """Keep the latest /map."""
        self.map = msg

    def _on_arrival(self, topic, raw):
        """Record a scan-path message's arrival and stamp, and keep the lidar frame when asked."""
        self.arrivals[topic].append([time.time(), self._sim(), bringup_test._cdr_stamp(raw)])
        if topic == "/livox/lidar" and self.lidar_frames is not None:
            self.lidar_frames.append(bytes(raw))

    def _on_scan(self, msg):
        """Record a /converted_scan with its ranges."""
        self.scans.append([time.time(), self._sim(), msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                           msg.angle_min, msg.angle_increment] + list(msg.ranges))

    def _check_scan_path(self, now):
        """Record the scan path to <out>/stall.txt, once, if it stalls.

        A stall: /converted_scan stopped while /livox/lidar still arrives, or slam_toolbox's
        map -> odom stopped advancing while /converted_scan arrives. Records the ROS graph of its
        topics and, for each of its processes (and their consumers), per thread, the state,
        kernel wait channel and what changed in 1 s (CPU time, context switches), and its UDP
        sockets' queues and drops.
        """
        points, lidar = self.arrivals["/cloud_body_filtered"], self.arrivals["/livox/lidar"]
        if self._stall_dumped or not self.scans or not points or not lidar or self.clock is None:
            return
        scan_stalled = now - self.scans[-1][0] >= SCAN_STALL_S and now - lidar[-1][0] <= 1.0
        map_odom = _tf_parts(self.lookup("map", "odom"))
        slam_stalled = (map_odom is not None and self.clock - map_odom[0] >= SCAN_STALL_S
                        and now - self.scans[-1][0] <= 1.0)
        if not (scan_stalled or slam_stalled):
            return
        self._stall_dumped = True
        lines = [f"{'/converted_scan' if scan_stalled else 'map -> odom'} stalled. /converted_scan last heard "
                 f"{now - self.scans[-1][0]:.1f} s ago (stamp {self.scans[-1][2]:.2f}), /cloud_body_filtered "
                 f"{now - points[-1][0]:.1f} s ago (stamp "
                 f"{points[-1][2]:.2f}), /livox/lidar {now - self.arrivals['/livox/lidar'][-1][0]:.1f} s ago; map -> "
                 f"odom stamp {map_odom[0] if map_odom else None}; sim t {self.clock}"]
        for topic in ("/livox/lidar", "/cloud_body_filtered", "/converted_scan"):
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
            """Return {tid: (name, state, wait channel, CPU s, context switches)} of a process."""
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
                        udp[c[9]] = (c[1], tx, rx, int(c[-1]))  # inode: local address, tx, rx queue, drops
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

    def samples(self, now):
        """Sample map -> pelvis and odom -> pelvis and check the scan path, when due."""
        if now >= self._next_sample:
            self._next_sample = now + TF_PERIOD_S
            self._check_scan_path(now)
            for key, frames in (("map_pelvis", ("map", "pelvis")), ("odom_pelvis", ("odom", "pelvis"))):
                tf = self.lookup(*frames)
                if tf is not None:
                    parts = _tf_parts(tf)
                    rows = self.tf_samples[key]
                    if not rows or rows[-1][2] != parts[0]:  # a new stamp
                        rows.append([now, self._sim()] + parts)

    def wait_walking(self, timeout=240.0):
        """Wait for the walking node's 'walking: policy started'; return False on a timeout or a refused launch."""
        end = time.time() + timeout
        while time.time() < end:
            if self._log_has("walking: policy started"):
                self.mark("walking")
                return True
            if any(self._log_has(refusal) for refusal in bringup_test.LAUNCH_REFUSALS):
                self.mark("launch_refused")
                return False
            for _ in range(20):
                self.spin()
        self.mark("walking_timeout")
        return False

    def wait_nav(self, timeout=180.0):
        """Wait until Nav2 is ready; return whether it became ready within `timeout` s of wall time.

        Ready: Nav2's lifecycle nodes active, /navigate_to_pose served, map -> pelvis in TF, and
        a /projected_map and a /cloud_body_filtered heard.
        """
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
                    and self.lookup("map", "pelvis") is not None
                    and self.projected_map is not None
                    and self.arrivals["/cloud_body_filtered"]):
                self.mark("nav_active", states=states)
                return True
            for _ in range(20):
                self.spin()
        self.mark("nav_timeout", states=states, action_server=self.navigate.server_is_ready(),
                  map_pelvis=self.lookup("map", "pelvis") is not None)
        return False

    def navigate_to(self, x, y, yaw):
        """Send one NavigateToPose goal in the map frame and wait for its result; return its success.

        Gives up after GOAL_TIMEOUT_S of sim time or on a scan-path stall, cancelling the goal.
        """
        from action_msgs.msg import GoalStatus
        from nav2_msgs.action import NavigateToPose
        msg = NavigateToPose.Goal()
        msg.pose.header.frame_id = "map"  # stamp 0: the latest transform
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
        """Keep the goal's latest feedback."""
        f = msg.feedback
        self.feedback = {"distance_remaining": f.distance_remaining,
                         "navigation_time": f.navigation_time.sec + f.navigation_time.nanosec * 1e-9,
                         "number_of_recoveries": f.number_of_recoveries}

    def finish(self):
        """Save the final map, what publishes /cmd_vel and /odom, the scan path's CPU and the samples."""
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
                 points=np.array(self.arrivals["/cloud_body_filtered"], float).reshape(-1, 3),
                 **{k: np.array(v, float).reshape(-1, 10) for k, v in self.tf_samples.items()})
        self.mark("end")


def run_drive(out, keep_lidar=False):
    """Run the drive role; return 0 when the goal succeeded."""
    path = os.path.join(out, "goal.json")
    with open(path) as f:
        goal = json.load(f)
    d = Driver(out, keep_lidar)
    d.info["goal"] = goal
    ok = False
    try:
        d.mark("drive_start")
        if "error" in goal:
            d.mark("no_goal", error=goal["error"])
            return 1
        if d.wait_walking() and d.wait_band_released() and d.wait_nav():
            d.spin_for(STAND_S)
            x, y, yaw = d.goal_in_map(goal)
            ok = d.navigate_to(x, y, yaw)
            d.spin_for(SETTLE_S)
        return 0 if ok else 1
    finally:
        try:
            d.finish()  # what was recorded, whatever happened
        finally:
            d.close()


def _interp_pose(s, t):
    """Return the sim's pelvis at sim times `t`: (xyz, unwrapped yaw, the nearest sample's WXYZ quaternion).

    xyz and yaw are interpolated linearly between the recorded samples.
    """
    st = s["t"]
    yaw = np.unwrap(yaw_of(s["quat"]))
    xyz = np.stack([np.interp(t, st, s["pelvis"][:, i]) for i in range(3)], axis=-1)
    k = np.clip(np.searchsorted(st, t), 1, len(st) - 1)
    k = np.where(np.abs(st[k - 1] - t) < np.abs(st[k] - t), k - 1, k)  # the nearest sample
    return xyz, np.interp(t, st, yaw), s["quat"][k]


def _quat_xyzw_to_matrix(q):
    """Return the rotation matrices of XYZW quaternions."""
    x, y, z, w = np.asarray(q, float).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _quat_wxyz_to_matrix(q):
    """Return the rotation matrices of WXYZ quaternions."""
    q = np.asarray(q, float)
    return _quat_xyzw_to_matrix(np.concatenate([q[..., 1:], q[..., :1]], axis=-1))


class Report(bringup_test.Report):
    """Check a nav run from its recordings and print the numbers."""

    def __init__(self, out):
        """Load the run's recordings, the goal and the drive's samples."""
        super().__init__(out)
        with open(os.path.join(out, "goal.json")) as f:
            self.goal = json.load(f)
        drive = os.path.join(out, "drive.npz")
        self.drive = dict(np.load(drive)) if os.path.exists(drive) else None

    def pelvis_at(self, t):
        """Return the recorded pelvis at sim time `t` (see _interp_pose)."""
        return _interp_pose(self.s, np.asarray(t, float))

    def world_from(self, frame_row):
        """Return the SE(2) world <- frame transform from one TF sample (stamp, x, y, z, qx, qy, qz, qw) of frame -> pelvis."""
        stamp, x, y, _, qx, qy, qz, qw = frame_row
        xyz, yaw, _ = self.pelvis_at(stamp)
        return se2(xyz[0], xyz[1], yaw) @ np.linalg.inv(se2(x, y, _yaw_xyzw(qx, qy, qz, qw)))

    def nav_test(self):
        """Check the nav test: the goal, the stack's start, the goal's result and position, then the per-topic checks."""
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
        # The stack: the odometry the test asked for, the walking node, Nav2.
        odom_pubs = self.info.get("odom_publishers")
        self.check("no sim /odom", odom_pubs == [],
                   f"/odom publishers {odom_pubs} (expected none: FAST_LIO is the odometry)")
        release = self.release()
        if release is None:
            self.check("stack up", False, "the sim's elastic band was never released: see sim.log and launch.log")
            return
        r = release[0]
        self.numbers["band_release_t"] = release[1]
        for name in ("walking", "band_released", "nav_active", "goal_sent"):
            if name not in ev:
                self.check("stack up", False, f"no '{name}' mark: see drive.log and launch.log")
                return
        first = ev["drive_start"]["wall"]
        self.numbers["stack_up_wall_s"] = {n: ev[n]["wall"] - first
                                           for n in ("walking", "band_released", "nav_active")}
        self.check("stack up", True,
                   f"band released at sim t {release[1]:.2f} s; the drive saw the walking node walk at sim t "
                   f"{ev['walking']['sim']:.1f} s, the band's release at {ev['band_released']['sim']:.1f} s and Nav2 "
                   f"active at {ev['nav_active']['sim']:.1f} s ({ev['walking']['wall'] - first:.0f}, "
                   f"{ev['band_released']['wall'] - first:.0f} and {ev['nav_active']['wall'] - first:.0f} s wall "
                   f"after it started, beside the launch)")
        # The goal.
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
        # Where the goal is in the world: the map frame as it was when it was sent.
        world_map = self.world_from(sent["map_pelvis"])
        goal_map = sent["goal_map"]
        goal_world = world_map @ se2(*goal_map)
        gwx, gwy, gwyaw = se2_parts(goal_world)
        self.numbers["goal_sent_world_vs_picked_m"] = float(math.hypot(gwx - gx, gwy - gy))
        # The drive's last mark: the sim stops soon after, with the robot standing.
        last = max(ev.values(), key=lambda e: e["wall"])["name"]
        i_end = self.at("goal_result") if "goal_result" in ev else self.at(last)
        end_xyz, end_yaw, _ = self.pelvis_at(s["t"][i_end])
        err = float(math.hypot(end_xyz[0] - gx, end_xyz[1] - gy))
        yaw_err = float(math.degrees(_wrap(end_yaw - gyaw)))
        nav = result.get("map_pelvis")
        nav_err = (float(math.hypot(nav[1] - goal_map[0], nav[2] - goal_map[1])) if nav else math.nan)
        nav_yaw = (float(math.degrees(_wrap(_yaw_xyzw(*nav[4:]) - goal_map[2]))) if nav else math.nan)
        start_xyz, _, _ = self.pelvis_at(sent["sim"])
        # And once the robot has stood SETTLE_S.
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
        # The pelvis per phase, for the record (run() checks the whole recording).
        a, b = self.at("goal_sent"), i_end
        self.phases([("under the band", 0, r), ("standing, walking node", r, a), ("navigating", a, b + 1),
                     ("after the result", b + 1, len(s["t"]))])
        self.cmd_vel()
        self.scan_path(sent["sim"], s["t"][i_end])
        self.map_check(world_map)
        self.scan_phantoms()
        self.drift(sent["sim"], s["t"][i_end])

    def scan_path(self, t0, t1):
        """Check /converted_scan kept coming while navigating, and record where the scan path stopped if it did."""
        if self.drive is None or "scans" not in self.drive or not len(self.drive["scans"]):
            self.check("scan path", False, "no /converted_scan heard")
            return
        numbers, text = {}, []
        for key, label in (("lidar", "/livox/lidar"), ("points", "/cloud_body_filtered"), ("scans", "/converted_scan")):
            rows = self.drive[key]
            stamps = rows[:, 2]
            inside = stamps[(stamps >= t0) & (stamps <= t1)]
            # The gaps over the window, its edges included.
            edges = np.concatenate([[t0], inside, [t1]])
            gap = float(np.diff(edges).max()) if len(edges) > 1 else math.inf
            numbers[label] = {"count": int(len(rows)), "in_window": int(len(inside)), "longest_gap_s": gap,
                              "last_stamp": float(stamps.max()) if len(stamps) else None}
            text.append(f"{label} {len(inside)} (longest gap {gap:.2f} s)")
        self.numbers["scan_path"] = numbers
        gap = numbers["/converted_scan"]["longest_gap_s"]
        stall = os.path.exists(os.path.join(self.out, "stall.txt"))
        self.check("scan path", gap <= MAX_SCAN_GAP_S,
                   f"while navigating (sim t {t0:.1f}-{t1:.1f} s): " + ", ".join(text)
                   + f"; limit {MAX_SCAN_GAP_S} s for /converted_scan" + ("; see stall.txt" if stall else ""))

    def scan_phantoms(self):
        """Check the /converted_scan returns against what the sim has in the scan's height band.

        Every return within SCAN_PHANTOM_RANGE is placed in the world with the sim's true pelvis
        at the scan's stamp (the scan is in base_footprint).
        """
        if self.drive is None or "scans" not in self.drive or not len(self.drive["scans"]):
            self.check("scan has no phantoms", False, "no /converted_scan heard")
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
        placed = np.where(ok, ranges, 0.0)  # no inf or NaN into the integer cells below
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
        """Check the pelvis' height and tilt from the sim over the whole recording."""
        z, tilt, t = self.s["pelvis"][:, 2], bringup_test._tilt_deg(self.s["upz"]), self.s["t"]
        self.numbers["upright"] = {"z_min": float(z.min()), "z_max": float(z.max()),
                                   "tilt_max_deg": float(tilt.max()), "sim_t": [float(t[0]), float(t[-1])]}
        self.check("never fell", z.min() >= MIN_PELVIS_Z and tilt.max() <= MAX_TILT_DEG,
                   f"the whole recording (sim t {t[0]:.1f}-{t[-1]:.1f} s): pelvis z {z.min():.3f}-"
                   f"{z.max():.3f} m, tilt max {tilt.max():.1f} deg (limits z >= {MIN_PELVIS_Z}, tilt <= "
                   f"{MAX_TILT_DEG})")

    def contacts(self):
        """Check no robot-environment contact over the whole recording."""
        path = os.path.join(self.out, "contacts.json")
        if not os.path.exists(path):
            self.check("no contact", False, "no contacts.json from the sim")
            return
        with open(path) as f:
            c = json.load(f)
        # The sim lists the first MAX_LISTED contacts, in time order.
        self.numbers["contacts"] = {"steps": c["steps"], "contact_steps": c["contact_steps"],
                                    "deepest": c["deepest"], "first": c["first"][:5]}
        if c["contact_steps"]:
            text = (f"{c['contact_steps']} physics steps with contact, deepest {c['deepest'] * 1000:.1f} mm, from "
                    f"sim t {c['first'][0][0]:.2f} s: {c['first'][:3]}")
        else:
            text = "no robot geom penetrated anything but the floor"
        self.check("no contact", not c["contact_steps"], f"{c['steps']} physics steps: {text}")

    def fall_log(self):
        """Check the sim's fall logger (golem_robocasa/g1/g1_mujoco.py), which looks at every physics step, logged no fall."""
        path = os.path.join(self.out, "sim.log")
        if not os.path.exists(path):
            self.check("no fall logged", False, "no sim.log")
            return
        with open(path, errors="replace") as f:
            falls = [line.strip() for line in f if "ROBOT FELL" in line]
        self.check("no fall logged", not falls, falls[0] if falls else "sim.log logs no fall")

    def cmd_vel(self):
        """Check every /cmd_vel is inside the speed caps and the velocity smoother is its only publisher."""
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
        """Check every occupied cell of the final /map is within MAP_TOL of something the sim has in the band."""
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
        # Distance from each cell of the sim's grid to the nearest band cell.
        band, origin, gres = scene["band"], scene["origin"], float(scene["res"])
        to_band = ndimage.distance_transform_edt(~band) * gres
        ix = np.floor((world[0] - origin[0]) / gres).astype(int)
        iy = np.floor((world[1] - origin[1]) / gres).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < band.shape[1]) & (iy < band.shape[0])
        dist = np.full(len(rows), np.inf)
        dist[inside] = to_band[iy[inside], ix[inside]]
        far = dist > MAP_TOL + 1e-9  # the grid's distances are multiples of 5 cm, in floats
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
        """Record the odometry's and the localisation's errors against the sim, and check the localisation's x, y.

        The odometry is odom -> pelvis and the localisation map -> pelvis; the check covers sim
        times t0 to t1, while navigating.
        """
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
            # The frame in the world from the first sample (odom is level, the floor z = 0).
            frame = self.world_from(rows[0, 2:])
            local = np.linalg.inv(frame) @ np.stack([xyz[:, 0], xyz[:, 1], np.ones(len(rows))])
            dyaw = math.atan2(frame[1, 0], frame[0, 0])
            xy_err = np.hypot(rows[:, 3] - local[0], rows[:, 4] - local[1])
            z_err = rows[:, 5] - xyz[:, 2]
            yaw_est = np.array([_yaw_xyzw(*q) for q in rows[:, 6:10]])
            yaw_err = np.degrees(_wrap(yaw_est - (yaw - dyaw)))
            # Tilt: the estimated pelvis z axis against the true one, turned into the frame.
            up_est = _quat_xyzw_to_matrix(rows[:, 6:10])[:, :, 2]
            up_true = _quat_wxyz_to_matrix(quat)[:, :, 2]
            c, si = math.cos(dyaw), math.sin(dyaw)
            up_true = up_true @ np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])  # R(-dyaw) u
            tilt_err = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", up_est, up_true), -1.0, 1.0)))
            path = float(np.sum(np.hypot(np.diff(xyz[:, 0]), np.diff(xyz[:, 1]))))
            numbers = {"samples": int(len(rows)), "sim_t": [float(stamp[0]), float(stamp[-1])],
                       "path_m": path, "xy_err_max": float(xy_err.max()), "xy_err_end": float(xy_err[-1]),
                       "yaw_err_deg_max_abs": float(np.abs(yaw_err).max()), "yaw_err_deg_end": float(yaw_err[-1])}
            text = (f"xy error max {xy_err.max() * 100:.1f} cm, at the end {xy_err[-1] * 100:.1f} cm; yaw error max "
                    f"{np.abs(yaw_err).max():.2f} deg, at the end {yaw_err[-1]:+.2f} deg")
            if key == "odom_pelvis":
                # odom is level with the floor at z = 0, so its z and tilt are compared with the sim's
                # too; for map -> pelvis only x, y and yaw are.
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
        # Drift of slam_toolbox's map -> odom against the sim's truth shows here.
        worst = (self.numbers.get("localisation") or {}).get("xy_err_max_navigating")
        window = f"while navigating (sim t {t0:.1f}-{t1:.1f} s)"
        self.check("localisation", worst is not None and worst <= MAX_LOCALISATION_ERR,
                   f"map -> pelvis against the sim's truth {window}: x, y error max {worst * 100:.2f} cm (limit "
                   f"{MAX_LOCALISATION_ERR * 100:.1f} cm)" if worst is not None
                   else f"no map -> pelvis sample {window}")

    def run(self, out):
        """Run every check, write report.json and print the verdict; return 0 if all passed."""
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
        # However far the drive got: the sim stops before the bringup, so the robot stands to the end.
        self.upright()
        self.contacts()
        self.fall_log()
        self.no_estop()
        ok = all(c["pass"] for c in self.checks)
        with open(os.path.join(out, "report.json"), "w") as f:
            json.dump({"pass": ok, "checks": self.checks, "numbers": self.numbers}, f, indent=1)
        print(f"{'PASSED' if ok else 'FAILED'}: {sum(c['pass'] for c in self.checks)}/{len(self.checks)} checks")
        return 0 if ok else 1


def main():
    """Run the role the command line names; return its exit status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=("sim", "drive", "report"))
    parser.add_argument("--out", required=True, help="the shared output directory")
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
        return run_drive(args.out, args.keep_lidar)
    return Report(args.out).run(args.out)


if __name__ == "__main__":
    sys.exit(main())
