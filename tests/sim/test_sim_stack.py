"""Closed-loop test of either robot's whole stack in the RoboCasa sim: the robot
stands unsupported once the elastic band is off, Nav2 walks it to a goal ahead
while /named_config moves its arms, and SLAM stays healthy throughout. The parts
that run in the containers; test_sim_stack.sh runs them and has the usage.

  sim     in the robocasa container: h12_mujoco.py as __main__ for GOLEM_ROBOT's
          robot (--h12 or --g1), recording from the sim's own state: after every
          physics step the elastic band's state and any penetrating contact between
          a robot geom and anything but the floor, and every RECORD_EVERY steps the
          pelvis (<out>/sim.npz, contacts.json). At startup, with the robot at the
          spawn, it maps the kitchen from the sim's geometry (<out>/scene.npz) and
          places the goal GOAL_AHEAD m ahead of the pelvis (<out>/goal.json). It
          stops on <out>/stop and saves sim.npz, then contacts.json
  drive   in the ros container, beside the robot's sim bringup with navigation:
          waits until the band is off and Nav2's lifecycle nodes are active, stands
          STAND_S, sends one NavigateToPose goal, and once the robot walks sends
          /named_config 't_pose', then 'home' once that has succeeded (each with a
          duration of ARM_GOAL_S); after the goal it stands SETTLE_S and waits for a
          /map published after the walk. Throughout it records /Odometry,
          /converted_scan, /map, the odom -> base_footprint TF, /cmd_vel and the
          publishers of /lowcmd (<out>/drive.npz, events.json)
  report  anywhere with numpy and scipy: checks the run, prints the numbers and
          writes <out>/report.json; exit 1 on a failure

The goal. The sim maps the kitchen around the spawn from the world bounding box
of every geom that is not the robot's or the floor's, visual or colliding,
between 2 cm and the robot's height (ROBOTS), on a 5 cm grid. The goal is GOAL_AHEAD m
from the pelvis along its heading, facing that way; the sim refuses it
(goal.json's "error", and the test fails) unless it is in open floor: at least
GOAL_CLEARANCE m from every occupied cell and reachable from the spawn through
cells PATH_CLEARANCE m clear. The drive sends it in the map frame: through odom
as FAST_LIO's odometry defines it, matched to the sim's pelvis at one stamp,
then slam_toolbox's map -> odom.

Pass if, in one run:
  - the stack comes up: the band is released and Nav2 is active before the goal;
  - the robot stands unsupported: from the band's release to the end of the
    recording the band stays off, the pelvis stays above the robot's
    min_pelvis_z and tilts at most MAX_TILT_DEG, and no robot geom penetrates
    anything but the floor (contacts under the band are reported); the sim's
    fall logger (sim.log) logs no fall;
  - the goal succeeds within GOAL_TIMEOUT_S of sim time, and the true pelvis ends
    within FINAL_TOL of it, at the result and SETTLE_S later;
  - while walking to it, 't_pose' and then 'home' succeed: 't_pose' sent once
    /cmd_vel moves the robot and before the goal's result, 'home' once 't_pose'
    has succeeded (each move takes seconds and the walk a few, so 'home' may end
    after the goal's result);
  - SLAM is healthy from Nav2's start to the drive's end: /Odometry at
    MIN_ODOM_HZ or more in every sim second, the odom -> base_footprint TF present
    at every sample and at most MAX_TF_AGE_S old, the known cells of /map more
    at the end than at Nav2's start, and /converted_scan's stamps at most
    MAX_SCAN_GAP_S apart;
  - safety_node is the only publisher of /lowcmd at every sample from its start
    (no other publisher at any sample), golem_safety (launch.log) latches no
    estop before the drive's end, and nothing publishes the sim's ground-truth
    /odom.

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
# the G1 tests' recorder helpers, geometry and log parsing (tests/g1)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "g1"))
import test_sim_bringup as bringup_test  # noqa: E402
import test_sim_nav as nav_test  # noqa: E402

SIM = bringup_test.SIM
RECORD_EVERY = bringup_test.RECORD_EVERY   # physics steps per sample (50 Hz)
# per robot: the pelvis's lowest height while standing or walking, below each
# robot's stand: HOMIE stands the G1 at 0.744 m (tests/g1); ALMI stands the H1-2
# at 0.977 m, and tests/stack's check_lowerbody allows 0.20 m below the stand.
# The robot's height, m: the top of the band the goal's clearance is taken over
ROBOTS = {"h12": {"min_pelvis_z": 0.77, "height": 1.80},
          "g1": {"min_pelvis_z": bringup_test.MIN_PELVIS_Z, "height": 1.40}}
# the goal, from the sim's geometry
GOAL_AHEAD = 1.5           # m from the spawn's pelvis along its heading
GOAL_CLEARANCE = 0.75      # m to the nearest occupied cell: the footprint's half diagonal + inflation 0.4
PATH_CLEARANCE = nav_test.PATH_CLEARANCE   # 0.35 m
# drive
NAV2_NODES = nav_test.NAV2_NODES
STAND_S = 3.0              # sim s standing before the goal, Nav2 up
SETTLE_S = 2.0             # sim s after the goal's result
GOAL_TIMEOUT_S = 90.0      # sim s from the goal to its result
# s: each /named_config goal's duration, frame_task_server's timeout for it (its default,
# controller.timeout, 5 s, ends the H1-2's move to 't_pose' before its IK converges)
ARM_GOAL_S = 15
ARM_TIMEOUT_S = 60.0       # wall s for each /named_config goal's result
# sim s after the goal's result to wait for a /map that slam_toolbox published after the walk
# (it publishes every map_update_interval, 5 s, and adds a scan every minimum_travel_distance, 0.5 m)
MAP_WAIT_S = 12.0
WALKING_CMD = 0.05         # m/s or rad/s: a /cmd_vel this large has the robot walking
TF_PERIOD_S = 0.1          # wall s between TF samples
GRAPH_PERIOD_S = 1.0       # wall s between samples of /lowcmd's publishers
# pass limits
MAX_TILT_DEG = bringup_test.MAX_TILT_DEG   # 15 deg
FINAL_TOL = 0.4            # m: the true pelvis to the goal (Nav2's xy tolerance 0.25)
MIN_ODOM_HZ = 5.0          # /Odometry messages in every sim second (FAST_LIO runs at the lidar's 10 Hz)
MAX_TF_AGE_S = 0.5         # sim s: odom -> base_footprint's stamp behind /clock
MAX_SCAN_GAP_S = nav_test.MAX_SCAN_GAP_S   # 0.5 sim s: the costmaps' expected_update_rate
LOWCMD_WRITER = "safety_node"


def robot_from_env():
    robot = os.environ.get("GOLEM_ROBOT", "").strip().lower()
    if robot not in ROBOTS:
        raise SystemExit(f"GOLEM_ROBOT must be one of {sorted(ROBOTS)}, not '{robot}'")
    return robot


# --------------------------------------------------------------------------- #
# sim: h12_mujoco.py with a recorder, a contact check and the goal
# --------------------------------------------------------------------------- #
def map_scene(out, robot, model, data, base_body):
    """At the spawn: the kitchen's occupancy (scene.npz) and the goal (goal.json)"""
    from scipy import ndimage
    is_robot, floor, low, high = nav_test.robot_and_floor(model, data, base_body)
    visual = model.geom_group == 1
    colliding = (model.geom_contype > 0) | (model.geom_conaffinity > 0)
    env = ~is_robot & ~floor
    pelvis = data.xpos[base_body].copy()
    yaw = float(nav_test.yaw_of(data.xquat[base_body]))
    res = nav_test.GRID_RES
    n = int(round(2 * nav_test.GRID_HALF / res))
    origin = np.round((pelvis[:2] - nav_test.GRID_HALF) / res) * res
    z_range = (0.02, ROBOTS[robot]["height"])
    solid = nav_test.raster(low, high, env & (visual | colliding), z_range, origin, (n, n))
    clearance = ndimage.distance_transform_edt(~solid) * res

    def cell(xy):
        return tuple(np.floor((np.asarray(xy)[::-1] - origin[::-1]) / res).astype(int))

    spawn_cell = cell(pelvis[:2])
    labels, _ = ndimage.label(clearance >= PATH_CLEARANCE)
    reachable = (labels == labels[spawn_cell]) & (labels[spawn_cell] > 0)
    gx, gy = pelvis[0] + GOAL_AHEAD * math.cos(yaw), pelvis[1] + GOAL_AHEAD * math.sin(yaw)
    goal_cell = cell((gx, gy))
    np.savez(os.path.join(out, "scene.npz"), origin=origin, res=res, solid=solid, clearance=clearance,
             reachable=reachable, spawn=pelvis, spawn_yaw=yaw)
    goal = {"robot": robot, "spawn": [float(v) for v in pelvis], "spawn_yaw": yaw,
            "spawn_clearance": float(clearance[spawn_cell]), "grid_origin": origin.tolist(), "res": res,
            "goal_z": list(z_range), "goal": [gx, gy, yaw], "goal_clearance": float(clearance[goal_cell]),
            "goal_distance": GOAL_AHEAD}
    if not reachable[goal_cell]:
        goal["error"] = (f"the goal {GOAL_AHEAD} m ahead ({gx:.2f}, {gy:.2f}) is not reachable from the spawn "
                         f"through cells {PATH_CLEARANCE} m clear")
    elif goal["goal_clearance"] < GOAL_CLEARANCE:
        goal["error"] = (f"the goal {GOAL_AHEAD} m ahead ({gx:.2f}, {gy:.2f}) is {goal['goal_clearance']:.2f} m "
                         f"from the nearest obstacle, not in open floor ({GOAL_CLEARANCE} m)")
    with open(os.path.join(out, "goal.json"), "w") as f:
        json.dump(goal, f, indent=1)
    print(f"[test_sim_stack] {robot}: spawn pelvis ({pelvis[0]:.3f}, {pelvis[1]:.3f}), yaw {math.degrees(yaw):.1f} "
          f"deg, {goal['spawn_clearance']:.2f} m of room; goal ({gx:.3f}, {gy:.3f}), "
          f"{goal['goal_clearance']:.2f} m of room" + (f"; {goal['error']}" if "error" in goal else ""), flush=True)


class Recorder:
    """The robot's state from the sim, after every physics step (the sim lock is held)"""

    MAX_LISTED = 50

    def __init__(self, out):
        self.out = out
        self.stop_file = os.path.join(out, "stop")
        self.band = None                # the sim's ElasticBand, from its MeasurementBridge
        self.rows = {k: [] for k in ("t", "wall", "pelvis", "quat", "upz")}
        self.steps = 0
        self.band_changes = []          # sim t, wall, state (1 on, 0 off, NaN no band) at each change
        self._band_state = None
        self._base = None
        self._masks = None
        self.contact_steps = 0          # physics steps with a penetrating robot-env contact
        self.contact_spans = []         # [first sim t, last sim t, steps] of consecutive such steps
        self.contacts = []              # the first MAX_LISTED: [sim t, robot geom, env geom, dist]
        self.deepest = 0.0

    def step(self, iface):
        m, d = iface.model, iface.data
        self.steps += 1
        if self._base is None:
            self._base = iface.resolver.body_id("pelvis")
            robot, floor, _, _ = nav_test.robot_and_floor(m, d, self._base)
            self._masks = (robot, ~robot & ~floor)
        on = None if self.band is None else bool(self.band.enabled)
        if not self.band_changes or on != self._band_state:
            self._band_state = on
            self.band_changes.append([float(d.time), time.time(), math.nan if on is None else float(on)])
        n = d.ncon
        if n:
            robot, env = self._masks
            geom = d.contact.geom[:n]
            dist = d.contact.dist[:n]
            hit = (dist < 0) & ((robot[geom[:, 0]] & env[geom[:, 1]]) | (robot[geom[:, 1]] & env[geom[:, 0]]))
            if hit.any():
                self.contact_steps += 1
                now = float(d.time)
                if self.contact_spans and now - self.contact_spans[-1][1] <= 1.5 * m.opt.timestep:
                    self.contact_spans[-1][1:] = [now, self.contact_spans[-1][2] + 1]
                else:
                    self.contact_spans.append([now, now, 1])
                self.deepest = min(self.deepest, float(dist[hit].min()))
                for g, gap in zip(geom[hit], dist[hit]):
                    if len(self.contacts) < self.MAX_LISTED:
                        r, e = (g[0], g[1]) if robot[g[0]] else (g[1], g[0])
                        self.contacts.append([float(d.time)] + nav_test._geom_names(m, (r, e)) + [float(gap)])
        base = self._base
        if self.steps % 100 == 0:
            # the pelvis at a stamp, for the drive to place the goal in the map frame
            rot = d.xmat[base].reshape(3, 3)
            snapshot = {"stamp": float(d.time),
                        "pose": [float(d.xpos[base][0]), float(d.xpos[base][1]),
                                 float(math.atan2(rot[1, 0], rot[0, 0]))]}
            temporary = os.path.join(self.out, "live_pose.tmp")
            with open(temporary, "w") as target:
                json.dump(snapshot, target)
            os.replace(temporary, os.path.join(self.out, "live_pose.json"))
        if self.steps % 500 == 0 and os.path.exists(self.stop_file):
            raise bringup_test.StopRecording()
        if self.steps % RECORD_EVERY:
            return
        rows = self.rows
        rows["t"].append(float(d.time))
        rows["wall"].append(time.time())
        rows["pelvis"].append(d.xpos[base].copy())
        rows["quat"].append(d.xquat[base].copy())
        rows["upz"].append(float(d.xmat[base][8]))

    def save(self):
        arrays = {k: np.array(v) for k, v in self.rows.items()}
        # written under another name and renamed, so a file named sim.npz is complete
        tmp = os.path.join(self.out, "sim.tmp.npz")
        np.savez(tmp, hz=500.0 / RECORD_EVERY, band_changes=np.array(self.band_changes, float).reshape(-1, 3),
                 **arrays)
        os.replace(tmp, os.path.join(self.out, "sim.npz"))
        with open(os.path.join(self.out, "contacts.json"), "w") as f:
            json.dump({"steps": self.steps, "contact_steps": self.contact_steps, "deepest": self.deepest,
                       "spans": self.contact_spans, "first": self.contacts}, f, indent=1)
        print(f"[test_sim_stack] saved {len(self.rows['t'])} samples to {self.out}/sim.npz", flush=True)


def run_sim(out, sim_args):
    import importlib
    import inspect
    import runpy
    import signal

    robot = robot_from_env()
    sys.path.insert(0, SIM)
    # each robot's own SimInterface: golem_robocasa's top-level unitree_interface is the H1-2's
    interface = importlib.import_module("unitree_interface" if robot == "h12" else "g1.unitree_interface")
    import measurement_bridge

    recorder = Recorder(out)
    after_step = interface.SimInterface.after_step
    init = interface.SimInterface.__init__
    bridge_init = measurement_bridge.MeasurementBridge.__init__

    def recorded_after_step(self):
        after_step(self)
        recorder.step(self)

    def mapped_init(self, *args, **kwargs):
        # the robot is at the spawn, in the sim's stance, before the first step and the bridges
        init(self, *args, **kwargs)
        map_scene(out, robot, self.model, self.data, self.resolver.body_id("pelvis"))

    def band_init(self, *args, **kwargs):
        # both sims hand their elastic band to the bridge that serves /elastic_band/toggle
        recorder.band = inspect.signature(bridge_init).bind(self, *args, **kwargs).arguments.get("elastic_band")
        bridge_init(self, *args, **kwargs)

    def interrupt(signum, frame):
        raise KeyboardInterrupt()

    interface.SimInterface.after_step = recorded_after_step
    interface.SimInterface.__init__ = mapped_init
    measurement_bridge.MeasurementBridge.__init__ = band_init
    signal.signal(signal.SIGTERM, interrupt)
    script = os.path.join(SIM, "h12_mujoco.py")
    sys.argv = [script, f"--{robot}"] + sim_args
    try:
        runpy.run_path(script, run_name="__main__")
    except (bringup_test.StopRecording, KeyboardInterrupt):
        print("[test_sim_stack] sim stopped", flush=True)
    finally:
        recorder.save()


# --------------------------------------------------------------------------- #
# drive: the goal and the arm moves against the running bringup
# --------------------------------------------------------------------------- #
def _known_cells(msg):
    return int(np.count_nonzero(np.asarray(msg.data, np.int8) >= 0))


class Driver:
    def __init__(self, out):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import OccupancyGrid, Odometry
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan
        from tf2_ros import Buffer, TransformListener
        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("test_sim_stack")
        self.out = out
        self.events, self.info = [], {}
        self.clock, self.clocks = None, 0
        self.node.create_subscription(Clock, "/clock", self._on_clock,
                                      QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT), raw=True)
        self.cmds = []       # wall, sim, vx, vy, wz
        self.walking = None  # the first /cmd_vel that moves the robot once the goal is sent
        self.node.create_subscription(Twist, "/cmd_vel", self._on_cmd, 100)
        self.maps = []       # wall, sim, stamp, known cells, width, height
        self.node.create_subscription(OccupancyGrid, "/map", self._on_map, QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # stamps only: wall, sim, stamp
        self.odometry, self.scans = [], []
        self.node.create_subscription(Odometry, "/Odometry", lambda raw: self._stamp(self.odometry, raw),
                                      QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE), raw=True)
        self.node.create_subscription(LaserScan, "/converted_scan", lambda raw: self._stamp(self.scans, raw),
                                      qos_profile_sensor_data, raw=True)
        # TF on a node of its own, spun on the listener's thread: /tf runs at hundreds of Hz
        self.tf_node = Node("test_sim_stack_tf")
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self.tf_node, spin_thread=True)
        self.base_tf = []    # wall, sim, odom -> base_footprint's stamp (NaN: none)
        self.lowcmd_writers = []   # wall, sim, the sorted node names publishing /lowcmd
        self._next_tf = self._next_graph = 0.0
        self.launch_log = os.path.join(out, "launch.log")
        self.sim_log = os.path.join(out, "sim.log")

    def _on_clock(self, raw):
        self.clock = bringup_test._cdr_stamp(raw)
        self.clocks += 1

    def _sim(self):
        return math.nan if self.clock is None else self.clock

    def _stamp(self, rows, raw):
        rows.append([time.time(), self._sim(), bringup_test._cdr_stamp(raw)])

    def _on_cmd(self, msg):
        self.cmds.append([time.time(), self._sim(), msg.linear.x, msg.linear.y, msg.angular.z])
        if (self.walking is None and "goal_sent" in {e["name"] for e in self.events}
                and max(abs(msg.linear.x), abs(msg.linear.y), abs(msg.angular.z)) >= WALKING_CMD):
            self.walking = self.cmds[-1]

    def _on_map(self, msg):
        self.maps.append([time.time(), self._sim(), msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                          _known_cells(msg), msg.info.width, msg.info.height])

    def mark(self, name, **extra):
        self.events.append({"name": name, "wall": time.time(), "sim": self.clock, **extra})
        print(f"[test_sim_stack] {name} {extra if extra else ''}", flush=True)
        with open(os.path.join(self.out, "events.json"), "w") as f:
            json.dump({"events": self.events, "info": self.info}, f, indent=1)

    def lookup(self, target, source):
        from rclpy.time import Time
        try:
            return self.tf.lookup_transform(target, source, Time())
        except Exception:
            return None

    def spin(self, timeout=0.05):
        """One callback, and the samples when they are due"""
        self.rclpy.spin_once(self.node, timeout_sec=timeout)
        now = time.time()
        if now >= self._next_tf:
            self._next_tf = now + TF_PERIOD_S
            tf = nav_test._tf_parts(self.lookup("odom", "base_footprint"))
            self.base_tf.append([now, self._sim(), math.nan if tf is None else tf[0]])
        if now >= self._next_graph:
            self._next_graph = now + GRAPH_PERIOD_S
            self.lowcmd_writers.append([now, self._sim(), sorted(
                i.node_name for i in self.node.get_publishers_info_by_topic("/lowcmd"))])

    def spin_for(self, seconds, wall_limit=120.0):
        """Spin for `seconds` of sim time; fail if the sim stalls"""
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

    def wait_band_released(self, timeout=300.0):
        """The sim logs the band's release, which the robot's leg node asks for once its
        policy runs and frame_task_server is up"""
        end = time.time() + timeout
        while time.time() < end:
            if bringup_test.file_has(self.sim_log, bringup_test.BAND_RELEASED_LINE):
                self.mark("band_released")
                return True
            if any(bringup_test.file_has(self.launch_log, refusal) for refusal in bringup_test.LAUNCH_REFUSALS):
                self.mark("launch_refused")
                return False
            for _ in range(20):
                self.spin()
        self.mark("band_timeout")
        return False

    def wait_nav(self, timeout=180.0):
        """Nav2's lifecycle nodes active, /navigate_to_pose and /named_config served,
        map -> base_footprint in TF, a /map and a /converted_scan heard"""
        from custom_ros_messages.action import NamedConfig
        from lifecycle_msgs.msg import State
        from lifecycle_msgs.srv import GetState
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        self.navigate = ActionClient(self.node, NavigateToPose, "navigate_to_pose")
        self.named_config = ActionClient(self.node, NamedConfig, "named_config")
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
                    and self.named_config.server_is_ready() and self.lookup("map", "base_footprint") is not None
                    and self.maps and self.scans):
                self.mark("nav_active", states=states)
                return True
            for _ in range(20):
                self.spin()
        self.mark("nav_timeout", states=states, action_server=self.navigate.server_is_ready(),
                  named_config=self.named_config.server_is_ready(),
                  map_base=self.lookup("map", "base_footprint") is not None, maps=len(self.maps),
                  scans=len(self.scans))
        return False

    def goal_in_map(self, goal):
        """The world goal in the map frame: the sim's pelvis at one stamp (live_pose.json,
        written by the sim, never published) against odom -> pelvis at that stamp, then
        map -> odom"""
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
                    raise RuntimeError("no odometry at the recorded pose's stamp")
                self.spin()
        _, ex, ey, _, qx, qy, qz, qw = nav_test._tf_parts(estimate)
        se2, yaw_of = nav_test.se2, nav_test._yaw_xyzw
        world_odom = se2(*truth["pose"]) @ np.linalg.inv(se2(ex, ey, yaw_of(qx, qy, qz, qw)))
        map_odom = self.lookup("map", "odom")
        if map_odom is None:
            raise RuntimeError("no map -> odom in TF")
        _, x, y, _, qx, qy, qz, qw = nav_test._tf_parts(map_odom)
        t = se2(x, y, yaw_of(qx, qy, qz, qw)) @ np.linalg.inv(world_odom) @ se2(*goal["goal"])
        return nav_test.se2_parts(t)

    def _send_named(self, name):
        from custom_ros_messages.action import NamedConfig
        goal = NamedConfig.Goal()
        goal.config_name, goal.plan = name, False
        goal.duration.sec = ARM_GOAL_S
        self.mark(f"{name}_sent")
        self.arm_error = None

        def feedback(msg):
            self.arm_error = float(msg.feedback.joint_error)
        return self.named_config.send_goal_async(goal, feedback_callback=feedback), time.time()

    def navigate_with_arms(self, x, y, yaw):
        """Sends the goal; once /cmd_vel moves the robot and while the goal runs, 't_pose',
        then 'home' once 't_pose' has succeeded; waits for the goal's result and the arm goals"""
        from action_msgs.msg import GoalStatus
        from nav2_msgs.action import NavigateToPose
        msg = NavigateToPose.Goal()
        msg.pose.header.frame_id = "map"   # stamp 0: the latest transform
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.mark("goal_sent", goal_map=[x, y, yaw],
                  map_base=nav_test._tf_parts(self.lookup("map", "base_footprint")))
        handle = self.wait_future(self.navigate.send_goal_async(msg), 10.0)
        if handle is None or not handle.accepted:
            self.mark("goal_result", accepted=False, status=None, success=False)
            return False
        result = handle.get_result_async()
        start, give_up = self.clock, time.time() + 4 * GOAL_TIMEOUT_S
        pending = ["t_pose", "home"]
        arm = None   # [name, the send's future, then the result's; wall at the send; sent or running]
        outcome = None
        # 't_pose' only while the goal runs; 'home' after a successful 't_pose', whenever
        while outcome is None or arm is not None or pending[:1] == ["home"]:
            if outcome is None and result.done():
                status = result.result().status
                outcome = status == GoalStatus.STATUS_SUCCEEDED
                self.mark("goal_result", status=status, success=outcome, sim_seconds=self.clock - start,
                          map_base=nav_test._tf_parts(self.lookup("map", "base_footprint")))
            elif outcome is None and (self.clock - start > GOAL_TIMEOUT_S or time.time() > give_up):
                self.mark("goal_timeout", sim_seconds=self.clock - start)
                self.wait_future(handle.cancel_goal_async(), 10.0)
                self.wait_future(result, 10.0)
                outcome = False
                self.mark("goal_result", status=result.result().status if result.done() else None, success=False,
                          sim_seconds=self.clock - start)
            if arm is None and pending and self.walking is not None and (outcome is None or pending[0] == "home"):
                name = pending.pop(0)
                arm = [name, *self._send_named(name), "sent"]
            elif arm is not None and arm[1].done():
                response = arm[1].result()
                if arm[3] == "sent":   # the goal's handle
                    if response is None or not response.accepted:
                        self.mark(f"{arm[0]}_result", accepted=False, success=False)
                        arm, pending = None, []
                    else:
                        arm[1], arm[3] = response.get_result_async(), "running"
                else:
                    success = response is not None and bool(response.result.success)
                    self.mark(f"{arm[0]}_result", status=None if response is None else response.status,
                              success=success, joint_error=self.arm_error)
                    arm = None
                    if not success:
                        pending = []
            elif arm is not None and time.time() - arm[2] > ARM_TIMEOUT_S:
                self.mark(f"{arm[0]}_result", success=False, timeout=True)
                arm, pending = None, []
            self.spin()
        if self.walking is not None:
            self.info["walking"] = {"wall": self.walking[0], "sim": self.walking[1], "cmd": self.walking[2:]}
        return outcome

    def wait_map_after(self, sim_t):
        """Spin until a /map arrives published after sim t, or MAP_WAIT_S of sim time"""
        give_up = time.time() + 120.0
        while not any(m[1] > sim_t for m in self.maps):
            if self.clock is not None and self.clock > sim_t + MAP_WAIT_S or time.time() > give_up:
                self.mark("map_timeout")
                return False
            self.spin()
        self.mark("map_after_walk", known=self.maps[-1][3])
        return True

    def finish(self):
        """What publishes /cmd_vel and /odom, and the recordings"""
        self.info["cmd_vel_publishers"] = sorted(
            i.node_name for i in self.node.get_publishers_info_by_topic("/cmd_vel"))
        self.info["odom_publishers"] = sorted(
            i.node_name for i in self.node.get_publishers_info_by_topic("/odom"))
        self.info["lowcmd_writers"] = self.lowcmd_writers
        np.savez(os.path.join(self.out, "drive.npz"), cmds=np.array(self.cmds, float).reshape(-1, 5),
                 maps=np.array(self.maps, float).reshape(-1, 6),
                 odometry=np.array(self.odometry, float).reshape(-1, 3),
                 scans=np.array(self.scans, float).reshape(-1, 3),
                 base_tf=np.array(self.base_tf, float).reshape(-1, 3))
        self.mark("end")

    def close(self):
        self.tf_node.destroy_node()
        self.node.destroy_node()
        self.rclpy.shutdown()


def run_drive(out):
    with open(os.path.join(out, "goal.json")) as f:
        goal = json.load(f)
    d = Driver(out)
    d.info["goal"] = goal
    ok = False
    try:
        d.mark("drive_start")
        if "error" in goal:
            d.mark("no_goal", error=goal["error"])
            return 1
        if d.wait_band_released() and d.wait_nav():
            d.spin_for(STAND_S)
            x, y, yaw = d.goal_in_map(goal)
            ok = d.navigate_with_arms(x, y, yaw)
            d.spin_for(SETTLE_S)
            d.wait_map_after(next(e["sim"] for e in d.events if e["name"] == "goal_result"))
        return 0 if ok else 1
    finally:
        try:
            d.finish()   # what was recorded, whatever happened
        finally:
            d.close()


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self, out):
        self.out = out
        sim = np.load(os.path.join(out, "sim.npz"))
        self.s = {k: sim[k] for k in sim.files if k != "hz"}
        with open(os.path.join(out, "events.json")) as f:
            ev = json.load(f)
        self.info = ev["info"]
        self.events = {e["name"]: e for e in ev["events"]}
        with open(os.path.join(out, "goal.json")) as f:
            self.goal = json.load(f)
        self.robot = self.goal["robot"]
        drive = os.path.join(out, "drive.npz")
        self.drive = dict(np.load(drive)) if os.path.exists(drive) else None
        self.checks, self.numbers = [], {"robot": self.robot}
        self.cut_at_sim_reset()

    # the G1 tests' report helpers, on this recording
    check = bringup_test.Report.check
    at = bringup_test.Report.at
    end_wall = bringup_test.Report.end_wall
    release = bringup_test.Report.release
    no_estop = bringup_test.Report.no_estop
    cut_at_sim_reset = bringup_test.Report.cut_at_sim_reset

    def pelvis_at(self, t):
        return nav_test._interp_pose(self.s, np.asarray(t, float))

    def window(self):
        """Sim t of Nav2's start and of the drive's end, the window SLAM is checked over"""
        end = self.events.get("end") or max(self.events.values(), key=lambda e: e["wall"])
        return self.events["nav_active"]["sim"], end["sim"]

    def stack_up(self):
        release = self.release()
        missing = [n for n in ("band_released", "nav_active", "goal_sent") if n not in self.events]
        if release is None or missing:
            self.check("stack up", False, ("the sim's elastic band was never released; " if release is None else "")
                       + (f"no {missing} mark: see drive.log and launch.log" if missing else ""))
            return None
        first = self.events["drive_start"]["wall"]
        self.numbers["band_release_t"] = release[1]
        self.check("stack up", True,
                   f"band released at sim t {release[1]:.2f} s; Nav2 active at sim t "
                   f"{self.events['nav_active']['sim']:.1f} s ({self.events['nav_active']['wall'] - first:.0f} s wall "
                   f"after the drive started, beside the launch)")
        return release

    def standing(self, release):
        """From the band's release to the end of the recording: the band stays off, the
        pelvis above the robot's floor and upright"""
        s = self.s
        r = release[0]
        z, tilt, t = s["pelvis"][r:, 2], bringup_test._tilt_deg(s["upz"][r:]), s["t"][r:]
        states = s["band_changes"][:, 2].tolist()
        floor = ROBOTS[self.robot]["min_pelvis_z"]
        under = bringup_test._tilt_deg(s["upz"][:max(1, r)])
        self.numbers["standing"] = {"sim_t": [float(t[0]), float(t[-1])], "z_min": float(z.min()),
                                    "z_max": float(z.max()), "tilt_max_deg": float(tilt.max()),
                                    "band_states": states, "under_band_tilt_max_deg": float(under.max()),
                                    "under_band_z": [float(s["pelvis"][:max(1, r), 2].min()),
                                                     float(s["pelvis"][:max(1, r), 2].max())]}
        self.check("standing unsupported", states == [1.0, 0.0] and z.min() >= floor and tilt.max() <= MAX_TILT_DEG,
                   f"from the band's release to the end (sim t {t[0]:.1f}-{t[-1]:.1f} s): pelvis z "
                   f"{z.min():.3f}-{z.max():.3f} m, tilt max {tilt.max():.1f} deg (limits z >= {floor}, tilt <= "
                   f"{MAX_TILT_DEG}); the band " + ("on, then off once" if states == [1.0, 0.0] else
                                                    f"states {states} (1 on, 0 off)")
                   + f"; under the band tilt max {under.max():.1f} deg")

    def contacts(self, release):
        """No robot geom penetrates anything but the floor from the band's release on (the
        robot unsupported); contacts under the band are given for the record"""
        path = os.path.join(self.out, "contacts.json")
        if not os.path.exists(path):
            self.check("no contact", False, "no contacts.json from the sim")
            return
        with open(path) as f:
            c = json.load(f)
        t_free = release[1]
        after = [span for span in c["spans"] if span[1] >= t_free]
        before = [span for span in c["spans"] if span[1] < t_free]
        self.numbers["contacts"] = {"steps": c["steps"], "contact_steps": c["contact_steps"], "deepest": c["deepest"],
                                    "spans_after_release": after[:10], "spans_under_band": before[:10],
                                    "first": c["first"][:5]}
        text = (f"from the band's release (sim t {t_free:.2f} s): "
                + (f"{sum(s[2] for s in after)} physics steps with contact, in {len(after)} spans from sim t "
                   f"{after[0][0]:.2f} s" if after else "no robot geom penetrated anything but the floor"))
        if before:
            text += (f"; under the band {sum(s[2] for s in before)} steps in {len(before)} spans from sim t "
                     f"{before[0][0]:.2f} s, e.g. {c['first'][0][1]} against {c['first'][0][2]}")
        self.check("no contact", not after, text)

    def fall_log(self):
        nav_test.Report.fall_log(self)

    def goal_reached(self):
        g, ev = self.goal, self.events
        gx, gy, gyaw = g["goal"]
        result = ev.get("goal_result", {})
        took = result.get("sim_seconds", math.nan)
        self.numbers["goal"] = {"world": g["goal"], "clearance": g["goal_clearance"], "status": result.get("status"),
                                "sim_s": took}
        if "goal_result" not in ev:
            self.check("goal reached", False, "no goal result: see drive.log")
            return
        i_end = self.at("goal_result")
        last = self.at("end") if "end" in ev else len(self.s["t"]) - 1
        end_xyz, end_yaw, _ = self.pelvis_at(self.s["t"][i_end])
        last_xyz, _, _ = self.pelvis_at(self.s["t"][last])
        start_xyz, _, _ = self.pelvis_at(ev["goal_sent"]["sim"])
        err = float(math.hypot(end_xyz[0] - gx, end_xyz[1] - gy))
        settled = float(math.hypot(last_xyz[0] - gx, last_xyz[1] - gy))
        self.numbers["goal"].update(true_xy_err=err, settled_xy_err=settled,
                                    yaw_err_deg=float(math.degrees(nav_test._wrap(end_yaw - gyaw))),
                                    travelled_straight=float(math.hypot(*(end_xyz[:2] - start_xyz[:2]))))
        ok = bool(result.get("success")) and took <= GOAL_TIMEOUT_S and max(err, settled) <= FINAL_TOL
        self.check("goal reached", ok,
                   f"({gx:.3f}, {gy:.3f}), {GOAL_AHEAD} m ahead of the spawn with {g['goal_clearance']:.2f} m of room: "
                   f"NavigateToPose status {result.get('status')} (4 = succeeded) after {took:.1f} s of sim time "
                   f"(limit {GOAL_TIMEOUT_S:.0f} s); the true pelvis at the result {err:.3f} m from it, "
                   f"{self.s['t'][last] - self.s['t'][i_end]:.1f} s later {settled:.3f} m (limit {FINAL_TOL} m); "
                   f"it walked {self.numbers['goal']['travelled_straight']:.2f} m from where the goal was sent")

    def arms_while_walking(self):
        """'t_pose' sent once the robot walks and before the goal's result, 'home' once
        't_pose' has succeeded, and both succeed. Each move takes seconds and the walk to the
        goal a few, so 'home' may run after the goal's result"""
        ev, walking = self.events, self.info.get("walking")
        names = ("t_pose_sent", "t_pose_result", "home_sent", "home_result")
        got = {n: ev.get(n) for n in names}
        self.numbers["arms"] = {n: (None if e is None else {k: e.get(k) for k in ("sim", "wall", "success",
                                                                                  "joint_error")})
                                for n, e in got.items()}
        goal_end = ev.get("goal_result", {}).get("wall", math.inf)
        if walking is None or any(e is None for e in got.values()):
            self.check("arms while walking", False,
                       ("the robot never walked on /cmd_vel after the goal; " if walking is None else "")
                       + f"marks {[n for n, e in got.items() if e is None]} missing"
                       + "".join(f"; {n[:-7]} failed, joint error {got[n].get('joint_error')} rad"
                                 for n in ("t_pose_result", "home_result") if got[n] and not got[n]["success"])
                       + ": see drive.log and launch.log")
            return
        in_walk = walking["wall"] <= got["t_pose_sent"]["wall"] <= goal_end
        ok = got["t_pose_result"]["success"] and got["home_result"]["success"] and in_walk
        sim0 = walking["sim"]

        def outcome(name):
            e = got[f"{name}_result"]
            return (f"{'succeeded' if e['success'] else 'FAILED'} +{e['sim'] - sim0:.1f} s (joint error "
                    f"{e.get('joint_error')} rad)")
        self.check("arms while walking", ok,
                   f"the robot walked from sim t {sim0:.1f} s; 't_pose' sent +{got['t_pose_sent']['sim'] - sim0:.1f} "
                   f"s, {outcome('t_pose')}; 'home' sent +{got['home_sent']['sim'] - sim0:.1f} s, {outcome('home')}; "
                   f"the goal's result +{ev['goal_result']['sim'] - sim0:.1f} s"
                   + ("" if in_walk else "; 't_pose' was not sent while walking"))

    def slam(self):
        """FAST_LIO's /Odometry, level_odom's odom -> base_footprint, slam_toolbox's /map and
        the scan, from Nav2's start to the drive's end"""
        if self.drive is None:
            self.check("SLAM healthy", False, "no drive.npz")
            return
        t0, t1 = self.window()
        d = self.drive
        # /Odometry in every whole sim second of the window
        stamps = d["odometry"][:, 2]
        edges = np.arange(t0, t1 - 1.0 + 1e-9, 1.0)
        per_s = np.array([np.count_nonzero((stamps >= a) & (stamps < a + 1.0)) for a in edges])
        worst = int(per_s.min()) if len(per_s) else 0
        self.numbers["odometry"] = {"count": int(len(stamps)), "per_s_min": worst,
                                    "per_s_median": float(np.median(per_s)) if len(per_s) else None}
        self.check("odometry rate", len(per_s) > 0 and worst >= MIN_ODOM_HZ,
                   f"/Odometry over sim t {t0:.1f}-{t1:.1f} s: at least {worst} messages in every sim second "
                   f"(median {self.numbers['odometry']['per_s_median']}; limit {MIN_ODOM_HZ:g})")
        # odom -> base_footprint at every sample
        rows = d["base_tf"]
        rows = rows[(rows[:, 1] >= t0) & (rows[:, 1] <= t1)]
        missing = int(np.count_nonzero(~np.isfinite(rows[:, 2])))
        age = rows[:, 1] - rows[:, 2]
        age_max = float(np.nanmax(age)) if len(rows) and missing < len(rows) else math.inf
        self.numbers["base_tf"] = {"samples": int(len(rows)), "missing": missing, "age_max_s": age_max}
        self.check("odom -> base_footprint", len(rows) > 0 and missing == 0 and age_max <= MAX_TF_AGE_S,
                   f"{len(rows)} TF samples over the window, {missing} without odom -> base_footprint; its stamp at "
                   f"most {age_max:.2f} s behind /clock (limit {MAX_TF_AGE_S} s)")
        # /map grows
        maps = d["maps"]
        before = maps[maps[:, 1] <= t0]
        after = maps[(maps[:, 1] > t0) & (maps[:, 1] <= t1)]
        first = int(before[-1, 3]) if len(before) else None
        last = int(after[-1, 3]) if len(after) else None
        self.numbers["map"] = {"messages_in_window": int(len(after)), "known_at_start": first, "known_at_end": last}
        self.check("map grows", first is not None and last is not None and last > first,
                   f"/map's known cells {first} at Nav2's start, {last} at the drive's end ({len(after)} maps between)")
        # /converted_scan gaps
        scans = d["scans"][:, 2]
        inside = np.sort(scans[(scans >= t0) & (scans <= t1)])
        gaps = np.diff(np.concatenate([[t0], inside, [t1]]))
        gap = float(gaps.max())
        self.numbers["scan"] = {"in_window": int(len(inside)), "longest_gap_s": gap}
        self.check("scan gaps", gap <= MAX_SCAN_GAP_S,
                   f"{len(inside)} /converted_scan over the window, longest gap {gap:.2f} s (limit {MAX_SCAN_GAP_S} s)")

    def writers(self):
        """safety_node alone publishes /lowcmd: no other publisher at any sample, and none
        missing once it is up; nothing publishes /odom"""
        samples = self.info.get("lowcmd_writers") or []
        up = next((i for i, s in enumerate(samples) if LOWCMD_WRITER in s[2]), len(samples))
        others = [s for i, s in enumerate(samples) if s[2] != [LOWCMD_WRITER] and (s[2] or i > up)]
        self.numbers["lowcmd_writers"] = {"samples": len(samples), "before_safety_node": up, "others": others[:5]}
        self.check("one /lowcmd writer", up < len(samples) and not others,
                   f"{len(samples) - up} samples of /lowcmd's publishers from safety_node's start to the drive's end "
                   f"({up} before, with none)"
                   + (f", {len(others)} not only {LOWCMD_WRITER}: e.g. {others[0][2]} at sim t {others[0][1]:.1f} s"
                      if others else f": each only {LOWCMD_WRITER}"))
        odom = self.info.get("odom_publishers")
        self.check("no sim /odom", odom == [], f"/odom publishers {odom} (expected none: FAST_LIO is the odometry)")

    def run(self, out):
        t, wall = self.s["t"], self.s["wall"]
        self.numbers["real_time_factor"] = float((t[-1] - t[0]) / (wall[-1] - wall[0]))
        print(f"     {self.robot}: {len(t)} samples, sim t {t[0]:.1f}-{t[-1]:.1f} s, real-time factor "
              f"{self.numbers['real_time_factor']:.2f}")
        g = self.goal
        release = None
        if "error" in g:
            self.check("goal in open floor", False, g["error"])
        else:
            self.check("goal in open floor", True,
                       f"({g['goal'][0]:.3f}, {g['goal'][1]:.3f}), {GOAL_AHEAD} m ahead of the spawn "
                       f"({g['spawn'][0]:.3f}, {g['spawn'][1]:.3f}, yaw {math.degrees(g['spawn_yaw']):.1f} deg), "
                       f"{g['goal_clearance']:.2f} m of room (limit {GOAL_CLEARANCE} m)")
            release = self.stack_up()
            if release is not None:
                self.standing(release)
                self.goal_reached()
                self.arms_while_walking()
                self.slam()
        if release is not None:
            self.contacts(release)
        self.fall_log()
        self.writers()
        self.no_estop()
        ok = all(c["pass"] for c in self.checks)
        with open(os.path.join(out, "report.json"), "w") as f:
            json.dump({"pass": ok, "checks": self.checks, "numbers": self.numbers}, f, indent=1)
        print(f"{'PASSED' if ok else 'FAILED'}: {sum(c['pass'] for c in self.checks)}/{len(self.checks)} checks")
        return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=("sim", "drive", "report"))
    parser.add_argument("--out", required=True, help="the shared output directory")
    parser.epilog = "sim: h12_mujoco.py's arguments follow a --; the robot is GOLEM_ROBOT's"
    argv = sys.argv[1:]
    sim_args = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:len(argv) - len(sim_args) - ("--" in argv)])
    if args.role == "sim":
        run_sim(args.out, sim_args)
        return 0
    if args.role == "drive":
        return run_drive(args.out)
    return Report(args.out).run(args.out)


if __name__ == "__main__":
    sys.exit(main())
