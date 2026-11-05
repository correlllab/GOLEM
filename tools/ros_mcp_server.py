#!/usr/bin/env python3
"""Serve ROS debugging tools over MCP (HTTP) for the H1-2 Mac sim.

It runs inside the golem_ros container, where the FastDDS graph is reachable;
docker/mac/scripts/launch_ros_mac.sh starts it with GOLEM_ROS_MCP=1. Colima
doesn't forward container ports to the Mac, so it binds localhost inside the VM
and is reached over the same SSH tunnel as the noVNC viewers
(docker/mac/scripts/mac_vnc_tunnel.sh). Register it on the Mac with:

    claude mcp add --transport http ros_debug http://localhost:6082/mcp

One rclpy node subscribes once to the frequent topics (costmap, plan, scan,
clock), listens to TF and caches the latest message, so the status tools answer
without a node start-up and DDS discovery per call. It commands only /cmd_vel.
The generic tools (echo, hz, service call for any topic) shell out to the ros2
CLI. rclpy spins in a background thread; the tool handlers read the locked cache.

The robot pose (robot_status, wait_for) comes from TF: base_footprint in map, or in
odom while there is no map yet (golem_slam's level_odom publishes odom -> base_footprint
and slam_toolbox map -> odom), and the pelvis in the same frame for posture. The
sim's ground-truth /odom is off by default, so nothing here reads it.

Environment: GOLEM_ROS_MCP_PORT (default 6082), ROS_DOMAIN_ID, RMW_IMPLEMENTATION.
The MCP tool functions' docstrings are the tool descriptions clients see.
"""
import asyncio
import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from mcp.server.fastmcp import FastMCP

PORT = int(os.environ.get("GOLEM_ROS_MCP_PORT", "6082"))
_LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=1)
# A best-effort reader matches both best-effort and reliable writers, so it gets
# the sensor topics whatever QoS the publisher chose; a reliable reader gets
# nothing from a best-effort writer such as /converted_scan's.
_SENSOR = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=10)
# The frames the robot pose is read in, in order: the SLAM map, then odom until there is one.
POSE_FRAMES = ("map", "odom")
BASE_FRAME = "base_footprint"
PELVIS_FRAME = "pelvis"


class RosDebugNode(Node):
    """Cache the latest costmap, plan, scan and clock under a lock, and publish /cmd_vel."""

    def __init__(self):
        """Create the subscriptions, the /cmd_vel publisher and the TF listener."""
        super().__init__("ros_debug_mcp")
        self._lock = threading.Lock()
        self._costmap = None
        self._plan = None
        self._scan = None
        self._clock = None
        self.create_subscription(LaserScan, "/converted_scan", self._cb("_scan"), _SENSOR)
        self.create_subscription(Clock, "/clock", self._cb("_clock"), _SENSOR)
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap",
                                 self._cb("_costmap"), _LATCHED)
        self.create_subscription(Path, "/plan", self._cb("_plan"), _SENSOR)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def _cb(self, attr):
        """Return a callback that stores its message in attribute attr."""
        def f(msg):
            with self._lock:
                setattr(self, attr, msg)
        return f

    def get(self, attr):
        """Return the cached message in attribute attr, or None."""
        with self._lock:
            return getattr(self, attr)

    def sim_time(self):
        """Return the latest /clock in s, or None before the first."""
        c = self.get("_clock")
        return None if c is None else c.clock.sec + c.clock.nanosec * 1e-9

    def _lookup(self, frame, child):
        """Return the latest transform of child in frame (a geometry_msgs Transform), or None."""
        try:
            return self._tf_buffer.lookup_transform(frame, child, Time()).transform
        except TransformException:
            return None

    def robot_state(self):
        """Return the base position (base_footprint in map, else odom) and posture; None without either.

        Posture comes from the pelvis in the same frame: its height above the footprint
        and its tilt. It is "unknown" when the pelvis has no transform.
        """
        for frame in POSE_FRAMES:
            base = self._lookup(frame, BASE_FRAME)
            if base is not None:
                break
        else:
            return None
        p = base.translation
        state = {"frame": f"{frame}->{BASE_FRAME}",
                 "x": round(p.x, 3), "y": round(p.y, 3), "z": round(p.z, 3)}
        pelvis = self._lookup(frame, PELVIS_FRAME)
        if pelvis is None:
            return {**state, "uprightness": None, "posture": "unknown"}
        q = pelvis.rotation
        upright = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)  # body-up . world-up
        height = pelvis.translation.z - p.z
        return {**state, "uprightness": round(upright, 3),
                "posture": ("standing" if (upright > 0.85 and height > 0.7)
                            else "fallen" if (upright < 0.5 or height < 0.4)
                            else "leaning/other")}

    def costmap_summary(self):
        """Return the global costmap's size and cell counts by class, or None before the first."""
        m = self.get("_costmap")
        if m is None:
            return None
        a = np.asarray(m.data, dtype=np.int16)
        return {"width": m.info.width, "height": m.info.height,
                "resolution": round(m.info.resolution, 3),
                "lethal": int((a >= 99).sum()),
                "inflated": int(((a >= 50) & (a < 99)).sum()),
                "free": int((a == 0).sum()),
                "unknown": int((a == -1).sum())}

    def plan_summary(self):
        """Return whether a global plan exists, its pose count and its length (m)."""
        pl = self.get("_plan")
        if pl is None or not pl.poses:
            return {"has_plan": False, "poses": 0, "length_m": 0.0}
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in pl.poses]
        length = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                     for i in range(1, len(pts)))
        return {"has_plan": True, "poses": len(pts), "length_m": round(length, 2)}

    def publish_cmd(self, vx, vy, wz):
        """Publish one /cmd_vel (vx, vy m/s, wz rad/s, body frame)."""
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
        self._cmd_pub.publish(t)


NODE: RosDebugNode = None  # set in main()


def _sh(*args, timeout=10):
    """Run a ros2 CLI command (the launcher sourced the environment) and return its output."""
    import subprocess
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"


mcp = FastMCP("ros_debug", host="127.0.0.1", port=PORT)


@mcp.tool()
def robot_status() -> dict:
    """Robot base state at a glance: base_footprint position (x,y,z) in the map frame, or
    in odom while there is no map ("frame" says which), pelvis uprightness (1.0 =
    vertical, <0.5 = fallen), posture, and current sim time. Instant."""
    st = NODE.robot_state()
    return {"error": f"no map->{BASE_FRAME} or odom->{BASE_FRAME} TF yet (is the "
                     "bringup with navigation running?)"} if st is None else {
        **st, "sim_time": NODE.sim_time()}


@mcp.tool()
def costmap_summary() -> dict:
    """nav2 GLOBAL costmap cell counts: lethal / inflated / free / unknown, plus
    size and resolution. lethal>0 means the planner is seeing obstacles."""
    s = NODE.costmap_summary()
    return {"error": "no /global_costmap/costmap (nav2 running?)"} if s is None else s


@mcp.tool()
def nav_status() -> dict:
    """Current nav2 global plan: whether one exists, pose count, and path length
    in metres (from /plan)."""
    return NODE.plan_summary()


@mcp.tool()
def scan_status() -> dict:
    """Is /converted_scan (the SLAM/costmap laser input) live? Reports the latest
    scan's stamp and range count, or that it's absent."""
    s = NODE.get("_scan")
    if s is None:
        return {"live": False, "note": "no /converted_scan (pointcloud_to_laserscan / SLAM running?)"}
    return {"live": True, "stamp_sec": s.header.stamp.sec,
            "ranges": len(s.ranges), "range_min": round(s.range_min, 2),
            "range_max": round(s.range_max, 2)}


@mcp.tool()
async def drive(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0,
                duration: float = 4.0) -> dict:
    """Publish /cmd_vel (body frame, m/s and rad/s) at 20 Hz for `duration`
    seconds, then stop. vx>0 forward, wz>0 turn left. Needs the sim launched
    with GOLEM_LOWERBODY=almi. Reports start/end pose so you can see if it moved."""
    duration = max(0.0, min(float(duration), 30.0))
    p0 = NODE.robot_state()
    hz, end = 0.05, time.time() + duration
    while time.time() < end:
        NODE.publish_cmd(vx, vy, wz)
        await asyncio.sleep(hz)
    NODE.publish_cmd(0.0, 0.0, 0.0)
    p1 = NODE.robot_state()
    moved = (round(math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"]), 3)
             if p0 and p1 else None)
    return {"ok": True, "commanded": {"vx": vx, "vy": vy, "wz": wz, "duration": duration},
            "start": p0, "end": p1, "moved_m": moved}


@mcp.tool()
async def wait_for(condition: str, timeout: float = 60.0) -> dict:
    """Block until a condition holds, then return (polls the warm cache ~5 Hz).
    Replaces sleep-poll loops. condition is one of:
      standing | fallen | stopped | moving | scan_live | nav_has_plan |
      sim_time>=<N>   (e.g. 'sim_time>=20')
    Returns {met: bool, waited_s, ...state}."""
    timeout = max(1.0, min(float(timeout), 600.0))
    t0 = time.time()
    last = None

    def check():
        nonlocal last
        st = NODE.robot_state()
        last = st
        if condition == "standing":
            return bool(st and st["posture"] == "standing")
        if condition == "fallen":
            return bool(st and st["posture"] == "fallen")
        if condition == "scan_live":
            return NODE.get("_scan") is not None
        if condition == "nav_has_plan":
            return NODE.plan_summary()["has_plan"]
        if condition.startswith("sim_time>="):
            try:
                thr = float(condition.split(">=", 1)[1])
            except ValueError:
                return False
            t = NODE.sim_time()
            return t is not None and t >= thr
        if condition in ("stopped", "moving"):
            a = NODE.robot_state()
            time.sleep(0.4)
            b = NODE.robot_state()
            if not a or not b:
                return False
            d = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            return (d < 0.01) if condition == "stopped" else (d >= 0.02)
        return False

    while time.time() - t0 < timeout:
        if check():
            return {"met": True, "waited_s": round(time.time() - t0, 1),
                    "state": last, "sim_time": NODE.sim_time()}
        await asyncio.sleep(0.2)
    return {"met": False, "waited_s": round(time.time() - t0, 1),
            "state": last, "sim_time": NODE.sim_time(),
            "note": f"condition '{condition}' not met within {timeout}s"}


@mcp.tool()
def list_topics(filter: str = "") -> list:
    """List active topics (optionally substring-filtered), as 'name  type'."""
    out = []
    for name, types in sorted(NODE.get_topic_names_and_types()):
        if not filter or filter in name:
            out.append(f"{name}  {','.join(types)}")
    return out


@mcp.tool()
def echo_topic(topic: str, timeout: float = 6.0) -> str:
    """One message from any topic (generic, via `ros2 topic echo --once`). Use the
    hot tools (robot_status/costmap_summary/scan_status) for the common ones."""
    return _sh("ros2", "topic", "echo", "--once", topic, timeout=int(timeout) + 2)


@mcp.tool()
def topic_hz(topic: str, window: float = 5.0) -> str:
    """Measured publish rate of a topic over `window` seconds."""
    import subprocess
    try:
        p = subprocess.run(["ros2", "topic", "hz", topic], capture_output=True,
                           text=True, timeout=window + 3)
        for line in (p.stdout or "").splitlines():
            if "average rate" in line:
                return line.strip()
        return "(no messages in window)"
    except subprocess.TimeoutExpired as e:
        for line in ((e.stdout or b"").decode(errors="ignore")).splitlines():
            if "average rate" in line:
                return line.strip()
        return "(no messages in window)"


@mcp.tool()
def call_service(name: str, type: str, args: str = "{}") -> str:
    """Call any service (generic, via `ros2 service call`). type is the full type
    e.g. 'std_srvs/srv/Trigger'; args is a YAML/JSON dict string."""
    return _sh("ros2", "service", "call", name, type, args, timeout=15)


@mcp.tool()
def node_list() -> list:
    """List active ROS nodes."""
    return sorted(f"{ns}/{n}".replace("//", "/") for n, ns in NODE.get_node_names_and_namespaces())


def _spin(executor):
    """Spin the executor until shutdown."""
    executor.spin()


def main():
    """Start the node's executor in a background thread and serve MCP over HTTP."""
    rclpy.init()
    global NODE
    NODE = RosDebugNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(NODE)
    threading.Thread(target=_spin, args=(ex,), daemon=True).start()
    NODE.get_logger().info(f"ros_debug MCP server on http://127.0.0.1:{PORT}/mcp")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
