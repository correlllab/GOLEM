"""Scripted drive of marl_deploy_node in the Robocasa sim, measured on FAST-LIO's /Odometry (as on the robot).

Run in the ROS container with lmg_sim_bringup policy:=<name> up (any non-zero ROS_DOMAIN_ID):
    python3 sim_drive.py [--out /tmp/sim_drive.json]

Confirms the engage, then stands, walks backward and forward, turns, side-steps, reaches a world-fixed arm goal and
walks again. For each
phase it reports the commanded velocity, the head IMU's mean heading-frame velocity and yaw rate (FAST-LIO's body,
differenced over 0.5 s windows), the cosine between commanded and actual planar velocity, the largest height drop and
tilt from standing, the node's states and the wrist errors. camera_init is levelled as h12_slam level_odom does it:
up = the mean /livox/imu specific force, rotated into camera_init, over the first stand. A drop of 0.4 m or a tilt
past 0.8 rad from standing counts as a fall.
"""

import argparse
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger

# (label, seconds, (vx, vy, wz)) or (label, seconds, "arm")
# The Robocasa spawn faces a fixture about 0.5 m ahead with open floor behind (h1_2_robosuite place_robot_clear):
# back up first, then walk forward over the same ground.
PHASES = [
    ("stand", 4.0, (0.0, 0.0, 0.0)),
    ("walk backward 0.3", 6.0, (-0.3, 0.0, 0.0)),
    ("walk forward 0.3", 5.0, (0.3, 0.0, 0.0)),
    ("stop", 2.0, (0.0, 0.0, 0.0)),
    ("turn left 0.4", 6.0, (0.0, 0.0, 0.4)),
    ("turn right 0.4", 6.0, (0.0, 0.0, -0.4)),
    ("walk backward + turn", 6.0, (-0.3, 0.0, 0.3)),
    ("walk sideways left 0.2", 4.0, (0.0, 0.2, 0.0)),
    ("walk sideways right 0.2", 4.0, (0.0, -0.2, 0.0)),
    ("stop again", 3.0, (0.0, 0.0, 0.0)),
    ("arm goal (world-fixed)", 8.0, "arm"),
    ("walk backward 0.3 again", 5.0, (-0.3, 0.0, 0.0)),
    ("stop at the end", 3.0, (0.0, 0.0, 0.0)),
]
ARM_GOAL = [(0.30, 0.25, 0.05), (0.30, -0.25, 0.05)]  # left, right wrist, pelvis frame (README example)


def rot(q):
    """Rotation matrix of an (x, y, z, w) quaternion, body -> world."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Driver(Node):
    def __init__(self):
        super().__init__("sim_drive")
        self.poses = []         # (t, position (3,), rotation (3, 3)) in camera_init
        self.up_samples = []    # unit specific force rotated into camera_init, while levelling
        self.levelling = True
        self.up = None
        self.states = []        # (t, state)
        self.wrist = []         # (t, left, right)
        self.create_subscription(Odometry, "/Odometry", self._on_odom, 50)
        self.create_subscription(Imu, "/livox/imu", self._on_imu, qos_profile_sensor_data)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)   # as the node publishes ~/state
        self.create_subscription(String, "/marl_deploy_node/state", self._on_state, latched)
        self.create_subscription(Float32MultiArray, "/marl_deploy_node/wrist_error", self._on_wrist, 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_goal = self.create_publisher(PoseArray, "/marl_deploy_node/arm_goal", 10)
        self.cli_confirm = self.create_client(Trigger, "/marl_deploy_node/confirm_engage")
        self.cli_end = self.create_client(Trigger, "/marl_deploy_node/end_arm_goal")

    def _on_odom(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        self.poses.append((time.monotonic(), np.array([p.x, p.y, p.z]), rot((o.x, o.y, o.z, o.w))))

    def _on_imu(self, m):
        if not self.levelling or not self.poses:
            return
        a = np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        if np.linalg.norm(a) > 1e-6:
            self.up_samples.append(self.poses[-1][2] @ (a / np.linalg.norm(a)))

    def finish_levelling(self):
        self.levelling = False
        if self.up_samples:
            u = np.mean(self.up_samples, axis=0)
            self.up = u / np.linalg.norm(u)

    def odom_rows(self, t0, t1):
        """(t, planar x, planar y, height, heading, tilt from standing) per FAST-LIO pose in [t0, t1]."""
        if self.up is None or not self.poses:
            return np.zeros((0, 6))
        up = self.up
        r_ref = self.poses[0][2]
        e1 = r_ref[:, 0] - np.dot(r_ref[:, 0], up) * up
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        z_ref = r_ref[:, 2]
        rows = []
        for t, p, r in self.poses:
            if not t0 <= t <= t1:
                continue
            fwd = r[:, 0]
            heading = math.atan2(np.dot(fwd, e2), np.dot(fwd, e1))
            tilt = math.acos(max(-1.0, min(1.0, float(np.dot(r[:, 2], z_ref)))))
            rows.append((t, np.dot(p, e1), np.dot(p, e2), np.dot(p, up), heading, tilt))
        return np.array(rows)

    def _on_state(self, m):
        self.states.append((time.monotonic(), m.data))

    def _on_wrist(self, m):
        if len(m.data) >= 2:
            self.wrist.append((time.monotonic(), float(m.data[0]), float(m.data[1])))

    @property
    def state(self):
        return self.states[-1][1] if self.states else ""

    def spin_for(self, seconds, cmd=None):
        end = time.monotonic() + seconds
        next_pub = 0.0
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
            if cmd is not None and time.monotonic() >= next_pub:
                t = Twist()
                t.linear.x, t.linear.y, t.angular.z = cmd
                self.pub_cmd.publish(t)
                next_pub = time.monotonic() + 0.1

    def call(self, client, name):
        if not client.wait_for_service(timeout_sec=10.0):
            return f"{name}: no service"
        fut = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        r = fut.result()
        return f"{name}: {r.success} {r.message}" if r else f"{name}: no reply"


def phase_stats(d, t0, t1, cmd, z_stand):
    od = d.odom_rows(t0, t1)
    out = {}
    if len(od) < 10:
        return {"samples": int(len(od))}
    # body-frame velocity over 0.5 s windows
    vb, wz = [], []
    j = 0
    for i in range(len(od)):
        while od[i, 0] - od[j, 0] > 0.5:
            j += 1
        if od[i, 0] - od[j, 0] < 0.3:
            continue
        dt = od[i, 0] - od[j, 0]
        dx, dy = od[i, 1] - od[j, 1], od[i, 2] - od[j, 2]
        yaw = od[j, 4] + 0.5 * math.remainder(od[i, 4] - od[j, 4], 2 * math.pi)
        c, s = math.cos(yaw), math.sin(yaw)
        vb.append((c * dx + s * dy) / dt * np.array([1.0, 0.0]) + (-s * dx + c * dy) / dt * np.array([0.0, 1.0]))
        wz.append(math.remainder(od[i, 4] - od[j, 4], 2 * math.pi) / dt)
    vb, wz = np.array(vb), np.array(wz)
    late = slice(len(vb) // 3, None)  # past the transition
    v, w = vb[late].mean(0), float(wz[late].mean())
    drop = z_stand - float(od[:, 3].min())
    out.update(vx=round(float(v[0]), 3), vy=round(float(v[1]), 3), wz=round(w, 3),
               max_drop_m=round(drop, 3), max_tilt_deg=round(math.degrees(float(od[:, 5].max())), 1))
    if cmd != "arm" and np.linalg.norm(cmd[:2]) > 0.05:
        cv = np.array(cmd[:2])
        out["cos_v"] = round(float(np.dot(v, cv) / (np.linalg.norm(v) * np.linalg.norm(cv) + 1e-9)), 3)
    if cmd != "arm" and abs(cmd[2]) > 0.05:
        out["yaw_sign_ok"] = bool(w * cmd[2] > 0)
    wr = [x for x in d.wrist if t0 <= x[0] <= t1]
    if wr and cmd == "arm":
        late_w = wr[len(wr) // 2:]
        out["wrist_err_cm_end"] = round(100 * float(np.mean([0.5 * (a + b) for _, a, b in late_w])), 1)
        out["wrist_err_cm_min"] = round(100 * float(min(0.5 * (a + b) for _, a, b in wr)), 1)
    out["states"] = sorted({s for t, s in d.states if t0 <= t <= t1} | ({d.state} if d.state else set()))
    out["fell"] = bool(drop > 0.4 or od[:, 5].max() > 0.8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/sim_drive.json")
    ap.add_argument("--engage_timeout", type=float, default=300.0)
    args = ap.parse_args()
    rclpy.init()
    d = Driver()
    log = {"events": [], "phases": []}
    t_start = time.monotonic()
    # wait for the node to pre-pose and level its attitude, then confirm
    while time.monotonic() - t_start < args.engage_timeout and d.state not in ("waiting_confirm", "nav"):
        d.spin_for(0.5)
        if d.state.startswith("fault"):
            break
    log["events"].append(f"t={time.monotonic() - t_start:.1f}s state {d.state!r} before confirm")
    if d.state != "nav":
        log["events"].append(d.call(d.cli_confirm, "confirm_engage"))
    t_wait = time.monotonic()
    while time.monotonic() - t_wait < 30.0 and d.state != "nav":
        d.spin_for(0.5)
    log["events"].append(f"t={time.monotonic() - t_start:.1f}s state {d.state!r} after confirm")
    log["events"].append(f"FAST-LIO poses so far: {len(d.poses)}")
    z_stand = None
    for label, seconds, cmd in PHASES:
        t0 = time.monotonic()
        if cmd == "arm":
            pa = PoseArray()
            for x, y, z in ARM_GOAL:
                p = Pose()
                p.position.x, p.position.y, p.position.z = x, y, z
                p.orientation.w = 1.0
                pa.poses.append(p)
            d.pub_goal.publish(pa)
            d.spin_for(seconds)
            log["events"].append(d.call(d.cli_end, "end_arm_goal"))
        else:
            d.spin_for(seconds, cmd)
        if d.levelling:  # the first phase stands: level camera_init from it
            d.finish_levelling()
            rows = d.odom_rows(t0, time.monotonic())
            z_stand = float(rows[:, 3].mean()) if len(rows) else 0.0
            log["events"].append(f"levelled camera_init from {len(d.up_samples)} IMU samples: up {np.round(d.up, 4).tolist() if d.up is not None else None}")
        stats = phase_stats(d, t0, time.monotonic(), cmd, z_stand)
        stats.update(phase=label, cmd=cmd if cmd == "arm" else list(cmd))
        log["phases"].append(stats)
        print(json.dumps(stats), flush=True)
    d.spin_for(0.5, (0.0, 0.0, 0.0))
    json.dump(log, open(args.out, "w"), indent=1)
    print("[EVENTS]", json.dumps(log["events"]), flush=True)
    d.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
