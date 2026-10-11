"""Game commander: drives marl_deploy_node with the command schedule the policy was trained on.

Training (Locomanipulation_game, ArmTargetsCommand) alternates two kinds of event, each drawn on its own:
  walking   a velocity command (vx, vy, wz) uniform in the training ranges (5% standing still), held for
            nav_time_range seconds;
  arm goal  one wrist pose per arm from the run's target tables, held arm_goal_s seconds; half are standing
            goals, half are built in a squat (arm_goals_<spread>_<drop>.npz, exported by Locomanipulation_game
            scripts/skrl/export_arm_goals.py);
with arm_goal_prob the chance of an arm goal. marl_deploy_node settles between the two (0.75 s before a reach,
1.5 s before walking) as training does; an arm goal that follows an arm goal replaces it without a settle.

The training world was an open floor; a kitchen isn't. Two guards keep the schedule from walking the robot into
the furniture, and both only shorten or redirect walking:
  obstacles  /livox/pointcloud in the pelvis frame: a walking segment ends early when a point between knee and
             head height lies within obstacle_distance of the pelvis in the direction of travel, and a new
             segment is drawn from the directions that are clear; an arm goal waits for clear_reach in front,
             turning in place until it is.
  geofence   with SLAM running, /Odometry (FAST-LIO, camera_init -> body): outside geofence_radius of the start,
             walking segments head back toward it.

mode "sequence" runs one scripted pass instead: back off 3 s, turn 4 s, walk 4 s, a standing reach, a crouch reach,
walk 4 s (the guards still apply).

Every event and a status line at 10 Hz (state, commands, wrist errors, safety heartbeat, SLAM pose) go to
log_file as JSON lines, for evaluation.

Interfaces
----------
sub  /marl_deploy_node/state         (std_msgs/String)
sub  /marl_deploy_node/wrist_error   (std_msgs/Float32MultiArray)  m, left and right
sub  /safety/heartbeat               (std_msgs/Bool)
sub  /livox/pointcloud               (sensor_msgs/PointCloud2)
sub  /Odometry                       (nav_msgs/Odometry)
sub  /lowstate                       (unitree_hg/LowState)  IMU tilt: a fall ends the run
pub  /cmd_vel                        (geometry_msgs/Twist)
pub  /marl_deploy_node/arm_goal      (geometry_msgs/PoseArray)
cli  /marl_deploy_node/end_arm_goal  (std_srvs/Trigger)
"""

import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32MultiArray, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from unitree_hg.msg import LowState

from lmg_deploy.rotations import projected_gravity, quat_apply

DRIVING_STATES = ("nav", "settle_to_arm", "arm_goal", "settle_to_nav")
FALLEN_TILT = 1.0    # rad: the IMU tilted this far for FALLEN_S ends the run (training's fall termination: 1 rad)
FALLEN_S = 0.5
REACH_TOL = 0.05     # m: both wrists this close for REACH_HOLD_S counts a goal reached (training: 5 cm, 1 s)
REACH_HOLD_S = 1.0


class GameCommanderNode(Node):
    def __init__(self):
        super().__init__("game_commander")
        self.declare_parameter("mode", "training")           # training | sequence
        self.declare_parameter("goals_file", "")
        self.declare_parameter("seed", 0)
        self.declare_parameter("arm_goal_prob", 0.5)
        self.declare_parameter("nav_time_range", [4.0, 8.0])
        self.declare_parameter("arm_goal_s", 4.0)
        self.declare_parameter("standing_prob", 0.05)
        self.declare_parameter("lin_vel_x", [-0.7, 0.7])
        self.declare_parameter("lin_vel_y", [-0.3, 0.3])
        self.declare_parameter("ang_vel_z", [-0.5, 0.5])
        self.declare_parameter("obstacle_distance", 0.6)     # m from the pelvis, direction of travel
        self.declare_parameter("clear_reach", 0.8)           # m in front before an arm goal starts
        self.declare_parameter("geofence_radius", 1.5)       # m from the start; 0 disables
        # the frame the cloud's points are in, when its header names a frame the TF tree lacks
        # (RoboCasa stamps "lidar_link"; the sensor sits at the URDF's livox_link)
        self.declare_parameter("cloud_frame", "")
        self.declare_parameter("duration_s", 0.0)            # stop commanding after this (sim time); 0: forever
        self.declare_parameter("log_file", "")

        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))
        self.mode = str(self.get_parameter("mode").value)
        goals_file = str(self.get_parameter("goals_file").value)
        if not goals_file or not os.path.isfile(goals_file):
            raise FileNotFoundError(f"goals_file {goals_file!r} not found")
        goals = np.load(goals_file)
        self.goal_poses, self.goal_low = goals["poses"], goals["low"].astype(bool)
        self.goal_crouch = goals["needs_crouch"].astype(bool)
        self.ranges = np.array([self.get_parameter(k).value for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")], dtype=float)
        self.log_path = str(self.get_parameter("log_file").value)
        self.log = open(self.log_path, "a", buffering=1) if self.log_path else None

        self.deploy_state = ""
        self.wrist_error = np.full(2, np.nan)
        self.heartbeat: bool | None = None
        self.points: np.ndarray | None = None    # (N, 3) pelvis frame, obstacle band only
        self.cloud_count = {"received": 0, "tf_failed": 0}
        self.odom: np.ndarray | None = None      # (x, y, yaw) in camera_init
        self.odom_start: np.ndarray | None = None
        self.cmd = np.zeros(3)
        self.event: dict | None = None
        self.event_end = 0.0
        self.start_time: float | None = None
        self.reach_since: float | None = None
        self.reached = False
        self.events_done = 0
        self.script: list[dict] = []
        self.finished = False

        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(String, "/marl_deploy_node/state", self._on_state, latched)
        self.create_subscription(Float32MultiArray, "/marl_deploy_node/wrist_error", self._on_wrist_error, 10)
        self.create_subscription(Bool, "/safety/heartbeat", self._on_heartbeat, 10)
        self.create_subscription(PointCloud2, "/livox/pointcloud", self._on_cloud, sensor)
        self.create_subscription(Odometry, "/Odometry", self._on_odom, 10)
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, sensor)
        self.tilt = 0.0
        self.tilted_since: float | None = None
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.goal_pub = self.create_publisher(PoseArray, "/marl_deploy_node/arm_goal", 10)
        self.end_cli = self.create_client(Trigger, "/marl_deploy_node/end_arm_goal")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.05, self._tick)
        self.create_timer(0.1, self._status)
        self.get_logger().info(f"game_commander: mode {self.mode}, {len(self.goal_poses)} goals from {goals_file}")

    # -- inputs -------------------------------------------------------------------------
    def _on_state(self, msg: String) -> None:
        if msg.data != self.deploy_state:
            self._write({"type": "deploy_state", "state": msg.data})
        self.deploy_state = msg.data

    def _on_wrist_error(self, msg: Float32MultiArray) -> None:
        self.wrist_error = np.asarray(msg.data, dtype=float)

    def _on_heartbeat(self, msg: Bool) -> None:
        if self.heartbeat is not False and not msg.data:
            self._write({"type": "estop", "detail": "safety heartbeat False"})
        self.heartbeat = bool(msg.data)

    def _on_cloud(self, msg: PointCloud2) -> None:
        self.cloud_count["received"] += 1
        try:
            frame = str(self.get_parameter("cloud_frame").value) or msg.header.frame_id
            tf = self.tf_buffer.lookup_transform("pelvis", frame, Time())
        except Exception as error:  # noqa: BLE001 - TF not up yet
            self.cloud_count["tf_failed"] += 1
            self.get_logger().warn(f"cloud TF: {error}", throttle_duration_sec=10.0)
            return
        cloud = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        if len(cloud) == 0:
            return
        if getattr(cloud, "dtype", None) is not None and cloud.dtype.names:
            xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=-1)
        else:
            xyz = np.asarray(list(cloud), dtype=np.float64).reshape(-1, 3)
        r = tf.transform.rotation
        t = tf.transform.translation
        quat = np.array([r.w, r.x, r.y, r.z])
        rot = _quat_matrix(quat)
        pts = xyz.astype(np.float64) @ rot.T + np.array([t.x, t.y, t.z])
        # between knee and head height in the pelvis frame (pelvis ~1 m up), beyond the robot's own body
        band = (pts[:, 2] > -0.6) & (pts[:, 2] < 0.8) & (np.hypot(pts[:, 0], pts[:, 1]) > 0.35)
        self.points = pts[band]

    def _on_lowstate(self, msg: LowState) -> None:
        gravity = projected_gravity(np.asarray(msg.imu_state.quaternion, dtype=np.float64))
        self.tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom = np.array([p.x, p.y, yaw])
        if self.odom_start is None:
            self.odom_start = self.odom.copy()

    # -- helpers -------------------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _write(self, record: dict) -> None:
        record = {"t": round(self._now(), 3), "wall": round(time.time(), 3), **record}
        if self.log:
            self.log.write(json.dumps(record) + "\n")
        if record["type"] not in ("status",):
            self.get_logger().info(json.dumps(record))

    def _clearance(self, direction: float, half_width: float = 0.35) -> float:
        """Free distance (m) from the pelvis along a pelvis-frame heading, in a corridor half_width wide."""
        if self.points is None or len(self.points) == 0:
            return math.inf
        c, s = math.cos(direction), math.sin(direction)
        along = self.points[:, 0] * c + self.points[:, 1] * s
        across = -self.points[:, 0] * s + self.points[:, 1] * c
        ahead = (along > 0.0) & (np.abs(across) < half_width)
        return float(along[ahead].min()) if ahead.any() else math.inf

    def _outside_fence(self) -> np.ndarray | None:
        """The direction home (pelvis frame, rad) when SLAM puts the robot outside the geofence, else None."""
        radius = float(self.get_parameter("geofence_radius").value)
        if radius <= 0.0 or self.odom is None or self.odom_start is None:
            return None
        d = self.odom_start[:2] - self.odom[:2]
        if np.hypot(*d) < radius:
            return None
        return math.atan2(d[1], d[0]) - self.odom[2]

    def _draw_velocity(self) -> np.ndarray:
        home = self._outside_fence()
        if home is not None:
            return np.array([0.4 * math.cos(home), 0.25 * math.sin(home), 0.0])
        if self.rng.random() < float(self.get_parameter("standing_prob").value):
            return np.zeros(3)
        for _ in range(20):
            cmd = self.rng.uniform(self.ranges[:, 0], self.ranges[:, 1])
            if np.hypot(cmd[0], cmd[1]) < 1e-3 or \
                    self._clearance(math.atan2(cmd[1], cmd[0])) > float(self.get_parameter("obstacle_distance").value) + 0.5:
                return cmd
        return np.array([0.0, 0.0, self.ranges[2, 1] * (1 if self.rng.random() < 0.5 else -1)])  # boxed in: turn

    def _draw_goal(self, kind: str | None = None) -> int:
        if kind == "standing":
            pool = np.flatnonzero(~self.goal_low & (self.goal_poses[:, :, 0].min(axis=1) > 0.25))
        elif kind == "crouch":
            pool = np.flatnonzero(self.goal_low & self.goal_crouch)
        else:
            pool = np.arange(len(self.goal_poses))
        return int(self.rng.choice(pool))

    # -- events -----------------------------------------------------------------------------
    def _start_walk(self, cmd: np.ndarray | None = None, duration: float | None = None) -> None:
        if self.event is not None and self.event["kind"] == "arm_goal":
            self._finish_goal()
            self.end_cli.call_async(Trigger.Request())
        lo, hi = self.get_parameter("nav_time_range").value
        self.cmd = self._draw_velocity() if cmd is None else np.asarray(cmd, dtype=float)
        duration = float(self.rng.uniform(lo, hi)) if duration is None else duration
        self.event = {"kind": "walk", "index": self.events_done, "cmd": self.cmd.round(3).tolist(), "duration": round(duration, 2)}
        self.event_end = self._now() + duration
        self._write({"type": "event", **self.event})

    def _start_goal(self, index: int, duration: float | None = None) -> None:
        if self.event is not None and self.event["kind"] == "arm_goal":
            self._finish_goal()
        self.cmd = np.zeros(3)
        poses = self.goal_poses[index]
        msg = PoseArray()
        msg.header.frame_id = "pelvis"
        for p in poses:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in p[:3])
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = (float(v) for v in p[3:])
            msg.poses.append(pose)
        self.goal_pub.publish(msg)
        duration = float(self.get_parameter("arm_goal_s").value) if duration is None else duration
        # the deploy node settles 0.75 s first when the robot was walking
        settle = 0.75 if self.event is None or self.event["kind"] != "arm_goal" else 0.0
        self.event = {"kind": "arm_goal", "index": self.events_done, "goal": index, "low": bool(self.goal_low[index]),
                      "needs_crouch": bool(self.goal_crouch[index]), "poses": poses.round(4).tolist(),
                      "duration": duration, "settle": settle, "best_error": math.inf}
        self.event_end = self._now() + settle + duration
        self.reach_since, self.reached = None, False
        self._write({"type": "event", **{k: v for k, v in self.event.items() if k != "best_error"}})

    def _finish_goal(self) -> None:
        ev = self.event
        self._write({"type": "goal_result", "index": ev["index"], "goal": ev["goal"], "low": ev["low"],
                     "needs_crouch": ev["needs_crouch"], "reached": self.reached,
                     "best_error": None if math.isinf(ev["best_error"]) else round(ev["best_error"], 4)})

    def _next_event(self) -> None:
        self.events_done += 1
        if self.mode == "sequence":
            if not self.script:
                self.finished = True
                self._write({"type": "done", "events": self.events_done})
                if self.event is not None and self.event["kind"] == "arm_goal":
                    self._finish_goal()
                    self.end_cli.call_async(Trigger.Request())
                self.event, self.cmd = None, np.zeros(3)
                return
            step = self.script.pop(0)
            if step["kind"] == "walk":
                self._start_walk(step["cmd"], step["duration"])
            else:
                self._start_goal(self._draw_goal(step["goal"]), step["duration"])
            return
        if self.rng.random() < float(self.get_parameter("arm_goal_prob").value):
            if self._clearance(0.0, 0.45) < float(self.get_parameter("clear_reach").value):
                self._start_walk(np.array([0.0, 0.0, self.ranges[2, 1]]), 2.0)   # face open space first
            else:
                self._start_goal(self._draw_goal())
        else:
            self._start_walk()

    # -- loop --------------------------------------------------------------------------------
    def _tick(self) -> None:
        now = self._now()
        if self.tilt > FALLEN_TILT:
            self.tilted_since = now if self.tilted_since is None else self.tilted_since
        else:
            self.tilted_since = None
        if not self.finished and self.tilted_since is not None and now - self.tilted_since > FALLEN_S:
            self.finished = True
            if self.event is not None and self.event["kind"] == "arm_goal":
                self._finish_goal()
            self.event, self.cmd = None, np.zeros(3)
            self._write({"type": "done", "events": self.events_done, "reason": f"fallen (IMU tilt {self.tilt:.2f} rad)"})
        if not self.finished and (self.deploy_state.startswith("fault") or self.heartbeat is False):
            # an e-stop or a latched deploy node ends the run: the robot is limp from here
            self.finished = True
            if self.event is not None and self.event["kind"] == "arm_goal":
                self._finish_goal()
            self.event, self.cmd = None, np.zeros(3)
            self._write({"type": "done", "events": self.events_done, "reason": self.deploy_state or "estop"})
        if self.finished or self.deploy_state not in DRIVING_STATES:
            self.cmd_pub.publish(Twist())
            return
        if self.start_time is None:
            # the obstacle guard needs the cloud in the pelvis frame (TF comes up a few seconds in)
            self.waiting_since = now if getattr(self, "waiting_since", None) is None else self.waiting_since
            if self.points is None and now - self.waiting_since < 20.0:
                self.cmd_pub.publish(Twist())
                return
            if self.points is None:
                self._write({"type": "warning", "detail": "no obstacle input after 20 s; starting without it"})
            self.start_time = now
            if self.mode == "sequence":
                # the robot starts facing a counter: back off and turn before walking forward
                self.script = [{"kind": "walk", "cmd": [-0.3, 0.0, 0.0], "duration": 3.0},
                               {"kind": "walk", "cmd": [0.0, 0.0, 0.5], "duration": 4.0},
                               {"kind": "walk", "cmd": [0.4, 0.0, 0.0], "duration": 4.0},
                               {"kind": "goal", "goal": "standing", "duration": 6.0},
                               {"kind": "goal", "goal": "crouch", "duration": 6.0},
                               {"kind": "walk", "cmd": [0.4, 0.0, 0.0], "duration": 4.0}]
            self._write({"type": "start", "mode": self.mode})
            self.events_done = -1
            self._next_event()
        duration = float(self.get_parameter("duration_s").value)
        if not self.finished and duration > 0.0 and now - self.start_time > duration:
            self.finished = True
            if self.event is not None and self.event["kind"] == "arm_goal":
                self._finish_goal()
                self.end_cli.call_async(Trigger.Request())
            self.event, self.cmd = None, np.zeros(3)
            self._write({"type": "done", "events": self.events_done})
        if self.finished:
            self.cmd_pub.publish(Twist())
            return

        ev = self.event
        if ev["kind"] == "walk" and np.hypot(self.cmd[0], self.cmd[1]) > 0.05:
            heading = math.atan2(self.cmd[1], self.cmd[0])
            if self._clearance(heading) < float(self.get_parameter("obstacle_distance").value):
                self._write({"type": "obstacle_stop", "heading": round(heading, 2)})
                self._next_event()
                ev = self.event
        if ev is not None and ev["kind"] == "arm_goal" and self.deploy_state == "arm_goal" and np.isfinite(self.wrist_error).all():
            err = float(self.wrist_error.max())
            ev["best_error"] = min(ev["best_error"], err)
            if err < REACH_TOL:
                self.reach_since = now if self.reach_since is None else self.reach_since
                self.reached |= now - self.reach_since >= REACH_HOLD_S
            else:
                self.reach_since = None
        if ev is not None and now >= self.event_end:
            self._next_event()
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.angular.z = (float(v) for v in self.cmd)
        self.cmd_pub.publish(twist)

    def _status(self) -> None:
        self._write({"type": "status", "state": self.deploy_state, "cmd": self.cmd.round(3).tolist(),
                     "event": None if self.event is None else self.event["kind"],
                     "wrist_error": [None if not np.isfinite(e) else round(float(e), 4) for e in self.wrist_error],
                     "heartbeat": self.heartbeat,
                     "odom": None if self.odom is None else self.odom.round(3).tolist(),
                     "front_clearance": round(min(self._clearance(0.0), 9.9), 2),
                     "cloud": {**self.cloud_count, "band_points": 0 if self.points is None else len(self.points)}})


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    return np.stack([quat_apply(q, e) for e in np.eye(3)], axis=1)


def main():
    rclpy.init()
    node = GameCommanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.log:
            node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
