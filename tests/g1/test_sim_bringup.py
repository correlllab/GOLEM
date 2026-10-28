"""Acceptance test of the G1 sim with the whole bringup: the parts that run in
the containers. test_sim_bringup.sh runs them; see it for the usage.

  sim     in the robocasa container: h12_mujoco.py --g1 as __main__ (its own
          CLI), recording the robot's state from the sim after every 10th physics
          step (50 Hz of sim time; GOLEM_TEST_RECORD_EVERY=1 for every step)
          to <out>/sim.npz, until <out>/stop appears
  drive   in the ros container, beside 'ros2 launch g1_bringup
          g1_sim_bringup.launch.py': the test's steps, with their wall times
          and results in <out>/events.json
  report  anywhere with numpy: checks the recording against the steps,
          prints the numbers and writes <out>/report.json; exit 1 on a failure

The arm test (default): the safety layer engages rt/arm_sdk (weight 0 -> 1),
frame_task_server takes the arms to 'home' at startup, a NamedConfig 't_pose'
goal succeeds and the arms reach it, /left/hand/close closes the left hand,
stopping frame_task_server hands the arms back (weight -> 0), and the stand-in
(HOMIE) keeps the robot standing throughout. The walk test (--walk, with
use_locomotion:=true): the stand-in hands over to the walking node's
rt/lowcmd, the robot keeps standing, then walks on /cmd_vel 0.3 m/s forward
for 5 s; after the sensors, frame_task_server alone is stopped while the
walking node balances the robot, and the waist roll and pitch stay at their
hold, since its shutdown command keeps them stiff on rt/lowcmd. Both tests
listen to the sensors from the ros container for 10 s
(after the controller is up, or after the walk): every sensor topic at its
rate in sim time throughout the window, with its frame_id and its message
contents, the camera's frame on /tf_static, the sim's node publishing nothing
else, and one /livox/lidar frame putting the floor where the upside-down mount
predicts.
The joint, base and sensor poses come from the sim's own state; no ground
truth is published.

The steps wait in sim time (/clock) and the checks measure motion in sim time,
so a sim slower than real time stretches the test instead of failing it; what
the bringup runs on wall time (the safety layer's weight ramps) is checked in
wall time.
"""
import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# the sim: the repo's golem_robocasa/, two levels up from tests/g1/ (the containers
# mount both under /home/code)
SIM = os.path.join(os.path.dirname(os.path.dirname(HERE)), "golem_robocasa")
# physics steps per sample: 50 Hz of sim time (1 to record every step)
RECORD_EVERY = int(os.environ.get("GOLEM_TEST_RECORD_EVERY", "10"))
WALK_SPEED = 0.3           # m/s forward
WALK_SECONDS = 5.0         # sim s of /cmd_vel
SENSOR_SECONDS = 10.0      # sim s of listening to the sensors
# pass limits
ARM_TOL = 0.1              # rad: measured arm joints vs the named config
HAND_TOL = 0.05            # rad: measured fingers vs the close pose
MIN_PELVIS_Z = 0.70        # m: the stand-in stands at 0.744
# pelvis z-axis from vertical: HOMIE stands at ~2 deg and leans ~6.5 deg with
# the arms in t_pose; well short of the walking node's 25 deg upright gate
MAX_TILT_DEG = 15.0
WEIGHT_STEP = 0.1          # largest rt/arm_sdk weight change per 20 ms of wall time
RAMP_WALL_S = (0.5, 2.0)   # the safety layer's 1 s weight ramps, in wall s
HOME_CMD_TOL = 0.01        # rad: the commanded arm targets vs home
WALK_SPEED_TOL = 0.3       # mean forward speed within 30% of WALK_SPEED
WAIST_HOLD_SECONDS = 4.0   # sim s after the controller stops, in the walk test
# rad: measured waist roll and pitch vs their hold once the controller has stopped. The arms
# drop when they go limp, and the held waist settles about 0.03-0.05 rad off its hold under
# the changed load (its last gravity torque was for the arms' pose); a limp waist slumps to
# its ~0.5 rad limits
WAIST_HOLD_TOL = 0.1
# sensor topics: the rate from the stamps (sim time) within these fractions of
# the configured one: the MID-360's are reliable, the camera's best effort
RATE_TOL = {"reliable": (0.99, 1.01), "best_effort": (0.95, 1.01)}
# the longest gap between stamps, in periods: the reliable topics lose nothing
# (the IMU's samples are 4 and 6 ms apart), the camera at most two frames in a row
MAX_GAP_PERIODS = {"reliable": 1.5, "best_effort": 3.0}
# the stamps span the listening window: the first at most one period plus this
# (discovery) after its start, the last at most one period plus this before its
# end (a lidar frame is published once its 0.1 s is cast, ~0.11 s after its
# stamp, and its ring may lag 0.2 s more), in sim s
COVER_START_S = 0.5
COVER_END_S = 0.3
FLOOR_TOL = 0.01           # m: a lidar point this close to z = 0 is on the floor
MIN_FLOOR_SHARE = 0.10     # of a frame's points (see lidar_floor)
HEAD_OPTICAL_FRAME = "head_color_optical_frame"
# camera_info's vertical FOV, deg: the D435's color stream (Intel: 69 x 42, +-1),
# which aligned depth follows, not the depth sensor's 58
HEAD_VFOV_DEG = (41.0, 44.0)
OPTICAL_IN_LINK_XYZW = (-0.5, 0.5, -0.5, 0.5)   # rpy (-pi/2, 0, -pi/2), REP 103's optical frame


class StopRecording(Exception):
    """<out>/stop appeared: end the sim"""


# --------------------------------------------------------------------------- #
# sim: h12_mujoco.py --g1 with a recorder on g1.unitree_interface.SimInterface.after_step
# --------------------------------------------------------------------------- #
class Recorder:
    def __init__(self, out):
        self.out = out
        self.stop_file = os.path.join(out, "stop")
        self.rows = {k: [] for k in ("t", "wall", "pelvis", "quat", "upz", "vel", "q", "dq", "gyro",
                                     "hands", "sdk_weight", "sdk_q", "w", "released", "lowcmd_q",
                                     "lowcmd_t", "leg_changes", "leg_hold_max", "lidar_pos", "lidar_mat")}
        self.steps = 0
        self.names = None
        self.dt = None
        self._lidar_site = None
        # every step: how often rt/lowcmd's leg targets change, and the longest
        # they stayed the same (physics steps) since the last sample
        self._legs, self.leg_changes, self._hold, self._hold_max = None, 0, 0, 0

    def step(self, iface):
        """After every physics step (the sim lock is held)"""
        self.steps += 1
        if iface._lowcmd is not None:
            legs = iface._lowcmd.q[:12]
            if self._legs is None or np.any(legs != self._legs):
                self._legs, self.leg_changes, self._hold = legs.copy(), self.leg_changes + 1, 0
            else:
                self._hold += 1
                self._hold_max = max(self._hold_max, self._hold)
        if self.steps % 500 == 0 and os.path.exists(self.stop_file):
            raise StopRecording()
        if self.steps % RECORD_EVERY:
            return
        d, r = iface.data, iface.resolver
        if self.names is None:
            self.names = {"body": list(r.body.names),
                          "hands": [n for h in r.hands.values() for n in h.names]}
            self.dt = float(iface.model.opt.timestep)
            self._lidar_site = r.site_id("mid360_link")
        sdk, lowcmd = iface._arm_sdk, iface._lowcmd
        gyro = r.sensor_adr["gyro"][0]
        rows = self.rows
        rows["t"].append(float(d.time))
        rows["wall"].append(time.time())
        rows["pelvis"].append(d.xpos[r.base_body].copy())
        rows["quat"].append(d.qpos[r.base_qpos + 3:r.base_qpos + 7].copy())
        rows["upz"].append(float(d.xmat[r.base_body][8]))
        rows["vel"].append(d.qvel[r.base_qvel:r.base_qvel + 3].copy())
        rows["q"].append(d.qpos[r.body.qpos].copy())
        rows["dq"].append(d.qvel[r.body.qvel].copy())
        rows["gyro"].append(d.sensordata[gyro:gyro + 3].copy())
        rows["hands"].append(np.concatenate([d.qpos[h.qpos] for h in r.hands.values()]))
        rows["sdk_weight"].append(math.nan if sdk is None else sdk.weight)
        rows["sdk_q"].append(np.full(17, math.nan) if sdk is None else sdk.q.copy())
        rows["w"].append(iface.arm_sdk_w)
        rows["released"].append(iface.released)
        rows["lowcmd_q"].append(np.full(29, math.nan) if lowcmd is None else lowcmd.q.copy())
        rows["lowcmd_t"].append(math.nan if lowcmd is None else lowcmd.time)
        rows["leg_changes"].append(self.leg_changes)
        rows["leg_hold_max"].append(self._hold_max)
        self._hold_max = 0
        # the MID-360's pose (livox_frame in the world), of the kinematics mj_step
        # left: at t - dt
        rows["lidar_pos"].append(d.site_xpos[self._lidar_site].copy())
        rows["lidar_mat"].append(d.site_xmat[self._lidar_site].copy())

    def save(self):
        arrays = {k: np.array(v) for k, v in self.rows.items()}
        np.savez(os.path.join(self.out, "sim.npz"), names=json.dumps(self.names),
                 hz=500.0 / RECORD_EVERY, dt=self.dt, **arrays)
        print(f"[test_sim_bringup] saved {len(self.rows['t'])} samples to {self.out}/sim.npz", flush=True)


def run_sim(out, sim_args):
    import runpy
    import signal

    sys.path.insert(0, SIM)
    # the G1's own classes: golem_robocasa's top-level unitree_interface is the H1-2's
    from g1 import unitree_interface
    from g1.g1_sensors import SensorConfig

    # what the drive expects to hear: the sensors as the sim reads its knobs
    with open(os.path.join(out, "sensors.json"), "w") as f:
        json.dump(SensorConfig.from_env()._asdict(), f, indent=1)
    recorder = Recorder(out)
    after_step = unitree_interface.SimInterface.after_step

    def recorded_after_step(self):
        after_step(self)
        recorder.step(self)

    def interrupt(signum, frame):
        raise KeyboardInterrupt()

    unitree_interface.SimInterface.after_step = recorded_after_step
    signal.signal(signal.SIGTERM, interrupt)
    script = os.path.join(SIM, "h12_mujoco.py")
    sys.argv = [script, "--g1"] + sim_args
    try:
        runpy.run_path(script, run_name="__main__")
    except (StopRecording, KeyboardInterrupt):
        print("[test_sim_bringup] sim stopped", flush=True)
    finally:
        recorder.save()


# --------------------------------------------------------------------------- #
# drive: the test's steps against the running bringup
# --------------------------------------------------------------------------- #
def _cdr_stamp(raw):
    """Sim seconds of the builtin_interfaces/Time that starts a serialized message
    (a Header's stamp, or rosgraph_msgs/Clock)"""
    sec, nanosec = struct.unpack_from("<iI", raw, 4)
    return sec + nanosec * 1e-9


def _cdr_frame_id(raw):
    """frame_id of a serialized message that starts with a std_msgs/Header"""
    size, = struct.unpack_from("<I", raw, 12)
    return raw[16:16 + size - 1].decode()


# the sim's sensor topics: (topic, type, QoS, frame_id, SensorConfig switch, its rate's field)
SENSOR_TOPICS = (
    ("/livox/lidar", "livox_ros_driver2.msg.CustomMsg", "reliable", "livox_frame", "lidar", "lidar_hz"),
    ("/livox/imu", "sensor_msgs.msg.Imu", "reliable", "livox_frame", "lidar", "imu_hz"),
    ("/realsense/head/color/image_raw", "sensor_msgs.msg.Image", "best_effort",
     "head_color_optical_frame", "camera", "camera_hz"),
    ("/realsense/head/color/image_raw/compressed", "sensor_msgs.msg.CompressedImage", "best_effort",
     "head_color_optical_frame", "camera", "camera_hz"),
    ("/realsense/head/aligned_depth_to_color/image_raw", "sensor_msgs.msg.Image", "best_effort",
     "head_color_optical_frame", "camera", "camera_hz"),
    ("/realsense/head/aligned_depth_to_color/image_raw/compressedDepth", "sensor_msgs.msg.CompressedImage",
     "best_effort", "head_color_optical_frame", "camera", "camera_hz"),
    ("/realsense/head/color/camera_info", "sensor_msgs.msg.CameraInfo", "best_effort",
     "head_color_optical_frame", "camera", "camera_hz"),
)


def _msg_type(path):
    import importlib
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


class Driver:
    def __init__(self, out):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("test_sim_bringup")
        self.out = out
        self.events = []
        self.info = {}
        # the sim's time, for waits in sim time
        self.clock = None
        self.clocks = 0   # /clock messages heard
        self.node.create_subscription(Clock, "/clock", self._on_clock,
                                      QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT), raw=True)

    def _on_clock(self, raw):
        self.clock = _cdr_stamp(raw)
        self.clocks += 1

    def mark(self, name, **extra):
        self.events.append({"name": name, "wall": time.time(), "sim": self.clock, **extra})
        print(f"[test_sim_bringup] {name} {extra if extra else ''}", flush=True)
        with open(os.path.join(self.out, "events.json"), "w") as f:
            json.dump({"events": self.events, "info": self.info}, f, indent=1)

    def spin_for(self, seconds, wall_limit=120.0):
        """Spin for `seconds` of sim time (from /clock), from the first /clock
        heard in it (the last one may be old: nothing spun since); fail after
        wall_limit s of wall time (the sim stalled), so the steps stop rather than
        run short. Returns the sim times it spun from and to."""
        give_up = time.time() + wall_limit
        heard = self.clocks
        while self.clocks == heard and time.time() < give_up:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.clocks == heard:
            raise RuntimeError("no /clock from the sim")
        start = self.clock
        while self.clock < start + seconds:
            if time.time() > give_up:
                self.mark("sim_stalled", sim_seconds=self.clock - start, wanted=seconds)
                raise RuntimeError(f"the sim's clock advanced {self.clock - start:.2f} of {seconds} s "
                                   f"in {wall_limit:.0f} s wall")
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        return start, self.clock

    def wait_future(self, future, timeout):
        end = time.time() + timeout
        while not future.done() and time.time() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        return future.result() if future.done() else None

    def wait_for_controller(self, timeout=240.0):
        """frame_task_server creates /named_config after its startup move to 'home'"""
        from rclpy.action import ActionClient
        from custom_ros_messages.action import NamedConfig
        self.named_config = ActionClient(self.node, NamedConfig, "named_config")
        if not self.named_config.wait_for_server(timeout_sec=timeout):
            self.mark("controller_timeout")
            return False
        self.mark("controller_up")
        return True

    def named_config_goal(self, name, timeout=60.0):
        from custom_ros_messages.action import NamedConfig
        goal = NamedConfig.Goal()
        goal.config_name, goal.plan = name, False
        self.mark(f"{name}_sent")
        handle = self.wait_future(self.named_config.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            self.mark(f"{name}_result", accepted=False, success=False)
            return False
        result = self.wait_future(handle.get_result_async(), timeout)
        success = bool(result is not None and result.result.success)
        self.mark(f"{name}_result", accepted=True, status=None if result is None else result.status,
                  success=success)
        return success

    def trigger(self, service, timeout=30.0):
        from std_srvs.srv import Trigger
        client = self.node.create_client(Trigger, service)
        if not client.wait_for_service(timeout_sec=timeout):
            self.mark(f"{service}_response", success=False, message="no service")
            return False
        self.mark(f"{service}_sent")
        response = self.wait_future(client.call_async(Trigger.Request()), timeout)
        success = bool(response is not None and response.success)
        self.mark(f"{service}_response", success=success,
                  message=None if response is None else response.message)
        return success

    def stop_frame_task_server(self):
        import signal
        import subprocess
        pids = subprocess.run(["pgrep", "-f", "lib/golem_upper_ik/frame_task_server"],
                              capture_output=True, text=True).stdout.split()
        for pid in pids:
            os.kill(int(pid), signal.SIGINT)   # as Ctrl-C
        self.mark("controller_stopped", pids=pids)

    def wait_for_lowcmd(self, timeout=60.0):
        """The split-mode safety layer's first rt/lowcmd (the sim releases its stand-in)"""
        from rclpy.qos import qos_profile_sensor_data
        from unitree_hg.msg import LowCmd
        seen = []
        sub = self.node.create_subscription(LowCmd, "lowcmd", seen.append, qos_profile_sensor_data)
        end = time.time() + timeout
        while not seen and time.time() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        self.node.destroy_subscription(sub)
        self.mark("lowcmd_first" if seen else "lowcmd_timeout")
        return bool(seen)

    def walk(self, speed, seconds, rate=20.0):
        """/cmd_vel forward at `rate` Hz of wall time for `seconds` of sim time"""
        from geometry_msgs.msg import Twist
        pub = self.node.create_publisher(Twist, "cmd_vel", 10)
        msg = Twist()
        msg.linear.x = speed
        self.mark("walk_start", speed=speed)
        end, give_up = self.clock + seconds, time.time() + 120.0
        while self.clock < end:
            if time.time() > give_up:
                self.mark("sim_stalled", wanted=seconds)
                raise RuntimeError(f"the sim's clock did not advance {seconds} s in 120 s wall")
            pub.publish(msg)
            wake = time.time() + 1.0 / rate
            while time.time() < wake:
                self.rclpy.spin_once(self.node, timeout_sec=max(0.0, wake - time.time()))
        self.mark("walk_stop")   # no more /cmd_vel: the node stops after commands.timeout

    def check_sensors(self, seconds=SENSOR_SECONDS):
        """Listen to the sensor topics for `seconds` of sim time, as their
        consumers would (the MID-360's reliable, the camera's best effort):
        every message's stamp, frame_id and arrival, and the window's sim start
        and end; one message of each in full (the IMU's all); the sim's node's
        publishers; /tf_static, as tf2 listens to it; one /livox/lidar frame to
        <out>/lidar_frame.npz. The report checks them."""
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                               qos_profile_sensor_data)
        from rclpy.serialization import deserialize_message
        from tf2_msgs.msg import TFMessage
        with open(os.path.join(self.out, "sensors.json")) as f:
            config = json.load(f)
        qos = {"reliable": QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                      history=HistoryPolicy.KEEP_LAST, depth=256),
               "best_effort": qos_profile_sensor_data}
        topics = [t for t in SENSOR_TOPICS if config[t[4]]]
        heard = {topic: [] for topic, *_ in topics}
        raw = {topic: [] for topic, *_ in topics}
        subs = []
        for topic, kind, reliability, *_ in topics:
            def received(msg, topic=topic):
                heard[topic].append((_cdr_stamp(msg), _cdr_frame_id(msg), time.time(), self.clock))
                if len(raw[topic]) < (100_000 if topic == "/livox/imu" else 1) or topic == "/livox/lidar":
                    raw[topic].append(msg)
            subs.append(self.node.create_subscription(_msg_type(kind), topic, received, qos[reliability],
                                                      raw=True))
        tf_static = []

        def static(msg):
            for tf in msg.transforms:
                p, r = tf.transform.translation, tf.transform.rotation
                tf_static.append([tf.header.frame_id, tf.child_frame_id, [p.x, p.y, p.z], [r.x, r.y, r.z, r.w]])
        subs.append(self.node.create_subscription(
            TFMessage, "/tf_static", static, QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                                                        durability=DurabilityPolicy.TRANSIENT_LOCAL)))
        self.mark("sensors_start")
        window = self.spin_for(seconds)
        self.mark("sensors_end", window_sim=list(window))
        for sub in subs:
            self.node.destroy_subscription(sub)
        result = {"config": config, "window_sim": list(window), "topics": {}, "samples": {}, "tf_static": tf_static,
                  "publishers": sorted([name, sorted(types)] for name, types in
                                       self.node.get_publisher_names_and_types_by_node("mujoco_sensors", "/"))}
        for topic, kind, reliability, frame_id, _, rate in topics:
            rows = heard[topic]
            stamps = np.array([r[0] for r in rows])
            walls = np.array([r[2] for r in rows])
            ages = np.array([r[3] - r[0] for r in rows if r[3] is not None])
            result["topics"][topic] = {
                "count": len(rows), "reliability": reliability, "expected_frame_id": frame_id,
                "expected_hz": config[rate], "frame_ids": sorted({r[1] for r in rows}),
                "stamps": [float(stamps[0]), float(stamps[-1])] if len(rows) else None,
                "increasing": bool(np.all(np.diff(stamps) > 0)) if len(rows) else None,
                "hz_sim": float((len(rows) - 1) / (stamps[-1] - stamps[0])) if len(rows) > 1 else None,
                "hz_wall": float((len(rows) - 1) / (walls[-1] - walls[0])) if len(rows) > 1 else None,
                "longest_gap_sim": float(np.diff(stamps).max()) if len(rows) > 1 else None,
                "age_sim_median": float(np.median(ages)) if ages.size else None}
        samples = result["samples"]
        for topic, kind, *_ in topics:
            if not raw[topic]:
                continue
            msg = deserialize_message(raw[topic][0], _msg_type(kind))
            if kind.endswith(".Image"):
                data = np.frombuffer(bytes(msg.data), np.uint8)
                sample = {"height": msg.height, "width": msg.width, "encoding": msg.encoding, "step": msg.step,
                          "bytes": int(data.size), "is_bigendian": msg.is_bigendian}
                if msg.encoding == "16UC1":
                    sample["valid_share"] = float((data.view("<u2") > 0).mean())
                samples[topic] = sample
            elif kind.endswith(".CompressedImage"):
                samples[topic] = {"format": msg.format, "bytes": len(msg.data)}
            elif kind.endswith(".CameraInfo"):
                samples[topic] = {"height": msg.height, "width": msg.width, "k": list(msg.k), "p": list(msg.p),
                                  "distortion_model": msg.distortion_model, "d": list(msg.d)}
            elif kind.endswith(".Imu"):
                imus = [deserialize_message(m, _msg_type(kind)) for m in raw[topic]]
                acc = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]
                                for m in imus])
                gyro = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for m in imus])
                q = imus[0].orientation
                samples[topic] = {"count": len(imus), "acc_mean_g": acc.mean(axis=0).tolist(),
                                  "acc_norm_range_g": [float(v) for v in np.percentile(
                                      np.linalg.norm(acc, axis=1), [1, 99])],
                                  "gyro_abs_max": float(np.abs(gyro).max()),
                                  "orientation_xyzw": [q.x, q.y, q.z, q.w]}
        if raw.get("/livox/lidar"):
            # the frame in the middle of the window, whole
            frame = raw["/livox/lidar"][len(raw["/livox/lidar"]) // 2]
            msg = deserialize_message(frame, _msg_type(SENSOR_TOPICS[0][1]))
            pts = msg.points
            np.savez(os.path.join(self.out, "lidar_frame.npz"),
                     x=np.array([p.x for p in pts], np.float32), y=np.array([p.y for p in pts], np.float32),
                     z=np.array([p.z for p in pts], np.float32),
                     offset_time=np.array([p.offset_time for p in pts], np.uint32),
                     line=np.array([p.line for p in pts], np.uint8),
                     reflectivity=np.array([p.reflectivity for p in pts], np.uint8),
                     tag=np.array([p.tag for p in pts], np.uint8), timebase=np.uint64(msg.timebase),
                     stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, frame_id=msg.header.frame_id,
                     lidar_id=msg.lidar_id, point_num=msg.point_num)
            samples["/livox/lidar"] = {"frames_kept": len(raw["/livox/lidar"]), "point_num": msg.point_num,
                                       "points": len(pts), "lidar_id": msg.lidar_id}
        self.info["sensors"] = result
        self.mark("sensors_checked")

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def run_drive(out, walk):
    from g1_dex3.dex3 import JOINT_NAMES, named_pose
    from golem_upper_ik.profile import load_profile

    profile = load_profile("g1")
    d = Driver(out)
    d.info = {"mode": "walk" if walk else "arm", "arm_joints": list(profile.enabled_joints),
              "home": profile.named_configs["home"].tolist(),
              "t_pose": profile.named_configs["t_pose"].tolist(),
              "left_hand_joints": JOINT_NAMES["left"],
              "left_close": named_pose("left", "close").tolist(),
              "walk_speed": WALK_SPEED, "walk_seconds": WALK_SECONDS}
    try:
        if walk:
            # split mode: the controller's first command and the node's start the
            # safety layer's rt/lowcmd, which releases the sim's stand-in
            if not d.wait_for_lowcmd(timeout=240.0):
                return 1
            d.spin_for(5.0)
            d.mark("stand_end")
            d.walk(WALK_SPEED, WALK_SECONDS)
            d.spin_for(2.0)
            d.check_sensors()   # standing again, in open floor
            # the controller alone stops while the walking node balances the robot
            d.stop_frame_task_server()
            d.spin_for(WAIST_HOLD_SECONDS)
        else:
            if not d.wait_for_controller():
                return 1
            d.check_sensors()   # while the arms are at home
            d.named_config_goal("t_pose")
            d.spin_for(1.0)
            d.trigger("/left/hand/close")
            d.spin_for(2.5)     # the hand check averages the last 0.5 s of it
            d.stop_frame_task_server()
            d.spin_for(6.0)
        d.mark("end")
        return 0
    finally:
        d.close()


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _tilt_deg(upz):
    return np.degrees(np.arccos(np.clip(upz, -1.0, 1.0)))


def _yaw(quat):
    w, x, y, z = np.asarray(quat).T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _largest_change(values, clock, span=0.02):
    """The largest change of values per `span` s of clock, each over the shortest
    run of samples that spans at least that long, whatever the sampling rate"""
    j = np.searchsorted(clock, clock + span)
    i = np.nonzero(j < len(clock))[0]
    if not i.size:
        return 0.0
    j = j[i]
    return float(np.max(np.abs(values[j] - values[i]) * span / (clock[j] - clock[i])))


class Report:
    def __init__(self, out):
        self.out = out
        sim = np.load(os.path.join(out, "sim.npz"))
        self.s = {k: sim[k] for k in sim.files if k not in ("names", "hz", "dt")}
        self.names = json.loads(str(sim["names"]))
        self.hz = float(sim["hz"]) if "hz" in sim.files else 50.0   # samples per sim second
        self.dt = float(sim["dt"]) if "dt" in sim.files else 0.002  # physics step, s
        with open(os.path.join(out, "events.json")) as f:
            ev = json.load(f)
        self.info = ev["info"]
        self.events = {e["name"]: e for e in ev["events"]}
        self.checks = []
        self.numbers = {}

    def check(self, name, ok, text):
        self.checks.append({"check": name, "pass": bool(ok), "detail": text})
        print(f"{'PASS' if ok else 'FAIL'} {name}: {text}")

    def at(self, event):
        """Index of the first sample at or after an event's wall time"""
        wall = self.events[event]["wall"]
        return int(min(np.searchsorted(self.s["wall"], wall), len(self.s["wall"]) - 1))

    def window(self, start, end, start_offset=0.0):
        """Sample indices between two events, start shifted by start_offset s of sim time"""
        t = self.s["t"]
        t0 = t[self.at(start)] + start_offset
        t1 = t[self.at(end)]
        return np.nonzero((t >= t0) & (t <= t1))[0]

    def joints(self, names, rows=slice(None)):
        idx = [self.names["body"].index(n) for n in names]
        return self.s["q"][rows][:, idx]

    def standing(self, rows, label):
        z = self.s["pelvis"][rows, 2]
        tilt = _tilt_deg(self.s["upz"][rows])
        xy = self.s["pelvis"][rows, :2]
        drift = np.linalg.norm(xy - xy[0], axis=1).max()
        t = self.s["t"][rows]
        self.numbers[f"standing_{label}"] = {"z_min": float(z.min()), "z_max": float(z.max()),
                                             "tilt_max_deg": float(tilt.max()), "xy_drift_m": float(drift),
                                             "sim_t": [float(t[0]), float(t[-1])]}
        self.check(f"standing ({label})", z.min() >= MIN_PELVIS_Z and tilt.max() <= MAX_TILT_DEG,
                   f"sim t {t[0]:.1f}-{t[-1]:.1f} s: pelvis z {z.min():.3f}-{z.max():.3f} m, tilt max "
                   f"{tilt.max():.1f} deg, xy drift {drift * 100:.1f} cm (limits z >= {MIN_PELVIS_Z}, "
                   f"tilt <= {MAX_TILT_DEG})")

    def phases(self, spans):
        """Pelvis height and tilt per phase, for the record"""
        z, tilt, t = self.s["pelvis"][:, 2], _tilt_deg(self.s["upz"]), self.s["t"]
        self.numbers["phases"] = {}
        for label, a, b in spans:
            if b <= a:
                continue
            self.numbers["phases"][label] = {"sim_t": [float(t[a]), float(t[b - 1])],
                                             "z_min": float(z[a:b].min()), "z_max": float(z[a:b].max()),
                                             "tilt_max_deg": float(tilt[a:b].max())}
            print(f"     {label:16s} sim t {t[a]:5.1f}-{t[b - 1]:5.1f} s: pelvis z {z[a:b].min():.3f}-"
                  f"{z[a:b].max():.3f} m, tilt max {tilt[a:b].max():4.1f} deg")

    def arm_error(self, label, target_name, rows):
        target = np.array(self.info[target_name])
        q = self.joints(self.info["arm_joints"], rows).mean(axis=0)
        err = np.abs(q - target)
        worst = self.info["arm_joints"][int(err.argmax())]
        cmd = self.s["sdk_q"][rows].mean(axis=0)
        arm_slots = [self.names["body"].index(n) - 12 for n in self.info["arm_joints"]]
        cmd_err = np.abs(cmd[arm_slots] - target).max()
        waist = self.joints(["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"], rows).mean(axis=0)
        self.numbers[label] = {"max_err": float(err.max()), "mean_err": float(err.mean()), "worst": worst,
                               "cmd_max_err": float(cmd_err), "waist": waist.tolist()}
        return err, worst, cmd_err, waist

    def weight_ramp(self, rows, rising):
        """(start, end) sim times of a 0 -> 1 or 1 -> 0 ramp of the received weight, its
        largest change per 20 ms of wall time (the safety layer ramps on wall time),
        its first weight and its wall-clock duration"""
        w = self.s["sdk_weight"][rows]
        t = self.s["t"][rows]
        wall = self.s["wall"][rows]
        valid = ~np.isnan(w)
        w, t, wall = w[valid], t[valid], wall[valid]
        lo, hi = (w <= 0.001), (w >= 0.999)
        if rising:
            if not hi.any():
                return None
            end = int(np.argmax(hi))
            start = int(np.nonzero(lo[:end])[0][-1]) if lo[:end].any() else 0
        else:
            if not hi.any() or not lo.any():
                return None
            last_hi = int(np.nonzero(hi)[0][-1])
            after = np.nonzero(lo[last_hi:])[0]
            if not after.size:
                return None
            start, end = last_hi, last_hi + int(after[0])
        step = _largest_change(w[start:end + 1], wall[start:end + 1])
        return float(t[start]), float(t[end]), step, float(w[0]), float(wall[end] - wall[start])

    def ramp_ok(self, wall, step):
        return RAMP_WALL_S[0] <= wall <= RAMP_WALL_S[1] and step <= WEIGHT_STEP

    def arm_test(self):
        s = self.s
        n = len(s["t"])
        # 1. the safety layer engages rt/arm_sdk: weight 0 -> 1
        ramp = self.weight_ramp(slice(0, self.at("t_pose_sent")), rising=True)
        if ramp is None:
            self.check("engage", False, "the rt/arm_sdk weight never reached 1")
        else:
            t0, t1, step, first, wall = ramp
            first_t = s["t"][np.nonzero(~np.isnan(s["sdk_weight"]))[0][0]]
            self.numbers["engage"] = {"first_frame_t": float(first_t), "first_weight": first,
                                      "ramp_start_t": t0, "ramp_end_t": t1, "ramp_wall_s": wall,
                                      "max_step": step}
            self.check("engage", first <= 0.05 and self.ramp_ok(wall, step),
                       f"first rt/arm_sdk frame at sim t {first_t:.2f} s with weight {first:.3f}; weight "
                       f"0 -> 1 over {wall:.2f} s wall (sim t {t0:.2f} to {t1:.2f} s, {t1 - t0:.2f} s), largest "
                       f"change per 20 ms of wall time {step:.3f} (limits {RAMP_WALL_S[0]}-{RAMP_WALL_S[1]} s "
                       f"wall, {WEIGHT_STEP})")
        # 2. frame_task_server took the arms to home at startup. In the sim the
        # stand-in already holds them at 0, which is home, so the targets it
        # commands and the weight it holds are checked too
        rows = self.window("controller_up", "t_pose_sent", start_offset=1.0)
        err, worst, cmd_err, waist = self.arm_error("home", "home", rows)
        span = s["t"][rows[-1]] - s["t"][rows[0]]
        w_min = float(s["w"][rows].min())
        self.numbers["home"]["weight_min"] = w_min
        self.check("home at startup", err.max() <= ARM_TOL and cmd_err <= HOME_CMD_TOL and w_min >= 0.999,
                   f"arms (14) vs home over the {span:.1f} s (sim) before the t_pose goal: max "
                   f"{err.max():.3f} rad ({worst}), mean {err.mean():.3f}; commanded max {cmd_err:.4f} "
                   f"(limit {HOME_CMD_TOL}), weight in effect min {w_min:.3f}; waist yaw/roll/pitch "
                   f"{waist[0]:+.3f} {waist[1]:+.3f} {waist[2]:+.3f}")
        # 3. NamedConfig t_pose succeeds and the arms reach it
        result = self.events.get("t_pose_result", {})
        rows = self.window("t_pose_result", "/left/hand/close_sent", start_offset=0.5)
        err, worst, cmd_err, waist = self.arm_error("t_pose", "t_pose", rows)
        took = result["wall"] - self.events["t_pose_sent"]["wall"]
        self.check("t_pose", bool(result.get("success")) and err.max() <= ARM_TOL,
                   f"goal success {result.get('success')} after {took:.1f} s wall; arms vs t_pose: max "
                   f"{err.max():.3f} rad ({worst}), mean {err.mean():.3f}; commanded max {cmd_err:.4f}; "
                   f"waist {waist[0]:+.3f} {waist[1]:+.3f} {waist[2]:+.3f}")
        # 4. /left/hand/close closes the left hand: the last 0.5 s of sim time
        # before frame_task_server is stopped, at least 2 s after the response
        response = self.events.get("/left/hand/close_response", {})
        rows = self.window("controller_stopped", "controller_stopped", start_offset=-0.5)
        settled = s["t"][rows[0]] - s["t"][self.at("/left/hand/close_response")]
        self.numbers["hand_settle_sim_s"] = float(settled)
        hands = s["hands"][rows].mean(axis=0)
        left = [self.names["hands"].index(n) for n in self.info["left_hand_joints"]]
        right = [i for i in range(len(self.names["hands"])) if i not in left]
        err = np.abs(hands[left] - np.array(self.info["left_close"]))
        before = s["hands"][self.at("/left/hand/close_sent")]
        right_moved = np.abs(hands[right] - before[right]).max()
        self.numbers["hand"] = {"max_err": float(err.max()), "per_joint": err.tolist(),
                                "left_before": before[left].tolist(), "right_moved": float(right_moved)}
        self.check("left hand close", bool(response.get("success")) and err.max() <= HAND_TOL and settled >= 1.9,
                   f"service success {response.get('success')}; left fingers vs the close pose, "
                   f"{settled:.1f}-{settled + 0.5:.1f} s (sim) after the response: max "
                   f"{err.max():.4f} rad, per joint {np.round(err, 4).tolist()} (from "
                   f"{np.round(before[left], 2).tolist()} before); the right hand moved {right_moved:.4f} "
                   f"rad meanwhile")
        # 5. stopping frame_task_server hands the arms back: weight 1 -> 0. Its
        # last command (zero gains) trips golem_safety's arm_sdk.waist_kp_min at
        # once, and the stop's mark is taken just after the signal, so the ramp
        # may start at the sample before the mark. It is looked for from two
        # samples before the mark: a ramp that started earlier has no sample at
        # weight 1 in that window, and fails
        stop = self.at("controller_stopped")
        ramp = self.weight_ramp(slice(max(0, stop - 2), n), rising=False)
        if ramp is None:
            self.check("hand back", False, "the rt/arm_sdk weight did not ramp 1 -> 0 from the stop on "
                                           "(it was below 1 two samples before the stop, or never reached 0)")
        else:
            t0, t1, step, _, wall = ramp
            arms_end = np.abs(self.joints(self.info["arm_joints"], slice(n - int(0.5 * self.hz), n))).max()
            w_end = float(s["w"][-1])
            self.numbers["hand_back"] = {"stop_t": float(s["t"][stop]), "ramp_start_t": t0,
                                         "ramp_start_after_stop_s": t0 - float(s["t"][stop]),
                                         "ramp_end_t": t1, "ramp_wall_s": wall, "max_step": step,
                                         "w_end": w_end, "arms_end_max_abs": float(arms_end)}
            self.check("hand back", self.ramp_ok(wall, step) and w_end == 0.0,
                       f"frame_task_server stopped at sim t {s['t'][stop]:.2f} s; weight 1 -> 0 over "
                       f"{wall:.2f} s wall (sim t {t0:.2f} to {t1:.2f} s, {t1 - t0:.2f} s, starting "
                       f"{t0 - s['t'][stop]:+.2f} s from the stop), largest change "
                       f"per 20 ms of wall time {step:.3f}; "
                       f"weight in effect at the end {w_end:.3f}; arms back at the stand-in's hold "
                       f"(max |q| {arms_end:.3f} rad)")
        # 6. HOMIE stood the robot throughout, and was never released
        self.phases([("stand-in alone", 0, self.at("controller_up")),
                     ("home", self.at("controller_up"), self.at("t_pose_sent")),
                     ("to t_pose", self.at("t_pose_sent"), self.at("t_pose_result")),
                     ("t_pose and hand", self.at("t_pose_result"), stop),
                     ("hand back", stop, self.at("end") + 1)])
        self.standing(slice(0, self.at("end") + 1), "whole run")
        self.check("stand-in kept", not s["released"].any(), "no rt/lowcmd released the stand-in")
        self.sensors()

    def walk_test(self):
        s = self.s
        n = len(s["t"])
        released = np.nonzero(s["released"])[0]
        if not released.size:
            self.check("hand-over", False, "the stand-in was never released (no valid rt/lowcmd)")
            return
        r = int(released[0])
        self.numbers["handover_t"] = float(s["t"][r])
        self.check("hand-over", True, f"the node's rt/lowcmd released the stand-in at sim t {s['t'][r]:.2f} s")
        if "leg_changes" in s:
            # diagnostic: HOMIE needs a fresh leg target every 10 physics steps; a
            # starved ros side delivers fewer, late and in bursts
            self.numbers["leg_targets"] = {}
            for label, e in (("in the first 2 s after the hand-over", min(n - 1, r + int(2 * self.hz))),
                             ("from the hand-over to the end", self.at("end"))):
                got = int(s["leg_changes"][e] - s["leg_changes"][r])
                expected = int(round((s["t"][e] - s["t"][r]) / 0.02))
                hold = float(s["leg_hold_max"][r + 1:e + 1].max() + 1) * 2.0
                self.numbers["leg_targets"][label] = {"received": got, "expected": expected,
                                                      "longest_hold_ms": hold}
                print(f"     leg targets {label}: {got} received of {expected} expected (one per 20 ms "
                      f"of sim time); the longest one held {hold:.0f} ms of sim time")
        self.standing(slice(0, r), "stand-in")
        self.standing(slice(r, self.at("walk_start")), "walking node, standing")
        # walk: base displacement in the heading at the start of the command
        a, b = self.at("walk_start"), self.at("walk_stop")
        c = min(n - 1, b + int(2 * self.hz))   # 2 s after the commands stop
        yaw = _yaw(s["quat"])
        heading = yaw[a]
        fwd = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-fwd[1], fwd[0]])
        d_stop = s["pelvis"][b, :2] - s["pelvis"][a, :2]
        d_end = s["pelvis"][c, :2] - s["pelvis"][a, :2]
        steady = slice(max(a, b - int(3 * self.hz)), b)   # the last 3 s of the command
        v = s["vel"][steady, :2]
        vx, vy = float((v @ fwd).mean()), float((v @ left).mean())
        dur = s["t"][b] - s["t"][a]
        self.numbers["walk"] = {"sim_seconds": float(dur), "forward_at_stop": float(d_stop @ fwd),
                                "lateral_at_stop": float(d_stop @ left), "forward_2s_after": float(d_end @ fwd),
                                "lateral_2s_after": float(d_end @ left), "yaw_change_deg":
                                float(math.degrees(yaw[c] - heading)), "steady_vx": vx, "steady_vy": vy}
        self.check("walk", abs(vx - WALK_SPEED) <= WALK_SPEED_TOL * WALK_SPEED,
                   f"/cmd_vel {WALK_SPEED} m/s for {dur:.1f} s of sim time: forward {d_stop @ fwd:.3f} m, "
                   f"lateral {d_stop @ left:+.3f} m at the stop; {d_end @ fwd:.3f} m, {d_end @ left:+.3f} m "
                   f"2 s later; yaw {math.degrees(yaw[c] - heading):+.1f} deg; mean speed over the last "
                   f"3 s of the command vx {vx:.3f}, vy {vy:+.3f} m/s")
        self.standing(slice(a, c + 1), "walking")
        # until the test ends: stopping the launch then stops rt/lowcmd and the
        # sim turns the motors off
        self.standing(slice(r, self.at("end") + 1), "after the hand-over")
        self.waist_after_controller_stop()
        self.sensors()

    def waist_after_controller_stop(self):
        """frame_task_server stopped while the walking node runs: its last command keeps
        the waist at its hold (golem_safety relays it on rt/lowcmd), so the waist roll and
        pitch stay there instead of slumping onto the walking policy"""
        s = self.s
        stop, end = self.at("controller_stopped"), self.at("end")
        pids = self.events["controller_stopped"].get("pids", [])
        names = ["waist_roll_joint", "waist_pitch_joint"]
        slots = [self.names["body"].index(n) for n in names]
        t = s["t"]
        before = np.nonzero((t >= t[stop] - 0.5) & (t < t[stop]))[0]
        hold = np.median(s["lowcmd_q"][before][:, slots], axis=0)
        rows = np.arange(stop, end + 1)
        err = np.abs(s["q"][rows][:, slots] - hold).max(axis=0)
        cmd_moved = np.abs(s["lowcmd_q"][rows][:, slots] - hold).max(axis=0)
        span = t[end] - t[stop]
        self.numbers["waist_after_controller_stop"] = {
            "hold": hold.tolist(), "max_err": err.tolist(), "cmd_max_change": cmd_moved.tolist(),
            "sim_seconds": float(span)}
        self.check("waist held after the controller stops",
                   bool(pids) and span >= WAIST_HOLD_SECONDS - 0.5 and err.max() <= WAIST_HOLD_TOL,
                   f"frame_task_server (pids {pids}) stopped at sim t {t[stop]:.2f} s; over the {span:.1f} s (sim) after, "
                   f"waist roll / pitch within {err[0]:.4f} / {err[1]:.4f} rad of the hold "
                   f"{hold[0]:+.3f} / {hold[1]:+.3f} rad (limit {WAIST_HOLD_TOL}); the commanded waist "
                   f"moved {cmd_moved[0]:.4f} / {cmd_moved[1]:.4f} rad")

    # ------------------------------------------------------------- sensors
    def sensors(self):
        """What the ros container heard from the sim's sensors (Driver.check_sensors)"""
        heard = self.info.get("sensors")
        if heard is None:
            self.check("sensor topics", False, "the drive did not listen to the sensors")
            return
        config = heard["config"]
        expected = {t[0] for t in SENSOR_TOPICS if config[t[4]]}
        # the sim's node publishes the drivers' topics and nothing else (no
        # /livox/pointcloud, and no /odom or /tf without GOLEM_SIM_ODOM)
        published = {name for name, _ in heard["publishers"]} - {"/rosout", "/parameter_events"}
        surface = expected | {"/clock"} | ({"/tf_static"} if config["camera"] else set())
        self.numbers["sensor_node_publishers"] = sorted(published)
        self.check("sensor topic surface", published == surface,
                   f"the sim's mujoco_sensors node publishes {sorted(published)}"
                   + (f"; missing {sorted(surface - published)}" if surface - published else "")
                   + (f"; extra {sorted(published - surface)}" if published - surface else ""))
        self.numbers["sensor_topics"] = heard["topics"]
        # the listening window, in sim time: each topic's stamps must span it, at
        # its rate throughout (a topic that stops partway fails one or the other)
        start, end = heard.get("window_sim") or (math.nan, math.nan)
        self.numbers["sensor_window_sim"] = [start, end]
        print(f"     listened to the sensors over sim t {start:.3f}-{end:.3f} s ({end - start:.2f} s)")
        for topic, info in heard["topics"].items():
            lo, hi = RATE_TOL[info["reliability"]]
            period = 1.0 / info["expected_hz"]
            ratio = info["hz_sim"] / info["expected_hz"] if info["hz_sim"] else 0.0
            first, last = info["stamps"] or (math.nan, math.nan)
            gap = info["longest_gap_sim"] if info["longest_gap_sim"] is not None else math.nan
            max_gap = MAX_GAP_PERIODS[info["reliability"]] * period
            ok = (lo <= ratio <= hi and info["frame_ids"] == [info["expected_frame_id"]] and info["increasing"]
                  and first - start <= period + COVER_START_S and end - last <= period + COVER_END_S
                  and gap <= max_gap)
            self.check(f"{topic}", ok,
                       f"{info['count']} messages ({info['expected_hz'] * (end - start):.0f} in the window at the "
                       f"rate), {info['hz_sim'] or 0:.2f} Hz of sim time from the stamps "
                       f"({info['expected_hz']:g} expected, {lo:.0%}-{hi:.0%}), {info['hz_wall'] or 0:.2f} Hz "
                       f"wall; first stamp at the window's start {first - start:+.3f} s, last at its end "
                       f"{last - end:+.3f} s (limits <= +{period + COVER_START_S:.3f}, >= "
                       f"-{period + COVER_END_S:.3f}); "
                       f"longest gap {gap:.4f} s (limit {max_gap:.4f}); frame_id {info['frame_ids']}; "
                       f"median age at arrival {info['age_sim_median'] or 0:.3f} s of sim time")
        samples = heard["samples"]
        if config["camera"]:
            self.camera_samples(config, samples)
            self.camera_tf(heard.get("tf_static", []))
        if config["lidar"]:
            self.imu_sample(samples.get("/livox/imu"))
            self.lidar_floor(config)

    def camera_samples(self, config, samples):
        w, h = config["camera_width"], config["camera_height"]
        ns = "/realsense/head"
        color = samples.get(f"{ns}/color/image_raw", {})
        depth = samples.get(f"{ns}/aligned_depth_to_color/image_raw", {})
        jpeg = samples.get(f"{ns}/color/image_raw/compressed", {})
        png = samples.get(f"{ns}/aligned_depth_to_color/image_raw/compressedDepth", {})
        info = samples.get(f"{ns}/color/camera_info", {})
        k = info.get("k", [0] * 9)
        vfov = 2 * math.degrees(math.atan(h / 2 / k[4])) if k[4] > 0 else math.nan
        hfov = 2 * math.degrees(math.atan(w / 2 / k[0])) if k[0] > 0 else math.nan
        ok = (color.get("encoding") == "rgb8" and (color.get("width"), color.get("height")) == (w, h)
              and color.get("bytes") == w * h * 3 and depth.get("encoding") == "16UC1"
              and (depth.get("width"), depth.get("height")) == (w, h) and depth.get("bytes") == w * h * 2
              and depth.get("valid_share", 0) >= 0.5
              and jpeg.get("format") == "rgb8; jpeg compressed bgr8" and png.get("format") == "16UC1; compressedDepth"
              and (info.get("width"), info.get("height")) == (w, h)
              and math.isclose(k[2], (w - 1) / 2) and math.isclose(k[5], (h - 1) / 2) and k[0] > 0
              and k[0] == k[4] and HEAD_VFOV_DEG[0] <= vfov <= HEAD_VFOV_DEG[1])
        self.check("head camera messages", ok,
                   f"color {color.get('width')}x{color.get('height')} {color.get('encoding')}, depth "
                   f"{depth.get('width')}x{depth.get('height')} {depth.get('encoding')} with depth on "
                   f"{depth.get('valid_share', 0):.0%} of the pixels, compressed {jpeg.get('format')!r} "
                   f"({jpeg.get('bytes')} B) and {png.get('format')!r} ({png.get('bytes')} B), camera_info "
                   f"{info.get('width')}x{info.get('height')} fx {k[0]:.1f} fy {k[4]:.1f} cx {k[2]:.1f} "
                   f"cy {k[5]:.1f}, FOV {hfov:.1f} x {vfov:.1f} deg (vertical {HEAD_VFOV_DEG[0]:g}-"
                   f"{HEAD_VFOV_DEG[1]:g})")

    def camera_tf(self, tf_static):
        """/tf_static puts head_color_optical_frame on the URDF's d435_link, rotated
        by the ROS optical-frame convention (z forward, x right, y down: rpy
        -pi/2, 0, -pi/2), and nothing else gives that frame a parent"""
        mine = [tf for tf in tf_static if tf[1] == HEAD_OPTICAL_FRAME]
        found = sorted({(parent, tuple(round(v, 6) for v in xyz), tuple(round(v, 6) for v in xyzw))
                        for parent, _, xyz, xyzw in mine})
        ok = (len(found) == 1 and found[0][0] == "d435_link" and np.allclose(found[0][1], 0.0, atol=1e-9)
              and np.allclose(found[0][2], OPTICAL_IN_LINK_XYZW, atol=1e-6))
        self.numbers["camera_tf_static"] = [list(f) for f in found]
        self.check("head camera frame on /tf_static", ok,
                   f"{len(mine)} transform(s) to {HEAD_OPTICAL_FRAME}: {found} (expected d435_link, translation 0, "
                   f"rotation xyzw {list(OPTICAL_IN_LINK_XYZW)}); {len(tf_static)} static transforms in all")

    def imu_sample(self, imu):
        """Standing, the MID-360's IMU reads 1 g up, which is its -z: it hangs upside down"""
        if not imu:
            self.check("/livox/imu contents", False, "no /livox/imu message")
            return
        acc = np.array(imu["acc_mean_g"])
        ok = (abs(np.linalg.norm(acc) - 1.0) <= 0.05 and acc[2] <= -0.95
              and imu["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0])
        self.check("/livox/imu contents", ok,
                   f"{imu['count']} samples: mean specific force {np.round(acc, 3).tolist()} g (norm "
                   f"{np.linalg.norm(acc):.3f}; 1st-99th percentile of the norm "
                   f"{imu['acc_norm_range_g'][0]:.3f}-{imu['acc_norm_range_g'][1]:.3f}), largest |gyro| "
                   f"{imu['gyro_abs_max']:.3f} rad/s, orientation {imu['orientation_xyzw']} (left at the default)")

    def lidar_floor(self, config):
        """One /livox/lidar frame against the sensor's pose in the sim: placed with
        the pose at each point's time, a share of its points lie on the floor
        (z = 0), none below it, and in livox_frame those are at +z, where the
        upside-down mount puts the floor"""
        path = os.path.join(self.out, "lidar_frame.npz")
        if not os.path.exists(path):
            self.check("lidar frame on the floor", False, "no /livox/lidar frame was saved")
            return
        f = np.load(path)
        p = np.stack([f["x"], f["y"], f["z"]], axis=1).astype(np.float64)
        when = int(f["timebase"]) * 1e-9 + f["offset_time"] * 1e-9
        # the recorded pose of the kinematics at t - dt nearest each point's time
        t_pose = self.s["t"] - self.dt
        k = np.clip(np.searchsorted(t_pose, when), 1, len(t_pose) - 1)
        k = np.where(np.abs(t_pose[k - 1] - when) < np.abs(t_pose[k] - when), k - 1, k)
        off = np.abs(t_pose[k] - when).max()
        rot = self.s["lidar_mat"][k].reshape(-1, 3, 3)
        world = np.einsum("nij,nj->ni", rot, p) + self.s["lidar_pos"][k]
        z = world[:, 2]
        floor = np.abs(z) <= FLOOR_TOL
        below = int((z < -FLOOR_TOL).sum())
        up = rot[len(rot) // 2].T @ [0.0, 0.0, 1.0]   # world up in livox_frame
        height = float(self.s["lidar_pos"][k[len(k) // 2], 2])
        lz = p[floor, 2]
        meta_ok = (str(f["frame_id"]) == "livox_frame" and int(f["lidar_id"]) == 192
                   and int(f["point_num"]) == len(p) and abs(float(f["stamp"]) - int(f["timebase"]) * 1e-9) < 1e-9
                   and np.all(np.diff(f["offset_time"].astype(np.int64)) >= 0)
                   and int(f["offset_time"].max()) <= 1e9 / config["lidar_hz"])
        share = float(floor.mean()) if len(p) else 0.0
        self.numbers["lidar_frame"] = {
            "points": len(p), "configured": config["lidar_points"], "floor_share": share,
            "floor_rms_mm": float(np.sqrt(np.mean(z[floor] ** 2)) * 1e3) if floor.any() else None,
            "below_floor": below, "floor_livox_z": [float(lz.min()), float(lz.max())] if lz.size else None,
            "sensor_height": height, "world_up_in_livox": up.tolist(), "pose_time_off_s": float(off)}
        self.check("lidar frame on the floor",
                   meta_ok and share >= MIN_FLOOR_SHARE and below == 0 and lz.size and lz.min() > 0
                   and len(p) >= 0.5 * config["lidar_points"] and off <= 1.0 / self.hz,
                   f"{len(p)} points ({config['lidar_points']} rays), lidar_id {int(f['lidar_id'])}, "
                   f"frame_id {str(f['frame_id'])}; {share:.0%} on the floor (within {FLOOR_TOL * 100:.0f} cm "
                   f"of z = 0, RMS {self.numbers['lidar_frame']['floor_rms_mm'] or 0:.2f} mm), {below} below it; "
                   f"the sensor {height:.3f} m up, world up = {np.round(up, 3).tolist()} in livox_frame, so the "
                   f"floor points are at livox z {lz.min() if lz.size else 0:+.3f} to "
                   f"{lz.max() if lz.size else 0:+.3f} m")

    def run(self, out):
        t, wall = self.s["t"], self.s["wall"]
        self.numbers["real_time_factor"] = float((t[-1] - t[0]) / (wall[-1] - wall[0]))
        # and over each second of sim time
        edges = np.searchsorted(t, np.arange(t[0], t[-1], 1.0))
        per_s = np.diff(t[edges]) / np.diff(wall[edges]) if len(edges) > 2 else np.array([math.nan])
        self.numbers["real_time_factor_per_s"] = {"min": float(per_s.min()), "p5": float(np.percentile(per_s, 5)),
                                                  "median": float(np.median(per_s))}
        # and over the test's steps, once the bringup is up (its start loads the host)
        first = next((e for e in ("controller_up", "lowcmd_first") if e in self.events), None)
        steps = ""
        if first and "end" in self.events:
            a, b = self.at(first), self.at("end")
            if b > a:
                self.numbers["real_time_factor_steps"] = float((t[b] - t[a]) / (wall[b] - wall[a]))
                steps = f"; {self.numbers['real_time_factor_steps']:.2f} from {first} to the end"
        print(f"     {len(t)} samples, sim t {t[0]:.1f}-{t[-1]:.1f} s, real-time factor "
              f"{self.numbers['real_time_factor']:.2f} (per sim second: min {per_s.min():.2f}, 5th percentile "
              f"{np.percentile(per_s, 5):.2f}, median {np.median(per_s):.2f}){steps}")
        try:
            self.walk_test() if self.info["mode"] == "walk" else self.arm_test()
        except KeyError as e:
            # the steps stopped early: say so, and report what was recorded
            self.check("test steps", False, f"the run has no {e} mark, so the test's steps did not "
                                            "finish (see drive.log)")
            released = np.nonzero(self.s["released"])[0]
            if released.size:
                print(f"     the stand-in was released at sim t {t[released[0]]:.2f} s")
            self.standing(slice(0, len(t)), "whole recording")
        ok = all(c["pass"] for c in self.checks)
        with open(os.path.join(out, "report.json"), "w") as f:
            json.dump({"pass": ok, "checks": self.checks, "numbers": self.numbers}, f, indent=1)
        print(f"{'PASSED' if ok else 'FAILED'}: {sum(c['pass'] for c in self.checks)}/{len(self.checks)} checks")
        return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=("sim", "drive", "report"))
    parser.add_argument("--out", required=True, help="the shared output directory")
    parser.add_argument("--walk", action="store_true", help="drive: the walk test")
    parser.epilog = "sim: the G1 sim's arguments (h12_mujoco.py --g1) follow a --"
    argv = sys.argv[1:]
    sim_args = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:len(argv) - len(sim_args) - ("--" in argv)])
    if args.role == "sim":
        run_sim(args.out, sim_args)
        return 0
    if args.role == "drive":
        return run_drive(args.out, args.walk)
    return Report(args.out).run(args.out)


if __name__ == "__main__":
    sys.exit(main())
