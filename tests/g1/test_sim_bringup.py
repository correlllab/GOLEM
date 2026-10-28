"""Acceptance test of the G1 sim with the whole bringup: the parts that run in
the containers. test_sim_bringup.sh runs them; see it for the usage.

  sim     in the robocasa container: h12_mujoco.py --g1 as __main__ (its own
          CLI), recording the robot's state from the sim after every 10th physics
          step (50 Hz of sim time; GOLEM_TEST_RECORD_EVERY=1 for every step)
          to <out>/sim.npz, until <out>/stop appears
  drive   in the ros container, beside 'ros2 launch golem_bringup
          g1_sim.launch.py': the test's steps, with their wall times
          and results in <out>/events.json
  report  anywhere with numpy: checks the recording against the steps,
          prints the numbers and writes <out>/report.json; exit 1 on a failure

Both tests start as the G1 starts in debug mode: the limp robot hangs from the
sim's elastic band while golem_safety streams zero-gain rt/lowcmd, HOMIE's
walking node pre-poses the legs and starts its policy under the band, and it
releases the band once frame_task_server is up. Both check that start-up: the
band holds the robot upright, golem_safety latches no estop, the walking node
starts, the band comes off once (after the first /left_ee_pose the test sees,
and within the walking node's band.max_wait of the policy's start), and the
robot stands for 5 s after it. The report also gives the pelvis and the feet
around the release, for the record.

The arm test (default), after the release: frame_task_server took the arms
to 'home' at startup, a NamedConfig 't_pose' goal succeeds and the arms reach
it, /left/hand/close closes the left hand, and stopping frame_task_server
leaves the waist roll and pitch at their hold (its shutdown command keeps them
stiff on rt/lowcmd) while the walking node keeps the robot standing. The walk
test (--walk) stands 5 s after the release, walks on /cmd_vel 0.3 m/s forward
for 5 s, and after the sensors stops frame_task_server alone, with the same
waist check. Both tests listen to the sensors from the ros container for 10 s
(after the release, or after the walk): every sensor topic at its rate in sim
time throughout the window, with its frame_id and its message contents, the
camera's frame on /tf_static, the sim's node publishing nothing else, and one
/livox/lidar frame putting the floor where the upside-down mount predicts.
The joint, base, feet and sensor poses and the band's state come from the
sim's own state; no ground truth is published.

The steps wait in sim time (/clock) and the checks measure motion in sim time,
so a sim slower than real time stretches the test instead of failing it; the
band's release against the policy's start is checked in wall time, as the
walking node times its band.max_wait.
"""
import argparse
import json
import math
import os
import re
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
MIN_PELVIS_Z = 0.70        # m: HOMIE stands at 0.744
# pelvis z-axis from vertical: the limp robot hangs from the band within 4 deg,
# HOMIE stands at ~2 deg and leans ~6.5 deg with the arms in t_pose; well short of
# the walking node's 25 deg upright gate
MAX_TILT_DEG = 15.0
HOME_CMD_TOL = 0.01        # rad: the commanded arm targets vs home
RELEASE_STANDING_S = 5.0   # sim s after the band's release that the robot must stand
RELEASE_MEASURE_S = 3.0    # sim s after the band's release that the report measures, for the record
# wall s past the walking node's band.max_wait that the release may come: the node
# polls at 10 Hz, then calls the sim's /elastic_band/toggle
BAND_POLL_SLACK_S = 0.5
WALK_SPEED_TOL = 0.3       # mean forward speed within 30% of WALK_SPEED
WAIST_HOLD_SECONDS = 4.0   # sim s after the controller stops
# rad: measured waist roll and pitch vs their hold once the controller has stopped. The arms
# drop when they go limp, and the held waist settles up to about 0.06 rad off its hold under
# the changed load (its last gravity torque was for the arms' pose), passing about 0.08 on
# the way after the walk; a limp waist slumps to its ~0.5 rad limits
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
# the feet whose world positions the sim records (the G1's ankle roll links)
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
# the sim's log line once the band is off (golem_robocasa/g1/g1_mujoco.py's fall logger)
BAND_RELEASED_LINE = "elastic band released at t="
# the walking node's log lines (g1_locomotion): its config, and the policy's start
LOCOMOTION_CONFIG = re.compile(r"Loaded (\S+\.yaml) and the policy")
POLICY_STARTED = "walking: policy started"
# a ROS log line in launch.log: level, wall time, logger, text
LOG_LINE = re.compile(r"\[(DEBUG|INFO|WARN|ERROR|FATAL)\] \[(\d+(?:\.\d+)?)\] \[([^\]]+)\]: (.*)")


def file_has(path, text):
    try:
        with open(path, errors="replace") as f:
            return text in f.read()
    except OSError:
        return False


def logged(path, logger):
    """(wall time, level, text) of every line the ROS logger `logger` wrote to a launch
    log, in order"""
    lines = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = LOG_LINE.search(line)
                if m and m.group(3) == logger:
                    lines.append((float(m.group(2)), m.group(1), m.group(4).rstrip()))
    except OSError:
        pass
    return lines


def safety_event(text):
    """The event of one of golem_safety's JSON log lines, else None"""
    if not text.startswith("{"):
        return None
    try:
        return json.loads(text).get("event")
    except ValueError:
        return None


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
                                     "hands", "feet", "band", "commanded", "lowcmd_q", "lowcmd_t",
                                     "leg_changes", "leg_hold_max", "lidar_pos", "lidar_mat")}
        self.steps = 0
        self.names = None
        self.dt = None
        self._lidar_site = None
        self._feet = None
        # every step: the band's state (1 on, 0 off, NaN no band) at each change, with
        # its sim and wall time, and the first step with SimInterface.commanded
        self._band = None
        self.band_changes = []
        self.commanded_at = None
        # every step: how often rt/lowcmd's leg targets change, and the longest
        # they stayed the same (physics steps) since the last sample
        self._legs, self.leg_changes, self._hold, self._hold_max = None, 0, 0, 0

    def step(self, iface):
        """After every physics step (the sim lock is held)"""
        self.steps += 1
        on = None if iface.band is None else bool(iface.band.enabled)
        band = math.nan if on is None else float(on)
        if not self.band_changes or on != self._band:
            self._band = on
            self.band_changes.append([float(iface.data.time), time.time(), band])
        if self.commanded_at is None and iface.commanded:
            self.commanded_at = [float(iface.data.time), time.time()]
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
            self._feet = [r.body_id(n) for n in FEET]
        lowcmd = iface._lowcmd
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
        rows["feet"].append(np.array([d.xpos[b] if b >= 0 else np.full(3, math.nan) for b in self._feet]))
        rows["band"].append(band)
        rows["commanded"].append(bool(iface.commanded))
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
                 hz=500.0 / RECORD_EVERY, dt=self.dt,
                 band_changes=np.array(self.band_changes, float).reshape(-1, 3),
                 commanded_at=np.array(self.commanded_at or [math.nan, math.nan], float), **arrays)
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
        # frame_task_server publishes /left_ee_pose (100 Hz) once its start-up moves are
        # done; the first one's stamp (sim time) is when the test saw the controller ready.
        # The drive spins in every wait, and the deep history keeps that first one
        # through a slow spin
        from geometry_msgs.msg import PoseStamped
        self.ee_pose_first = None
        self.node.create_subscription(PoseStamped, "/left_ee_pose", self._on_ee_pose, QoSProfile(
            depth=1000, reliability=ReliabilityPolicy.RELIABLE))
        self.launch_log = os.path.join(out, "launch.log")
        self.sim_log = os.path.join(out, "sim.log")

    def _on_clock(self, raw):
        self.clock = _cdr_stamp(raw)
        self.clocks += 1

    def _on_ee_pose(self, msg):
        if self.ee_pose_first is None:
            self.ee_pose_first = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.mark("left_ee_pose_first", stamp=self.ee_pose_first)

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

    def spin_until(self, ready, timeout, every=0.25):
        """Spin until ready(), asked every `every` s of wall time, is true; False after
        `timeout` s of wall time"""
        end = time.time() + timeout
        while not ready():
            if time.time() > end:
                return False
            until = time.time() + every
            while time.time() < until:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
        return True

    def wait_for_controller(self, timeout=240.0):
        """frame_task_server creates /named_config after its startup move to 'home'"""
        from rclpy.action import ActionClient
        from custom_ros_messages.action import NamedConfig
        self.named_config = ActionClient(self.node, NamedConfig, "named_config")
        if not self.spin_until(self.named_config.server_is_ready, timeout, every=0.1):
            self.mark("controller_timeout")
            return False
        self.mark("controller_up")
        return True

    def wait_band_released(self, timeout=240.0):
        """The sim logs the band's release, which the walking node asks for once its
        policy runs and frame_task_server is up. Records the walking node's
        band.max_wait, from the config it logged loading"""
        import yaml
        if not self.spin_until(lambda: file_has(self.sim_log, BAND_RELEASED_LINE)
                               or file_has(self.launch_log, "g1_core: not starting"), timeout):
            self.mark("band_timeout")
            return False
        if not file_has(self.sim_log, BAND_RELEASED_LINE):
            self.mark("launch_refused")
            return False
        self.mark("band_released")
        self.info["band_max_wait"] = None
        for _, _, text in logged(self.launch_log, "locomotion_node"):
            m = LOCOMOTION_CONFIG.match(text)
            if m:
                with open(m.group(1)) as f:
                    self.info["locomotion_config"] = m.group(1)
                    self.info["band_max_wait"] = float(yaml.safe_load(f)["band"]["max_wait"])
                break
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
            # the walking node releases the band once its policy runs and the
            # controller is up
            if not d.wait_band_released():
                return 1
            d.spin_for(5.0)
            d.mark("stand_end")
            d.walk(WALK_SPEED, WALK_SECONDS)
            d.spin_for(2.0)
            d.check_sensors()   # standing again, in open floor
        else:
            if not d.wait_for_controller() or not d.wait_band_released():
                return 1
            d.check_sensors()   # standing on HOMIE from the release, the arms at home
            d.named_config_goal("t_pose")
            d.spin_for(1.0)
            d.trigger("/left/hand/close")
            d.spin_for(2.5)     # the hand check averages the last 0.5 s of it
        # the controller alone stops while the walking node balances the robot
        d.stop_frame_task_server()
        d.spin_for(WAIST_HOLD_SECONDS)
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

    def standing(self, rows, label, name=None):
        z = self.s["pelvis"][rows, 2]
        tilt = _tilt_deg(self.s["upz"][rows])
        xy = self.s["pelvis"][rows, :2]
        drift = np.linalg.norm(xy - xy[0], axis=1).max()
        t = self.s["t"][rows]
        self.numbers[f"standing_{label}"] = {"z_min": float(z.min()), "z_max": float(z.max()),
                                             "tilt_max_deg": float(tilt.max()), "xy_drift_m": float(drift),
                                             "sim_t": [float(t[0]), float(t[-1])]}
        self.check(name or f"standing ({label})", z.min() >= MIN_PELVIS_Z and tilt.max() <= MAX_TILT_DEG,
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
        # the targets golem_safety wrote on rt/lowcmd, whose slots are the body's order
        cmd = self.s["lowcmd_q"][rows].mean(axis=0)
        arm_slots = [self.names["body"].index(n) for n in self.info["arm_joints"]]
        cmd_err = np.abs(cmd[arm_slots] - target).max()
        waist = self.joints(["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"], rows).mean(axis=0)
        self.numbers[label] = {"max_err": float(err.max()), "mean_err": float(err.mean()), "worst": worst,
                               "cmd_max_err": float(cmd_err), "waist": waist.tolist()}
        return err, worst, cmd_err, waist

    # ------------------------------------------------------------- start-up
    def release(self):
        """The band's first release, from the sim's record of every step: (index of the
        first sample at or after it, sim t, wall), or None"""
        changes = self.s["band_changes"]
        off = np.nonzero(changes[:, 2] == 0.0)[0]
        if not off.size:
            return None
        t, wall = float(changes[off[0], 0]), float(changes[off[0], 1])
        return int(min(np.searchsorted(self.s["t"], t), len(self.s["t"]) - 1)), t, wall

    def end_wall(self):
        """The wall time of the test's last step: its end mark, or the last mark it made"""
        if "end" in self.events:
            return self.events["end"]["wall"]
        return max(e["wall"] for e in self.events.values())

    def startup(self):
        """The G1's start-up: the band holds the limp robot, golem_safety latches no
        estop, the walking node starts and releases the band once frame_task_server is
        up, and the robot stands after. Returns the release's first sample, or None"""
        release = self.release()
        r = None if release is None else release[0]
        self.band_holds(r)
        self.no_estop()
        policy_wall = self.walking_node_starts()
        self.band_released(release, policy_wall)
        if r is not None:
            self.release_numbers(r, policy_wall)
            t = self.s["t"]
            rows = np.nonzero((t >= t[r]) & (t <= t[r] + RELEASE_STANDING_S))[0]
            if t[-1] < t[r] + RELEASE_STANDING_S:
                self.check("standing after release", False,
                           f"the recording ends {t[-1] - t[r]:.1f} s (sim) after the release, before "
                           f"{RELEASE_STANDING_S:g} s")
            else:
                self.standing(rows, f"the {RELEASE_STANDING_S:g} s after the release",
                              name="standing after release")
        return r

    def band_holds(self, r):
        """While the band is on, the robot hangs upright from it"""
        s = self.s
        rows = slice(0, len(s["t"]) if r is None else max(1, r))
        tilt = _tilt_deg(s["upz"][rows])
        z = s["pelvis"][rows, 2]
        t = s["t"][rows]
        on = s["band"][rows]
        first = s["commanded_at"]
        self.numbers["band_holds"] = {"sim_t": [float(t[0]), float(t[-1])], "tilt_max_deg": float(tilt.max()),
                                      "z_min": float(z.min()), "z_max": float(z.max()),
                                      "first_lowcmd_t": float(first[0])}
        self.check("band holds", bool(np.all(on == 1.0)) and tilt.max() <= MAX_TILT_DEG,
                   f"the band on from sim t {t[0]:.2f} to {t[-1]:.2f} s"
                   + ("" if r is not None else " (never released)")
                   + f": pelvis tilt max {tilt.max():.1f} deg (limit {MAX_TILT_DEG}), pelvis z "
                   f"{z.min():.3f}-{z.max():.3f} m; the first valid rt/lowcmd came at sim t {first[0]:.2f} s"
                   + ("" if np.all(on == 1.0) else "; the band was not on at every sample before the release"))

    def no_estop(self):
        """golem_safety (launch.log) latched no estop from its start to the end of the
        test's steps, the controller's stop included"""
        lines = logged(os.path.join(self.out, "launch.log"), "safety_node")
        started = [wall for wall, _, text in lines if safety_event(text) == "relay_started"]
        end = self.end_wall()
        estops = [(wall, text) for wall, _, text in lines if wall <= end and (
            safety_event(text) == "estop_command_written" or text.startswith("estop diagnostic"))]
        self.numbers["estops"] = [[wall, text] for wall, text in estops]
        if not started:
            text = "launch.log has no relay_started from golem_safety's safety_node"
        elif estops:
            wall, first = estops[0]
            text = (f"{len(estops)} estop line(s) from safety_node, the first {wall - started[0]:.1f} s (wall) "
                    f"after its start: {first}")
        else:
            text = (f"no estop from safety_node's start to the end of the test's steps "
                    f"({end - started[0]:.0f} s wall)")
        self.check("no estop during start-up", bool(started) and not estops, text)

    def walking_node_starts(self):
        """HOMIE's node (launch.log) started its policy once, and did not fault before the
        end of the test's steps. Returns the start's wall time, or None"""
        lines = logged(os.path.join(self.out, "launch.log"), "locomotion_node")
        end = self.end_wall()
        starts = [wall for wall, _, text in lines if text.startswith(POLICY_STARTED)]
        faults = [text for wall, _, text in lines if wall <= end and text.startswith("fault:")]
        wall = starts[0] if starts else None
        t = None
        if wall is not None:
            t = float(self.s["t"][int(min(np.searchsorted(self.s["wall"], wall), len(self.s["t"]) - 1))])
        self.numbers["policy_started"] = {"wall": wall, "sim_t": t, "starts": len(starts), "faults": faults}
        if not starts:
            text = "launch.log has no 'walking: policy started' from locomotion_node"
        else:
            text = f"the policy started at sim t {t:.2f} s"
            text += "" if len(starts) == 1 else f", {len(starts)} times"
        if faults:
            text += f"; it faulted: {faults[0]}"
        self.check("walking node starts", len(starts) == 1 and not faults, text)
        return wall

    def band_released(self, release, policy_wall):
        """The band comes off once and stays off: after the first /left_ee_pose the test saw
        (its stamp, sim time) and within band.max_wait of the policy's start (wall)"""
        changes = self.s["band_changes"]
        states = changes[:, 2].tolist()
        pose = self.events.get("left_ee_pose_first")
        max_wait = self.info.get("band_max_wait")
        reasons = [text for _, _, text in logged(os.path.join(self.out, "launch.log"), "locomotion_node")
                   if text.startswith("releasing the elastic band") or text.startswith("elastic band released")]
        self.numbers["band_release"] = {"changes_t_wall_on": changes.tolist(), "left_ee_pose_first": pose,
                                        "max_wait": max_wait, "walking_node": reasons}
        if release is None:
            self.check("band released after the controller is ready", False,
                       "the band was never released" + ("" if pose else ", and the test saw no /left_ee_pose"))
            return
        _, t_off, wall_off = release
        once = states == [1.0, 0.0]
        after_pose = pose is not None and t_off > pose["stamp"]
        since = wall_off - policy_wall if policy_wall is not None else math.nan
        in_time = max_wait is not None and 0.0 <= since <= max_wait + BAND_POLL_SLACK_S
        self.numbers["band_release"].update(sim_t=t_off, wall=wall_off, after_policy_wall_s=since,
                                            after_left_ee_pose_sim_s=None if pose is None else t_off - pose["stamp"])
        text = (f"off at sim t {t_off:.3f} s; " + ("on from the first step, then off once" if once else
                                                   f"its states over the run {states} (1 on, 0 off)")
                + "; " + (f"{t_off - pose['stamp']:.3f} s (sim) after the first /left_ee_pose's stamp "
                          f"{pose['stamp']:.3f} s" if pose else "the test saw no /left_ee_pose")
                + f"; {since:.2f} s (wall) after the policy's start (band.max_wait {max_wait} s, "
                f"+{BAND_POLL_SLACK_S} s for the node's poll)")
        if reasons:
            text += f"; locomotion_node: {' / '.join(reasons)}"
        self.check("band released after the controller is ready", once and after_pose and in_time, text)

    def release_numbers(self, r, policy_wall):
        """The pelvis and the feet around the release, for the record: under the band in the
        second before it, and over the RELEASE_MEASURE_S after it"""
        s = self.s
        t, z, tilt, feet = s["t"], s["pelvis"][:, 2], _tilt_deg(s["upz"]), s["feet"]
        before = np.nonzero((t >= t[r] - 1.0) & (t < t[r]))[0]
        after = np.nonzero((t >= t[r]) & (t <= t[r] + RELEASE_MEASURE_S))[0]
        if not before.size or not after.size:
            return
        last = before[-1]
        moved = np.linalg.norm(feet[after][:, :, :2] - feet[last][:, :2], axis=2)   # (samples, 2 feet)
        under = s["wall"][last] - policy_wall if policy_wall is not None else math.nan
        n = {"policy_under_band_wall_s": under,
             "before": {"pelvis_z": float(z[last]), "pelvis_z_range": [float(z[before].min()), float(z[before].max())],
                        "tilt_deg": float(tilt[last]), "tilt_max_deg": float(tilt[before].max())},
             "after": {"seconds": RELEASE_MEASURE_S, "tilt_max_deg": float(tilt[after].max()),
                       "pelvis_z_min": float(z[after].min()), "pelvis_drop_m": float(z[last] - z[after].min()),
                       "pelvis_z_end": float(z[after[-1]]),
                       "feet_moved_max_m": moved.max(axis=0).tolist(), "feet_moved_end_m": moved[-1].tolist()}}
        self.numbers["release"] = n
        b, a = n["before"], n["after"]
        print(f"     band release at sim t {t[r]:.2f} s, {under:.2f} s (wall) after the policy's start. Under "
              f"the band, the second before: pelvis z {b['pelvis_z_range'][0]:.3f}-{b['pelvis_z_range'][1]:.3f} "
              f"m, tilt max {b['tilt_max_deg']:.1f} deg (last sample {b['pelvis_z']:.3f} m, "
              f"{b['tilt_deg']:.1f} deg). The {RELEASE_MEASURE_S:g} s after: tilt max {a['tilt_max_deg']:.1f} "
              f"deg, pelvis down {a['pelvis_drop_m'] * 100:.1f} cm to {a['pelvis_z_min']:.3f} m (at the end "
              f"{a['pelvis_z_end']:.3f} m); feet moved left {a['feet_moved_max_m'][0] * 100:.1f} cm, right "
              f"{a['feet_moved_max_m'][1] * 100:.1f} cm at most ({a['feet_moved_end_m'][0] * 100:.1f} / "
              f"{a['feet_moved_end_m'][1] * 100:.1f} cm at the end)")

    # ------------------------------------------------------------- the tests
    def arm_test(self):
        s = self.s
        samples = len(s["t"])
        r = self.startup()
        # frame_task_server took the arms to home at startup, and commands it
        rows = self.window("controller_up", "t_pose_sent", start_offset=1.0)
        err, worst, cmd_err, waist = self.arm_error("home", "home", rows)
        span = s["t"][rows[-1]] - s["t"][rows[0]]
        self.check("home at startup", err.max() <= ARM_TOL and cmd_err <= HOME_CMD_TOL,
                   f"arms (14) vs home over the {span:.1f} s (sim) before the t_pose goal: max "
                   f"{err.max():.3f} rad ({worst}), mean {err.mean():.3f}; commanded max {cmd_err:.4f} "
                   f"(limit {HOME_CMD_TOL}); waist yaw/roll/pitch "
                   f"{waist[0]:+.3f} {waist[1]:+.3f} {waist[2]:+.3f}")
        # NamedConfig t_pose succeeds and the arms reach it
        result = self.events.get("t_pose_result", {})
        rows = self.window("t_pose_result", "/left/hand/close_sent", start_offset=0.5)
        err, worst, cmd_err, waist = self.arm_error("t_pose", "t_pose", rows)
        took = result["wall"] - self.events["t_pose_sent"]["wall"]
        self.check("t_pose", bool(result.get("success")) and err.max() <= ARM_TOL,
                   f"goal success {result.get('success')} after {took:.1f} s wall; arms vs t_pose: max "
                   f"{err.max():.3f} rad ({worst}), mean {err.mean():.3f}; commanded max {cmd_err:.4f}; "
                   f"waist {waist[0]:+.3f} {waist[1]:+.3f} {waist[2]:+.3f}")
        # /left/hand/close closes the left hand: the last 0.5 s of sim time
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
        # stopping frame_task_server leaves the waist at its hold, the walking node standing the robot
        self.waist_after_controller_stop()
        stop = self.at("controller_stopped")
        released = samples if r is None else r
        self.phases([("under the band", 0, released),
                     ("on HOMIE, home", released, self.at("t_pose_sent")),
                     ("to t_pose", self.at("t_pose_sent"), self.at("t_pose_result")),
                     ("t_pose and hand", self.at("t_pose_result"), stop),
                     ("controller stopped", stop, self.at("end") + 1)])
        if r is not None:
            self.standing(slice(r, self.at("end") + 1), "from the release to the end")
        self.sensors()

    def walk_test(self):
        s = self.s
        n = len(s["t"])
        r = self.startup()
        if r is None:
            return
        if "leg_changes" in s:
            # diagnostic: HOMIE needs a fresh leg target every 10 physics steps; a
            # starved ros side delivers fewer, late and in bursts
            self.numbers["leg_targets"] = {}
            for label, e in (("in the first 2 s after the release", min(n - 1, r + int(2 * self.hz))),
                             ("from the release to the end", self.at("end"))):
                got = int(s["leg_changes"][e] - s["leg_changes"][r])
                expected = int(round((s["t"][e] - s["t"][r]) / 0.02))
                hold = float(s["leg_hold_max"][r + 1:e + 1].max() + 1) * 2.0
                self.numbers["leg_targets"][label] = {"received": got, "expected": expected,
                                                      "longest_hold_ms": hold}
                print(f"     leg targets {label}: {got} received of {expected} expected (one per 20 ms "
                      f"of sim time); the longest one held {hold:.0f} ms of sim time")
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
        self.standing(slice(r, self.at("end") + 1), "from the release to the end")
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
        first = next((e for e in ("controller_up", "band_released") if e in self.events), None)
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
            release = self.release()
            if release is not None:
                print(f"     the band was released at sim t {release[1]:.2f} s")
                self.standing(slice(release[0], len(t)), "from the release to the end of the recording")
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
