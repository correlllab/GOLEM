#!/usr/bin/env python3
"""Records and checks the externally visible ROS/DDS surface of the RoboCasa sim.

Runs inside the robocasa container, next to the simulator it launches:

    probe.py --robot h12|g1 --out observed.json [--golden golden.json] [--record]
             [--window 10] [--sim-log sim.log] -- <launcher command ...>

It starts the launcher as a child process, waits for its "ROS bridges up" line,
measures the surface over a window of simulated time, stops the sim with
SIGINT (the loop runs until interrupted, and its `finally` block tears the
bridges down), and writes the JSON. With --golden it compares the observation
against the golden file.

Exit status: 0 success or match, 1 mismatch against the golden, 2 the probe
could not measure (sim failed to start, domain shared with another sim, a
required measurement missing, ...). A sim that crashes or hangs on SIGINT after
the surface was measured only prints a warning; the comparison decides.

Rates are per second of simulated time, taken from message header stamps where
a message has them and from the latest /clock sample otherwise, so they do not
depend on the machine's real-time factor. WALL_TIMER_TOPICS are the exception:
the sim publishes them from a wall-clock ROS timer, as the real driver does,
so their rate is wall-clock Hz.

Only the comparison code is stdlib-only; observing needs ROS, the message
workspaces and unitree_sdk2py, i.e. the robocasa image.
"""
import argparse
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback

# Relative tolerance for the keys in TOLERANT_KEYS; every other value is exact.
RTOL = 0.05
TOLERANT_KEYS = ("rate_hz", "livox_imu_mean_acc")

PROBE_NODE = "surface_probe"
BARE_DDS_NODE = "_CREATED_BY_BARE_DDS_APP_"   # how rmw names a non-ROS DDS participant
READY_LINE = re.compile(r"ROS bridges up")
IMU_TOPIC = "/livox/imu"

# Published from a wall-clock timer, so their sim-time rate scales with 1/RTF.
WALL_TIMER_TOPICS = re.compile(r"^/(left|right)/gripper/state$")

# Present on every rclpy node; not part of what a sim chooses to publish.
INFRA_TOPICS = ("/rosout", "/parameter_events")
PARAMETER_SERVICES = ("describe_parameters", "get_parameter_types", "get_parameters",
                      "list_parameters", "set_parameters", "set_parameters_atomically")

# Subscription queue depth; /clock and the IMU are far faster than the rest.
QUEUE_DEPTH = {"/clock": 5000, IMU_TOPIC: 2000}
DEFAULT_QUEUE_DEPTH = 200

CLEAN_SIM_EXIT = (0, 130, -signal.SIGINT)   # 130: a shell's view of death by SIGINT
NO_FACTS = dict(frame_id=None, encoding=None, width=None, height=None, point_fields=None)


MISSING = object()   # stands for a key one of the two surfaces lacks


class ProbeError(Exception):
    """The surface could not be measured (as opposed to differing from the golden)."""


# --------------------------------------------------------------------------- comparison

class Comparison:
    """Differences between a golden and an observation, and the worst tolerant error."""

    def __init__(self, rtol=RTOL):
        self.rtol = rtol
        self.diffs = []
        self.worst = (0.0, "")   # relative error and path of the worst tolerant value

    def walk(self, golden, observed, path="", key=""):
        if isinstance(golden, dict) and isinstance(observed, dict):
            for name in sorted(set(golden) | set(observed)):
                child = _join(path, name)
                if name not in observed:
                    self._add(child, golden[name], MISSING)
                elif name not in golden:
                    self._add(child, MISSING, observed[name])
                else:
                    self.walk(golden[name], observed[name], child, name)
        elif isinstance(golden, list) and isinstance(observed, list):
            # Order is not part of the surface.
            g, o = _canonical(golden), _canonical(observed)
            if g != o:
                self.diffs.append(f"{path}: only in golden {[x for x in g if x not in o]}, "
                                  f"only in observed {[x for x in o if x not in g]}")
        elif key in TOLERANT_KEYS and _is_number(golden) and _is_number(observed):
            error = abs(golden - observed) / max(abs(golden), abs(observed), 1e-12)
            if error > self.worst[0]:
                self.worst = (error, path)
            if not math.isclose(golden, observed, rel_tol=self.rtol):
                self._add(path, golden, observed)
        elif golden != observed or type(golden) is not type(observed):
            self._add(path, golden, observed)

    def _add(self, path, golden, observed):
        self.diffs.append(f"{path}: golden {_brief(golden)}, observed {_brief(observed)}")


def _brief(value):
    """repr for a value; a whole entry that is present on one side only shows just its type."""
    if value is MISSING:
        return "<missing>"
    if isinstance(value, dict):
        return f"<entry {value['type']}>" if "type" in value else f"<{len(value)} entries>"
    return repr(value)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _join(path, name):
    if not path:
        return name
    return f'{path}["{name}"]' if name.startswith("/") else f"{path}.{name}"


def _canonical(items):
    return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))


def compare(golden, observed, rtol=RTOL):
    """The Comparison of two surfaces; "meta" is not part of the surface."""
    result = Comparison(rtol)
    result.walk({k: v for k, v in golden.items() if k != "meta"},
                {k: v for k, v in observed.items() if k != "meta"})
    return result


# --------------------------------------------------------------------------- the sim process

class Sim:
    """The launcher as a child process, in its own process group."""

    def __init__(self, command, log_path):
        self.log = open(log_path, "w", buffering=1)
        self.ready = threading.Event()
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True)
        self._pump_thread = threading.Thread(target=self._pump, daemon=True, name="sim_log")
        self._pump_thread.start()

    def _pump(self):
        for line in self.proc.stdout:
            self.log.write(line)
            if READY_LINE.search(line):
                self.ready.set()

    def alive(self):
        return self.proc.poll() is None

    def wait_ready(self, timeout):
        deadline = time.monotonic() + timeout
        while not self.ready.wait(1.0):
            if not self.alive():
                raise ProbeError(f"the sim exited with status {self.proc.returncode} "
                                 "before it came up")
            if time.monotonic() > deadline:
                raise ProbeError(f"the sim was not up after {timeout:.0f} s")

    def stop(self, timeout=60.0):
        """Interrupts the sim and waits for it; returns its exit status (None if it hung)."""
        if self.alive():
            os.killpg(self.proc.pid, signal.SIGINT)
        try:
            status = self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
            status = None
        self._pump_thread.join(5.0)
        self.log.close()
        return status


# --------------------------------------------------------------------------- graph snapshot

def _fqn(namespace, name):
    return namespace.rstrip("/") + "/" + name


def _type_string(types):
    return ", ".join(sorted(types))


def _is_action_internal(name):
    return "/_action/" in name


def _shown_topic(name):
    return name not in INFRA_TOPICS and not _is_action_internal(name)


def _shown_service(node_fqn, name):
    prefix = node_fqn + "/"
    return not (_is_action_internal(name)
                or (name.startswith(prefix) and name[len(prefix):] in PARAMETER_SERVICES))


def snapshot_nodes(node):
    """Every sim node's publishers, subscriptions, services and actions, with types."""
    import rclpy.action
    nodes = {}
    for name, namespace in node.get_node_names_and_namespaces():
        if name == PROBE_NODE:
            continue
        fqn = _fqn(namespace, name)
        publishers = node.get_publisher_names_and_types_by_node(name, namespace)
        subscriptions = node.get_subscriber_names_and_types_by_node(name, namespace)
        services = node.get_service_names_and_types_by_node(name, namespace)
        actions = rclpy.action.get_action_server_names_and_types_by_node(node, name, namespace)
        nodes[fqn] = {
            "publishers": {t: _type_string(ty) for t, ty in publishers if _shown_topic(t)},
            "subscriptions": {t: _type_string(ty) for t, ty in subscriptions if _shown_topic(t)},
            "services": {s: _type_string(ty) for s, ty in services if _shown_service(fqn, s)},
            "actions": {a: _type_string(ty) for a, ty in actions},
        }
    return nodes


def snapshot_topics(node):
    """Returns ({topic: (type, publisher qos)} for ROS publishers, the bare DDS surface).

    The Unitree SDK's rt/lowstate and rt/lowcmd belong to a bare DDS participant
    with no ROS node, so they are listed apart from the nodes' topics.
    """
    topics, bare = {}, {"publishes": {}, "subscribes": {}}
    for name, types in node.get_topic_names_and_types():
        if not _shown_topic(name):
            continue
        type_name = _type_string(types)
        publishers = node.get_publishers_info_by_topic(name)
        subscribers = node.get_subscriptions_info_by_topic(name)
        if any(s.node_name == BARE_DDS_NODE for s in subscribers):
            bare["subscribes"][name] = type_name
        ros_publishers = [p for p in publishers if p.node_name != BARE_DDS_NODE]
        if len(ros_publishers) < len(publishers):
            bare["publishes"][name] = type_name
        if ros_publishers:
            topics[name] = (type_name, ros_publishers[0].qos_profile)
    return topics, bare


def require_empty_domain(node):
    time.sleep(3.0)   # discovery of anything already on the domain
    others = [n for n, _ in node.get_node_names_and_namespaces() if n != PROBE_NODE]
    topics = [t for t, _ in node.get_topic_names_and_types() if t not in INFRA_TOPICS]
    if others or topics:
        raise ProbeError(f"DDS domain {os.environ.get('ROS_DOMAIN_ID')} is not empty "
                         f"(nodes {others}, topics {topics}); one robot stack per domain")


def require_single_sim(node):
    """Fails if a second sim has come up on the domain: a node name or /clock twice."""
    names = [_fqn(namespace, name) for name, namespace in node.get_node_names_and_namespaces()
             if name != PROBE_NODE]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    clock_publishers = len(node.get_publishers_info_by_topic("/clock"))
    if duplicates or clock_publishers != 1:
        raise ProbeError(f"more than one sim is on DDS domain {os.environ.get('ROS_DOMAIN_ID')}: "
                         f"duplicate nodes {duplicates}, {clock_publishers} publishers on /clock")


# --------------------------------------------------------------------------- message statistics

def message_facts(msg):
    """frame_id, encoding, size and point layout of one message (None where n/a)."""
    facts = dict(NO_FACTS)
    header = getattr(msg, "header", None)
    if header is not None:
        facts["frame_id"] = header.frame_id
    if hasattr(msg, "fields") and hasattr(msg, "point_step"):      # PointCloud2
        facts["height"] = int(msg.height)
        # An unorganized cloud's width is its point count, which depends on the scene.
        if msg.height > 1:
            facts["width"] = int(msg.width)
        facts["point_fields"] = [[f.name, int(f.offset), int(f.datatype), int(f.count)]
                                 for f in msg.fields]
    elif hasattr(msg, "width") and hasattr(msg, "height"):         # Image, CameraInfo
        facts["width"], facts["height"] = int(msg.width), int(msg.height)
    if hasattr(msg, "encoding"):                                    # Image
        facts["encoding"] = msg.encoding
    elif hasattr(msg, "format"):                                    # CompressedImage
        facts["encoding"] = msg.format
    return facts


def message_stamp(msg):
    """Sim time (s) a message carries: Clock, a header stamp or a TF's first stamp."""
    if hasattr(msg, "clock"):
        stamp = msg.clock
    elif hasattr(msg, "header"):
        stamp = msg.header.stamp
    elif getattr(msg, "transforms", None):
        stamp = msg.transforms[0].header.stamp
    else:
        return None
    return stamp.sec + stamp.nanosec * 1e-9


class SimClock:
    """The latest /clock sample (s), written by the /clock subscription."""

    latest = None


class Collector:
    """What one subscription saw: when each message came, and the first one's facts."""

    def __init__(self, topic, clock):
        self.topic = topic
        self.clock = clock
        self.samples = []      # (wall time, latest /clock sample, message stamp)
        self.accel_norms = []  # |linear_acceleration| per sample, None for other types
        self.facts = None
        self.edges = set()     # "parent -> child" over a TFMessage's transforms

    def __call__(self, msg):
        stamp = message_stamp(msg)
        self.samples.append((time.monotonic(), self.clock.latest, stamp))
        if self.topic == "/clock":
            self.clock.latest = stamp
        if self.facts is None:
            self.facts = message_facts(msg)
        accel = getattr(msg, "linear_acceleration", None)
        self.accel_norms.append(
            None if accel is None else math.sqrt(accel.x ** 2 + accel.y ** 2 + accel.z ** 2))
        for t in getattr(msg, "transforms", ()):
            self.edges.add(f"{t.header.frame_id} -> {t.child_frame_id}")


def in_window(sample, window):
    _, clock, stamp = sample
    t = stamp if stamp is not None else clock
    return t is not None and window[0] <= t <= window[1]


def rate_hz(topic, samples, window):
    """Messages per second in the window, or None with fewer than two of them."""
    samples = [s for s in samples if in_window(s, window)]
    if len(samples) < 2:
        return None
    if WALL_TIMER_TOPICS.match(topic):
        span = samples[-1][0] - samples[0][0]
        rate = (len(samples) - 1) / span if span > 0 else None
    elif all(s[2] is not None for s in samples):
        stamps = sorted(s[2] for s in samples)
        deltas = [b - a for a, b in zip(stamps, stamps[1:])]
        # A best-effort drop is a gap of several periods; it is not the publish rate.
        typical = statistics.median(deltas)
        kept = [d for d in deltas if d <= 1.5 * typical]
        rate = len(kept) / sum(kept) if sum(kept) > 0 else None
    else:
        span = samples[-1][1] - samples[0][1]
        rate = (len(samples) - 1) / span if span > 0 else None
    return None if rate is None else round(rate, 3)


class LowStateReader:
    """rt/lowstate through unitree_sdk2py, the way the sim's own consumers read it."""

    def __init__(self, domain):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        ChannelFactoryInitialize(domain)
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        # Polled, not handler-driven: closing a handler-driven subscriber while
        # samples arrive makes the SDK's listener fail on the freed reader.
        self._subscriber.Init()

    def tick(self, timeout=5.0):
        """The tick of the next sample, or None if none arrives in `timeout` seconds."""
        sample = self._subscriber.Read(timeout)
        return None if sample is None else int(sample.tick)

    def close(self):
        self._subscriber.Close()


# --------------------------------------------------------------------------- observation

def wait_for(predicate, timeout, what):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise ProbeError(f"timed out waiting for {what}")
        time.sleep(0.1)


def observe(args, sim, node):
    """Measures the surface of the sim `sim` has started; `node` is the probe's node."""
    from rclpy.qos import HistoryPolicy, QoSProfile
    from rosidl_runtime_py.utilities import get_message

    domain = int(os.environ["ROS_DOMAIN_ID"])
    started = time.monotonic()
    sim.wait_ready(args.ready_timeout)
    print(f"[probe] sim up after {time.monotonic() - started:.0f} s", flush=True)
    wait_for(lambda: "/clock" in dict(node.get_topic_names_and_types()), 60, "/clock")
    time.sleep(args.settle)   # the rest of the graph is discovered by now
    require_single_sim(node)

    nodes = snapshot_nodes(node)
    topics, bare_dds = snapshot_topics(node)

    clock = SimClock()
    collectors = {}
    subscriptions = []
    for topic in sorted(topics, key=lambda t: t != "/clock"):
        type_name, qos = topics[topic]
        collectors[topic] = Collector(topic, clock)
        profile = QoSProfile(history=HistoryPolicy.KEEP_LAST,
                             depth=QUEUE_DEPTH.get(topic, DEFAULT_QUEUE_DEPTH),
                             reliability=qos.reliability, durability=qos.durability)
        subscriptions.append(node.create_subscription(
            get_message(type_name), topic, collectors[topic], profile))
        if topic == "/clock":
            wait_for(lambda: clock.latest is not None, 60, "the first /clock sample")

    lowstate = LowStateReader(domain)
    start = clock.latest + args.warmup
    window = (start, start + args.window)
    print(f"[probe] measuring sim time {window[0]:.1f}..{window[1]:.1f} s", flush=True)
    wall_limit = 60 + 20 * (args.warmup + args.window)
    wait_for(lambda: clock.latest >= window[0] or not sim.alive(), wall_limit, "the window to start")
    first_tick = lowstate.tick()
    wait_for(lambda: clock.latest >= window[1] or not sim.alive(), wall_limit, "the window to end")
    if not sim.alive():
        raise ProbeError("the sim exited during the measurement window")
    last_tick = lowstate.tick()
    lowstate.close()
    require_single_sim(node)
    ticks_advance = first_tick is not None and last_tick is not None and last_tick > first_tick

    stats = {}
    for topic, (type_name, qos) in sorted(topics.items()):
        collector = collectors[topic]
        stats[topic] = dict(
            collector.facts or NO_FACTS,
            type=type_name,
            qos={"reliability": qos.reliability.name, "durability": qos.durability.name},
            received=bool(collector.samples),
            rate_clock="wall" if WALL_TIMER_TOPICS.match(topic) else "sim",
            rate_hz=rate_hz(topic, collector.samples, window))
    imu = collectors.get(IMU_TOPIC)
    imu_norms = [] if imu is None else [
        norm for sample, norm in zip(imu.samples, imu.accel_norms) if in_window(sample, window)]
    return {
        "meta": {"robot": args.robot, "launcher": args.launcher, "window_sim_s": args.window},
        "nodes": nodes,
        "bare_dds": bare_dds,
        "topics": stats,
        "tf_static": sorted(collectors["/tf_static"].edges) if "/tf_static" in collectors else [],
        "tf": sorted(collectors["/tf"].edges) if "/tf" in collectors else [],
        "livox_imu_mean_acc": round(statistics.fmean(imu_norms), 4) if imu_norms else None,
        "lowstate_ticks_advance": ticks_advance,
    }


def run_probe(args):
    """Runs the sim and observes it; returns (observation, the sim's exit status)."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    rclpy.init()
    node = rclpy.create_node(PROBE_NODE)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True, name="probe_executor")
    spin_thread.start()
    sim = None
    try:
        require_empty_domain(node)
        sim = Sim(args.launcher, args.sim_log)
        observed = observe(args, sim, node)
    except ProbeError:
        if sim is not None:
            print(f"[probe] last lines of {args.sim_log}:\n{tail(args.sim_log)}", file=sys.stderr)
        raise
    finally:
        # Stop the sim only once the probe's subscriptions are gone, so its
        # teardown has no remote subscribers to encode and publish for.
        executor.shutdown()
        spin_thread.join(5.0)
        node.destroy_node()
        rclpy.shutdown()
        time.sleep(2.0)
        shutdown = sim.stop() if sim is not None else 0
    return observed, shutdown


def report_sim_exit(args, shutdown):
    """Records how the sim stopped. The surface is already measured, so a sim that
    crashes or hangs on SIGINT is a warning, not a failed measurement."""
    clean = shutdown in CLEAN_SIM_EXIT
    status = "did not exit and was killed" if shutdown is None else shutdown
    if args.sim_exit:
        with open(args.sim_exit, "w") as f:
            f.write(f"{'clean' if clean else 'unclean'} {'hung' if shutdown is None else shutdown}\n")
    if not clean:
        print(f"[probe] WARNING: the sim did not stop cleanly on SIGINT (status {status}); "
              f"the surface was measured before that. Kept sim log: {args.sim_log}\n"
              f"{tail(args.sim_log, 20)}", file=sys.stderr, flush=True)


def check_fit_to_record(observed):
    """What a golden must at least contain, so a broken run is never recorded."""
    problems = []
    if not observed["nodes"]:
        problems.append("no sim nodes were found")
    if "/clock" not in observed["topics"]:
        problems.append("/clock is not published")
    if not observed["lowstate_ticks_advance"]:
        problems.append("rt/lowstate ticks did not advance")
    if observed["livox_imu_mean_acc"] is None:
        problems.append(f"no {IMU_TOPIC} samples in the window")
    return problems


def tail(path, lines=40):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-lines:])
    except OSError:
        return ""


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot", required=True, choices=("h12", "g1"))
    parser.add_argument("--out", required=True, help="where to write the observed JSON")
    parser.add_argument("--golden", help="compare against this golden JSON")
    parser.add_argument("--record", action="store_true",
                        help="refuse an observation that is not fit to be a golden")
    parser.add_argument("--window", type=float, default=10.0, help="sim seconds measured")
    parser.add_argument("--warmup", type=float, default=1.0,
                        help="sim seconds between subscribing and the window")
    parser.add_argument("--settle", type=float, default=3.0,
                        help="wall seconds for graph discovery once the sim is up")
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--sim-log", default="sim.log")
    parser.add_argument("--sim-exit", help="write how the sim stopped (clean|unclean <status>) here")
    parser.add_argument("launcher", nargs=argparse.REMAINDER,
                        help="-- followed by the launcher command and its arguments")
    args = parser.parse_args(argv)
    if args.launcher[:1] == ["--"]:
        args.launcher = args.launcher[1:]
    if not args.launcher:
        parser.error("give the launcher command after --")

    try:
        observed, shutdown = run_probe(args)
    except ProbeError as e:
        print(f"[probe] ERROR: {e}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 2
    with open(args.out, "w") as f:
        json.dump(observed, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[probe] observed surface written to {args.out}", flush=True)
    report_sim_exit(args, shutdown)
    if args.record:
        problems = check_fit_to_record(observed)
        if problems:
            print("[probe] ERROR: not fit to record as a golden: " + "; ".join(problems),
                  file=sys.stderr)
            return 2
    if args.golden:
        with open(args.golden) as f:
            result = compare(json.load(f), observed)
        name = os.path.basename(args.golden)
        if result.diffs:
            print(f"SURFACE MISMATCH against {name}: {len(result.diffs)} difference(s)")
            for line in result.diffs:
                print(f"  {line}")
            return 1
        print(f"SURFACE MATCH against {name}: {len(observed['topics'])} topics, "
              f"{len(observed['nodes'])} nodes, worst rate/IMU deviation "
              f"{result.worst[0] * 100:.2f} % ({result.worst[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
