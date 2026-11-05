#!/usr/bin/env python3
"""Run an MJPC controller core from the mujoco_mpc fork as a child process, configured by ROS parameters.

The core (`mjpc_lowerbody_core` by default, or `mjpc_fullbody_core`) is the
fork's own thin main and deploy_common.cc, compiled from the submodule. This
node maps its ROS parameters to the core's flags, releases the sim's elastic
band after a delay, and forwards SIGINT and SIGTERM to the core, whose handler
publishes its damping safe-hold before it exits. No golem_bringup launch file
starts it; golem_bringup's config/h12/mjpc_sim.yaml and mjpc_real.yaml hold the
sim and real parameter sets.

SAFETY: the core must not share a process with rclcpp. rclcpp's
rmw_cyclonedds and unitree_sdk2's bundled CycloneDDS both export libddsc.so.0
and libddscxx.so.0, and one process resolving both to mismatched builds
corrupts the heap (`free(): invalid pointer` right after ChannelFactory init).
"""

import os
import signal
import subprocess
import sys
import threading

import rclpy
from ament_index_python.packages import get_package_prefix
from rclpy.node import Node
from rclpy.duration import Duration
from std_srvs.srv import Trigger


def _band_worker(node: Node, delay_sec: float, stop: threading.Event) -> None:
    """Call /elastic_band/toggle once, delay_sec after the node clock's first reading, unless stopped first.

    With use_sim_time the node clock follows /clock, so the delay is in sim
    seconds, the clock the robot lives on; the RoboCasa sim runs well below
    real time, so a wall-clock delay would fire at the wrong point of the
    start-up. Without use_sim_time the delay is in wall seconds. This
    function's spin_once() also serves the /clock subscription: nothing else
    spins the node.
    """
    t0 = None
    while not stop.is_set() and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        now = node.get_clock().now()
        if t0 is None:
            # With use_sim_time, now() is 0 until the first /clock arrives.
            if now.nanoseconds > 0:
                t0 = now
            continue
        if (now - t0) >= Duration(seconds=delay_sec):
            break
    if stop.is_set() or not rclpy.ok():
        return  # the controller exited first
    cli = node.create_client(Trigger, "/elastic_band/toggle")
    if not cli.wait_for_service(timeout_sec=2.0):
        node.get_logger().warning(
            "/elastic_band/toggle not available -- band NOT released "
            "(fine on the real robot; set drop_band:=false to silence)")
        return
    fut = cli.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    if fut.done() and fut.result() is not None:
        node.get_logger().info(f"elastic band released: {fut.result().message}")
    else:
        node.get_logger().warning("elastic band toggle call did not complete")


def main() -> None:
    """Start the core with the node's parameters, release the band if asked, and exit with the core's code."""
    rclpy.init()
    node = Node("mjpc_deploy_lowerbody_controller")

    _libdir = os.path.join(get_package_prefix("deploy_mjpc"),
                           "lib", "deploy_mjpc")

    p = node.declare_parameter
    # controller_exe picks the installed core: mjpc_lowerbody_core (legs only,
    # Stabilize task) or mjpc_fullbody_core (whole body, Lean task).
    # controller_binary, an absolute path, overrides it.
    exe = p("controller_exe", "mjpc_lowerbody_core").value
    binary = p("controller_binary", os.path.join(_libdir, exe)).value
    # Each parameter is the core's flag of the same name, with the same meaning
    # (mujoco_mpc/mjpc/deploy/h12_lower_body_controller.cc).
    args = [
        binary,
        "--task", str(p("task", "Stabilize H12 Magpie").value),
        "--strategy", str(p("strategy", 6).value),
        "--gravity_ff", str(p("gravity_ff", 0.85).value),
        # s; must equal the plant's /lowstate period (RoboCasa 0.002, real 0.001).
        "--twin_dt", str(p("twin_dt", 0.005).value),
        "--sportstate_topic", str(p("sportstate_topic", "rt/sportmodestate_est").value),
        "--imu_pitch_offset_deg", str(p("imu_pitch_offset_deg", 0.0).value),
        "--imu_roll_offset_deg", str(p("imu_roll_offset_deg", 0.0).value),
        "--ankle_roll_offset_l_deg", str(p("ankle_roll_offset_l_deg", 0.0).value),
        "--ankle_roll_offset_r_deg", str(p("ankle_roll_offset_r_deg", 0.0).value),
        "--domain_id", str(p("domain_id",
                             int(os.environ.get("ROS_DOMAIN_ID", "0"))).value),
        # Inert here: deploy_mjpc builds the cores without H12_NODE_GRPC, which
        # compiles out the monitor server this port is for.
        "--grpc_port", str(p("grpc_port", 0).value),
        "--plan_trajectories", str(p("plan_trajectories", 0).value),
        "--plan_threads", str(p("plan_threads", 0).value),
        "--stale_sec", str(p("stale_sec", 0.05).value),
        # The plant's real-time factor for latency compensation: 1.0 applies no
        # scaling (the real robot); a slower-than-real-time sim (RoboCasa) passes
        # its measured factor so the predict-forward horizon does not lead by 1/RTF.
        "--latency_rtf", str(p("latency_rtf", 1.0).value),
        # The plan for mjpc_debug_visualizer's ghost (/mjpc/plan in ROS), at 20 Hz.
        # Fixed, not a parameter: the visualizer reads the same topic by default.
        # The core publishes it from the planner thread, off the 200 Hz control
        # loop; run bare, the core publishes no plan.
        "--plan_topic", "rt/mjpc/plan",
        "--plan_hz", "20.0",
        # straighten_start, frc_parity and cost: both cores define all three, and
        # the defaults here are the cores' own.
        # straighten_start: hold the measured pose, wait for ENTER on the core's
        # stdin, then hand control to the planner. Pairs with strategy:=25 and
        # needs a terminal, so do not launch it detached.
        ("--straighten_start" if bool(p("straighten_start", False).value)
         else "--nostraighten_start"),
        # frc_parity is an int, not a bool: -1 takes the task's
        # deploy_frc_parity numeric (absent from Stabilize_H12_Magpie.xml, so
        # off); 0 forces it off; 1 tightens the planner's force ranges to
        # 0.9 x the safety layer's torque e-stop, the torque the core can emit.
        "--frc_parity", str(int(p("frc_parity", -1).value)),
        # cost: print a per-term cost breakdown to stderr once a second.
        ("--cost" if bool(p("cost", False).value) else "--nocost"),
    ]
    iface = str(p("network_interface", "").value)
    if iface:
        args += ["--network_interface", iface]
    # --arm_aware (the legs pre-compensate arm motion) exists only in the
    # lower-body core; the full-body core aborts on it as an unknown flag, so
    # has_arm_aware:=false leaves it out.
    if bool(p("has_arm_aware", True).value):
        args.append("--arm_aware" if bool(p("arm_aware", True).value)
                    else "--noarm_aware")

    drop_band = bool(p("drop_band", True).value)
    band_after = float(p("band_drop_after_secs", 20.0).value)

    if not os.path.isfile(binary):
        node.get_logger().fatal(
            f"controller binary not found at {binary} -- was the package built "
            "in the ros container (mjpc build tree + unitree_sdk2 hydrated)?")
        rclpy.shutdown()
        sys.exit(1)

    # The sourced ROS environment puts /opt/ros/humble/lib on LD_LIBRARY_PATH,
    # which wins over the binary's rpath: libddsc.so.0 would resolve to ROS's
    # CycloneDDS while libddscxx.so.0 stays unitree_sdk2's, and the mixed builds
    # corrupt the heap. Prepending the SDK's own lib dir makes both resolve there.
    env = os.environ.copy()
    for cand in ("/opt/unitree_install/lib",
                 os.path.expanduser("~/unitree_install/lib")):
        if os.path.isfile(os.path.join(cand, "libddsc.so.0")):
            env["LD_LIBRARY_PATH"] = cand + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            break

    node.get_logger().info("launching " + " ".join(args))
    child = subprocess.Popen(args, env=env)  # stdout and stderr go to the launch log

    # SAFETY: ros2 launch signals this process; the core must get the signal so
    # it publishes its damping safe-hold before exiting.
    def _forward(signum, _frame):
        """Pass the signal on to the core while it runs."""
        if child.poll() is None:
            child.send_signal(signum)
    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    stop = threading.Event()
    band_thread = None
    if drop_band:
        band_thread = threading.Thread(
            target=_band_worker, args=(node, band_after, stop), daemon=True)
        band_thread.start()

    rc = child.wait()
    stop.set()
    if band_thread is not None:
        band_thread.join(timeout=1.0)
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(rc if rc is not None and rc >= 0 else 1)


if __name__ == "__main__":
    main()
