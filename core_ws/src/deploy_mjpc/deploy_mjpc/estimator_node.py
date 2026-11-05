#!/usr/bin/env python3
"""Run the mujoco_mpc fork's proprioceptive base estimator, configured by ROS parameters.

The estimator is mjpc/deploy/helper_scripts/base_estimator_node.py in the
mujoco_mpc submodule: an RW-EKF (or complementary) leg-odometry and IMU-fusion
filter over rt/lowstate that publishes SportModeState_ on a side topic. This
node maps its ROS parameters to the estimator's arguments, shuts rclpy down,
and runs the script unmodified in this process, so the fork stays the one
source of the estimator and its unitree_sdk2py DDS I/O stays as it is.

The estimator's IMU_OFFSET must equal the controller core's kImuOffset
(deploy_common.h): both ends of the IMU-site -> pelvis reconstruction must
agree or the base pose is wrong.
"""

import os
import runpy
import sys

import rclpy
from rclpy.node import Node

# Defaults for the container's mounts; MJPC_ROOT overrides the fork's location.
_MJPC_ROOT = os.environ.get("MJPC_ROOT", "/home/code/mujoco_mpc")
_DEFAULT_SCRIPT = os.path.join(
    _MJPC_ROOT, "mjpc", "deploy", "helper_scripts", "base_estimator_node.py")
# Any H1-2 model with a free base, the 27 joints in lowstate motor order and
# *ankle_roll* foot bodies serves; this is the staged task XML the controller
# loads, with its assets beside it in the build tree. MJPC_TASKS_DIR overrides
# the tasks directory.
_DEFAULT_SCENE = os.path.join(
    os.environ.get("MJPC_TASKS_DIR",
                   os.path.join(_MJPC_ROOT, "build", "mjpc", "tasks")),
    "humanoid_bench", "stabilize", "Stabilize_H12_Magpie.xml")


def main() -> None:
    """Read the parameters, shut rclpy down, then run the fork's estimator script in this process."""
    rclpy.init()
    node = Node("deploy_mjpc_estimator")

    p = node.declare_parameter
    script = p("estimator_script", _DEFAULT_SCRIPT).value
    scene = p("scene", _DEFAULT_SCENE).value
    domain = p("domain_id", int(os.environ.get("ROS_DOMAIN_ID", "0"))).value
    iface = p("iface", "").value               # "": the estimator picks the robot subnet's interface
    no_auto_iface = p("no_auto_iface", False).value
    rate = p("rate", 200.0).value              # publish Hz
    # s of plant time per lowstate tick; 0 clocks the filter on the wall clock,
    # as on the real robot. A slower-than-real-time sim passes its physics step
    # (the estimator's --tick-dt).
    tick_dt = p("tick_dt", 0.0).value
    lowstate_topic = p("lowstate_topic", "rt/lowstate").value
    # A side topic, so the estimator is never a second writer of
    # rt/sportmodestate; the controller's sportstate_topic points here.
    out_topic = p("out_topic", "rt/sportmodestate_est").value
    vel_lpf_ms = p("vel_lpf_ms", 30.0).value
    imu_fusion = p("imu_fusion", True).value
    imu_tau_ms = p("imu_tau_ms", 300.0).value
    static_dq = p("static_dq", 0.3).value
    contact = p("contact", "min-speed").value  # min-speed | mean
    filt = p("filter", "rw-ekf").value         # rw-ekf | complementary
    ekf_amax = p("ekf_amax", 3.0).value
    ekf_r0 = p("ekf_r0", 0.02).value
    ekf_r1 = p("ekf_r1", 0.3).value
    ekf_lfloor = p("ekf_lfloor", 5.0).value
    ekf_chi2 = p("ekf_chi2", 11.34).value
    ekf_rejcap = p("ekf_rejcap", 50).value
    compare = p("compare", False).value
    truth_topic = p("truth_topic", "rt/sportmodestate").value
    selftest = p("selftest", False).value

    node.get_logger().info(
        f"running fork estimator {script} (filter={filt}, out={out_topic}, "
        f"domain={domain})")
    # The estimator is a DDS-only worker from here on.
    node.destroy_node()
    rclpy.shutdown()

    if not os.path.isfile(script):
        sys.exit(f"[estimator_node] fork estimator not found at {script} — "
                 "is the mujoco_mpc submodule mounted/checked out?")

    argv = [
        script,
        "--scene", str(scene),
        "--domain", str(domain),
        "--rate", str(rate),
        "--tick-dt", str(tick_dt),
        "--lowstate-topic", str(lowstate_topic),
        "--out-topic", str(out_topic),
        "--vel-lpf-ms", str(vel_lpf_ms),
        "--imu-tau-ms", str(imu_tau_ms),
        "--static-dq", str(static_dq),
        "--contact", str(contact),
        "--filter", str(filt),
        "--ekf-amax", str(ekf_amax),
        "--ekf-r0", str(ekf_r0),
        "--ekf-r1", str(ekf_r1),
        "--ekf-lfloor", str(ekf_lfloor),
        "--ekf-chi2", str(ekf_chi2),
        "--ekf-rejcap", str(ekf_rejcap),
        "--truth-topic", str(truth_topic),
    ]
    if iface:
        argv += ["--iface", str(iface)]
    if no_auto_iface:
        argv.append("--no-auto-iface")
    if not imu_fusion:
        argv.append("--no-imu-fusion")
    if compare:
        argv.append("--compare")
    if selftest:
        argv.append("--selftest")

    sys.argv = argv
    # run_name="__main__" runs the script's own entry point; SIGINT from ros2
    # launch reaches its loop as KeyboardInterrupt, as a terminal Ctrl-C does.
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
