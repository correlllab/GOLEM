#!/usr/bin/env python3
"""Run the mujoco_mpc fork's MJPC plan visualizer, configured by ROS parameters (debug only).

The visualizer is mjpc/deploy/helper_scripts/plan_visualizer.py in the
mujoco_mpc submodule: it renders the measured robot (joints from rt/lowstate,
base from the estimator's rt/sportmodestate_est) with a translucent ghost posed
at the planner's current best trajectory (rt/mjpc/plan), and writes an mp4 on
shutdown. As estimator_node.py does, this node maps its ROS parameters to the
worker's arguments, shuts rclpy down, and runs the script unmodified, so the
worker's unitree_sdk2py DDS I/O never shares a live process with rclpy's
CycloneDDS.

Nothing in the control path reads it, and it reads no ground truth: only
rt/lowstate, the estimator's estimate and the planner's own plan. It shows no
ghost unless the controller core publishes the plan, which it does only when
started through controller_launcher.py.
"""

import os
import runpy
import sys

import rclpy
from rclpy.node import Node

# Defaults for the container's mounts; MJPC_ROOT overrides the fork's location.
_MJPC_ROOT = os.environ.get("MJPC_ROOT", "/home/code/mujoco_mpc")
_DEFAULT_SCRIPT = os.path.join(
    _MJPC_ROOT, "mjpc", "deploy", "helper_scripts", "plan_visualizer.py")
# The staged task XML the controller loads: the ghost is posed from the plan's
# raw qpos rows, so the viewer's model must have the controller's qpos layout.
# MJPC_TASKS_DIR overrides the tasks directory.
_DEFAULT_SCENE = os.path.join(
    os.environ.get("MJPC_TASKS_DIR",
                   os.path.join(_MJPC_ROOT, "build", "mjpc", "tasks")),
    "humanoid_bench", "stabilize", "Stabilize_H12_Magpie.xml")


def main() -> None:
    """Read the parameters, shut rclpy down, then run the fork's visualizer script in this process."""
    rclpy.init()
    node = Node("mjpc_debug_visualizer")

    p = node.declare_parameter
    script = p("visualizer_script", _DEFAULT_SCRIPT).value
    scene = p("scene", _DEFAULT_SCENE).value
    domain = p("domain_id", int(os.environ.get("ROS_DOMAIN_ID", "0"))).value
    iface = p("iface", "").value               # "": the visualizer picks the robot subnet's interface
    no_auto_iface = p("no_auto_iface", False).value
    lowstate_topic = p("lowstate_topic", "rt/lowstate").value
    plan_topic = p("plan_topic", "rt/mjpc/plan").value
    # The estimator's output, as the controller's sportstate_topic. Never the
    # sim's ground-truth rt/sportmodestate: the real robot has no such topic.
    sportstate_topic = p("sportstate_topic", "rt/sportmodestate_est").value
    out_path = p("out_path", "/tmp/mjpc_plan.mp4").value
    fps = p("fps", 30.0).value                 # Hz, render and write rate
    width = p("width", 960).value
    height = p("height", 720).value
    camera = p("camera", "").value             # "": a free camera tracking the pelvis
    ghost_rgba = p("ghost_rgba", "0.25,0.45,1.0,0.35").value
    # sweep: the ghost's plan step cycles from now to the end of the horizon,
    # fading as it goes. A single ghost at step 0 would show nothing: plan row 0
    # is the measured state the core set.
    ghost_mode = p("ghost_mode", "sweep").value          # sweep | single
    sweep_rate = p("sweep_rate", 1.0).value              # plan s per wall s
    ghost_alpha_far = p("ghost_alpha_far", 0.15).value   # alpha at the far end of the sweep
    ghost_step = p("ghost_step", -1).value     # single mode only; -1 = last step
    ghost_trail = p("ghost_trail", 0).value    # > 0: a fading trail at every Nth step
    plan_stale_sec = p("plan_stale_sec", 1.0).value      # s; an older plan shows no ghost
    # Off by default: the ghost's base is the planner's belief and the measured
    # base the estimator's, and their divergence is signal. On compares posture alone.
    align_ghost_base = p("align_ghost_base", False).value
    max_seconds = p("max_seconds", 0.0).value  # s; 0: until shutdown

    node.get_logger().info(
        f"running fork plan visualizer {script} (plan={plan_topic}, "
        f"out={out_path}, domain={domain})")
    # The visualizer is a DDS-only worker from here on.
    node.destroy_node()
    rclpy.shutdown()

    if not os.path.isfile(script):
        sys.exit(f"[debug_visualizer_node] fork visualizer not found at {script} — "
                 "is the mujoco_mpc submodule mounted/checked out?")

    argv = [
        script,
        "--scene", str(scene),
        "--domain", str(domain),
        "--lowstate-topic", str(lowstate_topic),
        "--plan-topic", str(plan_topic),
        "--sportstate-topic", str(sportstate_topic),
        "--out", str(out_path),
        "--fps", str(fps),
        "--width", str(width),
        "--height", str(height),
        "--ghost-rgba", str(ghost_rgba),
        "--ghost-mode", str(ghost_mode),
        "--sweep-rate", str(sweep_rate),
        "--ghost-alpha-far", str(ghost_alpha_far),
        "--ghost-step", str(ghost_step),
        "--ghost-trail", str(ghost_trail),
        "--plan-stale-sec", str(plan_stale_sec),
        "--max-seconds", str(max_seconds),
    ]
    if camera:
        argv += ["--camera", str(camera)]
    if iface:
        argv += ["--iface", str(iface)]
    if no_auto_iface:
        argv.append("--no-auto-iface")
    if align_ghost_base:
        argv.append("--align-ghost-base")

    sys.argv = argv
    # run_name="__main__" runs the script's own entry point; SIGINT from ros2
    # launch reaches its handler, which unwinds the loop so the mp4 is finalised
    # (an unreleased VideoWriter leaves a truncated, unplayable file).
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
