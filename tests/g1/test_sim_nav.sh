#!/bin/bash
# Closed-loop navigation test of the G1 RoboCasa sim, run from the host: Nav2
# walks the G1 to a goal 2-3 m away in the kitchen, or to NAV_GOAL.
#
#   tests/g1/test_sim_nav.sh                     # odom_source fastlio (FAST_LIO on the sim's MID-360)
#   GOLEM_SIM_ODOM=1 tests/g1/test_sim_nav.sh    # odom_source sim_gt (the sim's ground truth)
#   NAV_GOAL=0.62,-5.08,180 tests/g1/test_sim_nav.sh   # the corridor behind the counter peninsula
#
# Builds core_ws for the G1 and starts the sim (h12_mujoco.py --g1) headless,
# with its sensors on, in the layout 9 / style 9 / seed 42 kitchen, the spawn
# backed NAV_BACKOFF m into open floor. In its network namespace a ros container
# then runs 'ros2 launch golem_bringup g1_sim.launch.py use_rviz:=false
# use_locomotion:=true use_nav:=true odom_source:=...' and the test's steps
# (test_sim_nav.py drive): wait for the walking node and Nav2, send one
# NavigateToPose goal (the sim's pick from its own geometry, or NAV_GOAL),
# record. Then it stops the sim first (OUT/stop), with the robot standing under
# the walking node, and the bringup once the sim has saved its recording:
# stopping the bringup first would stop rt/lowcmd (golem_safety stops publishing
# when it exits in split mode), and the sim, with every motor off after
# GOLEM_CMD_TIMEOUT, would drop the robot while it still runs.
# Finally it checks the run against the sim's recording (test_sim_nav.py report;
# see test_sim_nav.py for what is checked), the robot standing to the end of it
# and no fall in sim.log. Exits 0 only if every check passes. The camera
# renders on the GPU (EGL).
#
# The containers have no network of their own (common.sh), so no DDS leaves the
# host, and nothing talks to the robot. ROS_DOMAIN_ID must not be 0.
#
# Env:
#   GOLEM_SIM_ODOM  1: the sim publishes its ground truth and the bringup runs
#               odom_source:=sim_gt; unset or 0: fastlio (the shell's value;
#               docker/.env's does not apply)
#   CORE_WS     a core_ws to build in place and use (default the repo's, built
#               into container_cache/g1_tests; see common.sh)
#   OUT         output directory, new or empty (default
#               tests/results/g1-nav-<odom source>-<time>-<pid>)
#   SIM_IMAGE   default golem_sim_robocasa:latest
#   ROS_IMAGE   default golem_ros:latest
#   ROS_DOMAIN_ID   default 197
#   NAV_BACKOFF m the spawn is backed into open floor (default 2.0)
#   NAV_GOAL    "x,y[,yaw_deg]" in the world frame instead of the sim's pick (the heading
#               defaults to the direction from the spawn); the sim refuses one that is not
#               reachable from the spawn through cells 0.35 m clear (test_sim_nav.py's
#               PATH_CLEARANCE; goal.json's "error"), and the test fails
#   GOLEM_TEST_RECORD_EVERY  physics steps per recorded sample (default 10, 50 Hz)
#   NAV_KEEP_LIDAR  1: also keep the /livox/lidar frames from the goal on (lidar_frames.npz,
#               about 1 MB per sim second)
#   SIM_GPUS    the sim container's NVIDIA_VISIBLE_DEVICES (default all; empty for
#               none: Mesa's software EGL then renders the camera, far slower)
#   GOLEM_LIDAR_POINTS, GOLEM_LIDAR_HZ, GOLEM_CAMERAS, GOLEM_CAMERA_HZ,
#   GOLEM_CAMERA_SCALE   passed to the sim as the shell has them, docker/.env's
#               never (g1_sensors.SensorConfig)
set -eo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${1:-}" ]; then
    echo "usage: [GOLEM_SIM_ODOM=1] $0" >&2
    exit 2
fi
case "$(echo "${GOLEM_SIM_ODOM:-0}" | tr '[:upper:]' '[:lower:]')" in
    0|off|false|no|'') ODOM=fastlio; SIM_ODOM=0 ;;
    *) ODOM=sim_gt; SIM_ODOM=1 ;;
esac
# the lidar stays on: the test needs it
SIM_ENV=(-e "GOLEM_SIM_ODOM=$SIM_ODOM" -e "GOLEM_SPAWN_BACKOFF=${NAV_BACKOFF:-2.0}" -e "NAV_GOAL=${NAV_GOAL:-}"
         -e GOLEM_LIDAR=)
source "$HERE/common.sh"
if [ -n "${GOLEM_TEST_RECORD_EVERY:-}" ]; then
    SIM_ENV+=(-e "GOLEM_TEST_RECORD_EVERY=$GOLEM_TEST_RECORD_EVERY")
fi
g1_test_shell_knobs GOLEM_LIDAR_POINTS GOLEM_LIDAR_HZ GOLEM_CAMERAS GOLEM_CAMERA_HZ GOLEM_CAMERA_SCALE
DRIVE_ARGS="--odom-source $ODOM"
if [ "${NAV_KEEP_LIDAR:-0}" = 1 ]; then
    DRIVE_ARGS="$DRIVE_ARGS --keep-lidar"
fi
g1_test_setup test_sim_nav "nav-$ODOM" "$ODOM" 197
g1_test_log_load

echo "[test_sim_nav] odom_source $ODOM on DDS domain $ROS_DOMAIN_ID, output in $OUT"
g1_test_build_ros
g1_test_start_sim test_sim_nav.py "${SIM_ENV[@]}"
grep "\[test_sim_nav\]" "$OUT/sim.log" || true
echo "[test_sim_nav] sim up; starting the bringup with the walking node and navigation"

g1_test_run_ros 900 "ros2 launch golem_bringup g1_sim.launch.py use_rviz:=false use_locomotion:=true use_nav:=true \
        odom_source:=$ODOM > /out/launch.log 2>&1 &
    LAUNCH=\$!
    python3 -u /home/code/tests/g1/test_sim_nav.py drive --out /out $DRIVE_ARGS
    rc=\$?
    # the sim first, with the robot standing: it looks for /out/stop every 500
    # physics steps, then saves sim.npz and, last, contacts.json; then the bringup
    touch /out/stop
    for _ in \$(seq 1 60); do [ -e /out/contacts.json ] && break; sleep 1; done
    kill -INT \$LAUNCH
    wait \$LAUNCH
    exit \$rc"

g1_test_stop_sim
g1_test_report test_sim_nav.py
