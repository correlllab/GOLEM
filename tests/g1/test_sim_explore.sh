#!/bin/bash
# Closed-loop frontier-exploration test of the G1 RoboCasa sim, run from the host:
# g1_navigation's frontier explorer walks the G1 through Nav2 from the spawn until
# it reports the kitchen explored, or its time runs out.
#
#   tests/g1/test_sim_explore.sh                    # odom_source fastlio (FAST_LIO on the sim's MID-360)
#   GOLEM_SIM_ODOM=1 tests/g1/test_sim_explore.sh   # odom_source sim_gt (the sim's ground truth)
#
# Builds core_ws for the G1 and starts the sim (h12_mujoco.py --g1) headless,
# with its sensors on, in the layout 9 / style 9 / seed 42 kitchen, the spawn
# backed EXPLORE_BACKOFF m into open floor, as test_sim_nav.sh does. In its
# network namespace a ros container then runs 'ros2 launch g1_bringup
# g1_sim_bringup.launch.py use_rviz:=false use_locomotion:=true use_nav:=true
# frontier_explore:=true odom_source:=...' and the test's steps
# (test_sim_explore.py drive): wait for the walking node, Nav2 and
# /skill/frontier_explore, send one SkillFrontierExplore goal, record, wait for
# its result. Then it stops the sim first (OUT/stop), with the robot standing
# under the walking node, and the bringup once the sim has saved its recording:
# stopping the bringup first would let the safety layer's exit estop
# (sim_split.yaml: every motor off) drop the robot while the sim still runs.
# Finally it checks the run against the sim's recording (test_sim_explore.py
# report; see test_sim_explore.py for what is checked). Exits 0 only if every
# check passes. The camera renders on the GPU (EGL).
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
#               tests/results/g1-explore-<odom source>-<time>-<pid>)
#   SIM_IMAGE   default golem_sim_robocasa:latest
#   ROS_IMAGE   default golem_ros:latest
#   ROS_DOMAIN_ID   default 198
#   EXPLORE_BACKOFF m the spawn is backed into open floor (default 2.0)
#   GOLEM_TEST_RECORD_EVERY  physics steps per recorded sample (default 10, 50 Hz)
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
SIM_ENV=(-e "GOLEM_SIM_ODOM=$SIM_ODOM" -e "GOLEM_SPAWN_BACKOFF=${EXPLORE_BACKOFF:-2.0}" -e GOLEM_LIDAR=)
source "$HERE/common.sh"
if [ -n "${GOLEM_TEST_RECORD_EVERY:-}" ]; then
    SIM_ENV+=(-e "GOLEM_TEST_RECORD_EVERY=$GOLEM_TEST_RECORD_EVERY")
fi
g1_test_shell_knobs GOLEM_LIDAR_POINTS GOLEM_LIDAR_HZ GOLEM_CAMERAS GOLEM_CAMERA_HZ GOLEM_CAMERA_SCALE
g1_test_setup test_sim_explore "explore-$ODOM" "$ODOM" 198
g1_test_log_load

echo "[test_sim_explore] odom_source $ODOM on DDS domain $ROS_DOMAIN_ID, output in $OUT"
g1_test_build_ros
g1_test_start_sim test_sim_explore.py "${SIM_ENV[@]}"
echo "[test_sim_explore] sim up; starting the bringup with the walking node, navigation and the explorer"

g1_test_run_ros 1500 "ros2 launch g1_bringup g1_sim_bringup.launch.py use_rviz:=false use_locomotion:=true use_nav:=true \
        frontier_explore:=true odom_source:=$ODOM > /out/launch.log 2>&1 &
    LAUNCH=\$!
    python3 -u /home/code/tests/g1/test_sim_explore.py drive --out /out --odom-source $ODOM
    rc=\$?
    # the sim first, with the robot standing: it looks for /out/stop every 500
    # physics steps, then saves sim.npz and, last, contacts.json; then the bringup
    touch /out/stop
    for _ in \$(seq 1 60); do [ -e /out/contacts.json ] && break; sleep 1; done
    kill -INT \$LAUNCH
    wait \$LAUNCH
    exit \$rc"

g1_test_stop_sim
g1_test_report test_sim_explore.py
