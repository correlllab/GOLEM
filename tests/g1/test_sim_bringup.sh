#!/bin/bash
# Acceptance test of the G1 RoboCasa sim with the whole bringup, run from the host.
#
#   tests/g1/test_sim_bringup.sh           # the arm test
#   tests/g1/test_sim_bringup.sh --walk    # the walk test
#
# Builds core_ws for the G1, starts the sim (h12_mujoco.py --g1) headless in the
# layout 9 / style 9 / seed 42 kitchen, then, in its network namespace, a ros
# container that runs 'ros2 launch golem_bringup g1_sim.launch.py
# use_rviz:=false' and the test's steps (test_sim_bringup.py drive), and finally
# checks the sim's recording against them (test_sim_bringup.py report). See
# test_sim_bringup.py for what is checked. Exits 0 only if every check passes.
# The sim's lidar and head camera are on, as by default; the camera renders on
# the GPU (EGL).
#
# The containers have no network of their own (common.sh), so no DDS leaves the
# host, and nothing talks to the robot. ROS_DOMAIN_ID must not be 0.
#
# Env:
#   CORE_WS     a core_ws to build in place and use (default the repo's, built
#               into container_cache/g1_tests; see common.sh)
#   OUT         output directory, new or empty (default tests/results/g1-bringup-<time>-<pid>,
#               or g1-walk-...)
#   SIM_IMAGE   default golem_sim_robocasa:latest
#   ROS_IMAGE   default golem_ros:latest
#   ROS_DOMAIN_ID   default 195 for the arm test, 196 for the walk test
#   WALK_BACKOFF    m the walk test backs the spawn into open floor (default 2.0)
#   GOLEM_TEST_RECORD_EVERY  physics steps per recorded sample (default 10, 50 Hz)
#   SIM_GPUS    the sim container's NVIDIA_VISIBLE_DEVICES (default all; empty for
#               none: Mesa's software EGL then renders the camera, far slower)
#   GOLEM_LIDAR, GOLEM_LIDAR_POINTS, GOLEM_LIDAR_HZ, GOLEM_CAMERAS, GOLEM_CAMERA_HZ,
#   GOLEM_CAMERA_SCALE   passed to the sim as the shell has them, docker/.env's
#               never (g1_sensors.SensorConfig); the test expects what the sim reads
set -eo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST=arm
NAME=bringup
DOMAIN=195
LAUNCH_ARGS="use_rviz:=false"
SIM_ENV=(-e GOLEM_SIM_ODOM=0 -e GOLEM_SPAWN_BACKOFF=)
DRIVE_ARGS=""
if [ "${1:-}" = "--walk" ]; then
    TEST=walk
    NAME=walk
    DOMAIN=196
    # the spawn faces a counter: walk into open floor behind it instead
    SIM_ENV=(-e GOLEM_SIM_ODOM=0 -e "GOLEM_SPAWN_BACKOFF=${WALK_BACKOFF:-2.0}")
    DRIVE_ARGS="--walk"
elif [ -n "${1:-}" ]; then
    echo "usage: $0 [--walk]" >&2
    exit 2
fi
source "$HERE/common.sh"
if [ -n "${GOLEM_TEST_RECORD_EVERY:-}" ]; then
    SIM_ENV+=(-e "GOLEM_TEST_RECORD_EVERY=$GOLEM_TEST_RECORD_EVERY")
fi
g1_test_shell_knobs GOLEM_LIDAR GOLEM_LIDAR_POINTS GOLEM_LIDAR_HZ GOLEM_CAMERAS GOLEM_CAMERA_HZ GOLEM_CAMERA_SCALE
g1_test_setup test_sim_bringup "$NAME" "$TEST" "$DOMAIN"

echo "[test_sim_bringup] $TEST test on DDS domain $ROS_DOMAIN_ID, output in $OUT"
g1_test_build_ros
g1_test_start_sim test_sim_bringup.py "${SIM_ENV[@]}"
echo "[test_sim_bringup] sim up; starting the bringup"

g1_test_run_ros 900 "ros2 launch golem_bringup g1_sim.launch.py $LAUNCH_ARGS > /out/launch.log 2>&1 &
    LAUNCH=\$!
    python3 -u /home/code/tests/g1/test_sim_bringup.py drive --out /out $DRIVE_ARGS
    rc=\$?
    kill -INT \$LAUNCH
    wait \$LAUNCH
    exit \$rc"

g1_test_stop_sim
g1_test_report test_sim_bringup.py
