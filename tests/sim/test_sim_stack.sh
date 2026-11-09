#!/bin/bash
# Closed-loop test of either robot's whole stack in the RoboCasa sim, run from
# the host: the robot stands unsupported, Nav2 walks it to a goal 1.5 m ahead
# while /named_config moves its arms, and SLAM stays healthy throughout.
#
#   tests/sim/test_sim_stack.sh --h12
#   tests/sim/test_sim_stack.sh --g1
#
# Builds core_ws and starts the robot's sim (h12_mujoco.py --h12 or --g1)
# headless, with its sensors on, in the layout 9 / style 9 / seed 42 kitchen,
# the spawn backed 2.0 m into open floor. In its network namespace a ros
# container then runs the robot's sim bringup with navigation
# ('ros2 launch golem_bringup h12_sim.launch.py use_rviz:=false use_skills:=false'
# or 'g1_sim.launch.py use_rviz:=false') and the test's steps
# (test_sim_stack.py drive): wait for the elastic band's release and Nav2, send
# one NavigateToPose goal, move the arms to 't_pose' and back 'home' while the
# robot walks, record. Then it stops the sim first (OUT/stop), with the robot
# standing, and the bringup once the sim has saved its recording: stopping the
# bringup first would stop rt/lowcmd, and the sim, with every motor off after
# GOLEM_CMD_TIMEOUT, would drop the robot while it still runs. Finally it
# checks the run against the sim's recording (test_sim_stack.py report; see it
# for what is checked). Exits 0 only if every check passes. The cameras render
# on the GPU (EGL).
#
# The containers have no network of their own (tests/sim/common.sh), so no DDS
# leaves the host, and nothing talks to the robot. ROS_DOMAIN_ID must not be 0.
#
# Env:
#   GOLEM_SIM_ODOM must be unset: the test runs on FAST-LIO's odometry, never the
#               sim's ground truth.
#   CORE_WS     a core_ws to build in place and use (default the repo's, built
#               into container_cache/sim_tests; see tests/sim/common.sh)
#   OUT         output directory, new or empty (default
#               tests/results/sim-stack-<robot>-<time>-<pid>)
#   SIM_IMAGE   default golem_sim_robocasa:latest
#   ROS_IMAGE   default golem_ros:latest
#   ROS_DOMAIN_ID   default 199
#   GOLEM_TEST_RECORD_EVERY  physics steps per recorded sample (default 10, 50 Hz)
#   SIM_GPUS    the sim container's NVIDIA_VISIBLE_DEVICES (default all; empty for
#               none: Mesa's software EGL then renders the cameras, far slower)
#   GOLEM_CAMERAS, GOLEM_CAMERA_SCALE, and for the G1 GOLEM_LIDAR_POINTS,
#   GOLEM_LIDAR_HZ, GOLEM_CAMERA_HZ   passed to the sim as the shell has them,
#               docker/.env's never
#   GOLEM_LIDAR_BACKEND   passed to the sim as the shell has it, docker/.env's never:
#               unset is auto (the GPU caster when the sim has a GPU, see SIM_GPUS, else
#               the CPU's), cpu casts every ray on the CPU as on the Mac; the sim logs
#               the caster it built
#
# Exit: 0 when every check passes; 1 when a check, the test's steps or a container fails;
#   2 for bad arguments, a set GOLEM_SIM_ODOM, a bad domain or a non-empty OUT; 3 when the
#   core_ws build fails.
set -eo pipefail

# The robot, and the refusal of the sim's ground-truth odometry.

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
    --h12) ROBOT=h12 ;;
    --g1) ROBOT=g1 ;;
    *) echo "usage: $0 --h12|--g1" >&2; exit 2 ;;
esac
if [ $# -ne 1 ]; then
    echo "usage: $0 --h12|--g1" >&2
    exit 2
fi
if [ -n "${GOLEM_SIM_ODOM:-}" ]; then
    echo 'Unset GOLEM_SIM_ODOM: the test runs on FAST-LIO odometry, never the sim'"'"'s ground truth.' >&2
    exit 2
fi

# The sim's knobs: odometry off, the spawn backed into open floor, the shell's sensor knobs.
SIM_ENV=(-e GOLEM_SIM_ODOM=0 -e GOLEM_SPAWN_BACKOFF=2.0)
source "$HERE/common.sh"
if [ -n "${GOLEM_TEST_RECORD_EVERY:-}" ]; then
    SIM_ENV+=(-e "GOLEM_TEST_RECORD_EVERY=$GOLEM_TEST_RECORD_EVERY")
fi
if [ "$ROBOT" = g1 ]; then
    # The lidar stays on: navigation needs it.
    SIM_ENV+=(-e GOLEM_LIDAR=)
    sim_test_shell_knobs GOLEM_LIDAR_POINTS GOLEM_LIDAR_HZ GOLEM_CAMERAS GOLEM_CAMERA_HZ GOLEM_CAMERA_SCALE \
        GOLEM_LIDAR_BACKEND
    BRINGUP="g1_sim.launch.py use_rviz:=false"
else
    sim_test_shell_knobs GOLEM_CAMERAS GOLEM_CAMERA_SCALE GOLEM_LIDAR_BACKEND
    BRINGUP="h12_sim.launch.py use_rviz:=false use_skills:=false"
fi

# Containers, build, then the sim, up before the bringup.
sim_test_setup "$ROBOT" sim test_sim_stack "stack-$ROBOT" stack 199
sim_test_log_load

echo "[test_sim_stack] $ROBOT on DDS domain $ROS_DOMAIN_ID, output in $OUT"
sim_test_build_ros
sim_test_start_sim test_sim_stack.py "${SIM_ENV[@]}"
grep "\[test_sim_stack\]" "$OUT/sim.log" || true
echo "[test_sim_stack] sim up; starting $BRINGUP"

# The bringup in the background and the test's steps beside it, at most 900 s.

sim_test_run_ros 900 "ros2 launch golem_bringup $BRINGUP > /out/launch.log 2>&1 &
    LAUNCH=\$!
    python3 -u /home/code/tests/sim/test_sim_stack.py drive --out /out
    rc=\$?
    # The sim first, with the robot standing: it looks for /out/stop every 500
    # physics steps, then saves sim.npz and, last, contacts.json; then the bringup.
    touch /out/stop
    for _ in \$(seq 1 60); do [ -e /out/contacts.json ] && break; sleep 1; done
    kill -INT \$LAUNCH
    wait \$LAUNCH
    exit \$rc"

# Stop the sim, then check the run against its recording.
sim_test_stop_sim
sim_test_report test_sim_stack.py
