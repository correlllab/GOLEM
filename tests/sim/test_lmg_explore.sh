#!/bin/bash
# Run the H1-2's closed-loop frontier-exploration test on lmg_deploy's policy in the RoboCasa
# sim, from the host: golem_skills' frontier explorer walks the H1-2, on the loco-manipulation
# game's MARL policy, through Nav2 from the spawn until it reports the kitchen explored, or
# its time runs out.
#
#   tests/sim/test_lmg_explore.sh                         # lmg_deploy's policies/h12/
#   GOLEM_TEST_VIDEO=1 tests/sim/test_lmg_explore.sh      # and the video
#   tests/sim/test_lmg_explore.sh --video OUT             # re-render OUT/explore.mp4 of a finished run
#
# Builds core_ws and starts the sim (h12_mujoco.py --h12) headless, with its sensors on, in the
# layout 9 / style 9 / seed 42 kitchen, the spawn backed EXPLORE_BACKOFF m into open floor, as
# test_sim_stack.sh does. In its network namespace a ros container then runs
# 'ros2 launch golem_bringup lmg_sim.launch.py use_rviz:=false use_skills:=true'
# and the test's steps (test_lmg_explore.py drive): wait for the policy node's release of the
# elastic band, Nav2 and /skill/frontier_explore, send one SkillFrontierExplore goal, record,
# wait for its result. Then it stops the sim first (OUT/stop), with the robot standing under
# the policy, and the bringup once the sim has saved its recording: stopping the bringup first
# would stop rt/lowcmd and the sim would drop the robot while it still runs. With
# GOLEM_TEST_VIDEO=1 it renders OUT/explore.mp4 from the recordings (test_lmg_explore.py
# video). Finally it checks the run against the sim's recording (test_lmg_explore.py report;
# see test_lmg_explore.py for what is checked). The cameras render on the GPU (EGL).
#
# The containers have no network of their own (tests/sim/common.sh), so no DDS leaves the
# host, and nothing talks to the robot. ROS_DOMAIN_ID must not be 0.
#
# Env:
#   GOLEM_SIM_ODOM must be unset: the test runs on FAST-LIO's odometry, never the sim's
#               ground truth.
#   GOLEM_TEST_VIDEO  1: render OUT/explore.mp4 (chase camera, head camera, /map); default 0
#   CORE_WS     a core_ws to build in place and use (default the repo's, built
#               into container_cache/sim_tests; see tests/sim/common.sh)
#   OUT         output directory, new or empty (default
#               tests/results/sim-lmg-explore-<time>-<pid>)
#   SIM_IMAGE   default golem_sim_robocasa:latest
#   ROS_IMAGE   default golem_ros:latest
#   ROS_DOMAIN_ID   default 197
#   EXPLORE_BACKOFF m the spawn is backed into open floor (default 2.0)
#   GOLEM_TEST_RECORD_EVERY  physics steps per recorded sample (default 10, 50 Hz)
#   GOLEM_CAMERAS, GOLEM_CAMERA_SCALE   passed to the sim as the shell has them, docker/.env's never
#   GOLEM_SIM_RATE   passed to the sim as the shell has it, docker/.env's never: the target
#               real-time factor (unset is 1, real time; or 2, max, ...)
#
# Exit: 0 when every check passes; 1 when a check, the test's steps, the video or a container
#   fails; 2 for bad arguments or settings, a bad domain or a non-empty OUT; 3 when the core_ws
#   build fails.
set -eo pipefail

# Arguments and settings: none, and the sim's ground-truth odometry off.

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = --video ] && [ $# -eq 2 ] && [ -f "$2/sim.npz" ]; then
    # Only the video, from a finished run's recordings, in a disposable robocasa container with no network.
    OUT=$(cd -- "$2" && pwd)
    source "$HERE/../../docker/scripts/docker_common.sh"
    export GOLEM_ROBOT=h12 GOLEM_DISPLAY=headless ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-197}
    ROBOT_FLAG=
    load_config
    configure_robot robocasa
    configure_display
    prepare_mounts robocasa
    printf '%s\n' 'services:' '  robocasa:' "    image: \"${SIM_IMAGE:-golem_sim_robocasa:latest}\"" \
        '    network_mode: none' '    volumes:' \
        "      - {type: bind, source: \"$HERE/../g1\", target: /home/code/tests/g1, read_only: true}" \
        "      - {type: bind, source: \"$HERE\", target: /home/code/tests/sim, read_only: true}" \
        "      - {type: bind, source: \"$OUT\", target: /out}" > "$OUT/video_compose.yml"
    COMPOSE+=(-f "$OUT/video_compose.yml")
    if ! "${COMPOSE[@]}" run --rm -T --no-deps -e PYTHONDONTWRITEBYTECODE=1 -e MUJOCO_GL=egl robocasa \
            python3 /home/code/tests/sim/test_lmg_explore.py video --out /out > "$OUT/video.log" 2>&1; then
        tail -20 "$OUT/video.log" >&2
        exit 1
    fi
    # The container writes as root.
    "${COMPOSE[@]}" run --rm -T --no-deps --entrypoint chown robocasa -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true
    tail -1 "$OUT/video.log"
    exit 0
elif [ -n "${1:-}" ]; then
    echo "usage: $0 [--video OUT]" >&2
    exit 2
fi
if [ -n "${GOLEM_SIM_ODOM:-}" ]; then
    echo 'Unset GOLEM_SIM_ODOM: the test runs on FAST-LIO odometry, never the sim'"'"'s ground truth.' >&2
    exit 2
fi
BRINGUP="lmg_sim.launch.py use_rviz:=false use_skills:=true"

# The sim's knobs: odometry off, the spawn backed into open floor, the shell's sensor knobs.
SIM_ENV=(-e GOLEM_SIM_ODOM=0 -e "GOLEM_SPAWN_BACKOFF=${EXPLORE_BACKOFF:-2.0}")
source "$HERE/common.sh"
if [ -n "${GOLEM_TEST_RECORD_EVERY:-}" ]; then
    SIM_ENV+=(-e "GOLEM_TEST_RECORD_EVERY=$GOLEM_TEST_RECORD_EVERY")
fi
sim_test_shell_knobs GOLEM_CAMERAS GOLEM_CAMERA_SCALE GOLEM_SIM_RATE

# The containers, the build, then the sim, up before the bringup.
sim_test_setup h12 sim test_lmg_explore lmg-explore explore 197
sim_test_log_load

echo "[test_lmg_explore] DDS domain $ROS_DOMAIN_ID, output in $OUT"
sim_test_build_ros
sim_test_start_sim test_lmg_explore.py "${SIM_ENV[@]}"
grep "\[test_sim_stack\]" "$OUT/sim.log" || true
echo "[test_lmg_explore] sim up; starting $BRINGUP"

# The bringup in the background and the test's steps beside it, at most 1800 s.

sim_test_run_ros 1800 "ros2 launch golem_bringup $BRINGUP > /out/launch.log 2>&1 &
    LAUNCH=\$!
    python3 -u /home/code/tests/sim/test_lmg_explore.py drive --out /out
    rc=\$?
    # The sim first, with the robot standing: it looks for /out/stop every 500
    # physics steps, then saves sim.npz and, last, contacts.json; then the bringup.
    touch /out/stop
    for _ in \$(seq 1 60); do [ -e /out/contacts.json ] && break; sleep 1; done
    kill -INT \$LAUNCH
    wait \$LAUNCH
    exit \$rc"

# Stop the sim, render the video if asked, then check the run against its recording.
sim_test_stop_sim
if [ "${GOLEM_TEST_VIDEO:-0}" = 1 ]; then
    echo "[test_lmg_explore] rendering $OUT/explore.mp4"
    if ! "${COMPOSE[@]}" run --rm -T --no-deps -e PYTHONDONTWRITEBYTECODE=1 -e MUJOCO_GL=egl robocasa \
            python3 /home/code/tests/sim/test_lmg_explore.py video --out /out > "$OUT/video.log" 2>&1; then
        tail -20 "$OUT/video.log" >&2
        echo "[test_lmg_explore] the video failed; see $OUT/video.log" >&2
        VIDEO_FAILED=1
    fi
fi
if [ -n "${VIDEO_FAILED:-}" ]; then
    DRIVE_RC=1
fi
sim_test_report test_lmg_explore.py
