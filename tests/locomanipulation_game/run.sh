#!/usr/bin/env bash
# One loco-manipulation game run in RoboCasa: locomanipulationgame_sim_bringup.launch.py in a disposable ROS
# container, the canonical RoboCasa launcher with record_simulator.py as its entry point in a disposable sim
# container, then a rendered video. Artifacts go to tests/results/locomanipulation_game-<name>-<time>/.
#
#   tests/locomanipulation_game/run.sh <name> [training|sequence] [commander seconds] [extra launch args...]
#
# Environment: ROS_DOMAIN_ID (default 194; never 0), GOLEM_SIM_ARGS (default "--headless --seed 42 --task
# AirDryFruit"), GOLEM_TEST_CAMERAS (default 0: the policy uses no camera; the lidar stays on for SLAM),
# GOLEM_SPAWN_BACKOFF (m the spawn is moved back from the task's counter; default 0),
# LMG_WALL_TIMEOUT (default 1800 s), LMG_RENDER (default 1).
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NAME=${1:?Usage: run.sh <name> [training|sequence] [seconds] [launch args...]}
MODE=${2:-training}
SECONDS_SIM=${3:-180}
shift $(( $# < 3 ? $# : 3 ))
EXTRA_ARGS=("$@")
case "$MODE" in training|sequence) ;; *) echo "mode must be training or sequence" >&2; exit 2 ;; esac
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-194}
export GOLEM_TEST_CAMERAS=${GOLEM_TEST_CAMERAS:-0}
LMG_WALL_TIMEOUT=${LMG_WALL_TIMEOUT:-1800}
read -r -a SIM_ARGS <<< "${GOLEM_SIM_ARGS:---headless --seed 42 --task AirDryFruit}"
source "$ROOT/docker/scripts/docker_common.sh"
load_config
check_domain robocasa
export GOLEM_DISPLAY=headless
configure_display
prepare_mounts robocasa
prepare_mounts ros
result="$ROOT/tests/results/locomanipulation_game-$NAME-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$result/record" "$result/ros"
chmod 777 "$result/record" "$result/ros"
prefix="golem-lmg-${ROS_DOMAIN_ID}-$$"
ros="$prefix-ros"; sim="$prefix-sim"
cleanup() {
    local status=$?
    trap - EXIT
    docker stop -t 10 "$ros" "$sim" >/dev/null 2>&1 || true
    docker logs "$ros" >"$result/ros.log" 2>&1 || true
    docker logs "$sim" >"$result/simulator.log" 2>&1 || true
    docker rm -f "$ros" "$sim" >/dev/null 2>&1 || true
    git -C "$ROOT" rev-parse HEAD >"$result/commit.txt" 2>/dev/null || true
    git -C "$ROOT" status --short >>"$result/commit.txt" 2>/dev/null || true
    printf '%s\n' "mode=$MODE seconds=$SECONDS_SIM domain=$ROS_DOMAIN_ID sim_args=${SIM_ARGS[*]} launch_args=${EXTRA_ARGS[*]}" >"$result/run.txt"
    if [[ "${LMG_RENDER:-1}" == 1 && -s "$result/record/qpos.f32" ]]; then
        "${COMPOSE[@]}" run -T --rm --no-deps -e MUJOCO_GL=egl -v "$ROOT/tests/locomanipulation_game:/home/code/tests/locomanipulation_game:ro" \
            -v "$result:/result" --entrypoint python3 robocasa /home/code/tests/locomanipulation_game/render_recording.py \
            /result/record --overlay /result/ros/commander.jsonl --out /result/video.mp4 >"$result/render.log" 2>&1 || echo "render failed (see render.log)"
    fi
    echo "Artifacts: $result"
    exit "$status"
}
trap cleanup EXIT
tests_mount=(-v "$ROOT/tests/locomanipulation_game:/home/code/tests/locomanipulation_game:ro")
launch="ros2 launch h1_bringup locomanipulationgame_sim_bringup.launch.py commander_mode:=$MODE commander_duration_s:=$SECONDS_SIM commander_log:=/results/commander.jsonl ${EXTRA_ARGS[*]}"
"${COMPOSE[@]}" run -d --no-deps --name "$ros" "${tests_mount[@]}" -v "$result/ros:/results" --entrypoint bash ros \
    -c "/home/code/h12_sim_scripts/launch_ros.sh bash -c 'echo LMG_BUILD_READY; python3 /home/code/h12_sim_scripts/wait_for_sim.py && exec $launch'" >/dev/null
for ((attempt=0; attempt<1800; attempt++)); do
    if docker logs "$ros" 2>&1 | grep -q LMG_BUILD_READY; then break; fi
    [[ $(docker inspect -f '{{.State.Running}}' "$ros") == true ]] || { docker logs "$ros" | tail -40; exit 1; }
    sleep 1
done
docker logs "$ros" 2>&1 | grep -q LMG_BUILD_READY || { echo 'ROS build timed out' >&2; exit 1; }
# The canonical launcher with only the Python entry point swapped for the recorder.
python3 - "$ROOT" "$result/launcher.sh" <<'PY'
from pathlib import Path
import sys
root, output = sys.argv[1:]
source = (Path(root) / 'docker/scripts/launch_robocasa.sh').read_text()
assert source.count('h12_mujoco.py') == 1, 'Canonical launcher changed; update the recorder hook'
source = source.replace('"$(dirname "$0")/runtime_common.sh"', '"/home/code/h12_sim_scripts/runtime_common.sh"')
Path(output).write_text(source.replace('h12_mujoco.py', '/home/code/tests/locomanipulation_game/record_simulator.py'))
PY
"${COMPOSE[@]}" run -d --no-deps --name "$sim" "${tests_mount[@]}" -e GOLEM_CAMERAS="$GOLEM_TEST_CAMERAS" \
    -e GOLEM_SPAWN_BACKOFF="${GOLEM_SPAWN_BACKOFF:-0}" \
    -e GOLEM_RECORD_DIR=/record -v "$result/record:/record" -v "$result/launcher.sh:/lmg_launcher.sh:ro" \
    --entrypoint bash robocasa /lmg_launcher.sh "${SIM_ARGS[@]}" >/dev/null
for ((t=0; t<LMG_WALL_TIMEOUT; t++)); do
    if grep -q '"type": "done"' "$result/ros/commander.jsonl" 2>/dev/null; then echo "commander done"; break; fi
    for container in "$ros" "$sim"; do
        [[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]] || { echo "$container exited early"; exit 1; }
    done
    sleep 1
done
sleep 3
