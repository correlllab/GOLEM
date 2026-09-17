#!/usr/bin/env bash
# Real ROS controllers -> CycloneDDS -> simulation -> measured acceptance.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SIMULATOR=${1:?Usage: run.sh isaac|robocasa arms|locomotion}
TEST_CASE=${2:?Usage: run.sh isaac|robocasa arms|locomotion}
case "$SIMULATOR" in isaac|robocasa) ;; *) exit 2 ;; esac
case "$TEST_CASE" in arms|locomotion) ;; *) exit 2 ;; esac
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-193}
export GOLEM_TEST_WALL_TIMEOUT=${GOLEM_TEST_WALL_TIMEOUT:-900}
[[ "$GOLEM_TEST_WALL_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || exit 2
source "$ROOT/docker/scripts/docker_common.sh"
load_config
check_domain "$SIMULATOR"
export GOLEM_DISPLAY=headless
configure_display
prepare_mounts "$SIMULATOR"
prepare_mounts ros
result="$ROOT/tests/results/stack-$SIMULATOR-$TEST_CASE-$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$result/telemetry" "$result/metrics"
# Root in Docker writes the artifacts; allow ordinary host users to inspect and remove them.
chmod 777 "$result/telemetry" "$result/metrics"
prefix="golem-stack-test-${ROS_DOMAIN_ID}-$$"
stack="$prefix-ros"; simulator="$prefix-sim"; checker="$prefix-check"
cleanup() {
    local status=$?
    trap - EXIT
    for pair in "$stack:ros" "$simulator:simulator" "$checker:checker"; do
        docker logs "${pair%:*}" >"$result/${pair##*:}.log" 2>&1 || true
        docker rm -f "${pair%:*}" >/dev/null 2>&1 || true
    done
    echo "Stack test artifacts: $result"
    exit "$status"
}
trap cleanup EXIT
env_args=(-e "SIMULATOR=$SIMULATOR" -e "TEST_CASE=$TEST_CASE" -e "GOLEM_TEST_WALL_TIMEOUT=$GOLEM_TEST_WALL_TIMEOUT")
tests_mount=(-v "$ROOT/tests/stack:/home/code/tests/stack:ro")
"${COMPOSE[@]}" run -d --no-deps --name "$stack" "${env_args[@]}" "${tests_mount[@]}" --entrypoint bash ros /home/code/tests/stack/launch_ros_stack.sh >/dev/null
for ((attempt=0; attempt<600; attempt++)); do
    if docker logs "$stack" 2>&1 | grep STACK_BUILD_READY >/dev/null; then break; fi
    [[ $(docker inspect -f '{{.State.Running}}' "$stack") == true ]] || { docker logs "$stack"; exit 1; }
    sleep 1
done
docker logs "$stack" 2>&1 | grep STACK_BUILD_READY >/dev/null || { echo 'ROS test build timed out' >&2; exit 1; }
# Retain every setting in the canonical launcher; replace only the Python entry
# point with the passive recorder. No production source is modified.
python3 - "$ROOT" "$SIMULATOR" "$result/launcher.sh" <<'PY'
from pathlib import Path
import sys
root, sim, output = sys.argv[1:]
source = (Path(root)/f'docker/scripts/launch_{sim}.sh').read_text()
old = '/home/code/CL_isaaclab_sim/sim_main.py' if sim == 'isaac' else 'h12_mujoco.py'
assert source.count(old) == 1, 'Canonical launcher changed; update test instrumentation'
Path(output).write_text(source.replace(old, '/home/code/tests/stack/observe_simulator.py'))
PY
sim_args=(--headless)
if [[ "$SIMULATOR" == isaac ]]; then
    sim_args+=(--hand_type magpie)
    # Arm kinematics is isolated from balance; locomotion always uses floating base.
    [[ "$TEST_CASE" != arms ]] || sim_args+=(--fix_base)
fi
"${COMPOSE[@]}" run -d --no-deps --name "$checker" "${env_args[@]}" "${tests_mount[@]}" -v "$result/telemetry:/telemetry:ro" -v "$result/metrics:/results" --entrypoint bash ros /home/code/tests/stack/run_checker.sh >/dev/null
"${COMPOSE[@]}" run -d --no-deps --name "$simulator" "${env_args[@]}" "${tests_mount[@]}" -e GOLEM_TEST_TELEMETRY=/telemetry/state.jsonl -v "$result/telemetry:/telemetry" -v "$result/launcher.sh:/home/code/h12_sim_scripts/test_launcher.sh:ro" --entrypoint bash "$SIMULATOR" /home/code/h12_sim_scripts/test_launcher.sh "${sim_args[@]}" >/dev/null
for ((attempt=0; attempt<GOLEM_TEST_WALL_TIMEOUT+30; attempt++)); do
    [[ $(docker inspect -f '{{.State.Running}}' "$checker") == true ]] || break
    for container in "$stack" "$simulator"; do
        [[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]] || { docker logs "$container"; echo 'Required process exited before test completed' >&2; exit 1; }
    done
    sleep 1
done
docker logs "$checker"
[[ $(docker inspect -f '{{.State.Running}}' "$checker") == false ]] || { echo 'Stack checker timed out' >&2; exit 1; }
exit "$(docker inspect -f '{{.State.ExitCode}}' "$checker")"
