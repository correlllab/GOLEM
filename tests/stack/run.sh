#!/usr/bin/env bash
# Run one H1-2 stack case: the real controllers against a sim, scored from the sim's own telemetry.
#
# Usage: tests/stack/run.sh isaac|robocasa arms|locomotion
#   arms         both wrists 2 cm forward and back through /frame_task (check_frame_task.py)
#   locomotion   ALMI balance, walk and stop on /cmd_vel (check_lowerbody.py); RoboCasa only:
#                `isaac locomotion` is refused, since Isaac's rt/lowstate tick counts publishes
#                and the leg node never starts there (tests/stack/README.md)
# Environment:
#   ROS_DOMAIN_ID            DDS domain, default 193; check_domain refuses 0.
#   GOLEM_TEST_WALL_TIMEOUT  the checker's wall timeout, s, default 900.
#   GOLEM_TEST_CAMERAS       the sim's GOLEM_CAMERAS, default 0.
#   GOLEM_CPU_PINNING, GOLEM_SIM_CPUSET, GOLEM_ROS_CPUSET   the CPUs the RoboCasa sim and the
#                            ROS containers are pinned to, as the wrappers pin them
#                            (docker_common.sh's configure_cpus); Isaac is not pinned.
#   docker/.env and the shell, through docker_common.sh's load_config. GOLEM_SIM_RATE is cleared
#                            whatever they say: the sim runs at real time, the only rate the
#                            checks are gates at.
# Starts three disposable containers (golem-stack-test-<domain>-<pid>-{ros,sim,check}) and leaves
# their logs, telemetry, the checker's report and run.json in tests/results/stack-<sim>-<case>-*.
# Exit: the checker's status (0 pass); 1 when a container exits early or a wait times out;
#   2 for an unknown simulator or case, `isaac locomotion` or a bad GOLEM_TEST_WALL_TIMEOUT;
#   nonzero with the usage for a missing argument.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SIMULATOR=${1:?Usage: run.sh isaac|robocasa arms|locomotion}
TEST_CASE=${2:?Usage: run.sh isaac|robocasa arms|locomotion}
case "$SIMULATOR" in isaac|robocasa) ;; *) exit 2 ;; esac
case "$TEST_CASE" in arms|locomotion) ;; *) exit 2 ;; esac
if [[ "$SIMULATOR/$TEST_CASE" == isaac/locomotion ]]; then
    echo "run.sh: isaac locomotion cannot pass: Isaac's rt/lowstate tick counts publishes, so the leg node never starts (tests/stack/README.md)" >&2
    exit 2
fi
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-193}
export GOLEM_TEST_WALL_TIMEOUT=${GOLEM_TEST_WALL_TIMEOUT:-900}
# The controllers read no images, and CPU-rendered RoboCasa cameras slow physics several-fold
# while the wall-clock controllers keep real-time pace, which the real robot never has to absorb.
export GOLEM_TEST_CAMERAS=${GOLEM_TEST_CAMERAS:-0}
[[ "$GOLEM_TEST_WALL_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || exit 2
# Set up docker compose, the domain check and the mounts as the wrappers do, for the H1-2 only:
# a G1 selected in the shell or docker/.env does not apply here.
source "$ROOT/docker/scripts/docker_common.sh"
export GOLEM_ROBOT=h12
load_config
check_domain "$SIMULATOR"
export GOLEM_DISPLAY=headless
configure_display
# Pin the RoboCasa sim and the ROS containers to their CPUs, as the wrappers do (not Isaac).
[[ "$SIMULATOR" != robocasa ]] || configure_cpus
prepare_mounts "$SIMULATOR"
prepare_mounts ros

# The run's artifact directory. Root in the containers writes it; 777 lets host users read and
# remove what it writes.
result="$ROOT/tests/results/stack-$SIMULATOR-$TEST_CASE-$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$result/telemetry" "$result/metrics"
chmod 777 "$result/telemetry" "$result/metrics"
prefix="golem-stack-test-${ROS_DOMAIN_ID}-$$"
stack="$prefix-ros"; simulator="$prefix-sim"; checker="$prefix-check"

# Save each container's log, remove the containers, write run.json (git state, image IDs, camera
# setting, exit code) and exit with the run's status.
cleanup() {
    local status=$?
    trap - EXIT
    for pair in "$stack:ros" "$simulator:simulator" "$checker:checker"; do
        docker logs "${pair%:*}" >"$result/${pair##*:}.log" 2>&1 || true
        docker rm -f "${pair%:*}" >/dev/null 2>&1 || true
    done
    python3 - "$result" "$SIMULATOR" "$TEST_CASE" "$status" <<'PYREPORT'
import json, os, pathlib, subprocess, sys
path, simulator, case, status = sys.argv[1:]
def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()
report = dict(simulator=simulator, case=case, exit_code=int(status),
              commit=git('rev-parse', 'HEAD'),
              submodules=git('submodule', 'status', '--recursive'),
              working_tree=git('status', '--short'),
              cameras=os.environ['GOLEM_TEST_CAMERAS'],
              images={name: subprocess.check_output(['docker', 'image', 'inspect', '-f', '{{.Id}}', name], text=True).strip()
                      for name in ('golem_ros:latest', f'golem_sim_{simulator}:latest')})
pathlib.Path(path, 'run.json').write_text(json.dumps(report, indent=2)+'\n')
PYREPORT
    echo "Stack test artifacts: $result"
    exit "$status"
}
trap cleanup EXIT

# Start the ROS container: it builds the test install and prints STACK_BUILD_READY within 600 s.
env_args=(-e "SIMULATOR=$SIMULATOR" -e "TEST_CASE=$TEST_CASE" -e "GOLEM_TEST_WALL_TIMEOUT=$GOLEM_TEST_WALL_TIMEOUT" -e "GOLEM_CAMERAS=$GOLEM_TEST_CAMERAS" -e GOLEM_SIM_RATE=)
tests_mount=(-v "$ROOT/tests/stack:/home/code/tests/stack:ro")
"${COMPOSE[@]}" run -d --no-deps --name "$stack" "${env_args[@]}" "${tests_mount[@]}" --entrypoint bash ros /home/code/tests/stack/launch_ros_stack.sh >/dev/null
for ((attempt=0; attempt<600; attempt++)); do
    if docker logs "$stack" 2>&1 | grep STACK_BUILD_READY >/dev/null; then break; fi
    [[ $(docker inspect -f '{{.State.Running}}' "$stack") == true ]] || { docker logs "$stack"; exit 1; }
    sleep 1
done
docker logs "$stack" 2>&1 | grep STACK_BUILD_READY >/dev/null || { echo 'ROS test build timed out' >&2; exit 1; }

# Copy the sim's own launcher with only its Python entry point replaced by observe_simulator.py,
# which runs the real sim and records its physics privately. No checked-in source changes.
python3 - "$ROOT" "$SIMULATOR" "$result/launcher.sh" <<'PY'
from pathlib import Path
import sys
root, sim, output = sys.argv[1:]
source = (Path(root)/f'docker/scripts/launch_{sim}.sh').read_text()
entry = '/home/code/CL_isaaclab_sim/sim_main.py' if sim == 'isaac' else 'h12_mujoco.py'
old = entry if sim == 'isaac' else f'exec python -u {entry}'
assert source.count(old) == 1, 'Canonical launcher changed; update test instrumentation'
# The copy is mounted at /test_launcher.sh, away from runtime_common.sh, so that path is made absolute.
source = source.replace('"$(dirname "$0")/runtime_common.sh"', '"/home/code/h12_sim_scripts/runtime_common.sh"')
Path(output).write_text(source.replace(old, old.replace(entry, '/home/code/tests/stack/observe_simulator.py')))
PY

# The sim's arguments: headless and seeded; Isaac with the Magpie grippers, RoboCasa in one task scene.
sim_args=(--headless --seed 42)
if [[ "$SIMULATOR" == isaac ]]; then
    sim_args+=(--hand_type magpie)
    # The arm case, the only one Isaac runs, fixes its base to keep the arms apart from balance.
    sim_args+=(--fix_base)
else
    sim_args+=(--task AirDryFruit)
fi

# Start the checker (telemetry read-only, its report into metrics/), then the sim.
"${COMPOSE[@]}" run -d --no-deps --name "$checker" "${env_args[@]}" "${tests_mount[@]}" -v "$result/telemetry:/telemetry:ro" -v "$result/metrics:/results" --entrypoint bash ros /home/code/tests/stack/run_checker.sh >/dev/null
"${COMPOSE[@]}" run -d --no-deps --name "$simulator" "${env_args[@]}" "${tests_mount[@]}" -e GOLEM_TEST_TELEMETRY=/telemetry/state.jsonl -v "$result/telemetry:/telemetry" -v "$result/launcher.sh:/test_launcher.sh:ro" --entrypoint bash "$SIMULATOR" /test_launcher.sh "${sim_args[@]}" >/dev/null

# Wait for the checker to finish; the ROS container or the sim exiting first fails the run.
for ((attempt=0; attempt<GOLEM_TEST_WALL_TIMEOUT+30; attempt++)); do
    [[ $(docker inspect -f '{{.State.Running}}' "$checker") == true ]] || break
    for container in "$stack" "$simulator"; do
        [[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]] || { docker logs "$container"; echo 'Required process exited before test completed' >&2; exit 1; }
    done
    sleep 1
done

# The checker still running after the wait is a timeout; otherwise its exit code is the result.
docker logs "$checker"
[[ $(docker inspect -f '{{.State.Running}}' "$checker") == false ]] || { echo 'Stack checker timed out' >&2; exit 1; }
exit "$(docker inspect -f '{{.State.ExitCode}}' "$checker")"
