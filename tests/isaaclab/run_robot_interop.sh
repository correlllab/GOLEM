#!/usr/bin/env bash
# Drive the Isaac H1-2 from native ROS and check its state, hands and sensors (GPU).
#
# check_robot_ros.py in a ROS container commands a wrist and the hands and checks the measured
# motion and every sensor topic, against an 800-step, fixed-base Isaac sim.
# Usage: tests/isaaclab/run_robot_interop.sh [magpie|inspire]
# Environment: ISAAC_HAND_TYPE (default hand when no argument; magpie), ROS_DOMAIN_ID (default
#   191; check_domain refuses 0), docker/.env and the shell.
# Exit: the checker's status (0 pass); 1 when the sim ends first, the checker exceeds 300 s or a
#   container fails to start; 2 for an unknown hand type.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
hand_type="${1:-${ISAAC_HAND_TYPE:-magpie}}"
case "$hand_type" in magpie|inspire) ;; *) echo 'Expected magpie or inspire' >&2; exit 2 ;; esac
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-191}"

# Set up docker compose and the mounts for the H1-2 only: a G1 selected in the shell or
# docker/.env does not apply here.
source "$ROOT/docker/scripts/docker_common.sh"
export GOLEM_ROBOT=h12
load_config
check_domain isaac
export GOLEM_DISPLAY=headless
configure_display
prepare_mounts isaac
prepare_mounts ros

# Start the ROS checker and wait up to 180 s for it to be ready.
subscriber="isaac-robot-check-${ROS_DOMAIN_ID}-$$"
simulator="isaac-robot-sim-${ROS_DOMAIN_ID}-$$"
cleanup() { docker rm -f "$subscriber" "$simulator" >/dev/null 2>&1 || true; }
trap cleanup EXIT
"${COMPOSE[@]}" run -d --no-deps --name "$subscriber" -e "ISAAC_HAND_TYPE=$hand_type" -v "$ROOT/tests:/home/code/tests:ro" --entrypoint bash ros /home/code/tests/isaaclab/subscribe_robot_interop.sh >/dev/null
for attempt in $(seq 1 180); do
    if docker logs "$subscriber" 2>&1 | grep -q SUBSCRIBER_READY; then break; fi
    if [ "$(docker inspect --format '{{.State.Running}}' "$subscriber")" != true ]; then docker logs "$subscriber"; exit 1; fi
    sleep 1
done
docker logs "$subscriber" 2>&1 | grep -q SUBSCRIBER_READY

# Start the sim with a fixed base, so the wrist check needs no leg controller, and wait up to
# 300 s for the checker; the sim ending first fails the run.
"${COMPOSE[@]}" run -d --no-deps --name "$simulator" --entrypoint bash isaac /home/code/h12_sim_scripts/launch_isaac.sh --headless --hand_type "$hand_type" --fix_base --max_steps 800 >/dev/null
for attempt in $(seq 1 300); do
    if [ "$(docker inspect --format '{{.State.Running}}' "$subscriber")" != true ]; then break; fi
    if [ "$(docker inspect --format '{{.State.Running}}' "$simulator")" != true ]; then
        docker logs "$simulator"
        docker logs "$subscriber"
        echo 'Simulator ended before ROS interoperability check completed' >&2
        exit 1
    fi
    sleep 1
done

# The checker's exit status is the result.
docker logs "$subscriber"
if [ "$(docker inspect --format '{{.State.Running}}' "$subscriber")" = true ]; then
    docker logs "$simulator"
    echo 'ROS interoperability check exceeded 300 seconds' >&2
    exit 1
fi
status=$(docker inspect --format '{{.State.ExitCode}}' "$subscriber")
if [ "$status" != 0 ]; then docker logs "$simulator"; exit "$status"; fi
