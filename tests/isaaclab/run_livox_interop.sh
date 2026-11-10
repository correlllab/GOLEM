#!/usr/bin/env bash
# Check that a Livox CustomMsg written by Isaac's CycloneDDS publisher decodes in ROS Humble.
#
# No GPU simulation: publish_livox_sample.py writes one sample from an Isaac container and
# check_livox_ros.py reads it in a ROS container (subscribe_livox_interop.sh).
# Usage: tests/isaaclab/run_livox_interop.sh
# Environment: ROS_DOMAIN_ID (check_domain refuses 0), docker/.env and the shell.
# Exit: 0 when the subscriber decodes the sample; nonzero otherwise.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

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

# Start the ROS subscriber and wait up to 120 s for it to be ready.
subscriber="isaac-livox-check-${ROS_DOMAIN_ID}-$$"
cleanup() { docker rm -f "$subscriber" >/dev/null 2>&1 || true; }
trap cleanup EXIT
"${COMPOSE[@]}" run -d --no-deps --name "$subscriber" -v "$ROOT/tests:/home/code/tests:ro" --entrypoint bash ros /home/code/tests/isaaclab/subscribe_livox_interop.sh >/dev/null
for attempt in $(seq 1 120); do
    if docker logs "$subscriber" 2>&1 | grep -q 'SUBSCRIBER_READY'; then break; fi
    if [ "$(docker inspect --format '{{.State.Running}}' "$subscriber")" != true ]; then
        docker logs "$subscriber"
        exit 1
    fi
    sleep 1
done
docker logs "$subscriber" 2>&1 | grep -q SUBSCRIBER_READY

# Publish the sample from Isaac, then take the subscriber's exit status as the result.
"$ROOT/tests/isaaclab/run_in_container.sh" 'cd /home/code/CL_isaaclab_sim; PYTHONPATH=$PWD:$PYTHONPATH /isaac-sim/python.sh /home/code/tests/isaaclab/publish_livox_sample.py'
status=$(docker wait "$subscriber")
docker logs "$subscriber"
test "$status" = 0
