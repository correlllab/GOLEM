#!/usr/bin/env bash
# No GPU simulation: Isaac CycloneDDS writer -> native Humble ROS subscriber.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/docker/scripts/docker_common.sh"
load_config
check_domain isaac
export GOLEM_DISPLAY=headless
configure_display
prepare_mounts isaac
prepare_mounts ros
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
"$ROOT/tests/isaaclab/run_in_container.sh" 'cd /home/code/CL_isaaclab_sim; PYTHONPATH=$PWD:$PYTHONPATH /isaac-sim/python.sh /home/code/tests/isaaclab/publish_livox_sample.py'
status=$(docker wait "$subscriber")
docker logs "$subscriber"
test "$status" = 0
