#!/usr/bin/env bash
# Disposable test container using GOLEM's normal mounts and domain checks.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/docker/scripts/docker_common.sh"
load_config
check_domain isaac
export GOLEM_DISPLAY=headless
configure_display
prepare_mounts isaac
if [ "$#" = 0 ]; then
    set -- 'cd /home/code/CL_isaaclab_sim && PYTHONDONTWRITEBYTECODE=1 /isaac-sim/python.sh -m unittest discover -s /home/code/tests/isaaclab -p "test_*.py" -v'
fi
"${COMPOSE[@]}" run --rm -T --no-deps -v "$ROOT/tests:/home/code/tests:ro" --entrypoint bash isaac -lc "$*"
