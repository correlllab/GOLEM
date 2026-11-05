#!/usr/bin/env bash
# Run a command in a disposable, headless Isaac container with the wrappers' mounts and domain check.
#
# Usage: tests/isaaclab/run_in_container.sh [shell command]
#   With no command, runs every tests/isaaclab/test_*.py with Isaac's Python.
#   The command runs under `bash -lc` in /, with tests/ mounted read-only at /home/code/tests.
# Environment: ROS_DOMAIN_ID (check_domain refuses 0), and docker/.env and the shell through
#   docker_common.sh's load_config.
# Exit: the command's status; 1 when a wrapper check fails.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# Set up docker compose, the domain check, a headless display and the Isaac mounts.
source "$ROOT/docker/scripts/docker_common.sh"
load_config
check_domain isaac
export GOLEM_DISPLAY=headless
configure_display
prepare_mounts isaac

# Run the command, by default the CPU unit tests.
if [ "$#" = 0 ]; then
    set -- 'cd /home/code/CL_isaaclab_sim && PYTHONDONTWRITEBYTECODE=1 /isaac-sim/python.sh -m unittest discover -s /home/code/tests/isaaclab -p "test_*.py" -v'
fi
"${COMPOSE[@]}" run --rm -T --no-deps -v "$ROOT/tests:/home/code/tests:ro" --entrypoint bash isaac -lc "$*"
