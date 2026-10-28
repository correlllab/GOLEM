#!/usr/bin/env bash
# Run simulator tests in a disposable container without replacing a user's sim.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/docker/scripts/docker_common.sh"
# H1-2 only: a G1 selected in the shell or docker/.env does not apply here.
export GOLEM_ROBOT=h12
load_config
check_domain robocasa
export GOLEM_DISPLAY=headless
configure_display
prepare_mounts robocasa
exec "${COMPOSE[@]}" run --rm -T --no-deps -v "$ROOT/tests:/home/code/tests:ro" -e MUJOCO_GL=egl robocasa bash -lc \
    'source /opt/ros/humble/setup.bash && source /opt/livox_ws/install/setup.bash && cd /home/code/golem_robocasa && python -m unittest discover -s /home/code/tests/robocasa -v "$@"' bash "$@"
