#!/usr/bin/env bash
# Run the H1-2 RoboCasa sim's unit tests (tests/robocasa/test_*.py) in a disposable container.
#
# Usage: tests/robocasa/run_in_container.sh [unittest discover options]
#   e.g. -p test_sensor_bridge.py to run one file. A running sim container is left alone.
# Environment: ROS_DOMAIN_ID (check_domain refuses 0), GOLEM_CPU_PINNING and GOLEM_SIM_CPUSET
#   (configure_cpus), docker/.env and the shell through docker_common.sh's load_config.
# Exit: the unittest run's status; 1 when a wrapper check fails.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# Set up docker compose and the mounts for the H1-2 only, headless: a G1 selected in the shell
# or docker/.env does not apply here.
source "$ROOT/docker/scripts/docker_common.sh"
export GOLEM_ROBOT=h12
load_config
check_domain robocasa
export GOLEM_DISPLAY=headless
configure_display
# Pin the container to the sim's CPUs, as the sims run (docker_common.sh's configure_cpus).
configure_cpus
prepare_mounts robocasa

# EGL renders the cameras offscreen; the Livox workspace provides the CustomMsg type.
exec "${COMPOSE[@]}" run --rm -T --no-deps -v "$ROOT/tests:/home/code/tests:ro" -e MUJOCO_GL=egl robocasa bash -lc \
    'source /opt/ros/humble/setup.bash && source /opt/livox_ws/install/setup.bash && cd /home/code/golem_robocasa && python -m unittest discover -s /home/code/tests/robocasa -v "$@"' bash "$@"
