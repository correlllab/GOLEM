#!/usr/bin/env bash
# Run the G1 sim's unit tests in a disposable robocasa container with the G1 overlay's mounts.
#
# Usage: tests/robocasa/g1/run_in_container.sh [test_g1_*.py ...]
#   With no files, runs every test_g1_*.py here. A running sim container is left alone.
# The container has no network and uses DDS domain 97, whatever ROS_DOMAIN_ID says:
#   test_g1_unitree_interface.py opens real Unitree DDS channels.
# Environment: docker/.env and the shell through load_config; the sim knobs listed below are
#   cleared.
# Exit: 0 when every file passes; 1 when any fails or a wrapper check fails; 2 for an argument
#   that is not a test_g1_*.py file here.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)

# Accept only test files that exist here.
for name in "$@"; do
    case "$name" in test_g1_*.py) [ -f "$ROOT/tests/robocasa/g1/$name" ] || { echo "No tests/robocasa/g1/$name" >&2; exit 2; } ;;
                    *) echo "Usage: run_in_container.sh [test_g1_*.py ...]" >&2; exit 2 ;; esac
done

# The tests expect each knob's default, so a value from the shell or docker/.env is cleared
# (an interface named in G1_DDS_INTERFACE does not exist in the container).
for knob in GOLEM_CAMERAS GOLEM_CAMERA_SCALE GOLEM_CAMERA_HZ GOLEM_LIDAR GOLEM_LIDAR_POINTS \
            GOLEM_LIDAR_HZ GOLEM_LIDAR_BACKEND GOLEM_CMD_TIMEOUT GOLEM_SPAWN_BACKOFF \
            GOLEM_SIM_ODOM GOLEM_SIM_EXPERIMENT GOLEM_SIM_RATE G1_DDS_INTERFACE; do
    export "$knob="
done

# Set up docker compose with the G1 overlay and the G1 mounts, headless, on domain 97, whatever
# the shell or docker/.env selects.
source "$ROOT/docker/scripts/docker_common.sh"
export GOLEM_ROBOT=g1 ROS_DOMAIN_ID=97
load_config
configure_robot robocasa
check_domain robocasa
export GOLEM_DISPLAY=headless
configure_display
prepare_mounts robocasa

# A compose overlay that takes the container off every network.
no_network=$(mktemp "${TMPDIR:-/tmp}/golem-g1-unit-XXXXXX.yml")
trap 'rm -f "$no_network"' EXIT
printf 'services:\n  robocasa:\n    network_mode: none\n' > "$no_network"
COMPOSE+=(-f "$no_network")

# Run each file in turn and report the failures. golem_slam's configs are mounted because
# test_g1_sensors.py checks the lidar's IMU point against its FAST_LIO config.
"${COMPOSE[@]}" run --rm -T --no-deps \
    -v "$ROOT/tests:/home/code/tests:ro" \
    -v "$ROOT/core_ws/src/golem_slam/config:/home/code/core_ws/src/golem_slam/config:ro" \
    -e MUJOCO_GL=egl -e PYTHONDONTWRITEBYTECODE=1 robocasa bash -lc '
source /opt/ros/humble/setup.bash && source /opt/livox_ws/install/setup.bash
export PYTHONPATH=/home/code/golem_robocasa${PYTHONPATH:+:$PYTHONPATH}
cd /home/code/tests/robocasa/g1
[ "$#" -gt 0 ] || set -- test_g1_*.py
failed=()
for test in "$@"; do
    echo "=== $test"
    python3 -u "$test" || failed+=("$test")
done
if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo "All $# G1 sim test files passed"' bash "$@"
