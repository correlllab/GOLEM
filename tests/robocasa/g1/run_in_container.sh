#!/usr/bin/env bash
# Runs the G1 sim's unit tests in a disposable robocasa container with the G1
# overlay's mounts, leaving a running simulator alone.
#   run_in_container.sh                        every test_g1_*.py here
#   run_in_container.sh test_g1_sensors.py ... the named files
# The container has no network and uses DDS domain 97: test_g1_unitree_interface.py
# opens real Unitree DDS channels. Exits 1 if any file fails.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
for name in "$@"; do
    case "$name" in test_g1_*.py) [ -f "$ROOT/tests/robocasa/g1/$name" ] || { echo "No tests/robocasa/g1/$name" >&2; exit 2; } ;;
                    *) echo "Usage: run_in_container.sh [test_g1_*.py ...]" >&2; exit 2 ;; esac
done
# The tests use each knob's default: a value set in the shell or docker/.env
# must not change them (an interface in G1_DDS_INTERFACE does not exist here).
for knob in GOLEM_CAMERAS GOLEM_CAMERA_SCALE GOLEM_CAMERA_HZ GOLEM_LIDAR GOLEM_LIDAR_POINTS \
            GOLEM_LIDAR_HZ GOLEM_CMD_TIMEOUT GOLEM_ARM_SDK_TIMEOUT GOLEM_SPAWN_BACKOFF \
            GOLEM_SIM_ODOM GOLEM_SIM_EXPERIMENT G1_DDS_INTERFACE; do
    export "$knob="
done
source "$ROOT/docker/scripts/docker_common.sh"
# G1 only, whatever the shell or docker/.env selects; 97 is the G1 DDS unit tests' domain.
export GOLEM_ROBOT=g1 ROS_DOMAIN_ID=97
load_config
configure_robot robocasa
check_domain robocasa
export GOLEM_DISPLAY=headless
configure_display
prepare_mounts robocasa
no_network=$(mktemp "${TMPDIR:-/tmp}/golem-g1-unit-XXXXXX.yml")
trap 'rm -f "$no_network"' EXIT
printf 'services:\n  robocasa:\n    network_mode: none\n' > "$no_network"
COMPOSE+=(-f "$no_network")
# test_g1_sensors.py checks the lidar's IMU point against g1_navigation's FAST_LIO config.
"${COMPOSE[@]}" run --rm -T --no-deps \
    -v "$ROOT/tests:/home/code/tests:ro" \
    -v "$ROOT/core_ws/src/g1/g1_navigation/config:/home/code/core_ws/src/g1/g1_navigation/config:ro" \
    -e MUJOCO_GL=egl -e PYTHONDONTWRITEBYTECODE=1 robocasa bash -lc '
source /opt/ros/humble/setup.bash && source /opt/livox_ws/install/setup.bash
export PYTHONPATH=/home/code/h1_robocasa${PYTHONPATH:+:$PYTHONPATH}
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
