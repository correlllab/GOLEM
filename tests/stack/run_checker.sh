#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/code/core_ws/install_stack_tests/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp PYTHONUNBUFFERED=1
case "$TEST_CASE" in
    arms) exec python3 /home/code/tests/stack/check_frame_task.py --simulator "$SIMULATOR" --output /results/report.json --wall-timeout "$GOLEM_TEST_WALL_TIMEOUT" ;;
    locomotion) exec python3 /home/code/tests/stack/check_lowerbody.py --telemetry /telemetry/state.jsonl --output /results/report.json --wall-timeout "$GOLEM_TEST_WALL_TIMEOUT" --expected-walk-policy almi ;;
    *) exit 2 ;;
esac
