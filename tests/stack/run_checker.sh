#!/usr/bin/env bash
# Run one stack case's checker inside the checker container that run.sh starts.
#
# Usage (from run.sh): bash /home/code/tests/stack/run_checker.sh
# Environment: TEST_CASE (arms|locomotion), SIMULATOR (isaac|robocasa), GOLEM_TEST_WALL_TIMEOUT (s).
# Reads /telemetry/state.jsonl (locomotion) and writes /results/report.json.
# Exit: the checker's status (0 pass, nonzero fail); 2 for an unknown TEST_CASE.
set -eo pipefail
# The test install that launch_ros_stack.sh built, with its message packages.
source /opt/ros/humble/setup.bash
source /home/code/core_ws/install_stack_tests/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp PYTHONUNBUFFERED=1
case "$TEST_CASE" in
    arms) exec python3 /home/code/tests/stack/check_frame_task.py --simulator "$SIMULATOR" --output /results/report.json --wall-timeout "$GOLEM_TEST_WALL_TIMEOUT" ;;
    locomotion) exec python3 /home/code/tests/stack/check_lowerbody.py --telemetry /telemetry/state.jsonl --output /results/report.json --wall-timeout "$GOLEM_TEST_WALL_TIMEOUT" --expected-walk-policy almi ;;
    *) exit 2 ;;
esac
