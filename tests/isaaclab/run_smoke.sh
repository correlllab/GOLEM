#!/usr/bin/env bash
# Run the real Isaac Lab scene headless for 50 steps and check it shuts down normally (GPU).
#
# Usage: tests/isaaclab/run_smoke.sh
# Environment: ROS_DOMAIN_ID (default 187; check_domain refuses 0), docker/.env and the shell.
# Writes tests/results/isaac-smoke.log.
# Exit: 0 when the run completes 50 steps without ROS errors; nonzero otherwise, including a
#   300 s timeout.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$ROOT/tests/results"
# SIGINT at 300 s, then SIGKILL 30 s later, bound a run that hangs.
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-187} "$ROOT/tests/isaaclab/run_in_container.sh" \
 'timeout --signal=INT --kill-after=30s 300s /home/code/h12_sim_scripts/launch_isaac.sh --headless --max_steps 50 --enable_profiling --profile_interval 25' \
 > "$ROOT/tests/results/isaac-smoke.log" 2>&1
# Only a run that reaches its normal end prints the completion line; a timeout does not.
grep -q '\[isaac\] completed 50 steps' "$ROOT/tests/results/isaac-smoke.log"
# ROS discovery and deserialisation errors show only in the log.
if grep -Eq 'invalid data size|string data is not null-terminated|rcutils_set_error_state' "$ROOT/tests/results/isaac-smoke.log"; then
    echo 'ROS discovery or deserialization failed; see tests/results/isaac-smoke.log' >&2
    exit 1
fi
