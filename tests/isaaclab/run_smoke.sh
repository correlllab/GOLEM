#!/usr/bin/env bash
# NVIDIA integration: actual Isaac Lab construction, physics, sensors and cleanup.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$ROOT/tests/results"
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-187} "$ROOT/tests/isaaclab/run_in_container.sh" \
 'timeout --signal=INT --kill-after=30s 300s /home/code/h12_sim_scripts/launch_isaac.sh --headless --max_steps 50 --enable_profiling --profile_interval 25' \
 > "$ROOT/tests/results/isaac-smoke.log" 2>&1
# A successful bounded run must reach normal cleanup; timeouts are failures.
grep -q '\[isaac\] completed 50 steps' "$ROOT/tests/results/isaac-smoke.log"
