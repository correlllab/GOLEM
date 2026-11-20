#!/usr/bin/env bash
# Check the Isaac sim's elastic band holds, releases and re-holds the floating H1-2 (GPU).
#
# check_elastic_band.py runs in place of sim_main.py, under a copy of launch_isaac.sh, with
# --elastic_band for 50 s of sim time, and calls /elastic_band/toggle twice over ROS.
# Usage: tests/isaaclab/run_elastic_band.sh
# Environment: ROS_DOMAIN_ID (default 185; check_domain refuses 0), docker/.env and the shell.
# Writes tests/results/isaac-elastic-band.log.
# Exit: 0 when every check passes; nonzero otherwise, including a 900 s timeout.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$ROOT/tests/results"
LOG="$ROOT/tests/results/isaac-elastic-band.log"
# The launcher's environment (the shared CycloneDDS library, Isaac's Python) with the check as
# its entry point; SIGINT at 900 s, then SIGKILL 30 s later, bound a run that hangs.
status=0
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-185} "$ROOT/tests/isaaclab/run_in_container.sh" \
 'sed -e "s#/home/code/CL_isaaclab_sim/sim_main.py#/home/code/tests/isaaclab/check_elastic_band.py#" \
      -e "s#\"\$(dirname \"\$0\")/runtime_common.sh\"#/home/code/h12_sim_scripts/runtime_common.sh#" \
      /home/code/h12_sim_scripts/launch_isaac.sh > /tmp/launch_check.sh &&
  timeout --signal=INT --kill-after=30s 900s bash /tmp/launch_check.sh --headless' \
 > "$LOG" 2>&1 || status=$?
# The checks' lines, then the verdict: only a run that reaches its last step prints PASSED.
[ "$status" = 0 ] || echo "run_elastic_band: the sim exited with status $status" >&2
grep -E '^(PASS|FAIL|PASSED|FAILED)' "$LOG" || true
grep -q '^PASSED: ' "$LOG"
