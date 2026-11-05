#!/usr/bin/env bash
# Run every stack case: arms and locomotion against RoboCasa, then against Isaac.
#
# Usage: tests/stack/run_all.sh
# Environment: whatever run.sh reads (ROS_DOMAIN_ID, GOLEM_TEST_WALL_TIMEOUT, GOLEM_TEST_CAMERAS).
# Exit: 0 when every case passes, 1 when any fails. Every case runs, even after a failure.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
status=0
for simulator in robocasa isaac; do
    for case_name in arms locomotion; do
        if ! "$ROOT/tests/stack/run.sh" "$simulator" "$case_name"; then
            status=1
        fi
    done
done
exit "$status"
