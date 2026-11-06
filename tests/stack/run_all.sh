#!/usr/bin/env bash
# Run every stack case that can pass: arms and locomotion against RoboCasa, then arms against
# Isaac. Isaac's locomotion is skipped: its rt/lowstate tick counts publishes, not ms of the
# node's clock, so the leg node never starts there, and Isaac serves no /elastic_band/toggle.
#
# Usage: tests/stack/run_all.sh
# Environment: whatever run.sh reads (ROS_DOMAIN_ID, GOLEM_TEST_WALL_TIMEOUT, GOLEM_TEST_CAMERAS).
# Exit: 0 when every case passes, 1 when any fails. Every case runs, even after a failure.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
status=0
for simulator in robocasa isaac; do
    for case_name in arms locomotion; do
        if [ "$simulator/$case_name" = isaac/locomotion ]; then
            echo "run_all: skipping isaac locomotion (tests/stack/README.md)"
            continue
        fi
        if ! "$ROOT/tests/stack/run.sh" "$simulator" "$case_name"; then
            status=1
        fi
    done
done
exit "$status"
