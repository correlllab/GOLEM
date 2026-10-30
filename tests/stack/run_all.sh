#!/usr/bin/env bash
# Run every case, retaining all failures rather than stopping at the first one.
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
