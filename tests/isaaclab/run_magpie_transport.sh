#!/usr/bin/env bash
# Run check_magpie_sidecar.py: Isaac's bundled Humble and its native Humble Magpie process on one graph.
#
# No GPU. Fails on a nonzero check or on any serialisation or runtime error in the output.
# Usage: tests/isaaclab/run_magpie_transport.sh
# Environment: as run_in_container.sh.
# Exit: 0 on success; 1 when the output shows a serialisation or runtime error; else the check's status.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
logfile=$(mktemp)
trap 'rm -f "$logfile"' EXIT
# Isaac's CycloneDDS build and its bundled Humble bridge libraries, as the sim runs them.
"$ROOT/tests/isaaclab/run_in_container.sh" 'export LD_LIBRARY_PATH=/workspace/golem/isaac/cyclonedds/install/lib:/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; export CYCLONEDDS_HOME=/workspace/golem/isaac/cyclonedds/install; cd /home/code/CL_isaaclab_sim; PYTHONPATH=$PWD:$PYTHONPATH /isaac-sim/python.sh /home/code/tests/isaaclab/check_magpie_sidecar.py' 2>&1 | tee "$logfile"
# A mismatch between the two ROS runtimes shows only as these messages, not as an exit status.
if grep -E -q 'invalid data size|string data is not null-terminated|rcutils_set_error_state|Traceback|Segmentation fault|Fatal Python error|ModuleNotFoundError' "$logfile"; then
    echo 'FAIL: mixed ROS runtimes reported a serialization/runtime error' >&2
    exit 1
fi
