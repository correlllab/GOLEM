#!/usr/bin/env bash
# Mixed Isaac bundled-Humble and native-Humble graph regression, without GPU.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
logfile=$(mktemp)
trap 'rm -f "$logfile"' EXIT
"$ROOT/tests/isaaclab/run_in_container.sh" 'export LD_LIBRARY_PATH=/workspace/golem/isaac/cyclonedds/install/lib:/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; export CYCLONEDDS_HOME=/workspace/golem/isaac/cyclonedds/install; cd /home/code/CL_isaaclab_sim; PYTHONPATH=$PWD:$PYTHONPATH /isaac-sim/python.sh /home/code/tests/isaaclab/check_magpie_sidecar.py' 2>&1 | tee "$logfile"
if grep -E -q 'invalid data size|string data is not null-terminated|rcutils_set_error_state|Traceback|Segmentation fault|Fatal Python error|ModuleNotFoundError' "$logfile"; then
    echo 'FAIL: mixed ROS runtimes reported a serialization/runtime error' >&2
    exit 1
fi
