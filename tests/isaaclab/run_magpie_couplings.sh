#!/usr/bin/env bash
# Real USD-schema validation without starting Kit or allocating GPU simulation.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
exec "$ROOT/tests/isaaclab/run_in_container.sh" '
set -e
usd_dirs=(/isaac-sim/extscache/omni.usd.libs-*.cp311)
physx_dirs=(/isaac-sim/extscache/omni.usd.schema.physx-*.cp311.*)
usd_schema=${usd_dirs[0]}
physx_schema=${physx_dirs[0]}
export PYTHONPATH="$usd_schema:$physx_schema:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$usd_schema/bin:$physx_schema/bin:${LD_LIBRARY_PATH:-}"
export PXR_PLUGINPATH_NAME="$physx_schema/plugins/PhysxSchema/resources:${PXR_PLUGINPATH_NAME:-}"
/isaac-sim/python.sh -m unittest discover -s /home/code/tests/isaaclab -p test_magpie_couplings.py -v'
