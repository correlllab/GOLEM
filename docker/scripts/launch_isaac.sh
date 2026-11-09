#!/bin/bash
# The isaac container's command: run CL_isaaclab_sim's sim_main.py (the H1-2)
# with Isaac Sim's bundled Python.
#
# Usage: launch_isaac.sh [TASK | --task TASK] [--headless] [--reset-cache] [-- sim_main.py args...]
#   TASK           the gym task id (default: base, the one CL_isaaclab_sim
#                  registers, in tasks/h1-2_tasks/stack_rgyblock_h12_27dof_inspire)
#   --headless     no viewer; added anyway without a DISPLAY
#   --reset-cache  delete Isaac's texture cache first
#   --             pass the rest to sim_main.py; so does any other --option,
#                  with everything after it (e.g. --hand_type inspire, --fix_base)
# Environment: ROS_DOMAIN_ID (1..232, 1 when unset), DISPLAY, ISAACLAB_PATH
#   (default /workspace/golem/isaac/IsaacLab).
# Exit status: 1 on a refused domain, 2 on --task without a name; otherwise
# the simulator's.
set -e
source "$(dirname "$0")/runtime_common.sh"
validate_sim_domain
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# The Unitree Python DDS binding and the ROS RMW must load the same native
# CycloneDDS library: the image's build, ahead of Isaac's bundled Humble bridge
# libraries, which the simulator publishes ROS topics through.
export CYCLONEDDS_HOME=/workspace/golem/isaac/cyclonedds/install

ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/golem/isaac/IsaacLab}"
ISAAC_PY="${ISAACLAB_PATH}/_isaac_sim/python.sh"
export LD_LIBRARY_PATH=${CYCLONEDDS_HOME}/lib:/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib:${LD_LIBRARY_PATH:-}


TASK="base"
HEADLESS_FLAG=""
# Run headless when no display is reachable (SSH without X11, CI).
if [ -z "$HEADLESS_FLAG" ] && [ -z "${DISPLAY:-}" ]; then
    echo "[launch_isaac] no DISPLAY set — forcing --headless"
    HEADLESS_FLAG="--headless"
fi

EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --reset-cache) rm -rf "$HOME/.cache/ov/texturecache"; shift ;;
        --headless) HEADLESS_FLAG="--headless"; shift ;;
        --task)
            [ "$#" -ge 2 ] || { echo '--task needs a task name' >&2; exit 2; }
            TASK=$2; shift 2 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        --*) EXTRA_ARGS+=("$@"); break ;;
        *) TASK=$1; shift ;;
    esac
done

export PYTHONUNBUFFERED=1
# The simulator runs under IsaacLab's python.sh, Isaac Sim's Kit interpreter:
# python.sh sets CARB_APP_PATH, EXP_PATH, PYTHONPATH and LD_PRELOAD, without
# which isaacsim does not import. The .bashrc `python` alias is not available
# here, as Compose runs this script non-interactively.

# CL_isaaclab_sim's src/python/envs/common/{scene,robots}.py load their USDs
# from /workspace/golem/isaac/mateo_ws/CL_Assets, while the assets are mounted
# at /home/code/CL_Assets: link the one to the other. TODO(max): drop the link
# once CL_isaaclab_sim takes its asset root from the environment.
MATEO_WS=/workspace/golem/isaac/mateo_ws
mkdir -p "$MATEO_WS"
ln -sfn /home/code/CL_Assets "$MATEO_WS/CL_Assets"

exec "$ISAAC_PY" -u \
    /home/code/CL_isaaclab_sim/sim_main.py \
    --device cuda \
    --task "$TASK" \
    $HEADLESS_FLAG \
    --enable_cameras \
    --robot_type h1_2 "${EXTRA_ARGS[@]}"
