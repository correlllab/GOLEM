#!/bin/bash
# Usage: ./launch_isaac.sh [task_name] [--headless] [--reset-cache]
# Defaults to task "base" — the only env CL_isaaclab_sim registers at the current
# submodule pin (tasks/h1-2_tasks/stack_rgyblock_h12_27dof_inspire/__init__.py).
# HEADLESS=1 also forces --headless.
#
# ROS publishing uses Isaac-Sim's bundled isaacsim.ros2.bridge extension
# (loaded by Kit at app startup); use its interpreter for simulation imports.
set -e
source "$(dirname "$0")/runtime_common.sh"
validate_sim_domain
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Python DDS and ROS RMW must load the same native Cyclone library.
export CYCLONEDDS_HOME=/workspace/golem/isaac/cyclonedds/install

ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/golem/isaac/IsaacLab}"
ISAAC_PY="${ISAACLAB_PATH}/_isaac_sim/python.sh"
export LD_LIBRARY_PATH=${CYCLONEDDS_HOME}/lib:/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib:${LD_LIBRARY_PATH:-}


TASK="base"
HEADLESS_FLAG=""
[ "${HEADLESS:-0}" = "1" ] && HEADLESS_FLAG="--headless"
# Auto-headless if no display is reachable (SSH without X11, cloud VM, CI).
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
# Use Isaac Sim's bundled Kit interpreter via IsaacLab's python.sh wrapper.
# Compose runs this script non-interactively, so ~/.bashrc (which defines the
# `python` alias) is not sourced, and python.sh is what sets CARB_APP_PATH /
# EXP_PATH / PYTHONPATH / LD_PRELOAD=libcarb.so — a bare `python3` cannot
# import isaacsim.

# Shim for hardcoded developer paths inside CL_isaaclab_sim: src/python/envs/
# common/{scene,robots}.py load USDs from /workspace/golem/isaac/mateo_ws/
# CL_Assets (one dev's personal workspace layout), but compose mounts the assets
# at /home/code/CL_Assets. Symlinked here rather than patched in the submodule —
# the sim packages are owned elsewhere (CLAUDE.md §5/§8). Drop this once the
# asset root is made relative/env-driven upstream.
MATEO_WS=/workspace/golem/isaac/mateo_ws
mkdir -p "$MATEO_WS"
ln -sfn /home/code/CL_Assets "$MATEO_WS/CL_Assets"

exec "$ISAAC_PY" -u \
    /home/code/CL_isaaclab_sim/sim_main.py \
    --device cuda \
    --task "$TASK" \
    $HEADLESS_FLAG \
    --enable_inspire_dds \
    --enable_cameras \
    --robot_type h1_2 "${EXTRA_ARGS[@]}"
