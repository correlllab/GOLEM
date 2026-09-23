#!/bin/bash
# Usage: ./launch_robocasa.sh [--headless]
# Windowed MuJoCo sim by default; pass --headless to disable the passive viewer
# (auto-applied when no DISPLAY is reachable, e.g. SSH without X11 / CI).
# The sim publishes rt/lowstate over CycloneDDS plus its camera / lidar / clock /
# gripper topics on ROS 2 (ROS_DOMAIN_ID=1 unless overridden). mujoco_ros_bridge.py
# is the authority on that list.
set -e
source "$(dirname "$0")/runtime_common.sh"
validate_sim_domain

source /opt/ros/humble/setup.bash
# livox_ros_driver2 (CustomMsg/CustomPoint) is baked into the robocasa image at
# /opt/livox_ws by RobocasaDockerfile. Source it so mujoco_ros_bridge.py can
# import livox_ros_driver2.msg.
source /opt/livox_ws/install/setup.bash

# Run colcon every launch — it's a fast no-op when nothing's changed.
# build/install/log are bind-mounted from the host at container_cache/msgs_ws/
# (see docker-compose.yml), so they persist across `docker compose run --rm`
# cycles. Wipe that host directory if you need a clean rebuild.
MSGS_WS=/home/code/msgs_ws
echo "[launch_robocasa] building $MSGS_WS"
(cd "$MSGS_WS" && colcon build --symlink-install \
    --packages-select magpie_msgs custom_ros_messages)
source "$MSGS_WS/install/setup.bash"
[ "${GOLEM_PREPARE_ONLY:-0}" != 1 ] || exit 0

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

# Auto-add --headless if no display is reachable (SSH without X11, cloud VM, CI).
case " $* " in
    *" --headless "*) ;;
    *)
        if [ -z "${DISPLAY:-}" ]; then
            echo "[launch_robocasa] no DISPLAY set — forcing --headless"
            set -- "$@" --headless
        fi
        ;;
esac

# MUJOCO_GL selects the offscreen renderer for the RGBD cameras. EGL needs no
# display and can render on the camera worker thread; the passive viewer opens
# its own GLFW window independently of it.
export MUJOCO_GL=egl

cd /home/code/h1_robocasa
# -u: docker-compose captures stdout via a pipe, which block-buffers python's
# print()s — the "[h12_mujoco] ROS bridges up" readiness line otherwise sits in
# the buffer indefinitely on a quiet headless run.
exec python -u h12_mujoco.py "$@"
