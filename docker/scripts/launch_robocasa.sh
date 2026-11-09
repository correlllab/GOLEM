#!/bin/bash
# The robocasa container's command: build msgs_ws, then run the RoboCasa
# simulator (golem_robocasa/h12_mujoco.py) of the selected robot.
#
# Usage: launch_robocasa.sh [h12_mujoco.py arguments...]
#   e.g. --headless or --layout 9 --style 9 --seed 42. Without a DISPLAY,
#   --headless is added (no passive viewer).
# Environment: ROS_DOMAIN_ID (1..232, 1 when unset), DISPLAY,
#   GOLEM_PREPARE_ONLY=1 (build msgs_ws and exit 0; docker_stack.sh), and the
#   simulator's own knobs (docker-compose.yml, docker-compose.g1.yml).
# The simulator's topics are listed by its sensor modules (h12_sensors.py and
# magpie_hand_bridge.py, or g1/g1_sensors.py) and CLAUDE.md §8.
# Exit status: 1 on a refused domain; colcon's status when the msgs_ws build
# fails; otherwise the simulator's.
set -e
source "$(dirname "$0")/runtime_common.sh"
validate_sim_domain

source /opt/ros/humble/setup.bash
# livox_ros_driver2's CustomMsg, baked into the image at /opt/livox_ws.
source /opt/livox_ws/install/setup.bash

# Build magpie_msgs and custom_ros_messages, the message packages the
# simulator serves. build/install/log persist on the host in
# container_cache/msgs_ws, so an unchanged rebuild is a quick no-op; delete
# that directory for a clean build.
MSGS_WS=/home/code/msgs_ws
echo "[launch_robocasa] building $MSGS_WS"
(cd "$MSGS_WS" && colcon build --symlink-install \
    --packages-select magpie_msgs custom_ros_messages)
source "$MSGS_WS/install/setup.bash"
[ "${GOLEM_PREPARE_ONLY:-0}" != 1 ] || exit 0

# Run headless when no display is reachable (SSH without X11, CI).
case " $* " in
    *" --headless "*) ;;
    *)
        if [ -z "${DISPLAY:-}" ]; then
            echo "[launch_robocasa] no DISPLAY set — forcing --headless"
            set -- "$@" --headless
        fi
        ;;
esac

# EGL renders the RGBD cameras and the MID-360's rays offscreen on the GPU, needs
# no display, and can render on the sensors' worker threads; the passive viewer
# opens its own GLFW window.
export MUJOCO_GL=egl

cd /home/code/golem_robocasa
# -u: Compose reads stdout through a pipe, which would block-buffer print() and
# hold back the "ROS bridges up" readiness line on a quiet headless run.
exec python -u h12_mujoco.py "$@"
