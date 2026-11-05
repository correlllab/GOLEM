#!/bin/bash
# The Mac robocasa container's command: build msgs_ws, then run the RoboCasa
# simulator of the H1-2 (golem_robocasa/h12_mujoco.py) with CPU rendering.
#
# Usage: launch_robocasa_mac.sh [h12_mujoco.py arguments...]
# Environment:
#   ROS_DOMAIN_ID       1..232 (1 when unset); domain 0 is refused
#   GOLEM_DISPLAY       headless (default): OSMesa offscreen, no viewer, and
#                       --headless is added. vnc: the interactive viewer,
#                       rendered with llvmpipe into an in-container Xvfb (:99)
#                       and served over noVNC on 6080
#                       (http://localhost:6080/vnc.html through
#                       docker/mac/scripts/mac_vnc_tunnel.sh); a GL window
#                       cannot be forwarded to XQuartz on Apple Silicon.
#   VNC_GEOMETRY        that display's WxHxD (default 1280x800x24)
#   GOLEM_PREPARE_ONLY  1: build msgs_ws and exit 0 (docker_stack.sh)
#   the simulator's knobs (docker/mac/docker-compose.yml)
# Exit status: 1 on a refused domain or when Xvfb does not start; colcon's
# status when the msgs_ws build fails; otherwise the simulator's.
set -e
source "/home/code/golem_common/runtime_common.sh"
validate_sim_domain

GOLEM_DISPLAY="${GOLEM_DISPLAY:-headless}"

# The viewer's VNC display and ports (GOLEM_DISPLAY=vnc), bound to localhost in
# the Colima VM. Colima does not forward host-network container ports, so the
# Mac reaches them through mac_vnc_tunnel.sh's SSH tunnel.
VNC_DISPLAY=:99
VNC_PORT=5900
NOVNC_PORT=6080
VNC_GEOMETRY="${VNC_GEOMETRY:-1280x800x24}"

# Start Xvfb, fluxbox, x11vnc and websockify in the background and point
# DISPLAY at the Xvfb display; exit 1 when Xvfb does not come up within 15 s.
# cleanup_vnc_stack stops them on exit.
start_vnc_stack() {
    echo "[launch_robocasa_mac] starting VNC stack on $VNC_DISPLAY ($VNC_GEOMETRY)"
    # `docker restart` keeps the container's filesystem: clear a previous run's
    # X lock and socket, or Xvfb aborts with "Server is already active".
    dnum="${VNC_DISPLAY#:}"
    pkill -f "Xvfb ${VNC_DISPLAY}" 2>/dev/null || true
    sleep 0.3
    rm -f "/tmp/.X${dnum}-lock" "/tmp/.X11-unix/X${dnum}" 2>/dev/null || true
    Xvfb "$VNC_DISPLAY" -screen 0 "$VNC_GEOMETRY" +extension GLX +render -noreset \
        >/tmp/xvfb.log 2>&1 &
    export DISPLAY="$VNC_DISPLAY"
    for _ in $(seq 1 30); do
        xdpyinfo -display "$VNC_DISPLAY" >/dev/null 2>&1 && break
        sleep 0.5
    done
    xdpyinfo -display "$VNC_DISPLAY" >/dev/null 2>&1 \
        || { echo "[launch_robocasa_mac] Xvfb failed to start"; cat /tmp/xvfb.log; exit 1; }

    fluxbox >/tmp/fluxbox.log 2>&1 &
    sleep 1

    # -localhost: VNC accepts connections only from inside the VM, where the
    # SSH tunnel ends; that is why -nopw (no password) is acceptable.
    x11vnc -display "$VNC_DISPLAY" -rfbport "$VNC_PORT" -localhost \
        -forever -shared -nopw -quiet -bg >/tmp/x11vnc.log 2>&1
    sleep 1

    websockify --web /usr/share/novnc "127.0.0.1:$NOVNC_PORT" "localhost:$VNC_PORT" \
        >/tmp/websockify.log 2>&1 &
    sleep 1

    echo "[launch_robocasa_mac] noVNC ready. From the Mac run mac_vnc_tunnel.sh, then open:"
    echo "[launch_robocasa_mac]   http://localhost:${NOVNC_PORT}/vnc.html?autoconnect=1&resize=scale"
}

# Stop the VNC stack's processes.
cleanup_vnc_stack() {
    pkill -f "websockify .*${NOVNC_PORT}" 2>/dev/null || true
    pkill -x x11vnc 2>/dev/null || true
    pkill -x fluxbox 2>/dev/null || true
    pkill -f "Xvfb ${VNC_DISPLAY}" 2>/dev/null || true
}

source /opt/ros/humble/setup.bash
# livox_ros_driver2's CustomMsg for h12_sensors.py, baked into the image at
# /opt/livox_ws.
source /opt/livox_ws/install/setup.bash

# Build the message-only msgs_ws (magpie_msgs, custom_ros_messages). build/
# and install/ are named volumes on the VM's own disk: a --symlink-install tree
# written onto the virtiofs bind mount stalls for minutes. They persist, so an
# unchanged rebuild is a no-op.
MSGS_WS=/home/code/msgs_ws
MSGS_BUILD=/opt/msgs_ws/build
MSGS_INSTALL=/opt/msgs_ws/install
echo "[launch_robocasa_mac] building $MSGS_WS -> $MSGS_INSTALL"
(cd "$MSGS_WS" && colcon build --symlink-install \
    --build-base "$MSGS_BUILD" --install-base "$MSGS_INSTALL" \
    --packages-select magpie_msgs custom_ros_messages)
source "$MSGS_INSTALL/setup.bash"
[ "${GOLEM_PREPARE_ONLY:-0}" != 1 ] || exit 0

if [ "$GOLEM_DISPLAY" = "vnc" ]; then
    # The windowed viewer (GLFW) on the Xvfb display, rendered with llvmpipe
    # (no GPU under Colima); no --headless.
    trap cleanup_vnc_stack EXIT
    start_vnc_stack
    export LIBGL_ALWAYS_SOFTWARE=1
    export MUJOCO_GL=glfw
else
    # OSMesa software rendering offscreen, and --headless: the VM has no X
    # display.
    export MUJOCO_GL=osmesa
    case " $* " in
        *" --headless "*) ;;
        *) set -- "$@" --headless ;;
    esac
fi

cd /home/code/golem_robocasa
echo "[launch_robocasa_mac] GOLEM_DISPLAY=$GOLEM_DISPLAY MUJOCO_GL=$MUJOCO_GL ROS_DOMAIN_ID=$ROS_DOMAIN_ID args: $*"
python -u h12_mujoco.py "$@"
