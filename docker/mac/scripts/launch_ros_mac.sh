#!/bin/bash
# The Mac ros container's command: build the packages the Mac bringup runs
# (not the vision / grasp ML stack, MJPC or the G1's), then start
# h12_sim_mac.launch.py against the running simulator.
#
# Usage: launch_ros_mac.sh [command...]
#   command  run it instead of the bringup once the build is sourced (e.g. bash)
# Environment:
#   ROS_DOMAIN_ID       1..232 (1 when unset); domain 0 is refused
#   GOLEM_RVIZ          vnc or 1: RViz2 rendered with llvmpipe into its own Xvfb
#                       (:100) and served over noVNC on 6081
#                       (http://localhost:6081/vnc.html through
#                       docker/mac/scripts/mac_vnc_tunnel.sh)
#   RVIZ_GEOMETRY       that display's WxHxD (default 1600x900x24)
#   RVIZ_CONFIG         the .rviz file (default golem_bringup's rviz/h12/main.rviz)
#   GOLEM_ROS_MCP       1: start tools/ros_mcp_server.py, the ROS debugging MCP
#                       server, on localhost:GOLEM_ROS_MCP_PORT (default 6082)
#   GOLEM_WAIT_FOR_SIM  1: run wait_for_sim.py before the bringup (the stack)
#   GOLEM_LOWERBODY, GOLEM_SLAM, GOLEM_NAV2: read by h12_sim_mac.launch.py
# Exit status: 1 on a refused domain or without the baked livox_ros_driver2;
# colcon's or wait_for_sim.py's status when they fail; otherwise the bringup's
# or the command's.
set -e
source /home/code/golem_common/runtime_common.sh
validate_sim_domain

GOLEM_RVIZ="${GOLEM_RVIZ:-0}"
GOLEM_ROS_MCP="${GOLEM_ROS_MCP:-0}"
GOLEM_ROS_MCP_PORT="${GOLEM_ROS_MCP_PORT:-6082}"

# RViz's VNC display, bound to localhost in the VM (the SSH tunnel reaches it),
# on ports beside the MuJoCo viewer's 5900 / 6080 so both can run. The display
# must differ from the simulator's :99: both containers share the host network
# namespace, which scopes X11's abstract socket and its TCP ports, so two
# Xvfb on :99 would collide and both noVNC pages would show one screen.
RVIZ_DISPLAY=:100
RVIZ_VNC_PORT=5901
RVIZ_NOVNC_PORT=6081
RVIZ_GEOMETRY="${RVIZ_GEOMETRY:-1600x900x24}"
# Empty: golem_bringup's installed rviz/h12/main.rviz, resolved in
# start_rviz_stack once the workspace is sourced.
RVIZ_CONFIG="${RVIZ_CONFIG:-}"

# Start Xvfb, fluxbox, x11vnc, websockify and RViz2 in the background; return
# 1 when Xvfb does not come up within 15 s.
start_rviz_stack() {
    echo "[launch_ros_mac] starting RViz VNC stack on $RVIZ_DISPLAY ($RVIZ_GEOMETRY)"
    # `docker restart` keeps the container's filesystem: clear a previous run's
    # X lock and socket, or Xvfb aborts with "Server is already active".
    dnum="${RVIZ_DISPLAY#:}"
    pkill -f "Xvfb ${RVIZ_DISPLAY}" 2>/dev/null || true
    sleep 0.3
    rm -f "/tmp/.X${dnum}-lock" "/tmp/.X11-unix/X${dnum}" 2>/dev/null || true
    Xvfb "$RVIZ_DISPLAY" -screen 0 "$RVIZ_GEOMETRY" +extension GLX +render -noreset \
        >/tmp/xvfb_rviz.log 2>&1 &
    export DISPLAY="$RVIZ_DISPLAY"
    for _ in $(seq 1 30); do
        xdpyinfo -display "$RVIZ_DISPLAY" >/dev/null 2>&1 && break
        sleep 0.5
    done
    xdpyinfo -display "$RVIZ_DISPLAY" >/dev/null 2>&1 \
        || { echo "[launch_ros_mac] Xvfb failed to start"; cat /tmp/xvfb_rviz.log; return 1; }

    fluxbox >/tmp/fluxbox_rviz.log 2>&1 &
    sleep 1
    x11vnc -display "$RVIZ_DISPLAY" -rfbport "$RVIZ_VNC_PORT" -localhost \
        -forever -shared -nopw -quiet -bg >/tmp/x11vnc_rviz.log 2>&1
    sleep 1
    websockify --web /usr/share/novnc "127.0.0.1:$RVIZ_NOVNC_PORT" "localhost:$RVIZ_VNC_PORT" \
        >/tmp/websockify_rviz.log 2>&1 &
    sleep 1

    # llvmpipe: there is no GPU under Colima.
    export LIBGL_ALWAYS_SOFTWARE=1
    if [ -z "$RVIZ_CONFIG" ]; then
        _pfx="$(ros2 pkg prefix golem_bringup 2>/dev/null)"
        [ -n "$_pfx" ] && RVIZ_CONFIG="$_pfx/share/golem_bringup/rviz/h12/main.rviz"
    fi
    local cfg_arg=()
    [ -f "$RVIZ_CONFIG" ] && cfg_arg=(-d "$RVIZ_CONFIG")
    rviz2 "${cfg_arg[@]}" >/tmp/rviz2.log 2>&1 &
    echo "[launch_ros_mac] RViz launched. From the Mac run mac_vnc_tunnel.sh, then open:"
    echo "[launch_ros_mac]   http://localhost:${RVIZ_NOVNC_PORT}/vnc.html?autoconnect=1&resize=scale"
}

# Start tools/ros_mcp_server.py in the background, with ROS sourced; pip
# installs the `mcp` package on first use (the slim image lacks it). Return 1
# when that install fails.
start_ros_mcp() {
    echo "[launch_ros_mac] starting ROS debug MCP server on 127.0.0.1:${GOLEM_ROS_MCP_PORT}"
    python3 -c "import mcp" 2>/dev/null || {
        echo "[launch_ros_mac] installing the 'mcp' pip package (first run)..."
        pip install --quiet --disable-pip-version-check mcp >/tmp/pip_mcp.log 2>&1 \
            || { echo "[launch_ros_mac] pip install mcp FAILED:"; tail -5 /tmp/pip_mcp.log; return 1; }
    }
    GOLEM_ROS_MCP_PORT="$GOLEM_ROS_MCP_PORT" \
        python3 /home/code/tools/ros_mcp_server.py >/tmp/ros_mcp.log 2>&1 &
    sleep 1
    echo "[launch_ros_mac] MCP server -> /tmp/ros_mcp.log. Register on the Mac (after mac_vnc_tunnel.sh):"
    echo "[launch_ros_mac]   claude mcp add --transport http ros_debug http://localhost:${GOLEM_ROS_MCP_PORT}/mcp"
}

source /opt/ros/humble/setup.bash
# livox_ros_driver2, baked into the image at /opt/livox_ws
# (RosDockerfile.slim.arm64): fast_lio builds against its CustomMsg.
if [ ! -f /opt/livox_ws/install/setup.bash ]; then
    echo '[launch_ros_mac] missing baked livox_ros_driver2; rebuild with docker/mac/scripts/docker_build_mac.sh ros' >&2
    exit 1
fi
source /opt/livox_ws/install/setup.bash

WS=/home/code/core_ws
# build/ and install/ are named volumes on the VM's own disk
# (docker/mac/docker-compose.yml): a --symlink-install tree written onto the
# virtiofs bind mount stalls for minutes. The volumes persist, so builds are
# incremental.
BUILD_BASE=/opt/core_ws/build
INSTALL_BASE=/opt/core_ws/install
cd "$WS"

# PKGS: what h12_sim_mac.launch.py runs. The state and control blocks
# (golem_ros2_models, golem_safety, golem_upper_ik), the leg controller
# (golem_lowerbody_rl and its unitree_hg; GOLEM_LOWERBODY) and golem_slam's
# navigation graph (golem_slam, fast_lio; GOLEM_SLAM / GOLEM_NAV2), the message
# packages, and golem_bringup itself, so `ros2 launch golem_bringup ...` resolves.
# BRINGUP_STUB_DEPS: golem_bringup's in-workspace exec_depends the Mac does not
# build (real-robot drivers, the ML stack, MJPC, the loco-manipulation game,
# the G1's, and livox_ros_driver2, which comes from /opt/livox_ws). colcon
# refuses to build golem_bringup until each has a package.sh in the install
# space, so each gets an empty one; with no ament index entry they do not
# appear in `ros2 pkg list`. A package added to golem_bringup's package.xml
# goes into one list or the other (CLAUDE.md §6).
PKGS="custom_ros_messages magpie_msgs golem_ros2_models golem_safety golem_upper_ik golem_slam fast_lio golem_bringup unitree_hg golem_lowerbody_rl"
BRINGUP_STUB_DEPS="estop magpie_control livox_ros_driver2 model_server cl_realsense lmg_deploy dex_3 g1_homie deploy_mjpc golem_skills"

for _dep in $BRINGUP_STUB_DEPS; do
    mkdir -p "$INSTALL_BASE/$_dep/share/$_dep"
    [ -e "$INSTALL_BASE/$_dep/share/$_dep/package.sh" ] || : > "$INSTALL_BASE/$_dep/share/$_dep/package.sh"
done

echo "[launch_ros_mac] colcon build --packages-select $PKGS"
colcon build --symlink-install \
    --build-base "$BUILD_BASE" --install-base "$INSTALL_BASE" \
    --packages-select $PKGS \
    --cmake-args -DBUILD_TESTING=OFF

source "$INSTALL_BASE/setup.bash"

# RViz and the MCP server start before the bringup, so they also run with a
# command instead of the bringup; both pick up topics as they appear.
if [ "$GOLEM_RVIZ" = "vnc" ] || [ "$GOLEM_RVIZ" = "1" ]; then
    start_rviz_stack || echo "[launch_ros_mac] RViz stack failed to start (continuing without it)"
fi

if [ "$GOLEM_ROS_MCP" = "1" ] || [ "$GOLEM_ROS_MCP" = "vnc" ]; then
    start_ros_mcp || echo "[launch_ros_mac] MCP server failed to start (continuing without it)"
fi

if [ "${GOLEM_WAIT_FOR_SIM:-0}" = 1 ]; then
    python3 /home/code/golem_common/wait_for_sim.py
fi
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

# The Mac bringup, launched by path.

echo "[launch_ros_mac] launching the Mac bringup (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
exec ros2 launch "$WS/src/golem_bringup/launch/h12/h12_sim_mac.launch.py"
