#!/bin/bash
# Apple-Silicon / CPU launcher for the SLIM ROS workspace.
# Builds only the core bringup package subset (skips the vision / grasp / nav2
# / FAST-LIO packages that need the heavy ML stack), then launches the minimal
# Mac bringup (robot_state_publisher, joint_state_publisher, Pink IK
# frame_task_server, safety_node) against the running MuJoCo sim.
#
# Pass `bash` as the first arg to drop to a shell instead of launching.
#
# Set GOLEM_RVIZ=vnc to also run RViz2 rendered with software GL (llvmpipe) into
# an in-container Xvfb and streamed over VNC/noVNC on port 6081 (separate from
# the MuJoCo viewer's 6080). RViz's GL window can't forward to XQuartz on Apple
# Silicon, so we ship pixels. View from the Mac: docker/mac/scripts/mac_vnc_tunnel.sh
# then open http://localhost:6081/vnc.html.
set -e
source /home/code/golem_common/runtime_common.sh
validate_sim_domain

GOLEM_RVIZ="${GOLEM_RVIZ:-0}"
# GOLEM_ROS_MCP=1 starts the ROS debugging MCP server (tools/ros_mcp_server.py)
# on localhost:6082 inside the VM — reach it from the Mac via the SSH tunnel and
# register with `claude mcp add --transport http ros_debug http://localhost:6082/mcp`.
GOLEM_ROS_MCP="${GOLEM_ROS_MCP:-0}"
GOLEM_ROS_MCP_PORT="${GOLEM_ROS_MCP_PORT:-6082}"

# VNC/noVNC for RViz (only used when GOLEM_RVIZ=vnc). Localhost-bound in the VM;
# reachable from the Mac only via the SSH tunnel (mac_vnc_tunnel.sh). Ports are
# offset from the robocasa MuJoCo viewer's (5900/6080) so both run together.
#
# DISPLAY MUST differ from robocasa's :99. Both containers run network_mode:host,
# so they share ONE network namespace — and X11's abstract socket
# (@/tmp/.X11-unix/X<n>) plus TCP 60<nn> are namespace-scoped. Two Xvfb on :99
# would collide: one wins the abstract socket and BOTH viewers land on that single
# display (both noVNC ports then show the same mixed screen). :100 keeps RViz on
# its own X server.
RVIZ_DISPLAY=:100
RVIZ_VNC_PORT=5901
RVIZ_NOVNC_PORT=6081
RVIZ_GEOMETRY="${RVIZ_GEOMETRY:-1600x900x24}"
# main.rviz ships in the golem_bringup package (rviz/h12/). Resolved from the installed
# share dir inside start_rviz_stack (after the workspace is built + sourced).
RVIZ_CONFIG="${RVIZ_CONFIG:-}"

start_rviz_stack() {
    echo "[launch_ros_mac] starting RViz VNC stack on $RVIZ_DISPLAY ($RVIZ_GEOMETRY)"
    # `docker restart` reuses the container FS: clear any stale X lock/socket from
    # a previous run so Xvfb doesn't abort with "Server is already active".
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

    fluxbox >/tmp/fluxbox_rviz.log 2>&1 &   # bg set by the fbsetbg shim (Dockerfile)
    sleep 1
    x11vnc -display "$RVIZ_DISPLAY" -rfbport "$RVIZ_VNC_PORT" -localhost \
        -forever -shared -nopw -quiet -bg >/tmp/x11vnc_rviz.log 2>&1
    sleep 1
    websockify --web /usr/share/novnc "127.0.0.1:$RVIZ_NOVNC_PORT" "localhost:$RVIZ_VNC_PORT" \
        >/tmp/websockify_rviz.log 2>&1 &
    sleep 1

    export LIBGL_ALWAYS_SOFTWARE=1   # force llvmpipe; no GPU under Colima
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

# Background ROS debugging MCP server (warm rclpy node + HTTP MCP tools). Needs ROS
# sourced (done before this is called). Installs the `mcp` pip package on first run.
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

WS=/home/code/core_ws
# build/install go to /opt paths backed by named docker volumes on the VM's
# native ext4: writing colcon's --symlink-install tree onto the virtiofs
# bind-mount stalls for minutes. Named volumes are fast AND persistent, so the
# build caches incrementally across runs.
BUILD_BASE=/opt/core_ws/build
INSTALL_BASE=/opt/core_ws/install
cd "$WS"

# The packages the minimal Mac bringup needs. golem_bringup (ament_python, just
# launch/config/rviz files) is included so `ros2 launch golem_bringup <file>`
# resolves with tab completion — but colcon-ros refuses to build it until its
# in-workspace exec_depends have a package.sh in the install space, and we
# deliberately don't build the heavy ones (estop, livox_ros_driver2, fast_lio,
# model_server, cl_realsense, lmg_deploy — the LIO/ML/driver stack and the
# loco-manipulation game) or the G1's (the Mac runs the H1-2 only). Empty
# package.sh stubs (below) satisfy that check; they carry no ament index marker
# so they never show up in `ros2 pkg list`. NOTE only h12_sim_mac.launch.py runs
# here — the other launch files exec-launch the omitted packages (nav2,
# FAST_LIO, model_server...).
# unitree_hg + h12_lowerbody_rl provide the optional lower-body stack
# (walk/FAME policies). They build cheaply and are only *launched* when
# GOLEM_LOWERBODY is set (see the bringup), but building them always keeps
# `ros2 run h12_lowerbody_rl fame_node` available in a shell. unitree_hg needs
# ros-humble-rosidl-generator-dds-idl and torch, both in the image. Use
# GOLEM_LOWERBODY=almi to stand and walk (see docs/MACOS.md).
PKGS="custom_ros_messages magpie_msgs golem_ros2_models golem_safety golem_upper_ik h12_slam golem_bringup unitree_hg h12_lowerbody_rl"
BRINGUP_STUB_DEPS="estop livox_ros_driver2 fast_lio model_server cl_realsense lmg_deploy dex_3 g1_homie g1_navigation"

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

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

# Optionally bring up RViz (rendered to Xvfb, streamed over noVNC). Started
# before the bringup so it's up whether we launch or drop to a shell; RViz
# tolerates topics/TF arriving after it starts.
if [ "$GOLEM_RVIZ" = "vnc" ] || [ "$GOLEM_RVIZ" = "1" ]; then
    start_rviz_stack || echo "[launch_ros_mac] RViz stack failed to start (continuing without it)"
fi

# Optional ROS debugging MCP server (started before bringup so it's up whether we
# launch or drop to a shell; the warm subscribers just fill in as topics appear).
if [ "$GOLEM_ROS_MCP" = "1" ] || [ "$GOLEM_ROS_MCP" = "vnc" ]; then
    start_ros_mcp || echo "[launch_ros_mac] MCP server failed to start (continuing without it)"
fi

if [ "${GOLEM_WAIT_FOR_SIM:-0}" = 1 ]; then
    python3 /home/code/golem_common/wait_for_sim.py
fi
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

echo "[launch_ros_mac] launching minimal bringup (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
exec ros2 launch "$WS/src/golem_bringup/launch/h12/h12_sim_mac.launch.py"
