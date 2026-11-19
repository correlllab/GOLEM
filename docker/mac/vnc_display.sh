#!/bin/bash
# Start a virtual X display for RViz in the Mac golem_ros container, then run
# the container's command.
#
# Usage: the Mac golem_ros image's ENTRYPOINT (docker/mac/RosDockerfile):
#   vnc_display.sh <command...>
# A Mac has no X server a container can render through (XQuartz gives no
# OpenGL on Apple Silicon), so RViz draws on Xvfb display :99 with Mesa's
# software OpenGL (llvmpipe), shown by x11vnc on port 5901 and by noVNC in a
# browser on port 6080. The containers run on the VM's host network, which
# OrbStack and Docker Desktop forward to the Mac's localhost:
#   http://localhost:6080/vnc.html   or   vnc://localhost:5901 (Screen Sharing)
# Environment: DISPLAY (:99, docker-compose.mac.yml); GOLEM_VNC=0 starts no
#   display (the command still runs); GOLEM_VNC_SIZE (default 1600x1000).
# Exit status: the command's. A display that fails to start is logged and the
#   command runs anyway: RViz then fails alone.
set -u

# Start Xvfb, x11vnc and noVNC in the background, once per container.
start_display() {
    local display=${DISPLAY:-:99} size=${GOLEM_VNC_SIZE:-1600x1000}
    local n=${display#:}
    # A container restarted with `docker restart` keeps its /tmp, and a stale
    # lock makes Xvfb refuse the display.
    if [ -e "/tmp/.X${n}-lock" ] && ! pgrep -x Xvfb >/dev/null; then
        rm -f "/tmp/.X${n}-lock" "/tmp/.X11-unix/X${n}"
    fi
    pgrep -x Xvfb >/dev/null && return 0
    Xvfb "$display" -screen 0 "${size}x24" -nolisten tcp >/tmp/xvfb.log 2>&1 &
    # Wait up to 5 s for the display's socket.
    for _ in $(seq 50); do [ -S "/tmp/.X11-unix/X${n}" ] && break; sleep 0.1; done
    if [ ! -S "/tmp/.X11-unix/X${n}" ]; then
        echo "vnc_display: Xvfb did not start on $display (/tmp/xvfb.log)" >&2
        return 0
    fi
    # No password: the VM's network is reachable only from the Mac.
    x11vnc -display "$display" -rfbport 5901 -forever -shared -nopw -quiet \
        >/tmp/x11vnc.log 2>&1 &
    websockify --web /usr/share/novnc 6080 localhost:5901 >/tmp/novnc.log 2>&1 &
    echo "vnc_display: RViz on http://localhost:6080/vnc.html (or vnc://localhost:5901)" >&2
}

[ "${GOLEM_VNC:-1}" = 0 ] || start_display
exec "$@"
