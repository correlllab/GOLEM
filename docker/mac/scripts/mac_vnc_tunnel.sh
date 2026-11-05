#!/bin/bash
# Run on the Mac: forward the in-VM viewer and MCP ports to the Mac's
# localhost through an SSH tunnel over Colima's own SSH config. Colima does not
# forward host-network container ports to the Mac, so a tunnel is the way in.
# Forwarding a port nothing listens on yet is harmless.
#   MuJoCo viewer   http://localhost:6080/vnc.html  (VNC localhost:5900; GOLEM_DISPLAY=vnc)
#   RViz            http://localhost:6081/vnc.html  (VNC localhost:5901; GOLEM_RVIZ=vnc)
#   ROS MCP server  http://localhost:6082/mcp       (GOLEM_ROS_MCP=1)
#
# Usage:
#   mac_vnc_tunnel.sh          open the tunnel (replacing an old one), print the URLs
#   mac_vnc_tunnel.sh --open   ... and open both noVNC pages in the browser
#   mac_vnc_tunnel.sh --stop   close the tunnel
# Environment: COLIMA_PROFILE (default: default).
# Exit status: 0 on success; 1 without colima, when colima ssh-config fails or
# has no Host entry; ssh's status when the tunnel cannot be opened.
set -euo pipefail

PROFILE="${COLIMA_PROFILE:-default}"
# noVNC and VNC of the MuJoCo viewer, then of RViz, then the MCP server.
PORTS=(6080 5900 6081 5901 6082)
MUJOCO_URL="http://localhost:6080/vnc.html?autoconnect=1&resize=scale"
RVIZ_URL="http://localhost:6081/vnc.html?autoconnect=1&resize=scale"

# Kill every ssh process listening on one of PORTS. It matches by listening
# port (lsof), not by command line: `ssh -f` daemonizes and truncates its argv,
# so pgrep on "-L <port>" would miss it.
stop_tunnel() {
    local killed=0 pid port
    for port in "${PORTS[@]}"; do
        for pid in $(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null); do
            if ps -p "$pid" -o comm= 2>/dev/null | grep -q '^ssh'; then
                kill "$pid" 2>/dev/null && killed=1
            fi
        done
    done
    [ "$killed" = 1 ] && echo "tunnel closed" || echo "no tunnel running"
}

case "${1:-}" in
    --stop) stop_tunnel; exit 0 ;;
esac

command -v colima >/dev/null 2>&1 || { echo "ERROR: colima not found on PATH"; exit 1; }

# Colima's SSH config for the VM, and the Host alias in it.
CFG="$(mktemp -t colima-ssh.XXXXXX)"
if ! colima ssh-config --profile "$PROFILE" >"$CFG" 2>/dev/null; then
    echo "ERROR: 'colima ssh-config' failed — is colima running? (colima start)"; rm -f "$CFG"; exit 1
fi
HOST_ALIAS="$(awk '/^Host /{print $2; exit}' "$CFG")"
[ -n "$HOST_ALIAS" ] || { echo "ERROR: could not read Host alias from colima ssh-config"; rm -f "$CFG"; exit 1; }

# Close an existing tunnel first; a port still bound locally would trip
# ExitOnForwardFailure.
stop_tunnel >/dev/null
sleep 1

# One -L per port, each to the same port on the VM's loopback.
FWD_ARGS=()
for port in "${PORTS[@]}"; do
    FWD_ARGS+=(-L "${port}:127.0.0.1:${port}")
done

ssh -F "$CFG" -f -N -o ExitOnForwardFailure=yes "${FWD_ARGS[@]}" "$HOST_ALIAS"

echo "Tunnel up (Mac -> Colima VM):"
echo "  MuJoCo viewer : $MUJOCO_URL"
echo "  RViz          : $RVIZ_URL"
echo "  (raw VNC: localhost:5900 = MuJoCo, localhost:5901 = RViz)"
echo "Close it with: $0 --stop"

if [ "${1:-}" = "--open" ]; then
    open "$MUJOCO_URL"
    open "$RVIZ_URL"
fi
