#!/usr/bin/env bash
# Start, restart, stop or follow the Apple-Silicon simulator and ros container
# together: docker_stack.sh with GOLEM_PLATFORM=mac. The ros container builds
# its packages, waits for the simulator's rt/lowstate, then starts
# h12_sim_mac.launch.py.
#
# Usage: docker_stack_mac.sh up|restart|stop|logs robocasa [--headless|--gui]
#   --gui  the MuJoCo viewer over noVNC (GOLEM_DISPLAY=vnc)
# Arguments, environment and exit status: docker/scripts/docker_stack.sh.
set -eo pipefail
export GOLEM_PLATFORM=mac
exec "$(dirname "$0")/../../scripts/docker_stack.sh" "$@"
