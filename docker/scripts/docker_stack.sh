#!/usr/bin/env bash
# Start, restart, stop or follow a simulator and the ros container together.
#
# Usage: docker_stack.sh up|restart|stop|logs robocasa|isaac [--headless|--gui] [--g1|--h12]
#   up        build the workspaces, then start the pair detached; fails if
#             either container is already running
#   restart   remove both containers (also ones docker_run.sh made), then up
#   stop      stop both containers; caches are kept
#   logs      follow both containers' logs
#   --headless / --gui  the display mode (Mac: --gui is vnc)
#   --g1 / --h12        the robot (anywhere in the arguments)
# The ros container builds core_ws, waits in wait_for_sim.py for a fresh
# rt/lowstate from a simulator of the selected robot, then starts its bringup
# (docker-compose.stack.yml): h12_sim.launch.py, or g1_sim.launch.py for the G1.
# Environment: GOLEM_ROBOT=h12|g1, GOLEM_DISPLAY, GOLEM_SIM_TIMEOUT (s, the
# wait), GOLEM_USE_RVIZ (true|false; default true when DISPLAY is set),
# ROS_DOMAIN_ID and the rest of docker_common.sh; GOLEM_PLATFORM=mac from
# docker_stack_mac.sh.
# Exit status: 0 once both containers are started or stopped; 1 when a check
# fails; otherwise the failing docker command's.
set -eo pipefail
source "$(dirname "$0")/docker_common.sh"
parse_robot_flags "$@"
set -- "${WRAPPER_ARGS[@]}"
ACTION=${1:?Usage: docker_stack.sh up|restart|stop|logs robocasa|isaac [--headless|--gui]}
SIM=${2:?Select robocasa or isaac}
shift 2
load_config
validate_service "$SIM"
[ "$SIM" != ros ] || fail 'Choose a simulator: robocasa or isaac.'
case "$ACTION" in up|restart|stop|logs) ;; *) fail "Unknown action: $ACTION";; esac
for arg in "$@"; do
    case "$arg" in
        --headless) export GOLEM_DISPLAY=headless;;
        --gui) if [ "${GOLEM_PLATFORM:-linux}" = mac ]; then export GOLEM_DISPLAY=vnc; else export GOLEM_DISPLAY=gui; fi;;
        *) fail "Unknown option: $arg";;
    esac
done
# stop and logs come before the display and mount checks, so they work with
# the display or a source disk gone.
if [ "$ACTION" = stop ]; then
    for service in ros "$SIM"; do
        name=$(container_name "$service")
        if docker inspect --type container "$name" >/dev/null 2>&1; then docker stop "$name"; fi
    done
    exit 0
fi
if [ "$ACTION" = logs ]; then exec "${COMPOSE[@]}" logs -f "$SIM" ros; fi
configure_robot "$SIM"
echo "robot: $GOLEM_ROBOT" >&2
if [ "$GOLEM_ROBOT" = g1 ]; then
    # Read by docker-compose.stack.yml. Set here, from the selected robot,
    # because a value left in the shell or docker/.env would start the H1-2
    # bringup against the G1 sim.
    export GOLEM_BRINGUP_LAUNCH=g1_sim.launch.py
fi
check_domain "$SIM"
configure_display
prepare_mounts "$SIM"
prepare_mounts ros
if [ "$ACTION" = restart ]; then
    for service in ros "$SIM"; do
        name=$(container_name "$service")
        if docker inspect --type container "$name" >/dev/null 2>&1; then docker rm -f "$name"; fi
    done
else
    for service in "$SIM" ros; do
        name=$(container_name "$service")
        if [ "$(docker inspect --type container -f '{{.State.Running}}' "$name" 2>/dev/null || true)" = true ]; then
            fail "$name is already running. Use the stack restart command to restart both coherently."
        fi
    done
fi
# Build the workspaces before any physics runs: a simulator left alone while
# colcon builds would time out its motor commands. The ros build runs without
# the wait, RViz or the MCP server; the robocasa run builds msgs_ws and exits
# (GOLEM_PREPARE_ONLY).
"${COMPOSE[@]}" run --rm -T -e GOLEM_WAIT_FOR_SIM=0 -e GOLEM_RVIZ=0 -e GOLEM_ROS_MCP=0 \
    ros "/home/code/h12_sim_scripts/launch_ros${LAUNCH_SUFFIX}.sh" /bin/true
if [ "$SIM" = robocasa ]; then
    "${COMPOSE[@]}" run --rm -T -e GOLEM_PREPARE_ONLY=1 robocasa
fi
# Add the stack overlay (wait for the sim, then bringup).
if [ "${GOLEM_PLATFORM:-linux}" = mac ]; then
    COMPOSE+=(-f "$GOLEM_ROOT/docker/mac/docker-compose.stack.yml")
else
    if [ -n "${DISPLAY:-}" ]; then
        export GOLEM_USE_RVIZ=${GOLEM_USE_RVIZ:-true}
    else
        export GOLEM_USE_RVIZ=${GOLEM_USE_RVIZ:-false}
    fi
    COMPOSE+=(-f "$GOLEM_ROOT/docker/docker-compose.stack.yml")
fi
"${COMPOSE[@]}" up -d --no-build --force-recreate "$SIM" ros
echo "Started $SIM + ros. Use the stack logs command to follow readiness and bringup."
