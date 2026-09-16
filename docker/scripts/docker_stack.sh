#!/usr/bin/env bash
# Usage: docker_stack.sh up|restart|stop|logs robocasa|isaac [--headless|--gui]
# Pre-build the ROS workspace before starting physics; wait for DDS before bringup.
set -eo pipefail
source "$(dirname "$0")/docker_common.sh"
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
# Stopping or reading logs must work even if a display or a source disk is gone.
if [ "$ACTION" = stop ]; then
    for service in ros "$SIM"; do
        name=$(container_name "$service")
        if docker inspect "$name" >/dev/null 2>&1; then docker stop "$name"; fi
    done
    exit 0
fi
if [ "$ACTION" = logs ]; then exec "${COMPOSE[@]}" logs -f "$SIM" ros; fi
check_domain "$SIM"
configure_display
prepare_mounts "$SIM"
prepare_mounts ros
if [ "$ACTION" = restart ]; then
    # Explicit restart also replaces one-off containers created by docker_run.sh.
    for service in ros "$SIM"; do
        name=$(container_name "$service")
        if docker inspect "$name" >/dev/null 2>&1; then docker rm -f "$name"; fi
    done
else
    for service in "$SIM" ros; do
        name=$(container_name "$service")
        if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true)" = true ]; then
            fail "$name is already running. Use the stack restart command to restart both coherently."
        fi
    done
fi
# Preparation suppresses optional GUI/MCP processes and does not start controllers.
"${COMPOSE[@]}" run --rm -T -e GOLEM_WAIT_FOR_SIM=0 -e GOLEM_RVIZ=0 -e GOLEM_ROS_MCP=0 \
    ros "/home/code/h12_sim_scripts/launch_ros${LAUNCH_SUFFIX}.sh" /bin/true
if [ "$SIM" = robocasa ]; then
    "${COMPOSE[@]}" run --rm -T -e GOLEM_PREPARE_ONLY=1 robocasa
fi
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
