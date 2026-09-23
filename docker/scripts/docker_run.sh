#!/usr/bin/env bash
# Usage: docker_run.sh SERVICE [--restart] [command | launcher flags...]
# GOLEM_DISPLAY=auto|gui|headless selects Linux desktop mounts.
set -eo pipefail
source "$(dirname "$0")/docker_common.sh"
SIM=${1:?Usage: docker_run.sh SERVICE [--restart] [command or flags...]}
shift
load_config
validate_service "$SIM"
RESTART=0
if [ "${1:-}" = --restart ]; then RESTART=1; shift; fi
check_domain "$SIM"
for arg in "$@"; do [ "$arg" != --headless ] || export GOLEM_DISPLAY=headless; done
configure_display
prepare_mounts "$SIM"
NAME=$(container_name "$SIM")
if docker inspect --type container "$NAME" >/dev/null 2>&1; then
    [ "$RESTART" = 1 ] || fail "$NAME already exists. Use --restart explicitly, or attach with docker exec -it $NAME bash."
    docker rm -f "$NAME"
fi
if [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; then
    set -- "/home/code/h12_sim_scripts/launch_${SIM}${LAUNCH_SUFFIX}.sh" "$@"
fi
exec "${COMPOSE[@]}" run --rm --name "$NAME" "$SIM" "$@"
