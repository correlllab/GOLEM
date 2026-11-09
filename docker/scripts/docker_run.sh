#!/usr/bin/env bash
# Start one GOLEM container in the foreground (docker compose run --rm).
#
# Usage: docker_run.sh SERVICE [--restart] [command... | launcher flags...]
#   SERVICE     robocasa, ros or isaac; the container is
#               golem_ros or golem_sim_<SERVICE>
#   --restart   replace an existing container of that name; without it an
#               existing container is an error
#   command     run instead of the image's launcher (e.g. bash)
#   flags       a first argument starting with - runs the service's launcher
#               (launch_<SERVICE>.sh) with all the arguments, e.g. --headless
#   --g1/--h12  anywhere: the simulator's robot (not for ros, which serves both)
# Environment: GOLEM_ROBOT=h12|g1 (simulators), GOLEM_DISPLAY=auto|gui|headless,
# GOLEM_CPU_PINNING, GOLEM_SIM_CPUSET and GOLEM_ROS_CPUSET (the CPUs robocasa
# and ros are pinned to; on a hybrid CPU, the performance and efficiency
# cores), ROS_DOMAIN_ID and the rest of docker_common.sh. Prints the robot and
# the CPUs it pinned to (`cpus: ...`) on stderr.
# Exit status: 1 when a check fails or the domain-0 prompt is declined;
# otherwise the container command's.
set -eo pipefail
source "$(dirname "$0")/docker_common.sh"
parse_robot_flags "$@"
set -- "${WRAPPER_ARGS[@]}"
SIM=${1:?Usage: docker_run.sh SERVICE [--restart] [command or flags...]}
shift
load_config
validate_service "$SIM"
# The simulators run one robot, chosen here; the ros container serves both.
if [ "$SIM" = ros ]; then
    [ -z "$ROBOT_FLAG" ] || fail "docker_run.sh ros takes no --$ROBOT_FLAG flag: one ros container serves both robots (H1-2 and G1)."
else
    configure_robot "$SIM"
    echo "robot: $GOLEM_ROBOT" >&2
fi
RESTART=0
if [ "${1:-}" = --restart ]; then RESTART=1; shift; fi
check_domain "$SIM"
# A --headless for the launcher also skips the desktop mounts.
for arg in "$@"; do [ "$arg" != --headless ] || export GOLEM_DISPLAY=headless; done
configure_display
# Pin the container to its CPUs (robocasa, ros; not the Isaac simulator).
[ "$SIM" = isaac ] || configure_cpus
prepare_mounts "$SIM"
# Never remove an existing container implicitly.
NAME=$(container_name "$SIM")
if docker inspect --type container "$NAME" >/dev/null 2>&1; then
    [ "$RESTART" = 1 ] || fail "$NAME already exists. Use --restart explicitly, or attach with docker exec -it $NAME bash."
    docker rm -f "$NAME"
fi
if [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; then
    set -- "/home/code/h12_sim_scripts/launch_${SIM}.sh" "$@"
fi
exec "${COMPOSE[@]}" run --rm --name "$NAME" "$SIM" "$@"
