#!/usr/bin/env bash
# Build the GOLEM images (docker/BUILD.md).
#
# Usage: docker_build.sh [--pull] [--no-cache] [isaac|robocasa|ros ...]
#   isaac, robocasa, ros  the images to build; none: all three. robocasa and
#                         ros build golem_base first.
#   --pull                refresh the public base images (CUDA, Isaac Sim)
#   --no-cache            rebuild every layer
# Environment: docker/.env and the shell as in docker_common.sh.
# Exit status: 0 when every build succeeds; 1 on a bad argument or a missing
# source; otherwise the failing docker command's status.
set -eo pipefail
source "$(dirname "$0")/docker_common.sh"
load_config
SERVICES=()
BUILD_FLAGS=()
CHILD_FLAGS=()
# --pull goes only to builds with a public base; --no-cache to every build.
for arg in "$@"; do
    case "$arg" in
        --pull) BUILD_FLAGS+=("$arg");;
        --no-cache) BUILD_FLAGS+=("$arg"); CHILD_FLAGS+=("$arg");;
        --help|-h) echo 'Usage: docker_build.sh [--pull] [--no-cache] [isaac|robocasa|ros ...]'; exit 0;;
        *) validate_service "$arg"; SERVICES+=("$arg");;
    esac
done
[ ${#SERVICES[@]} -gt 0 ] || SERVICES=(robocasa ros isaac)
# Check the COPY sources and resolve the MJPC commit before building anything.
# The ros image builds MJPC at the mujoco_mpc submodule's commit, so the baked
# build matches the source docker-compose.yml mounts.
NEEDS_BASE=0
for service in "${SERVICES[@]}"; do
    case "$service" in
        ros|robocasa) NEEDS_BASE=1; require_file unitree_sdk2_python/setup.py;;
    esac
    if [ "$service" = robocasa ]; then require_file core_ws/src/livox_ros_driver2/package_ROS2.xml; fi
    if [ "$service" = ros ]; then
        export MJPC_REF
        MJPC_REF=$(git rev-parse HEAD:mujoco_mpc) || fail 'Cannot resolve the committed mujoco_mpc pin.'
        echo "MJPC_REF=$MJPC_REF"
    fi
done
# golem_base first, under the local tag the children's FROM names.
if [ "$NEEDS_BASE" = 1 ]; then
    docker build --platform linux/amd64 "${BUILD_FLAGS[@]}" -t golem_base:latest -f docker/BaseDockerfile .
fi
# Isaac has a public base, so it gets --pull. The children build on the local
# golem_base just built: --pull there would try to download golem_base from
# Docker Hub.
CHILDREN=()
for service in "${SERVICES[@]}"; do
    if [ "$service" = isaac ]; then
        "${COMPOSE[@]}" build "${BUILD_FLAGS[@]}" isaac
    else
        CHILDREN+=("$service")
    fi
done
if [ "${#CHILDREN[@]}" -gt 0 ]; then
    "${COMPOSE[@]}" build "${CHILD_FLAGS[@]}" "${CHILDREN[@]}"
fi
