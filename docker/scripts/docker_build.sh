#!/usr/bin/env bash
# Usage: docker_build.sh [--pull] [--no-cache] [isaac|robocasa|ros ...]
set -eo pipefail
source "$(dirname "$0")/docker_common.sh"
load_config
SERVICES=()
BUILD_FLAGS=()
CHILD_FLAGS=()
for arg in "$@"; do
    case "$arg" in
        --pull) BUILD_FLAGS+=("$arg");;
        --no-cache) BUILD_FLAGS+=("$arg"); CHILD_FLAGS+=("$arg");;
        --help|-h) echo 'Usage: docker_build.sh [--pull] [--no-cache] [isaac|robocasa|ros ...]'; exit 0;;
        *) validate_service "$arg"; SERVICES+=("$arg");;
    esac
done
if [ ${#SERVICES[@]} -eq 0 ]; then
    SERVICES=(robocasa ros)
    [ "${GOLEM_PLATFORM:-linux}" = mac ] || SERVICES+=(isaac)
fi
NEEDS_BASE=0
for service in "${SERVICES[@]}"; do
    case "$service" in
        ros|robocasa) NEEDS_BASE=1; require_file unitree_sdk2_python/setup.py;;
    esac
    if [ "$service" = robocasa ]; then require_file core_ws/src/livox_ros_driver2/package_ROS2.xml; fi
    if [ "$service" = ros ] && [ "${GOLEM_PLATFORM:-linux}" != mac ]; then
        export MJPC_REF
        MJPC_REF=$(git rev-parse HEAD:mujoco_mpc) || fail 'Cannot resolve the committed mujoco_mpc pin.'
        echo "MJPC_REF=$MJPC_REF"
    fi
done
if [ "$NEEDS_BASE" = 1 ]; then
    if [ "${GOLEM_PLATFORM:-linux}" = mac ]; then
        docker build --platform linux/arm64 "${BUILD_FLAGS[@]}" -t golem_base:arm64 -f docker/mac/BaseDockerfile.arm64 .
    else
        docker build --platform linux/amd64 "${BUILD_FLAGS[@]}" -t golem_base:latest -f docker/BaseDockerfile .
    fi
fi
# Child images inherit the local base we just built; --pull there would try
# downloading golem_base from Docker Hub. Isaac has a public upstream base.
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
