#!/usr/bin/env bash
# Build the Apple-Silicon images: docker_build.sh with GOLEM_PLATFORM=mac
# (golem_base:arm64, then golem_sim_robocasa:arm64 and golem_ros:arm64).
#
# Usage: docker_build_mac.sh [--pull] [--no-cache] [robocasa|ros ...]
# Arguments, environment and exit status: docker/scripts/docker_build.sh.
set -eo pipefail
export GOLEM_PLATFORM=mac
exec "$(dirname "$0")/../../scripts/docker_build.sh" "$@"
