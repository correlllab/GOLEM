#!/usr/bin/env bash
# Usage: docker_build_mac.sh [--pull] [--no-cache] [robocasa|ros ...]
set -eo pipefail
export GOLEM_PLATFORM=mac
exec "$(dirname "$0")/../../scripts/docker_build.sh" "$@"
