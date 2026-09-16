#!/usr/bin/env bash
# Usage: docker_run_mac.sh robocasa|ros [--restart] [command or flags...]
set -eo pipefail
export GOLEM_PLATFORM=mac
exec "$(dirname "$0")/../../scripts/docker_run.sh" "$@"
