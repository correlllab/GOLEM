#!/usr/bin/env bash
# Usage: docker_stack_mac.sh up|restart|stop|logs robocasa [--headless|--gui]
set -eo pipefail
export GOLEM_PLATFORM=mac
exec "$(dirname "$0")/../../scripts/docker_stack.sh" "$@"
