#!/usr/bin/env bash
# Start one Apple-Silicon container: docker_run.sh with GOLEM_PLATFORM=mac.
# Start robocasa and ros together with docker_stack_mac.sh instead: a
# simulator left alone for a few seconds times out its motor commands and the
# robot collapses.
#
# Usage: docker_run_mac.sh robocasa|ros [--restart] [command or flags...]
# Arguments, environment and exit status: docker/scripts/docker_run.sh (the
# Mac refuses isaac, --g1 and domain 0).
set -eo pipefail
export GOLEM_PLATFORM=mac
exec "$(dirname "$0")/../../scripts/docker_run.sh" "$@"
