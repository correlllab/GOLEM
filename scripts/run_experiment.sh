#!/bin/bash
set -euo pipefail
HERE="$(/usr/bin/dirname -- "$(/usr/bin/readlink -f -- "${BASH_SOURCE[0]}")")"
exec "$HERE/environment/launch_controller.sh" --experiment "$@"
