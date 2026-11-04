#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(/usr/bin/dirname -- "$(/usr/bin/readlink -f -- "${BASH_SOURCE[0]}")")"
if [[ $# == 0 ]]; then set -- --help; fi
for arg in "$@"; do
    if [[ $arg == -h || $arg == --help ]]; then
        exec /usr/bin/env -i HOME="$HOME" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
            PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 "$SCRIPT_DIR/setup_simulator.py" "$@"
    fi
done
UV="$(command -v uv || true)"
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    printf '%s\n' 'error: ROS 2 Humble /opt/ros/humble/setup.bash is required.' >&2
    exit 1
fi
# No inherited conda, Docker overlays, user site, or DDS/loader configuration.
exec /usr/bin/env -i HOME="$HOME" USER="$(/usr/bin/id -un)" \
    PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    ROS_DOMAIN_ID=1 GOLEM_NATIVE_UV="$UV" \
    /bin/bash --noprofile --norc -c '
        set -e
        source /opt/ros/humble/setup.bash
        exec /usr/bin/python3 "$@"
    ' native-simulator "$SCRIPT_DIR/setup_simulator.py" "$@"
