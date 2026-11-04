#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(/usr/bin/dirname -- "$(/usr/bin/readlink -f -- "${BASH_SOURCE[0]}")")"

# Help must work without ROS, uv, an activated environment, or a particular cwd.
if [[ $# == 0 ]]; then
    set -- --help
fi
for arg in "$@"; do
    if [[ $arg == -h || $arg == --help ]]; then
        exec /usr/bin/env -i HOME="$HOME" PATH=/usr/bin:/bin \
            PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
            /usr/bin/python3 "$SCRIPT_DIR/build_ros_workspace.py" "$@"
    fi
done

UV="$(command -v uv || true)"
if [[ ${1-} == setup && -z "$UV" ]]; then
    printf '%s\n' 'error: uv is required for setup; install it in your user account first.' >&2
    exit 1
fi
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    printf '%s\n' 'error: missing /opt/ros/humble/setup.bash; ROS 2 Humble is required.' >&2
    exit 1
fi

# Discard conda, user Python, foreign overlays, DDS configuration, and loader paths.
# Only the installed Humble underlay is sourced; no workspace setup is auto-sourced.
exec /usr/bin/env -i \
    HOME="$HOME" USER="$(/usr/bin/id -un)" \
    PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 ROS_DOMAIN_ID=1 GOLEM_NATIVE_UV="$UV" \
    /bin/bash --noprofile --norc -c '
        set -e
        source /opt/ros/humble/setup.bash
        exec /usr/bin/python3 "$@"
    ' ros-workspace "$SCRIPT_DIR/build_ros_workspace.py" "$@"
