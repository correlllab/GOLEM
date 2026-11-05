#!/bin/bash
# Execute only a clean Humble child, independently of the caller's shell/overlays.
golem_clean_python() {
    local mode="$1" helper="$2"
    shift 2
    local uv="${GOLEM_NATIVE_UV:-$(command -v uv || true)}"
    local -a clean=(HOME="$HOME" USER="$(/usr/bin/id -un)" PATH=/usr/bin:/bin
        LANG=C.UTF-8 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}" GOLEM_NATIVE_UV="$uv")
    if [[ $mode == runtime ]]; then
        local name
        for name in DISPLAY XAUTHORITY WAYLAND_DISPLAY XDG_RUNTIME_DIR MUJOCO_GL \
                GOLEM_SIM_ODOM GOLEM_CAMERA_SCALE GOLEM_CAMERA_DEPTH GOLEM_CAMERA_POINTCLOUD; do
            if [[ -v $name ]]; then clean+=("$name=${!name}"); fi
        done
    fi
    exec /usr/bin/env -i "${clean[@]}" /bin/bash --noprofile --norc -c '
        set -e
        if [[ ! -f /opt/ros/humble/setup.bash ]]; then
            printf "%s\n" "error: install ROS 2 Humble; missing /opt/ros/humble/setup.bash" >&2
            exit 1
        fi
        source /opt/ros/humble/setup.bash
        exec /usr/bin/python3 "$@"
    ' golem-environment "$helper" "$@"
}
