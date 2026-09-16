#!/usr/bin/env bash
# Shared host-side configuration for Linux and Apple-Silicon wrappers.
GOLEM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

fail() { echo "ERROR: $*" >&2; exit 1; }
require_file() {
    [ -f "$1" ] || fail "Missing $1. Initialize submodules and fetch Git LFS assets (see docs/SETUP.md)."
    local first_line
    IFS= read -r first_line < "$1" || true
    [ "$first_line" != 'version https://git-lfs.github.com/spec/v1' ] || fail "$1 is a Git LFS pointer; fetch its LFS contents before launching."
}
require_dir() { [ -d "$1" ] || fail "Missing mount directory: $1"; }

load_config() {
    cd "$GOLEM_ROOT"
    # Match Compose precedence: explicitly exported shell values win over .env.
    local saved_env
    saved_env=$(export -p | sed 's/^declare -x /export /')
    if [ -f docker/.env ]; then
        set -a
        source docker/.env
        set +a
        eval "$saved_env"
    fi
    [ -z "$(env | sed -n '/^HAMS_[A-Z0-9_]*=/p')" ] || fail 'Rename HAMS_* variables to GOLEM_*.'
    export GOLEM_ASSETS_DIR="${GOLEM_ASSETS_DIR:-$GOLEM_ROOT/CL_Assets}"
    export GOLEM_CACHE_DIR="${GOLEM_CACHE_DIR:-$GOLEM_ROOT/container_cache}"
    # Relative user paths are always relative to the checkout, on both platforms.
    case "$GOLEM_ASSETS_DIR" in /*) ;; *) GOLEM_ASSETS_DIR="$GOLEM_ROOT/$GOLEM_ASSETS_DIR";; esac
    case "$GOLEM_CACHE_DIR" in /*) ;; *) GOLEM_CACHE_DIR="$GOLEM_ROOT/$GOLEM_CACHE_DIR";; esac
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
    case "$ROS_DOMAIN_ID" in ''|*[!0-9]*) fail 'ROS_DOMAIN_ID must be an integer from 0 to 232.';; esac
    [ "${#ROS_DOMAIN_ID}" -le 3 ] || fail 'ROS_DOMAIN_ID must be from 0 to 232.'
    ROS_DOMAIN_ID=$((10#$ROS_DOMAIN_ID))
    [ "$ROS_DOMAIN_ID" -le 232 ] || fail 'ROS_DOMAIN_ID must be from 0 to 232.'
    ENV_FILE="$GOLEM_ROOT/docker/.env"
    [ -f "$ENV_FILE" ] || ENV_FILE=/dev/null
    if [ "${GOLEM_PLATFORM:-linux}" = mac ]; then
        COMPOSE=(docker compose --env-file "$ENV_FILE" -f "$GOLEM_ROOT/docker/mac/docker-compose.yml")
        LAUNCH_SUFFIX=_mac
    else
        COMPOSE=(docker compose --env-file "$ENV_FILE" -f "$GOLEM_ROOT/docker/docker-compose.yml")
        LAUNCH_SUFFIX=
    fi
}

validate_service() {
    case "$1" in
        robocasa|ros) ;;
        isaac) [ "${GOLEM_PLATFORM:-linux}" != mac ] || fail 'Isaac requires NVIDIA and is unavailable on Mac.';;
        *) fail "Unknown service '$1'; use robocasa, ros, or isaac (Linux only).";;
    esac
}

check_domain() {
    if [ "$ROS_DOMAIN_ID" = 0 ]; then
        [ "$1" = ros ] && [ "${GOLEM_PLATFORM:-linux}" != mac ] || fail 'Domain 0 is reserved for the real robot.'
        local reply
        read -r -p 'Domain 0 is the REAL ROBOT bus. Proceed? [y/N] ' reply
        case "$reply" in y|Y|yes|YES) ;; *) fail 'Real-robot launch cancelled.';; esac
    fi
}

configure_display() {
    [ "${GOLEM_PLATFORM:-linux}" != mac ] || return 0
    local mode=${GOLEM_DISPLAY:-auto}
    [ "$mode" != auto ] || { if [ -n "${DISPLAY:-}" ]; then mode=gui; else mode=headless; fi; }
    case "$mode" in
        headless) export DISPLAY= ;;
        gui)
            [ -n "${DISPLAY:-}" ] || fail 'GUI mode needs DISPLAY. Use GOLEM_DISPLAY=headless.'
            export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
            require_file "$XAUTHORITY"
            require_dir /tmp/.X11-unix
            COMPOSE+=(-f "$GOLEM_ROOT/docker/docker-compose.gui.yml")
            ;;
        *) fail 'Linux GOLEM_DISPLAY must be auto, gui, or headless.';;
    esac
}

prepare_mounts() {
    local service=$1
    require_file "$GOLEM_ASSETS_DIR/ros_assets/h1_2_magpie_ros.urdf"
    case "$service" in
        isaac)
            require_file "$GOLEM_ROOT/CL_isaaclab_sim/sim_main.py"
            mkdir -p "$GOLEM_CACHE_DIR/isaac/ov" "$GOLEM_CACHE_DIR/isaac/nvidia" \
                     "$GOLEM_CACHE_DIR/isaac/local_share_ov" "$GOLEM_CACHE_DIR/isaac/isaacsim"
            ;;
        robocasa)
            require_file "$GOLEM_ROOT/h1_robocasa/h12_mujoco.py"
            require_file "$GOLEM_ROOT/core_ws/src/magpie_msgs/package.xml"
            require_file "$GOLEM_ROOT/core_ws/src/custom_ros_messages/package.xml"
            mkdir -p "$GOLEM_CACHE_DIR/msgs_ws/src/magpie_msgs" "$GOLEM_CACHE_DIR/msgs_ws/src/custom_ros_messages"
            ;;
        ros)
            require_file "$GOLEM_ROOT/core_ws/src/h1_bringup/package.xml"
            if [ "${GOLEM_PLATFORM:-linux}" != mac ]; then
                require_file "$GOLEM_ROOT/core_ws/cyclonedds.xml"
                require_file "$GOLEM_ROOT/mujoco_mpc/CMakeLists.txt"
                require_dir "$GOLEM_ASSETS_DIR/graspgenx/assets/magpie"
                require_dir "$GOLEM_ASSETS_DIR/graspgenx/assets/meshes/magpie"
                export GOLEM_MJPC_GIT_DIR="${GOLEM_MJPC_GIT_DIR:-$(git -C "$GOLEM_ROOT/mujoco_mpc" rev-parse --absolute-git-dir 2>/dev/null || echo "$GOLEM_ROOT/.git/modules/mujoco_mpc")}"
                require_dir "$GOLEM_MJPC_GIT_DIR"
                mkdir -p "$GOLEM_CACHE_DIR/mjpc_build" "$GOLEM_ROOT/mujoco_mpc/build"
            fi
            ;;
    esac
}

container_name() {
    if [ "$1" = ros ]; then echo golem_ros; else echo "golem_sim_$1"; fi
}
