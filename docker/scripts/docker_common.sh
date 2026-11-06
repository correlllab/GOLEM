#!/usr/bin/env bash
# Shared host-side helpers for the Linux and Mac wrappers (docker_build.sh,
# docker_run.sh, docker_stack.sh and the docker/mac/scripts wrappers, which
# set GOLEM_PLATFORM=mac). Sourced, not run.
#
# Environment read:
#   GOLEM_PLATFORM     linux (default) or mac: which compose file and launchers
#   ROS_DOMAIN_ID      0..232, 1 when unset; 0 is the real robots' bus
#   GOLEM_ROBOT        h12 (default) or g1, for the simulator services
#   GOLEM_ASSETS_DIR   host CL_Assets (default: the checkout's)
#   GOLEM_CACHE_DIR    host cache root (default: the checkout's container_cache)
#   GOLEM_DISPLAY      Linux: auto (default), gui or headless
#   DISPLAY, XAUTHORITY  the X server for gui mode
#   GOLEM_MJPC_GIT_DIR the mujoco_mpc submodule's git directory (default: found
#                      with git)
#   docker/.env        any of the above; exported shell values win
# Globals set: GOLEM_ROOT, COMPOSE (the docker compose command line, extended
# by configure_robot and configure_display), LAUNCH_SUFFIX, ENV_FILE,
# ROBOT_FLAG, WRAPPER_ARGS.
# Every check fails through fail(): an "ERROR:" line on stderr and exit 1.
GOLEM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Print an error and exit 1.
fail() { echo "ERROR: $*" >&2; exit 1; }
# Fail unless file $1 exists and is not a Git LFS pointer.
require_file() {
    [ -f "$1" ] || fail "Missing $1. Initialize submodules and fetch Git LFS assets (see docs/SETUP.md)."
    local first_line
    IFS= read -r first_line < "$1" || true
    [ "$first_line" != 'version https://git-lfs.github.com/spec/v1' ] || fail "$1 is a Git LFS pointer; fetch its LFS contents before launching."
}
# Fail unless directory $1 exists.
require_dir() { [ -d "$1" ] || fail "Missing mount directory: $1"; }

# Load docker/.env, resolve the mount roots and the domain, and pick the
# platform's compose file. Changes into the checkout.
load_config() {
    cd "$GOLEM_ROOT"
    # Values exported in the shell win over docker/.env, as in Compose: save
    # them, source the file, then restore them.
    local saved_env
    saved_env=$(export -p | sed 's/^declare -x /export /')
    if [ -f docker/.env ]; then
        set -a
        source docker/.env
        set +a
        eval "$saved_env"
    fi
    export GOLEM_ASSETS_DIR="${GOLEM_ASSETS_DIR:-$GOLEM_ROOT/CL_Assets}"
    export GOLEM_CACHE_DIR="${GOLEM_CACHE_DIR:-$GOLEM_ROOT/container_cache}"
    # Relative paths are relative to the checkout, on both platforms.
    case "$GOLEM_ASSETS_DIR" in /*) ;; *) GOLEM_ASSETS_DIR="$GOLEM_ROOT/$GOLEM_ASSETS_DIR";; esac
    case "$GOLEM_CACHE_DIR" in /*) ;; *) GOLEM_CACHE_DIR="$GOLEM_ROOT/$GOLEM_CACHE_DIR";; esac
    # The domain: at most three decimal digits, read base 10 (bash would read
    # 08 as invalid octal), at most 232. 00 is domain 0.
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
    case "$ROS_DOMAIN_ID" in ''|*[!0-9]*) fail 'ROS_DOMAIN_ID must be an integer from 0 to 232.';; esac
    [ "${#ROS_DOMAIN_ID}" -le 3 ] || fail 'ROS_DOMAIN_ID must be from 0 to 232.'
    ROS_DOMAIN_ID=$((10#$ROS_DOMAIN_ID))
    [ "$ROS_DOMAIN_ID" -le 232 ] || fail 'ROS_DOMAIN_ID must be from 0 to 232.'
    # Compose gets the same env file explicitly (/dev/null when there is none).
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

# Fail unless $1 is a service of this platform: robocasa, ros, or (Linux only)
# isaac.
validate_service() {
    case "$1" in
        robocasa|ros) ;;
        isaac) [ "${GOLEM_PLATFORM:-linux}" != mac ] || fail 'Isaac requires NVIDIA and is unavailable on Mac.';;
        *) fail "Unknown service '$1'; use robocasa, ros, or isaac (Linux only).";;
    esac
}

# Split the robot selectors --g1 / --h12 out of the arguments, wherever they
# appear: ROBOT_FLAG is the selector (empty if none), WRAPPER_ARGS the rest.
# Fails on both selectors at once.
parse_robot_flags() {
    ROBOT_FLAG=
    WRAPPER_ARGS=()
    local arg
    for arg in "$@"; do
        case "$arg" in
            --g1|--h12)
                [ -z "$ROBOT_FLAG" ] || [ "$ROBOT_FLAG" = "${arg#--}" ] || fail 'Choose one robot: --g1 or --h12.'
                ROBOT_FLAG=${arg#--}
                ;;
            *) WRAPPER_ARGS+=("$arg");;
        esac
    done
}

# Select the robot of simulator service $1 and export it as GOLEM_ROBOT (h12 or
# g1): ROBOT_FLAG, else the shell's or docker/.env's GOLEM_ROBOT, else h12. The
# G1 adds docker-compose.g1.yml and runs only on x86 Linux with RoboCasa.
configure_robot() {
    local service=$1 robot
    robot=$(printf '%s' "${ROBOT_FLAG:-${GOLEM_ROBOT:-h12}}" | tr '[:upper:]' '[:lower:]')
    case "$robot" in
        h12) ;;
        g1)
            [ "${GOLEM_PLATFORM:-linux}" != mac ] || fail 'G1 is not available on Mac; use an x86 Linux host.'
            [ "$service" != isaac ] || fail 'G1 is not available with the Isaac simulator; use robocasa.'
            COMPOSE+=(-f "$GOLEM_ROOT/docker/docker-compose.g1.yml")
            ;;
        *) fail "GOLEM_ROBOT must be h12 or g1, not '$robot'.";;
    esac
    export GOLEM_ROBOT=$robot
}

# SAFETY: domain 0 is the real robots' command bus. Refuse it for every service
# but the x86 workspace container, and for that one ask on stdin. The prompt
# goes to stderr whether or not stdin is a terminal, so a log shows that
# domain 0 was confirmed.
check_domain() {
    if [ "$ROS_DOMAIN_ID" = 0 ]; then
        [ "$1" = ros ] && [ "${GOLEM_PLATFORM:-linux}" != mac ] || fail 'Domain 0 is reserved for the real robot.'
        local reply
        {
            echo 'Domain 0 is the REAL ROBOT bus.'
            printf 'Proceed? [y/N] '
        } >&2
        read -r reply
        case "$reply" in y|Y|yes|YES) ;; *) fail 'Real-robot launch cancelled.';; esac
    fi
}

# Linux: resolve GOLEM_DISPLAY (auto: gui when DISPLAY is set) and, for gui,
# check the X socket and Xauthority file and add docker-compose.gui.yml;
# headless clears DISPLAY. The Mac's display is the compose file's GOLEM_DISPLAY.
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

# Fail unless the CL_Assets/g1 files service $1 reads exist and are not Git LFS
# pointers: the G1 sim builds its robot from the MJCF and URDF, and every ros
# container builds golem_ros2_models, which installs the G1's URDF, sphere
# model and SRDF. Other services read none.
require_g1_model() {
    local service=$1
    local g1="$GOLEM_ASSETS_DIR/g1"
    local hint="Fetch the G1 model: git -C $GOLEM_ASSETS_DIR lfs pull (CL_Assets is a submodule: git submodule update --init CL_Assets first)"
    local files first_line path
    case "$service" in
        robocasa) files=(g1_description/g1_29dof_with_hand_rev_1_0.urdf g1_description/g1_29dof_with_hand_rev_1_0.xml) ;;
        ros) files=(g1_description/g1_29dof_with_hand_rev_1_0.urdf g1_sphere.urdf g1_sphere_collision.srdf LICENSE.unitree_ros) ;;
        *) return 0 ;;
    esac
    files=("${files[@]/#/$g1/}")
    for path in "${files[@]}"; do
        [ -f "$path" ] || fail "Missing $path. $hint"
    done
    local meshes=("$g1"/g1_description/meshes/*.STL)
    [ -f "${meshes[0]}" ] || fail "No meshes in $g1/g1_description/meshes. $hint"
    for path in "${files[@]}" "${meshes[@]}"; do
        IFS= read -r first_line < "$path" || true
        [ "$first_line" != 'version https://git-lfs.github.com/spec/v1' ] || fail "$path is a Git LFS pointer. $hint"
    done
}

# Fail unless the H1-2 files golem_ros2_models installs in every ros container
# (its URDF, sphere model and SRDF, and the body and Magpie gripper meshes)
# exist and are not Git LFS pointers.
require_h12_model() {
    local hint="Fetch the H1-2 model: git -C $GOLEM_ASSETS_DIR lfs pull (CL_Assets is a submodule: git submodule update --init CL_Assets first)"
    local files=(h1_2_magpie.urdf h1_2_magpie_sphere.urdf h1_2_magpie_sphere_collision.srdf)
    local first_line path dir
    files=("${files[@]/#/$GOLEM_ASSETS_DIR/ros_assets/}")
    for path in "${files[@]}"; do
        [ -f "$path" ] || fail "Missing $path. $hint"
    done
    local meshes=()
    for dir in h1_2 magpie; do
        local found=("$GOLEM_ASSETS_DIR/meshes/$dir"/*.[sS][tT][lL])
        [ -f "${found[0]}" ] || fail "No meshes in $GOLEM_ASSETS_DIR/meshes/$dir. $hint"
        meshes+=("${found[@]}")
    done
    for path in "${files[@]}" "${meshes[@]}"; do
        IFS= read -r first_line < "$path" || true
        [ "$first_line" != 'version https://git-lfs.github.com/spec/v1' ] || fail "$path is a Git LFS pointer. $hint"
    done
}

# Check the bind sources service $1 needs, before any container starts, and
# create the cache directories it mounts (Compose never creates a bind source).
prepare_mounts() {
    local service=$1
    # The robot models each service reads.
    case "$service" in
        ros)
            require_h12_model
            require_g1_model ros
            ;;
        robocasa) [ "${GOLEM_ROBOT:-h12}" != g1 ] || require_g1_model robocasa ;;
    esac
    # The service's own sources and caches.
    case "$service" in
        isaac)
            require_file "$GOLEM_ROOT/CL_isaaclab_sim/sim_main.py"
            mkdir -p "$GOLEM_CACHE_DIR/isaac/ov" "$GOLEM_CACHE_DIR/isaac/nvidia" \
                     "$GOLEM_CACHE_DIR/isaac/local_share_ov" "$GOLEM_CACHE_DIR/isaac/isaacsim"
            ;;
        robocasa)
            require_file "$GOLEM_ROOT/golem_robocasa/h12_mujoco.py"
            require_file "$GOLEM_ROOT/core_ws/src/magpie_msgs/package.xml"
            require_file "$GOLEM_ROOT/core_ws/src/custom_ros_messages/package.xml"
            # Mount points of the two read-only package mounts inside msgs_ws.
            mkdir -p "$GOLEM_CACHE_DIR/msgs_ws/src/magpie_msgs" "$GOLEM_CACHE_DIR/msgs_ws/src/custom_ros_messages"
            ;;
        ros)
            require_file "$GOLEM_ROOT/core_ws/src/golem_bringup/package.xml"
            if [ "${GOLEM_PLATFORM:-linux}" != mac ]; then
                require_file "$GOLEM_ROOT/core_ws/cyclonedds.xml"
                require_file "$GOLEM_ROOT/mujoco_mpc/CMakeLists.txt"
                require_dir "$GOLEM_ASSETS_DIR/graspgenx/assets/magpie"
                require_dir "$GOLEM_ASSETS_DIR/graspgenx/assets/meshes/magpie"
                # mujoco_mpc/.git is a file naming the real git directory;
                # docker-compose.yml mounts that directory.
                export GOLEM_MJPC_GIT_DIR="${GOLEM_MJPC_GIT_DIR:-$(git -C "$GOLEM_ROOT/mujoco_mpc" rev-parse --absolute-git-dir 2>/dev/null || echo "$GOLEM_ROOT/.git/modules/mujoco_mpc")}"
                require_dir "$GOLEM_MJPC_GIT_DIR"
                mkdir -p "$GOLEM_CACHE_DIR/mjpc_build" "$GOLEM_ROOT/mujoco_mpc/build"
            fi
            ;;
    esac
}

# Print the container name of service $1: golem_ros, or golem_sim_<service>.
container_name() {
    if [ "$1" = ros ]; then echo golem_ros; else echo "golem_sim_$1"; fi
}
