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

# Splits the robot selectors --g1 / --h12 out of the arguments, wherever they
# appear. ROBOT_FLAG is the selector (empty if none); WRAPPER_ARGS is the rest.
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

# Selects the robot for simulator service $1 and exports it as GOLEM_ROBOT (h12 or g1).
# Precedence: ROBOT_FLAG, then the shell, then docker/.env (see load_config).
# G1 adds its compose overlay; it runs on x86 Linux, never with Isaac.
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

# The confirmation is printed whether or not stdin is a terminal, so a log shows
# that domain 0 was confirmed.
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

# The CL_Assets/g1 files a service reads: the G1 sim builds its robot from the
# MJCF and URDF, and every ros container builds golem_ros2_models, which installs
# the G1's URDF, sphere model and SRDF for both robots. The meshes are Git LFS.
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

# The CL_Assets files of the H1-2 that golem_ros2_models installs in every ros
# container: its URDF, sphere model and SRDF, and the body and Magpie gripper
# meshes, which are Git LFS.
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

# HOMIE's walking policy, which docker-compose.yml bind-mounts into every ros
# container (launch_ros.sh builds both robots' packages there).
require_g1_policy() {
    local policy="$GOLEM_ROOT/core_ws/src/g1_homie/policy"
    local name
    for name in deploy.onnx g1.yaml homie_policy.py; do
        require_file "$policy/$name"
    done
}

prepare_mounts() {
    local service=$1
    case "$service" in
        ros) require_h12_model; require_g1_model ros; require_g1_policy ;;
        robocasa) [ "${GOLEM_ROBOT:-h12}" != g1 ] || require_g1_model robocasa ;;
    esac
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
            mkdir -p "$GOLEM_CACHE_DIR/msgs_ws/src/magpie_msgs" "$GOLEM_CACHE_DIR/msgs_ws/src/custom_ros_messages"
            ;;
        ros)
            require_file "$GOLEM_ROOT/core_ws/src/golem_bringup/package.xml"
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
