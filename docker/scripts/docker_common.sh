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

# Selects the robot for service $1 and exports it as GOLEM_ROBOT (h12 or g1).
# Precedence: ROBOT_FLAG, then the shell, then docker/.env (see load_config).
# G1 adds its compose overlay; it runs on x86 Linux, never with Isaac. A set
# G1_DDS_INTERFACE must be a Linux interface name (at most IFNAMSIZ-1 = 15
# characters), because write_g1_dds_pin puts it into XML. It is meant for the
# ros container on domain 0 only: the sim's ROS nodes are never pinned, so on a
# sim domain it can split the ROS graph across interfaces.
configure_robot() {
    local service=$1 robot
    # Letters spelled out: a range like A-Z follows the locale's collation, which
    # can take in letters such as é.
    local ifname_re='^[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.@:-]{1,15}$'
    robot=$(printf '%s' "${ROBOT_FLAG:-${GOLEM_ROBOT:-h12}}" | tr '[:upper:]' '[:lower:]')
    case "$robot" in
        h12) ;;
        g1)
            [ "${GOLEM_PLATFORM:-linux}" != mac ] || fail 'G1 is not available on Mac; use an x86 Linux host.'
            [ "$service" != isaac ] || fail 'G1 is not available with the Isaac simulator; use robocasa.'
            if [ -n "${G1_DDS_INTERFACE:-}" ]; then
                [[ $G1_DDS_INTERFACE =~ $ifname_re ]] || \
                    fail "G1_DDS_INTERFACE=$(printf '%q' "$G1_DDS_INTERFACE") is not a network interface name (letters, digits and _ . @ : -, at most 15 characters; see: ip -br link)."
                [ "$ROS_DOMAIN_ID" = 0 ] || \
                    echo "WARNING: G1_DDS_INTERFACE=$G1_DDS_INTERFACE pins only the ros container, for the real robot on domain 0; leave it empty for the sim (domain $ROS_DOMAIN_ID), whose ROS nodes may otherwise use another interface (docs/g1/dds.md)." >&2
            fi
            COMPOSE+=(-f "$GOLEM_ROOT/docker/docker-compose.g1.yml")
            ;;
        *) fail "GOLEM_ROBOT must be h12 or g1, not '$robot'.";;
    esac
    export GOLEM_ROBOT=$robot
}

# For the G1, writes $GOLEM_CACHE_DIR/cyclonedds_local.xml from
# G1_DDS_INTERFACE (empty: CycloneDDS chooses); the ros container loads it after
# core_ws/cyclonedds.xml. Call it only right before a ros container starts.
# The overlay bind-mounts the file into the ros container as a single file, which
# pins the inode the container started with, and CycloneDDS rereads it in every
# process that starts there. So the file is replaced by a rename, never written
# in place: a running container keeps the pin it started with.
write_g1_dds_pin() {
    [ "${GOLEM_ROBOT:-h12}" = g1 ] || return 0
    local pin=$GOLEM_CACHE_DIR/cyclonedds_local.xml tmp
    mkdir -p "$GOLEM_CACHE_DIR"
    tmp=$(mktemp "$pin.XXXXXX") || fail "Cannot create a temporary file beside $pin."
    if ! {
        if [ -n "${G1_DDS_INTERFACE:-}" ]; then
            printf '<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="%s"/></Interfaces></General></Domain></CycloneDDS>\n' \
                "$G1_DDS_INTERFACE" > "$tmp"
        else
            echo '<CycloneDDS/>' > "$tmp"
        fi && chmod 0644 "$tmp" && mv -f "$tmp" "$pin"
    }; then
        rm -f "$tmp"
        fail "Cannot write $pin."
    fi
}

# G1 on DDS domain 0 reaches the real robot, so the CycloneDDS interface pin
# must name a host port that is up and addressed, not whatever DDS would pick.
# Sets G1_DDS_IPV4 to the port's IPv4 addresses (CIDR, space-separated).
check_dds_interface() {
    if [ -z "${G1_DDS_INTERFACE:-}" ]; then
        fail "ROS_DOMAIN_ID=0 (the real robot) needs G1_DDS_INTERFACE, the laptop's wired port to"$'\n'"       the robot (ours: enp130s0), in docker/.env. Without it DDS may pick another interface."
    fi
    command -v ip >/dev/null 2>&1 || fail "'ip' (iproute2) is needed to check G1_DDS_INTERFACE on DDS domain 0."
    ip link show dev "$G1_DDS_INTERFACE" >/dev/null 2>&1 || \
        fail "G1_DDS_INTERFACE=$(printf '%q' "$G1_DDS_INTERFACE") is not a network interface on this host (see: ip -br link)."
    local link_state
    link_state=$(ip -br link show dev "$G1_DDS_INTERFACE" | awk '{print $2}') || true
    if [ "$link_state" != UP ]; then
        fail "G1_DDS_INTERFACE=$G1_DDS_INTERFACE is ${link_state:-in an unknown state}, not UP."$'\n'"       Check the cable to the robot: ip -br link show dev $G1_DDS_INTERFACE"
    fi
    G1_DDS_IPV4=$(ip -o -4 addr show dev "$G1_DDS_INTERFACE" | awk '{print $4}' | paste -sd ' ' -) || true
    if [ -z "$G1_DDS_IPV4" ]; then
        fail "G1_DDS_INTERFACE=$G1_DDS_INTERFACE has no IPv4 address (ours: 192.168.123.222/24)."$'\n'"       Check the cable and the port's address: ip -br addr show dev $G1_DDS_INTERFACE"
    fi
}

# The confirmation is printed whether or not stdin is a terminal, so a log shows
# which robot and port were confirmed.
check_domain() {
    if [ "$ROS_DOMAIN_ID" = 0 ]; then
        [ "$1" = ros ] && [ "${GOLEM_PLATFORM:-linux}" != mac ] || fail 'Domain 0 is reserved for the real robot.'
        local robot=${GOLEM_ROBOT:-h12} reply
        [ "$robot" != g1 ] || check_dds_interface
        {
            echo 'Domain 0 is the REAL ROBOT bus.'
            echo "  robot: $robot"
            [ "$robot" != g1 ] || echo "  interface: $G1_DDS_INTERFACE (IPv4 $G1_DDS_IPV4)"
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

# The files the G1 overlay bind-mounts: its robot description (a submodule) and
# the walking policy.
require_g1_files() {
    local description="$GOLEM_ROOT/third_party/leprechaun_g1/assets/g1_description"
    local policy="$GOLEM_ROOT/core_ws/src/g1/g1_locomotion/policy"
    local name
    for name in g1_29dof_with_hand_rev_1_0.urdf g1_29dof_with_hand_rev_1_0.xml; do
        [ -f "$description/$name" ] || fail "Missing $description/$name. Initialize the G1 robot description: git submodule update --init third_party/leprechaun_g1"
        require_file "$description/$name"
    done
    for name in deploy.onnx g1.yaml homie_policy.py; do
        require_file "$policy/$name"
    done
}

prepare_mounts() {
    local service=$1
    require_file "$GOLEM_ASSETS_DIR/ros_assets/h1_2_magpie_ros.urdf"
    if [ "${GOLEM_ROBOT:-h12}" = g1 ]; then
        case "$service" in robocasa|ros) require_g1_files;; esac
    fi
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
