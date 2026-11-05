#!/bin/bash
# The x86 ros container's command: build the bind-mounted core_ws (both robots'
# packages) incrementally, source it, then run a command. It does not start
# a bringup unless that is the command (docker-compose.stack.yml).
#
# Usage: launch_ros.sh [command...]   (default: bash)
# Environment:
#   GOLEM_WAIT_FOR_SIM  1: run wait_for_sim.py (GOLEM_ROBOT, ROS_DOMAIN_ID,
#                       GOLEM_SIM_TIMEOUT) after the build and before the command
# build/, install/ and log/ live in the host's core_ws, so builds persist across
# containers.
# Exit status: 1 without the baked Unitree SDK or with a stale installed
# package; colcon's or wait_for_sim.py's status when they fail; otherwise the
# command's.
set -e

source /opt/ros/humble/setup.bash
# deploy_mjpc links the Unitree C++ SDK the image installs (RosDockerfile).
if [ ! -f /opt/unitree_install/lib/libunitree_sdk2.a ]; then
    echo '[launch_ros] missing baked Unitree SDK; rebuild with docker/scripts/docker_build.sh ros' >&2
    exit 1
fi

# MuJoCo MPC's persistent build tree (docker-compose.yml mounts it inside the
# mujoco_mpc source): fill it from the image's warm seed when it is empty.
MJPC_SRC=/home/code/mujoco_mpc
MJPC_BUILD=/home/code/mujoco_mpc/build
MJPC_SEED=/opt/mjpc-build-seed
if [ -f "$MJPC_SRC/CMakeLists.txt" ] && [ ! -e "$MJPC_BUILD/CMakeCache.txt" ] \
   && [ -d "$MJPC_SEED" ]; then
    echo "[launch_ros] hydrating MJPC build cache from seed ($MJPC_SEED -> $MJPC_BUILD)"
    mkdir -p "$MJPC_BUILD"
    cp -a "$MJPC_SEED/." "$MJPC_BUILD/"
fi

# On a build tree's first use, touch every mounted source file once. The
# mounted source can differ from what the seed was built from even at the same
# commit (local edits, untracked files), and its mtimes prove nothing, so ninja
# must recompile everything that depends on it; the seeded dependencies stay
# built. The marker is written right after the touch: a later failed build
# still retries, because the sources stay newer than its objects.
if [ -f "$MJPC_SRC/CMakeLists.txt" ] && [ -e "$MJPC_BUILD/CMakeCache.txt" ] \
   && [ ! -f "$MJPC_BUILD/.golem-source-revalidated-v1" ]; then
    find "$MJPC_SRC" \( -path "$MJPC_BUILD" -o -name .git \) -prune -o \
         -exec touch -h {} +
    touch "$MJPC_BUILD/.golem-source-revalidated-v1"
fi

# Rebuild agent_server and the staged task assets from the mounted source
# (a no-op ninja scan when nothing changed) before colcon links deploy_mjpc
# against the build tree.
if [ -f "$MJPC_SRC/CMakeLists.txt" ] && [ -e "$MJPC_BUILD/CMakeCache.txt" ]; then
    /home/code/h12_sim_scripts/rebuild_mjpc.sh
fi

WS=/home/code/core_ws
cd "$WS"

if [ ! -d src ] || [ -z "$(ls -A src 2>/dev/null)" ]; then
    echo "[launch_ros] $WS/src is empty — did you forget 'git submodule update --init --recursive'?"
fi

# livox_ros_driver2 ships package_ROS1.xml and package_ROS2.xml, and upstream's
# build.sh, which picks one, deletes workspace caches and hides colcon errors.
# So colcon builds the driver from a private copy under build/.golem-src, with
# package_ROS2.xml as its package.xml, and the mounted submodule stays as it
# is. colcon checks every input on every start; package.xml timestamps alone
# would miss changed C++, headers, interfaces and launch files.
LIVOX_DIR="$WS/src/livox_ros_driver2"
COLCON_PATHS=(src)
if [ -f "$LIVOX_DIR/package_ROS2.xml" ]; then
    STAGE_ROOT="$WS/build/.golem-src"
    LIVOX_STAGE="$STAGE_ROOT/livox_ros_driver2"
    mkdir -p "$STAGE_ROOT"
    touch "$STAGE_ROOT/COLCON_IGNORE"
    # A checkout or copy that keeps timestamps can change content without
    # making CMake's inputs newer. Compare the driver tree with the last copy
    # and, when it differs, drop only the driver's build and install. The
    # copy's package.xml comes from package_ROS2.xml, so the submodule's own
    # package.xml is not compared.
    if [ -d "$LIVOX_STAGE" ] && \
       ! diff -qr --exclude=.git --exclude=package.xml "$LIVOX_DIR" "$LIVOX_STAGE" >/dev/null; then
        rm -rf "$WS/build/livox_ros_driver2" "$WS/install/livox_ros_driver2"
    fi
    rm -rf "$LIVOX_STAGE"
    mkdir -p "$LIVOX_STAGE"
    cp -a "$LIVOX_DIR/." "$LIVOX_STAGE/"
    cp -p "$LIVOX_STAGE/package_ROS2.xml" "$LIVOX_STAGE/package.xml"
    # A CMake cache is tied to its absolute source directory: drop the
    # driver's build and install when its cache names another source path.
    LIVOX_CACHE="$WS/build/livox_ros_driver2/CMakeCache.txt"
    if [ -f "$LIVOX_CACHE" ] && \
       ! grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=$LIVOX_STAGE" "$LIVOX_CACHE"; then
        rm -rf "$WS/build/livox_ros_driver2" "$WS/install/livox_ros_driver2"
    fi
    # The copy, and every other package directory in src/.
    COLCON_PATHS=("$LIVOX_STAGE")
    for package in "$WS"/src/*; do
        [ -d "$package" ] || continue
        [ "$package" = "$LIVOX_DIR" ] || COLCON_PATHS+=("$package")
    done
fi

# colcon never removes the install of a package that is gone from src/, and
# sourcing it would run stale code. Refuse to go on until those caches are
# deleted (CLAUDE.md §10 lists the commands).
CURRENT_PACKAGES=$(colcon list --names-only --base-paths "${COLCON_PATHS[@]}")
for marker in "$WS"/install/*/share/colcon-core/packages/*; do
    [ -f "$marker" ] || continue
    package=${marker##*/}
    if ! grep -Fxq -- "$package" <<< "$CURRENT_PACKAGES"; then
        echo "[launch_ros] stale installed package '$package' has no discovered source; remove its core_ws/build/$package and core_ws/install/$package caches, then retry" >&2
        exit 1
    fi
done

# The CMake arguments select livox_ros_driver2's ROS 2 / Humble build.
echo "[launch_ros] colcon build (incremental)"
colcon build --symlink-install --base-paths "${COLCON_PATHS[@]}" \
    --cmake-args -DROS_EDITION=ROS2 -DDISTRO_ROS=humble

source install/setup.bash

if [ "${GOLEM_WAIT_FOR_SIM:-0}" = 1 ]; then
    python3 /home/code/h12_sim_scripts/wait_for_sim.py
fi
[ "$#" -gt 0 ] || set -- bash
exec "$@"
