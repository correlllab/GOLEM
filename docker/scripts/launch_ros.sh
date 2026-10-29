#!/bin/bash
# Build and source core_ws, then execute a command (default: bash).
# core_ws is bind-mounted from the host; build/install/log persist there too.
set -e

source /opt/ros/humble/setup.bash
if [ ! -f /opt/unitree_install/lib/libunitree_sdk2.a ]; then
    echo '[launch_ros] missing baked Unitree SDK; rebuild with docker/scripts/docker_build.sh ros' >&2
    exit 1
fi

# Hydrate the persistent MJPC build tree once. Source timestamps cannot prove
# equivalence with the image: local edits may exist even at the same git HEAD.
MJPC_SRC=/home/code/mujoco_mpc
MJPC_BUILD=/home/code/mujoco_mpc/build
MJPC_SEED=/opt/mjpc-build-seed
if [ -f "$MJPC_SRC/CMakeLists.txt" ] && [ ! -e "$MJPC_BUILD/CMakeCache.txt" ] \
   && [ -d "$MJPC_SEED" ]; then
    echo "[launch_ros] hydrating MJPC build cache from seed ($MJPC_SEED -> $MJPC_BUILD)"
    mkdir -p "$MJPC_BUILD"
    cp -a "$MJPC_SEED/." "$MJPC_BUILD/"
fi

# The marker also migrates persistent caches created by launchers that backdated
# sources. Only successful input invalidation writes it; build failures remain
# retryable because source mtimes still exceed the old objects.
if [ -f "$MJPC_SRC/CMakeLists.txt" ] && [ -e "$MJPC_BUILD/CMakeCache.txt" ] \
   && [ ! -f "$MJPC_BUILD/.golem-source-revalidated-v1" ]; then
    # Revalidate every mounted input on first use, including dirty/untracked
    # files and sources checked out before the image was built. Keep the seeded
    # dependency build tree; subsequent launches use normal incremental builds.
    find "$MJPC_SRC" \( -path "$MJPC_BUILD" -o -name .git \) -prune -o \
         -exec touch -h {} +
    touch "$MJPC_BUILD/.golem-source-revalidated-v1"
fi

# --- MJPC incremental rebuild (ninja no-op scan when already warm) ---
# Brings libmjpc/threadpool + the staged task assets + the dist-packages
# agent_server current with the mounted submodule BEFORE colcon links
# h12_deploy_mjpc against the build tree (after a hydrate the tree is at the
# image's MJPC_REF; source inputs above require revalidation).
if [ -f "$MJPC_SRC/CMakeLists.txt" ] && [ -e "$MJPC_BUILD/CMakeCache.txt" ]; then
    /home/code/h12_sim_scripts/rebuild_mjpc.sh
fi

WS=/home/code/core_ws
cd "$WS"

if [ ! -d src ] || [ -z "$(ls -A src 2>/dev/null)" ]; then
    echo "[launch_ros] $WS/src is empty — did you forget 'git submodule update --init --recursive'?"
fi

# Let colcon/CMake check all inputs on every launch: package.xml timestamps do
# not account for C++, headers, interfaces, launch files, or build configuration.
# The upstream Livox build.sh deletes workspace caches and masks colcon errors.
# Select its ROS 2 manifest in a private copy, preserving the mounted submodule.
LIVOX_DIR="$WS/src/livox_ros_driver2"
COLCON_PATHS=(src)
if [ -f "$LIVOX_DIR/package_ROS2.xml" ]; then
    STAGE_ROOT="$WS/build/.golem-src"
    LIVOX_STAGE="$STAGE_ROOT/livox_ros_driver2"
    mkdir -p "$STAGE_ROOT"
    touch "$STAGE_ROOT/COLCON_IGNORE"
    # Timestamp-preserving checkouts/copies can change content without making
    # CMake inputs newer. Compare the small driver tree before refreshing it;
    # invalidate only its package outputs when content changes. package.xml is
    # selected from package_ROS2.xml below, so the original variant is irrelevant.
    if [ -d "$LIVOX_STAGE" ] && \
       ! diff -qr --exclude=.git --exclude=package.xml "$LIVOX_DIR" "$LIVOX_STAGE" >/dev/null; then
        rm -rf "$WS/build/livox_ros_driver2" "$WS/install/livox_ros_driver2"
    fi
    rm -rf "$LIVOX_STAGE"
    mkdir -p "$LIVOX_STAGE"
    cp -a "$LIVOX_DIR/." "$LIVOX_STAGE/"
    cp -p "$LIVOX_STAGE/package_ROS2.xml" "$LIVOX_STAGE/package.xml"
    # A CMake cache is tied to its absolute source directory. Discard only the
    # driver's old cache/install when migrating from the mounted source path.
    LIVOX_CACHE="$WS/build/livox_ros_driver2/CMakeCache.txt"
    if [ -f "$LIVOX_CACHE" ] && \
       ! grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=$LIVOX_STAGE" "$LIVOX_CACHE"; then
        rm -rf "$WS/build/livox_ros_driver2" "$WS/install/livox_ros_driver2"
    fi
    COLCON_PATHS=("$LIVOX_STAGE")
    for package in "$WS"/src/*; do
        [ -d "$package" ] || continue
        [ "$package" = "$LIVOX_DIR" ] || COLCON_PATHS+=("$package")
    done
fi

# colcon does not remove isolated install prefixes when packages disappear.
# Refuse to source stale code until the user explicitly cleans those caches.
CURRENT_PACKAGES=$(colcon list --names-only --base-paths "${COLCON_PATHS[@]}")
for marker in "$WS"/install/*/share/colcon-core/packages/*; do
    [ -f "$marker" ] || continue
    package=${marker##*/}
    if ! grep -Fxq -- "$package" <<< "$CURRENT_PACKAGES"; then
        echo "[launch_ros] stale installed package '$package' has no discovered source; remove its core_ws/build/$package and core_ws/install/$package caches, then retry" >&2
        exit 1
    fi
done

echo "[launch_ros] colcon build (incremental)"
colcon build --symlink-install --base-paths "${COLCON_PATHS[@]}" \
    --cmake-args -DROS_EDITION=ROS2 -DDISTRO_ROS=humble

source install/setup.bash

if [ "${GOLEM_WAIT_FOR_SIM:-0}" = 1 ]; then
    python3 /home/code/h12_sim_scripts/wait_for_sim.py
fi
[ "$#" -gt 0 ] || set -- bash
exec "$@"
