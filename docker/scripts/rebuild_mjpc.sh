#!/bin/bash
# Rebuild MuJoCo MPC's agent_server from the mounted mujoco_mpc submodule, in
# the persistent build tree, and install it where `from mujoco_mpc import
# agent` starts it. Runs inside the x86 ros container; launch_ros.sh runs it
# at every start (a no-op when nothing changed).
#
# Usage:
#   docker exec -it golem_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh            # a C++ edit
#   docker exec -it golem_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh --install  # + task assets, protos, libmujoco
# A C++ edit takes seconds: the tree was built warm in the image and restored
# by launch_ros.sh. --install also reinstalls the Python package, for changed
# task XMLs / meshes or Python API.
# Exit status: 1 without the mounted source or when no agent_server was built;
# otherwise the failing command's (set -e).
set -euo pipefail

MJPC_SRC=/home/code/mujoco_mpc
MJPC_BUILD=/home/code/mujoco_mpc/build      # in-tree; host copy in container_cache/mjpc_build
PKG=/usr/local/lib/python3.10/dist-packages/mujoco_mpc

[ -f "$MJPC_SRC/CMakeLists.txt" ] || {
    echo "[rebuild_mjpc] ERROR: no source at $MJPC_SRC — is the ../mujoco_mpc submodule mounted/checked out?" >&2
    exit 1
}

# 1. Configure only an empty build tree (launch_ros.sh normally restores the
#    seed, so this is skipped), with RosDockerfile's flags and generator, so
#    setup.py's build_ext never reconfigures. With _deps already present,
#    FetchContent must not download again.
if [ ! -f "$MJPC_BUILD/CMakeCache.txt" ]; then
    echo "[rebuild_mjpc] cold build dir — full configure (first build is ~15 min: gRPC/MuJoCo/abseil)"
    FC_FLAG=()
    [ -d "$MJPC_BUILD/_deps" ] && FC_FLAG=(-DFETCHCONTENT_FULLY_DISCONNECTED=ON)
    cmake -S "$MJPC_SRC" -B "$MJPC_BUILD" -G Ninja \
        -DCMAKE_C_COMPILER=clang-13 -DCMAKE_CXX_COMPILER=clang++-13 \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        -DMJPC_BUILD_GRPC_SERVICE=ON \
        -DMJPC_BUILD_TESTS=OFF -DMJPC_GRPC_BUILD_TESTS=OFF -DPYMJPC_BUILD_TESTS=OFF \
        "${FC_FLAG[@]}"
fi

# 1b. The seed's _deps have their .git directories trimmed, so a reconfigure
#     must never run FetchContent's git steps: git in a .git-less _deps
#     directory climbs to the submodule's read-only git metadata and fails.
#     FETCHCONTENT_FULLY_DISCONNECTED is set in the cache once. A dependency
#     that needs fetching means deleting container_cache/mjpc_build and a cold
#     build.
if [ -d "$MJPC_BUILD/_deps" ] && \
   ! grep -q '^FETCHCONTENT_FULLY_DISCONNECTED:BOOL=ON$' "$MJPC_BUILD/CMakeCache.txt"; then
    echo "[rebuild_mjpc] pinning FETCHCONTENT_FULLY_DISCONNECTED=ON into the warm cache"
    cmake -DFETCHCONTENT_FULLY_DISCONNECTED=ON "$MJPC_BUILD"
fi

# 2. Build agent_server incrementally, with the seed build's compilers.
echo "[rebuild_mjpc] cmake --build (incremental) --target agent_server"
CC=clang-13 CXX=clang++-13 CMAKE_GENERATOR=Ninja \
    cmake --build "$MJPC_BUILD" --target agent_server -j"$(nproc)"

# Find the binary: build/bin/agent_server, else wherever the build put it.
BIN="$MJPC_BUILD/bin/agent_server"
[ -x "$BIN" ] || BIN="$(find "$MJPC_BUILD" -name agent_server -type f -perm -u+x -print -quit)"
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "[rebuild_mjpc] ERROR: agent_server binary not found after build" >&2; exit 1; }

# 3. Copy it into the installed package: `from mujoco_mpc import agent` starts
#    <package>/mjpc/agent_server, so a C++ edit takes effect only once copied.
cp -a "$BIN" "$PKG/mjpc/agent_server"
echo "[rebuild_mjpc] refreshed $PKG/mjpc/agent_server  (from $BIN)"

# 4. --install: reinstall the package (task XMLs and meshes, generated protos).
if [ "${1:-}" = "--install" ]; then
    echo "[rebuild_mjpc] full setup.py install (assets + proto)"
    ( cd "$MJPC_SRC/python" \
      && CC=clang-13 CXX=clang++-13 CMAKE_GENERATOR=Ninja \
         python setup.py install --single-version-externally-managed \
                                 --record /tmp/mjpc-install-files.txt \
      && rm -f /tmp/mjpc-install-files.txt )
    # The build's own libmujoco: agent_server needs exactly the library it was
    # built against (RosDockerfile, MJPC step 4).
    cp -a "$MJPC_BUILD"/lib/libmujoco.so* /usr/local/lib/ && ldconfig
    # Clamp setuptools / wheel / numpy back (RosDockerfile step 6) only when
    # setuptools left 59.x, so a normal run needs no network.
    if ! python -c "import setuptools,sys; sys.exit(0 if setuptools.__version__.startswith('59.') else 1)"; then
        echo "[rebuild_mjpc] re-clamping setuptools/wheel to the colcon-safe pins"
        pip install --no-cache-dir --no-deps "numpy<2" "setuptools==59.6.0" "wheel<0.44"
        rm -f /usr/local/lib/python3.10/dist-packages/distutils-precedence.pth
    fi
    echo "[rebuild_mjpc] dist-packages mujoco_mpc fully refreshed"
fi

echo "[rebuild_mjpc] done — restart your MJPC node so it re-spawns the updated agent_server"
