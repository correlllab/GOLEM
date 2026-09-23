# GOLEM build system

GOLEM (Generalized Open Layered Embodied Modules) builds into **four Docker
images** driven by two independent build systems that meet at runtime:

1. **Docker** builds the images (system deps, Python envs, C++ libraries baked at
   image-build time).
2. **colcon** builds the ROS 2 workspace(s) at *container start*, from
   bind-mounted source, into a host-persisted cache.

A third, standalone **CMake** build (MuJoCo MPC) is layered on top of the `ros`
image with a dev loop that lets you rebuild it in-container without rebuilding
the image — see [MuJoCo MPC dev loop](#mujoco-mpc-mjpc-dev-loop).

The guiding principle throughout: **bake stable/heavy things into the image; keep
fast-moving source on the host and bind-mount it in.** Almost every dependency is
a git submodule, and almost every source tree is mounted, not copied — so the
Docker build context stays tiny and iterating on code never requires a rebuild.

---

## 1. Source topology

Nearly everything is a git submodule. `git submodule update --init --recursive`
is required before anything builds.

**Top-level submodules**

| Path | Purpose |
|---|---|
| `CL_Assets` | URDF meshes, MuJoCo XML, Isaac USD (Git-LFS) |
| `CL_isaaclab_sim` | Isaac Sim task/runtime code |
| `unitree_sdk2_python` | Unitree DDS SDK (Python) |
| `mujoco_mpc` | MuJoCo MPC fork (`badinkajink @ extended_hw_patched`) — see §7 |

**`core_ws/src` submodules:** `cl_realsense`, `custom_ros_messages`, `estop`,
`h12_ros2_controller`, `h12_ros2_model`, `h12_safety_layer`, `livox_ros_driver2`,
`magpie_control`, `magpie_msgs`, `unitree_ros2`.

**`core_ws/src` in-tree packages** (versioned directly in this repo, *not*
submodules): `FAST_LIO`, `h12_deploy_mjpc`, `h12_lowerbody_rl`,
`h12_skills`, `h1_bringup`, `model_server`.

Large binary assets (meshes, XML, USD) are tracked with **Git-LFS**; run
`git lfs install` first. Some weights are *not* in git and are fetched manually
(e.g. SAM3 `sam3.pt`, GraspGenX checkpoints) — see `docs/SETUP.md`.

---

## 2. Image graph

```
nvidia/cuda:12.2.0-devel-ubuntu22.04
        │
        ▼
   golem_base  ───────────┬───────────────┐
   (ROS 2 Humble,         │               │
    Python 3.10, torch    ▼               ▼
    cu130, CycloneDDS   golem_ros     golem_sim_robocasa
    0.10, unitree SDK)  (workspace)   (MuJoCo + RoboCasa)

nvcr.io/nvidia/isaac-sim:5.1.0
        │
        ▼
   golem_sim_isaac  (Isaac Sim's bundled Python + pinned IsaacLab)
```

- `golem_ros` and `golem_sim_robocasa` inherit from `golem_base`.
- `golem_sim_isaac` is **self-contained**, based on NVIDIA's Isaac Sim image.
  Its bundled interpreter and ROS bridge are separate from the Humble/Python 3.10
  stack in `golem_base`. The Dockerfile also installs ROS Jazzy tooling.

All containers interoperate over **CycloneDDS on one ROS domain** (default
`ROS_DOMAIN_ID=1`; domain `0` is reserved for the real robot).

---

## 3. `golem_base` (`docker/BaseDockerfile`)

The shared foundation for the ROS and RoboCasa images.

- **Base:** `nvidia/cuda:12.2.0-devel-ubuntu22.04`; env pins `ROS_DISTRO=humble`,
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, US apt mirror.
- **apt:** `gcc-12`/`g++-12` (set as default via update-alternatives), cmake,
  git/git-lfs, Python 3.10 (`python-is-python3`), and the X11/GL/Vulkan/EGL/OSMesa
  runtime libs both simulators dlopen.
- **ROS 2 Humble:** `ros-humble-ros-base` + `rmw-cyclonedds-cpp` + a few msg/TF
  packages (`sensor-msgs-py`, `tf2-ros`, `geometry-msgs`, `cv-bridge`, …).
- **uv:** installed to `/usr/local/bin` for fast, deterministic system pip installs.
- **CycloneDDS 0.10.x from source** → `CYCLONEDDS_HOME=/cyclonedds/install`. Pinned
  because the PyPI `cyclonedds` wheel (a transitive dep of `unitree_sdk2_python`)
  references `dds/ddsi/q_radmin.h`, which was removed after 0.10.
- **`unitree_sdk2_python`:** `COPY`'d from the local submodule checkout, installed
  `uv pip --system`, then its sub-packages copied into site-packages.
- **PyTorch cu130** (`torch`, `torchvision`) — the CUDA-13 wheels have native
  `sm_120`/Blackwell (RTX 5070 Ti) support and bundle their own CUDA runtime.
- **Kinematics libs:** `pin` (pinocchio), `pink`, `mink` — shared IK stack.
- Build-time `http(s)_proxy` ARGs are cleared at the end so they don't leak into
  child images.

---

## 4. `golem_ros` (`docker/RosDockerfile`)

The workspace image. Layers are ordered **most-stable → least-stable** so that
iterating on volatile pins doesn't bust the heavy layers above. `FROM golem_base`.

1. **apt build/workspace deps:** `colcon`, `rosdep`, `vcstool`; apt
   numpy/scipy/yaml/transforms3d (must match system C-extension ABIs); C++ libs
   (`libpcl-dev`, `libeigen3-dev`, `libyaml-cpp-dev`, `libopencv-dev`); the
   ament/rosidl build system; many `ros-humble-*` message/description/PCL/TF/RViz
   packages; and **apt `pinocchio`** (C++ symbols) alongside base's pip `pin`.
2. **Livox-SDK2 from source** — required by `livox_ros_driver2`. Placed before the
   Python layers so pip iteration doesn't rebuild it.
3. **Controller / IK pip stack:** `numpy<2` forced first (binds C-extensions to the
   1.x ABI), `pin-pink`, `qpsolvers` + `quadprog`/`proxsuite`, `meshcat`, `open3d`, …
4. **Vision ML stack** (exact-pinned): `transformers==4.47.1`, `google-genai`,
   `timm`, `ultralytics`, `opencv-python-headless`, CLIP (from a git SHA), etc.
5. **SAM3** — cloned to `/opt/sam3` and exposed via `PYTHONPATH` (its `setup.py`
   produces a bogus version, so it is *not* pip-installed).
6. **setuptools/wheel/numpy clamp:** restores `numpy<2`, `setuptools==59.6.0`,
   `wheel<0.44` (later pip steps bump them). This clamp recurs several times in the
   file — colcon's `python_setup_py` and torch require `setuptools<80`, and the
   80+-era `distutils-precedence.pth` otherwise spams `_distutils_hack` errors.
7. **Nav2 / SLAM / rqt / teleop / rosbag2 / foxglove** — appended late.
8. **GraspGenX** (NVlabs) — installed in-place with an elaborate constraint dance:
   build a modern-setuptools wheel in a staging dir (the system setuptools 59.6 is
   too old for its PEP 621 metadata), install `--no-deps`, freeze the whole env as
   an *additive constraint*, then add deps + downgrade `huggingface-hub` for
   `diffusers`, and add `viser` — all pinned so nothing already-installed moves.
9. **MuJoCo MPC** — see §7.

**Runtime build:** `launch_ros.sh` runs `colcon build --symlink-install` on
`core_ws` **at container start** (see §8). The image bakes only the *toolchain* and
Python deps; the ROS packages themselves are compiled from the bind-mounted source.

---

## 5. `golem_sim_robocasa` (`docker/RobocasaDockerfile`)

Single-stage, `FROM golem_base`. Provides the MuJoCo + RoboCasa kitchen simulator.

- `MUJOCO_GL=egl` baked as default; `launch_robocasa.sh` forces `glfw` for a
  windowed viewer unless `--headless` (then `egl` for offscreen GPU rendering).
- **`mujoco==3.3.1`** (pinned to match RoboCasa's hard pin), `numpy>=2.2.6`, Pillow.
- **RoboCasa + robosuite** installed from git (neither fully on PyPI; robosuite
  from `master`). Also `robosuite_models`, `mimicgen`, `mink==0.0.5` — installed
  only to silence import-time "not installed" warnings.
  - *Side effect:* `lerobot` (a RoboCasa transitive) pins `torch==2.7.1`, which
    downgrades base's cu130 torch to a CPU/cu121 build **in this image only**
    (accepted; Isaac uses its own NVIDIA image).
- **Kitchen assets (~10 GB)** downloaded at build time so the container is
  ready-to-run (comment out that `RUN` to fetch them manually instead).
- **`msgs_ws` toolchain** (colcon + ament + rosidl) + an empty `/home/code/msgs_ws/src`
  mountpoint — this container builds *only* the IDL packages (`magpie_msgs`,
  `custom_ros_messages`) at start, not the full `core_ws`.
- **Livox baked at `/opt/livox_ws`:** `livox_ros_driver2` is `COPY`'d (not
  bind-mounted — a mount would collide with the `ros` container's `core_ws/src`
  mount, since `build.sh` rewrites its own source tree) and built via upstream
  `build.sh humble`. `build/` is deliberately *not* wiped afterward because
  `--symlink-install` leaves the generated message modules symlinked into it.

---

## 6. `golem_sim_isaac` (`docker/IsaacDockerfile`)

The main stage is based on `nvcr.io/nvidia/isaac-sim:5.1.0`. A separate Humble
stage packages the Magpie gripper runtime and message interfaces. IsaacLab is checked out
at `b4c321024792976150ca55fddb26fa34480d974e`; its `_isaac_sim` link points to
`/isaac-sim`. Run simulator Python through that installation's `python.sh`,
which sets Kit's required environment. The image installs CycloneDDS 0.10.x,
the Unitree Python SDK, simulator dependencies, and ROS Jazzy tooling.

Source and assets are bind-mounted. Persistent caches default to
`container_cache/isaac/{ov,nvidia,local_share_ov,isaacsim}`. Set `GOLEM_CACHE_DIR`
to relocate them. Existing `CL_isaaclab_sim/.isaac_cache` contents may be copied
there with containers stopped; otherwise caches regenerate on first launch.

`launch_isaac.sh` accepts a task name or `--task NAME`, `--headless`, and
`--reset-cache`, `--hand_type magpie|inspire`, and `--fix_base` for a bench
fixture. Magpie is the default floating-base robot. Output uses
normal container stdout/stderr. The image has no shell entrypoint, so command
overrides such as `docker_run.sh isaac bash` work normally.

The simulator's Unitree DDS manager reads `ROS_DOMAIN_ID` (default 1), rejects
0, and communicates directly with the ROS stack on that domain. There is no
cross-domain relay. This requires the accompanying `CL_isaaclab_sim` source
change; distribute it through the submodule repository before updating the
superproject pin. The Magpie gripper process uses private Humble Python 3.10 and the
same DDS domain, while the main simulator uses Isaac's bundled interpreter.
Humble's discovery schema matches the ROS container and bundled sensor bridge.
The original `magpie_msgs` source is built into the image; rebuild Isaac when
those interfaces change. The helper runtime leaves Isaac's Python and the
container's libc unchanged.

---

## 7. MuJoCo MPC (MJPC) dev loop

MJPC is a CMake/FetchContent project (it fetches and builds its own MuJoCo 3.2.3 +
gRPC + abseil + glfw). It is **built standalone, outside colcon** — the thin ROS
bridge `h12_deploy_mjpc` is the only mjpc-related colcon package.

**At image build** (`RosDockerfile` MJPC block): the `mujoco_mpc` submodule's
pinned commit is cloned (step 2); then `agent_server` is compiled with
clang-13/Ninja (step 4), the importable
`mujoco_mpc` package is installed into `dist-packages` (the durable import target),
the build's own `libmujoco.so` is staged into `/usr/local/lib`, and — instead of
deleting `build/` — the whole warm build tree is stashed to `/opt/mjpc-build-seed`.
The source clone is then removed (the submodule bind mount re-supplies it).

The three build fixes that used to be `sed`'d in at build time now live as **real
commits on the fork's `extended_hw_patched` branch** (drop `ui_agent_server`,
magpie `patch`→`copy`, case-insensitive `.STL` globs), so the baked build and the
mounted source are byte-identical — a requirement for the warm cache below.

**At runtime:** `docker-compose` mounts the submodule source over
`/home/code/mujoco_mpc` and a persistent build cache
(`container_cache/mjpc_build`) at the exact in-tree build path. On first launch
`launch_ros.sh` hydrates the cache from the seed and refreshes source timestamps to avoid
reusing stale objects for local edits. The first `docker exec … rebuild_mjpc.sh` revalidates source objects while retaining built dependencies; subsequent launches
use normal incremental builds. `rebuild_mjpc.sh` rebuilds `agent_server` and copies it
into `dist-packages` (the path `from mujoco_mpc import agent` auto-spawns).

> The submodule gitlink and `MJPC_REF` are pinned to the **same** patched-fork SHA.
> Bumping mjpc = rebase the patches, push, move both pins, rebuild the image.

---

## 8. Build & run orchestration

**Build** — `docker/scripts/docker_build.sh [--pull] [--no-cache] [service ...]`.
Defaults to all three services. Builds `golem_base` first for ROS/RoboCasa;
propagates the committed MJPC pin into the ROS build. `--pull` refreshes public
upstream images (base and Isaac); child builds use the freshly built local
`golem_base`, rather than attempting to pull that private local tag.

**Run one service** — `docker/scripts/docker_run.sh SERVICE [--restart] [cmd...]`.
A leading flag invokes that service's launcher. Existing containers are never
removed implicitly; `--restart` explicitly replaces the selected container.

**Run a pair** — `docker/scripts/docker_stack.sh up|restart|stop|logs SIM`.
The startup command builds the ROS and RoboCasa message workspaces without
starting physics, then starts the simulator and ROS bringup. ROS waits for a
fresh `rt/lowstate` sample on the configured domain. `GOLEM_SIM_TIMEOUT` bounds
that wait; a timeout fails ROS startup and directs users to simulator logs.
`restart` stops both services before recreating them. `stop` preserves caches.

Both wrapper families source `docker/scripts/docker_common.sh`. They explicitly
load `docker/.env`, preserving exported shell overrides, validate the domain,
check required source paths, and create only cache/mountpoint directories.
`GOLEM_ASSETS_DIR` and `GOLEM_CACHE_DIR` accept absolute paths or paths relative
to the checkout. Required bind sources use `create_host_path: false`.

Linux Compose is headless by itself. The wrapper selects
`docker-compose.gui.yml` when `GOLEM_DISPLAY=gui`, or automatically when
`DISPLAY` is set. GUI mode requires an Xauthority file; it does not change the
host's X access policy. `--headless` avoids desktop mounts entirely. NVIDIA
runtime and host networking remain in the Linux configuration.

The Mac wrappers expose the same build/run/stack operations for `robocasa` and
`ros` only. Images use the `:arm64` tag and `linux/arm64` platform. Mac uses
FastDDS, host networking/IPC, and native VM named volumes for colcon build and
install trees. Keep those named volumes: host filesystem shares make symlink
builds slow. VNC is opt-in; there are no NVIDIA or host X11 requirements.

The ROS image bakes the pinned Unitree C++ SDK at `/opt/unitree_install`.
No SDK clone or SDK-cache mount is needed at startup. The separate MJPC warm
build cache and source mounts remain for controller development.

`.dockerignore` restricts build context to sources actually copied by the
Dockerfiles: the Unitree Python SDK and Livox driver. Simulation source, model
weights, and assets stay outside the context and arrive through bind mounts.

---

## 9. The colcon workspace(s)

colcon is invoked **at container start**, never at image-build time, against
bind-mounted source, with output persisted on the host:

| Container | Workspace | Built | Cache |
|---|---|---|---|
| `ros` | `core_ws` (full) | every start, incremental every launch | host `core_ws/{build,install,log}` |
| `robocasa` | `msgs_ws` (IDL only) | every start (fast no-op) | host `container_cache/msgs_ws` |

`--symlink-install` is used throughout so Python nodes and model weights resolve
via the install symlinks without a manual copy. Because build/install/log live on
host-side bind mounts, incremental rebuilds across `docker compose run --rm` cycles
are near-instant; wipe the host dir for a clean rebuild.

---

## 10. Cross-cutting invariants & gotchas

- **`setuptools==59.6.0`, `wheel<0.44`, `numpy<2`** are re-clamped after any pip
  step that bumps them — required for colcon's `python_setup_py`, torch, and the
  pinocchio/scipy ABI. Watch for `_distutils_hack` errors (a stale
  `distutils-precedence.pth`).
- **CycloneDDS is pinned to 0.10.x from source** in every base/Isaac image (PyPI
  wheel needs the removed `q_radmin.h`).
- **MuJoCo versions are pinned per image and must not drift:** `3.2.3` in `ros`
  (matches the mjpc C++ server's ABI byte-for-byte), `3.3.1` in `robocasa` (matches
  RoboCasa's pin).
- **Layer-order discipline:** heavy/stable layers first; volatile pins appended
  last. New deps go at the *end* of a Dockerfile so they don't bust cached layers.
- **CUDA wheel split:** `ros`/base use torch **cu130**; Isaac inherits NVIDIA's
  bundled stack; RoboCasa resolves its own torch through `lerobot`. Inspect
  the built images when diagnosing GPU/torch compatibility.

---

## 11. Common commands

```bash
# one-time
git submodule update --init --recursive
git lfs install
cp docker/.env.example docker/.env      # set GEMINI_API_KEY, ROS_DOMAIN_ID

# build
docker/scripts/docker_build.sh              # base + all profiles
docker/scripts/docker_build.sh robocasa ros # subset

# run
docker/scripts/docker_run.sh robocasa            # windowed sim
docker/scripts/docker_run.sh ros                 # workspace shell (auto colcon build)
docker/scripts/docker_run.sh isaac

# inside the ros container
ros2 launch h1_bringup h1_sim_bringup.launch.py
colcon build --symlink-install                   # force a rebuild

# MJPC iteration (inside golem_ros)
docker exec -it golem_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh            # C++ edit
docker exec -it golem_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh --install  # + assets/proto
```
