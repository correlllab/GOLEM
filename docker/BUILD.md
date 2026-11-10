# GOLEM build system

GOLEM (Generalized Open Layered Embodied Modules) builds into **four Docker
images** driven by two independent build systems that meet at runtime:

1. **Docker** builds the images (system deps, Python envs, C++ libraries baked at
   image-build time).
2. **colcon** builds the ROS 2 workspace(s) at *container start*, from
   bind-mounted source, into a host-persisted cache.

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
| `CL_Assets` | both robots' URDFs, MJCFs and meshes (the G1's in `g1/`), Isaac USD (Git-LFS) |
| `CL_isaaclab_sim` | Isaac Sim task/runtime code |
| `unitree_sdk2_python` | Unitree DDS SDK (Python) |

**`core_ws/src` submodules:** `cl_realsense`, `custom_ros_messages`, `estop`,
`livox_ros_driver2`, `magpie_control`, `magpie_msgs`,
`unitree_ros2`.

**`core_ws/src` in-tree packages** (versioned directly in this repo, *not*
submodules): `FAST_LIO`, `golem_ros2_models`, `golem_safety`, `golem_upper_ik`,
`golem_lowerbody_rl`, `golem_slam`, `golem_skills`, `golem_bringup`, `lmg_deploy`,
`model_server`, and the G1's own package `dex_3`.

`golem_safety` (C++, `ament_cmake`), `golem_upper_ik` and `golem_lowerbody_rl`
(Python) are the safety layer, arm controller and leg controller of both
robots, so they build for both; the
`ros` image has `golem_safety`'s build dependencies (`rclcpp`, `libyaml-cpp-dev`,
`zlib1g-dev`).

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
`ROS_DOMAIN_ID=1`; domain `0` is reserved for the real robots), one robot
stack per domain.

---

## 3. `golem_base` (`docker/BaseDockerfile`)

The shared foundation for the ROS and RoboCasa images: Ubuntu 22.04 on the
digest-pinned CUDA 12.2 devel image, the dated Ubuntu and ROS apt snapshots
(`docker/locks/apt-sources.sh`), ROS 2 Humble's `ros-base` with the CycloneDDS
RMW, CycloneDDS 0.10.x built from source (the PyPI `cyclonedds` binding that
`unitree_sdk2_python` needs includes a header removed after 0.10), the Unitree
Python SDK from the submodule, PyTorch cu130 (native Blackwell support, its own
CUDA runtime) and the kinematics libraries. The Dockerfile's comment above each
step says what it installs and why; the package lists are there, not here.

---

## 4. `golem_ros` (`docker/RosDockerfile`)

The workspace image, `FROM golem_base`. Layers are ordered **most-stable →
least-stable** so that iterating on volatile pins doesn't bust the heavy layers
above. In order:

1. **apt:** the colcon / ament / rosidl build system, the C++ libraries the
   workspace links, the ROS packages it builds against, and apt `pinocchio`.
2. **Livox-SDK2 from source**, which `livox_ros_driver2` links, at the commit
   every image uses.
3. **Controller / IK pip stack:** base's NumPy-2 pinocchio closure is removed,
   then `numpy<2` binds the C extensions to the 1.x ABI of apt's scipy and
   pinocchio.
4. **Vision ML stack** (exact-pinned) and CLIP at a fixed commit.
5. **SAM3** at a fixed commit, on `PYTHONPATH` (its `setup.py` builds a bogus
   package name, so it is not pip-installed).
6. **The setuptools / wheel / numpy clamp** (§9).
7. **Tkinter, Nav2 / SLAM / rqt / teleop / rosbag2.**
8. **GraspGenX** (NVlabs): its wheel is built against a newer setuptools in a
   staging directory (the system setuptools 59.6 predates its PEP 621
   metadata) and installed `--no-deps`; the whole environment is then frozen as
   an *additive constraint*, so its dependencies, `diffusers` (with an older
   `huggingface-hub`) and `viser` install without moving anything installed.
9. **The RealSense driver, OctoMap and the G1's additions**, the G1's through
   lock step `ros-amd64-11.txt`.
10. **The clamp again**, last: the GraspGenX dependency step moves setuptools.

Each step's comment in the Dockerfile names its packages and why each is there.

**Runtime build:** `launch_ros.sh` runs `colcon build --symlink-install` on
`core_ws` **at container start** (see §7). The image bakes only the *toolchain* and
Python deps; the ROS packages themselves are compiled from the bind-mounted source.

---

## 5. `golem_sim_robocasa` (`docker/RobocasaDockerfile`)

Single-stage, `FROM golem_base`. Provides the MuJoCo + RoboCasa kitchen simulator.

- `MUJOCO_GL=egl` baked in, for shells in the container too; `launch_robocasa.sh`
  exports `egl` in both display modes. The sims render the cameras and the
  MID-360's rays through EGL on the NVIDIA GPU and stop at start without one
  (`golem_robocasa/sim_common.py`'s `require_gpu`); the passive viewer opens
  its own GLFW window.
- **`mujoco==3.17.0`**, `numpy>=2.2.6`, Pillow. RoboCasa's setup.py pins
  `mujoco==3.3.1`, so the first step installs 3.3.1 and the last step
  (lock `robocasa-amd64-05.txt`) replaces it with 3.17.0, the newest MuJoCo
  with a Python 3.10 wheel (4.0 drops 3.10). That step also applies
  `docker/patches/robosuite-mujoco-3.17.0.patch` and
  `robocasa-mujoco-3.17.0.patch` with `git apply`, at the pinned commits, so a
  patch that no longer applies fails the build; each file's header says why it
  is needed. `pip check` still reports robosuite's `mujoco<3.10` and
  RoboCasa's `mujoco==3.3.1` requirements, which the image overrides.
  `golem_robocasa` calls 3.17.0's API directly.
- **RoboCasa + robosuite** installed from git at fixed commits (neither fully on
  PyPI). Also `robosuite_models`, `mimicgen`, `mink==0.0.5` — installed only to
  silence import-time "not installed" warnings.
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
  `build.sh humble`. At the pinned driver commit `build.sh` runs a plain
  `colcon build` (no `--symlink-install`) and ends in `popd`, so it exits 0
  even when that build fails (`CLAUDE.md` §8). `build/` is left in place.
- The G1 sim reads none of HOMIE's files, so this image has no onnxruntime.
  The G1 sim's robot description arrives through the base file's `CL_Assets`
  bind mount (§7), not the image.

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
to relocate them; caches regenerate on first launch.

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

## 7. Build & run orchestration

**Build** — `docker/scripts/docker_build.sh [--pull] [--no-cache] [service ...]`.
Defaults to all three services. Builds `golem_base` first for ROS/RoboCasa.
`--pull` refreshes public
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

**The G1 overlay.** Simulator and stack wrappers select the robot with
`--g1`/`--h12` anywhere in their arguments, else `GOLEM_ROBOT` (shell, then
`docker/.env`), else `h12` (`docker_run.sh ros` takes no `--g1` or `--h12`
flag). For the G1 they add `docker/docker-compose.g1.yml` after
`docker-compose.yml`:

- the robocasa service gets `GOLEM_ROBOT=g1` and the G1 sim's knobs
  (`CLAUDE.md` §8);
- the ros service needs no overlay: the base file mounts `tools/` (at
  `/home/code/tools`) into every ros container, read-only with
  `create_host_path: false`, it uses `core_ws/cyclonedds.xml` directly, the
  G1's model is in `CL_Assets/g1`, which the base file mounts for both robots,
  and HOMIE's weights are installed by `golem_lowerbody_rl` from the
  `core_ws` mount.

`prepare_mounts` checks the `CL_Assets/g1` files each service reads for missing
files and LFS pointers before any container starts: in every ros container,
whose `golem_ros2_models` installs both robots' models, and in the robocasa
service for the G1.
`launch_ros.sh` builds every package, the G1's included, whatever
the robot: one `ros` container runs either robot's bringup. For the G1,
`docker_stack.sh` sets `GOLEM_BRINGUP_LAUNCH=g1_sim.launch.py` for the stack
overlay, and `wait_for_sim.py` refuses a simulator of the other robot. The G1 uses the same
images and container names as the H1-2, and the wrappers refuse it with Isaac.

Compose is headless by itself. The wrapper selects
`docker-compose.gui.yml` when `GOLEM_DISPLAY=gui`, or automatically when
`DISPLAY` is set. GUI mode requires an Xauthority file; it does not change the
host's X access policy. `--headless` avoids desktop mounts entirely. Every
service uses the NVIDIA runtime and host networking.

`.dockerignore` restricts build context to sources actually copied by the
Dockerfiles: the Unitree Python SDK and Livox driver. Simulation source, model
weights, and assets stay outside the context and arrive through bind mounts.

---

## 8. The colcon workspace(s)

colcon is invoked **at container start**, never at image-build time, against
bind-mounted source, with output persisted on the host:

| Container | Workspace | Built | Cache |
|---|---|---|---|
| `ros` | `core_ws` (full, both robots' packages) | every start, incremental every launch | host `core_ws/{build,install,log}` |
| `robocasa` | `msgs_ws` (IDL only) | every start (fast no-op) | host `container_cache/msgs_ws` |

`--symlink-install` is used throughout so Python nodes and model weights resolve
via the install symlinks without a manual copy. Because build/install/log live on
host-side bind mounts, incremental rebuilds across `docker compose run --rm` cycles
are near-instant; wipe the host dir for a clean rebuild.

---

## 9. Cross-cutting invariants & gotchas

- **`setuptools==59.6.0`, `wheel<0.44`, `numpy<2`** are re-clamped after any pip
  step that bumps them — required for colcon's `python_setup_py`, torch, and the
  pinocchio/scipy ABI. Watch for `_distutils_hack` errors (a stale
  `distutils-precedence.pth`).
- **CycloneDDS is pinned to 0.10.x from source** in every base/Isaac image (PyPI
  wheel needs the removed `q_radmin.h`).
- **MuJoCo is pinned per image:** `3.17.0` in `robocasa` (with the patches in
  `docker/patches/`, §5); `ros` keeps `golem_base`'s, which nothing the
  workspace runs imports (HOMIE's vendored `mujoco_deploy_g1.py` does; nothing
  runs it).
- **Layer-order discipline:** heavy/stable layers first; volatile pins appended
  last. New deps go at the *end* of a Dockerfile so they don't bust cached layers;
  in `RosDockerfile`, before the final setuptools/wheel/numpy clamp, which stays
  the last pip step (`tests/docker/test_source_pins.py`).
- **CUDA wheel split:** `ros`/base use torch **cu130**; Isaac inherits NVIDIA's
  bundled stack; RoboCasa resolves its own torch through `lerobot`. Inspect
  the built images when diagnosing GPU/torch compatibility.

---

## 10. Common commands

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
ros2 launch golem_bringup h12_sim.launch.py
colcon build --symlink-install                   # force a rebuild
```
