# Audit 4: Docker/build infra, repo hygiene, docs, shared non-G1 packages

Scope: compare **G1_GOLEM `main` HEAD `df64d43`** with **GOLEM `camera_ready` HEAD `42d6559`**
(HEAD-state comparison, per the scope change). Common ancestors used only for attribution:

- `c00feca`: merge-base of GOLEM camera_ready and the spatial-memory branch.
- `c0af423` (GOLEM `origin/spatial-memory-robocasa-benchmark`), which is G1_GOLEM `9c8d5e0`: the G1 fork point.

So a difference is attributed to the **spatial-memory branch** (`c00feca..c0af423`), the **G1 work**
(`9c8d5e0..df64d43`) or **camera_ready** (`c00feca..42d6559`).

Legend: **[V]** means verified by reading the files or by a git command. **[I]** means inferred and not executed.
Nothing was built or run. Neither repo was modified.

---

## 0. Executive summary

1. **The two docker layers diverged in opposite directions.** [V]
   - camera_ready **locked every build input**:
     - Ubuntu and ROS apt snapshots;
     - FROM digests;
     - exact SHAs for every `git clone`;
     - per-step `pip -c /opt/golem-locks/*.txt` constraint files.
   - camera_ready also refactored the scripts into `docker_common.sh`, `docker_stack.sh` and `runtime_common.sh`.
   - camera_ready also renamed **HAMS to GOLEM**: images and containers became `golem_*`, env vars became `GOLEM_*`, and stale `HAMS_*` vars are hard-rejected.
   - The G1 fork never saw any of that. It still has unlocked `hams_*` images, `HAMS_*` env vars and the fork-base script shape.
   - On top of that, G1 removed Isaac, the mac port, MJPC and CL_Assets, and added:
     - small deps: onnxruntime, pynput, jsp-gui, evdev and xlib;
     - a NIC pin (`G1_DDS_INTERFACE` plus the generated `cyclonedds_local.xml`);
     - stricter domain-0 checks.
2. **The G1 image deltas are small and additive.** [V] GOLEM's images are already a superset of G1's except for:
   - `onnxruntime==1.23.2` in both ros and robocasa;
   - `pynput==1.8.1`, apt `python3-evdev`, apt `python3-xlib` and `ros-humble-joint-state-publisher-gui` in ros.

   Nav2, slam_toolbox, pointcloud_to_laserscan, pinocchio, pin-pink, OMPL, tk and the livox msgs are all already present. **No separate G1 image is warranted.**
3. **Hard port blockers in GOLEM's infra.** [V]
   - `docker_common.sh:25` aborts if any `HAMS_*` var is set. G1 reads 11 `HAMS_*` vars, in `sim/` and its tests.
   - `tests/docker/test_source_pins.py:71-79` fails any Dockerfile `pip install` that lacks `-c /opt/golem-locks/...`. G1's onnxruntime and pynput lines fail it. So would G1's `SpatialMemoryDockerfile` and `VLMapsDockerfile`, which also have unpinned FROMs.
   - GOLEM `.gitignore:70` (`core`) silently ignores **26 tracked G1 files** under `g1_control/core/` and `g1_safety_layer/core/`. G1 added `!core/` for this.
   - `prepare_mounts` (`docker_common.sh:81-109`) requires H1-only files (CL_Assets URDF, mujoco_mpc, magpie GraspGenX assets) for every service.
4. **Version conflicts to validate.** [I]
   - The GOLEM ros lock pins **Pinocchio 2.7 (`pin==2.7.0`) with `pin-pink==4.4.0`**. G1 was validated on an unlocked image. g1_control only uses APIs that exist in 2.7 (grep), but this needs a test run.
   - GOLEM robocasa pins **robosuite `5ce6643` / robocasa `4f8a298`**. G1's `g1_robosuite.py` was developed against whatever master was at build time.
   - MuJoCo pins agree: 3.3.1 in robocasa on both sides. GOLEM ros is 3.2.3 for MJPC, but G1 ROS-side code imports mujoco only in an offline sim2sim script and lazily in the sim-side test modes.
5. **Shared packages.**
   - Submodule pins are identical on both sides: livox `13eb05e`, unitree_ros2 `668d1ec`, unitree_sdk2_python `c158c8d`. [V]
   - FAST_LIO: only G1 changed it. It reverted the launch to upstream and deleted a stray README. Both are safe to take.
   - model_server: camera_ready is newer (Gemini ER-2, depth bundles, hand cams). G1 only renamed the msgs package and removed the magpie default. Keep camera_ready's version.
   - cl_realsense: G1 only deleted H1 files; the G1 stack does not use it. Keep GOLEM's submodule.
6. **The dirty livox `build.sh` in GOLEM** (`--symlink-install` added) is a leftover from the fork-base `launch_ros.sh` sed-patch. Neither side's current `launch_ros.sh` runs or patches `build.sh`. It still leaks into GOLEM's robocasa image, because that Dockerfile COPYs the working tree and runs `build.sh`. [V]

---

## 1. Docker files: HEAD vs HEAD

### 1.1 File inventory [V]

| File | GOLEM camera_ready | G1_GOLEM | Notes |
|---|---|---|---|
| `BaseDockerfile` | locked | unlocked (fork-base content) | G1 diff vs GOLEM: comments, `COPY third_party/unitree_sdk2_python`. GOLEM added the digest-pinned FROM (`BaseDockerfile:10`), `apt-sources.sh` snapshots, ROS snapshot key, `uv` 0.12.15, CycloneDDS `5041f35` (G1: `--branch releases/0.10.x --depth 1`, G1 `BaseDockerfile:74`), and `-c base-amd64-0N.txt` |
| `RosDockerfile` | 606 lines | 431 lines | see §1.2 |
| `RobocasaDockerfile` | 155 lines | 162 lines | see §1.3 |
| `IsaacDockerfile` | yes | removed by G1 | H1-only |
| `SpatialMemoryDockerfile`, `VLMapsDockerfile` | no | yes | **from the spatial-memory branch** (`c0af423`), not G1 work. G1 only changed the VLMaps CMD module path. Unpinned `FROM pytorch/pytorch:2.7.1-...` and bare `pip install`. |
| `docker-compose.yml` | 277 lines, long-form binds with `create_host_path: false` | 127 lines, short-form binds | see §1.4 |
| `docker-compose.gui.yml`, `docker-compose.stack.yml` | yes (camera_ready) | no | GUI overlay plus paired-startup overlay (hardcodes `h12_bringup h12_sim_bringup.launch.py`) |
| `locks/` (55 files) | yes (camera_ready, 2026-09-17) | no | `SOURCES.md` documents the policy |
| `mac/` | yes (camera_ready refactored it) | removed by G1 | |
| `scripts/docker_build.sh` | uses docker_common; `--pull`, `--no-cache`; isaac/robocasa/ros; exports `MJPC_REF` | standalone; `--no-cache`, `--skip-base`; robocasa/ros; builds `hams_base` | both diverged from the fork base |
| `scripts/docker_run.sh` | 24 lines over `docker_common.sh`; no auto-`rm -f` (needs `--restart`) | 116 lines; G1 added NIC pin and domain checks | both diverged; conflict |
| `scripts/docker_common.sh` | yes (camera_ready) | no | `load_config`, `check_domain`, `configure_display`, `prepare_mounts`, `container_name` |
| `scripts/docker_stack.sh`, `runtime_common.sh`, `wait_for_sim.py`, `package_humble_runtime.py` | yes (camera_ready) | no | `wait_for_sim.py` waits on `rt/lowstate` `unitree_hg LowState_`, which works unchanged for the G1 sim **[I]** (the G1 sim publishes hg LowState per G1 CLAUDE.md) |
| `scripts/launch_ros.sh` | stages a private livox copy, revalidates the MJPC cache, refuses stale installs, requires the baked `/opt/unitree_install` | copies `package_ROS2.xml` to `package.xml` in place, stamp-file rebuild | both diverged; see §6.4 |
| `scripts/launch_robocasa.sh` | `validate_sim_domain`, builds `msgs_ws` (magpie + custom_ros_messages), `cd /home/code/h1_robocasa`, `h12_mujoco.py` | no domain validation here (the sim's `_domain_id()` refuses ≤0), no msgs_ws, `cd /home/code/sim`, `g1_mujoco.py` | |
| `scripts/rebuild_mjpc.sh`, `launch_isaac.sh` | yes | removed by G1 | |
| `scripts/spatial_memory_camera.sh` | no | yes (spatial-memory branch; G1 only changed paths) | uses `hams_spatial_memory`, `hams_vlmaps` |
| `scripts/dds_bridge.py` | removed (camera_ready) | removed (G1) | removed independently on both sides |
| `BUILD.md` | "GOLEM build system", 11 sections incl. Isaac and MJPC | "HAMS build system", 9 sections | rewritten on both sides |
| `.env.example` | `ROS_DOMAIN_ID`, `GEMINI_API_KEY`, optional `GOLEM_*` | `GEMINI_API_KEY`, `ROS_DOMAIN_ID=1`, `G1_DDS_INTERFACE=` | |
| `docker/tests/__pycache__/*.pyc` | stray untracked pyc (tests moved to `tests/docker/`) | no | harmless hygiene nit |

### 1.2 RosDockerfile [V]

Shared core, identical in intent on both sides:
- apt build tooling;
- the Livox-SDK2 build;
- the Pink/QP/OMPL pip stack;
- the model_server ML stack;
- CLIP and SAM3;
- the numpy, setuptools and wheel clamp;
- tk;
- Nav2, slam_toolbox, pointcloud_to_laserscan, rqt, teleop, rosbag2 and foxglove;
- GraspGenX with the freeze-constraint trick, plus viser.

**Only on camera_ready** (all added after `c00feca`, except MJPC, which predates the fork):
- Every pip step uses `-c /opt/golem-locks/ros-amd64-01..12.txt`.
- `RUN pip uninstall -y pin libpinocchio coal ... cmeel-*` (`:99-101`) removes the base image's Pinocchio 4 / NumPy-2 closure before reinstalling. The ROS lock is **Pinocchio 2.7 / Boost 1.83 / eigenpy 3.5.1 / hpp-fcl 2.4.4, pin-pink 4.4.0** (`locks/ros-amd64.txt:192-193`, `SOURCES.md`).
- SAM3, GraspGenX and Livox-SDK2 are pinned to SHAs (`:83`, `:151`, `:298`).
- MJPC (the pre-fork block, kept): clang-13 toolchain, `ARG MJPC_REF` (`:452`), mujoco==3.2.3, the build seed and its import check.
- `ros-humble-realsense2-camera`, `image-transport-plugins` (`:583-586`).
- unitree_sdk2 C++ at `63c6f53` into `/opt/unitree_install` (`:590-596`). **`launch_ros.sh` refuses to start without it.**
- `ros-humble-octomap-server`, `octomap-rviz-plugins` (`:603-606`).

**Only on G1:**
- No MJPC, realsense, unitree_sdk2 C++, octomap or locks. MJPC was removed by G1; the others were never in G1.
- `faiss-cpu==1.14.3`, `sentencepiece==0.2.1` (G1 `:395-397`), from the spatial-memory branch.
- apt `ros-humble-joint-state-publisher-gui python3-evdev python3-xlib` (G1 `:408-412`).
- `pip freeze --all > /opt/g1-constraints.txt && PIP_CONSTRAINT=... pip install onnxruntime==1.23.2 pynput==1.8.1` (G1 `:428-431`).
- A version-agnostic dist-info ghost sweep, without GOLEM's Pillow assert.

### 1.3 RobocasaDockerfile [V]

| Item | GOLEM | G1 |
|---|---|---|
| robosuite | `5ce6643` (`:43-45`) | `--depth 1` master at build time |
| robocasa | `4f8a298` (`:47-49`) | master |
| robosuite_models / mimicgen | `2d63a6b` / `72bd767` | master |
| kitchen asset digest check | yes (`:86-88`, `locks/robocasa-assets.digest`) | no |
| Livox-SDK2 | `08f523c` | master |
| WORKDIR | `/home/code/h1_robocasa` | `/home/code/g1_robocasa` (stale; compose `working_dir: /home/code/sim`) |
| msgs_ws toolchain | used (magpie_msgs + custom_ros_messages built at every start) | kept but unused |
| livox build | `COPY core_ws/src/livox_ros_driver2` then `./build.sh humble` (`:147-155`) | same, from `core_ws/src/third_party/...`; comment corrected (see §6.4) |
| onnxruntime | — | `uv pip install -c /opt/g1-constraints.txt onnxruntime==1.23.2` (G1 `:160-162`) |
| mujoco / numpy | 3.3.1 / numpy 2.2.6 then robocasa moves it to 2.2.5 | same |

### 1.4 Compose [V]

**robocasa service:**
- GOLEM mounts `h1_robocasa`, `CL_Assets`, `scripts` (ro) and `msgs_ws` plus two message packages. All are long-form with `create_host_path: false`, so `prepare_mounts` must pre-create them.
- GOLEM passes `GOLEM_SPAWN_MARGIN`, `CAMERAS`, `CAMERA_SCALE/DEPTH/POINTCLOUD`, `HAND_CAMERA_RES`, `SIM_ODOM` (default 0), `SIM_EXPERIMENT`, `TETHER_*` and `SPAWN_OFFSET`.
- G1 mounts `../sim` → `/home/code/sim`, `../third_party/leprechaun_g1` → `/home/code/third_party/leprechaun_g1:ro`, `../core_ws/src/g1/g1_locomotion/policy` → `/home/code/g1_locomotion:ro`, `/dev/input` and `/dev/dri`.
- G1 passes `HAMS_SIM_ODOM`, `ARM_SDK_TIMEOUT`, `CMD_TIMEOUT`, `LIDAR*` and `CAMERA*` (G1 compose `:36-44`).

**ros service:**
- GOLEM sets `CYCLONEDDS_URI=file:///home/code/cyclonedds.xml` (`:188`).
- GOLEM mounts CL_Assets, the magpie GraspGenX asset leaves, the `mujoco_mpc` source, `mjpc_build` and the mjpc gitdir.
- G1 sets `CYCLONEDDS_URI=...cyclonedds.xml,file:///home/code/cyclonedds_local.xml` (G1 `:88`) and adds `G1_DDS_INTERFACE` (`:91`).
- G1 mounts leprechaun_g1 and the policy (`:122,125`).
- G1 also carries spatial-memory extras: `EMBODIED_AGENT_ROOT`, `../../EmbodiedAgent:/opt/EmbodiedAgent:ro`, `container_cache/g1_spatial_memory` and `huggingface`. The EmbodiedAgent bind is short-form, so docker creates a missing host dir. Under GOLEM's long-form `create_host_path:false` convention a missing sibling checkout would **fail**, so it needs to be an optional overlay.
- `GRASPGENX_CHECKPOINT_DIR` differs only by path (`perception/` in G1).

Display: GOLEM moved `DISPLAY`/X11 into `docker-compose.gui.yml`, selected by `GOLEM_DISPLAY`. G1 keeps them inline and calls `xhost`.

---

## 2. Naming map (G1 to GOLEM) [V]

| Kind | G1_GOLEM | GOLEM camera_ready | Port action |
|---|---|---|---|
| base image | `hams_base:latest` | `golem_base:latest` (`golem_base:arm64` on mac) | use golem |
| ros image / container | `hams_ros:latest` / `hams_ros` | `golem_ros:latest` / `golem_ros` | use golem |
| robocasa image / container | `hams_sim_robocasa` | `golem_sim_robocasa` | use golem |
| demo image | `hams_ros_rviz` (`demo/navigation/docker/Dockerfile`, `FROM hams_ros:latest`) | — | becomes `golem_ros_rviz` |
| spatial-memory images | `hams_spatial_memory`, `hams_vlmaps` | — | out of scope (spatial-memory branch) |
| scripted-test containers | `g1_sim_test_<pid>`, `g1_sim_nav_<pid>`, `g1_sim_explore_<pid>`, `g1_demo_walk_<pid>` (+`_ros_`) | `golem-stack-test-<domain>-<pid>` | keep `g1_*`; change the default `SIM_IMAGE`/`ROS_IMAGE` to `golem_*` |
| lock dir in image | — | `/opt/golem-locks` | add G1 deps to the locks |
| scripts mount | `/home/code/h12_sim_scripts` (rw) | `/home/code/h12_sim_scripts` (ro) | same path |
| sim source mount | `../sim` → `/home/code/sim` | `../h1_robocasa` → `/home/code/h1_robocasa` | G1 sim at its own path. The G1 sim derives `_REPO = dirname(sim)` and needs `/home/code/third_party/leprechaun_g1` as a sibling mount. |
| robot assets | `third_party/leprechaun_g1` → `/home/code/third_party/leprechaun_g1:ro` | `${GOLEM_ASSETS_DIR:-../CL_Assets}` → `/home/code/CL_Assets` | new mount |
| policy | `core_ws/src/g1/g1_locomotion/policy` → `/home/code/g1_locomotion:ro` (both services) | — | new mount |
| cache | `../container_cache` (hardcoded) | `${GOLEM_CACHE_DIR:-../container_cache}` | use `GOLEM_CACHE_DIR` |
| DDS config | `/home/code/cyclonedds.xml` + `/home/code/cyclonedds_local.xml` | `/home/code/cyclonedds.xml` | add the local file |
| `core_ws/cyclonedds.xml` | identical except "HAMS"→"GOLEM" and a doc link | | none |

Env vars. The GOLEM guard `docker_common.sh:25` rejects every `HAMS_*`.

| G1 | Readers | GOLEM equivalent | Note |
|---|---|---|---|
| `HAMS_SIM_ODOM` | `sim/g1_mujoco.py:198`, nav tests, the `g1_core.launch.py:284` message | `GOLEM_SIM_ODOM` (exists, same semantics, default 0) | reuse |
| `HAMS_SPAWN_BACKOFF` | `sim/g1_mujoco.py:175`, tests, demo | `GOLEM_SPAWN_BACKOFF` (exists, read by `h12_mujoco.py:180`) | reuse |
| `HAMS_CMD_TIMEOUT` | `sim/unitree_interface.py:91` | `GOLEM_CMD_TIMEOUT` (exists, `h1_robocasa/unitree_interface.py:84`, 0.5 sim-s) | reuse; x86 compose does not pass it today (mac does) |
| `HAMS_ARM_SDK_TIMEOUT` | `sim/unitree_interface.py:87` | — | new `GOLEM_ARM_SDK_TIMEOUT` |
| `HAMS_CAMERAS` | `sim/mujoco_ros_bridge.py:171` | `GOLEM_CAMERAS` (exists, same 0=off) | reuse |
| `HAMS_CAMERA_HZ`, `HAMS_CAMERA_SIZE` | `mujoco_ros_bridge.py:154-167` | none; GOLEM has `GOLEM_CAMERA_SCALE` and `GOLEM_HAND_CAMERA_RES` (different semantics) | new `GOLEM_CAMERA_HZ`/`GOLEM_CAMERA_SIZE`, or align with SCALE |
| `HAMS_LIDAR`, `_POINTS`, `_HZ` | `mujoco_ros_bridge.py:162-169` | none (GOLEM has `GOLEM_LIDAR_SELF_FILTER`) | new `GOLEM_LIDAR*` |
| `HAMS_SIM_TIME` | `g1_spatial_memory/launch/spatial_memory.launch.py:55` | — | `GOLEM_SIM_TIME` if spatial memory is ported |
| `G1_DDS_INTERFACE` | docker_run, compose, sim `unitree_interface.py:254`, `g1_control controller_config.py:307`, `g1_locomotion config.py:174`, `g1_safety_layer core/config.py:163`, `g1_dex3 hand_node.py:75`, plus their tests | none (H1 configs use `interface: ''`) | recommend `GOLEM_DDS_INTERFACE` (robot-agnostic, also useful for the H1 desktop); otherwise keep `G1_` (not rejected by the guard) |
| `G1_SIM_TEST_RECORD_EVERY`, `G1_SAFETY_LOG_DIR` | tests, g1_safety_layer | — | package-local; fine to keep |

---

## 3. Dependencies and version pins

### 3.1 What GOLEM images must gain to run the G1 stack and the G1 sim [V for the lists, I for compatibility]

**golem_ros**, appended at the end per the layer-order rule, after the octomap block:
- apt: `ros-humble-joint-state-publisher-gui` (`g1_bringup` and `g1_description` declare `exec_depend joint_state_publisher_gui`), `python3-evdev`, `python3-xlib`.
- pip: `onnxruntime==1.23.2` and `pynput==1.8.1`, both via `-c /opt/golem-locks/ros-amd64-13.txt`.
- Add to the `ros-amd64.txt` common set: `onnxruntime`, `coloredlogs`, `flatbuffers`, `humanfriendly`, `pynput`, plus `evdev`/`python-xlib` at the apt versions so the exact-pin test passes. `protobuf` (7.36.1), `sympy` (1.14.0) and `six` are already locked. The G1 constraint is that nothing already installed may move. G1 enforces it with a full `pip freeze --all`; GOLEM's lock files serve the same purpose.
- Everything else the G1 packages need is already in golem_ros: nav2 (smac, RPP, velocity_smoother, bt_navigator), slam_toolbox, pointcloud_to_laserscan, apt pinocchio, pin-pink, qpsolvers, OMPL 2.0.1, meshcat, tk, scipy, pyyaml, livox msgs (built in core_ws), and `unitree_sdk2py` (same SHA).

**golem_sim_robocasa:**
- pip: `onnxruntime==1.23.2` via `-c /opt/golem-locks/robocasa-amd64-05.txt`.
- Add `onnxruntime`, `coloredlogs`, `flatbuffers` and `humanfriendly` to `robocasa-amd64.txt`. scipy 1.15.3, pyyaml, PIL and protobuf 3.19.6 are already there.
- **Validate** that onnxruntime 1.23.2 imports with protobuf 3.19.6. [I]

### 3.2 Pins and conflicts

| Package | GOLEM ros | GOLEM robocasa | G1 ros | G1 robocasa | Risk |
|---|---|---|---|---|---|
| MuJoCo | **3.2.3** (MJPC ABI) | 3.3.1 | unlocked (base `mink` pulls a recent mujoco; [I]) | 3.3.1 | Low [I]: G1 ROS-side code imports mujoco only in `g1_locomotion/policy/mujoco_deploy_g1.py` (offline sim2sim) and lazily in `tests/test_sim_*.py` sim/report modes (`test_sim_bringup.py:201`), which run in the robocasa image |
| numpy | 1.26.4 | 2.2.5 | `numpy<2` | 2.2.5 | none |
| Pinocchio (pip) | **`pin==2.7.0`**, pin-pink 4.4.0 | pin 4.1.0 | unlocked: base `pin` (4.x) not uninstalled, then pin-pink reinstalled with `--ignore-installed` [I] | — | **Medium**: g1_control was validated on a different Pinocchio. Its APIs (`buildModelsFromUrdf`, `buildReducedModel`, `computeCollisions`, `GeometryData`, `removeCollisionPairs`, `framesForwardKinematics`, `computeFrameJacobian`, `rnea`, `crba`, ...) exist in 2.7 per the grep, but run the g1_control tests in golem_ros |
| robosuite / robocasa | — | `5ce6643` / `4f8a298` | — | master HEAD at build time (2026-09-30..10-06) [I] | **Medium**: `sim/g1_robosuite.py` registers a custom robot. Run `tests/test_sim_bringup.sh` against the GOLEM image |
| torch | 2.14.0+cu130 | 2.7.1 (lerobot) | unlocked cu130 | 2.7.1 | none (HOMIE uses onnx) |
| CycloneDDS (C) | `5041f35` (0.10.x) | same | `releases/0.10.x` HEAD | same | none (same series) |
| Livox-SDK2 | `08f523c` | `08f523c` | master | master | none while livox stays `13eb05e` |
| unitree_sdk2_python | `c158c8d` (root `unitree_sdk2_python/`) | | `c158c8d` (`third_party/unitree_sdk2_python/`) | | none (path only) |
| unitree_sdk2 C++ | `63c6f53` in `/opt/unitree_install` | | absent | | G1 does not need it; GOLEM `launch_ros.sh` requires it |

---

## 4. DDS and the domain-0 logic [V]

| | GOLEM (`docker_common.sh`) | G1 (`docker_run.sh`) |
|---|---|---|
| Parse | digits only, 0..232, leading zeros normalized (`10#`) (`:31-35`) | regex `^(0|[1-9][0-9]*)$`, rejects `00` and `0\r` (`:52-57`) |
| Unset | defaults to 1 | defaults to 1 |
| Domain 0, sim | refused (also `runtime_common.sh validate_sim_domain` inside the container) | refused; the sim's `_domain_id()` also refuses ≤0 |
| Domain 0, ros | y/N prompt only (`:55-62`), refused on mac | requires `G1_DDS_INTERFACE`, `ip link` existing, state `UP` and an IPv4 address, then the y/N prompt (`:58-97`) |
| NIC pin | none (commented `<Interfaces>` hint in `cyclonedds.xml`) | writes `container_cache/cyclonedds_local.xml` every run (`:35-40`) and mounts it as the second `CYCLONEDDS_URI` |
| Precedence | shell exports win over `docker/.env` (`load_config`) | `.env` overwrites the shell |
| Scripted tests | compose, host networking, unique domains 186-193 (`tests/stack/run.sh`, CI) | `docker run --network none`; the ros container joins the sim's netns (`--network container:$SIM`); `--cap-add NET_ADMIN` plus an ioctl enables multicast on `lo`; `G1_DDS_INTERFACE=lo`; scripts don't read `.env` |

Implications:
- **H1's real robot runs natively on the onboard PC** with `ROS_DOMAIN_ID=0`. The dockerized domain-0 path in GOLEM is only the companion desktop (rviz, cameras, models).
- **The G1 commands the robot from the laptop's `ros` container on domain 0.** In a GOLEM with G1 support, the docker domain-0 path becomes safety-critical, and the NIC pin is functionally required (WiFi plus a wired port).
- Port G1's interface checks into `check_domain`, gated on the robot (or on a set interface), so H1 desktop users without a pin are not blocked.
- Always write `cyclonedds_local.xml` (empty `<CycloneDDS/>` when unset) into `$GOLEM_CACHE_DIR` in `prepare_mounts`, because GOLEM's long-form binds need the host file to exist.
- The Unitree SDK ignores `CYCLONEDDS_URI`. The G1 nodes read the interface env var themselves, so the env var must also be passed to the ros service.
- G1 also documents `systemd-inhibit` for laptop sessions (CLAUDE.md §9).
- `wait_for_sim.py` (GOLEM paired startup) should work unchanged for the G1 sim. [I]

---

## 5. Repo hygiene

### 5.1 `.gitmodules` and pins [V]

| Submodule | GOLEM path / URL / pin | G1 path / URL / pin |
|---|---|---|
| livox_ros_driver2 | `core_ws/src/livox_ros_driver2`, ssh, `13eb05e` (working tree dirty, see §6.4) | `core_ws/src/third_party/livox_ros_driver2`, https, `13eb05e` |
| unitree_ros2 | `core_ws/src/unitree_ros2`, ssh, `668d1ec` | `core_ws/src/third_party/unitree_ros2`, https, `668d1ec` |
| unitree_sdk2_python | `unitree_sdk2_python`, ssh (Oya-Tomo fork), `c158c8d` | `third_party/unitree_sdk2_python`, https, `c158c8d` |
| leprechaun_g1 | — | `third_party/leprechaun_g1`, https, branch main, `3bee054` |
| CL_Assets, CL_isaaclab_sim, h12_ros2_controller, h12_ros2_model, h12_safety_layer, custom_ros_messages, magpie_msgs, magpie_control, estop, cl_realsense, mujoco_mpc | ssh submodules | removed, or imported in-tree as `g1_msgs`, `g1_control`, `g1_safety_layer`, `perception/cl_realsense` |

- GOLEM uses SSH URLs throughout, so a G1 laptop clone needs a GitHub SSH key. G1 deliberately switched to anonymous HTTPS.
- The G1 `.gitmodules` keys keep the old names (`[submodule "core_ws/src/livox_ros_driver2"]`) with new paths.
- **leprechaun_g1 path coupling:** `core_ws/src/g1/g1_description/leprechaun_g1` is a tracked symlink to `../../../../third_party/leprechaun_g1` (verified). Keeping `core_ws/src/g1/` and a top-level `third_party/leprechaun_g1` in GOLEM avoids touching the symlink, `setup.py` or the sim's `G1_ASSETS` (`sim/g1_robosuite.py:41`).

### 5.2 Ignore and attribute files [V]

- **`.gitignore`:**
  - GOLEM line 70 is `core` (crash dumps). Run against every G1 tracked path, it matches **26 files**: `g1_control/g1_control/core/**` and `g1_safety_layer/g1_safety_layer/core/**`. G1 added `!core/` right after `core`. **Add `!core/` to GOLEM before importing**, or `git add` silently drops these files.
  - G1 also adds `/site/` (mkdocs).
  - GOLEM adds `/core_ws/{build,install,log}_stack_tests/`, `/bags/`, `*.mcap`, `*.db3`, `/work/` and `/.claude/`.
  - G1's `demo/navigation/report*.log` are re-included by a nested `demo/navigation/.gitignore`.
- **`.dockerignore`:**
  - The deny-all pattern is the same.
  - GOLEM whitelists `core_ws/src/livox_ros_driver2`, `unitree_sdk2_python`, `core_ws/src/magpie_msgs`, `docker/scripts/package_humble_runtime.py` and `docker/locks/**`.
  - G1 whitelists the `third_party/` equivalents.
  - No G1 addition needs a COPY, so there is no change unless the layout changes.
- **`.gitattributes`:**
  - GOLEM has none at the root, only nested ones: `core_ws/src/model_server/weights/.gitattributes`, `lmg_deploy/{bags,policies}/.gitattributes`, `CL_Assets`.
  - G1 root has `demo/**/*.mp4 filter=lfs`, for 2 LFS videos in `demo/navigation/`.
  - Add it if the demo videos are ported.

### 5.3 CI and docs [V]

- **GOLEM `.github/workflows/simulator-tests.yml`:**
  - The CPU job runs `tests/isaaclab`, `tests/robocasa/test_loop_timing.py`, **`tests/docker`** (workflow and source-pin regressions) and `tests/stack`.
  - The GPU job is manual (`workflow_dispatch`) and builds isaac, robocasa and ros.
  - Its path filter covers `h1_robocasa/**`, `docker/**` and `core_ws/**`. A G1 sim dir must be added.
- **G1 `.github/workflows/docs.yml`:** builds MkDocs (`mkdocs build --strict`, `docs/requirements-docs.txt`: mkdocs 1.6.1, material 9.7.7, pymdown 12.1) and deploys to GitHub Pages on push to main (`site_url: https://correlllab.github.io/G1_GOLEM/`). It has no test CI.
- **Docs:**
  - GOLEM has a short `README.md` (index), `docs/{SETUP,RUNNING,MACOS,DDS_TUNING,DOCKER_TESTING,NAVIGATION_DEMO,ROS_MCP_DEBUG}.md`, `docker/BUILD.md`, `CLAUDE.md` and a tracked **`AGENTS.md`**, a near-copy of CLAUDE.md that is already drifting (missing the `lmg_*` rows and the `rt/lowcmd` invariant).
  - G1 has a 716-line README that doubles as the setup guide, the `docs/` MkDocs site (install, run, robot, reference, testing) plus `G1_FIRST_REAL_RUN.md` and `SPATIAL_MEMORY_ROBOCASA.md`, `mkdocs.yml` and `CLAUDE.md` only.
- **Robot-scoped invariants needed in GOLEM CLAUDE.md** (also inferred from other scopes):
  - GOLEM states "`h12_safety_layer` is the only publisher of `rt/lowcmd` … there is no `rt/arm_sdk` path". G1's `g1_safety_layer` publishes `rt/arm_sdk` and `rt/lowcmd`.
  - GOLEM's real-robot flow is native on the onboard PC. G1's is laptop plus docker on domain 0.
- **`docs/SETUP.md` "Upgrading from HAMS"** says to retag `hams_*` to `golem_*`. **This does not work for G1 images.** They lack `/opt/unitree_install` (required by GOLEM `launch_ros.sh:7-10`), MJPC and the locks, so G1 users must rebuild.

---

## 6. Shared non-G1 packages

### 6.1 FAST_LIO (GOLEM `core_ws/src/FAST_LIO` in-tree vs G1 `core_ws/src/third_party/FAST_LIO`) [V]

- camera_ready: no FAST_LIO commits since `c00feca`.
- G1 (`38e1f0c`) restored upstream's `launch/mapping.launch.py`. GOLEM's version is the H1 rewrite: a body→pelvis static TF for the H1, pointcloud_to_laserscan, slam_toolbox and a Nav2 include whose params file does not exist.
- G1 also deleted `config/README.md` (a lab TODO note) and `core_ws/tf_tree.pdf`.
- **Nothing in GOLEM includes `fast_lio/mapping.launch.py`.** `h12_slam/launch/h12_navigation.launch.py:27` starts `fastlio_mapping` directly, and the config dirs are otherwise identical. G1's cleanup is safe to take.
- Layout: keep `core_ws/src/FAST_LIO` in GOLEM.

### 6.2 model_server (GOLEM `core_ws/src/model_server` vs G1 `core_ws/src/perception/model_server`) [V]

**camera_ready-only changes:**
- `gemini_server.py:30` uses `gemini-robotics-er-2-preview` (G1 has `-er-1.6-preview`).
- YOLO bundles carry aligned depth and `camera_info`. It adds `set_depth`, `set_caminfo` and `DEPTH_STREAM`/`CAMERA_INFO_STREAM`; the `DetectionBundle.msg` fields are already identical on both sides.
- The hand-camera topics are in the YOLO default list.
- Comment rewrites. Weights are tracked in LFS (`sam3.pt` local; `yolo_world_battery_best.pt` and graspgen `epoch_*.pth` tracked).

**G1-only changes:**
- `custom_ros_messages` became `g1_msgs` (imports, `package.xml`, README).
- `graspgen_server` `DEFAULT_GRIPPER = ''` (no magpie), plus empty-name handling.
- YOLO subscribes to the head camera only.
- The `model_logging.py` path became `/home/code/core_ws/src/perception/<pkg>`.
- Weight `.pth`/`.pt` files were removed from git, and `weights/README.md` says where to get them.

**Recommendation:**
- Keep camera_ready's model_server.
- G1 needs nothing from it: the model servers are not in the G1 bringup.
- If G1 should use it, make the image topics and default gripper parameters per robot, not code forks.
- The `custom_ros_messages` vs `g1_msgs` choice is a cross-cutting interface decision (other audit).

### 6.3 cl_realsense (GOLEM submodule `1f5d921` vs G1 in-tree `core_ws/src/perception/cl_realsense`) [V]

- G1 imported it in-tree from history, at or near `fc9a656`, the fork-base pin.
- G1 deleted `launch/h12_rs_cams.launch.py` and the `Rviz/` dir and rewrote the README.
- camera_ready advanced the submodule to `1f5d921`: head and hand launch split (`h12_headcamera`, `h12_hand_cameras`), calibration, fps.
- The G1 stack does not use cl_realsense ("not part of the G1 stack"; `pc_acc` imports the missing `arpa_control`).
- **Keep GOLEM's submodule; there is nothing to port.**

### 6.4 livox_ros_driver2, unitree_ros2, unitree_sdk2_python [V]

The pins are identical (§5.1). The livox rule ("never bump off `13eb05e`") is respected on both sides.

**The dirty `build.sh` in GOLEM's livox checkout** (`git -C core_ws/src/livox_ros_driver2 diff`):

```diff
-    colcon build --cmake-args -DROS_EDITION=${VERSION_ROS2} -DDISTRO_ROS=${ROS_DISTRO}
+    colcon build --symlink-install --cmake-args -DROS_EDITION=${VERSION_ROS2} -DDISTRO_ROS=${ROS_DISTRO}
```

There is also an ignored `package.xml` left by `build.sh`.

- **Origin:** the fork-base `docker/scripts/launch_ros.sh` (`c00feca`, lines 107-122) sed-patched `--symlink-install` into the host checkout's `build.sh` and then ran `./build.sh humble`. This checkout ran that version at some point.
- **Current state:** neither side runs or patches `build.sh` at container launch anymore. Both pass the same cmake args (`-DROS_EDITION=ROS2 -DDISTRO_ROS=humble`) to a direct `colcon build --symlink-install`:
  - GOLEM `launch_ros.sh:53-104` copies the driver to `core_ws/build/.golem-src/livox_ros_driver2` and selects `package_ROS2.xml` there, so the submodule is never mutated. This is covered by `tests/docker/test_launch_ros_cache.py:68` "livox is staged without running or mutating upstream build script".
  - G1 `launch_ros.sh:16-27,46` copies `package_ROS2.xml` to `package.xml` in place, relying on Livox's `.gitignore`.
  - So G1's approach is the same fix as GOLEM's, done independently. GOLEM's is stricter.
- **Remaining effect:**
  - Both RobocasaDockerfiles still `COPY` the working-tree driver and run `./build.sh humble` inside the image (GOLEM `:147-155`). In this GOLEM checkout the image therefore builds with `--symlink-install`; a clean clone builds without it. Both variants work.
  - GOLEM's comment (`:148-152`, "build.sh uses --symlink-install") describes the patched file as if it were upstream. G1 corrected that comment.
  - `build.sh` still exits 0 on colcon failure. Neither Dockerfile asserts `from livox_ros_driver2.msg import CustomMsg`, which CLAUDE.md recommends.
- **Action (not done; read-only):**
  - Restore with `git -C core_ws/src/livox_ros_driver2 checkout -- build.sh` and delete the stray `package.xml`.
  - Adopt G1's corrected comment.
  - Optionally add the import assertion after the robocasa livox build.

---

## 7. Recommendations: how docker/infra should absorb G1

**One image set, with a robot selector at run time. No G1 image profile.** The G1 additions are about 100 MB of pip/apt on top of GOLEM's images, and every other G1 need is already there. A second image set would double the 80+ GB footprint and the lock maintenance. Laptop users do pay for the MJPC seed (3-8 GB) and Isaac is skipped by profile, which is acceptable.

### Concrete shared-infra changes (ordered)

1. **`.gitignore`:** add `!core/` after `core` (blocker; verified). Add `/site/` only if MkDocs is adopted. Optionally add the root `.gitattributes` for `demo/**/*.mp4` (LFS).
2. **Locks plus Dockerfiles:**
   - In `RosDockerfile`, append one block after octomap: apt `ros-humble-joint-state-publisher-gui python3-evdev python3-xlib`, then `pip install -c /opt/golem-locks/ros-amd64-13.txt --no-cache-dir onnxruntime==1.23.2 pynput==1.8.1`.
   - In `RobocasaDockerfile`, append `uv pip install -c /opt/golem-locks/robocasa-amd64-05.txt --system --no-cache onnxruntime==1.23.2`.
   - Add the new numbered lock files and the exact pins to `ros-amd64.txt` and `robocasa-amd64.txt` (onnxruntime, coloredlogs, flatbuffers, humanfriendly, pynput, and the evdev/xlib versions in ros).
   - Update `locks/SOURCES.md` and `BUILD.md`.
   - Must keep `tests/docker/test_source_pins.py` green.
   - **Do not copy** G1's freeze-based `g1-constraints.txt` approach. It fails `test_python_installs_use_committed_constraints`.
3. **`docker_common.sh`:**
   - Add `GOLEM_ROBOT=h12|g1` (default `h12`, so H1 defaults stay byte-identical, per the shared-infra rule).
   - `prepare_mounts` per robot. G1 requires `third_party/leprechaun_g1/...` and the policy dir. Decide whether a G1 checkout must still have CL_Assets and mujoco_mpc initialized; if not, move the H1-only binds into an h12 overlay.
   - `check_domain`: for G1 (or whenever an interface is set) require `GOLEM_DDS_INTERFACE` to exist, be UP and have IPv4 before the prompt (port G1 `docker_run.sh:64-89`).
   - Always write `$GOLEM_CACHE_DIR/cyclonedds_local.xml`.
4. **Compose:**
   - Add a `docker-compose.g1.yml` overlay selected by `docker_common.sh`, like the gui overlay.
   - robocasa: mount the G1 sim dir at `/home/code/<g1 sim>`, `leprechaun_g1` at `/home/code/third_party/leprechaun_g1:ro`, and the policy at `/home/code/g1_locomotion:ro`. Pass the G1 sim knobs as `GOLEM_*` and set `working_dir`.
   - ros: add the leprechaun and policy mounts and `GOLEM_DDS_INTERFACE`.
   - Base ros service: add `cyclonedds_local.xml` to `CYCLONEDDS_URI` and its bind (harmless for H1).
   - Spatial-memory mounts (EmbodiedAgent, HF cache) only in an optional overlay, or not at all.
5. **`launch_robocasa.sh`:** branch on `GOLEM_ROBOT`. For g1, `cd` to the G1 sim and `exec python -u g1_mujoco.py`. The msgs_ws build can be skipped. Keep `validate_sim_domain`.
6. **`docker_stack.sh` / `docker-compose.stack.yml`:** use the bringup package and launch file per robot (`g1_bringup g1_sim_bringup.launch.py`). `wait_for_sim.py` is reusable.
7. **Env var policy:**
   - All docker and sim toggles are `GOLEM_*`. Rename the G1 `HAMS_*` to `GOLEM_*` in `sim/`, its unit tests, `tests/`, `demo/` and `g1_core.launch.py`.
   - Reuse `GOLEM_SIM_ODOM`, `SPAWN_BACKOFF`, `CMD_TIMEOUT` and `CAMERAS`. Add `GOLEM_ARM_SDK_TIMEOUT`, `LIDAR`, `LIDAR_POINTS`, `LIDAR_HZ`, `CAMERA_HZ` and `CAMERA_SIZE`.
   - Rename `G1_DDS_INTERFACE` to `GOLEM_DDS_INTERFACE` in the 5 G1 packages, the sim and their tests, or keep `G1_` if the G1 package owners prefer. `G1_` is not rejected.
   - Images and containers are `golem_*`. Change the `SIM_IMAGE`/`ROS_IMAGE` defaults in `tests/test_sim_*.sh` and `demo/walk/demo_walk.sh`, and the demo `FROM`.
8. **Submodules:** add `leprechaun_g1` (pick the SSH convention to match `.gitmodules`, or document HTTPS). Keep livox, unitree_ros2 and unitree_sdk2_python at GOLEM's paths and pins (identical SHAs).
9. **Layout:**
   - Keep GOLEM's top-level `core_ws/src/<vendored>` paths. GOLEM `launch_ros.sh:57` and its tests hardcode `core_ws/src/livox_ros_driver2`, and the Dockerfiles and `.dockerignore` COPY from there.
   - Put G1 packages under `core_ws/src/g1/`. GOLEM's `launch_ros.sh` passes each `src/*` dir as a `--base-paths` entry, and colcon discovers packages recursively under it [I].
   - Don't bring G1's `perception/` or `third_party/` package moves.
10. **Tests and CI:**
    - Extend the `tests/docker/test_workflows.py` fixtures for the G1 overlay.
    - Bring G1's `tests/test_sim_*.{sh,py}`, keeping their `--network none` isolation, into e.g. `tests/g1/`. Fix the `/home/code/tests` mount paths.
    - Add the G1 sim dir to the CI path filter.
    - Optionally add `sim/test_*.py` to the CPU job.
    - These scripts call `launch_ros.sh` with no args. GOLEM's version then execs `bash` on `/dev/null` and exits 0, so it is compatible [I]. It does require the GOLEM image, because of `/opt/unitree_install`.
11. **Docs:**
    - Fold G1's README, `docs/robot/*` and `G1_FIRST_REAL_RUN.md` into `docs/G1*.md` (or `docs/g1/`). Keep plain markdown unless the team wants MkDocs plus Pages for GOLEM as a whole.
    - Add robot-scoped sections to `CLAUDE.md` and sync or retire `AGENTS.md`.
    - State that HAMS→GOLEM retagging does not apply to G1 images.
12. **Out of scope / defer:** `SpatialMemoryDockerfile`, `VLMapsDockerfile`, `spatial_memory_camera.sh`, `g1_spatial_memory` and its mounts. These come from the spatial-memory branch, not G1 support, and would fail GOLEM's pin tests as-is.

### Validation needed after porting [I]

- `python -m unittest discover -s tests/docker` (lock and workflow guards).
- In golem_ros: the g1_control, g1_safety_layer, g1_locomotion and g1_dex3 package tests. This checks Pinocchio 2.7 and pin-pink 4.4.0, and onnxruntime against the locked numpy and protobuf.
- In golem_sim_robocasa: `sim/test_*.py` with `--network none`, then `tests/test_sim_bringup.sh` (and `--walk`), `test_sim_nav.sh` and `test_sim_explore.sh`. This checks robosuite `5ce6643` / robocasa `4f8a298` with `g1_robosuite.py`, and onnxruntime with protobuf 3.19.6.
- H1 regressions: `tests/stack/run_all.sh` to confirm the default (`GOLEM_ROBOT` unset) behavior is unchanged.
