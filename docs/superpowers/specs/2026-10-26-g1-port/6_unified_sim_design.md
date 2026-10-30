# 6. One RoboCasa sim runner for the H1-2 and the G1: design

Scope: GOLEM `camera_ready` @ `42d6559` (`h1_robocasa/`, `docker/`, `tests/`) and G1_GOLEM `main`
@ `df64d43` (`sim/`, `tests/`, `docker/`, `core_ws/src/g1/g1_locomotion/policy/homie_policy.py`).
Read-only: nothing was built or run, and neither repo was changed.

User decision this implements: **one simulator runner for both robots, `--h12` / `--g1`, h12 the
default, one sensor bridge, no sibling sim directory.** This replaces Option A in
`3_sim_tests_demo.md` §9.

Legend: **[V]** verified by reading the code at the cited `file:line`; **[I]** inferred, needs a
build or a run. Paths without a repo prefix are GOLEM's. `g1:` marks a G1_GOLEM path.

---

## 0. The design in brief

- **Entry point.** `h1_robocasa/h12_mujoco.py` stays the only entry point. It resolves the robot
  (`--h12`/`--g1`, else `GOLEM_ROBOT`, else `h12`) *before* its H1 imports. For `g1` it hands
  over to `h1_robocasa/g1/g1_mujoco.py`. The H1 loop stays in `h12_mujoco.py` byte for byte, so
  the stack-test guard (`tests/stack/run.sh:69-73`) and the observer
  (`tests/stack/observe_simulator.py:56-78`) need no change.
- **Layout.** The G1 sim becomes a Python package, `h1_robocasa/g1/`. There is no new top-level
  directory and no rename of `h1_robocasa` or of its mount.
- **Shared code.** The robots share `measurement_bridge.py` (as is), `sim_timing.py` (as is), a new
  `sim_common.py` (robot selection, CLI, task resolution, viewer framing), and one
  `mujoco_ros_bridge.py`. That module becomes the single `mujoco_sensors` node: `/clock`,
  ground-truth odom, ROS init and shutdown, and the sensor lifecycle. It hosts a robot's
  **sensor suite**. The sensor models stay per robot (`h12_sensors.py`, `g1/g1_sensors.py`),
  because each reproduces a different real driver configuration.
- **Robot-specific code.** These modules stay per robot, with no shared abstraction: the robosuite
  registration and placement, `sim_names`, `unitree_interface`, the support (elastic band or HOMIE
  stand-in), the grippers or hands, and the main loop.
- **Docker.** The base robocasa compose service is unchanged. G1 runs add a
  `docker/docker-compose.g1.yml` overlay (the leprechaun and policy mounts, `GOLEM_ROBOT=g1`, the
  G1 knobs). `docker_common.sh` selects it when `GOLEM_ROBOT=g1`. A new numbered lock step
  appended to the end of `RobocasaDockerfile` adds onnxruntime 1.23.2.
- **Migration.** Eight steps follow a step-0 baseline. Each step is gated on an H1 topic-surface
  golden, the existing H1 tests, and the G1 tests ported in step 2.

---

## 1. Entry point and robot selection

### 1.1 Constraints found in the code

| # | Constraint | Evidence |
|---|---|---|
| C1 | The stack test copies `launch_robocasa.sh` and asserts that the literal `h12_mujoco.py` appears **exactly once**, then swaps it for the observer. Even a launcher *comment* naming `h12_mujoco.py` breaks it (today's comment says `[h12_mujoco]`, which does not match). | `tests/stack/run.sh:69-73`, `docker/scripts/launch_robocasa.sh:49-51` [V] |
| C2 | The observer patches `mujoco.mj_step`. It records only calls whose caller frame is `/home/code/h1_robocasa/h12_mujoco.py` and has the locals `band` and `resolver` (`resolver.body_id('pelvis')`, `resolver.motor_qpos`). **The H1 loop must stay in `h12_mujoco.py` with those local names**, or the test records nothing. | `tests/stack/observe_simulator.py:56-78` [V] |
| C3 | `prepare_mounts robocasa` requires `h1_robocasa/h12_mujoco.py`. `test_workflows.py` builds it as a fixture and asserts the error names it. | `docker/scripts/docker_common.sh:90-95`; `tests/docker/test_workflows.py:19-23,78-83` [V] |
| C4 | `h12_mujoco.py` imports `magpie_hand_bridge` at module level, and that module raises without `magpie_msgs` (built only by the launcher's msgs_ws step). `battery_env` and `h1_2_robosuite` assume `CL_Assets`. G1's acceptance containers source only `/opt/ros` and `/opt/livox_ws` and mount no msgs_ws. | `h1_robocasa/h12_mujoco.py:15-24`; `h1_robocasa/magpie_hand_bridge.py:33-45`; `docker/scripts/launch_robocasa.sh:22-26`; `g1:tests/test_sim_bringup.sh:96-98` [V] |
| C5 | Module-name collisions. Both sims have top-level `unitree_interface`, `sim_names`, `mujoco_ros_bridge` and `measurement_bridge`. Once `h12_mujoco.py` has imported H1's `unitree_interface`, a flat `import unitree_interface` in G1 code gets **H1's** module. | `h12_mujoco.py:22,24`; `g1:sim/g1_mujoco.py:14-22` [V] |
| C6 | The G1 acceptance tests run the sim with `runpy` as `__main__` and monkeypatch `SimInterface.after_step` / `__init__` on the class they import. | `g1:tests/test_sim_bringup.py:162-191`; `g1:tests/test_sim_nav.py:307-345` [V] |
| C7 | The name `h1_robocasa` appears in 18 tracked files outside the directory: compose files (x86 and mac), the launchers (x86 and mac), the observer, `run_in_container.sh`, `docker_common.sh`, `test_workflows.py`, the CI path filter, the Dockerfiles and the docs. | `git grep -c h1_robocasa` [V] |

### 1.2 Design: `h12_mujoco.py` dispatches before the H1 imports

`h12_mujoco.py` stays the canonical entry point (C1-C3), and the directory is not renamed (C7).
The robot is resolved first; for `g1` the file hands over to the G1 package before any H1-only
import (C4, C5):

```python
# h1_robocasa/h12_mujoco.py, top of file
import sys

import sim_common

# The G1 runs from this entry point too. It is dispatched before the H1-2 imports below,
# which need magpie_msgs and CL_Assets; the G1 needs neither, and its test containers
# provide neither.
if __name__ == "__main__" and sim_common.select_robot(sys.argv[1:]) == "g1":
    from g1 import g1_mujoco
    sys.exit(g1_mujoco.main(sys.argv[1:]))

import argparse  # noqa: E402  (unchanged H1 imports follow)
...
```

```python
# h1_robocasa/sim_common.py (stdlib-only at import time)
ROBOTS = ("h12", "g1")

def add_robot_args(parser: argparse.ArgumentParser) -> None:
    """--h12 / --g1, mutually exclusive, into dest 'robot'."""

def select_robot(argv: Sequence[str], env: Mapping[str, str] = os.environ) -> str:
    """--h12/--g1 in argv, else GOLEM_ROBOT (h12|g1, case-insensitive), else 'h12'.
    A command-line flag beats the environment; an unknown GOLEM_ROBOT exits with an error."""
```

- H1 runs (no flag, `--h12`, or `GOLEM_ROBOT` unset or `h12`) fall through to the existing code.
  The H1 `__main__` parser only gains `sim_common.add_robot_args(parser)`, so it accepts `--h12`
  and `--help` lists both flags. H1 imports, the loop, the spawn, the sensors and the env reads are
  unchanged [V by construction].
- The observer still runs `/home/code/h1_robocasa/h12_mujoco.py` with `runpy`, and `mj_step` is
  still called from `h12_mujoco.sim_loop` with `band` and `resolver` in scope (C2).
- The G1 acceptance tests switch from `runpy g1_mujoco.py` to `runpy h12_mujoco.py` with `--g1` in
  the arguments. `SystemExit` from `sys.exit(g1_mujoco.main(...))` passes through their
  `try/finally`, which still saves the recording [V: `g1:tests/test_sim_bringup.py:185-191`].
- `g1/g1_mujoco.py` replaces its `__main__` block (`g1:sim/g1_mujoco.py:422-451`) with
  `main(argv) -> None`, which has the same CLI plus `--g1`.

Alternatives rejected:
- **Rename the entry point** (e.g. `golem_mujoco.py`). This would change C1, C2, C3 and about 10
  doc and comment sites. It is cosmetic only, so it is left as an open question (§8).
- **Lazy H1 imports.** Moving the `magpie_hand_bridge` import into `sim_loop` would edit the H1
  loop and still load `H1_2`, the Magpie grippers and `Battery` into G1 processes. A G1 run
  would then break on any H1 import-time error.
- **Dispatch in the launcher.** The launcher would need a second python line, which breaks C1, and
  it would not cover the `runpy` tests (C6).

### 1.3 How the robot reaches the process

| Layer | `--h12` (default) | `--g1` | Change |
|---|---|---|---|
| Python (`h12_mujoco.py`) | `select_robot` → h12 | `--g1` or `GOLEM_ROBOT=g1` | new (§1.2) |
| `launch_robocasa.sh` | unchanged | unchanged: flags pass through `"$@"`, `GOLEM_ROBOT` through the env; both robots use `MUJOCO_GL=egl` | only the usage comment; **never write `h12_mujoco.py` in it** (C1) |
| `docker-compose.yml` robocasa service | unchanged | unchanged | none, so `test_workflows.py:123-136` keeps passing |
| `docker/docker-compose.g1.yml` (new overlay) | not used | `GOLEM_ROBOT: g1`, the G1 knob pass-throughs (§5) and two read-only mounts (§6) | new |
| `docker-compose.gui.yml` | unchanged | unchanged (`robocasa: *gui` already covers the same service) | none |
| `docker_common.sh` | — | new `configure_robot`: validates `GOLEM_ROBOT`, fails on mac and for isaac, appends `-f docker-compose.g1.yml`. `prepare_mounts robocasa` adds the G1 file checks only when `GOLEM_ROBOT=g1` | new function and one branch |
| `docker_run.sh` | — | scans the args for `--g1`/`--h12` (as it does for `--headless`, `docker_run.sh:13`) and exports `GOLEM_ROBOT`, then calls `configure_robot` before `prepare_mounts`. The flag is still forwarded to the launcher (`:21-23`) | small |
| `docker_stack.sh` | — | the option loop (`:13-19`) accepts `--g1`/`--h12`; `configure_robot`; exports `GOLEM_BRINGUP_PACKAGE=g1_bringup`, `GOLEM_BRINGUP_LAUNCH=g1_sim_bringup.launch.py` | small |
| `docker-compose.stack.yml` | `${GOLEM_BRINGUP_PACKAGE:-h12_bringup}` `${GOLEM_BRINGUP_LAUNCH:-h12_sim_bringup.launch.py}` (identical when unset) | the G1 bringup | two lines |
| `docker/.env.example` | document `GOLEM_ROBOT=h12` | — | doc |
| Container name | `golem_sim_robocasa` | the same (one simulator, two modes; one robot per domain anyway) | none (see open question 6) |

**`GOLEM_ROBOT` maps to the flag.** It is the only way `docker_stack.sh up`, which runs the
compose `command:` without arguments, can pick the robot. The precedence is the same everywhere:
a CLI flag, then `GOLEM_ROBOT`, then `h12`. The base service does not pass `GOLEM_ROBOT`
through, so an H1 container never sees a stray host value. The H1 test wrappers
(`tests/stack/run.sh`, `tests/robocasa/run_in_container.sh`) never call `configure_robot`, so a
`GOLEM_ROBOT=g1` in `docker/.env` cannot switch them [V: they only call
`load_config`/`check_domain`/`prepare_mounts`, `run.sh:16-22`].

Optional robot-identity guard: the ros container's `wait_for_sim.py` already reads one `LowState_`
before bringup (`docker/scripts/wait_for_sim.py:16-24`). The G1 sim sets `mode_machine=5`
(`g1:sim/unitree_interface.py:261-262`); the H1 sim never sets it, so it reads 0 [V]. A
`mode_machine == 5` ⇔ `GOLEM_ROBOT == g1` check therefore catches a G1 stack started against an
H1 sim (and the reverse) before any command is sent. Both sims use the same `rt/*` names and
`unitree_hg` IDL.

---

## 2. Directory and module layout

### 2.1 Tree after the port

```
h1_robocasa/                       # name, mount (/home/code/h1_robocasa) and H1 files unchanged
  h12_mujoco.py                    # MOD   entry for both robots; H1 loop stays here (C2)
  sim_common.py                    # NEW   robot selection, common CLI, task resolution, viewer framing
  mujoco_ros_bridge.py             # MOD   the one sensor bridge: host node + SensorSuite protocol (§4)
  h12_sensors.py                   # NEW   H1 sensor suite, moved verbatim out of mujoco_ros_bridge.py
  measurement_bridge.py            # same  shared as is (a superset of G1's)
  sim_timing.py                    # same  shared (H1 uses it; G1 can adopt it, step 7)
  h1_2_robosuite.py sim_names.py unitree_interface.py magpie_hand_bridge.py mujoco_env.py
  realsense_profile.py realsense_profile.yaml battery_env.py test_spawn_collision.py   # same
  g1/                              # NEW   the G1 sim, a package (imported only as g1.<module>)
    __init__.py
    g1_mujoco.py                   #   G1 sim_loop + _settle_stance + main(argv)   <- g1:sim/g1_mujoco.py
    g1_robosuite.py                #   G1Dex3 registration, MJCF build, placement   <- g1:sim/g1_robosuite.py
    sim_names.py                   #   G1 slot tables, NameResolver                  <- g1:sim/sim_names.py
    unitree_interface.py           #   G1 DDS (lowcmd/arm_sdk/dex3, CRC)             <- g1:sim/unitree_interface.py
    stand_in.py                    #   HOMIE stand-in                                <- g1:sim/stand_in.py
    g1_sensors.py                  #   G1 sensor suite (SensorConfig, G1Sensors)     <- g1:sim/mujoco_ros_bridge.py (step 5)
    g1_lidar.py g1_head_camera.py g1_sensor_msgs.py   #   sensor models              <- g1:sim/ (verbatim)
docker/docker-compose.g1.yml       # NEW   G1 overlay for the robocasa service
docker/locks/robocasa-amd64-05.txt # NEW   onnxruntime lock step (or see §6.1 on naming)
tests/robocasa/surface/            # NEW   topic-surface probe + goldens (step 0)
tests/robocasa/g1/                 # NEW   G1 sim unit tests <- g1:sim/test_*.py, + run_in_container.sh
tests/g1/                          # NEW   G1 acceptance tests <- g1:tests/test_sim_{bringup,nav,explore}.{sh,py}
```

`tests/robocasa/g1/` has no `__init__.py`, so `tests/robocasa/run_in_container.sh`'s
`unittest discover -s /home/code/tests/robocasa` does not recurse into it. The G1 tests are
script-style `main()` files anyway [V: `g1:sim/test_*.py` headers].

Why a package and not flat `g1_*.py` files: C5 forbids top-level `unitree_interface`/`sim_names`
for the G1, and a package gives the G1 owner a directory boundary (GOLEM CLAUDE.md §5). Imports
inside `g1/` are package-relative (`from . import g1_lidar`). Shared modules are imported
absolutely (`import mujoco_ros_bridge`, `from measurement_bridge import MeasurementBridge`); they
resolve because `h1_robocasa` is `sys.path[0]` for the runner and `PYTHONPATH` for the tests.
**`h1_robocasa/g1` must never be put on `sys.path` itself.** If it were, a test that patches a
flat `unitree_interface` would patch the wrong class, and its recorder would silently never fire
(C6).

Not copied from G1: `sim/measurement_bridge.py` (GOLEM's is a superset, §4.6), `sim/LICENSE`
(identical to `h1_robocasa/LICENSE`, per audit 3 §1), G1's `launch_robocasa.sh` and its
compose service, and the X11 mounts in G1's base service, which `test_workflows.py:123-136` would
reject.

### 2.2 File actions

| Action | File | What |
|---|---|---|
| create | `h1_robocasa/sim_common.py` | `ROBOTS`, `add_robot_args`, `select_robot`, `add_common_args`, `runnable_tasks`, `random_task`, `resolve_task`, `VIEW_CAM_*`, `frame_viewer_on_robot` (signatures in §3.2) |
| create | `h1_robocasa/h12_sensors.py` | `H12Sensors` plus the H1 packers `_serialize_livox_scan`, `_camera_info`, `_CameraPub`, `PC2_*`, `LIVOX_POINT_DTYPE`, `CLOUD_FIELDS`, `_COMPRESSED_DEPTH_HEADER`, `_JPEG_QUALITY`, moved from `mujoco_ros_bridge.py:70-161,164-823` |
| create | `h1_robocasa/g1/*` | §2.1. Path roots fixed (§6.2), `HAMS_*` → `GOLEM_*` (§5), relative imports |
| create | `docker/docker-compose.g1.yml`, `docker/locks/robocasa-amd64-05.txt` | §1.3, §6 |
| create | `tests/robocasa/surface/{run.sh,probe.py,h12.json,g1.json}` | step 0 |
| create | `tests/robocasa/test_robot_select.py` (CPU), `tests/robocasa/test_bridge_host.py` (container) | steps 2, 4 |
| create | `tests/robocasa/g1/run_in_container.sh` + ported `test_g1_sensor_msgs.py`, `test_g1_sensors.py`, `test_g1_sensor_bridge.py` (was `test_mujoco_ros_bridge.py`), `test_g1_unitree_interface.py`, `test_g1_spawn_collision.py` | step 2 |
| create | `tests/g1/test_sim_{bringup,nav,explore}.{sh,py}` | step 2 (they run once the G1 ROS packages are in `core_ws`) |
| modify | `h1_robocasa/h12_mujoco.py` | step 2: early dispatch + `add_robot_args`. Step 3: use the `sim_common` helpers. Step 4: build the bridge as host + `H12Sensors` |
| modify | `h1_robocasa/mujoco_ros_bridge.py` | step 4: becomes the host (§4.3) |
| modify | `tests/robocasa/test_sensor_bridge.py` | step 4: fixture wiring only; every assertion stays (§4.4) |
| modify | `docker/RobocasaDockerfile` | append one locked onnxruntime `RUN` at the end |
| modify | `docker/scripts/docker_common.sh`, `docker_run.sh`, `docker_stack.sh`, `docker/docker-compose.stack.yml`, `docker/.env.example` | §1.3 |
| modify (optional) | `docker/scripts/wait_for_sim.py` | robot-identity check (§1.3) |
| modify | `tests/docker/test_workflows.py` | add G1 cases; the existing cases stay as they are |
| modify | `.github/workflows/simulator-tests.yml` | CPU: run `test_robot_select.py`; optionally `test_g1_spawn_collision.py` if CI checks out leprechaun and the policy dir |
| modify | `CLAUDE.md` §2/§3/§8, `docs/RUNNING.md`, `docker/BUILD.md`, `tests/README.md`, `tests/robocasa/README.md`, `docker/scripts/launch_robocasa.sh` (comment only) | step 8 |
| unchanged | the H1 files listed in §2.1, the base compose robocasa service, `docker-compose.gui.yml`, mac compose and launcher, `tests/stack/*` | — |

`h1_robocasa/tests/` holds only an untracked `__pycache__` [V], so it guards nothing.

### 2.3 Shared vs robot-specific

| Shared (one copy) | Robot-specific (dispatch, not merged) |
|---|---|
| `sim_common.py`: selection, CLI, `runnable_tasks`/`random_task`/`resolve_task` (identical in both repos except H1's Battery hook and the log tag [V: `h12_mujoco.py:523-595` vs `g1:sim/g1_mujoco.py:351-419`]), viewer framing (identical [V: `h12_mujoco.py:34-37,85-97` vs `g1:sim/g1_mujoco.py:32-35,66-78`]) | Main loop (`h12_mujoco.sim_loop`, `g1.g1_mujoco.sim_loop`) |
| `mujoco_ros_bridge.py` host: node, `/clock`, odom, ROS init and shutdown, sensor lifecycle (identical in both today [V: `mujoco_ros_bridge.py:60-67,521-555,826-852` vs `g1:sim/mujoco_ros_bridge.py:100-107,492-554`]) | Sensor suites (`h12_sensors.H12Sensors`, `g1.g1_sensors.G1Sensors`) and sensor models |
| `measurement_bridge.py` | robosuite registration and placement (`h1_2_robosuite`, `g1.g1_robosuite`) |
| `sim_timing.py` | `sim_names`, `unitree_interface`, support (`mujoco_env.ElasticBand` / `g1.stand_in.StandIn`), grippers (`magpie_hand_bridge` / Dex3 inside `g1.unitree_interface`) |

---

## 3. The robot seam

### 3.1 What differs per robot

| Aspect | H1-2 | G1 | Evidence | Lives in |
|---|---|---|---|---|
| robosuite robot | `H1_2` from the static `CL_Assets/robosuite_assets/robots/h1_2/robot.xml`; Magpie grippers; head cameras placed from the RealSense profile | `G1Dex3`, MJCF generated **at import** from leprechaun_g1's MJCF and URDF into a temp dir; reads HOMIE `g1.yaml` at import; `MasslessNullGripper`; `head_cam` fovy 42.5 at `d435_link` | `h1_2_robosuite.py:32-37,138-168`; `g1:sim/g1_robosuite.py:40-57,133-224,230-243` [V] | robosuite module |
| RoboCasa offset | `H1_2: [0, 0, 1.05]` | `G1Dex3: [0, -0.30, 0.80]` | `h1_2_robosuite.py:490-492`; `g1:sim/g1_robosuite.py:497-499` [V] | robosuite module |
| spawn pose | 27-motor bent knee (`INIT_LEG_POS`), arms at 0 | legs at HOMIE `default_angles`, everything else 0, then a 3 s stand-in settle on a copy of `MjData` | `h12_mujoco.py:41-57,173-176`; `g1:sim/g1_mujoco.py:81-116,167-171,205-214` [V] | runner |
| placement | margin-based: `GOLEM_SPAWN_MARGIN` (default 0.20 or `env.spawn_margin`), limp-arm poses, `max_iters` 50, `GOLEM_SPAWN_OFFSET` | contact-based: `max_iters` 25, clearance 0.001, floor fit on colliding geoms only | `h1_2_robosuite.py:234-433`, `h12_mujoco.py:180-201`; `g1:sim/g1_robosuite.py:341-459`, `g1:sim/g1_mujoco.py:50,176-180` [V] | robosuite module |
| support before control | `ElasticBand` on the torso (k 1e4, d 2e3, optional rotational hold); `/elastic_band/toggle` or SPACE | HOMIE `StandIn` on the 29 body motors until the first valid `rt/lowcmd`; `rt/arm_sdk` blends the waist and arms | `h12_mujoco.py:206-243`, `measurement_bridge.py:52-54`; `g1:sim/stand_in.py`, `g1:sim/unitree_interface.py:394-410` [V] | runner + `mujoco_env` / `g1.stand_in` |
| Unitree DDS | 27 motors; `rt/lowstate` from a wall-clock `RecurrentThread`, tick = step count, no CRC or mode; `rt/lowcmd` unvalidated; 10 ms watchdog thread | 29 body + 2×7 Dex3; `rt/lowstate` after every step, tick in ms, `mode_machine` 5, CRC; `rt/lowcmd` CRC- and mode-checked; `rt/arm_sdk`; `rt/dex3/*`; `G1_DDS_INTERFACE` | `unitree_interface.py:15-175`; `g1:sim/unitree_interface.py:51-476` [V] | `unitree_interface` per robot |
| grippers / hands | Magpie over ROS (`/{left,right}/gripper/*`, DeliGrasp) | Dex3 over DDS; no ROS gripper surface | `h12_mujoco.py:349-353`; `g1:sim/unitree_interface.py:54-55,324-333` [V] | per robot |
| lidar | 360×56 grid at 10 Hz; `lidar_link`; `lidar_id` 0; reflectivity 0; lines `el % 6`; plus `/livox/pointcloud`; QoS depth 20; cast on a worker from a `copy.copy(data)` snapshot; self filter is a knob | Fibonacci pattern, 5000 points at 10 Hz; `livox_frame`; `lidar_id` 192; reflectivity 100; lines `i % 4`; no pointcloud; QoS depth 256; ring of geom poses every 5 steps | `h12_mujoco.py:318-344`, `mujoco_ros_bridge.py:253-320,728-823`; `g1:sim/mujoco_ros_bridge.py:85-93,201-302`, `g1:sim/g1_sensor_msgs.py:46-117`, `g1:sim/g1_lidar.py:72-91` [V] | suites |
| `/livox/imu` | 200 Hz; **m/s²**; orientation from a framequat; `lidar_link`; QoS 10; next sample at `last + period` | 200 Hz; **g**; identity orientation (as the real driver publishes it); `livox_frame`; QoS 256; exact schedule; stamped `t − dt` | `mujoco_ros_bridge.py:322-326,472-519`; `g1:sim/g1_sensor_msgs.py:120-128`, `g1:sim/mujoco_ros_bridge.py:183-198,276-296` [V] | suites |
| cameras | head (profile intrinsics, 1280×720 × `GOLEM_CAMERA_SCALE` 0.5) plus two hand D405s (`GOLEM_HAND_CAMERA_RES` 256x256); 15 Hz; depth stream and cloud (knobs); compressed topics only with a subscriber; compressedDepth header `(0, 9, 0)`; waits for the previous frame | head D435 only, 1280×720 at 6 Hz, colour plus aligned depth, `color/camera_info` only; compressed always encoded; header all zero; PNG level 3; depth over 10 m zeroed in compressedDepth; drops a frame while busy | `h12_mujoco.py:291-309`, `mujoco_ros_bridge.py:135-139,557-726`; `g1:sim/mujoco_ros_bridge.py:305-407`, `g1:sim/g1_sensor_msgs.py:54-194` [V] | suites |
| TF | `/tf_static` profile tree (mount and driver edges) plus driver edges on `/tf` at 10 Hz | one `/tf_static` edge, `d435_link → head_color_optical_frame` | `realsense_profile.yaml`, `mujoco_ros_bridge.py:295-305,479-482`; `g1:sim/mujoco_ros_bridge.py:337-345` [V] | suites |
| stamps | `/clock` and sensors at `data.time`, ns truncated | `/clock` at `data.time` (truncated); sensors at `data.time − dt`, ns rounded | `mujoco_ros_bridge.py:60-67,456-458`; `g1:sim/mujoco_ros_bridge.py:482-485`, `g1:sim/g1_sensor_msgs.py:63-66` [V] | host (`/clock`), suites |
| ground-truth odom | `GOLEM_SIM_ODOM`: `odom → pelvis` + `/odom` at 50 Hz from the free-joint body | same | `h12_mujoco.py:259-268`; `g1:sim/g1_mujoco.py:198-199` [V] | host |
| timeouts | `GOLEM_CMD_TIMEOUT` 0.5 sim-s, `float()` parse (**an empty string raises**) | `CMD_TIMEOUT` once released, plus `ARM_SDK_TIMEOUT`; finite-positive parser (empty = default) | `unitree_interface.py:84`; `g1:sim/unitree_interface.py:70-91` [V] | `unitree_interface` per robot |
| loop extras | band forces; fall logger on the torso (0.5 / 0.5 m); `tick()` outside the lock; threaded 60 Hz viewer with its own `MjData`; `RealtimePacer`; `--timing` | unstable-reset detection; `after_step()`; fall logger on the pelvis (0.5 / 0.2 m); `tick()` inside the lock; inline viewer sync; inline pacer + RTF log; GIL switch interval 0.5 ms; `--duration` | `h12_mujoco.py:410-509`; `g1:sim/g1_mujoco.py:37-63,217,276-339` [V] | runners |
| task extras | `Battery` (`battery_env.canonical_task`) | none | `h12_mujoco.py:573-575` [V] | runner hook into `sim_common.resolve_task` |

### 3.2 Interfaces (exact names and signatures)

**No `RobotSpec` dataclass.** With the loops dispatched, shared code needs only two things from a
robot: its sensor suite (to the bridge) and its task hook and log tag (to `resolve_task`). A
dataclass would wrap two arguments.

**(a) `h1_robocasa/sim_common.py`** (new; stdlib, plus `mujoco`/`robocasa` imported inside functions)

```python
ROBOTS: tuple[str, ...] = ("h12", "g1")
def add_robot_args(parser: argparse.ArgumentParser) -> None
def select_robot(argv: Sequence[str], env: Mapping[str, str] = os.environ) -> str
def add_common_args(parser: argparse.ArgumentParser) -> None
    # --h12/--g1, --headless, --task, --layout, --style, --seed, --timing, --duration
def runnable_tasks() -> tuple[dict[str, type], set[str]]
def random_task(seed: int | None = None) -> str
def resolve_task(name: str, seed: int | None = None, *, tag: str,
                 robot_task: Callable[[str], str | None] = lambda name: None) -> str
VIEW_CAM_DISTANCE, VIEW_CAM_AZIMUTH, VIEW_CAM_ELEVATION, VIEW_CAM_LOOKAT_DZ
def frame_viewer_on_robot(handle, data, body_id: int) -> None
```

`resolve_task` keeps each robot's log prefix: `tag="h12_mujoco"` or `"g1_mujoco"`. H1 passes
`robot_task=battery_env.canonical_task`. A G1 run given `--task Battery` exits with the error that
the task is H1-2 only.

**(b) Per-robot runner, one signature for both** (after step 3)

```python
# h1_robocasa/h12_mujoco.py   (the loop stays here, C2)
def sim_loop(task: str, viewer: bool = True, layout: int | None = None, style: int | None = None,
             seed: int | None = None, timing: bool = False, duration: float = 0.0) -> None
# h1_robocasa/g1/g1_mujoco.py
def sim_loop(task: str, viewer: bool = True, layout: int | None = None, style: int | None = None,
             seed: int | None = None, timing: bool = False, duration: float = 0.0) -> None
def main(argv: Sequence[str]) -> None   # parse (sim_common.add_common_args), resolve the task, sim_loop()
```

Until step 7, `--timing` with `--g1` and `--duration` with `--h12` exit with "not supported for
this robot yet". The parameters exist so the signature is shared. For H1, `duration=0` keeps the
loop as `while True`.

**(c) Sensor seam: `h1_robocasa/mujoco_ros_bridge.py`** (step 4)

```python
class SensorSuite(Protocol):
    def tick(self, sim_t: float, stamp: TimeMsg) -> None: ...   # after /clock and odom, every tick
    def shutdown(self) -> None: ...                              # stop threads; free GL on its thread
    def stats(self) -> str: ...                                  # "" when there is nothing to report

class RosSensorBridge(Node):                                     # node name "mujoco_sensors"
    def __init__(self, model, data, make_sensors: Callable[[Node], SensorSuite], *,
                 base_body_id: int = -1, odom_frame: str = "odom", base_frame: str = "pelvis",
                 odom_rate_hz: float = 50.0, sim_lock=None) -> None
    def tick(self) -> None
    def shutdown(self) -> None
    def stats(self) -> str

def init_ros() -> bool
def shutdown_ros() -> None
def _sim_time_to_msg(sim_time: float) -> TimeMsg
```

**(d) The suites**

```python
# h1_robocasa/h12_sensors.py: today's RosSensorBridge arguments without the odom and band ones
class H12Sensors:
    def __init__(self, node, model, data, cameras=(), camera_tf=(), camera_tf_rate_hz=10.0,
                 depth_stream=True, pointcloud=True, lidar_body="livox_link", lidar_frame="lidar_link",
                 lidar_exclude_body="torso_link", cam_rate_hz=15.0, lidar_az_rays=180, lidar_el_rays=28,
                 lidar_rate_hz=5.0, lidar_max_range=30.0, lidar_min_range=0.1,
                 imu_quat_sensor="livox_imu_quat", imu_gyro_sensor="livox_imu_gyro",
                 imu_acc_sensor="livox_imu_acc", imu_frame="lidar_link", imu_rate_hz=100.0,
                 lidar_self_prefixes=(), sim_lock=None, timings=None) -> None
    def tick(self, sim_t, stamp) -> None     # imu, camera TF, cameras, lidar (today's order)
    def shutdown(self) -> None; def stats(self) -> str; def get_logger(self)   # -> node.get_logger()

# h1_robocasa/g1/g1_sensors.py
class SensorConfig(NamedTuple): ...          # from_env(env=os.environ) reads the GOLEM_* names of §5
class G1Sensors:
    def __init__(self, node, model, data, prefix: str = "robot0_", config: SensorConfig | None = None) -> None
    def tick(self, sim_t, stamp) -> None     # MID-360 (ring submit + IMU), head camera scene, at sim_t - dt
    def shutdown(self) -> None; def stats(self) -> str
```

**(e) Stays per robot with no shared protocol.** These include `NameResolver`, `SimInterface`,
`place_robot_collision_free`, `make_kitchen_env`, `ElasticBand` and `StandIn`. Each loop calls
only its own robot's versions, so a protocol would have no caller. If the loops are merged later
(open question 3), the de facto contract would be:
- `write_ctrl()` before `mj_step`, under the lock (both have it);
- `after_step()` (H1: a no-op);
- `physics_reset()` (H1: a no-op);
- `shutdown()` (H1 has none today: its `RecurrentThread`s are never stopped [V: `unitree_interface.py:64-89`]).

---

## 4. One sensor bridge

### 4.1 What the two bridges do the same way today [V]
- Node name `mujoco_sensors` (`mujoco_ros_bridge.py:197`; `g1:sim/mujoco_ros_bridge.py:423`).
- `/clock` on every tick, QoS depth 10, stamped `_sim_time_to_msg(data.time)` (identical function).
- Ground-truth odom: identical code, `odom → pelvis` TF + `/odom` at 50 Hz, created only when
  `base_body_id >= 0`. H1 reads it under `sim_lock` (`:521-555`); G1 reads it with the caller's
  lock held (`g1:...:492-525`).
- `init_ros()` / `shutdown_ros()`: identical.
- Lifecycle: the sensors are built in the bridge constructor, so a missing GL fails at startup
  and the EGL context exists before the passive viewer's GLFW window (`:358-365`;
  `g1:...:328-336`). `shutdown()` frees GL on the owning thread.

**The host keeps exactly that and nothing else.** It creates only `/clock` (plus `/odom` and
`/tf` when odom is on), so the G1 acceptance check "`mujoco_sensors` publishes the drivers'
topics and nothing else" still holds (`g1:tests/test_sim_bringup.py:787-797`) [V].

### 4.2 What stays per robot, and why

Every sensor difference in §3.1 (lidar pattern, frame, `lidar_id`, reflectivity, pointcloud, QoS,
IMU units and orientation, camera set, compressed encodings, stamps, TF) is deliberate: each sim
reproduces its own robot's driver configuration. Two examples:
- G1's `lidar_id` 192 comes from `g1_bringup`'s `MID360_config.json` [V `g1_sensor_msgs.py:50-53`].
- H1's compressedDepth header is "as captured from the robot" [V `mujoco_ros_bridge.py:135-138`].

Each robot's FAST_LIO configuration assumes its own IMU units and frame, so one publisher could
not serve both without per-robot parameters on every field. Merging the models would therefore
change at least one robot's output. **The camera and lidar models and encoders stay in the
suites.** A robot "profile" never selects between sensor behaviours inside one class.

### 4.3 The host and its lock contract

```python
def tick(self):
    with self.sim_lock or nullcontext():
        sim_t = float(self.data.time)
    stamp = _sim_time_to_msg(sim_t)
    self.pub_clock.publish(Clock(clock=stamp))
    if self.base_body_id >= 0 and sim_t - self._last_odom_sim_t >= self.odom_period:
        self._last_odom_sim_t = sim_t
        self._publish_odom(stamp)       # takes sim_lock if given
    self.suite.tick(sim_t, stamp)
```

This is the H1 order of operations: `/clock`, odom, then imu, camera TF, cameras and lidar
[V `mujoco_ros_bridge.py:456-496`]. It is also G1's: `/clock`, odom, MID-360, camera
[V `g1:...:467-490`].

There are two lock modes, and each preserves its robot's current behaviour exactly:
- **H1** passes `sim_lock` and calls `tick()` outside the lock (`h12_mujoco.py:500`). `H12Sensors`
  takes the lock only for its snapshots, which `test_sensor_bridge.py` asserts.
- **G1** passes `sim_lock=None` and keeps calling `tick()` inside its lock
  (`g1:sim/g1_mujoco.py:280-317`), as today. `G1Sensors` reads `MjData` directly and its threads
  never touch it.

Moving G1 to H1's mode would move the G1 `/clock` and sensor captures outside the lock. That is a
behaviour change, so it is an optional convergence (step 7), not part of the port.

### 4.4 H1 half (step 4): a mechanical move
- Move `mujoco_ros_bridge.py:70-161` (packers and constants) and the sensor parts of
  `RosSensorBridge` (`:198-326`, `:345-823` without `/clock`, odom and the band) into
  `h12_sensors.H12Sensors`. Replace `self.create_publisher` with `self.node.create_publisher`, and
  `StaticTransformBroadcaster(self)` / `TransformBroadcaster(self)` with `(self.node)`. Add
  `get_logger()` returning `self.node.get_logger()`, which the test overrides as today.
- **Drop the bridge's `elastic_band` argument and its `/elastic_band/toggle` service**
  (`:193,200,340-343,414-422`). `h12_mujoco.py` never passes `elastic_band` to the bridge
  (`:311-347`), so the service is never created. `MeasurementBridge` provides it
  (`h12_mujoco.py:355`, `measurement_bridge.py:52-54`) [V]. Removing it changes no output.
- `h12_mujoco.py:311-347` becomes `RosSensorBridge(model, data, lambda node: H12Sensors(node,
  model, data, <same arguments>, sim_lock=sim_lock, timings=timings),
  base_body_id=odom_base_body_id, sim_lock=sim_lock)`.
- `tests/robocasa/test_sensor_bridge.py`: import from `h12_sensors`, and build the fixture with
  `object.__new__(H12Sensors)` (same attributes). `test_tick_preserves_sim_time_schedule_and_publication_order`
  builds `host = object.__new__(RosSensorBridge)` with `host.suite = b`, `host.pub_clock`,
  `host.data`, `host.sim_lock` and `host.base_body_id = -1`, and calls `host.tick()`. **No
  assertion line changes**, which is the evidence that the move is behaviour-preserving.
- New `tests/robocasa/test_bridge_host.py` covers:
  - the order `/clock` → odom → `suite.tick(sim_t, stamp)`;
  - odom only when `base_body_id >= 0`;
  - with a lock, the lock is held only for the reads and never during `publish`;
  - with `sim_lock=None`, no lock is taken;
  - `shutdown()` and `stats()` delegate to the suite;
  - the host itself creates only `/clock` (plus `/odom` and `/tf` when odom is on).

### 4.5 G1 half (step 5)
- Move `g1:sim/mujoco_ros_bridge.py:85-407` (`SensorConfig`, `_Schedule`, `_LidarScans`, `_Mid360`,
  `_HeadCamera`, the constants) verbatim into `g1/g1_sensors.py`. Add `G1Sensors`, which wraps
  `_Mid360`/`_HeadCamera` with `tick(sim_t, stamp)`: it computes `t = sim_t − dt` exactly as
  `g1:...:482-490` does, and keeps its existing `shutdown()` and `stats()` (`:449-459`).
- The G1 runner builds `RosSensorBridge(model, data, lambda node: G1Sensors(node, model, data,
  prefix=pfx, config=sensors), base_body_id=odom_base_body_id, sim_lock=None)`.
- Delete the step-2 copy `g1/mujoco_ros_bridge.py`. Rename `g1:sim/test_mujoco_ros_bridge.py` to
  `tests/robocasa/g1/test_g1_sensor_bridge.py` (`bridge.SensorConfig` / `_LidarScans` /
  `_HeadCamera` now come from `g1.g1_sensors`).

### 4.6 `measurement_bridge.py`: shared as is
G1's file is GOLEM's minus the band service and the `GOLEM_SIM_EXPERIMENT` surface; the diff is
otherwise docstring only [V: diff above, `measurement_bridge.py:41,49-54,66-153`]. The G1 runner
already calls `MeasurementBridge(env, task_name=task)` and calls `tick()` under the lock
(`g1:sim/g1_mujoco.py:231,315-317`). That matches GOLEM's requirement that `_reset_objects` runs
under the loop's lock (`measurement_bridge.py:117,180-182`) [V].

The G1 gains the opt-in `/sim/ground_truth` and `/sim/reset_objects`. The feet lookup
`{pfx}{side}_ankle_roll_link` should resolve in the G1 MJCF [I: leprechaun_g1 is not checked out
in G1_GOLEM, `git submodule status` shows `-3bee054`]. The default stays off.

### 4.7 `unitree_interface.py`: per robot, no common interface needed
The two share only the topic names and the `unitree_hg` IDL; the semantics differ (§3.1). Each
loop calls its own class, so no protocol is needed (§3.2e).

Two import-time env reads stay per module:
- H1's `DOMAIN_ID = int(os.getenv("ROS_DOMAIN_ID"))` (`unitree_interface.py:21`);
- G1's `ARM_SDK_TIMEOUT`/`LOWCMD_TIMEOUT` (`g1:sim/unitree_interface.py:87,91`).

Early dispatch (§1.2) means a G1 process never imports H1's module.

### 4.8 Optional later convergence (each needs its own decision and tests)
- **One CustomMsg CDR packer** with `lidar_id`/`reflectivity`/`time_rounding` parameters. Both
  sides already test byte equality with rclpy: `tests/robocasa/test_sensor_bridge.py:289-308` and
  `g1:sim/test_g1_sensor_msgs.py`.
- **One Image / CameraInfo builder** with `jpeg_quality`, `png_level`, `depth_header` and
  `depth_max_mm` parameters.
- **G1 to H1's lock mode** (§4.3); G1 acceptance would revalidate it.
- **Pacing.** G1 adopts `RealtimePacer`/`LoopTimings`, but the semantics differ slightly:
  - `RealtimePacer` keeps up to 0.1 s of catch-up after a stall (`sim_timing.py:27-30`);
  - G1's inline pacer drops the whole lag once it is past 0.1 s (`g1:sim/g1_mujoco.py:324-330`) [V].

---

## 5. Environment variables

`docker_common.sh:25` rejects any `HAMS_*` [V], so every G1 knob is renamed.

| G1_GOLEM name | GOLEM name | Honoured by | Default h12 / g1 | Notes |
|---|---|---|---|---|
| `HAMS_SIM_ODOM` | `GOLEM_SIM_ODOM` | both | off / off | same parser, frames and rate [V `h12_mujoco.py:260`, `g1:sim/g1_mujoco.py:198`]. Base compose already passes `${GOLEM_SIM_ODOM:-0}` |
| `HAMS_SPAWN_BACKOFF` | `GOLEM_SPAWN_BACKOFF` | both | 0 / 0 | same meaning; each robot's own placement algorithm. G1 walk test uses 2.0 |
| `HAMS_CMD_TIMEOUT` | `GOLEM_CMD_TIMEOUT` | both | 0.5 / 0.5 | G1 applies it only after the stand-in is released. **H1 raises on an empty string** (`unitree_interface.py:84`), so only the G1 overlay may pass `${GOLEM_CMD_TIMEOUT:-}`; the base service keeps not passing it (mac passes `:-0.5`) |
| `HAMS_CAMERAS` | `GOLEM_CAMERAS` | both | on / on | covers 3 cameras on H1, the head only on G1; same parsing |
| `HAMS_CAMERA_SIZE` (WxH, 1280x720) | **`GOLEM_CAMERA_SCALE`** (fraction of 1280×720) | both | 0.5 / **1.0** | becomes robot-aware: G1 computes `round(1280·s) × round(720·s)`, keeping fovy 42.5. This drops non-16:9 sizes. Alternative: keep WxH as `GOLEM_G1_CAMERA_SIZE` (open question 5) |
| `HAMS_CAMERA_HZ` | `GOLEM_CAMERA_HZ` | g1 (H1 fixed at 15 Hz [V `mujoco_ros_bridge.py:177`, not overridden at `h12_mujoco.py:311-347`]) | — / 6 | ≤ 30 |
| `HAMS_LIDAR` | `GOLEM_LIDAR` | g1 (the H1 lidar cannot be turned off) | — / on | |
| `HAMS_LIDAR_POINTS` | `GOLEM_LIDAR_POINTS` | g1 | — / 5000 | |
| `HAMS_LIDAR_HZ` | `GOLEM_LIDAR_HZ` | g1 (H1 fixed at 10 Hz) | — / 10 | ≤ 100 |
| `HAMS_ARM_SDK_TIMEOUT` | `GOLEM_ARM_SDK_TIMEOUT` | g1 (`rt/arm_sdk` exists only on the G1) | — / 0.5 | no `G1_` infix needed: no existing var means anything else |
| `G1_DDS_INTERFACE` | `G1_DDS_INTERFACE` (unchanged) | g1 | empty | not `HAMS_`, so `load_config` accepts it. The G1 ROS side uses it too (open question 7) |
| `G1_SIM_TEST_RECORD_EVERY` | `GOLEM_TEST_RECORD_EVERY` | G1 acceptance tests | 10 | test-only, follows GOLEM's `GOLEM_TEST_*` |
| — | **`GOLEM_ROBOT`** | wrappers + `select_robot` | `h12` | new |
| — | `GOLEM_BRINGUP_PACKAGE`, `GOLEM_BRINGUP_LAUNCH` | `docker-compose.stack.yml` | `h12_bringup`, `h12_sim_bringup.launch.py` | set by `docker_stack.sh` for g1 |

GOLEM vars the G1 **ignores** (the H1 meaning has no G1 counterpart yet):
- `GOLEM_SPAWN_MARGIN`: G1 placement has no margin. If placement is ever shared, G1's default
  must be 0, or its tuned spawn moves.
- `GOLEM_SPAWN_OFFSET`: easy to add to the G1 in step 7.
- `GOLEM_SIM_TETHER_ROT_STIFFNESS`, `GOLEM_SIM_TETHER_ROT_BODY`: the G1 has no band.
- `GOLEM_HAND_CAMERA_RES`, `GOLEM_CAMERA_DEPTH`, `GOLEM_CAMERA_POINTCLOUD`: the G1 has no such
  streams.
- `GOLEM_LIDAR_SELF_FILTER`: the G1 always drops robot first hits, as no return
  [V `g1:sim/g1_lidar.py` docstring "Self returns"].

`GOLEM_SIM_EXPERIMENT` works for both through the shared `measurement_bridge.py`.

Optional, in `sim_common`: warn once when a knob the selected robot ignores is set to a
non-default value. The base compose always sets `GOLEM_SIM_TETHER_ROT_STIFFNESS=0` and
`GOLEM_SIM_ODOM=0`, so "set" must mean "differs from the compose default".

The test and demo variables (`SIM_IMAGE`, `ROS_IMAGE`, `SIM_GPUS`, `CORE_WS`, `OUT`,
`WALK_BACKOFF`, `NAV_BACKOFF`, `NAV_GOAL`, `NAV_KEEP_LIDAR`, `EXPLORE_BACKOFF`) are not `HAMS_`
and are not rejected. Keep them, and change the image defaults to `golem_sim_robocasa:latest` and
`golem_ros:latest`.

---

## 6. Runtime dependencies, mounts, version risks

### 6.1 Image (`docker/RobocasaDockerfile`)
- **onnxruntime 1.23.2.** Append one locked step at the very end, after the Livox build at
  `:153-155`:
  ```dockerfile
  RUN uv pip install -c /opt/golem-locks/robocasa-amd64-05.txt --system --no-cache "onnxruntime==1.23.2" \
      && python3 -c "import onnxruntime"
  ```
  `robocasa-amd64-05.txt` contains `-c robocasa-amd64.txt` plus the new pins. The full lock also
  gains `onnxruntime`, `coloredlogs`, `humanfriendly` and `flatbuffers`; resolve the exact
  versions in the image with uv. `protobuf==3.19.6`, `sympy==1.14.0`, `packaging` and
  `numpy==2.2.5` are already locked [V `docker/locks/robocasa-amd64.txt:170,180,224`,
  `-04.txt:4`]. Whether onnxruntime 1.23.2 accepts protobuf 3.19.6 is [I] (no upper bound is
  expected).
  - `test_source_pins.py:71-91` requires `pip install -c /opt/golem-locks/…` and exact pins [V].
    G1's `pip freeze` constraint (`g1:docker/RobocasaDockerfile:159-162`) would fail it.
  - **Cache cost.** `COPY docker/locks/robocasa-*.txt` at `RobocasaDockerfile:8` picks up the new
    file, so the first build reruns every layer, including the RoboCasa asset download [V]. The
    alternative is a lock named outside the glob (e.g. `docker/locks/g1-robocasa-amd64.txt`),
    COPYed right before the new `RUN` (open question 8).
  - The H1 image gains one additive package [I: no effect on the H1 sim]. No arm64 lock: the G1
    is x86-only (open question 9).
- Already in the image [V via the lock contents]: pyyaml 6.0.3, pillow 12.3.0, scipy 1.15.3,
  `/opt/livox_ws`, and `unitree_sdk2py` from the base.

### 6.2 Mounts (`docker/docker-compose.g1.yml`, robocasa service)
| Host | Container | Why |
|---|---|---|
| `../third_party/leprechaun_g1` (ro, `create_host_path: false`) | `/home/code/third_party/leprechaun_g1` | `g1_robosuite` builds the MJCF from it **at import** [V `g1:sim/g1_robosuite.py:40-43,224`] |
| `../core_ws/src/g1/g1_locomotion/policy` (ro) | `/home/code/g1_locomotion` | `g1.yaml` is read at import (`g1_robosuite.py:48-57`); `homie_policy.py` and `deploy.onnx` are used by `stand_in.py:19-24`; `homie_policy.load_config` resolves `../third_party/...` from there [V `homie_policy.py:25-45`] |
| `h1_robocasa` (already mounted) | `/home/code/h1_robocasa` | carries `g1/` |

Path roots in the moved code change from "parent of `sim/`" to "two levels up from
`h1_robocasa/g1/`". That gives `/home/code` in the container and the repo root on the host,
in both `g1/g1_robosuite.py` (`_REPO`) and `g1/stand_in.py` (`_ROOT`). Both existing candidate
policy locations then keep working: `/home/code/g1_locomotion` and
`core_ws/src/g1/g1_locomotion/policy`. The G1 test file `test_unitree_interface.py` builds its
leprechaun path from `HERE/..`, which must also be rebased [V `g1:sim/test_unitree_interface.py:27-28`].

The mounts stay out of the base service: there `create_host_path: false` would make every H1 run
fail on a checkout without the new submodule initialized [I: compose refuses a missing bind
source]. `prepare_mounts robocasa` with `GOLEM_ROBOT=g1` adds `require_file` checks for:
- the leprechaun `g1_29dof_with_hand_rev_1_0.{xml,urdf}`;
- `policy/{deploy.onnx,g1.yaml,homie_policy.py}`, which also catches LFS pointers.

**Order dependency on the rest of the port.** The sim cannot import without
`third_party/leprechaun_g1` and `core_ws/src/g1/g1_locomotion/policy/`. In the audit-summary
order, "Locomotion" (step 6) comes after "Sim" (step 5), so at least `policy/` (with its CC BY-NC
`LICENSE`) must land before or with the sim.

The launcher's msgs_ws colcon step (`launch_robocasa.sh:22-26`) still runs for G1 runs. It is
harmless: the G1 path never imports those messages.

`MUJOCO_GL`: GOLEM's launcher exports `egl` in both display modes (`launch_robocasa.sh:45`); G1's
used `glfw` with the viewer. The G1 head camera makes its GL context on its own thread inside the
bridge constructor (`g1:sim/mujoco_ros_bridge.py:328-336`), before `launch_passive`
(`g1:sim/g1_mujoco.py:222,247`) [V]. That is the order EGL needs, so G1 should run under `egl`
[I: verify GUI mode in step 2].

### 6.3 Version risks
| Item | GOLEM pin | G1 validated on | Risk |
|---|---|---|---|
| robosuite | `5ce6643` (2026-09-17) | master at ~2026-10-06 | **[I] medium.** G1-only symbols to check at the pin: `robosuite.models.grippers.null_gripper.NullGripper` and `env.control_freq` (H1 already exercises `LeggedManipulatorModel`, `register_robot_class` and `register_gripper`) |
| robocasa + kitchen assets | `4f8a298`, assets digest-pinned | master | **[I] medium.** `layout 9 / style 9 / seed 42` may differ, so G1 nav/explore expectations (goal distance, coverage) and spawn offsets tuned on master may shift. `env_utils.set_robot_base/_ROBOT_POS_OFFSETS` is already used by H1 |
| MuJoCo | 3.3.1 | 3.3.1 | none |
| numpy | 2.2.5 (final lock) | 2.2.6 | [I] low |
| onnxruntime | — | 1.23.2 | protobuf 3.19.6 compatibility [I] |
| `unitree_sdk2_python`, `livox_ros_driver2` | `c158c8d`, `13eb05e` | identical | none (audit 3 §4) |

---

## 7. Migration sequence

Test gates referred to below:
- **H1-CPU**: the CI cpu job — `tests/robocasa -p test_loop_timing.py`, `tests/docker`,
  `tests/stack -p 'test_*.py'`.
- **H1-UNIT**: `ROS_DOMAIN_ID=187 tests/robocasa/run_in_container.sh` (`test_sensor_bridge`,
  `test_realsense_profile`, `test_loop_timing`).
- **H1-SURFACE**: `tests/robocasa/surface/run.sh h12`, compared against `h12.json`.
- **H1-STACK**: `ROS_DOMAIN_ID=193 tests/stack/run.sh robocasa arms` and `... locomotion`.
- **H1-TIMING**: `docker_run.sh robocasa --headless --task OpenFridge --layout 1 --style 1 --seed 42 --timing`,
  stage means compared with the baseline on the same worker.
- **G1-UNIT**: `tests/robocasa/g1/run_in_container.sh`.
- **G1-SURFACE**: `tests/robocasa/surface/run.sh g1`, compared against `g1.json`.
- **G1-SMOKE**: `docker_run.sh robocasa --g1 --headless --layout 9 --style 9 --seed 42 --duration 30`.
  It must exit 0, log "stance settled" and "ROS bridges up", and log no `[fall-logger]`.
- **G1-ACCEPT**: `tests/g1/test_sim_bringup.sh`, `--walk`, `test_sim_nav.sh` (`fastlio` and
  `GOLEM_SIM_ODOM=1`), `test_sim_explore.sh`. They need the G1 ROS packages in `core_ws` (another
  track); until then G1-SURFACE and G1-SMOKE stand in.

| # | Step (small commits; each G1-sourced commit cites `df64d43`) | Proves H1 unchanged | Proves G1 works |
|---|---|---|---|
| 0 | **Baseline, no production change.** Add `tests/robocasa/surface/probe.py` + `run.sh`: compose-run the canonical launcher headless (`--task AirDryFruit --seed 42`) on a dedicated domain (194), with an rclpy probe in the same container. Over 10 sim-s it records: <ul><li>nodes, and each sim node's publishers, subscriptions, services and actions with types;</li><li>per-topic frame_id, sim-time rate, encoding, size and point layout;</li><li>`/tf_static` edges and `/livox/imu` mean |acc| (≈9.8 for h12, ≈1.0 for g1);</li><li>`rt/lowstate` ticks advancing (`unitree_sdk2py`).</li></ul> Compare names, types and frames exactly, rates within ±5 %. Record `h12.json` at `42d6559`, and `g1.json` from G1_GOLEM `df64d43` with its own image and launcher before any porting. Also keep H1-UNIT, H1-STACK and H1-TIMING baseline artifacts. | baseline | baseline (`g1.json`) |
| 1 | **Image and plumbing.** Lock step and onnxruntime `RUN`; `docker-compose.g1.yml`; `configure_robot`; G1 branch in `prepare_mounts`; `--g1`/`--h12` in `docker_run.sh`/`docker_stack.sh`; new `test_workflows.py` cases (g1 overlay mounts and env; g1 missing-asset failure; g1 rejected on mac and isaac; default args produce compose calls identical to today). Prerequisite: `third_party/leprechaun_g1` and `core_ws/src/g1/g1_locomotion/policy/` are in the repo. | H1-CPU with all existing `test_workflows`/`test_source_pins` unmodified; after the rebuild, H1-UNIT, H1-SURFACE, H1-STACK | `docker_run.sh robocasa --g1 bash -lc 'python3 -c "import onnxruntime"; ls /home/code/third_party/leprechaun_g1 /home/code/g1_locomotion'`; new workflow tests |
| 2 | **Import the G1 sim and dispatch.** Commit 2a: copy `sim/*` into `h1_robocasa/g1/` verbatim (G1's own bridge as `g1/mujoco_ros_bridge.py`; not its measurement bridge or LICENSE). 2b: package imports, shared `measurement_bridge`, path roots. 2c: `HAMS_*` → `GOLEM_*` per §5, with `GOLEM_CAMERA_SCALE` for G1. 2d: `sim_common.select_robot`/`add_robot_args`, early dispatch in `h12_mujoco.py`, `g1_mujoco.main(argv)`. 2e: port the G1 unit tests to `tests/robocasa/g1/` (patch `g1.unitree_interface.SimInterface`, never a flat import) and the acceptance tests to `tests/g1/` (entry `h12_mujoco.py --g1`, `golem_*` images, results under `tests/results/g1-*` with a `run.json` like `tests/stack/run.sh`). Add `test_robot_select.py` to the CPU CI. | H1-CPU (+ `test_robot_select`), H1-UNIT, H1-SURFACE (identical), H1-STACK: the guard and observer are untouched | G1-UNIT, G1-SMOKE (also with the GUI, for EGL), G1-SURFACE == `g1.json` apart from knob renames, G1-ACCEPT when available |
| 3 | **Shared helpers.** `runnable_tasks`/`random_task`/`resolve_task`/viewer framing/`add_common_args` move into `sim_common`, and both runners use them. Runner signatures are unified (§3.2b), with unsupported flags rejected per robot. | H1-SURFACE; H1-STACK; `--help` and log-line diff (same `[h12_mujoco]` tags); a container test that `resolve_task('OpenDoor', seed=1)` picks the same subclass as before | G1-SMOKE, G1-SURFACE |
| 4 | **One bridge, H1 half** (§4.4): host + `h12_sensors.py`; `test_sensor_bridge.py` rewired, every assertion line unchanged; `test_bridge_host.py`. | H1-UNIT, H1-SURFACE (exact `mujoco_sensors` publishers), H1-STACK, H1-TIMING | G1-UNIT (G1 not yet on the host) |
| 5 | **One bridge, G1 half** (§4.5): `g1/g1_sensors.py` + `G1Sensors` on the host with `sim_lock=None`; delete `g1/mujoco_ros_bridge.py`; port the bridge test. | H1-UNIT, H1-SURFACE (host untouched in this step) | G1-UNIT, G1-SURFACE, G1-ACCEPT sensor checks (topic surface, rates, gaps, frames, the floor in the lidar frame) |
| 6 | **Stack wiring:** the bringup selection in `docker-compose.stack.yml`; optional robot-identity check in `wait_for_sim.py`. Needs `g1_bringup` ported. | `test_workflows` stack cases; manual `docker_stack.sh up robocasa` | `docker_stack.sh up robocasa --g1`, then the G1-ACCEPT arm test against it |
| 7 | **Optional convergence** (one commit each, user-approved): G1 `--timing` (LoopTimings) and pacing; H1 `--duration`; G1 `GOLEM_SPAWN_OFFSET`; H1 honours `GOLEM_CAMERA_HZ`/`GOLEM_LIDAR_HZ` with its current defaults; shared CDR/image packers; G1 to H1's lock mode; shared placement primitives (`place_robot_freejoint`, `_backward_dir`, `_robot_env_contacts`, `_euler_to_wxyz`, `install_robocasa_patches(predicate, offset)`, `make_kitchen_env(robots=…)`), with `lowest_robot_point_z(colliding_only=…)` per-robot defaults | H1-SURFACE + H1-STACK + H1-TIMING on every commit | G1-SURFACE + G1-ACCEPT |
| 8 | **Docs:** `CLAUDE.md` §2 (layout), §3 (`--g1`, `GOLEM_ROBOT`, overlay), §8 (G1 topic surface, ground truth, knobs; "trust `h12_sensors.py` / `g1/g1_sensors.py`" replaces "trust `mujoco_ros_bridge.py`"); `docs/RUNNING.md`; `tests/README.md`; the launcher usage comment (C1) | — | — |

---

## 8. Risks and open questions

**Risks**
1. **The stack-test guard is literal** (C1). Any later edit to `launch_robocasa.sh` that names
   `h12_mujoco.py` again, even in a comment, fails `tests/stack/run.sh`.
2. **Silent test no-ops.**
   - The observer depends on the H1 loop's file and the local names `band`/`resolver` (C2), so a
     later refactor of the H1 loop must update the observer in the same commit.
   - G1 tests that patch a flat `unitree_interface` would patch H1's class (C5/C6).
   - Mitigation: both recorders should assert that they recorded at least one sample.
3. **Same DDS names and IDL for both robots.** A G1 stack against an H1 sim, or the reverse, is
   not rejected by the H1 sim. Mitigations: one robot per domain, `docker_stack` selecting the
   bringup and the sim from the same `GOLEM_ROBOT`, and the optional `mode_machine` check (§1.3).
4. **robosuite/robocasa pins differ** from what the G1 was validated on (§6.3). The G1 nav and
   explore thresholds may need re-tuning on GOLEM's kitchen assets.
5. **The image rebuild** busts the whole robocasa cache once (§6.1) and changes the H1 image too
   (onnxruntime added).
6. **Port ordering:** the sim needs `g1_locomotion/policy/` and leprechaun before the
   "Locomotion" step of the overall port (§6.2).
7. **No H1 topic-surface test exists in GOLEM today** [V: no `get_publisher_names_and_types` in
   `tests/`]. Step 0's probe is the only exact check on H1 output. Without it, "H1 unchanged"
   rests on `test_sensor_bridge.py` and the stack tests, which run with cameras off
   (`run.sh:14`).

**Open questions for the user**
1. **Early dispatch at the top of `h12_mujoco.py`** (code before imports, §1.2): OK? The
   alternatives are lazy H1 imports or renaming the entry point.
2. **Entry name:** keep `h12_mujoco.py` for both robots (recommended now), or rename later (e.g.
   `golem_mujoco.py`) and update C1, C2, C3, both launchers and the docs?
3. **Loops:** keep two loop functions behind one entry (recommended, per "dispatch, don't
   merge"), or plan a single loop with robot hooks later (§3.2e)?
4. **G1 lock mode:** keep `tick()` under the lock for G1 (bit-for-bit G1 behaviour), or converge
   on H1's capture-lock model in step 7?
5. **G1 camera size:** adopt `GOLEM_CAMERA_SCALE` with a robot-aware default (0.5 h12, 1.0 g1),
   or keep WxH as `GOLEM_G1_CAMERA_SIZE`?
6. **Container name:** one `golem_sim_robocasa` for both robots (recommended), or a distinct
   `golem_sim_robocasa_g1`?
7. **`G1_DDS_INTERFACE`:** keep it (shared with the G1 ROS side), or rename it
   `GOLEM_DDS_INTERFACE` repo-wide with the ROS-side port?
8. **Lock naming:** `robocasa-amd64-05.txt` (convention, full rebuild), or an out-of-glob name
   COPYed at the end (keeps the cache)?
9. **Mac:** G1 sim x86-only (`configure_robot` rejects it on mac), or add an arm64 onnxruntime
   lock and test it?
10. **Ignored-knob warnings** (§5): add them, or rely on documentation?
11. **G1 unit tests in CPU CI:** `test_g1_spawn_collision.py` runs on the host, but CI would have
    to check out leprechaun and the policy dir.
