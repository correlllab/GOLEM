# Audit 3: G1 simulator, acceptance tests, demos and the spatial-memory benchmark

Scope: G1_GOLEM `main` HEAD `df64d43` (`sim/`, `tests/`, `demo/`, `spatial_memory/`, and the
docker pieces they use) against GOLEM `camera_ready` HEAD `42d6559` (`h1_robocasa/`, `tests/`,
`docker/`). The fork base `c00feca` (GOLEM) is used only as the common ancestor for a 3-way
picture; per the scope change, no per-commit history walks were done.

Legend: **[V]** verified by reading code or running read-only git/diff commands; **[I]** inferred,
not verified (needs a build or a run).

---

## 1. File-by-file lineage: G1 `sim/*` vs GOLEM `h1_robocasa/*`

Base = `git show c00feca:h1_robocasa/<file>`. Numbers are changed lines (`diff | grep '^[<>]'`
split into deleted/added). "Kept by both" = base lines that survive unchanged in both HEADs.

| G1 `sim/` file | GOLEM `h1_robocasa/` file | base | G1 | GOLEM | base→G1 (−/+) | base→GOLEM (−/+) | kept by both | Unify? |
|---|---|---|---|---|---|---|---|---|
| `mujoco_ros_bridge.py` | `mujoco_ros_bridge.py` | 643 | 554 | 852 | 497 / 408 | 188 / 397 | 140 | **No.** Two independent rewrites |
| `measurement_bridge.py` | `measurement_bridge.py` | 79 | 65 | 188 | 19 / 5 | 8 / 117 | 58 | **Yes, trivially.** G1 is a strict subset of GOLEM's |
| `unitree_interface.py` | `unitree_interface.py` | 175 | 476 | 175 | 155 / 456 | 1 / 1 | 20 | **No.** G1 is a full rewrite; GOLEM's is the base plus the env rename |
| `sim_names.py` | `sim_names.py` | 130 | 126 | 130 | 86 / 82 | 0 / 0 | 44 | **No.** Robot tables; GOLEM's is unchanged since the base |
| `test_spawn_collision.py` | `test_spawn_collision.py` | 210 | 210 | 304 | 15 / 15 | 0 / 94 | 195 | Only with a shared placement module |
| `g1_mujoco.py` | `h12_mujoco.py` | 448 | 451 | 627 | 228 / 231 | 55 / 234 | 210 | Partly. Shared helpers only |
| `g1_robosuite.py` | `h1_2_robosuite.py` | 494 | 588 | 581 | 158 / 252 | 35 / 122 | 303 | Partly. Placement and patch code |
| `LICENSE` | `LICENSE` | 21 | 21 | 21 | 0 | 0 | all | identical |
| `stand_in.py`, `g1_lidar.py`, `g1_head_camera.py`, `g1_sensor_msgs.py`, `test_g1_*.py`, `test_mujoco_ros_bridge.py`, `test_unitree_interface.py` | n/a | | | | new in G1 | | | G1-only |
| n/a | `battery_env.py`, `magpie_hand_bridge.py`, `mujoco_env.py` (ElasticBand), `realsense_profile.py/.yaml`, `sim_timing.py` | | | | removed or absent in G1 | | | H1-only (see §5) |

All rows are [V]. Per file:

- **`measurement_bridge.py`** [V]. G1 only removed the elastic-band `/elastic_band/toggle`
  service and reworded the docstring. GOLEM camera_ready kept the band and added the opt-in
  experiment surface: `GOLEM_SIM_EXPERIMENT` at `h1_robocasa/measurement_bridge.py:68` adds
  `/sim/ground_truth` and `/sim/reset_objects`. GOLEM's `MeasurementBridge(env, elastic_band=None, task_name=...)`
  already behaves exactly like G1's. The experiment surface finds the robot by its
  `robot0_` free joint and `{pfx}{side}_ankle_roll_link`. G1 has those links, so it should work
  for G1 unchanged [I: link names come from the leprechaun MJCF, whose submodule is not initialized].
  **This file can be shared as is.**
- **`unitree_interface.py`** [V]. GOLEM's is the base code with `HAMS_CMD_TIMEOUT` renamed to
  `GOLEM_CMD_TIMEOUT` (`:84`). It has 27 motors, no CRC/mode checks, publishes `rt/lowstate`
  from a wall-clock `RecurrentThread` with `tick` = step count, and runs the watchdog on a 10 ms
  wall-clock thread that measures sim time. G1's is a new design that shares only about 20 lines:
  - 29 body motors plus 2x7 Dex3 motors;
  - fast CRC (`:127`);
  - `rt/lowcmd` validated for CRC, `mode_machine` 5 and `mode_pr` 0 (`:162`);
  - `rt/arm_sdk` blended by the weight in slot 29 over slots 12-28 (`:62-63`, `:394-410`),
    with a ramp on timeout (`:338-367`);
  - `rt/dex3/{left,right}/cmd|state` at 100 Hz;
  - `rt/lowstate` after every physics step with `tick` in ms;
  - `physics_reset()` after a MuJoCo auto-reset;
  - `G1_DDS_INTERFACE` (`:254`), and a validated `ROS_DOMAIN_ID` (`:198`).

  Both use the `unitree_hg` `LowCmd_`/`LowState_` IDL on the same topic names (see the risk in §8).
- **`sim_names.py`** [V]. These are robot tables. G1's has 29 `BODY_JOINTS` in DDS slot order,
  `HAND_JOINTS`, the pelvis IMU sensor names, a `Motors` tuple with `tau_min`/`tau_max`, and
  `base_qpos`/`base_qvel`. GOLEM's is unchanged since the base. Keep them separate.
- **`g1_mujoco.py` vs `h12_mujoco.py`** [V]. About 210 base lines survive in both:
  - viewer framing (`_frame_viewer_on_robot`);
  - `_runnable_tasks`, `_random_task`, `_resolve_task`;
  - argparse and the clean-reset scaffold.

  G1 removed the elastic band, the Magpie bridges and the 27-motor init pose. G1 added:
  - the HOMIE `StandIn` and `_settle_stance` (`:81-116`);
  - a fall logger on the pelvis;
  - detection of MuJoCo unstable resets (`:62-63`, `:289-296`);
  - a GIL switch interval of 0.5 ms (`:60`);
  - inline real-time pacing with `MAX_LAG` 0.1 s and RTF logging;
  - a `--duration` argument.

  GOLEM added:
  - `battery_env` tasks;
  - the RealSense-profile cameras;
  - `GOLEM_SPAWN_MARGIN` / `GOLEM_SPAWN_OFFSET` and limp-upper-body poses for the margin;
  - a rotational tether;
  - a 60 Hz viewer thread with its own `MjData`;
  - `sim_timing.RealtimePacer` / `LoopTimings` and the `--timing` flag.

  A 3-way merge into one runner is not practical. Shared helpers can be factored out.
- **`g1_robosuite.py` vs `h1_2_robosuite.py`** [V]. The placement primitives are nearly
  identical between G1 and the base (`b_place` diff: only print prefixes, plus G1 skipping
  non-colliding geoms in `lowest_robot_point_z`, `g1_robosuite.py:341-359`). The shared
  primitives are `place_robot_freejoint`, `lowest_robot_point_z`, `place_robot_clear`,
  `_backward_dir`, `_robot_env_contacts`, `place_robot_collision_free`,
  `install_robocasa_patches`, `make_kitchen_env` and `_euler_to_wxyz`.

  GOLEM rewrote `place_robot_collision_free` around a clearance margin (`h1_2_robosuite.py:306`
  `_robot_env_gap`, `:347` `_overlaps_behind`, `:354` `_placed_gap`, `:377`
  `place_robot_collision_free(..., max_iters=50, margin=0.0, poses=())`).

  The robot model halves are entirely different:
  - G1 generates the MJCF at import from `leprechaun_g1` (`build_robot_xml`, `:133-208`) and uses
    a `MasslessNullGripper`.
  - G1 names its class `G1Dex3` because robosuite_models already registers `G1`.
  - G1 places with the offset `[0.0, -0.30, 0.80]` (`:499`).
  - H1 uses a static `CL_Assets` XML and Magpie grippers.

  Both patch `robocasa.utils.env_utils.set_robot_base` / `set_robot_to_position` behind an
  `isinstance` guard, so the patches would chain safely even if both modules were imported in
  one process [I].
- **`test_spawn_collision.py`** [V]. G1's is the base file with `hr`/`h1_2_robosuite` renamed
  to `gr`/`g1_robosuite` and a `NullGripper` stub. GOLEM's adds 94 lines of margin tests
  (`_gap`, `test_margin_*`). The two follow whichever placement implementation they import.
- **`mujoco_ros_bridge.py`** [V]. The two are independent redesigns of the same file (§2.4).
  The only common parts are `_sim_time_to_msg`, the `/odom` + `odom->pelvis` publisher,
  `init_ros`/`shutdown_ros` and `/clock`. Do not 3-way merge.

**GOLEM's camera_ready changes to `h1_robocasa` since the base, by theme** [V] (14 commits; diffstat 1737+/340−):
- renamed `HAMS_*` to `GOLEM_*`;
- 1x real time with all sensors (`sim_timing.py`, threaded sensor jobs, viewer thread);
- RealSense driver fidelity (`realsense_profile.py/.yaml`; frames, TF, intrinsics, depth
  stream, point cloud, and `GOLEM_CAMERA_*` / `GOLEM_HAND_CAMERA_RES`);
- spawn clearance margin and spawn offset;
- Battery workcell task (`battery_env.py`);
- force-limited Magpie gripper;
- elastic band damping and rotational hold;
- the opt-in experiment surface;
- nav/ALMI fixes.

None of this is in G1.

---

## 2. G1 sim architecture (HEAD) [V]

### 2.1 `g1_mujoco.py` (entry point, 451 lines)
1. `SensorConfig.from_env()` is read first, so a bad knob fails fast (`:146`).
2. `g1_robosuite.make_kitchen_env(task, robots="G1Dex3")` builds the env, then `env.reset()`.
3. Clean reset: legs at HOMIE's `default_angles` and every other joint at 0, then
   `place_robot_collision_free(clearance=SPAWN_CLEARANCE=0.001, extra_backoff=HAMS_SPAWN_BACKOFF)`
   (`:167-183`).
4. `_settle_stance` runs the HOMIE stand-in for 3 s on a copy of `MjData` and copies the settled
   joints back to the spawn (`:81-116`, `:205-214`).
5. It builds `SimInterface` (Unitree DDS), `RosSensorBridge` (sensors) and `MeasurementBridge`
   (`/robocasa/*`). Only the measurement node runs in a background executor.
6. Main loop under `sim_lock`, per step:
   - `write_ctrl()`, then `mj_step`;
   - unstable-reset check;
   - `after_step()`: stand-in policy at 50 Hz, `rt/lowstate`, `rt/dex3/*/state`;
   - fall logger;
   - `ros_bridge.tick()`;
   - `env.update_state()` and the measurement tick at `control_freq` (20 Hz).

   Real-time pacing follows the loop; the viewer is synced inline (not threaded as on GOLEM).

### 2.2 `g1_robosuite.py`
- Generates the robot MJCF into a temp dir at import, from
  `<repo>/third_party/leprechaun_g1/assets/g1_description/g1_29dof_with_hand_rev_1_0.xml` and its URDF (`:40-43`).
- Adds sensor sites `mid360_link` / `d435_link` and camera `head_cam` with fovy 42.5° (`:82-91`).
- Adds palm eef bodies, the `{arm}_center` sites and the pelvis `framequat`.
- Joint dynamics and the standing stance come from HOMIE's `g1.yaml`, read **at import** from
  `<repo>/g1_locomotion/g1.yaml` or `<repo>/core_ws/src/g1/g1_locomotion/policy/g1.yaml` (`:48-57`).
  The sim therefore needs the policy directory mounted even to build the robot.

### 2.3 `stand_in.py` and `unitree_interface.py`
- `StandIn` runs HOMIE (`homie_policy.HomiePolicy`, an onnxruntime session) on the legs at
  50 Hz with command (0,0,0) and height 0.74 m. It holds the waist and arms at 0 with HOMIE's
  gains, and imports the policy dir via `sys.path` (`stand_in.py:19-24`).
- `SimInterface`:
  - Until the first valid `rt/lowcmd` (CRC, `mode_machine==5`, `mode_pr==0`), the stand-in owns
    all 29 body motors. `rt/arm_sdk` blends slots 12-28:
    `tau = w*PD(arm_sdk) + (1-w)*hold`, with `w = clip(motor_cmd[29].q, 0, 1)`.
  - `rt/arm_sdk` silent for `HAMS_ARM_SDK_TIMEOUT` (0.5 sim-s) ramps `w` to 0 at 1/s, and back
    up when frames resume.
  - After release, `rt/lowcmd` drives every motor and `rt/arm_sdk` is ignored. `rt/lowcmd`
    silent for `HAMS_CMD_TIMEOUT` (0.5 sim-s) zeroes every motor.
  - Dex3 commands are applied per the mode byte (status enabled, no timeout bit).
  - Torques are clipped to `actuatorfrcrange`. `rt/lowstate` carries `tau_est` from
    `qfrc_actuator`, IMU quat/gyro/acc/rpy, and is CRC-signed every step.

### 2.4 Sensors (`g1_lidar.py`, `g1_head_camera.py`, `g1_sensor_msgs.py`, bridge)
- **MID-360**:
  - Golden-ratio (Fibonacci) non-repetitive pattern, elevation −7..52° in the sensor frame.
  - The sensor is mounted upside down per the URDF.
  - 5000 points per frame (`HAMS_LIDAR_POINTS`) at 10 Hz (`HAMS_LIDAR_HZ`, ≤100).
  - Rays are cast on the `mid360_scan` thread from a 20-slot ring of geom poses captured every
    5 physics steps. The thread holds its own model copy and syncs geom/material alpha.
  - Per-point `offset_time` matches the pose each point was cast from. The first hit on a robot
    geom gives no point (the body still occludes).
  - `/livox/lidar` CustomMsg is serialized as CDR bytes in `livox_frame` with `lidar_id` 192.
    QoS is reliable, depth 256.
  - No `/livox/pointcloud`.
- **IMU**: `/livox/imu` at 200 Hz in `livox_frame`. Angular velocity in rad/s, specific force
  **in g**, orientation left at identity, as the real driver publishes it.
- **D435 head camera** (`/realsense/head/*`):
  - Topics: `color/image_raw`, `color/image_raw/compressed`,
    `aligned_depth_to_color/image_raw` (16UC1 mm), `.../compressedDepth`, `color/camera_info`.
  - 1280x720 at 6 Hz by default (`HAMS_CAMERA_SIZE`, `HAMS_CAMERA_HZ` ≤30).
  - One static TF, `d435_link -> head_color_optical_frame`.
  - Rendered on a `head_camera` thread that owns the GL context. Due frames are dropped while a
    render is in flight. Compressed topics are always encoded.
- **Optional ground truth**: `/odom` + `odom->pelvis` only with `HAMS_SIM_ODOM=1` (`g1_mujoco.py:198`).

### 2.5 Topic surface: G1 sim vs GOLEM camera_ready H1 sim [V]

| | G1 sim (`sim/`) | H1 sim (`h1_robocasa/`, camera_ready) |
|---|---|---|
| Unitree DDS published | `rt/lowstate` (every step, tick = ms, `mode_machine` 5, CRC), `rt/dex3/{l,r}/state` | `rt/lowstate` (wall `RecurrentThread`, tick = step count, no `mode_machine`) |
| Unitree DDS consumed | `rt/lowcmd` (CRC/mode-validated), `rt/arm_sdk`, `rt/dex3/{l,r}/cmd` | `rt/lowcmd` (unvalidated, 27 motors) |
| Hold before control | HOMIE stand-in (Unitree controller emulation) | elastic band on torso + `/elastic_band/toggle` |
| `/clock` | yes | yes |
| Lidar | `/livox/lidar` (5k pts/frame, `livox_frame`) | `/livox/lidar` (360x56=20k pts/frame, `lidar_link`), **plus `/livox/pointcloud`** |
| IMU | `/livox/imu` 200 Hz, **g**, identity orientation, `livox_frame` | `/livox/imu` 200 Hz, **m/s²**, orientation filled, `lidar_link` (`mujoco_ros_bridge.py:498-519`) |
| Cameras | head only: color(+compressed), aligned depth(+compressedDepth), color/camera_info; 1280x720@6 Hz | head + left_hand + right_hand; also depth/image_rect_raw(+compressedDepth), depth/camera_info, aligned_depth camera_info, depth/color/points; head 640x360 (scale 0.5) @15 Hz |
| TF | `/tf_static` one edge | `/tf_static` profile tree + driver edges on `/tf` @10 Hz |
| Grippers | none (Dex3 over DDS) | `/{left,right}/gripper/*` services, state, DeliGrasp |
| Task signals | `/robocasa/{task_name,task_goal,success,reward}` | same |
| Optional | `/odom`+TF (`HAMS_SIM_ODOM`) | `/odom`+TF (`GOLEM_SIM_ODOM`), `/sim/ground_truth`+`/sim/reset_objects` (`GOLEM_SIM_EXPERIMENT`) |

Both follow the "ground truth off by default" rule. G1's acceptance tests read the truth inside
the sim process (a recorder patched onto `SimInterface.after_step`) and never publish it, which
matches GOLEM's `tests/stack/observe_simulator.py` pattern.

---

## 3. Environment variables

### 3.1 Every knob read [V]
- **G1 sim**:
  - `HAMS_SPAWN_BACKOFF` (`g1_mujoco.py:175`), `HAMS_SIM_ODOM` (`:198`);
  - `HAMS_ARM_SDK_TIMEOUT` (`unitree_interface.py:87`), `HAMS_CMD_TIMEOUT` (`:91`);
  - `G1_DDS_INTERFACE` (`:254`), `ROS_DOMAIN_ID` (`:199`, `mujoco_ros_bridge.py:529`);
  - `HAMS_LIDAR`, `HAMS_LIDAR_POINTS`, `HAMS_LIDAR_HZ`, `HAMS_CAMERAS`, `HAMS_CAMERA_HZ`,
    `HAMS_CAMERA_SIZE` (`mujoco_ros_bridge.py:153-173`).
- **G1 launcher and compose**: `MUJOCO_GL` is forced to glfw with the viewer and egl when
  headless. `DISPLAY` decides headless mode. `ROS_DOMAIN_ID` defaults to 1. Compose passes
  the HAMS sensor/timeout/odom knobs but **not** `HAMS_SPAWN_BACKOFF`.
- **G1 tests and demos**: `SIM_IMAGE`, `ROS_IMAGE`, `SIM_GPUS`, `CORE_WS`, `OUT`,
  `WALK_BACKOFF`, `NAV_BACKOFF`, `NAV_GOAL`, `NAV_KEEP_LIDAR`, `EXPLORE_BACKOFF`,
  `G1_SIM_TEST_RECORD_EVERY`, plus the HAMS sensor knobs.
- **GOLEM H1 sim**:
  - spawn: `GOLEM_SPAWN_BACKOFF`, `GOLEM_SPAWN_MARGIN`, `GOLEM_SPAWN_OFFSET` (`h12_mujoco.py:180-187`);
  - tether: `GOLEM_SIM_TETHER_ROT_STIFFNESS`, `GOLEM_SIM_TETHER_ROT_BODY` (`:233-234`);
  - odom: `GOLEM_SIM_ODOM` (`:260`);
  - cameras: `GOLEM_CAMERAS`, `GOLEM_CAMERA_SCALE`, `GOLEM_HAND_CAMERA_RES`,
    `GOLEM_CAMERA_DEPTH`, `GOLEM_CAMERA_POINTCLOUD` (`:293-304`);
  - lidar: `GOLEM_LIDAR_SELF_FILTER` (`:341`);
  - experiment: `GOLEM_SIM_EXPERIMENT` (`measurement_bridge.py:68`);
  - watchdog and DDS: `GOLEM_CMD_TIMEOUT` (`unitree_interface.py:84`), `ROS_DOMAIN_ID`
    (`unitree_interface.py:21`, `mujoco_ros_bridge.py:827`);
  - renderer: `MUJOCO_GL` (`mujoco_ros_bridge.py:360`).
- **GOLEM launcher, compose and scripts**:
  - `launch_robocasa.sh` forces `MUJOCO_GL=egl` in both display modes (`:45`), reads
    `GOLEM_PREPARE_ONLY` and runs `validate_sim_domain`.
  - The x86 compose passes `GOLEM_SPAWN_MARGIN`, `GOLEM_CAMERAS`, `GOLEM_CAMERA_SCALE`,
    `GOLEM_CAMERA_DEPTH`, `GOLEM_CAMERA_POINTCLOUD`, `GOLEM_HAND_CAMERA_RES`, `GOLEM_SIM_ODOM`,
    `GOLEM_SIM_EXPERIMENT`, `GOLEM_SIM_TETHER_ROT_*` and `GOLEM_SPAWN_OFFSET`. It does **not**
    pass `GOLEM_SPAWN_BACKOFF`, `GOLEM_CMD_TIMEOUT` or `GOLEM_LIDAR_SELF_FILTER`; the mac
    compose passes the first two.
  - Script-level: `GOLEM_ASSETS_DIR`, `GOLEM_CACHE_DIR`, `GOLEM_DISPLAY`, `GOLEM_PLATFORM`,
    `GOLEM_WAIT_FOR_SIM`, `GOLEM_SIM_TIMEOUT`, `GOLEM_USE_RVIZ`, `GOLEM_TEST_*`.

### 3.2 Mapping [V for code, I where marked]

| G1 | GOLEM | Relation |
|---|---|---|
| `HAMS_SIM_ODOM` | `GOLEM_SIM_ODOM` | **Same meaning.** Same parser and frames (`odom->pelvis`, `/odom` at 50 Hz), default off. The base name was `HAMS_SIM_ODOM`, renamed in GOLEM only |
| `HAMS_SPAWN_BACKOFF` | `GOLEM_SPAWN_BACKOFF` | Same intent. Implementation differs: GOLEM backs off while keeping its margin; G1 stops before the first contact behind |
| `HAMS_CMD_TIMEOUT` | `GOLEM_CMD_TIMEOUT` | Same meaning (`rt/lowcmd` silence → zero torque, sim seconds, default 0.5). G1 validates finite and positive, and it applies only after the stand-in is released |
| `HAMS_CAMERAS` | `GOLEM_CAMERAS` | Same meaning (0 = no cameras). Scope differs: G1 has the head only, H1 all three |
| `HAMS_CAMERA_SIZE` (WxH, default 1280x720) | `GOLEM_CAMERA_SCALE` (fraction of 1280x720, default 0.5) and `GOLEM_HAND_CAMERA_RES` (WxH, hands) | **Collision in encoding and default.** Same concept, different units. G1 renders full resolution by default, H1 half |
| `HAMS_CAMERA_HZ` | none (H1 hard-codes 15 Hz) | G1-only |
| `HAMS_LIDAR`, `HAMS_LIDAR_POINTS`, `HAMS_LIDAR_HZ` | none (H1 lidar always on, 360x56 @10 Hz) | G1-only |
| `HAMS_ARM_SDK_TIMEOUT` | none | G1-only (`rt/arm_sdk` does not exist on H1) |
| `G1_DDS_INTERFACE` | none (the H1 sim's Unitree SDK uses default interface selection) | G1-only; no name clash |
| (always on in G1's lidar model) | `GOLEM_LIDAR_SELF_FILTER` | Near-equivalent. G1 always drops robot first-hits; H1 makes it a default-on knob |
| none | `GOLEM_SPAWN_MARGIN`, `GOLEM_SPAWN_OFFSET` | H1-only. GOLEM CLAUDE.md says `GOLEM_SPAWN_MARGIN` applies "on both platforms" (mac and x86). For the G1 it would be a silent no-op unless the placement is shared, and then its 0.20 m default would change G1's spawn [I] |
| none | `GOLEM_CAMERA_DEPTH`, `GOLEM_CAMERA_POINTCLOUD`, `GOLEM_HAND_CAMERA_RES` | H1-only (G1 has no depth stream, cloud or hand cameras) |
| none | `GOLEM_SIM_TETHER_ROT_*` | H1-only (G1 has no band) |
| none | `GOLEM_SIM_EXPERIMENT` | H1-only now; G1 gets it free by adopting GOLEM's `measurement_bridge.py` |
| `MUJOCO_GL` (glfw with viewer, egl headless) | `MUJOCO_GL` (egl always) | **Launcher conflict.** GOLEM makes the EGL camera context before the GLFW viewer. G1 makes its renderer on its camera thread before the viewer, so it should work under EGL as well [I] |
| `ROS_DOMAIN_ID` | `ROS_DOMAIN_ID` | Same. Both reject 0 |

**Hard blocker** [V]: GOLEM's `docker/scripts/docker_common.sh:25` refuses to run if any `HAMS_*`
variable is in the environment ("Rename HAMS_* variables to GOLEM_*."). G1's knobs and docs
(`HAMS_SIM_ODOM=1 tests/test_sim_nav.sh`, and others) therefore cannot coexist with GOLEM's run
scripts. The port must rename `HAMS_*` to `GOLEM_*` (or `GOLEM_G1_*` where the meaning differs,
e.g. `HAMS_CAMERA_SIZE`). For the shared names (`SIM_ODOM`, `SPAWN_BACKOFF`, `CMD_TIMEOUT`,
`CAMERAS`), the GOLEM name can be reused as is.

---

## 4. Runtime needs

**Image** [V]. The G1 sim runs in the robocasa image. G1's `RobocasaDockerfile` is GOLEM's
older one renamed (`hams_base`), plus:
1. `onnxruntime==1.23.2` appended at the end, installed via `pip freeze > /opt/g1-constraints.txt`
   (G1 `docker/RobocasaDockerfile:153-162`).
2. **Unpinned** `git clone --depth 1` of robosuite, robocasa, robosuite_models, mimicgen and
   Livox-SDK2.
3. Livox built from `core_ws/src/third_party/livox_ros_driver2`.
4. A stale `WORKDIR /home/code/g1_robocasa`, overridden by compose `working_dir: /home/code/sim`.

**Versions**:

| | GOLEM camera_ready | G1 (at its build) |
|---|---|---|
| MuJoCo | 3.3.1 (`locks/robocasa-*`) | 3.3.1 |
| robosuite | `5ce6643f` | master at build time |
| robocasa | `4f8a2980` | master at build time |
| robosuite_models | `2d63a6b4` | master at build time |
| mimicgen | `72bd767c` | master at build time |
| Livox-SDK2 | `08f523c` | master at build time |
| numpy | 2.2.5 / 2.2.6 (locked) | `numpy>=2.2.6` |
| `livox_ros_driver2` gitlink | `13eb05e` | `13eb05e` (identical) |
| `unitree_sdk2_python` | `c158c8d` | `c158c8d` (identical) |

GOLEM's pins were recorded 2026-09-17 (`docker/locks/SOURCES.md`). G1's images were built about
2026-10-06 from master. **The G1 sim has never run against GOLEM's pinned robosuite/robocasa
[I: plausible but unverified; needs a build plus `tests/test_sim_bringup.sh`].**

**What a port must change in the image** [V]: add `onnxruntime==1.23.2` through a committed
GOLEM lock. `tests/docker/test_source_pins.py:71-79` requires every `pip install` to use
`-c /opt/golem-locks/...`, and `:16-26` requires every `git clone` to be pinned. G1's Dockerfile
pattern would fail both. Do not copy G1's Dockerfile; append one locked `uv pip install` line at
the end of GOLEM's (layer discipline), plus an arm64 lock if mac matters. Everything else the G1
sim imports is already in GOLEM's robocasa image [V for the lock contents]: pyyaml 6.0.3,
pillow 12.3.0, scipy 1.15.3 (test reports), `/opt/livox_ws`, and `unitree_sdk2py` from base.

**Mounts the G1 sim expects** [V] (G1 `docker/docker-compose.yml:2-59`, plus the test scripts' `COMMON`):

| Mount | Purpose |
|---|---|
| `../sim:/home/code/sim` (rw in compose, ro in tests) | the sim code |
| `../third_party/leprechaun_g1:/home/code/third_party/leprechaun_g1:ro` | robot MJCF/URDF/meshes, read at import |
| `../core_ws/src/g1/g1_locomotion/policy:/home/code/g1_locomotion:ro` | `deploy.onnx` (1.8 MB, plain git), `g1.yaml`, `homie_policy.py`, needed at import |
| `./scripts:/home/code/h12_sim_scripts` | launcher |
| `../tests:/home/code/tests:ro` | acceptance tests only |
| X11 socket, Xauthority, `/dev/input`, `/dev/dri` | in the base compose service |

Path assumptions in code: `g1_robosuite._REPO = dirname(sim)/..` (`:40`) and
`stand_in._ROOT` (`:19-23`) resolve the leprechaun and policy dirs relative to the sim dir's
parent. 15 files hard-code `sim` / `/home/code/sim`: tests, demos, spatial_memory, launcher,
compose, Dockerfile, BUILD.md.

**GOLEM's mounts for `h1_robocasa`** [V] (`docker/docker-compose.yml:64-156`):
- `../h1_robocasa:/home/code/h1_robocasa` (rw) and `CL_Assets` (ro);
- `./scripts:/home/code/h12_sim_scripts` (ro);
- `container_cache/msgs_ws` with `magpie_msgs` and `custom_ros_messages` ro inside it. The
  launcher colcon-builds these for `magpie_hand_bridge`; the G1 sim does not need them, but the
  build is harmless.

Other differences:
- GUI mounts live in `docker-compose.gui.yml`. `tests/docker/test_workflows.py:123-136` asserts
  the base robocasa service has no X11/Xauthority mounts. G1's pattern (X11 in the base service)
  would fail that test if copied.
- `prepare_mounts robocasa` (`docker_common.sh:90-95`) requires `h1_robocasa/h12_mujoco.py` and
  the msgs packages.
- G1 test containers use `--gpus all`, `--network none` and `--cap-add NET_ADMIN` (to set
  MULTICAST on `lo`).

---

## 5. camera_ready `h1_robocasa` features vs the G1 sim

| H1 feature | Relevance to G1 | Conflict? |
|---|---|---|
| `realsense_profile.py/.yaml` | The code is generic; the YAML is H1-only (anchor `head_camera_link`, H1 hand-eye calibration, H1 D435i intrinsics, hand D405 trees). G1 deliberately models the D435 colour stream from its URDF `d435_link` with fovy 42.5° and no captured profile (the real head camera is out of scope in G1). A G1 profile could be added later once a real G1 capture exists. | Both use the frame id `head_color_optical_frame`, but with different parent trees. Fine as long as each robot's URDF provides its parent |
| `sim_timing.py` (`RealtimePacer`, `LoopTimings`) | Directly reusable. G1's inline pacer has the same semantics (resync past 0.1 s lag). Adopting it gives G1 `--timing` and GOLEM's `tests/robocasa/test_loop_timing.py` | none |
| `magpie_hand_bridge.py` | H1-only (Magpie grippers). G1 uses Dex3 over DDS | none if left out |
| `battery_env.py` | Registers `Battery` and is built `robots="H1_2"` from `CL_Assets/mujoco_assets/battery`. G1 does not mount `CL_Assets` | none if left out; usable later |
| `mujoco_env.py` (`ElasticBand`) | G1 replaces the band with the HOMIE stand-in, which is the more faithful emulation of the G1's real start-up | none |
| Spawn margin (`GOLEM_SPAWN_MARGIN`, `_robot_env_gap`, limp poses) | If placement is shared, G1 must keep margin=0 and floor clearance 0.001 by default to preserve its tested spawn (G1's offsets and backoffs were tuned without a margin). G1's "colliding geoms only" floor fit must become a parameter (default False for H1), or the H1 spawn height may change | **Yes** (§8) |
| Spawn offset (`GOLEM_SPAWN_OFFSET`) | Easy to add to G1 | none |
| Experiment surface (`GOLEM_SIM_EXPERIMENT`) | Free if `measurement_bridge.py` is shared | none |
| Camera knobs | Different encoding (§3) | naming only |
| `h1_robocasa/tests/` | Contains only an untracked `__pycache__` [V]. GOLEM's sim tests live in `tests/robocasa/` (`test_loop_timing`, `test_realsense_profile`, `test_sensor_bridge`, all importing `h1_robocasa`) and run via `tests/robocasa/run_in_container.sh`. `h1_robocasa/test_spawn_collision.py` still sits in-dir | conventions differ (§6) |

---

## 6. Tests

### 6.1 G1 acceptance tests [V]
All three runners share the same pattern:
1. Start the sim headless in RoboCasa layout 9 / style 9 / seed 42 inside a `docker run
   --network none --cap-add NET_ADMIN --gpus all` container, setting MULTICAST on `lo` and
   `G1_DDS_INTERFACE=lo`. The ros container joins with `--network container:<sim>`.
2. The sim runs `tests/test_sim_*.py sim`, which uses `runpy` on `g1_mujoco.py` with
   `SimInterface.after_step` (and, for nav, `__init__`) monkeypatched to record state into
   `<out>/sim.npz`. **This couples the tests to the `SimInterface` class and method names.**
3. The ros container runs `launch_ros.sh` (builds `core_ws`), `ros2 launch g1_bringup
   g1_sim_bringup.launch.py ...` and the `drive` role.
4. `report` checks the recordings offline in the sim image.

Other conventions:
- Images default to `hams_sim_robocasa:latest` and `hams_ros:latest`; containers are named
  `g1_sim_test[_ros]_<pid>`, `g1_sim_nav*`, `g1_sim_explore*`.
- Output goes to `container_cache/{sim_bringup_test,sim_nav_test,sim_explore_test}/<time>_<case>`.
- The scripts do not read `docker/.env`. `ROS_DOMAIN_ID` must be greater than 0 (default 1).
- Each run ends with `PASSED: n/n checks`.

What each checks:
- **`test_sim_bringup.sh`** (19 checks, arms):
  - `rt/arm_sdk` engage ramp 0→1 in about 1 s of wall time;
  - `home` at startup (≤0.1 rad);
  - NamedConfig `t_pose`;
  - `/left/hand/close` (≤0.05 rad);
  - hand-back ramp 1→0 after killing `frame_task_server`;
  - the stand-in kept and the robot standing (pelvis ≥0.70 m, tilt ≤15°);
  - a 10 s sensor window: rate and gap per topic in sim time, frame ids, contents, the exact
    topic surface of the sim node, the camera TF, IMU contents, and a lidar frame placing the
    floor where the upside-down mount predicts.
- **`--walk`** (18 checks): the stand-in is handed over to the walking node's `rt/lowcmd`, the
  robot keeps standing, and walks 0.3 m/s ±30% for 5 s, plus the sensor checks.
  `HAMS_SPAWN_BACKOFF=2.0`.
- **`test_sim_nav.sh`** (12 checks):
  - the sim maps the kitchen from its own geometry and picks a goal 2-3 m away (or `NAV_GOAL`);
  - Nav2 must reach it within the timeout, and the true pelvis must end within tolerance;
  - `/cmd_vel` caps with the velocity smoother as the only publisher;
  - no `/scan` stall;
  - no phantom cells in the map or scan;
  - `/odom` published only for `sim_gt`;
  - localisation error, never fell, no robot-environment contact, no fall in `sim.log`.

  Runs with `fastlio` by default or `sim_gt` with `HAMS_SIM_ODOM=1`.
- **`test_sim_explore.sh`** (9 checks): frontier exploration with a 120 s sim budget, the
  never-fell, contact, `/cmd_vel` and map-phantom checks, coverage of the reachable floor, and
  `goals_reached` consistency.

**Runtime** (G1 README, 2026-10-05, their machine): 114 s arms (63 s of it the `core_ws` build),
49 s walk, 41 s nav, 55 s explore, about 4.3 min in all. In GOLEM the first run's `core_ws` build
covers far more packages (`model_server`, `h12_*`, MJPC; `h12_deploy_mjpc` auto-skips its C++
controller without the mjpc build mount), so expect a much longer first run [I].

**G1 sim unit tests** (`sim/test_*.py`, script style with a `main()`):
- run in the robocasa image with `--network none`, and `ROS_DOMAIN_ID=97` for
  `test_unitree_interface`;
- `test_g1_sensors` / `test_mujoco_ros_bridge` need EGL;
- `test_spawn_collision` runs on the host.

### 6.2 GOLEM camera_ready `tests/` conventions [V]
- `tests/README.md` describes the suites: CPU unittests (`tests/isaaclab`, `tests/robocasa`,
  `tests/docker`, `tests/stack` scoring) run in CI (`.github/workflows/simulator-tests.yml`,
  path filter includes `h1_robocasa/**`, `tests/**`). GPU/ROS integration wrappers run on
  trusted workers.
- Wrappers source `docker/scripts/docker_common.sh` (`load_config`, `check_domain`,
  `prepare_mounts`) and use `compose run --rm` (host networking) with a dedicated domain per
  test (186-193).
- Artifacts go to `tests/results/<name>` (gitignored), with `run.json` recording the git state
  and image IDs. Container names are prefixed `golem-stack-test-<domain>-<pid>`.
- `tests/stack/run.sh:69-73` copies `launch_robocasa.sh` and asserts the literal `h12_mujoco.py`
  appears **exactly once**, then swaps in an observer.

### 6.3 How the G1 tests would fit
- **Place them** in `tests/g1/` (or `tests/robocasa_g1/`): the shell runners plus `test_sim_*.py`.
  Move the G1 sim unit tests to `tests/robocasa_g1/` with a `run_in_container.sh` modelled on
  `tests/robocasa/run_in_container.sh`, or keep them beside the sim as `h1_robocasa` does with
  `test_spawn_collision.py`.
- **Retarget**:
  - default images `golem_sim_robocasa:latest` and `golem_ros:latest`;
  - output to `tests/results/g1-*`, with a `run.json` like `tests/stack/run.sh`;
  - container prefix `golem-g1-*`;
  - `HAMS_*` → `GOLEM_*`;
  - the sim path and mounts (§4).
- **Keep `--network none`.** It is stricter than GOLEM's domain isolation and compatible with
  GOLEM's rules. Optionally source `docker_common.sh` for `check_domain` only; `load_config`
  rejects `HAMS_*`, which the rename fixes.
- **Dependencies**: the tests need the G1 ROS packages in GOLEM's `core_ws` and the G1 ROS
  dependencies (onnxruntime, pink, ...) in `golem_ros` (another auditor's scope).
- **CI**: add the G1 sim directory to the workflow path filter. GPU acceptance would be a
  manual job, like `tests/stack/run_all.sh`.

---

## 7. `demo/` and `spatial_memory/`

### 7.1 `demo/` [V]

| File | Size | Storage |
|---|---|---|
| `walk/demo_walk.sh` + `demo_walk.py` | 5.9 KB + 3.1 KB | plain git |
| `navigation/g1_nav2_kitchen.mp4` | 14.6 MB | **Git LFS** (`.gitattributes`: `demo/**/*.mp4 filter=lfs`; oid `0d2be7069b`) |
| `navigation/g1_nav2_kitchen_obstacle.mp4` | 31.1 MB | **Git LFS** (oid `d69ae489c3`) |
| `navigation/*_path.png`, `*_tracking.png` (4 files) | 143-255 KB each, ~756 KB total | plain git |
| `navigation/record_nav_demo.sh`, `render_replay.py`, `rviz_capture.py`, `plot_run.py`, `nav_demo.rviz`, `bag_qos.yaml`, `report*.log`, `README.md` | small | plain git (`.gitignore` re-includes the logs) |
| `navigation/docker/Dockerfile` | `FROM hams_ros:latest` plus xvfb, unpinned `apt-get` | plain git (outside `docker/`, so GOLEM's pin tests would not scan it) |

What each demo does:
- **`demo_walk.sh`**: sim with viewer plus the walking bringup plus a `/cmd_vel` route. Same
  `--network none` pattern; containers `g1_demo_walk[_ros]_<pid>`.
- **`record_nav_demo.sh`**: runs the nav test with `ros2 bag record`, replays RViz in
  `hams_ros_rviz` under Xvfb, and renders a MuJoCo chase view beside it.

Port notes:
- GOLEM has no root `.gitattributes`; it uses per-directory ones
  (`core_ws/src/lmg_deploy/*/.gitattributes`, `model_server/weights/.gitattributes`). A
  `demo/.gitattributes` would be needed.
- The two LFS objects must be pushed to GOLEM's LFS remote.
- The README cites G1 SHA `c99fdf9`, which is meaningless in GOLEM because the history was
  rewritten.
- Consider leaving the 46 MB of videos out of GOLEM and linking to G1_GOLEM instead.

### 7.2 `spatial_memory/` benchmark [V]

**What it is.** A RoboCasa spatial-memory benchmark:
1. **Capture**: `spatial_memory_scan.py`, `spatial_memory_live.py` and
   `spatial_memory_benchmark.py` render a *virtual camera* in a RoboCasa kitchen after **parking
   the robot below the scene** and hiding robot geoms in group 5.
2. **Evaluation**: `evaluate.py`, `metrics.py`, `report.py`, `compare.py`, `quota.py`,
   `run_metadata.py`, `resize_dataset.py`, `rerank_validation.py`.
3. **Memory adapters**: latest-only, the EmbodiedAgent variants (recency, VLM rerank via
   Gemini), and VLMaps.

It is driven by `docker/scripts/spatial_memory_camera.sh` (1024 lines). That script runs the
capture via `docker compose --profile robocasa run` (sim image, working dir `/home/code/sim`) and
the backends in `hams_spatial_memory` / `hams_vlmaps` images (`docker/SpatialMemoryDockerfile`,
`docker/VLMapsDockerfile`). External dependencies: an `EmbodiedAgent` checkout, optionally
`Humanoid_Simulation`/VLMaps, and `GEMINI_API_KEY`.

**Lineage.** On GOLEM, `origin/spatial-memory-robocasa-benchmark` has its tip at `c0af423`; it
has not moved and is **not** an ancestor of camera_ready. There the code lived in three places:
- `benchmarks/spatial_memory/*` (the package `benchmarks.spatial_memory`);
- `h1_robocasa/spatial_memory_{benchmark,geometry,live,scan}.py`, importing `h1_2_robosuite`;
- `tests/test_spatial_memory_*.py`.

That branch also added `h12_skills` spatial_memory node files, docs, compose and Dockerfiles.
G1 moved the code to a top-level package `spatial_memory/` with `spatial_memory/tests/`.

The content is near-verbatim:
- 14 of 16 benchmark modules are byte-identical.
- `evaluate.py` changed 12 lines (adapter module paths, repo-root depth).
- The 3 capture scripts changed 7-13 lines each: `import g1_robosuite` via `sys.path` to
  `<repo>/sim` instead of `h1_2_robosuite`, and the docstring wording.
- The 13 tests changed 2-14 lines each (import paths).
- `spatial_memory_camera.sh` changed 15 lines (mounts and module paths); the SpatialMemory
  Dockerfile is unchanged; the VLMaps Dockerfile changed 2 lines.

**Robot-specificity.** The benchmark is essentially robot-agnostic. Its only robot touchpoints
are `make_kitchen_env(...)` and `place_robot_freejoint(...)`, which have the same signature in
both robosuite modules, and the `robot0_`/`gripper0_` prefix used to hide the robot. The robot
choice only changes the import's dependencies:
- G1 needs leprechaun plus the HOMIE `g1.yaml` at import;
- H1 needs `CL_Assets`.

camera_ready has none of this work. Porting it is a separate decision from G1 support.

---

## 8. Port risks and conflicts

1. **`HAMS_*` env vars are rejected by GOLEM's scripts** (`docker_common.sh:25`) [V].
   A rename is mandatory.
2. **Shared-infra rule** [V for the rule]. Any change to `h1_robocasa/` or
   `launch_robocasa.sh` must keep H1 defaults identical. Watch three places:
   - `tests/stack/run.sh:69-73` requires `h12_mujoco.py` exactly once in `launch_robocasa.sh`;
   - `tests/docker/test_workflows.py:123-136` forbids X11 mounts in the base robocasa service;
   - `prepare_mounts` hard-requires H1 files.
3. **Same DDS topics and IDL for both robots** [V]. Both sims publish `unitree_hg` `LowState_` on
   `rt/lowstate` and read `rt/lowcmd`. Nothing but the domain separates a G1 stack from an H1 sim.
   The G1 safety layer gates on `mode_machine` 5, which the H1 sim never sets [I from G1
   CLAUDE.md], but the H1 sim accepts any `LowCmd_` without CRC or mode checks. Recommendations:
   - give the two sims distinct container names and compose profiles;
   - make `docker_stack.sh` robot-aware (its bringup command is hard-coded to
     `h12_sim_bringup.launch.py`; `wait_for_sim.py` is robot-agnostic);
   - never run both sims on one domain.
4. **`/livox/imu` units differ** [V]: g in G1 versus m/s² in H1, plus different frame ids
   (`livox_frame` vs `lidar_link`) and orientation fields. Each robot's FAST_LIO config assumes
   its own sim. Never share one IMU publisher without a per-robot setting.
5. **Placement divergence** [V]. GOLEM's margin-based `place_robot_collision_free` (default
   `max_iters` 50 and margin semantics) and G1's contact-based one (`max_iters` 25, colliding-geom
   floor fit) are both tuned to their robot. A shared module needs per-robot defaults, or a
   behaviour change slips into one of the sims.
6. **Unpinned upstream** [V]. G1 was validated against robosuite/robocasa master (about
   2026-10-06); GOLEM pins 2026-09-17 SHAs. Re-run G1 acceptance on GOLEM's image [I risk: low
   to moderate].
7. **onnxruntime in the robocasa image** must come from a GOLEM lock file [V]. Adding it changes
   the image for H1 too: an additive pip package appended last [I: no effect on the H1 sim].
8. **Hard-coded paths** [V]. `sim/` and `/home/code/sim` appear in 15 files. The
   `third_party/leprechaun_g1` and `core_ws/src/g1/g1_locomotion/policy` locations are resolved
   relative to the sim dir's parent (`g1_robosuite.py:40-51`, `stand_in.py:19-23`). GOLEM has no
   `third_party/` dir and a flat `core_ws/src`; the chosen GOLEM locations must be reflected
   there and in every mount.
9. **Tests depend on internals** [V]. The acceptance tests monkeypatch
   `unitree_interface.SimInterface.after_step` / `__init__` and read `_lowcmd`, `_arm_sdk`,
   `resolver`. A refactor of the G1 sim must keep these hooks or update the tests together.
10. **LFS and history** [V]. Two LFS videos (46 MB) need a GOLEM `.gitattributes` and an LFS
    push. G1 SHAs cited in docs (e.g. `c99fdf9`) do not exist in GOLEM.
11. **HOMIE policy licence** is non-commercial; keep `policy/LICENSE` with the files [V, per G1 CLAUDE.md].

---

## 9. Recommended layout

### Option A (recommended now): a sibling sim directory, `g1_robocasa/` next to `h1_robocasa/`
Copy G1 `sim/*` as is into `g1_robocasa/` (G1's own original name). Then:
- rename `HAMS_*` to `GOLEM_*`;
- fix the repo-relative paths for leprechaun and the policy dir;
- adopt GOLEM's `measurement_bridge.py` by copying or importing it (G1's is a strict subset).

**Pros**: zero change to the H1 sim's code or defaults (shared-infra rule); G1's tested
behaviour is preserved, so the acceptance tests port with mechanical edits; ownership stays clear.

**Cons**: about 300 lines of duplicated RoboCasa glue (placement, patch, task resolution,
viewer framing, odom, `init_ros`), and two sensor stacks to maintain.

### Option B (incremental, after A passes): factor out a small shared module
Create something like `robocasa_common/` (mounted into the robocasa image) and move only the
robot-agnostic parts into it:
- `measurement_bridge.py` (GOLEM's version);
- `sim_timing.py`;
- the task resolution (`_runnable_tasks`, `_random_task`, `_resolve_task`, with the Battery hook
  optional);
- viewer framing and `_euler_to_wxyz`;
- the placement primitives, with per-robot defaults (G1: margin 0, clearance 0.001,
  colliding-only floor fit; H1: current);
- `install_robocasa_patches(robot_cls, offset)`.

Robot directories keep the robosuite registration, `sim_names`, `unitree_interface`, the hold
logic (band vs stand-in), the sensors and the main loop. Each extraction step must pass GOLEM's
`tests/robocasa`, `tests/stack` (H1) and G1's acceptance tests.

### Option C (not recommended): one runner with `--robot h1|g1` and one sensor bridge
The bridges are deliberately different (lidar model, IMU units, camera pipeline and profile), so
this would force behaviour changes on at least one robot.

### Shared-infra touch points for Option A
1. `docker/RobocasaDockerfile`: one appended `uv pip install -c /opt/golem-locks/robocasa-amd64-05.txt onnxruntime==1.23.2`
   with a new lock file (and arm64 if mac G1 is wanted).
2. `docker/docker-compose.yml`: a new service and profile, e.g. `robocasa_g1` (container
   `golem_sim_robocasa_g1`, image `golem_sim_robocasa`, `runtime: nvidia`, host network as today).
   - Mounts: `../g1_robocasa`, the leprechaun assets dir (ro), the HOMIE policy dir (ro),
     `./scripts` (ro).
   - Env: the renamed knobs.
   - No X11 mounts in the base file.

   `docker/docker-compose.gui.yml`: add `robocasa_g1: *gui`.
3. `docker/scripts/docker_common.sh`: a `prepare_mounts robocasa_g1` case. The current
   `require_file .../h1_2_magpie_ros.urdf` for every service is harmless but H1-specific.
4. `docker/scripts/docker_run.sh` and `docker_build.sh`: profile lists. `docker_stack.sh`: a
   robot or sim target that picks the G1 bringup launch.
5. Launcher: a new `docker/scripts/launch_robocasa_g1.sh`, so `launch_robocasa.sh` stays
   untouched and keeps the `tests/stack/run.sh` literal check working. It should force EGL as
   GOLEM does and validate the domain via `runtime_common.sh`.
6. `tests/g1/` (acceptance) and `tests/robocasa_g1/` (unit), CI path filter, `tests/README.md`.
7. `demo/` (G1 demos, `demo/.gitattributes` for LFS) or leave the videos out.
8. Docs: GOLEM `CLAUDE.md` §2, §3, §8 (topic surface and env knobs for the G1 sim) and
   `docs/RUNNING.md`.
9. Spatial memory, as a separate decision coordinated with the branch owner: port it as a
   robot-agnostic top-level `spatial_memory/`, with the capture scripts taking `--robot h1|g1`
   (import `h1_2_robosuite` or `g1_robosuite`), plus `spatial_memory_camera.sh`, two Dockerfiles,
   compose wiring and docs.
