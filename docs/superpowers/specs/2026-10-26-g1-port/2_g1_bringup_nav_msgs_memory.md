# Audit 2: G1 bringup, navigation, interfaces and spatial memory

Scope: G1_GOLEM `main` @ df64d43, packages `core_ws/src/g1/{g1_bringup, g1_navigation,
g1_spatial_memory, g1_msgs}`, compared with GOLEM `camera_ready` @ 42d6559
(`core_ws/src/{h12_bringup, h12_slam, h12_skills, custom_ros_messages @ c60205c, magpie_msgs}`)
and, for spatial memory, GOLEM `c0af423` (tip of `origin/spatial-memory-robocasa-benchmark`).

Method: HEAD-state comparison (per coordinator's scope change): `diff -r` of extracted trees
(`git archive <sha> | tar -x`), `git show <sha>:<path>`, and reading files. No history walks
over package logs. Read-only; nothing in either repo was modified. Extracted trees live under
`/tmp/claude-1000/-home-max-GOLEM/f8c735f2-cb0b-46cc-a631-673fdc69ab41/scratchpad/trees/`.

Legend: **[V]** verified by reading/diffing code; **[I]** inferred/judgement.

---

## 0. Lineage facts that matter for this scope

- **[V]** At the fork tree (G1_GOLEM `9c8d5e0` = GOLEM `c0af423`), GOLEM's bringup was the old
  `h1_bringup` package, which also held navigation (`h1_navigation.launch.py`, `mid360.yaml`,
  `nav2_config.yaml`, `slam_toolbox_h1.yaml`, `MID360_config.json`). There was no `h12_bringup`
  and no `h12_slam`. The same is true at the merge-base `c00feca` (`git ls-tree c00feca core_ws/src/`).
  `c00feca..c0af423` touched `h1_bringup` not at all and `h12_skills` only additively (spatial memory,
  7 files, +1247/-0).
- **[V]** camera_ready renamed `h1_bringup` -> `h12_bringup` and split navigation out into a new
  `h12_slam` package (`git diff --stat -M c00feca HEAD`: 71 files, +4867/-2059). So both G1 and
  camera_ready rewrote the same fork-era nav code independently.
- **[V]** The fork tree pins `custom_ros_messages` at **`bb54bbd`** ("Add spatial memory retrieval
  action", only on `origin/spatial-memory-retrieval-action` in the custom_ros_messages repo). It is
  **not** an ancestor of camera_ready's pin `c60205c`; the two share base `2d7d2ff`.
  `2d7d2ff..bb54bbd` = 1 commit (SkillRetrieveMemory); `2d7d2ff..c60205c` = 6 commits (slow_mode,
  relative, grasp fields, Lmg services).

---

## 1. `g1_msgs` vs `custom_ros_messages`

### 1.1 What "imported as custom_ros_messages" means  [V]

- The directory was imported **with history** from the lab's `CorrellLabCustomROSMessages` repo
  (where the package is `custom_ros_messages`) at the bb54bbd lineage, then renamed
  (`g1_msgs/README.md:8-9`, G1 `docker/BUILD.md:37`, G1 `README.md:650-653`).
- After the rename **everything** says `g1_msgs`: `package.xml:3` `<name>g1_msgs</name>`,
  `CMakeLists.txt:2` `project(g1_msgs)`, every Python import (`from g1_msgs.action import ...`),
  every CLI type string (`g1_msgs/action/NamedConfig`). `grep -rn custom_ros_messages` over G1_GOLEM
  finds only prose (README, CLAUDE.md, BUILD.md, g1_msgs/README). No G1 code imports
  `custom_ros_messages`.
- Consequence: ROS 2 type names embed the package name, so `g1_msgs/action/FrameTask` and
  `custom_ros_messages/action/FrameTask` are **different types on the wire**. H1 tools/clients and
  G1 servers cannot talk to each other while two packages exist.

### 1.2 Interface diff  [V]

g1_msgs HEAD vs the fork pin bb54bbd (what G1 imported):

| Change | Interfaces |
|---|---|
| Removed (other hardware) | `action/HandAdaptiveForce`, `action/ParkerGotoPose`, `msg/MotorCmd`, `msg/MotorCmds`, `msg/MotorState`, `msg/MotorStates`, `srv/EthernetMotor`, `srv/ParkerGetPoseTrigger`, `srv/SetHandAngles`, `srv/UR16BehaviorTrigger` |
| Added | `srv/SetHandPose.srv` (Dex3-1: `float64[] positions` (7, rad, HandCmd_ order), `float64 duration` -> `bool success, string message`) |
| Comment-only | `SkillFrontierExplore.action` `goals_reached` comment (now "reached, or frontier seen on the way"; semantic change in the G1 explorer, not in the type); `GraspGen.srv` `gripper_name` comment ("required, no server default") |
| Renamed | package/project `custom_ros_messages` -> `g1_msgs`; README rewritten |
| Identical | `SkillRetrieveMemory.action` (byte-identical to bb54bbd), `ScanBattery.action`, all other Skill*, DualArm, FrameTask, NamedConfig, msgs, srvs |

g1_msgs HEAD vs camera_ready pin c60205c (what GOLEM has now):

| Change | Interfaces |
|---|---|
| Only in g1_msgs | `action/SkillRetrieveMemory`, `srv/SetHandPose` |
| Only in c60205c | the 10 "other hardware" interfaces above, plus `srv/LmgSetMode`, `srv/LmgSetArmGoals` (camera_ready lmg_deploy) |
| Fields only in c60205c | `FrameTask`: `bool slow_mode`, `bool relative`; `NamedConfig`: `bool slow_mode`; `SkillGrasp`: `bool top_down`, `float32 min_downward_pitch_deg`, `bool visual_servo` (+ arm ""/"none" = auto, timeout 0 = no deadline); `SkillPickPlace`: `bool top_down`, `bool visual_servo` |
| Comment-only | SkillFrontierExplore, GraspGen (as above) |

**Correction to the brief:** `ScanBattery.action` is **not G1-only**: it is in c60205c and identical.
The only G1-only interfaces are **`SkillRetrieveMemory`** and **`SetHandPose`**.

### 1.3 Interfaces G1 code actually uses  [V]

| Interface | Users |
|---|---|
| `action/FrameTask`, `action/NamedConfig`, `action/DualArm`, `msg/StringArray` | `g1_control/ros2/{frame_task_server,frame_task_client,dual_arm_server,dual_arm_client}.py`; `tests/test_sim_bringup.py:290,299` |
| `srv/SetHandPose` | `g1_dex3/hand_node.py:16` |
| `action/SkillFrontierExplore` | `g1_navigation/frontier_explore.py:68`, `g1_navigation/test/test_frontier_explore.py:14`, `tests/test_sim_explore.py:104,127` |
| `action/SkillRetrieveMemory` | `g1_spatial_memory/skills/spatial_memory.py:40` |
| `srv/GeminiQuery`, `srv/SamSegment`, `srv/GraspGen`, `msg/Detection`, `msg/DetectionBundle` | `perception/model_server/{gemini,sam,graspgen,yolo}_server.py` |

Unused by G1: ScanBattery, the 11 other Skill* actions, GoHome, Query, RemovePart, ResetBeliefs,
UpdateBeliefs, UpdateTrackedObject, Dummy. G1 does not use `magpie_msgs` at all (only a README mention).

### 1.4 Single `custom_ros_messages` vs separate `g1_msgs`  [I, strongly recommended]

Keep **one** `custom_ros_messages` and add the two G1 interfaces:

- Both additions are purely additive, and no existing type changes, so there is zero impact on H1 code.
  `SkillRetrieveMemory` already exists upstream (bb54bbd on `spatial-memory-retrieval-action`).
  Its CMake hunk (after `SkillFrontierExplore`, CMakeLists ~L51) does not overlap c60205c's hunk (Lmg
  srvs, ~L34), so merging that branch should be clean **[I: hunk positions read, merge not executed]**.
  `SetHandPose.srv` is new (Dex3 is 7-DoF + duration; the existing `SetHandAngles` is fixed
  `float64[6]` for Inspire, used by Isaac's Inspire path, so it cannot be reused).
- A second package would duplicate ~30 identical interfaces under a different type namespace. The
  shared `model_server` would then need either two variants or a dependency on both (G1's model_server
  imports `g1_msgs`, GOLEM's imports `custom_ros_messages`). camera_ready's RoboCasa sim also builds
  `custom_ros_messages` at start (G1 `docker/BUILD.md:184-188` says the H1-2 sim did, and camera_ready
  `docker/scripts/launch_robocasa.sh` references it).
- G1 migration cost: mechanical. Import/type-string rename in ~13 Python files plus `<depend>`
  in 5 package.xml (g1_bringup, g1_control, g1_dex3, g1_navigation, g1_spatial_memory, model_server).
- **Semantic risk:** with c60205c, `FrameTask.Goal` gains `relative` and `slow_mode`, and
  `NamedConfig.Goal` gains `slow_mode`. G1's `frame_task_server.py` has no reference to either (grep).
  A `relative: true` goal would be **silently executed as absolute** on the G1. The G1 controller
  must reject (or implement) `relative` and decide on `slow_mode`. That belongs to the g1_control
  audit, but it is caused by unifying the messages.
- Ownership: `custom_ros_messages` is a submodule owned by others (GOLEM CLAUDE.md §5). The additions
  must be committed and pushed upstream (ideally by merging `spatial-memory-retrieval-action` and adding
  `SetHandPose`), then the pin bumped in GOLEM. This is an interface change, so announce it.
  Keep every interface G1 deleted (Lmg*, SetHandAngles, Parker*, Motor*, etc.): GOLEM still uses some.

---

## 2. `g1_bringup` vs `h12_bringup`

### 2.1 G1 structure  [V]

- `g1_bringup.launch.py` (real, from the laptop). It includes `g1_core` with `sim:=false`,
  `use_sim_time:=false`, `controller_config:=''`. Args: `use_rviz`, `use_hands`, `safety_config`
  (default `arm_sdk`), `use_locomotion` (false), `start_position_verified` (false), `use_nav` (false).
- `g1_sim_bringup.launch.py` (sim). It includes `g1_core` with `sim:=true`. Args: `use_rviz`,
  `use_hands`, `use_sim_time` (true), `safety_config`, `use_locomotion`, `use_nav`, `odom_source`
  (`fastlio` | `sim_gt`).
- `g1_core.launch.py`: a single `OpaqueFunction(_setup)` (L139-249) that validates and then starts:
  - `robot_state_publisher` (URDF read via `g1_description.paths.urdf_path('with_hand', ros=True)`)
  - `g1_control/joint_state_publisher`
  - `g1_safety_layer/safety_node --config <safety>.yaml` at t=0
  - `frame_task_server --config <controller>.yaml` after `controller_delay` (2.0 s)
  - `dex3_hands` (`use_hands`)
  - `rviz2 rviz/g1.rviz` (`use_rviz`)
  - `locomotion_node --config <g1|sim>.yaml` at `controller_delay + locomotion_delay` (2 + 2 s) when
    `use_locomotion`
  - an include of `g1_navigation.launch.py` (`odom_source`, `sim`, `use_sim_time`) when `use_nav`
- Interlocks and refusals live in Python (`startup_refusal`, L124-136), and a refusal raises
  `RuntimeError`:
  - `nav_refusal` runs first. It refuses `use_nav` on the real robot whatever else is set, refuses it
    without `use_locomotion`, and accepts only `odom_source` ∈ {fastlio, sim_gt}.
  - `pairing_refusal` comes from `g1_control/utility/config_pairing.py` (pairs `arm_sdk`↔`g1`,
    `split`↔`g1_split`, `sim_arm_sdk`↔`sim`, `sim_split`↔`sim_split`; real never takes `sim_*` and
    vice versa).
  - `locomotion_refusal`: on the real robot, `use_locomotion` needs `start_position_verified:=true`.
    The sim auto-switches to `sim_split` (L146-147).
- Tests (`test/test_locomotion_launch.py` 225 lines, 7 tests; `test/test_nav_launch.py` 145 lines,
  6 tests) evaluate `core._setup()` with a synthetic `LaunchContext` and assert node sets, delays,
  arguments and refusals. They need g1_bringup, g1_description, g1_control and g1_navigation
  installed.
- `config/MID360_config.json` is **byte-identical** to camera_ready
  `h12_slam/config/MID360_config.json` (H1-2 network: host IPs 192.168.123.164, lidar .120,
  roll 180). No G1 launch file references it; it is a leftover from h1_bringup.
- `package.xml` still `exec_depend`s `fast_lio`, `pointcloud_to_laserscan`, `slam_toolbox`,
  `nav2_bt_navigator`, `python3-opencv`, `scipy` (h1_bringup leftovers; g1_navigation carries its own).
- Configs are passed by **stem** (`f'{safety_config}.yaml'`), while H1 passes full filenames
  (`'sim_safety_split.yaml'`).

### 2.2 GOLEM camera_ready structure  [V]

- Top-level files (`h12_sim_bringup`, `h12_real_robot_bringup`, `h12_real_desktop_bringup`,
  `h12_safety_grippers`, `lmg_sim_bringup`, `lmg_real_bringup`, `h12_sim_bringup_mac`) include
  declarative building blocks: `h12_state`, `h12_control` (estop -> safety after `safety_delay` 3 s
  -> frame_task after `frame_task_delay` 5 s; the sim uses 0/0), `h12_lowerbody` (RL controller +
  optional MJPC estimator gated by `start_position_verified` via `IfCondition(AndSubstitution)`),
  `h12_models`, `h12_real_drivers`, `h12_grippers`, `h12_locomanipulationgame`, and
  `h12_slam/h12_navigation`.
- Params live in `config/*.yaml` (lowerbody_{sim,real}, mjpc_{sim,real}). No cross-argument
  validation in Python, except `lmg_sim_bringup` refusing `ROS_DOMAIN_ID` unset/0 (L39-43).
- URDF comes from `CL_Assets` (`ASSETS_DIR='/home/code/CL_Assets'` in sim files;
  `GOLEM_ASSETS_DIR` or `/home/unitree/GOLEM/CL_Assets` in `h12_state.launch.py:13`).
- `h12_sim_bringup` defaults `use_nav:=true`; real bringup always includes nav (L258).
- Tests: only `test_real_lidar_config.py` (MID360 IMU/point frame agreement, single config set)
  plus linters.
- `package.xml` exec_depends on H1-only packages: `estop`, `h12_ros2_controller`,
  `h12_safety_layer`, `h12_lowerbody_rl`, `lmg_deploy`, `magpie_msgs`, `cl_realsense`,
  `model_server`, `h12_slam`, ...

### 2.3 Fit into GOLEM  [I]

**Recommendation: keep a separate `g1_bringup` package** next to `h12_bringup`; do not put G1
top-level files into `h12_bringup`.

- Putting them into `h12_bringup` would force `h12_bringup/package.xml` to exec_depend on every
  `g1_*` package, and the reverse is already heavy. Building or using either robot's bringup would
  then require the other robot's whole stack. The package would also keep an `h12_` name while
  owning G1 files, which breaks the per-package ownership rule (GOLEM CLAUDE.md §5).
- G1's `g1_core` design (one validating OpaqueFunction plus launch-logic unit tests) is stronger than
  H1's declarative blocks for cross-argument interlocks. Porting it as-is keeps its 13 tests
  meaningful. Forcing it into the h12 building-block shape would lose the refusal logic, which needs
  all args in one place.
- Reuse candidates:
  - `h12_bringup/launch/h12_models.launch.py` is robot-agnostic except for `use_skills` (graspgen +
    `h12_skills`). G1 could include it with `use_skills:=false` to get gemini/sam/yolo, but that
    makes g1_bringup depend on h12_bringup and its H1 deps. Launching `model_server` from a neutral
    place is cleaner.
  - camera_ready's `livox_imu_upright` (`h12_bringup/h12_bringup/livox_imu_upright.py`) is the proven
    fix for exactly the G1's open `TODO(lidar)` (see §3.5).
- GOLEM CLAUDE.md §6 and §4 need a G1 section. The `docker/mac` `PKGS` list is irrelevant to G1
  (x86 only).

### 2.4 G1 bringup assumptions that need adapting  [V for the facts, I for the remedies]

- **`HAMS_SIM_ODOM`**: mentioned in `g1_core.launch.py:30-31,284`, `g1_sim_bringup.launch.py:7,20,53`
  and `g1_navigation.launch.py:17,19,240`. These are documentation/argument descriptions only; the
  launch files never read it. GOLEM's equivalent is `GOLEM_SIM_ODOM` (`h1_robocasa/h12_mujoco.py:260`).
  **camera_ready `docker/scripts/docker_common.sh:25` fails if any `HAMS_*` variable is set**
  ("Rename HAMS_* variables to GOLEM_*"), so every `HAMS_*` knob in the G1 sim and tests must
  become `GOLEM_*`.
- **`G1_DDS_INTERFACE`** (`g1_bringup.launch.py:7`) is a G1 docker/.env variable for the laptop's
  wired port. camera_ready docker has no equivalent (`CYCLONEDDS_URI=file:///home/code/cyclonedds.xml`
  only). It is needed for the G1 real-robot flow, and it is docker scope.
- **`/home/code/g1_locomotion`**: `g1_core` does not pass `policy_dir`, so `locomotion_node` uses its
  default `DEFAULT_POLICY_DIR = '/home/code/g1_locomotion'` (`g1_locomotion/locomotion_node.py:16`).
  That only exists if the GOLEM compose mounts `core_ws/src/g1/g1_locomotion/policy` there. Install
  the policy to the package share instead, or add the mount.
- **Unscoped Humble launch configs**: `tests/test_sim_explore.sh:132-133` passes
  `frontier_explore:=true` to `g1_sim_bringup`, which does not declare it, and `g1_core` does not
  forward it. It reaches `g1_navigation.launch.py` only because Humble launch configurations leak into
  includes. It works, but it is fragile (the same Humble caveat as GOLEM CLAUDE.md §6). Declare and
  forward it explicitly.
- Domain: G1 assumes sim domain 1 and real domain 0 from the laptop. This matches GOLEM's convention.
- Python cross-package imports at launch load: `g1_description.paths`,
  `g1_control.utility.config_pairing` (`g1_core.launch.py:68-73`), and `g1_navigation.frames`
  (`g1_navigation.launch.py:53-54`).
- `rviz/g1.rviz` uses fixed frame `pelvis` and `/left_ee_pose`, `/right_ee_pose` topics. No paths.

---

## 3. `g1_navigation` vs `h12_slam`

### 3.1 What G1 took at the fork  [V]

Very little beyond the idea. The fork-era `h1_bringup/launch/h1_navigation.launch.py` (161 lines)
had:
- FAST-LIO on `mid360.yaml` (identity extrinsic)
- a **hard-coded** `body -> pelvis` static TF (H1-2 numbers)
- `pointcloud_to_laserscan` on `/cloud_registered_body` with `target_frame: pelvis`
- slam_toolbox with `base_frame: pelvis`, `odom_frame: camera_init`
- Nav2 via `nav2_bringup/navigation_launch.py` with the stock BT

G1's `g1_navigation.launch.py` (256 lines) is a rewrite. `g1_slam_toolbox.yaml` says "Started from
the H1's slam_toolbox_h1.yaml" (L15). `g1_fastlio*.yaml` start from FAST_LIO upstream values. The
G1 Nav2 config notes that the inherited H1 Nav2 file nested `ros__parameters` under costmap layers,
which silently dropped layer params (`g1_nav2.yaml:15-17`). The fork-era file indeed had that bug
(`c00feca` nav2_config.yaml L130, L138). camera_ready's rewritten `nav2_config.yaml` does **not** have
it any more [V].

### 3.2 Side by side (HEAD states)  [V]

| Aspect | G1 `g1_navigation` | camera_ready `h12_slam` |
|---|---|---|
| Odometry | FAST_LIO (`g1_fastlio.yaml` real, untested; `g1_fastlio_sim.yaml`), or the sim's GT (`odom_source:=sim_gt`, sim only) | FAST-LIO `mid360.yaml`, **one config for real and both sims** (voxels 0.4/0.4, scan_line 6, blind 0.5). Extrinsic computed from URDF `livox_link`/`livox_imu_link` at launch |
| odom -> camera_init | static, computed from URDF + `pelvis_height` (0.744) assuming the robot stands level at start (`frames.fastlio_transforms`) | static, computed online by `level_odom` (IMU gravity average + lidar floor-plane fit) |
| body -> pelvis | `body_to_pelvis` node, **through the measured waist** via TF | static from URDF with `torso_joint` at zero |
| Base frame | `base_footprint` under `pelvis -> pelvis_level -> base_footprint` (`level_frame`) | `base_footprint` directly under `odom` (`level_odom`, per FAST-LIO odometry) |
| map -> odom | identity by design: slam_toolbox `use_scan_matching: false`, `do_loop_closing: false` | slam_toolbox stock: `use_scan_matching: true`, `do_loop_closing: true` (map -> odom corrects drift) |
| Scan source | raw `/livox/lidar` -> `livox_to_pointcloud` (CustomMsg -> PointCloud2, gyro deskew in sim only, TF to `pelvis_level`) -> `pointcloud_to_laserscan` with **no TF** -> `/scan` (band -0.59..0.45 m, 0.5-8 m) | FAST-LIO `/cloud_registered_body` -> `self_filter` (URDF collision boxes) -> `/cloud_body_filtered` -> `pointcloud_to_laserscan` with `target_frame: base_footprint` -> `/converted_scan` (0.25-1.7 m, 0.3-3 m) |
| 3D map | none | `octomap_server` -> `/projected_map` (static layer of both costmaps) |
| Nav2 | own node list + lifecycle manager (no `nav2_bringup`): Smac 2D + **RPP**; costmaps on `/scan` + `/map`; footprint `[[0.44,±0.24],[-0.20,±0.24]]`; smoother caps vx -0.3..0.5, **vy 0**, wz ±0.6 | `nav2_bringup/navigation_launch.py` + custom BT `navigate_to_pose_humanoid.xml`: Smac 2D + **MPPI**; global costmap = `/projected_map`, local = projected_map + `/converted_scan`; footprint `[[0.20,±0.275],[-0.20,±0.275]]`; caps vx -0.2..0.3, **vy ±0.2**, wz ±0.5 |
| Real-robot MID-360 | not done (`TODO(lidar)`, real bringup refuses `use_nav`) | done: `h12_real_drivers` routes `/livox/imu` through `livox_imu_upright` (rotates IMU by MID360_config roll 180), guarded by `test_real_lidar_config.py` |
| Tests | 7 test files (frames, nodes, FAST_LIO on synthetic scans, Nav2 smoke, velocity smoother, launch, frontier) + closed-loop `tests/test_sim_nav.sh` / `test_sim_explore.sh` | `test_level_odom`, `test_mid360_frames`, `test_robot_geometry` + linters |
| Extras | `frontier_explore` node (§4) | `h12_real_slam.launch.py`, `record_slam_bag`, `slam.rviz`, `nav2_config_mac.yaml` |

Frame-convention convergence [V]: both ended up with level `odom`, `base_footprint` as the
slam_toolbox/Nav2 base, and floor at z = 0. They differ in where `base_footprint` hangs (child of
`odom` vs. child of `pelvis_level`) and in whether map -> odom is the identity.

### 3.3 Robot-specific vs reusable  [I]

- **h12_slam is H1-2-bound by constants**: `LIDAR_FRAME='livox_link'`, `IMU_FRAME='livox_imu_link'`,
  and the URDF default `.../CL_Assets/ros_assets/h1_2_magpie_ros.urdf` (`h12_navigation.launch.py:15-17,80-84`).
  The G1 URDF has `mid360_link` and no separate IMU link; G1 takes the lidar-to-IMU extrinsic from
  the FAST_LIO yaml. The Nav2 params are tuned for H1-2 (footprint, vy, MPPI).
- **Robot-agnostic pieces in h12_slam**: `level_odom` (IMU + floor fit; no standing-height
  assumption, so it works on a gantry), `self_filter` + `robot_geometry` (any URDF on
  `/robot_description`), `record_slam_bag`, and `livox_imu_upright` (in h12_bringup).
- **Robot-agnostic pieces in g1_navigation**: `livox_to_pointcloud` (deskew + TF; frame names are
  parameters), `level_frame`, `body_to_pelvis` (frames from params plus the FAST_LIO yaml),
  `frontier_explore`. `frames.py` hard-codes `BASE_LINK='pelvis'`, `LIDAR_LINK='mid360_link'`,
  `LIVOX_FRAME='livox_frame'`.
- **Cross-pollination worth flagging to owners [I]:**
  - G1 found that `pointcloud_to_laserscan` with a `target_frame` (its own TF message filter) hung
    for good in 7 of 27 sim runs (`g1_navigation.launch.py:158-162`, G1 CLAUDE.md pitfalls).
    camera_ready's `h12_navigation.launch.py:141` uses `target_frame: base_footprint`, so the same
    latent hazard may exist there. Unverified on H1-2.
  - G1's `body_to_pelvis` exists because a fixed body -> pelvis "counts the waist twice" when the
    waist moves. camera_ready's static `body_to_pelvis_broadcaster` assumes `torso_joint` = 0. If the
    H1-2 torso joint moves during navigation, it has the same error. Not checked whether it does.
  - camera_ready's `livox_imu_upright` + shared `MID360_config.json` + `test_real_lidar_config.py`
    already solve the G1's real-lidar `TODO(lidar)` problem (driver rotates points, not IMU). The G1
    MID-360 is also mounted inverted (`frames.py:3-4`: `mid360_joint rpy pi 0.0511 0`). G1's README
    proposes flipping `extrinsic_R`; G1 hangs `camera_init` off a URDF-derived `odom`, so that may
    work for G1 even though it failed for the H1-2 setup (memory note). The simplest G1 path is
    either a zero extrinsic in a G1-specific `MID360_config.json` (identical to the sim) or reusing
    `livox_imu_upright`.
  - The G1 `MID360_config.json` host IP `192.168.123.164` is the H1-2/robot-PC address. The G1
    stack runs on a laptop, so it needs a G1-specific file once the lidar is connected.

**Recommendation [I]:** keep `g1_navigation` as a separate G1 package (frames, URDF, waist-aware
TF, HOMIE speed limits vy=0, RPP tuning, a sim-validated pipeline). Do not try to merge it into
`h12_slam` now: the pipelines are architecturally different (raw-lidar scan + identity map->odom vs
FAST-LIO cloud + octomap + scan-matching). Longer term, factor the agnostic nodes
(`livox_imu_upright`, `self_filter`, `level_odom`, `livox_to_pointcloud`, frontier explorer) into a
shared package. Both stacks use the same topic names (`/livox/lidar`, `/livox/imu`, `/map`,
`/navigate_to_pose`, `/cmd_vel`), which is fine because only one robot runs at a time. They differ
in scan topics (`/scan` vs `/converted_scan`).

### 3.4 How h12_slam diverged on camera_ready (tree comparison c00feca -> HEAD)  [V]

New since the fork-era `h1_bringup` nav:
- package split
- `level_odom`, `self_filter`, `robot_geometry`, `record_slam_bag`
- octomap + `/projected_map` costmaps
- MPPI controller
- humanoid BT XML via `RewrittenYaml`
- URDF-derived FAST-LIO extrinsic and body -> pelvis
- unified `mid360.yaml` (real-verified values)
- stock `slam_toolbox_h1.yaml`
- `h12_real_slam.launch.py`
- tests
- real-robot IMU handling (`livox_imu_upright` in h12_bringup)

None of this is in G1.

---

## 4. Frontier exploration  [V]

- **Same action type and goal fields.** `SkillFrontierExplore` goal: `min_frontier_cells`,
  `blacklist_radius`, `min_goal_distance`, `goal_timeout`, `timeout`. Result: `success`,
  `message`, `goals_reached`. Feedback: `phase`, `progress`, `frontiers_remaining`. The defaults are
  the same (6, 0.6, 0.7, 120). Same server name `/skill/frontier_explore`, and both use Nav2
  `/navigate_to_pose` and `/map`. The only `.action` difference is the `goals_reached` comment.
- **Common ancestor:** the fork-era mixin `h12_skills/skills/frontier_explore.py` (174 lines).
- **camera_ready (427 lines, still a `SkillsNode` mixin):**
  - registered in `base.py:230` `SKILL_ACTIONS`, inherited by `node.py:49-51`
  - requires the whole H1 skills node (magpie grippers, perception, etc.)
  - added "abyss safety": goals backed off into known-free space with graduated standoffs
    `_STANDOFFS=(0.35,0.25,0.18,0.12)` and a wall margin, against RoboCasa doorways opening onto a
    void
  - added a min-advance gate vs the goal tolerance, progress-based blacklisting (`_PROGRESS_CELLS`),
    stuck-round limits, and RViz `MarkerArray` on `/frontier_explore/frontiers`
  - `_BASE_FRAME='pelvis'`
- **G1 (537 lines, standalone `Node` + `MultiThreadedExecutor` in g1_navigation, launch arg
  `frontier_explore:=true`):**
  - frontiers must be walkable and reachable through the **global costmap**
    (`/global_costmap/costmap`, inscribed threshold)
  - the goal is the cluster cell nearest the centroid whose **footprint** fits (`G1_FOOTPRINT`
    default, overridden from the nav2 yaml by the launch file)
  - in-flight goals are cancelled when the frontier is seen on the way (not blacklisted) or
    blacklisted when the footprint becomes lethal
  - deadlines on the node clock (sim time)
  - `goals_reached` also counts "seen on the way"
  - configurable `map_frame`, `base_frame` (default `pelvis`), topics, and footprint
- **Divergence:** two independent re-designs of the same policy with complementary safety features.
  G1's lacks the abyss standoff, and camera_ready's lacks costmap reachability and footprint checks.
  G1's `goals_reached` semantics differ from camera_ready's (Nav2 successes only).
- **Port implications [I]:**
  - Never run both on one robot; they would both serve `/skill/frontier_explore`. On G1 there is no
    `SkillsNode`, so there is no clash.
  - G1's standalone, parameterised node is the better base for a robot-agnostic explorer. camera_ready's
    abyss standoff should be ported into it before an H1-2 switch.
  - Behavioural differences mean H1 callers may see different `goals_reached` counts.
  - Wire compatibility is guaranteed once both use `custom_ros_messages` (comments are not part of
    the type).

---

## 5. `g1_spatial_memory` vs `c0af423` `h12_skills` spatial memory

### 5.1 Diff  [V]

- `skills/spatial_memory.py` (739 -> 740 lines): **3 hunks only**. Docstring "H12 spatial-memory
  skill" -> "G1 spatial-memory node"; `from custom_ros_messages.action` -> `from g1_msgs.action`
  (L40); node name `h12_spatial_memory` -> `g1_spatial_memory` (L508).
- `spatial_memory_replay.py`: docstring only. `test/test_spatial_memory_backend.py`: docstring +
  import path. `launch/spatial_memory.launch.py`: docstring, `package=`, `name=`.
  `setup.py`: package name, dropped `skills = h12_skills.node:main`.
  `package.xml`: name; deps trimmed to `g1_msgs` (dropped `custom_ros_messages`, `magpie_msgs`,
  `std_srvs`, `nav2_msgs`, `sensor_msgs_py`, `visualization_msgs`, `cv_bridge`, `python3-scipy`).
  `skills/__init__.py` is reduced to a docstring.
- Conclusion: **no functional change**. G1 kept the branch's node as-is, minus everything else from
  `h12_skills`.

### 5.2 Dependencies  [V]

- ROS: `SkillRetrieveMemory` action on `/skill/retrieve_memory`; subscribes
  `/realsense/head/color/image_raw/compressed` (default `DEFAULT_CAMERA_TOPIC`, L491); TF
  `odom -> pelvis` (`world_frame`, `robot_frame` params, L515-516).
- External code: `EmbodiedAgent` checkout at `EMBODIED_AGENT_ROOT` (default `/opt/EmbodiedAgent`,
  L511; replay L150; launch L18), loaded via `sys.path.insert` (L84-85).
- Data: `/data/spatial_memory` (L513; launch L19).
- Env: `GEMINI_API_KEY`/`GOOGLE_API_KEY` for rerank (L353); `HAMS_SIM_TIME` in
  `launch/spatial_memory.launch.py:55` (**actual env read**, inherited unchanged from c0af423).
- Docker (G1, inherited from the c0af423 branch): `RosDockerfile` final layer
  `pip install --no-deps faiss-cpu==1.14.3 sentencepiece==0.2.1` (G1 RosDockerfile:392-396; c0af423
  RosDockerfile:546-551); compose env `EMBODIED_AGENT_ROOT=/opt/EmbodiedAgent`, mounts
  `${EMBODIED_AGENT_ROOT:-../../EmbodiedAgent}:/opt/EmbodiedAgent:ro` and
  `../container_cache/g1_spatial_memory:/data/spatial_memory` (G1 compose L95-113);
  `docker/SpatialMemoryDockerfile` (image `hams_spatial_memory`) and
  `docker/scripts/spatial_memory_camera.sh` for the benchmark.
- **camera_ready has none of the docker pieces**: no faiss/sentencepiece, no EmbodiedAgent env or
  mount, no `/data/spatial_memory`, no SpatialMemoryDockerfile (grep).
- Benchmark code: c0af423 `benchmarks/spatial_memory/` -> G1 `spatial_memory/` (only import-path
  renames in `evaluate.py`). The capture scripts `h1_robocasa/spatial_memory_{benchmark,geometry,live,scan}.py`
  -> G1 `spatial_memory/`, where `live`/`scan` import the robot's robosuite module
  (`h1_2_robosuite` -> `g1_robosuite` plus a `sys.path` hack to `sim/`).

### 5.3 Robot-specific?  [V/I]

The node is **robot-agnostic**. It needs a head colour camera on
`/realsense/head/color/image_raw/compressed` and a TF chain `odom -> pelvis`. camera_ready's RoboCasa
bridge publishes exactly that topic (`h1_robocasa/mujoco_ros_bridge.py:9,377`) [V]. The H1-2 has
`pelvis` under `odom` via FAST-LIO, or `GOLEM_SIM_ODOM=1` [V]. Only the benchmark capture scripts are
(trivially) robot-specific.

### 5.4 Merge into camera_ready `h12_skills` or separate package?  [I]

- Mechanically, either is easy. The c0af423 changes to `h12_skills` are additive (7 files), and
  camera_ready changed `h12_skills/package.xml` by only one line since c00feca, so the edits to
  `package.xml`/`setup.py` (launch data_files, 2 console_scripts, launch/launch_ros/python3-pil deps)
  are trivial.
- **Recommend a separate, robot-agnostic package** (essentially `g1_spatial_memory` renamed, e.g.
  `spatial_memory` or `golem_spatial_memory`, importing `custom_ros_messages`):
  - Importing `h12_skills.skills.spatial_memory` executes `h12_skills/skills/__init__.py`, which
    imports every H1 skill mixin (grasp -> `custom_ros_messages`, `base`, `model_logging`,
    `perception_utils`, ...). That couples a robot-agnostic node to the H1 manipulation stack.
  - camera_ready's `h12_skills` has grown (grasp_synthesis, run telemetry) and has its own owner.
  - G1 must run it without `h12_skills`.
- Also needed:
  - add `SkillRetrieveMemory` to custom_ros_messages (§1.4)
  - rename `HAMS_SIM_TIME` -> `GOLEM_SIM_TIME`
  - add the docker pieces at the end of camera_ready's RosDockerfile and compose (docker scope, owned
    separately)
  - decide where the benchmark lives (`benchmarks/spatial_memory` vs `spatial_memory/`)

---

## 6. Hard-coded G1_GOLEM naming, paths and env vars to adapt  [V]

In-scope packages (code and config; READMEs listed separately):

| File:line | Item | Kind |
|---|---|---|
| `g1_bringup/launch/g1_core.launch.py:30-31, 284` | `HAMS_SIM_ODOM` | doc / arg description |
| `g1_bringup/launch/g1_sim_bringup.launch.py:7, 20, 53` | `HAMS_SIM_ODOM` | doc / arg description |
| `g1_bringup/launch/g1_bringup.launch.py:7` | `G1_DDS_INTERFACE`, `ROS_DOMAIN_ID=0` in docker/.env | doc (real-robot flow) |
| `g1_bringup/launch/g1_core.launch.py:68-73` | imports `g1_description.paths`, `g1_control.utility.config_pairing` | package names |
| `g1_bringup/package.xml:20` (+14-19) | `g1_msgs` (+ `g1_*`) exec_depends | dependency |
| `g1_bringup/config/MID360_config.json` | H1-2 network IPs (host .164), roll 180; unused | stale config |
| implicit via `g1_core` | `g1_locomotion/locomotion_node.py:16` `/home/code/g1_locomotion` policy dir (not overridden by bringup) | container mount |
| `g1_navigation/launch/g1_navigation.launch.py:17, 19, 240` | `HAMS_SIM_ODOM` | doc / arg description |
| `g1_navigation/launch/g1_navigation.launch.py:53` | `g1_description.paths` import | package name |
| `g1_navigation/g1_navigation/frontier_explore.py:5, 8, 68` | `g1_msgs` (doc + import) | import |
| `g1_navigation/g1_navigation/frontier_explore.py:100, 313-314` | `G1_FOOTPRINT` default, `base_frame` `pelvis` | robot default (parameterised) |
| `g1_navigation/g1_navigation/frames.py:BASE_LINK/LIDAR_LINK/LIVOX_FRAME` | `pelvis`, `mid360_link`, `livox_frame` | robot frames |
| `g1_navigation/test/test_frontier_explore.py:14` | `g1_msgs` import | import |
| `g1_navigation/package.xml:34` | `g1_msgs` | dependency |
| `g1_spatial_memory/g1_spatial_memory/skills/spatial_memory.py:4` | "HAMS owns..." | docstring |
| `.../spatial_memory.py:40` | `from g1_msgs.action import SkillRetrieveMemory` | import |
| `.../spatial_memory.py:508` | node name `g1_spatial_memory` | name |
| `.../spatial_memory.py:511, 513` | `/opt/EmbodiedAgent` (via `EMBODIED_AGENT_ROOT`), `/data/spatial_memory` | container paths |
| `g1_spatial_memory/g1_spatial_memory/spatial_memory_replay.py:150` | `/opt/EmbodiedAgent` | container path |
| `g1_spatial_memory/launch/spatial_memory.launch.py:18-19` | `/opt/EmbodiedAgent`, `/data/spatial_memory` | container paths |
| `g1_spatial_memory/launch/spatial_memory.launch.py:55` | **`HAMS_SIM_TIME` env read** | env var (rejected by GOLEM docker_common.sh) |
| `g1_spatial_memory/launch/spatial_memory.launch.py:32, 34` | package/name `g1_spatial_memory` | name |
| `g1_spatial_memory/package.xml:11` | `g1_msgs` | dependency |
| `g1_msgs/CMakeLists.txt:2`, `g1_msgs/package.xml:3` | `g1_msgs` | package name |

READMEs: `g1_navigation/README.md` (9 lines with HAMS_/test/docker refs) and
`g1_spatial_memory/README.md` (7 lines: `HAMS_SIM_ODOM`, "relative to HAMS", docker scripts).

Out of scope but coupled (for the other auditors):
- `g1_msgs` imports in `g1_control` (4 files), `g1_dex3/hand_node.py:16`, `model_server`
  (4 servers), `tests/test_sim_bringup.py`, `tests/test_sim_explore.py`
- G1 test scripts mount `docker/scripts` at `/home/code/h12_sim_scripts` and set `HAMS_SIM_ODOM`,
  `HAMS_SPAWN_BACKOFF`, `HAMS_LIDAR*`, `HAMS_CAMERA*`
- G1 compose mounts `container_cache/g1_spatial_memory`
- G1 `sim/g1_mujoco.py:198` reads `HAMS_SIM_ODOM`

All `HAMS_*` must become `GOLEM_*`: camera_ready `docker/scripts/docker_common.sh:25` refuses to run
with any `HAMS_*` set.

---

## 7. Recommendations (summary)  [I]

1. **Messages:** one `custom_ros_messages`. Upstream: merge `spatial-memory-retrieval-action`
   (SkillRetrieveMemory) and add `SetHandPose.srv`, then bump GOLEM's pin. Rename `g1_msgs` ->
   `custom_ros_messages` in all G1 code. Make G1's `frame_task_server` reject `relative:=true`
   (and decide on `slow_mode`). Keep every interface G1 deleted.
2. **Bringup:** port `g1_bringup` as its own package (launch files + tests unchanged apart from
   renames). Drop its dead `MID360_config.json` and nav exec_depends. Declare and forward
   `frontier_explore`. Rename `HAMS_SIM_ODOM` -> `GOLEM_SIM_ODOM` in docs. Solve the
   `/home/code/g1_locomotion` mount or install path. Update GOLEM CLAUDE.md §4/§6.
3. **Navigation:** port `g1_navigation` as its own package. Do not merge it with `h12_slam` now.
   Flag the p2l TF-filter hang and the waist-in-body->pelvis issue to the h12_slam owner. For the
   G1 real lidar, reuse camera_ready's `livox_imu_upright` approach or a zero-extrinsic G1
   `MID360_config.json` with the laptop's IP.
4. **Frontier exploration:** port G1's standalone node as the G1 server. Plan to make it the shared
   robot-agnostic explorer after porting camera_ready's abyss standoff into it. Never run two
   `/skill/frontier_explore` servers.
5. **Spatial memory:** port as a separate robot-agnostic package, not into `h12_skills`. Add the
   faiss/sentencepiece layer, the EmbodiedAgent mount/env and the `/data/spatial_memory` mount to
   camera_ready docker (owners: docker). Rename `HAMS_SIM_TIME`.
