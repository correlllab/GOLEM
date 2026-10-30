# Audit 1: G1 robot-control packages (G1_GOLEM → GOLEM camera_ready port)

Scope: `G1_GOLEM/core_ws/src/g1/{g1_control, g1_safety_layer, g1_locomotion, g1_dex3, g1_description}` at
G1_GOLEM `main` HEAD `df64d43`, compared against GOLEM `camera_ready` HEAD `42d6559` and its submodule
checkouts `core_ws/src/h12_ros2_controller` (`b53a6b2`) and `core_ws/src/h12_safety_layer` (`ad06903`).
Both submodule checkouts were confirmed clean at those pins.

Method: HEAD-state comparison. Trees were copied to the scratchpad and compared with `diff -r` and
per-file `diff`, with the module directory renamed to a common name so files align. Git history was used
only to pin the imported upstream versions.

Legend: **[V]** verified by reading code or running a read-only command. **[I]** inferred, not verified.

Scratch trees used (they can be deleted):
`/tmp/claude-1000/-home-max-GOLEM/f8c735f2-cb0b-46cc-a631-673fdc69ab41/scratchpad/trees/`
(`c87` = controller import base, `h12c` = GOLEM controller pin, `g1c` = G1 g1_control; `s87`, `h12s` and `g1s` are the same three for the safety layer).

---

## 0. Executive summary

- **What exists [V]:** five ament_python packages, about 12.7k lines of Python source plus 5.9k lines of tests:
  - `g1_control`: upper-body Pink IK controller, waist held, 14 arm joints solved.
  - `g1_safety_layer`: DDS relay with arm_sdk, split and full modes.
  - `g1_locomotion`: HOMIE ONNX walking policy on the legs.
  - `g1_dex3`: Dex3-1 hand driver.
  - `g1_description`: URDFs from the `leprechaun_g1` submodule, a generated joint table and the DDS slot layout.
- **Lineage [V]:**
  - `g1_control` was imported from h12_ros2_controller `87bca34` (2026-07-15). The import tree is identical to upstream except that `figures/` and `data/` were dropped.
  - `g1_safety_layer` was imported from h12_safety_layer `8ad5988` (2026-07-14). Its import tree hash `7647136` is identical to upstream. In G1_GOLEM's rewritten history the commit is `1ffa65d`.
  - Both were exactly what GOLEM pinned at the fork point `c00feca`.
  - Since then, upstream h12_ros2_controller has 110 commits. It added about 26.7k LOC, mostly H1 counter-balance/ZMP research.
  - Upstream h12_safety_layer gained only small changes: `/safety/heartbeat`, a configurable upper-stale watchdog, a logger interface, and mode_machine/mode_pr echoed in the split merge.
- **Depth of the rewrite [V]:** both are G1-only forks, not supersets.
  - Every shared file differs, including module renames.
  - H1-only features were deleted: Inspire hands, gravity-compensation server, benchmarks/plots, `mode: debug|sport`, the external serial estop and the collision-model requirement.
  - The robot identity is baked in through module-level imports of `g1_description`: 16 of the g1_control modules import the joint tables.
  - The G1 core changes are robot-agnostic hardening (optional collision model, measured-pose seeding, follow gate, startup timeouts, single-writer and engage gates, BOOTTIME watchdogs). They could later flow back to the H1 packages.
- **Recommendation:** keep them as **separate packages** beside the H1 ones; do not merge them into the H1 packages now. Reasons:
  - The H1 packages are owned submodules with heavy active research.
  - The H1 runtime depends on features G1 removed (`/safety/heartbeat`, the external estop).
  - The G1 safety semantics (arm_sdk weight hand-over, follow gate, config pairing) do not exist on H1.
- **Build-level collisions:** none in colcon. All package names and top-level Python modules differ, and console scripts install per package under `lib/<pkg>/`.
- **Runtime collisions on a shared DDS domain are real and silent.** Both stacks use:
  - the same node names (`safety_node`, `frame_task_server`, `joint_state_publisher`, `robot_state_publisher`);
  - the same action names with **different types** (`/frame_task`, `/named_config`, `/dual_arm`: `custom_ros_messages` versus `g1_msgs`, and the FrameTask/NamedConfig definitions have diverged);
  - the **same Unitree DDS topics with the identical IDL type** (`rt/lowstate`, `rt/lowcmd`, `rt/safety/lowcmd_{upper,lower}_in`, all `unitree_hg::LowCmd_/LowState_`).

  The rule must be "one robot per DDS domain/network". The G1 safety layer's engage gate (mode_machine 5, pose within 0.15 rad) is the only cross-robot guard. The H1 safety layer has none.
- **Port prerequisites:**
  - onnxruntime in GOLEM's ROS image (absent today; only the Isaac image has it).
  - The `leprechaun_g1` submodule at `<repo>/third_party/leprechaun_g1`, which the symlink and `parents[3]` assume. Otherwise `g1_description/setup.py` raises and aborts the whole colcon build.
  - `g1_msgs` (or a reconciliation with `custom_ros_messages`).
  - A mount for `policy/` at `/home/code/g1_locomotion`.
  - Renaming or keeping `G1_DDS_INTERFACE` / `G1_SAFETY_LOG_DIR`.
  - None of the five packages reads any `HAMS_*` variable.

---

## 1. Per-package inventory

LOC is non-blank and blank lines of tracked `*.py` files at HEAD **[V]**:

| Package | src py | test py | yaml | Notes |
|---|---|---|---|---|
| g1_control | 8,160 | 1,523 | 368 | + `uv.lock`, a 505-line `requirements.txt` and `environment.yml` (conda env `control_env`) inherited from H1 |
| g1_safety_layer | 2,317 | 2,231 | 387 | + `docs/architecture.md`, `docs/config.md` |
| g1_locomotion | 1,264 (incl. policy/ 3 py) | 1,514 | 172 | `policy/deploy.onnx` is 1.8 MB, committed as a plain blob (not LFS) |
| g1_dex3 | 592 | 555 | 0 | |
| g1_description | 413 | 62 | 0 | `leprechaun_g1` symlink (mode 120000) → `../../../../third_party/leprechaun_g1` |

### 1.1 g1_control [V]

- **Purpose:**
  - Pinocchio + Pink IK on the 14 arm joints, with palm frames `left/right_hand_palm_link` as end effectors (`utility/joint_definition.py:44-49`).
  - The waist (3 joints, slots 12-14) is held at `controller.waist_target` and never IK-driven (`core/controller/upper_controller.py:_apply_joint_position`, `_move_waist_to_target`).
  - Gravity-compensation feedforward.
  - Publishes **only** to `rt/safety/lowcmd_upper_in` (`core/channel_interface.py:20-21`).
- **Console scripts** (`setup.py`): `joint_state_publisher`, `dual_arm_server`, `dual_arm_client`, `frame_task_server`, `frame_task_client`. These are the same executable names as h12_ros2_controller, minus H1's `hand_controller_node`, `gravity_comp_server` and `hand_cmd_gui`.
- **Launch:** `launch/{desktop,full,robot,robot_safety,robot_tf}_launch.py`, the same filenames as H1. `robot_safety_launch.py` uses `utility/config_pairing.py`.
- **Configs:** `config/g1.yaml` (real, safety_out `rt/arm_sdk`, conservative v_lim 0.15 / w_lim 0.5 / dq_lim 0.3), `g1_split.yaml` (stiff waist kp 200 for walking, safety_out `rt/lowcmd`), `sim.yaml`, `sim_split.yaml` (domain 1, relaxed estop ratios 2.0).
- **ROS surface:**
  - Node names: `frame_task_server`, `dual_arm_server`, `joint_state_publisher`, `frame_task_client`, `dual_arm_client`.
  - Actions: `frame_task` (`g1_msgs/FrameTask`), `named_config` (`g1_msgs/NamedConfig`), `dual_arm` (`g1_msgs/DualArm`). All are relative names, resolving to `/frame_task` and so on.
  - Topics published: `joint_states`, `frame_names` (`g1_msgs/StringArray`), `frame_targets`, `frame_poses` (PoseArray), `left/right_ee_pose`, `left/right_ee_target` (PoseStamped).
- **Unitree DDS surface:**
  - Subscribes to `rt/lowstate` with a callback `ChannelSubscriber` (queue 10).
  - Publishes `LowCmd_` on `rt/safety/lowcmd_upper_in` at 500 Hz.
  - **Reads**, never writes, the safety layer's output (`rt/arm_sdk` or `rt/lowcmd`) via `SafetyOutputSubscriber` (`core/channel_interface.py:81-100`). The follow gate uses it.
- **Key G1 mechanisms:**
  - `_wait_for_state` (`upper_controller.py:148`): waits up to 5 s for a fresh, finite `rt/lowstate` before enabling anything.
  - Seeds every slot from the measured pose with gravity torque.
  - **Follow gate** (`core/follow_gate.py`): commands stay within `follow_tolerance` (0.05 rad) of the measured joints until a safety-output sample carries them with kp > 0 (and weight ≥ 0.999 on `rt/arm_sdk`).
  - Startup timeouts `startup_timeout` and `startup_move_timeout`.
  - Estop means going silent (`channel_interface.py:227`), so the safety layer hands back.
  - `plan=true` aborts with `PLANNER_UNAVAILABLE` (`upper_controller.py:27`), because there is no G1 self-collision model.
  - `controller_config.py:178-181` rejects a `mode:` key; `:212-223` enforces `topics.low_cmd` starting with `rt/safety/` and `topics.safety_out` being one of `rt/arm_sdk` or `rt/lowcmd`.
- **Config pairing** (`utility/config_pairing.py`): the pairs are arm_sdk↔g1, split↔g1_split, sim_arm_sdk↔sim and sim_split↔sim_split. Sim and real configs are told apart **by name prefix `sim`**. With locomotion enabled, the split pair is required. `g1_bringup/g1_core.launch.py` imports it, which ties g1_bringup to g1_control.
- **Dependencies:**
  - package.xml: `g1_msgs`, `g1_description`, exec `g1_safety_layer`, `robot_state_publisher`, `action_msgs`, `sensor_msgs`, `geometry_msgs`.
  - Python, not declared in package.xml: `pinocchio` (`pin`), `pink` (`pin-pink`), `qpsolvers`, `meshcat`, `meshcat_shapes`, `pyquaternion`, `tqdm`, `ompl` (only if the planner is enabled), `unitree_sdk2py`, `cyclonedds`, `yaml`, `numpy`. GOLEM's ROS image already ships pin/pink [V via RosDockerfile grep]. Whether ompl and meshcat are in GOLEM's image is [I] likely, since H1 uses the same stack.
- **Tests:**
  - `test/test_g1_controller.py` (473 lines), `test_controller_loop_dds.py` (737, in-process DDS on domain 97 with a fake G1), `test_controller_config.py`, `test_reduced_joint_planner.py`, plus ament flake8/pep257/copyright.
  - Run with `PYTHONPATH=.:../g1_description python3 -m pytest test/...` (README:215). This relies on `g1_description` being a **sibling directory**.
  - URDF-dependent tests skip when leprechaun_g1 is absent (`test_g1_controller.py:43`, `test_controller_loop_dds.py:60`).

### 1.2 g1_safety_layer [V]

- **Purpose:** the sole writer of `rt/arm_sdk` and `rt/lowcmd`. It relays, with clipping, a latched estop and three modes (`core/safety_layer.py:42`):
  - `arm_sdk_mode` (default, `config.py:221`): waist+arms on `rt/arm_sdk`, weight slot 29 ramped 0→1 over `engage_seconds` and back to 0 over `release_seconds` on estop and on SIGINT/SIGTERM/SIGHUP (`core/arm_sdk.py`, `core/exit_signals.py`). Legs are zeroed in the message because they are Unitree's.
  - `split_mode`: `rt/lowcmd` with legs from `rt/safety/lowcmd_lower_in` and slots 12-28 from `rt/safety/lowcmd_upper_in`.
  - `full_body_mode`: `rt/lowcmd` from `rt/safety/lowcmd_in`.
  - On `rt/lowcmd`, the header is stamped with `mode_pr=MODE_PR(0)` and `mode_machine` echoed from `rt/lowstate` (`safety_layer.py:516-522`).
- **Gates and watchdogs:**
  - Engage gate (`safety_checks.py:105-139`): fresh `rt/lowstate` (0.1 s), `mode_machine == engage.mode_machine` (default 5), and every slot of the command within 0.15 rad of the measured q.
  - Single-writer check: a second DDS publication handle on an input latches the estop (`safety_layer.py:304-321`). It relies on cyclonedds `sample_info.publication_handle`.
  - `state_stale_estop_seconds` (0.1 s), `lower_stale_estop_seconds` (0.2 s) and the upper-stale watchdog (2 s, hard-coded constant).
  - `arm_sdk.waist_kp_min` (40).
  - `estop_response` is `zero` or `damp`.
  - All watchdogs run on CLOCK_BOOTTIME (`core/clock.py`).
- **Config rules:**
  - Every config must declare `robot: real|sim`. A sim config refuses DDS domain 0 (`config.py:157-162, 269-275`).
  - A top-level `estop:` section, which is the H1 external serial estop, is **rejected** (`config.py:216-219`).
- **Configs:** `arm_sdk.yaml` (real default), `split.yaml`, `full.yaml`, `sim_arm_sdk.yaml`, `sim_split.yaml`.
- **Console scripts:** `safety_node`. There is also a non-ROS `g1_safety_layer/script/safety_layer_main.py`.
- **ROS surface:**
  - Node `safety_node`. **No ROS publishers or services** (no heartbeat).
  - `rclpy.init(domain_id=...)` is forced to the Unitree domain (`ros2/safety_node.py:94`).
  - Default `--config arm_sdk.yaml` (`:78`).
- **Environment:** `G1_DDS_INTERFACE` (DDS NIC) and `G1_SAFETY_LOG_DIR` (per-run log dir; `config.py:27`). The default log dir is `/tmp/g1_safety_layer`.
- **Dependencies:** package.xml declares only `g1_description` (no `rclpy` exec_depend). Python: `unitree_sdk2py`, `cyclonedds`, `numpy`, `yaml`.
- **Tests:**
  - `test_g1_safety.py` (546), `test_arm_sdk_mode_dds.py`, `test_lowcmd_modes_dds.py`, `test_hardening_dds.py` (656), `test_node_exit.py` (383; runs the node as a subprocess and signals it).
  - DDS tests use domain 97. `conftest.py` monkey-patches `IDL.serialize` with a lock to work around a cyclonedds-python 0.10 cross-thread serialization race (about 3% corrupted samples in their probe). This only matters in tests, where many threads write.
  - Run with `PYTHONPATH=.:../g1_description python3 -m pytest ...` (README:185).

### 1.3 g1_locomotion [V]

- **Purpose:**
  - `locomotion_node` runs HOMIE's ONNX policy at 50 Hz on `rt/lowstate` tick time.
  - It publishes legs 0-11 to `rt/safety/lowcmd_lower_in` (split mode).
  - Inputs: `/cmd_vel` (Twist) and `/locomotion/height` (Float32, 0.24 to 0.74 m).
  - It forces `use_sim_time` false (`locomotion_node.py:35`).
- **Console script:** `locomotion_node`. Node name `locomotion_node`. Parameter `policy_dir`, default **`/home/code/g1_locomotion`** (`locomotion_node.py:16`). Option `--config g1.yaml|sim.yaml`.
- **DDS:**
  - Its own cyclonedds `DomainParticipant` with keep-last-1 best-effort `DataReader`s on `rt/lowstate`, and on `rt/safety/lowcmd_upper_in`, which it only reads to know when the arm controller has started (`driver.py:26, 70-77`).
  - It publishes with a `unitree_sdk2py` `ChannelPublisher` (`driver.py:68`).
  - `config.py:63` enforces an `rt/safety/` prefix on `low_cmd`.
- **Faults:** stale, non-finite or fallen state, a tick going backwards, latency above `faults.max_latency` (25 ms) or an off-rate tick latch the node silent. The safety layer's lower watchdog then estops. There is **no `/safety/heartbeat` dependency**, unlike H1's lower-body nodes.
- **Policy dir** `policy/`:
  - Contents: `deploy.onnx` (1.8 MB, `pytorch 2.4.1` export), `g1.yaml` (joints, gains, scales; `xml_path: ../third_party/leprechaun_g1/...xml` at `policy/g1.yaml:6`, resolved relative to the policy dir and used only by `mujoco_deploy_g1.py`), `homie_policy.py` (`HomiePolicy`, single-threaded `policy_session()`), `mujoco_deploy_g1.py` and `test_homie_policy.py`.
  - The node loads `homie_policy.py` by file path with `importlib` (`locomotion.py:55-72`). Nothing is copied or installed: `setup.py` does not install `policy/`.
  - Both containers mount `policy/` read-only at `/home/code/g1_locomotion` (G1_GOLEM `docker/docker-compose.yml:55, 125`). The G1 **sim's stand-in** (`sim/stand_in.py`) shares this module and model. A port must keep both consumers pointing at one copy.
- **Dependencies:** `onnxruntime==1.23.2` (G1 `RosDockerfile:415-430`, `RobocasaDockerfile:153-162`). **GOLEM's ROS image has no onnxruntime**; only `IsaacDockerfile:110` installs it (1.22.1) [V].
- **License [V]:**
  - The package `LICENSE` is MIT text with **no copyright line**. package.xml and setup.py declare MIT.
  - `policy/LICENSE` is **CC BY-NC 4.0**.
  - `policy/README.md` says OpenHomie's README claims CC BY-NC-SA 4.0 and that it should be treated as the stricter BY-NC-SA. Implications:
    - non-commercial use only;
    - attribution and indicating modifications are required (the README documents the changes and credits);
    - under SA, adapted versions must be shared under the same license;
    - it must not be relicensed as MIT.
  - **Port implication:** keep `policy/LICENSE` and `policy/README.md` (credits and change list) beside the files. Keep `homie_policy.py` inside `policy/` and do not merge it into MIT code. Note the license in GOLEM's top-level docs, because GOLEM is otherwise MIT/BSD. [I] Any commercial use of GOLEM would need this directory excluded.
- **Tests:** `test/test_locomotion.py`, `test_locomotion_node.py`, `test_locomotion_dds.py` (domain 97, fake G1). `conftest.py` disables gc to avoid crashes from unclosed channels. `policy/test_homie_policy.py`. Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:../g1_description:$PYTHONPATH python3 -m pytest test` (README:256).

### 1.4 g1_dex3 [V]

- **Purpose:** the Dex3-1 hand driver over Unitree DDS `rt/dex3/{left,right}/cmd` (`unitree_hg HandCmd_`) and `.../state` (`HandState_`) (`driver.py:64-66`). It **bypasses the safety layer by design**. The GOLEM precedent is H1's `rt/inspire/cmd` from `h12_ros2_controller`'s `HandPublisher` and the Magpie grippers over Dynamixel; neither goes through the safety layer.
- **Console script:** `dex3_hands`. Node `dex3_hands`. Launch `launch/hands.launch.py`.
- **ROS surface:**
  - Services per side: `/<side>/hand/set_pose` (`g1_msgs/SetHandPose`), `/<side>/hand/{open,close,release}` (`std_srvs/Trigger`).
  - Topics: `/<side>/hand/pressure` (Float32MultiArray), and finger joints on `/joint_states`.
  - These **do not collide** with H1's `/<side>/gripper/*`.
- **Parameters:** `sides`, `kp` 1.5, `kd` 0.1, `rate_hz` 100, `pose_duration`, `joint_state_hz`, `network_interface`, `domain_id` (no fallback; must agree with `ROS_DOMAIN_ID`), `on_shutdown` (`release` or `silent`), `max_speed`. Environment: `G1_DDS_INTERFACE`.
- **Behavior:** silent until a hand's first command, which starts from the measured pose.
- **Tests:** `test_dex3.py`, `test_driver_dds.py` (domain 97, fake hand), `test_hand_node.py` (needs rclpy).

### 1.5 g1_description [V]

- **Purpose:**
  - Installs the `leprechaun_g1` G1 URDFs: `g1_29dof_rev_1_0.urdf` (body) and `g1_29dof_with_hand_rev_1_0.urdf` (with Dex3-1), plus `*_ros.urdf` copies with `package://g1_description/assets/meshes/` URIs, all STL meshes and `LICENSE.unitree_ros`.
  - Python API:
    - `paths.urdf_path(variant, ros)`.
    - `joints.py`: `BODY` (29), `LEFT_HAND`/`RIGHT_HAND` (7 each); each entry has name, slot, lower/upper limits, velocity and effort.
    - `layout.py`: `NUM_BODY_MOTORS=29`, `LOWCMD_MOTOR_SLOTS=35`, `LEG_SLOTS 0-11`, `WAIST_SLOTS 12-14`, arm slots 15-21 and 22-28, `ARM_SDK_SLOTS 12-28`, `ARM_SDK_WEIGHT_SLOT=29`, `MODE_PR=0`/`MODE_AB=1` (`layout.py:13-30`).
  - `launch/display.launch.py` runs RSP, the joint_state_publisher_gui and RViz.
- **How it reaches leprechaun_g1:** the git symlink `g1_description/leprechaun_g1` points to `../../../../third_party/leprechaun_g1`. That only resolves when the package sits at `<repo>/core_ws/src/g1/g1_description`.
  - `setup.py:16-24` raises `RuntimeError` if the assets directory is missing. In a monorepo colcon build, that aborts the build (colcon stops scheduling on the first failure by default) [I on colcon default; consistent with GOLEM's `launch_ros.sh:103`, which builds everything without `--continue-on-error`].
  - At build time `setup.py` writes `_install_staging/` into the **source tree**. It is gitignored. H1's `h12_ros2_model` uses the same pattern.
  - In docker, G1 mounts the submodule at `/home/code/third_party/leprechaun_g1` so the symlink resolves in-container.
- **Generated joints.py:**
  - `tools/generate_joints.py` reads the with-hand URDF for limits and uses a hard-coded `BODY_ORDER` copied from Unitree's `G1JointIndex`. It cross-checks against the body MJCF actuators and has a `--check` mode for staleness.
  - It assumes `REPO = PKG.parents[3]` (`generate_joints.py:17-19`), which is the same layout depth assumption.
  - `test/test_joints.py:56-61` runs the check and skips if leprechaun_g1 is absent.
- **Compared with GOLEM's `h12_ros2_model`:**
  - h12_ros2_model is a separate submodule that installs H1 URDFs and meshes from top-level `CL_Assets` via `../../../CL_Assets`, and also raises if they are missing.
  - It has **no joint table or DDS layout**. H1's joint tables are duplicated as hard-coded lists in `h12_ros2_controller/utility/joint_definition.py` (`NUM_MOTOR = 27`, `:13`) and `h12_safety_layer/core/joint_limits.py` (`MOTOR_COUNT = 27`, `:3`).
  - g1_description is therefore a better-factored single source of truth. In the G1 stack it is a hard dependency of g1_control, g1_safety_layer, g1_locomotion and g1_dex3.
- **Binary/LFS:** g1_description itself contains no binaries [V]. The leprechaun_g1 submodule (pinned `3bee054`, `correlllab/leprechaun_g1`, branch main) is **not checked out locally**, so its size and whether it uses LFS for its STL/XML files are **unverified [I]**. G1_GOLEM's own `.gitattributes` uses LFS only for `demo/**/*.mp4`. `deploy.onnx` is a regular 1.8 MB blob.

---

## 2. Lineage vs GOLEM's H1 submodules

### 2.1 Which upstream versions were imported [V]

| G1 package | Import commit (G1_GOLEM) | Upstream pin in import message | Same tree in GOLEM's submodule repo | GOLEM pin at fork point c00feca | GOLEM camera_ready pin | Upstream commits since |
|---|---|---|---|---|---|---|
| g1_control | `1ad922f` (subtree split `320861f`) | `87bca34` (2026-07-15, "Sync dev to main") | `87bca34` exists locally; import tree = upstream minus `figures/`, `data/` (diff -rq shows only those) | `87bca34` | `b53a6b2` (2026-10-20) | 110 (`git rev-list --count 87bca34..b53a6b2`) |
| g1_safety_layer | `89cb029` (subtree split `1ffa65d`) | `1ffa65d` (SHA rewritten) | `8ad5988` (2026-07-14): **identical tree hash `7647136`** | `8ad5988` | `ad06903` (2026-10-20) | 14 (`git rev-list --count 8ad5988..ad06903`; mostly config tweaks) |

### 2.2 Upstream h12_ros2_controller since the import (`87bca34` → `b53a6b2`), code-level [V]

File-level three-way comparison (lines changed, base→H1 / base→G1):

| Area | Upstream H1 change | Relevance to G1 |
|---|---|---|
| `core/latest_reader.py` (new) + `channel_interface.py` (+163) | rt/lowstate read **keep-last-1, best-effort, on demand** (`LatestReader`), tick unwrapping, `TickLagTracker`, `staleness_ms`; `ArmSDKPublisher` deleted; `LowCmdPublisher` refuses non-`rt/safety/` topics | **Relevant.** g1_control still uses a callback subscriber with queue 10; g1_locomotion independently adopted the same keep-last-1 pattern. Worth back-porting to g1_control. The prefix guard already exists in G1 (`controller_config.py:212`) |
| `upper_controller.py` (+265), `frame_task_server.py` (+193) | `StaleStateError` / `check_state_fresh()` abort goals on stale state; `_command_seed` (seed commands from the measured state, `COMMAND_SEED_MAX_GAP` 0.1 s); steady-state limit-block reporting; `speed_scale`/`slow_mode`; **relative frame targets**; planned-path Meshcat visualization | Stale abort: relevant (G1 checks freshness only at startup) [I]. Seeding: G1 has its own version. slow_mode and relative are **interface features**; see 2.4 |
| `utility/controller_config.py` (+157) | `mode` must be `debug` (`sport` removed); `rt/safety/` prefix validator; many `zmp`/`counter_ddp`/`iteration5*` research sections | Only the prefix guard matters; G1 has it already. G1 removed `mode` entirely |
| `core/controller/counter_balance/*` (25 files), `zmp_*`, `momentum_controller.py`, `support_region.py`, `robot_dynamics.py` (+747), `reduced_joint_planner.py` (+149), 57 docs, 28 tests | Counter-balance, ZMP and DDP research on the H1-2 (22 `torso_joint`/27/magpie references in those files) | **Not relevant** to G1; H1-specific research |
| `named_config.py` (+27), `planner_process.py` (+48), `robot_model.py` (+148), `robot_visualizer.py` (+82) | Planner speed overrides, visualization, model additions for counter-balance | Mostly not relevant |
| `config/` | New `balance_safety_split`, `counter_safety_split`, `real_benchmark_upper`, `legacy/`; `sport.yaml` removed | Not relevant |

H1 package totals: 36,650 lines of Python, of which 26,686 are in files that G1 does not have. g1_control has 9,683.

### 2.3 Upstream h12_safety_layer since the import (`8ad5988` → `ad06903`) [V]

The complete code diff is small:
- `ros2/safety_node.py:20,54`: publishes **`/safety/heartbeat` (std_msgs/Bool, 10 Hz)**, True until the estop latches. **GOLEM's H1 lower-body stack depends on it**: `h12_lowerbody_rl/scripts/lowerbody_controller_node.py:80` and `lmg_deploy/marl_deploy_node.py:138` latch silent without it.
- `core/safety_layer.py`: `estop.upper_stale_seconds` is configurable (≤0 disables); the split merge now copies `mode_machine`/`mode_pr` from the last state (a merge bug fix); `estop_triggered`/`estop_reason` properties.
- `core/safety_logger.py` (new): routes status to the ROS logger.
- `package.xml`: exec_depend `rclpy` and `std_msgs`.
- Configs: `relax_safety_split.yaml` (+97), `sim_safety_split.yaml`, new `locomotion_safety_full.yaml`.

Relevance to G1:
- The mode_machine/mode_pr fix already exists in G1 in a stronger form (`_stamp_header_locked`, which echoes from lowstate and stamps `MODE_PR`).
- G1 has its own fixed 2 s upper-stale watchdog plus the stricter state and lower watchdogs.
- The heartbeat is **missing in G1**. G1 nodes don't need it: g1_locomotion reads DDS directly, and the safety layer's lower watchdog estops. Adding it would help tooling and parity [I].

### 2.4 Interface divergence in the action definitions [V]

`diff` of `custom_ros_messages` at GOLEM pin `c60205c` against G1 `g1_msgs`:
- `FrameTask.action`: GOLEM added `bool slow_mode` and `bool relative`. G1 has neither.
- `NamedConfig.action`: GOLEM added `bool slow_mode`.
- `DualArm.action`: identical.

If g1_control is ever switched to `custom_ros_messages` to share one interface package, it must implement or **explicitly reject** `relative=true`. If it ignored the field, an offset goal would be executed as an absolute pelvis-frame target, which is unsafe.

### 2.5 How deep is the rewrite? Superset or fork?

**A G1-only fork of both packages [V]:**

- **Every file shared with H1 differs.** At minimum every import was renamed (`h12_ros2_controller.*` → `g1_control.*`). Substantive changes vs base: `channel_interface` 177 lines, `upper_controller` 303, `low_cmd_handler` 105, `robot_model` 112, `controller_config` 87, `joint_definition` 218 (rewritten to import g1_description), `joint_limits` 159, `named_config` 70.
- **Deleted from H1:** `benchmark/`, `plot/`, 13 examples, `gravity_comp_controller`/`gravity_comp_server`, `hand_controller(_node)` (Inspire hands on `rt/inspire/*`), `hand_cmd_gui`, `inspect_*`, the `debug`/`sport`/`safety_*` configs and the `handless` model option.
- **Robot identity is resolved at import time** through module constants:
  - G1: `NUM_MOTOR = NUM_BODY_MOTORS` (29) from `g1_description`; 16 modules import the joint tables or `g1_description`.
  - H1: `NUM_MOTOR = 27`, `ALL_JOINTS` including Magpie gripper joints, and URDF paths into `h12_ros2_model`/`CL_Assets`; 24 modules import them.
  - Waist handling is structurally different: three held joints (G1) against one `torso_joint` (H1; 14 references).
  - The end-effector frames differ: `*_hand_palm_link` against `*_wrist_yaw_link`.
  - `channel_interface` imports `ARM_SDK_WEIGHT_SLOT` from g1_description.
- **Generic improvements in G1's core:**
  - `robot_model._keep_body_joints` freezes every non-body joint by name and asserts the slot order.
  - The collision model is optional.
  - The IMU frame is configurable.
  - Measured-pose seeding, the follow gate and startup timeouts.

  These are robot-agnostic and could be back-ported to H1.
- **Safety layer:**
  - The G1 version is a **feature superset in modes** (adds arm_sdk_mode, the engage gate, single-writer detection, BOOTTIME watchdogs, signal hand-back, damp estop and the `robot: real|sim` domain guard).
  - It **lacks H1 runtime features**: the external serial estop (`h12/estop_status_raw` DDS via `estop_subscriber.py`, which G1 rejects at `config.py:216`), `/safety/heartbeat` and configurable `upper_stale_seconds`.
  - The configs are mutually incompatible: H1 configs have `estop:` and no `robot:`; G1 configs need `robot:` and reject `estop:`.
  - The limits tables are robot-specific (`MOTOR_COUNT` 27 vs 29).
  - Coincidentally, both use `SPLIT_UPPER_START = 12` (legs 0-11 on both robots).

**Could one package serve both robots via config?**
- **Safety layer:** technically feasible with moderate effort [I]. Steps:
  - a `robot_spec` (motor count, limits table, slot groups, arm_sdk support, expected mode_machine);
  - re-adding the external estop and the heartbeat to the G1 code base;
  - unifying the config schemas.

  The core relay logic is slot-indexed and nearly robot-agnostic. It is still not advisable for the port: h12_safety_layer is a separately owned submodule in active use on the real H1, and the G1 version has had no real-robot run yet (per G1 CLAUDE.md).
- **Controller:** much harder.
  - Upstream H1 has diverged by about 26k LOC of research code.
  - The startup and waist logic differ.
  - Several mechanisms are G1-only: the follow gate reading `rt/arm_sdk` weight, config pairing, arm_sdk semantics.
  - A shared core library (robot_model, ik_solver, planner, channel_interface) is a possible later refactor. Today it would be a three-way merge of two independently hardened forks.
- **Recommendation:** add `g1_control` and `g1_safety_layer` as separate packages. Optionally open follow-ups:
  - back-port `LatestReader` and the stale-goal abort into g1_control;
  - add `/safety/heartbeat` to g1_safety_layer;
  - eventually extract shared IK and model code.

---

## 3. Architectural differences that matter for coexistence

| Topic | H1 (GOLEM camera_ready) [V] | G1 (G1_GOLEM) [V] |
|---|---|---|
| Who writes `rt/lowcmd` | `h12_safety_layer` only. It is config-driven (`low_cmd_out`, default `rt/lowcmd`, `core/config.py:184-191`; publisher at `safety_layer.py:84`). Searching all of GOLEM `core_ws` finds no other runtime writer, apart from the safety layer's own `unsafe/send_{fixed,random}_positions.py --raw` debug tools and vendored unitree_ros2 examples. **GOLEM CLAUDE.md's claim holds** | `g1_safety_layer` only (rt/lowcmd in split/full; rt/arm_sdk in arm_sdk_mode) |
| `rt/arm_sdk` | **None.** The only reference is a read-only debug subscriber constant at `h12_ros2_controller/utility/topic_subscriber.py:12`; `ArmSDKPublisher` was deleted upstream | Default real mode. Weight in slot 29 `.q`, ramp in 1 s and out 3 s. The controller reads `rt/arm_sdk` for the follow gate |
| Modes | `split_mode`, `full_body_mode` | + `arm_sdk_mode` (default) |
| Header fields | Split merge copies `mode_machine`/`mode_pr` from the last state | Stamps `mode_pr=0` and echoes `mode_machine` from `rt/lowstate`; the engage gate requires `mode_machine==5` |
| Engage | Relays as soon as inputs arrive | Silent until the engage gate passes. In split/full it needs **every** input (the walking node and the controller) before it publishes anything |
| External estop | `estop_node` (serial) → DDS `h12/estop_status_raw` → `EStopSubscriber` (real bringup `use_estop:=true`) | Removed; the config rejects `estop:` |
| Liveness to other nodes | `/safety/heartbeat` (Bool, 10 Hz); consumed by h12_lowerbody_rl and lmg_deploy | None; the walking node watches DDS itself |
| Controller estop | Zero-gain burst, then the safety layer stops | The controller goes silent; the safety layer hands back (arm_sdk) or zeroes/damps (lowcmd) |
| Hands | Inspire `rt/inspire/{cmd,state}` from h12_ros2_controller; Magpie grippers via ROS services; both bypass the safety layer | Dex3 `rt/dex3/{side}/{cmd,state}` from g1_dex3; bypass the safety layer |
| Lower body | h12_lowerbody_rl (TorchScript; ROS-side `/lowstate`, `/safety/lowcmd_lower_in` via unitree_hg ROS msgs, which is the same wire topic as DDS `rt/safety/lowcmd_lower_in`; `/cmd_vel`; heartbeat-gated), MJPC | g1_locomotion (ONNX; raw DDS; `/cmd_vel`, `/locomotion/height`) |
| Config pairing | None (bringup passes configs) | `g1_control/utility/config_pairing.py`; sim and real told apart by name prefix |
| DDS NIC selection | `network.interface` in YAML | `G1_DDS_INTERFACE` env, overriding YAML |
| Real-robot domain | 0 (onboard PC) | 0 (laptop wired to the robot) |

**Shared code or patterns between g1_locomotion and h12_lowerbody_rl:** no shared code [V]. Shared patterns:
- `/cmd_vel` Twist input.
- Split-mode lower channel `rt/safety/lowcmd_lower_in`.
- Bringup interlock `start_position_verified`.
- Keep-last-1 state reads (H1 controller's `LatestReader` and G1 locomotion's `NEWEST_ONLY` are convergent designs).

Differences: TorchScript vs ONNX, ROS topics vs raw DDS, heartbeat-gated vs DDS-watchdog-gated.

---

## 4. Name collisions if added to GOLEM `core_ws`

### 4.1 Build level (single colcon workspace) [V]

| Item | H1 | G1 | Collides? |
|---|---|---|---|
| ROS package names | h12_ros2_controller, h12_safety_layer, h12_ros2_model, h12_lowerbody_rl, … | g1_control, g1_safety_layer, g1_locomotion, g1_dex3, g1_description | **No.** No `g1_*` package or file exists in GOLEM today |
| Top-level Python modules | `h12_ros2_controller`, `h12_safety_layer` | `g1_control`, `g1_safety_layer`, `g1_locomotion`, `g1_dex3`, `g1_description` | No. `homie_policy` is loaded by file path, not installed |
| Console scripts | `frame_task_server`, `frame_task_client`, `dual_arm_server`, `dual_arm_client`, `joint_state_publisher` (controller); `safety_node` (safety) | same names | **No.** ament_python installs to `lib/<pkg>/` (setup.cfg `script_dir=$base/lib/<pkg>`), and launch files use `package=` + `executable=` |
| Launch/config/rviz file names | `robot_launch.py` etc. | same filenames | No (`share/<pkg>/…`) |
| Interface packages | `custom_ros_messages` | `g1_msgs` (dependency of g1_control and g1_dex3) | No at build. Duplicate type *names* in different packages are fine |
| uv project names (`pyproject.toml`) | `h12-ros2-controller`, `h12-safety-layer` | **`h12-ros2-controller`, `h12-safety-layer`** (unchanged in g1_control/g1_safety_layer `pyproject.toml:2`) | Only for uv workflows (colcon uses setup.py). Rename in the port |
| uv/requirements paths | (H1 its own) | `../../../../third_party/unitree_sdk2_python` (`pyproject.toml:29`, `requirements.txt:3`) | GOLEM keeps the SDK at `<repo>/unitree_sdk2_python` (same pin `c158c8d` in both repos [V]); the path would be wrong in GOLEM for uv users |

### 4.2 Runtime on a shared ROS/DDS domain [V for names; impact I]

| Item | Collides? | Harmful? |
|---|---|---|
| Node names `safety_node`, `frame_task_server`, `joint_state_publisher`, `robot_state_publisher`, `dual_arm_server` | **Yes**, identical in h12_bringup and g1_bringup (`g1_core.launch.py:167-229`) | Duplicate node names break parameter services and introspection; only if both stacks share a domain |
| Actions `/frame_task`, `/named_config`, `/dual_arm` | **Yes**, same names, **different types** (`custom_ros_messages/*` vs `g1_msgs/*`), and FrameTask/NamedConfig fields differ | Clients would see two servers or fail type matching; skills targeting `/frame_task` would hit the wrong robot |
| Topics `joint_states`, `frame_names` (StringArray from different packages), `frame_*`, `left/right_ee_*`, `/robot_description`, `/cmd_vel` | Yes | Mixed robots in TF/RViz; `/cmd_vel` would drive both lower bodies |
| Unitree DDS `rt/lowstate`, `rt/lowcmd`, `rt/safety/lowcmd_in`, `rt/safety/lowcmd_upper_in`, `rt/safety/lowcmd_lower_in` | **Yes, same names and identical IDL type** (`unitree_hg::msg::dds_::LowCmd_/LowState_`, 35-slot arrays) | **Dangerous:** no type-level separation; slot layouts differ (H1 27 motors with torso at 12; G1 29 with waist 12-14 and arm_sdk weight at 29). The G1 safety layer's engage gate would refuse an H1 `rt/lowstate` (mode_machine ≠ 5 in the H1 RoboCasa sim, which never sets it [V by grep]; real H1-2 value unknown [I]) and its 0.15 rad pose check would refuse mismatched commands. **The H1 safety layer has no such gate** and would relay a G1 controller's upper channel (slots 27-28 unchecked, since MOTOR_COUNT 27). The single-writer check in G1 would latch on a second writer |
| `rt/arm_sdk`, `rt/dex3/*`, `/<side>/hand/*`, `/locomotion/height` | G1-only | Harmless |
| `/safety/heartbeat`, `rt/inspire/*`, `/<side>/gripper/*`, `h12/estop_status_raw`, `/lowerbody/*` | H1-only | Harmless; G1 would not publish `/safety/heartbeat` |
| Real-robot domain 0 | Both robots use domain 0 | If an H1 onboard PC and a G1 laptop share a LAN segment on domain 0, cross-talk is possible [I]. Recommend documenting distinct domains or physically separate networks, and that a sim domain must never host both stacks at once |

**Port rule to add to GOLEM CLAUDE.md:** one robot stack per DDS domain. The invariant "h12_safety_layer is the only publisher of rt/lowcmd; there is no rt/arm_sdk path" becomes per robot: on a G1, `g1_safety_layer` is the sole writer of `rt/lowcmd` **and** `rt/arm_sdk`.

---

## 5. g1_locomotion specifics (HOMIE)

Covered in §1.3. Additional points:
- **Policy dir mount** [V]: `locomotion_node.py:16` defaults `policy_dir` to `/home/code/g1_locomotion`. G1 compose mounts `../core_ws/src/g1/g1_locomotion/policy` there (ros service `:125`, robocasa `:55`). GOLEM's compose has no such mount. The port needs one of the following:
  - add the mount to GOLEM's `ros` (and the robocasa G1 sim) service;
  - or pass `policy_dir:=$(ros2 pkg prefix …)/…`;
  - or install `policy/` via `setup.py data_files`. That copies CC BY-NC files into `install/`, which is legal non-commercially but contradicts the "used in place" design.
- **Shared with the sim** [V]: the G1 sim's HOMIE stand-in imports the same `homie_policy.py`. The policy dir is effectively shared infra between `g1_locomotion` and `sim/`.
- **CPU sensitivity** [V via G1 CLAUDE.md §8 and README]: one single-threaded onnxruntime session; a default session starved the node.
- **Relation to h12_lowerbody_rl** [V]: none in code (see §3). The H1 bringup's `h12_lowerbody.launch.py` and `lowerbody_*.yaml` do not apply.

## 6. g1_description specifics

Covered in §1.5. To port it:
- GOLEM needs a `third_party/leprechaun_g1` submodule, or a different symlink target.
- **The symlink depth `../../../../`** assumes `core_ws/src/g1/<pkg>`. So does `parents[3]` in `generate_joints.py:18` and `g1_control/utility/path_definition.py:6`.
- Keep the same `core_ws/src/g1/` layout in GOLEM to avoid edits.
- Add the corresponding bind mount `/home/code/third_party/leprechaun_g1`.
- Today GOLEM's ros container mounts `CL_Assets` at `/home/code/CL_Assets` for h12_ros2_model, which is the analogous pattern.

## 7. Hardcoded G1_GOLEM paths/names in these five packages [V]

| Item | Where | Port action |
|---|---|---|
| `HAMS_*` env vars, `hams_*` names | **None** in these five packages (only in G1_GOLEM docker/sim/tests) | none |
| `G1_DDS_INTERFACE` | `g1_control/utility/controller_config.py:304-307`, `core/controller/upper_controller.py:163` (message), `g1_safety_layer/core/config.py:163-166`, `g1_locomotion/config.py:164-174`, `g1_dex3/hand_node.py:69-75`, and tests | Keep (robot-scoped and descriptive) or rename to a `GOLEM_*` scheme; GOLEM has no DDS-interface env var today |
| `G1_SAFETY_LOG_DIR` | `g1_safety_layer/core/config.py:27` and config comments | Keep, or rename alongside a GOLEM log-dir convention |
| `/home/code/g1_locomotion` | `g1_locomotion/locomotion_node.py:16`, `policy/homie_policy.py:28-29` (docstring), tests' comments | Add the mount in GOLEM compose, or change the default |
| `../third_party/leprechaun_g1/...xml` | `g1_locomotion/policy/g1.yaml:6` (sim2sim script only) | Resolves when policy/ is mounted beside `/home/code/third_party`; keep the layout |
| `../../../../third_party/leprechaun_g1` symlink; `parents[3] / 'third_party'` | `g1_description/leprechaun_g1`, `g1_description/setup.py:8-24`, `tools/generate_joints.py:17-19`, `g1_control/utility/path_definition.py:5-6` | Requires `<repo>/third_party/leprechaun_g1` and the `core_ws/src/g1/` depth |
| `../../../../third_party/unitree_sdk2_python` | `g1_control/pyproject.toml:29`, `requirements.txt:3`; same in g1_safety_layer | Point to GOLEM's `../../../../unitree_sdk2_python` or drop the uv files |
| uv project names `h12-ros2-controller` / `h12-safety-layer` | `g1_control/pyproject.toml:2-4`, `g1_safety_layer/pyproject.toml:2-4` | Rename to avoid confusion with the H1 packages |
| Sibling-dir test invocations `PYTHONPATH=.:../g1_description` | package READMEs | Fine if the `core_ws/src/g1/` layout is kept |
| Maintainers | g1_control/g1_safety_layer list `tonyzyt2000` (the H1 owner); g1_locomotion/g1_description/g1_dex3 list Niraj Pudasaini | Decide ownership under GOLEM's per-package ownership rule |
| Domain defaults | Real configs `network.domain_id: 0`, sim `1` | Matches GOLEM's convention (0 real, 1 sim) |

---

## 8. Port risks and recommendations

1. **Keep separate packages** (`g1_control`, `g1_safety_layer`, `g1_locomotion`, `g1_dex3`, `g1_description`) under `core_ws/src/g1/`, in-tree as in G1_GOLEM. Don't merge them into the H1 submodules. Build-level coexistence is clean.
2. **Don't break H1 users' builds.** `g1_description/setup.py` raises without leprechaun_g1, and GOLEM's `launch_ros.sh` builds every package. Either:
   - make leprechaun_g1 a required submodule, as CL_Assets already is; or
   - make g1_description degrade gracefully, or gate the G1 packages with `COLCON_IGNORE` or a build flag.
3. **Docker:** add `onnxruntime` (G1 pins 1.23.2) **at the end** of GOLEM's RosDockerfile, per its layer discipline. Add the `third_party/leprechaun_g1` mount and the `policy/` mount for `ros` (and for the G1 RoboCasa sim if it is ported). GOLEM and G1 share CycloneDDS 0.10.x and `unitree_sdk2_python` pin `c158c8d` [V], so the single-writer `sample_info` feature should work.
4. **Interfaces:** g1_control and g1_dex3 need `g1_msgs` (for SetHandPose, FrameTask, NamedConfig, DualArm and StringArray). FrameTask/NamedConfig have diverged from `custom_ros_messages`. Decide one of:
   - keep `g1_msgs` (simplest), and accept the action-name and type clash, which only matters when both stacks share a domain;
   - or move g1_control onto `custom_ros_messages` + a `SetHandPose` addition, and implement or reject `relative`/`slow_mode`.
5. **Runtime isolation:** document "one robot stack per DDS domain". Consider namespacing G1 actions (for example `/g1/frame_task`) if multi-robot sessions are ever wanted. Consider adding an engage-style mode_machine/robot check to h12_safety_layer [I].
6. **Docs to update in GOLEM:**
   - CLAUDE.md §8, "Other invariants": the safety layer is the sole `rt/lowcmd` writer **per robot**; the G1 also has `rt/arm_sdk`.
   - §4 package table.
   - The HOMIE non-commercial license note.
7. **Optional follow-ups (not needed for the port):**
   - Back-port H1's `LatestReader` and stale-goal abort into g1_control.
   - Add `/safety/heartbeat` to g1_safety_layer.
   - Upstream G1's hardening (engage gate, single-writer, BOOTTIME, signal hand-back) to h12_safety_layer.
   - Long term: a shared IK/model core.
8. **Unverified:**
   - leprechaun_g1 contents, size and LFS usage (submodule not checked out).
   - The real H1-2's `mode_machine` value.
   - Whether GOLEM's ROS image has `ompl` and `meshcat` (likely, since the H1 controller uses the same stack).
   - colcon abort-on-error behavior as it applies to GOLEM's launcher.
