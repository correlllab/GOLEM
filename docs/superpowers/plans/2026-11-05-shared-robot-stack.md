# One ROS Stack for Both Robots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The H1-2 and the G1 run the same `golem_safety`, `golem_upper_ik` and `golem_slam` code and node graph, configured per robot, with the MID-360 in its raw frame on both, the Mac on the x86 launch architecture, and the audit fixes the user chose.

**Architecture:** Robot data moves out of code into `config/<robot>/` directories; each shared package owns its launch files and both per-robot bringups include them, passing configs, delays and order. `golem_slam` builds one graph whose only inputs are the robot name (to pick the config directory) and config values. The safety layer's runtime behaviour does not change.

**Tech Stack:** ROS 2 Humble (rclpy, rclcpp, launch), Python 3.10, YAML, pytest, MuJoCo/RoboCasa, Unitree SDK2 Python, Docker.

**Spec:** `docs/superpowers/specs/2026-10-31-shared-robot-stack-design.md` (read it first; its goal numbers are cited below as G1–G13).

## Global Constraints

- Work in `/home/max/GOLEM` on branch `shared_robot_stack`. Stage files by name; never `git add -A`, `git add .` or `git commit -a`. Never commit `core_ws/src/golem_bringup/rviz/g1/g1.rviz` or the `core_ws/src/livox_ros_driver2` submodule state (the user's uncommitted work).
- Build and test only inside disposable containers, never on the host (exception: `python3 -m pytest tests/docker -q` runs on the host). The colcon wrapper is `ROS_DOMAIN_ID=87 bash /home/max/GOLEM/.superpowers/sdd/2026-11-05-shared-robot-stack/run_ros.sh --h12 '<command>'`: it builds `core_ws` in a container named `golem_ros_sdd`, then runs the command in `/home/code`. Package tests run as `cd /home/code/core_ws && colcon test --packages-select <pkg> --event-handlers console_direct+; colcon test-result --verbose`. Sim suites use their own runners (`tests/robocasa/run_in_container.sh`, `tests/robocasa/g1/run_in_container.sh`, `tests/robocasa/surface/run.sh h12|g1`, `tests/stack/run.sh`, `tests/g1/test_sim_*.sh`).
- Never use `ROS_DOMAIN_ID=0`. Never stop, restart, replace or `docker exec` into the user's `golem_ros` container (or any container you did not start).
- `golem_safety`'s runtime behaviour (configs, limits, estop responses, start-up, C++ core) does not change. Values moved from code into config files keep exactly the same value (for floats, the same `repr`).
- Comments and docstrings describe the code as it is (CLAUDE.md §8): no change narration. When behaviour changes, rewrite the nearby comments, the package README, and the CLAUDE.md/AGENTS.md text that describes it **in the same commit**. CLAUDE.md and AGENTS.md are identical apart from their title line and the `tools/` row (Claude Code vs Codex); edit both the same way.
- `CL_Assets` is a submodule: commit inside it on a branch named `shared_robot_stack` (create it from its current `main` HEAD `963c93f`), then commit the new pin in the superproject. Do not push anything.
- The G1's Unitree model files in `CL_Assets/g1/` stay byte-for-byte upstream. HOMIE's `core_ws/src/g1_homie/policy/` files are untouched.
- Do not modify Isaac (`CL_isaaclab_sim`, Isaac USD), `deploy_mjpc`, `mujoco_mpc`, or `lmg_deploy` beyond Task 11. Do not bump `livox_ros_driver2` (pin `13eb05e`).
- Ground truth stays off by default everywhere (CLAUDE.md §8).
- Implementers never dispatch subagents.

## Review Focus

- A robot started without its hand driver (`g1_sim.launch.py use_hands:=false`): navigation must still publish `/converted_scan` and `/map` (Task 6 test).
- A torso or waist that moves (H1-2 `torso_joint` ≠ 0, e.g. under `lmg_*` or a limp torso): `body → pelvis` must follow it on both robots (Task 5 test at `torso_joint` = 0.3 rad).
- A bad `rt/lowcmd` reaching the H1-2 sim (bad CRC, `mode_machine` 5, `mode_pr` 1, a NaN): dropped, motors stay off, a rate-limited warning (Task 9 tests).
- An operator confirm sent early, a heartbeat that never comes, a policy that throws or returns NaN on the H1-2 leg node: refused / waits / latches silent (Task 10 tests).
- The heartbeat going silent while a skill's arm goal is in flight: the goal is cancelled and the skill fails (Task 12 test).

## File Structure

- `core_ws/src/golem_upper_ik/config/{h12,g1}/robot.yaml` (new): per-robot arm data; `config/{h12,g1}/pairing.yaml` (new): safety→controller pairs. `golem_upper_ik/robots/` deleted.
- `core_ws/src/golem_upper_ik/launch/upper_ik.launch.py` (frame_task_server + pairing check), `launch/robot_state.launch.py` (new: robot_state_publisher + joint_state_publisher); `core_ws/src/golem_safety/launch/safety.launch.py` (safety_node).
- `core_ws/src/golem_slam/config/{h12,g1}/` (new): `fastlio.yaml`, `MID360_config.json`, `nav2.yaml`, `octomap.yaml`, `pipeline.yaml`, plus `g1/fastlio_sim.yaml`; `config/slam_toolbox.yaml` shared. `launch/navigation.launch.py` (new, replaces `h12_navigation.launch.py` and `g1_navigation.launch.py`). `golem_slam/body_to_pelvis.py` (renamed from `g1_body_to_pelvis.py`).
- `core_ws/src/h12_lowerbody_rl/h12_lowerbody_rl/supervisor.py` (new): ROS-free start-up and fault logic for the H1-2 leg node.
- `golem_robocasa/unitree_common.py` (new, optional): DDS command validation helpers shared by both sims' `unitree_interface.py`.
- `tests/sim/` (new): the closed-loop test for both robots.

---

### Task 1: `golem_upper_ik` robot data and pairing as config (G1, G4)

**Files:**
- Create: `core_ws/src/golem_upper_ik/config/h12/robot.yaml`, `config/g1/robot.yaml`, `config/h12/pairing.yaml`, `config/g1/pairing.yaml`
- Modify: `golem_upper_ik/profile.py`, `config.py`, `pairing.py`, `ros2/utility.py`, `core/planner/planner_process.py`, `setup.py` (install every `config/<robot>/` by glob), `docker/mac/scripts/robot_cli.sh` (comment path), tests in `golem_upper_ik/test/`
- Delete: `golem_upper_ik/robots/` (all three modules)

**Interfaces:**
- Produces, in `golem_upper_ik/profile.py`: `known_robots() -> tuple[str, ...]` (sorted; a robot is known when `share/golem_safety/robots/<robot>.yaml` and `share/golem_upper_ik/config/<robot>/robot.yaml` both exist); `robot_config(robot: str) -> dict` (the parsed, validated `robot.yaml`; raises `ValueError` naming the robot and the known ones for an unknown robot, and naming the key for a missing or unknown key). `load_profile` keeps its signature and `RobotProfile` its fields.
- Produces, in `golem_upper_ik/pairing.py`: `pairing_refusal(robot: str, controller: str, safety: str, sim: bool | None = None) -> str | None` and `default_controller_config(robot: str, safety: str) -> str`, one rule for every robot (below). `config_stem` and `is_sim_config` keep their behaviour. The `H12_*`/`G1_*` table constants are deleted.
- `robot.yaml` keys (all required, no others): `display_name`, `enabled_joints`, `held_joints`, `shutdown_keeps_held_joints`, `ee_frames`, `planner_frames`, `imu_frame`, `named_configs` (name → list of 14 floats), `startup_sequence`, `config_defaults` (mapping), `default_config`.
- `pairing.yaml`: `pairs:` mapping safety config stem → controller config stem. H1-2: the 8 entries of today's `H12_SAFETY_CONTROLLER_CONFIGS`. G1: `split: g1_split`, `sim_split: sim_split`.
- The one pairing rule, checked in this order, each a refusal string: the safety config is not in the robot's `pairs`; the controller config is not the one paired with it; with `sim` given, a safety or controller config whose `is_sim_config` differs from `sim`. Ruling: this tightens the H1-2 (names outside its table and sim/real mixes are now refused); no H1-2 bringup passes such a pair.

- [ ] **Step 1: Capture today's values.** At the branch base, dump every field of `robots/h12.py` and `robots/g1.py` (named configs via `ndarray.tolist()`) and both pairing tables to a JSON file in the plan workspace with a throwaway script run in the container. It is the reference for Step 5 and is not committed.
- [ ] **Step 2: Write the failing tests** in `test/test_profile.py`, `test/test_pairing.py`, `test/test_config.py`:
  - `test_robot_config_matches_reference[h12|g1]`: `robot_config(robot)` equals the Step 1 JSON field by field (named configs compared with `==` on Python floats, not approx).
  - `test_known_robots`: `known_robots() == ('g1', 'h12')`.
  - `test_robot_config_rejects_unknown_and_missing_keys`: a temp `robot.yaml` with an extra key and one with a key removed both raise `ValueError` naming the key.
  - `test_pairing_one_rule[robot, safety, controller, sim, expected]`: G1 `('split','g1_split',False) → None`, `('full', ..., False)` refused (not in pairs), `('sim_split','sim_split',False)` refused (sim mismatch); H1-2 `('relax_safety_split','safety_split',False) → None`, `('sim_safety_split','sim_safety_split',True) → None`, `('relax_safety_split','sim_safety_split',None)` refused (wrong controller), `('debug_unknown','safety_split',None)` refused (not in pairs).
  - Keep the existing per-robot tests that compare paired safety/controller configs' topics and gains, iterating `known_robots()` and each robot's `pairs` instead of the deleted tables.
- [ ] **Step 3: Run them and see them fail** (`colcon test --packages-select golem_upper_ik`): import errors for `known_robots`/`robot_config`.
- [ ] **Step 4: Implement.** Write the two YAML files per robot from the Step 1 values (G1 postures as their exact float values, e.g. `1.5707963267948966`). Implement `known_robots`, `robot_config`, the pairing rule; repoint every reader listed in the spec research (`profile.load_profile`, `config.load_controller_config`'s defaults, `ros2/utility.node_parameters`' default config, the planner's unavailable message, which now names `golem_upper_ik/config/<robot>/robot.yaml`). Delete `robots/`. Replace each test file's local `ROBOTS = ('h12', 'g1')` with `known_robots()`.
- [ ] **Step 5: Run all `golem_upper_ik` tests** and the Step 1 reference comparison: all pass, `colcon test-result` shows 0 failures.
- [ ] **Step 6: Commit** `golem_upper_ik: robot data and safety pairing are per-robot config files`.

### Task 2: Package launch files, included by both bringups (G2)

**Files:**
- Modify: `core_ws/src/golem_safety/launch/safety.launch.py`, `core_ws/src/golem_upper_ik/launch/upper_ik.launch.py`; create `core_ws/src/golem_upper_ik/launch/robot_state.launch.py`; `golem_upper_ik/package.xml` (`exec_depend robot_state_publisher`)
- Modify: `core_ws/src/golem_bringup/launch/h12/h12_state.launch.py`, `h12_control.launch.py`, `launch/g1/g1_core.launch.py`
- Test: `golem_safety/test/test_node_ros.py`, `golem_upper_ik/test/test_frame_task_server.py` (launch test), `golem_bringup/test/test_bringup_packages.py`, `test_g1_launch.py`, `test_nav_launch.py`

**Interfaces:**
- `golem_safety safety.launch.py` args: `robot` (required), `config` (required), `sim` (default `false`), `use_estop` (default `false`), `use_sim_time` (default `false`), `start_delay` (default `0.0` s; the node runs inside a `TimerAction` of that period). Node: `golem_safety/safety_node`, name `safety_node`, params `{robot, config, sim, use_estop, use_sim_time}`, `emulate_tty=True`.
- `golem_upper_ik upper_ik.launch.py` args: `robot`, `config`, `safety_config` (all required), `sim` (required), `use_sim_time` (default `false`), `start_delay` (default `0.0`). First entity: an `OpaqueFunction` that calls `pairing_refusal(robot, config, safety_config, sim)` and raises `RuntimeError(f'upper_ik: not starting: {reason}')`. Then `frame_task_server` (name `frame_task_server`, params `{robot, config, use_sim_time}`, `emulate_tty=True`) in a `TimerAction(start_delay)`.
- `golem_upper_ik robot_state.launch.py` args: `robot` (required), `use_sim_time` (default `false`), `urdf_file` (default `''`, meaning `golem_ros2_models.paths.urdf_path(robot, ros=True)`). Starts `robot_state_publisher` (`robot_description` = the file's contents) and `golem_upper_ik/joint_state_publisher` (params `{robot, use_sim_time}`), both `emulate_tty=True`.
- Each bringup includes `upper_ik.launch.py` **before** any other node-starting entity, so a refused pairing stops the launch before anything starts. Delays stay the bringups' choice: H1-2 `safety_delay` 3.0 / `frame_task_delay` 5.0 (sim top-levels pass 0.0), G1 safety 0 s, controller `controller_delay` 2.0, locomotion controller + `locomotion_delay` 2.0. `h12_control` keeps `use_frame_task` (no include when false) and its `estop_node`. `h12_state` keeps its `urdf_file` argument and passes it through. `g1_core` keeps its `OpaqueFunction` only for its own plumbing (config defaults per platform, `locomotion_refusal`, conditional includes); its refusal messages that remain keep the `g1_core: not starting:` prefix.

- [ ] **Step 1: Update the tests to the new structure** and see them fail: bringup tests walk `IncludeLaunchDescription` entities (resolve the included file path and its `launch_arguments`) as well as `Node` and `TimerAction`. Assertions to keep, restated against includes: G1 safety at delay 0, controller at 2.0, locomotion at 4.0; the configs each node gets; pairing refusals now raise `upper_ik: not starting: ...` for both robots; H1-2 top-levels pass `sim`/`use_estop`; the G1 includes `navigation` only with `use_nav`. `test_frame_task_server.py`'s launch test asserts `upper_ik.launch.py` starts only `frame_task_server` with args `{robot, config, safety_config, sim, use_sim_time, start_delay}`. `test_node_ros.py`'s launch test passes `use_sim_time:=false`.
- [ ] **Step 2: Implement** the three package launches and rewire `h12_state`, `h12_control`, `g1_core` to include them.
- [ ] **Step 3: Run** `colcon test --packages-select golem_safety golem_upper_ik golem_bringup`: all pass.
- [ ] **Step 4: Live check.** In the `run_ros.sh` container on domain 87, launch `h12_sim.launch.py use_rviz:=false use_skills:=false` and `g1_sim.launch.py use_rviz:=false` (no sim needed) for 30 s each, list nodes (`ros2 node list`), and confirm the same node names as before this task (`safety_node`, `frame_task_server`, `joint_state_publisher`, `robot_state_publisher`, plus the rest of each bringup). Paste the lists in the report.
- [ ] **Step 5: Update** CLAUDE.md/AGENTS.md §4 (`golem_safety`/`golem_upper_ik` rows: their launch files are used by the bringups) and §6 (building-block table and G1 core text). **Commit** `Bringups include golem_safety's and golem_upper_ik's launch files`.

### Task 3: H1-2 models carry the upside-down MID-360 (G5, models)

**Files:**
- Modify in `CL_Assets` (branch `shared_robot_stack`): `ros_assets/h1_2_magpie.urdf`, `h1_2_magpie_sphere.urdf`, `h1_2_handless.urdf`, `h1_2_handless_sphere.urdf`, `h1_2_magpie_homie.urdf`; regenerate `ros_assets/h1_2_magpie_ros.urdf` and `h1_2_handless_ros.urdf` with `ros_assets/gen_ros_urdf.py`; `mujoco_assets/h1_2_magpie.xml`; regenerate `robosuite_assets/robots/h1_2/robot.xml` with `robosuite_assets/build_assets.py`
- Modify: `golem_robocasa/h12_sensors.py` (elevation fan and its comment)
- Test: `core_ws/src/golem_slam/test/test_mid360_frames.py`; a new `tests/robocasa/test_h12_lidar_mount.py`

**Interfaces:**
- New frames, physically at the same points, each rotated 180° about its own x axis: `livox_joint` rpy `(π, 0.1531, 0)` (xyz unchanged `0.04874 0 0.67980`); `livox_imu_joint` xyz `(0.011, 0.02329, -0.04412)`, rpy `(-0.0016, -0.0119, 0)`. Write π as `3.141592653589793`. In the MJCF, the `livox_link` body and the `livox_site`/`livox_imu_site` sites get the equivalent orientation (compute quaternions; MuJoCo's default `eulerseq` is not URDF rpy).
- Rewrite the URDF/MJCF comments: the frames are the MID-360's raw frames (upside down); the driver applies no rotation. Remove every mention of `livox_imu_upright` and "upright (MID360_config.json roll 180)" from `CL_Assets`.
- `h12_sensors.py`: the local elevation fan becomes `(-7, +52)` degrees (the G1's `ELEVATION_DEG`), with a comment that the fan is in the sensor's raw frame.

- [ ] **Step 1: Write the failing tests.** `test_mid360_frames.py`: from `h1_2_magpie_ros.urdf`, `livox_link` in `livox_imu_link` stays under 2° and 0.1 m; `livox_imu_link` in `pelvis` has `R[2,2] < -0.99` at `torso_joint` = 0 (upside down); for each H1-2 URDF variant in `CL_Assets/ros_assets/` that has `livox_link`, its world pose equals the old pose (hard-code the old xyz/rpy in the test) times `Rx(π)` within 1e-9. `tests/robocasa/test_h12_lidar_mount.py`: load `robosuite_assets/robots/h1_2/robot.xml` and `mujoco_assets/h1_2_magpie.xml` with MuJoCo at `qpos0`; the `livox_link` body and `livox_imu_site` world rotations equal the URDF's (`torso_joint` 0) within 1e-6; a sim ray fan from `h12_sensors` hits the floor (some points below the sensor in the world) and the ceiling side (some above).
- [ ] **Step 2: Run them** (`colcon test --packages-select golem_slam`; `tests/robocasa/run_in_container.sh`) and see them fail.
- [ ] **Step 3: Edit the models, regenerate the derived files, change the fan.**
- [ ] **Step 4: Run** the two tests, `colcon test --packages-select golem_slam golem_ros2_models golem_upper_ik`, `tests/robocasa/run_in_container.sh`, `tests/robocasa/surface/run.sh h12` (the golden must still pass: frame ids and |acc| are unchanged), and `tests/stack/run.sh` (the H1-2 still stands, walks and maps in the sim). All pass.
- [ ] **Step 5: Commit** inside `CL_Assets` (`h1_2: the MID-360 frames are the sensor's raw, upside-down frames`), then in the superproject the pin plus `h12_sensors.py` and the tests (`H1-2 models carry the upside-down MID-360`).

### Task 4: The H1-2 lidar in its raw frame end to end (G5, stack)

**Files:**
- Modify: `core_ws/src/golem_slam/config/MID360_config.json` (extrinsic all zeros, as `g1_MID360_config.json`), `core_ws/src/golem_bringup/launch/h12/h12_real_drivers.launch.py` (the driver publishes `/livox/imu` directly), `core_ws/src/golem_bringup/setup.py`, `golem_bringup/test/test_real_lidar_config.py`
- Delete: `core_ws/src/golem_bringup/golem_bringup/livox_imu_upright.py`
- Modify: `golem_robocasa/h12_sensors.py` (lidar and IMU `frame_id` `livox_frame`), `core_ws/src/golem_slam/golem_slam/navigation_launch.py` (H1-2 path: a static identity `livox_link → livox_frame`, as the G1 path's `mid360_link → livox_frame`), `golem_bringup/launch/h12/h12_sim_mac.launch.py` (its `livox_link → lidar_link` static becomes `livox_link → livox_frame`), `golem_bringup/rviz/h12/main.rviz` if it names `lidar_link`
- Re-record: `tests/robocasa/surface/h12.json` (`run.sh h12 --record`)
- Docs: CLAUDE.md/AGENTS.md §6 (`h12_real_drivers` row), `core_ws/src/golem_slam/config/mid360.yaml` comments, `lmg_deploy` is not touched (spec: known consequence)

- [ ] **Step 1: Update tests to fail first:** `test_real_lidar_config.py` asserts the H1-2 driver config has a zero extrinsic, the driver node publishes `/livox/imu` with no remap, and no `livox_imu_upright` entry point exists in `golem_bringup/setup.py`. A `navigation_launch` test asserts the H1-2 graph has the `livox_link → livox_frame` static.
- [ ] **Step 2: Implement**, delete the node, re-record the H1-2 surface golden (only the three `/livox/*` frame ids may change; confirm with `git diff tests/robocasa/surface/h12.json`).
- [ ] **Step 3: Run** `colcon test --packages-select golem_bringup golem_slam`, `tests/robocasa/surface/run.sh h12`, `tests/robocasa/run_in_container.sh`, `tests/stack/run.sh`: all pass.
- [ ] **Step 4: Commit** `H1-2 lidar in its raw frame: the driver applies no rotation; livox_imu_upright deleted`.

### Task 5: One `golem_slam` graph for both robots (G3, G7)

**Files:**
- Create: `core_ws/src/golem_slam/config/h12/{fastlio.yaml, MID360_config.json, nav2.yaml, octomap.yaml, pipeline.yaml}`, `config/g1/{fastlio.yaml, fastlio_sim.yaml, MID360_config.json, nav2.yaml, octomap.yaml, pipeline.yaml}` (moved with `git mv` from `mid360.yaml`, `MID360_config.json`, `nav2_config.yaml`, `octomap.yaml`, `g1_fastlio.yaml`, `g1_fastlio_sim.yaml`, `g1_MID360_config.json`, `g1_nav2.yaml`, `g1_octomap.yaml`, `g1_pipeline.yaml`); `config/slam_toolbox.yaml` (moved from `slam_toolbox_h1.yaml`)
- Create: `launch/navigation.launch.py`; delete `launch/h12_navigation.launch.py`, `launch/g1_navigation.launch.py`
- Rename: `golem_slam/g1_body_to_pelvis.py` → `golem_slam/body_to_pelvis.py` (executable `body_to_pelvis`)
- Modify: `golem_slam/navigation_launch.py`, `golem_slam/level_odom.py` (moving base transform is the only mode: the `dynamic_base_transform` parameter and the cached branch are deleted), `setup.py` (configs by glob), `golem_slam/record_slam_bag.py` if it names configs
- Modify includers: `golem_bringup/launch/h12/{h12_sim,h12_real_robot,lmg_sim,lmg_real,h12_real_drivers}.launch.py`, `launch/g1/{g1_core,g1_real_drivers}.launch.py`, `golem_slam/launch/h12_real_slam.launch.py`, `tests/g1/*` and `tests/stack/*` where they name configs or launch files
- Test: `golem_slam/test/test_navigation_launch.py`, `test_g1_pipeline.py` → `test_body_to_pelvis.py`, `test_level_odom.py`, `golem_bringup/test/test_nav_launch.py`

**Interfaces:**
- `navigation.launch.py` args: `robot` (required), `use_sim_time` (default `false`), `fast_lio_config` (default `config/<robot>/fastlio.yaml`), `nav2_params_file`, `octomap_params_file` (defaults from `config/<robot>/`), `slam_params_file` (default `config/slam_toolbox.yaml`). The G1 sim passes `fast_lio_config:=<share>/config/g1/fastlio_sim.yaml`.
- `pipeline.yaml` keys (both robots, all required): `lidar_link` (`livox_link` / `mid360_link`), `scan: {min_height, max_height}` (H1-2 `0.25, 1.7`; G1 `0.15, 1.3`), `level_odom: {min_floor_depth}` (H1-2 `1.0`; G1 `0.5`).
- The graph, identical on both robots (same node names, executables and remaps; only parameter values differ): static `<lidar_link> → livox_frame` (name `livox_frame_tf`); `fast_lio` (name `fast_lio`) on `fast_lio_config`; `body_to_pelvis` (params `fastlio_config`, `lidar_frame: livox_frame`); `level_odom`; `self_filter`; `level_cloud` (`/cloud_body_filtered` → `/cloud_base_filtered`); `octomap_server`; `pointcloud_to_laserscan` (name `cloud_registered_to_laserscan`, `cloud_in` `/cloud_base_filtered`, `target_frame ''`); `slam_toolbox`; Nav2 inside a `GroupAction` with `SetRemap('behavior_server:cmd_vel', 'cmd_vel_nav')`.
- H1-2 `fastlio.yaml` gains `extrinsic_T`/`extrinsic_R`: `livox_link` in `livox_imu_link` from the Task 3 URDF (values written out; `extrinsic_est_en` stays `true`). The launch no longer reads the URDF for FAST-LIO.
- Tuning values move unchanged (spec anti-goal).

- [ ] **Step 1: Write the failing tests.**
  - `test_one_graph`: for `h12` and `g1`, the set of `(package, executable, name, remappings)` of every node and the SetRemap are equal.
  - `test_profiles_drive_the_differences`: scan heights and `min_floor_depth` per robot equal the `pipeline.yaml` values above; FAST-LIO, Nav2 and OctoMap files resolve to `config/<robot>/`.
  - `test_h12_extrinsic_matches_urdf`: H1-2 `fastlio.yaml` extrinsic equals `livox_link` in `livox_imu_link` from `h1_2_magpie_ros.urdf` within 1e-6.
  - `test_body_to_pelvis[h12-torso-0.3, g1-waist-0.0256]`: with the URDF chain at `torso_joint` = 0.3 rad (H1-2) or waist pitch 0.0256 (G1), `world_body · body_to_pelvis == world_pelvis` within 1e-9.
  - `test_level_odom_follows_base`: the existing dynamic-mode test, now the only mode.
  - Bringup `test_nav_launch.py`: G1 real includes `navigation.launch.py` with `{robot: g1, use_sim_time: false}`, G1 sim with `fast_lio_config` ending `config/g1/fastlio_sim.yaml`; every H1-2 includer passes `robot: h12`.
- [ ] **Step 2: Run them** and see them fail.
- [ ] **Step 3: Implement** the moves, the launch, the renamed node (its link name comes from `pipeline.yaml`'s `lidar_link` through the launch), the includer updates.
- [ ] **Step 4: Run** `colcon test --packages-select golem_slam golem_bringup`; `tests/g1/test_sim_nav.sh`; `tests/stack/run.sh`: all pass. Live-check both bringups as in Task 2 Step 4 and confirm the node lists of the navigation part are equal.
- [ ] **Step 5: Update** CLAUDE.md/AGENTS.md §4 `golem_slam` row and §6 (one `navigation.launch.py`), `golem_slam` README, `docs/g1/navigation.md`, `docs/NAVIGATION_DEMO.md` paths. **Commit** `golem_slam: one navigation graph for both robots, configured per robot`.

### Task 6: Navigation keeps running without a hand driver (G6)

**Files:** Modify `core_ws/src/golem_slam/golem_slam/self_filter.py`; test `core_ws/src/golem_slam/test/test_self_filter.py` (new)

**Interfaces:** `SelfFilter` keeps its parameters. Per link it tracks whether a TF has ever been found. A cloud is filtered with every link that has a TF now; a link that has **never** had a TF is skipped (a warning listing such links once, then again only if the set changes); a link that had a TF before and has none now drops the cloud, as today.

- [ ] **Step 1: Failing tests** with a fake TF buffer: `test_never_seen_links_are_skipped` (two links, one never resolvable → the cloud is published, filtered by the other link's box); `test_lost_link_drops_cloud` (a link resolvable on cloud 1, not on cloud 2 → cloud 2 not published); `test_warning_once` (three clouds, one never-seen link → one warning).
- [ ] **Step 2: Run, see fail. Step 3: Implement. Step 4: Run** `colcon test --packages-select golem_slam`, then a live check: `tests/g1/test_sim_nav.sh` with the bringup's `use_hands:=false` (add a pass-through so the script can set it, or run the bringup by hand in the sim) shows `/converted_scan` and `/map` publishing. Report how you ran it.
- [ ] **Step 5: Commit** `self_filter: skip links that never had a TF`; fix the `use_hands:=false` note in `docs/g1/first-session.md` if it warns about this.

### Task 7: A test that shared packages carry no robot branches (G4)

**Files:** Create `tests/docker/test_shared_packages.py`; modify whatever it finds.

**Interfaces:** Scans `core_ws/src/golem_safety/{src,include,launch}`, `core_ws/src/golem_upper_ik/{golem_upper_ik,launch,setup.py}`, `core_ws/src/golem_slam/{golem_slam,launch,setup.py}` (not `test/`, `tools/`, `config/`, `robots/` data YAML). Robot names are the stems of `core_ws/src/golem_safety/robots/*.yaml`. Fails on: a quoted string literal equal to a robot name or starting with `<robot>_` or `<robot>/`; `robot ==`/`== robot`/`robot in (` comparisons against literals; a file or module name starting with `<robot>_`. Messages name file and line.

- [ ] **Step 1: Write the test, run it** (`python3 -m pytest tests/docker/test_shared_packages.py -q`). **Step 2:** fix every hit it finds (error messages listing robots use `known_robots()`; `golem_safety`'s "h12 or g1" message lists the profiles found). **Step 3:** run it and the three packages' tests: pass. **Step 4: Commit** `Test that the shared packages carry no robot names in code`.

### Task 8: The Mac on the launch architecture (G13)

**Files:**
- Rewrite: `core_ws/src/golem_bringup/launch/h12/h12_sim_mac.launch.py`
- Delete: `core_ws/src/golem_slam/config/nav2_config_mac.yaml`
- Modify: `docker/mac/scripts/launch_ros_mac.sh` (`PKGS` adds `fast_lio` and what the shared pipeline launches; `BRINGUP_STUB_DEPS` drops them), `docker/mac/RosDockerfile.slim.arm64` (read `docker/BUILD.md` first: add PCL, `pcl_ros`/`pcl_conversions`, APR, `ros-humble-octomap-server`, Livox-SDK2 at the same pinned ref as the x86 images, and `livox_ros_driver2` built the way the x86 `RosDockerfile` builds it), `docker/mac/scripts/robot_cli.sh` (`rob_odom` reads `/Odometry`), `docker/mac/docker-compose.yml` comments
- Docs: CLAUDE.md/AGENTS.md §3 (mac toggles), §6 (the Mac row), §8 (ground truth: no stack opts in), `docs/MACOS.md`, `docs/NAVIGATION_DEMO.md`, `docs/RUNNING.md`
- Test: `core_ws/src/golem_bringup/test/test_mac_launch.py` (new), `test_bringup_packages.py` (add `h12_sim_mac` to the `h12_control` parametrization), `tests/docker` source-pin tests

**Interfaces:** `h12_sim_mac.launch.py` includes `h12_state` (`use_sim_time true`), `h12_control` (`sim true`, `safety_config`/`frame_task_config` `sim_safety_split.yaml`, both delays `0.0`, `use_sim_time true`), `h12_lowerbody` when `GOLEM_LOWERBODY` is `fame|walk|almi` (`start_position_verified true`, `lowerbody_params` `lowerbody_sim.yaml`, `lowerbody:=<value>`), and `golem_slam navigation.launch.py` (`robot h12`, `use_sim_time true`) when `GOLEM_SLAM` or `GOLEM_NAV2` is `1|true|on`. An unset or empty `GOLEM_LOWERBODY` means no leg node; any other value raises an error naming the allowed ones. The x86 `h12_sim.launch.py` is the reference for arguments.

- [ ] **Step 1: Failing tests** (`test_mac_launch.py`, env patched per case): no toggles → state + control only; `GOLEM_LOWERBODY=almi` → the lowerbody include with `lowerbody: almi`; `GOLEM_LOWERBODY=bogus` → error; `GOLEM_NAV2=1` alone → the navigation include with `robot: h12`; `GOLEM_SLAM=1` → the same include. Source-pin test: the Mac Dockerfile's Livox-SDK2 ref equals the x86 ones.
- [ ] **Step 2: Run, see fail. Step 3: Implement.**
- [ ] **Step 4: Run** `colcon test --packages-select golem_bringup`, `python3 -m pytest tests/docker -q`, and a live check in the x86 `run_ros.sh` container: run `h12_sim_mac.launch.py` by path with `GOLEM_LOWERBODY=almi GOLEM_NAV2=1` for 45 s and list nodes (expect the same nodes as `h12_sim.launch.py use_skills:=false use_rviz:=false`, minus the models and rviz). The arm64 image cannot be built on this host (no arm64 emulation); say so in the report.
- [ ] **Step 5: Commit** `The Mac launches the same blocks and navigation as the x86 H1-2 sim`.

### Task 9: The H1-2 sim's command path matches the G1 sim's (G12)

**Files:**
- Modify: `golem_robocasa/unitree_interface.py`, `golem_robocasa/h12_mujoco.py` (main loop and shutdown); optional create `golem_robocasa/unitree_common.py` and import it from `golem_robocasa/g1/unitree_interface.py` (no G1 behaviour change)
- Test: `tests/robocasa/test_h12_unitree_interface.py` (new, modelled on `tests/robocasa/g1/test_g1_unitree_interface.py`), `tests/robocasa/test_h12_mode_machine.py`

**Interfaces:**
- `rt/lowcmd`: a frame is dropped when its CRC is wrong, `mode_machine != 6`, `mode_pr != 0`, or any of q/dq/tau/kp/kd over the 27 body motors is not finite; drops are warned at most once per reason per 5 s with a suppressed count. A valid frame is stored as one immutable command swapped in by a single assignment, stamped with sim time; the DDS handler never touches `MjData`.
- Every motor is off (ctrl 0) until the first valid frame. Motor-on stays `mode == 1`; no torque clip is added.
- Timeout: `GOLEM_CMD_TIMEOUT` (default 0.5, sim s) must be finite and > 0 (error at start-up otherwise); checked inside the physics step against the command's sim time, applying only after the first valid frame; one log line when commands stop and one when they return. The watchdog thread is deleted.
- `rt/lowstate`: published from the main loop after every `mj_step`, under the sim lock (the `RecurrentThread` publisher is deleted). `tick = round(data.time * 1000) & 0xFFFFFFFF`; `mode_machine` 6, `mode_pr` 0; `imu_state.rpy` filled from the quaternion; `crc` set. q/dq/tau_est sources unchanged.
- `SimInterface.after_step()`, `SimInterface.shutdown()`; `h12_mujoco.py` calls `after_step()` right after `mj_step` and `shutdown()` in its `finally`.

- [ ] **Step 1: Failing tests:** `test_motors_off_until_first_valid_lowcmd`; `test_invalid_lowcmd_ignored` (bad CRC, `mode_machine` 5, `mode_pr` 1, NaN q → uncommanded, zero ctrl); `test_first_valid_lowcmd_drives_motors`; `test_lowcmd_timeout_turns_motors_off` (in sim time); `test_timeout_knob_must_be_finite_and_positive`; `test_lowstate_per_step` (5 steps → 5 writes, `tick == round(time*1000)`, `(mode_machine, mode_pr) == (6, 0)`, `Crc(msg) == msg.crc`, `rpy` matches the quaternion). Update `test_h12_mode_machine.py`'s fakes to the new publishing path.
- [ ] **Step 2: Run** `tests/robocasa/run_in_container.sh`, see fail. **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/robocasa/run_in_container.sh`, `tests/robocasa/g1/run_in_container.sh` (G1 unchanged), `tests/robocasa/surface/run.sh h12` (re-record only if a rate the golden tracks moves outside tolerance; say which), `tests/stack/run.sh`: all pass.
- [ ] **Step 5: Update** CLAUDE.md/AGENTS.md §8 (the H1-2 RoboCasa surface paragraph: valid-command rule and motors off until then). **Commit** `golem_robocasa: the H1-2 sim validates rt/lowcmd and publishes rt/lowstate every physics step`.

### Task 10: H1-2 leg node runtime checks; band release is one call on both leg nodes (G10, G11)

**Files:**
- Create: `core_ws/src/h12_lowerbody_rl/h12_lowerbody_rl/supervisor.py`; `core_ws/src/h12_lowerbody_rl/test/test_supervisor.py`
- Modify: `h12_lowerbody_rl/scripts/lowerbody_controller_node.py`, `golem_bringup/config/h12/lowerbody_sim.yaml`, `lowerbody_real.yaml`, `tests/stack/minimal_stack.launch.py` if it sets node params
- Modify: `core_ws/src/g1_homie/g1_homie/band.py`, `g1_homie/locomotion_node.py`, `g1_homie/test/test_band.py`, `test_locomotion_node.py`
- Docs: CLAUDE.md/AGENTS.md §6 (H1-2 start-up, G1 band text), both packages' READMEs

**Interfaces:**
- `supervisor.py`, no ROS imports: `class LegSupervisor` with phases `WAITING, RAMPING, HOLDING, RUNNING, FAULTED`; `on_lowstate(state, now) -> None`, `on_heartbeat(alive: bool, now) -> None`, `check_stale(now) -> None`, `confirm() -> tuple[bool, str]`, `set_cmd_vel(vx, vy, wz, now) -> None`, `cmd_vel(now) -> np.ndarray`, `step_policy(fn) -> np.ndarray | None` (calls the policy; a raised exception or a non-finite/wrong-shape output latches `FAULTED` with reason `policy step failed` / `policy output not finite` and returns None), `latch(reason) -> None`, properties `phase`, `fault`. Thresholds as `g1_homie`: upright gravity z < -0.9 to leave `WAITING`, fallen gravity z > -0.5 latches in `RAMPING`/`HOLDING`/`RUNNING`, non-finite state or a quaternion off unit by > 0.05 latches, `/lowstate` gap > `max_state_gap` latches, heartbeat false or silent > `heartbeat_timeout` after the first beat latches. `cmd_vel(now)` returns zeros when the last command is older than `cmd_vel_timeout`; a non-finite command is stored as zeros with a warning; a non-finite squat command is ignored. `confirm()` is accepted only in `HOLDING`.
- Node parameters added: `max_state_gap` (sim YAML `0.5`, real `0.05`), `cmd_vel_timeout` (`0.5` both). Existing parameters keep their names and values. In `FAULTED` the node publishes nothing, ever again, and stays up (the error is logged once); `golem_safety` holds the last command (CLAUDE.md §8).
- Band, both nodes: the toggle is called once, when the service is ready (no call while it is not); the reply is final. Success → released (the H1-2 node resets the active policy's history then, not when the call is sent; the G1 node as today). `None`, an exception or `success: false` → an error log `elastic band release failed: <why>`, the band stays, no further call. `g1_homie.band.BandRelease.result(False)` makes `poll` return None for good.

- [ ] **Step 1: Failing tests** in `test_supervisor.py` (fake clock, fake policy): waits for the first beat; waits while not upright; a false first beat latches; silence after the first beat latches; fallen latches in each of `RAMPING/HOLDING/RUNNING`; a stale `/lowstate` latches; confirm refused in `WAITING`/`RAMPING` and accepted in `HOLDING`; `cmd_vel` zero after 0.5 s; NaN `cmd_vel` → zeros; a policy raising → `FAULTED` and `step_policy` returns None; a policy returning NaN → `FAULTED`. In `g1_homie/test/test_band.py`: `test_failed_release_is_not_retried`. A node-level test for the H1-2 band (fake toggle service on domain 97): one call when ready, none before; a `success: false` reply → no second call.
- [ ] **Step 2: Run** (`colcon test --packages-select h12_lowerbody_rl g1_homie`), see fail. **Step 3: Implement** the supervisor, rewire the node's `_tick` onto it, change both band paths.
- [ ] **Step 4: Run** the two packages' tests, `tests/stack/run.sh` (H1-2 stands and walks), `tests/g1/test_sim_bringup.sh --walk`: all pass.
- [ ] **Step 5: Commit** `h12_lowerbody_rl: g1_homie's runtime checks; the band release is one call on both leg nodes`.

### Task 11: The LMG pre-pose waits for the first heartbeat (G9)

**Files:** Modify `core_ws/src/lmg_deploy/lmg_deploy/marl_deploy_node.py`; test `core_ws/src/lmg_deploy/test/test_heartbeat_gate.py` (new)

**Interfaces:** In `_tick`, while `_hb_last is None` the node publishes nothing, sets `~/state` to `waiting_heartbeat` and logs that once every 5 s; the pre-pose ramp (and `_prepose_start`) begins on the first tick after the first true beat. Nothing else changes.

- [ ] **Step 1: Failing test:** a node (or the smallest unit of it that owns `_tick`, with fakes for the policy files) receiving fresh `/lowstate` and no heartbeat publishes nothing on `/safety/lowcmd_lower_in` or `/safety/lowcmd_upper_in` for 1 s; after one true beat it publishes. **Step 2: Run, fail. Step 3: Implement. Step 4: Run** `colcon test --packages-select lmg_deploy`: pass. **Step 5:** update `lmg_deploy/README.md` and the CLAUDE.md/AGENTS.md §6 line that `lmg_deploy` reads the heartbeat. **Commit** `lmg_deploy: pre-pose only after the first safety heartbeat`.

### Task 12: Skills halt the arms on a silent heartbeat (G8)

**Files:** Modify `core_ws/src/golem_skills/golem_skills/base.py`; test `core_ws/src/golem_skills/test/test_heartbeat_halt.py` (new)

**Interfaces:**
- `/safety/heartbeat` handling: store the beat time (`time.monotonic()`); a false beat latches `_safety_latched = True` for the node's life.
- `arm_halt_reason() -> str | None`: `'safety heartbeat not received yet'` before the first true beat; `'golem_safety estop latched'` once latched; `'safety heartbeat silent for {s:.1f} s'` when the last beat is older than the parameter `heartbeat_timeout` (default `0.5` s); else today's `_arm_halt_reason`.
- While a `/frame_task` or `/named_config` goal is in flight, `_wait_future` checks `arm_halt_reason()` every poll; on a halt it cancels the goal on the server (the existing cancel path) and returns failure, so the skill fails with the halt reason. Grippers, navigation and `frontier_explore` are unaffected.

- [ ] **Step 1: Failing tests** with a fake clock and a fake action client: each of the three halt reasons; a true beat after a false one stays halted; a silent heartbeat during an in-flight `named_config` goal cancels it and the call returns False. **Step 2: Run** `colcon test --packages-select golem_skills`, fail. **Step 3: Implement. Step 4: Run** it: pass. **Step 5:** update `golem_skills` README and CLAUDE.md/AGENTS.md §6/§7 where they describe the heartbeat. **Commit** `golem_skills: the arms halt on a false or silent safety heartbeat`.

### Task 13: One closed-loop sim test for both robots, and the full run (Done)

**Files:**
- Create: `tests/sim/test_sim_stack.sh`, `tests/sim/test_sim_stack.py`, `tests/sim/README.md`; move shared harness pieces from `tests/g1/common.sh` into `tests/sim/common.sh` if both need them (`tests/g1` keeps working)
- Docs: `tests/README.md`, CLAUDE.md/AGENTS.md §9 (closed-loop checks), a final sweep of CLAUDE.md/AGENTS.md for text this branch made stale (header's verified-on line names 2026-11-05 for the parts this branch re-checked)

**Interfaces:** `tests/sim/test_sim_stack.sh --h12|--g1`, DDS domain 199 unless `ROS_DOMAIN_ID` is set (0 refused), `GOLEM_SIM_ODOM` unset (refused if set), spawn backed off into open floor (`GOLEM_SPAWN_BACKOFF=2.0`), bringup with navigation (`h12_sim.launch.py use_rviz:=false use_skills:=false` / `g1_sim.launch.py use_rviz:=false`). Pass criteria, all in one run:
- the robot stands unsupported after the band release (no fall; pelvis tilt ≤ 15°; pelvis height above a per-robot floor taken from the robot's stand in `tests/stack` / `tests/g1`);
- a Nav2 goal 1.5 m ahead in open floor is reached within 0.4 m;
- while walking to it, `/named_config` `t_pose` and then `home` both succeed;
- SLAM healthy throughout: `/Odometry` at ≥ 5 Hz, the `odom → base_footprint` TF present, `/map` grows, `/converted_scan` gaps ≤ 0.5 s;
- `safety_node` is the only publisher of `/lowcmd`; no estop latched.
Reuse `tests/g1/test_sim_nav.py`'s machinery where it fits (process control, in-sim state recording, checks).

- [ ] **Step 1: Write the test** and run it on the G1 first, then the H1-2; fix what it finds (fixes in the owning package, with a test there).
- [ ] **Step 2: Run everything in the spec's Done list:** every package's `colcon test`, `tests/robocasa/run_in_container.sh`, `tests/robocasa/g1/run_in_container.sh`, `tests/robocasa/surface/run.sh h12` and `g1`, `python3 -m pytest tests/docker -q`, `tests/stack/run.sh`, `tests/g1/test_sim_bringup.sh --walk`, `tests/g1/test_sim_nav.sh`, `tests/g1/test_sim_explore.sh`, `tests/sim/test_sim_stack.sh --h12`, `tests/sim/test_sim_stack.sh --g1`. Record each command's result in the report.
- [ ] **Step 3: Commit** `tests/sim: one closed-loop test of standing, walking, arms and SLAM for both robots`.
