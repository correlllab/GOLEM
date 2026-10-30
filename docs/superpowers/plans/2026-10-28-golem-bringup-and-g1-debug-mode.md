# golem_bringup, golem_robocasa and the G1's debug-mode start-up: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `h1_robocasa` → `golem_robocasa` and `h12_bringup` (+ `g1_bringup`) → `golem_bringup`; delete the four unused packages and `arm_sdk` mode; make the G1 start the way the H1-2 does (sim band, zero-gain stream from boot, HOMIE as the leg node that pre-poses, waits for a confirm on the real robot and releases the band in the sim).

**Architecture:** Three mechanical moves first (Tasks 1-3), each verified end to end, so later diffs are not mixed with renames. Then the G1 behavior change bottom-up: remove `arm_sdk` (4), one start-up rule in `golem_safety` (5), the sim (6), `g1_locomotion` (7), the bringups (8), the acceptance tests (9), the docs (10). Between Task 4 and Task 9 the G1 sim bringup is not expected to run end to end; each task's own tests are its gate.

**Tech Stack:** ROS 2 Humble, rclcpp (C++17) + gtest, rclpy + pytest, launch_ros, MuJoCo/robosuite (RoboCasa), unitree_sdk2py DDS (sim side and `g1_locomotion`), Docker compose wrappers.

**Spec:** `docs/superpowers/specs/2026-10-28-golem-bringup-and-g1-debug-mode-design.md` — the authority; read it before your task.

## Global Constraints

- **Branch** `golem_bringup_g1_debug` (on top of `golem_upper_ik_safety`). Commits end with the session's attribution trailer. Nothing is pushed.
- **Builds and tests run in containers.** `RUN <cmd>` below means `docker/scripts/docker_run.sh ros --restart --g1 bash -lc '<cmd>'` (G1 mode builds every package). Only `tests/docker` runs on the host (`python3 -m unittest discover -s tests/docker`).
  - Every pytest command starts with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
  - `CT <pkg...>` means `RUN 'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 && cd /home/code/core_ws && colcon build --symlink-install --packages-up-to <pkg...> && source install/setup.bash && colcon test --packages-select <pkg...> --event-handlers console_direct+; colcon test-result --verbose'`. Expected: `0 errors, 0 failures` (the bringup's pre-existing `test_copyright` skips are fine).
  - The H1-2 build: `docker/scripts/docker_run.sh ros --restart --h12 bash -lc 'cd /home/code/core_ws && colcon build --symlink-install --packages-ignore-regex "^g1_"'` must succeed without `g1_*` or `third_party/leprechaun_g1`.
- **Test DDS domains:** `golem_safety` 91, `golem_upper_ik` 92, `g1_locomotion` 97, `tests/stack` 193, surface probe 194, `tests/g1` 195-198. Never 0.
- **The H1-2 must not change behavior.** `tests/robocasa/surface/run.sh h12` compares unchanged against `h12.json` in every task that touches the sim; `tests/stack` arms and locomotion pass in Tasks 3 and 9.
- **No compatibility shims:** no package, launch file, symlink or config left under an old name (`h1_robocasa`, `h12_bringup`, `g1_bringup`, the four packages, `arm_sdk`, `engage_gate`).
- **Removed packages leave build artifacts.** When a task removes or renames a colcon package, delete its generated directories on this host: `core_ws/build/<pkg>`, `core_ws/install/<pkg>`, `container_cache/g1_tests/{build,install}/<pkg>` (list them first; they are build outputs, never source).
- **Submodules** are not edited, except that Task 2 removes the two deleted ones. `CL_Assets`, `cl_realsense`, `mujoco_mpc` keep their mentions of old names.
- **HOMIE's policy** stays in `core_ws/src/g1/g1_locomotion/policy/` with its `LICENSE` and `README.md`; after Task 6 the sim reads none of its files.
- **Comments describe the code as it is** (CLAUDE.md §8). Dated docs under `docs/superpowers/` are never edited, except this plan's change report.
- **Change report:** `docs/superpowers/specs/2026-10-28-golem-bringup-and-g1-debug-mode/change-report.md`, created in Task 1. Every task appends each change a user must know (a new name, a removed option, a behavior that differs per robot, a manual step), with the robot, before, after, why.
- **Fixed values** (spec):
  - Pre-pose ramp 2.0 s in both `g1_locomotion` configs.
  - Operator confirm: `/lowerbody/confirm_engage`, `std_srvs/srv/Trigger`, real config only.
  - Band: `/elastic_band/toggle`, `std_srvs/srv/Trigger`, `mujoco_env.ElasticBand` on `torso_link`, anchor 0.01 m above the torso's spawn position, stiffness 1e4 N/m, damping 2e3 N*s/m.
  - G1 heartbeat watchdog: false beat, or 0.5 s without a beat after the first, stops `locomotion_node` for good.
  - The sim's G1 joint dynamics and stance come from Unitree's `unitree_rl_gym` (BSD-3): `resources/robots/g1_description/g1_12dof.xml` default joint `damping="0.001" armature="0.01" frictionloss="0.1"`; `legged_gym/envs/g1/g1_config.py` `default_joint_angles` hip pitch -0.1, knee 0.3, ankle pitch -0.2, every other joint 0.
  - G1 `estop_response`: `mode: zero`, `after: keep_writing` (unchanged).
  - G1 safety configs after Task 4: `split`, `sim_split`, `full`; controller configs: `g1_split`, `sim_split`.

## Review Focus

1. **A band toggle sent twice** (a retry while a call is in flight, a late response) re-hangs or drops the robot mid-walk. Expected: at most one call in flight, never another after a success. Test: `test_band_release_one_call_in_flight` and `test_band_release_never_after_success` (Task 7).
2. **The heartbeat goes false or silent during the pre-pose or the confirm hold.** Expected: the node stops for good in every phase after the first beat. Test: `test_heartbeat_lost_in_every_phase_faults` (Task 7).
3. **A limit latch at boot** while the G1's legs hang at zero gain. Expected: no estop event during the whole start-up. Test: the `no estop during start-up` check in `tests/g1` (Task 9).
4. **A stale install of a removed or renamed package** keeps answering to its old name. Expected: after the documented cleanup, `ros2 pkg prefix <old>` fails. Test: the Task 3 Step 6 check, and the change-report entry.
5. **The G1 sim with nothing commanding it.** Expected: hanging from the band, motors off, upright; it falls only after the band is released. Test: `test_band_holds_limp_robot` and `test_release_drops_limp_robot` (Task 6).

---

### Task 1: `h1_robocasa` → `golem_robocasa`

**Files:**
- Move: `h1_robocasa/` → `golem_robocasa/` (`git mv`)
- Modify: `docker/docker-compose.yml:116-117`, `docker/mac/docker-compose.yml:70-71`, `docker/RobocasaDockerfile:4,90`, `docker/mac/RobocasaDockerfile.arm64:6,68`, `docker/scripts/launch_robocasa.sh:48`, `docker/mac/scripts/launch_robocasa_mac.sh:117`, `docker/scripts/docker_common.sh:212`, `docker/scripts/wait_for_sim.py:12-14`, `.github/workflows/simulator-tests.yml`, `.gitignore:69`, and every other living file from `git grep -l h1_robocasa -- ':!docs/superpowers' ':!CL_Assets'` (tests/robocasa, tests/g1, tests/stack, tests/docker, tests/README.md, core_ws/src/g1, README.md, CLAUDE.md, AGENTS.md, docs/)
- Create: `docs/superpowers/specs/2026-10-28-golem-bringup-and-g1-debug-mode/change-report.md`

**Interfaces:** Produces the path `golem_robocasa/` and the container path `/home/code/golem_robocasa`; every later task uses them.

- [ ] **Step 1:** Record `git status --ignored --short h1_robocasa` before moving.
- [ ] **Step 2:** `git mv h1_robocasa golem_robocasa`, then update every reference. Only the directory changes: `h12_mujoco.py` and every module keep their names.
- [ ] **Step 3:** Look at what is left in `h1_robocasa/`. Delete `__pycache__` directories; report anything else to the controller instead of deleting it.
- [ ] **Step 4: Verify.**
  - `git grep -n h1_robocasa -- ':!docs/superpowers' ':!CL_Assets'` prints nothing.
  - `python3 -m unittest discover -s tests/docker`: PASS.
  - `tests/robocasa/run_in_container.sh` and `tests/robocasa/g1/run_in_container.sh`: PASS.
  - `tests/robocasa/surface/run.sh h12` and `tests/robocasa/surface/run.sh g1`: PASS against the unchanged goldens. These use the existing `golem_sim_robocasa` image, which proves an image built before the rename still runs.
- [ ] **Step 5:** Create the change report (title, one line per entry format: robot / before / after / why). First entry: the rename, the new container path, and that sim containers must be restarted after pulling.
- [ ] **Step 6: Commit** `golem_robocasa: rename h1_robocasa, the RoboCasa sim of both robots`.

### Task 2: Delete `h12_safety_layer`, `h12_ros2_controller`, `g1_safety_layer`, `g1_control`

**Files:**
- Delete: `core_ws/src/h12_safety_layer`, `core_ws/src/h12_ros2_controller` (`git rm`, submodules: their `.gitmodules` entries go), `core_ws/src/g1/g1_safety_layer`, `core_ws/src/g1/g1_control` (`git rm -r`)
- Delete: `core_ws/src/golem_safety/tools/crc_fixtures.py`, `tools/snapshot_old_limits.py`, `core_ws/src/golem_upper_ik/tools/snapshot_old_controller.py`, `tools/snapshot_old_fk.py`
- Create: `core_ws/src/golem_safety/tools/h12_joint_limits.py` (the H1-2 table, copied verbatim from `h12_safety_layer/h12_safety_layer/core/joint_limits.py` before deleting it)
- Create: `core_ws/src/golem_safety/test/fixtures/README.md`, `core_ws/src/golem_upper_ik/test/fixtures/README.md`
- Modify: `core_ws/src/golem_safety/tools/generate_robot_data.py`, `robots/h12.yaml` (header), configs and comments in `golem_safety` and `golem_upper_ik` that name an old package, `core_ws/src/h12_lowerbody_rl/h12_lowerbody_rl/scripts/{lowerbody_controller_node,fame_node,walking_node}.py` (comment lines only), `tests/stack/test_launch_packages.py`, `core_ws/src/h12_bringup/test/test_bringup_packages.py`, `core_ws/src/g1/g1_bringup/test/test_bringup_packages.py`, `docker/BUILD.md`, `CLAUDE.md`, `AGENTS.md`, `docs/g1/{packages,safety,first-session}.md`

**Interfaces:** Produces `golem_safety/tools/h12_joint_limits.py` exposing the same names the generator imported from `h12_safety_layer.core.joint_limits`.

- [ ] **Step 1:** Record the provenance before deleting anything: `git ls-tree HEAD core_ws/src/h12_safety_layer core_ws/src/h12_ros2_controller` (the submodule SHAs) and `git rev-parse HEAD` (the in-tree packages). Write both fixture READMEs: what each fixture file holds, and the package and commit it was recorded from.
- [ ] **Step 2:** Copy the H1-2 table into `tools/h12_joint_limits.py`. Point `generate_robot_data.py` at it, drop its cross-check against `h12_ros2_controller`'s copy, and set the H1-2 header to `# Generated by tools/generate_robot_data.py from tools/h12_joint_limits.py. Do not edit by hand.`
- [ ] **Step 3:** Delete the four packages and the four tools. Delete the stale build artifacts (Global Constraints).
- [ ] **Step 4:** Reword every remaining mention outside `docs/superpowers`, `mujoco_mpc` and `cl_realsense`. A value's origin becomes what the value is. `logging.base_dir` becomes `/tmp/golem_safety` in every config that names `/tmp/h12_safety_layer` or `/tmp/g1_safety_layer`. CLAUDE.md and AGENTS.md lose the "awaiting deletion" rows and paragraph.
- [ ] **Step 5: Verify.**
  - `RUN 'cd /home/code/core_ws/src/golem_safety && python3 tools/generate_robot_data.py'`, then `git diff --stat core_ws/src/golem_safety/robots`: only `h12.yaml`'s first line changed.
  - `CT golem_safety golem_upper_ik h12_bringup g1_bringup h12_lowerbody_rl`: PASS.
  - The H1-2 build: PASS.
  - `git grep -nE 'h12_safety_layer|h12_ros2_controller|g1_safety_layer|g1_control\b' -- ':!docs/superpowers' ':!mujoco_mpc' ':!core_ws/src/cl_realsense' ':!*/fixtures/README.md' ':!docs/superpowers/specs/2026-10-28-golem-bringup-and-g1-debug-mode/change-report.md'` prints nothing.
- [ ] **Step 6:** Append to the change report: the deletions, the `logging.base_dir` change (G1 logs now go to `/tmp/golem_safety`), and the stale-install cleanup users run once.
- [ ] **Step 7: Commit** `Delete the unused safety layers and arm controllers of both robots`.

### Task 3: `golem_bringup`

**Files:**
- Move: `core_ws/src/h12_bringup` → `core_ws/src/golem_bringup` (`git mv`); every `core_ws/src/g1/g1_bringup` file into it; the file map is the spec's section 1 table. The `h12_bringup` Python module becomes `golem_bringup` and `resource/h12_bringup` becomes `resource/golem_bringup`.
- Modify: `golem_bringup/setup.py` and `package.xml`, every launch file, `docker/scripts/docker_stack.sh:37-38`, `docker/docker-compose.stack.yml:13-14`, `docker/scripts/docker_common.sh:218`, `docker/RosDockerfile:220`, `docker/mac/scripts/launch_ros_mac.sh:40,73-74,110-125,164`, `docker/mac/RosDockerfile.slim.arm64:153-154`, `core_ws/src/h12_slam/launch/h12_real_slam.launch.py`, the READMEs of `golem_safety`, `golem_upper_ik`, `g1_locomotion`, `g1_navigation` (and `config/g1_nav2.yaml`'s comment), `lmg_deploy`, `h12_lowerbody_rl/policies/almi`, `golem_robocasa/g1/g1_sensor_msgs.py`, `golem_upper_ik/golem_upper_ik/pairing.py` docstrings, `tests/g1/*`, `tests/stack/*`, `tests/docker/*`, `golem_bringup/common_cmds.md`, and the commands and package names in `CLAUDE.md`, `AGENTS.md`, `README.md`, `docs/`
- Test: `core_ws/src/golem_bringup/test/test_launch_layout.py` (new), the moved and merged tests

**Interfaces:**
- Produces the package `golem_bringup` with top-level files `launch/h12/{h12_sim,h12_sim_mac,h12_real_robot,h12_real_desktop,h12_safety_grippers,lmg_sim,lmg_real}.launch.py` and `launch/g1/{g1_sim,g1_real_robot}.launch.py`.
- Produces the shared G1 core `launch/g1/g1_core.launch.py`, `config/h12/*.yaml`, `rviz/h12/main.rviz` and `rviz/g1/g1.rviz`.
- Behavior does not change in this task.

- [ ] **Step 1: Write the failing test** `test_launch_layout.py`:
  - `test_launch_names_unique`: the basenames of every `launch/**/*.launch.py` are unique.
  - `test_top_level_files_exist`: each of the nine top-level files above exists under its subdirectory.
  - `test_setup_installs_every_launch_file`: every `launch/**/*.launch.py`, `config/**/*.yaml` and `rviz/**/*.rviz` appears in `setup.py`'s `data_files` under `share/golem_bringup/<same relative dir>`.
- [ ] **Step 2: Run it to make sure it fails.** `CT golem_bringup` fails: the package does not exist yet.
- [ ] **Step 3:** Do the moves. Write `setup.py` `data_files` per subdirectory (`launch/h12`, `launch/g1`, `config/h12`, `rviz/h12`, `rviz/g1`). Make `package.xml` the union of both packages' `exec_depend`s. Point every include and every share path in the launch files at the new subdirectories and names. Merge the two `test_bringup_packages.py` into one, and keep a single copyright/flake8/pep257 trio.
- [ ] **Step 4:** Update every reference outside the package (Files list). `docker_stack.sh` exports `GOLEM_BRINGUP_PACKAGE=golem_bringup` and `GOLEM_BRINGUP_LAUNCH=g1_sim.launch.py` for the G1. The stack compose file defaults to `golem_bringup` and `h12_sim.launch.py`. The Mac launcher builds `golem_bringup` (H1-2 only, no `g1_*`) and launches `$WS/src/golem_bringup/launch/h12/h12_sim_mac.launch.py`. Read the Mac launcher's build command and keep it building without any `g1_*` package; the Mac cannot be run here, so report what you checked.
- [ ] **Step 5:** Delete the stale build artifacts of `h12_bringup` and `g1_bringup`.
- [ ] **Step 6: Verify.**
  - `CT golem_bringup golem_upper_ik`: PASS.
  - The H1-2 build: PASS.
  - For each of the nine top-level files, `RUN 'cd /home/code/core_ws && source install/setup.bash && ros2 launch golem_bringup <file> --show-args'` exits 0.
  - `ros2 pkg prefix h12_bringup` and `ros2 pkg prefix g1_bringup` fail.
  - `python3 -m unittest discover -s tests/docker`: PASS.
  - `git grep -nE 'h12_bringup|g1_bringup|_bringup\.launch\.py' -- ':!docs/superpowers'` prints nothing, apart from `h12_slam`'s own `h12_real_slam.launch.py` name if matched.
- [ ] **Step 7: End to end, unchanged behavior.**
  - `tests/g1/test_sim_bringup.sh` (the arm test, still arm_sdk): PASSED.
  - `ROS_DOMAIN_ID=193 tests/stack/run.sh robocasa arms` and `... locomotion`: PASS.
- [ ] **Step 8:** Append to the change report: the old→new command table, and the stale-install cleanup.
- [ ] **Step 9: Commit** `golem_bringup: one bringup package for both robots, launch files split by robot`.

### Task 4: Delete `arm_sdk` mode

**Files:**
- `golem_safety`:
  - Modify: `include/golem_safety/{config,robot_profile,safety_core}.hpp`, `src/{config,robot_profile,safety_core,safety_node,lowcmd,dump_config_main}.cpp`, `robots/g1.yaml` (regenerated), `tools/generate_robot_data.py`, `tools/relay_latency_probe.py`, `README.md`.
  - Delete: `config/g1/arm_sdk.yaml`, `config/g1/sim_arm_sdk.yaml`, `test/fixtures/limits/g1/{arm_sdk,sim_arm_sdk}.json`, and the arm_sdk cases in `test/fixtures/lowcmd_crc.json`.
  - Tests: `test/test_{config,robot_profile,lowcmd,safety_core_startup,safety_core_relay}.cpp`, `test/{test_node_ros,test_limit_parity,test_robot_data,conftest}.py`.
- `golem_upper_ik`:
  - Delete: `config/g1/g1.yaml`, `config/g1/sim.yaml`.
  - Modify: `golem_upper_ik/{config,profile,pairing}.py`, `core/{follow_gate,low_cmd_handler,ros_io}.py`, `robots/g1.py` (comment), `README.md`.
  - Tests: `test/*` that name arm_sdk, and `test/fixtures/old_controller/{g1,g1_pairing}.json` (drop the deleted configs' entries).
- `g1_description`: `g1_description/layout.py` (`ARM_SDK_SLOTS`, `ARM_SDK_WEIGHT_SLOT`, the `rt/arm_sdk` text), `test/test_joints.py`, `README.md`.
- `golem_bringup`: `launch/g1/g1_core.launch.py` (defaults), `launch/g1/g1_real_robot.launch.py`, `launch/g1/g1_sim.launch.py`, and the G1 launch tests.

**Interfaces:**
- Produces the G1 profile `modes: [split_mode, full_body_mode]`, with no `arm_sdk_weight_slot`.
- Produces `golem_upper_ik.pairing`:
  - `G1_UPPER_CHANNEL_SAFETY_CONFIGS = ('split', 'sim_split')`
  - `G1_CONTROLLER_CONFIGS = ('g1_split', 'sim_split')`
  - `G1_SAFETY_CONTROLLER_CONFIGS = {'split': 'g1_split', 'sim_split': 'sim_split'}`
  - `G1_LOCOMOTION_*` unchanged until Task 8.
- `Profile` loses `arm_sdk_weight_slot`.
- The G1 bringups default to `split` / `sim_split` (controllers `g1_split` / `sim_split`). `use_locomotion` stays until Task 8.

- [ ] **Step 1: Write the failing tests.**
  - gtest `Config.ArmSdkModeRefused`: a G1 config with `mode: arm_sdk_mode` is refused, and the message names the mode and the G1's modes.
  - gtest `Config.ArmSdkKeysRefused`: a config with an `arm_sdk:` section is refused as a removed key, with the reason `arm_sdk mode was removed: the G1 runs split_mode or full_body_mode`.
  - gtest `RobotProfile.G1HasNoArmSdk`: modes == {split, full}, and there is no weight slot.
  - pytest `test_robot_data.py::test_g1_profile_has_no_arm_sdk`.
  - pytest `test_pairing.py::test_g1_arm_sdk_configs_refused`: `pairing_refusal('g1', 'g1', 'arm_sdk', sim=False)` refuses.
  - pytest `test_pairing.py::test_g1_default_pairs`: `default_controller_config('g1', 'split') == 'g1_split'` and `default_controller_config('g1', 'sim_split') == 'sim_split'`.
  - pytest `test_config.py::test_arm_sdk_output_refused`: a controller config whose `topics.safety_out` is `/arm_sdk` raises `ValueError` naming the topic.
- [ ] **Step 2: Run them to make sure they fail.** `CT golem_safety golem_upper_ik`.
- [ ] **Step 3: Implement the removal.**
  - Remove the mode and everything the spec's section 4 lists. The generator writes `upper_slots` from `layout.WAIST_SLOTS.start` to `layout.RIGHT_ARM_SLOTS[-1]`, with no weight slot.
  - Delete the arm_sdk tests named in the files list (`Startup.ArmSdkEngageRamp`, `Startup.WaistKpBelowMinLatches`, `Quiet.ArmSdkUpperHeldAtFullWeight`, `Exit.ArmSdkRampsThenStops`, and the rest that exercise the mode).
  - Rewrite every README paragraph that describes arm_sdk mode.
- [ ] **Step 4: Verify.**
  - `CT golem_safety golem_upper_ik g1_description golem_bringup g1_locomotion`: PASS.
  - `git grep -niE 'arm_sdk|arm sdk' -- core_ws/src`: only the loader's removed-key message and its test.
- [ ] **Step 5:** Append to the change report: arm_sdk mode removed (G1; real and sim defaults now `split` / `sim_split`); G1 default bringups do not run end to end until Task 9.
- [ ] **Step 6: Commit** `Delete arm_sdk mode: the G1 runs split or full-body mode only`.

### Task 5: The G1 gets the H1-2's start-up; `engage_gate` is deleted

**Files:**
- Modify: `core_ws/src/golem_safety/include/golem_safety/{config,safety_core}.hpp`, `src/{config,safety_core,safety_node,dump_config_main}.cpp`, `config/g1/{split,sim_split,full}.yaml`, `config/h12/*.yaml` (the `startup:` section only), `README.md`
- Modify: `core_ws/src/golem_upper_ik/golem_upper_ik/ros2/frame_task_server.py` (the start-up timeout's list of causes), `README.md`; `core_ws/src/golem_bringup/launch/g1/g1_core.launch.py` docstring
- Tests: `golem_safety/test/test_{config,safety_core_startup,safety_core_relay}.cpp`, `test/test_node_ros.py`; `core_ws/src/g1/g1_locomotion/test/test_locomotion_dds.py`

**Interfaces:**
- There is one start-up path for both robots: zero-gain stream from boot, each input engaging on its own first valid command, and limit checks from boot.
- The `startup` and `engage` config sections become removed keys.
- `StartupRule` and `estop_latched_silent` are gone.

- [ ] **Step 1: Write the failing tests.**
  - gtest `Startup.G1StreamsZeroFromFirstStep`: G1 `sim_split`. The first `step()` with no `/lowstate` writes `zero_command` on all 35 slots, with `mode_machine` 5.
  - gtest `Startup.G1EngagesEachInputOnItsOwn`: a G1 upper command alone engages the upper slots, and the leg slots stay at zero gain until the lower input's first valid command.
  - gtest `Startup.G1LimitChecksFromBoot`: before any input, a G1 `/lowstate` with a knee past `limits.estop` latches. `keep_writing` then writes the estop command every cycle.
  - gtest `Config.StartupAndEngageKeysRefused`: each section is refused as a removed key, and the reason says every robot streams zero gains from boot.
  - pytest `test_node_ros.py::test_g1_streams_lowcmd_from_boot`: a G1 `safety_node` publishes `/lowcmd` at about 500 Hz with no input and no `/lowstate`.
- [ ] **Step 2: Run them to make sure they fail.** `CT golem_safety`.
- [ ] **Step 3:** Delete `engage_gate` and its keys, reasons and tests (`Startup.EngageGate*`). Remove the `startup:` section from every config. Rewrite the G1 config headers to describe the zero-gain stream. The H1-2 tests other than the deleted ones stay unchanged and must pass. `frame_task_server`'s start-up error no longer lists the engage gate. Adapt `test_locomotion_dds.py` to a safety node that streams from boot.
- [ ] **Step 4: Verify.** `CT golem_safety golem_upper_ik g1_locomotion golem_bringup`: PASS.
- [ ] **Step 5:** Append to the change report: G1 start-up is now zero-gain from boot; limit checks start at boot; logs `estop_latched_silent` and `engage_refused` reasons about position distance are gone.
- [ ] **Step 6: Commit** `golem_safety: one start-up for both robots, a zero-gain stream from boot`.

### Task 6: The G1 sim: the band, no stand-in, no HOMIE

**Files:**
- Delete: `golem_robocasa/g1/stand_in.py`
- Modify:
  - `golem_robocasa/g1/g1_mujoco.py`: no stand-in, no `_settle_stance`; the band; the viewer key callback and band drawing; `MeasurementBridge(env, elastic_band=band, task_name=task)`; logs.
  - `golem_robocasa/g1/unitree_interface.py`: no stand-in, no `rt/arm_sdk`; motors off until the first valid `rt/lowcmd`.
  - `golem_robocasa/g1/g1_robosuite.py`: its own joint dynamics and stance.
  - `golem_robocasa/mujoco_env.py`: gains `draw_band(handle, anchor, body_pos)`, moved from `h12_mujoco._draw_band`.
  - `golem_robocasa/h12_mujoco.py`: calls `draw_band`; no other change.
  - `docker/docker-compose.g1.yml`: the sim service loses the policy mount and `GOLEM_ARM_SDK_TIMEOUT`.
  - `docker/scripts/docker_common.sh`: the policy check applies to the `ros` service only.
  - `tests/robocasa/surface/g1.json`: re-recorded.
- Tests: `tests/robocasa/g1/test_g1_unitree_interface.py`, `tests/robocasa/g1/test_g1_band.py` (new), `tests/robocasa/g1/test_g1_spawn_collision.py`

**Interfaces:**
- `SimInterface(model, data, resolver)`: no stand-in argument.
- `SimInterface.commanded: bool` is True from the first valid `rt/lowcmd`. It replaces `released`.
- `g1_robosuite.JOINT_DAMPING = 0.001`, `JOINT_ARMATURE = 0.01`, `JOINT_FRICTIONLOSS = 0.1`.
- `g1_robosuite.STANCE`: the legs at Unitree's default angles (Global Constraints), every other joint 0.
- `sim_loop` builds `band = ElasticBand(anchor, length=0, stiffness=1e4, damping=2e3)` on `torso_link`.

- [ ] **Step 1: Write the failing tests.**
  - `test_g1_unitree_interface.py`:
    - `test_motors_off_until_first_valid_lowcmd`: zero ctrl before any frame.
    - `test_first_valid_lowcmd_drives_motors`.
    - `test_invalid_lowcmd_ignored`: bad CRC, `mode_machine` 6, `mode_pr` 1.
    - `test_lowcmd_timeout_turns_motors_off`.
    - `test_no_arm_sdk_subscription`.
  - `test_g1_band.py` builds the G1 env headless, as the spawn-collision test does:
    - `test_band_holds_limp_robot`: 3 s of physics with no command. The `torso_link` z stays within 0.05 m of the anchor, and the pelvis tilt stays ≤ 15°.
    - `test_release_drops_limp_robot`: `band.toggle()`, then 1.5 s later the pelvis z is at least 0.15 m lower.
    - `test_sim_reads_no_policy`: importing `g1.g1_robosuite` in a subprocess with neither `/home/code/g1_locomotion` nor `core_ws/` on any path succeeds.
- [ ] **Step 2: Run them to make sure they fail.** `tests/robocasa/g1/run_in_container.sh`.
- [ ] **Step 3: Implement** the spec's section 6.
  - Cite `unitree_rl_gym` (file paths and BSD-3) in the comment beside the constants.
  - The `ROBOT FELL` line keeps its prefix, and its "since" text gives the time since the band was released.
  - The sim's ready line no longer lists `rt/arm_sdk`.
  - The spawn places the hanging robot without penetrating the floor or the kitchen (the spawn-collision test).
- [ ] **Step 4: Verify.**
  - `tests/robocasa/g1/run_in_container.sh` and `tests/robocasa/run_in_container.sh`: PASS.
  - `tests/robocasa/surface/run.sh h12`: PASS, unchanged.
  - `tests/robocasa/surface/run.sh g1 --record`. `git diff tests/robocasa/surface/g1.json` shows only `/arm_sdk` removed and `/elastic_band/toggle` (`std_srvs/srv/Trigger`) added; any other difference is reported to the controller before committing.
  - `python3 -m unittest discover -s tests/docker`: PASS.
- [ ] **Step 5:** Append to the change report: the G1 sim hangs from the band until `/elastic_band/toggle` (SPACE in the viewer); nothing stands it up any more; it no longer consumes `rt/arm_sdk`; `GOLEM_ARM_SDK_TIMEOUT` is gone; the sim container no longer mounts HOMIE's policy.
- [ ] **Step 6: Commit** `golem_robocasa: the G1 hangs from the elastic band, as the H1-2 does`.

### Task 7: `g1_locomotion` takes the lower-body controller's role

**Files:**
- Modify: `core_ws/src/g1/g1_locomotion/g1_locomotion/{locomotion,locomotion_node,config,driver}.py`, `config/sim.yaml`, `config/g1.yaml`, `README.md`
- Create: `core_ws/src/g1/g1_locomotion/g1_locomotion/band.py`
- Test: `test/test_locomotion.py`, `test/test_band.py` (new), `test/test_locomotion_node.py`, `test/test_locomotion_dds.py`

**Interfaces:**
- Config keys, both files, all required:
  - `startup.ramp_seconds: 2.0`
  - `startup.wait_for_confirm`: `false` in sim, `true` in g1
  - `heartbeat.timeout: 0.5`
  - `band.release`: `true` in sim, `false` in g1
  - `band.max_wait: 30.0`
- `locomotion.HOLDING = 'holding'`, between `RAMPING` and `WALKING`.
- `LocomotionController.on_heartbeat(self, beat: bool, now: float) -> None`.
- `LocomotionController.confirm(self) -> tuple[bool, str]`: accepted only in `HOLDING`.
- `LocomotionController.walking_since -> float | None`: monotonic time of entering `WALKING`.
- `band.BandRelease(enabled: bool, max_wait: float)`:
  - `.poll(now: float, walking_since: float | None, frame_task_ready: bool, service_ready: bool) -> str | None` returns the reason when a call must be sent now, and marks it in flight.
  - `.result(success: bool) -> None`.
  - `.released: bool`.
- Node: subscribes `/safety/heartbeat` (`std_msgs/Bool`) and `/left_ee_pose` (`geometry_msgs/PoseStamped`); serves `/lowerbody/confirm_engage` (`std_srvs/Trigger`); calls `/elastic_band/toggle` (`std_srvs/Trigger`) from a 10 Hz timer through `BandRelease`.

- [ ] **Step 1: Write the failing tests** (`test_locomotion.py` drives the controller with injected time, as its tests do now).
  - `test_waits_for_first_heartbeat`: steady upright state and a fresh arm command, but no beat. The phase stays `WAITING` with no command. After `on_heartbeat(True)` it is `RAMPING`.
  - `test_false_first_beat_faults`.
  - `test_heartbeat_lost_in_every_phase_faults`: for each of `RAMPING`, `HOLDING`, `WALKING`, either `on_heartbeat(False)` or 0.51 s with no beat (`check_stale`) gives `FAULTED`, and `command()` stays the last one with no new publish. Beats every 0.1 s keep it running.
  - `test_ramp_takes_two_seconds`: with each shipped config, the targets reach the default pose after 100 policy steps (2.0 s), not before.
  - `test_holds_until_confirm`: `wait_for_confirm: true`. After the ramp the phase is `HOLDING`, the targets are the default pose and the policy is not stepped. `confirm()` returns `(True, ...)` and the phase becomes `WALKING`.
  - `test_confirm_refused_outside_holding`: in `WAITING`, `confirm()` returns `(False, msg)` with the phase name in `msg`.
  - `test_sim_config_does_not_hold`: `WALKING` follows the ramp directly.
  - `test_band.py`:
    - `test_band_release_waits_for_walking`.
    - `test_band_release_waits_for_frame_task`: no call until `frame_task_ready`, unless `max_wait` has passed since `walking_since`, in which case the reason names the maximum wait.
    - `test_band_release_one_call_in_flight`.
    - `test_band_release_retries_after_failure`.
    - `test_band_release_never_after_success`.
    - `test_band_release_disabled_never_calls`.
  - `test_configs`: the shipped values above. A missing new key is refused, naming the key.
  - `test_locomotion_node.py::test_confirm_service_and_band_client`: the node serves `/lowerbody/confirm_engage`. With the controller forced to `WALKING`, a fake `/elastic_band/toggle` and one `/left_ee_pose`, the fake service is called exactly once.
- [ ] **Step 2: Run them to make sure they fail.** `CT g1_locomotion`.
- [ ] **Step 3: Implement.**
  - The heartbeat watchdog runs in `check_stale`.
  - The hold publishes the default-pose targets every policy step, with the timing faults active.
  - `test_locomotion_dds.py` runs against the real `safety_node`, whose heartbeat now gates the start.
  - Rewrite the README's Behavior section to match.
- [ ] **Step 4: Verify.** `CT g1_locomotion golem_bringup`: PASS.
- [ ] **Step 5:** Append to the change report: the G1 leg node waits for the heartbeat, pre-poses over 2 s (was 0.5 s sim, 3.0 s real), waits for `/lowerbody/confirm_engage` on the real robot, and releases the sim's band.
- [ ] **Step 6: Commit** `g1_locomotion: heartbeat, pre-pose, operator confirm and band release, as the H1-2's leg node`.

### Task 8: The G1 bringups always run HOMIE

**Files:**
- Modify: `core_ws/src/golem_bringup/launch/g1/{g1_core,g1_sim,g1_real_robot}.launch.py`, `core_ws/src/golem_upper_ik/golem_upper_ik/pairing.py`, `README.md`
- Test: `core_ws/src/golem_bringup/test/test_locomotion_launch.py` (rename to `test_g1_launch.py`), `test_nav_launch.py`, `core_ws/src/golem_upper_ik/test/test_pairing.py`

**Interfaces:**
- `startup_refusal(sim, safety_config, controller_config, locomotion_config, use_nav=False, odom_source='fastlio') -> str | None`: `use_locomotion` and `start_position_verified` are no longer parameters.
- `pairing_refusal(robot, controller_config, safety_config, sim=None) -> str | None`: `locomotion` is gone.
- `G1_LOCOMOTION_SAFETY_CONFIGS` and `G1_LOCOMOTION_CONTROLLER_CONFIGS` are gone.

- [ ] **Step 1: Write the failing tests.**
  - `test_g1_launch.py::test_sim_always_starts_locomotion`: the sim core's actions include `locomotion_node` with `sim.yaml`.
  - `test_real_starts_locomotion_only_when_verified`: without `start_position_verified:=true` there is no `locomotion_node`, and the arms and safety start. With it, `locomotion_node` starts with `g1.yaml`.
  - `test_use_locomotion_not_declared`: neither top-level file declares `use_locomotion`.
  - `test_real_refuses_sim_configs`.
  - `test_sim_refuses_real_configs`.
  - `test_nav_launch.py::test_sim_nav_without_use_locomotion`.
  - `test_real_refuses_nav`.
  - `test_pairing.py::test_g1_pairs`: `split`↔`g1_split`, `sim_split`↔`sim_split`, and sim/real mixing is refused.
- [ ] **Step 2: Run them to make sure they fail.** `CT golem_bringup golem_upper_ik`.
- [ ] **Step 3: Implement** the spec's section 8. The real file's docstring states:
  - the operator puts the G1 into debug mode, on a gantry, before launching;
  - `start_position_verified` gates the legs;
  - `/lowerbody/confirm_engage` starts the policy.
- [ ] **Step 4: Verify.**
  - `CT golem_bringup golem_upper_ik`: PASS.
  - `ros2 launch golem_bringup g1_sim.launch.py --show-args` and `ros2 launch golem_bringup g1_real_robot.launch.py --show-args` exit 0.
- [ ] **Step 5:** Append to the change report: `use_locomotion` is removed; the G1 sim always walks-ready; the real G1's legs need `start_position_verified:=true` and the confirm.
- [ ] **Step 6: Commit** `golem_bringup: the G1 bringups always start HOMIE`.

### Task 9: Acceptance tests and the whole-stack regression

**Files:**
- Modify: `tests/g1/{common.sh,test_sim_bringup.py,test_sim_bringup.sh,test_sim_nav.py,test_sim_nav.sh,test_sim_explore.py,test_sim_explore.sh,README.md}`

**Interfaces:**
- Consumes `SimInterface.commanded` (Task 6) and the band object from `sim_loop`; record its `enabled` state every step.
- Consumes the `locomotion_node` log line `walking: policy started` (Task 7).

- [ ] **Step 1: Rewrite the arm test's checks.**
  - **Remove:** `engage` (weight ramp), `hand back`, `stand-in kept`, and the walk test's `hand-over`.
  - **Add:**
    - `band holds`: while the band is on, pelvis tilt ≤ `MAX_TILT_DEG`;
    - `no estop during start-up`: no estop event in `golem_safety`'s log before the test's own stop step;
    - `walking node starts`;
    - `band released after the controller is ready`: the band turns off after the first `/left_ee_pose` the test node sees, and within `band.max_wait` of the policy's start;
    - `standing after release`: 5 s after the release, `standing` holds.
  - **Keep:** `home at startup` (without the weight term), `t_pose`, `left hand close`, `waist held after the controller stops`, `standing`, and the sensor checks.
  - The nav and explore tests drop `use_locomotion:=true` and wait for the band's release before their first goal.
  - Update the README's description of each test.
- [ ] **Step 2: Run** the G1 suites:
  - `tests/g1/test_sim_bringup.sh`;
  - `tests/g1/test_sim_bringup.sh --walk`;
  - `tests/g1/test_sim_nav.sh`;
  - `GOLEM_SIM_ODOM=1 tests/g1/test_sim_nav.sh`;
  - `tests/g1/test_sim_explore.sh`.

  Each must end `PASSED`. Expected flakes: the documented `t_pose` threshold flake; G1 `home` goals fail in the sim (pre-existing), and no check sends one. Any other failure goes to the controller with the run directory.
- [ ] **Step 3: H1-2 regression.**
  - `ROS_DOMAIN_ID=193 tests/stack/run.sh robocasa arms` and `... locomotion`: PASS.
  - `tests/robocasa/surface/run.sh h12` and `... g1`: PASS.
  - `python3 -m unittest discover -s tests/docker`: PASS.
  - `CT golem_safety golem_upper_ik golem_bringup g1_locomotion g1_description h12_lowerbody_rl`: PASS.
  - The H1-2 build: PASS.
- [ ] **Step 4: Commit** `tests/g1: the G1 starts from the band, HOMIE releases it`.

### Task 10: Docs

**Files:**
- Modify: `CLAUDE.md`, `AGENTS.md`, `README.md`, `docs/RUNNING.md`, `docs/DOCKER_TESTING.md`, `docs/SETUP.md`, `docs/g1/*.md`, `core_ws/src/golem_bringup/common_cmds.md`, `tests/README.md`, and any living doc `git grep` finds naming `use_locomotion`, `arm_sdk`, `stand-in`, `engage gate`, `GOLEM_ARM_SDK_TIMEOUT`

- [ ] **Step 1: Write the content.**
  - CLAUDE.md and AGENTS.md:
    - package tables (`golem_bringup`; no old packages);
    - the §3 and §6 commands and bringup tables;
    - the G1 start-up in debug mode (zero-gain stream, heartbeat, HOMIE pre-pose, confirm on the real robot, the band in the sim, always HOMIE);
    - the §8 G1 sim surface and knobs (no `rt/arm_sdk`, the band service, no `GOLEM_ARM_SDK_TIMEOUT`, `golem_robocasa`);
    - the invariants (writers, estop latch triggers and responses without arm_sdk or the engage gate, the controller's exit on the G1 in split mode only);
    - §9 commands;
    - a pitfall for stale installs: delete `core_ws/{build,install}/{h12_bringup,g1_bringup,h12_safety_layer,h12_ros2_controller,g1_safety_layer,g1_control}` once after pulling.
  - `docs/g1/walking.md`: walking needs no launch argument now. The drive-it-yourself flow waits for the band's release.
  - `docs/g1/first-session.md`: rewritten for debug mode:
    - the gantry;
    - the laptop steps;
    - `g1_real_robot.launch.py` with `start_position_verified`;
    - what you should see (the zero-gain stream, the heartbeat, the arms' start-up, the pre-pose, the confirm);
    - how to stop.
    - The page carries a section **"Entering debug mode (not yet written)"**: the procedure for entering debug mode on our G1, confirming it before launching, and when the robot comes off the gantry has not been provided, and nobody runs the first session until it is. Invent no button combination or app step.
- [ ] **Step 2: Verify.**
  - `git grep -nE 'h1_robocasa|h12_bringup|g1_bringup|use_locomotion|arm_sdk|stand-in|engage gate|GOLEM_ARM_SDK_TIMEOUT|h12_safety_layer|h12_ros2_controller|g1_safety_layer|g1_control\b' -- ':!docs/superpowers' ':!mujoco_mpc' ':!CL_Assets' ':!core_ws/src/cl_realsense'` prints only the loader's removed-key message, its test and the fixture READMEs.
  - `docker run --rm -v "$PWD:/repo" -w /repo python:3.11-slim sh -c "pip install -q -r docs/requirements-docs.txt && mkdocs build --strict"`: PASS.
- [ ] **Step 3: Commit** `docs: golem_bringup, golem_robocasa and the G1's debug-mode start-up`.
