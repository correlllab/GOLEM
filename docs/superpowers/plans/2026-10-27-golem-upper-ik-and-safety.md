# golem_upper_ik and golem_safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `h12_safety_layer` + `g1_safety_layer` with one C++ `golem_safety`, and `h12_ros2_controller` + `g1_control` with one Python `golem_upper_ik`. Both talk only ROS 2, and both real robots and all sims run them.

**Architecture:**
- **`golem_safety`** is an rclcpp node. A ROS-free `SafetyCore` holds all decisions, and the node only moves messages, so every rule is unit-testable with injected time.
- **`golem_upper_ik`** is the H1-2 controller's live code with the G1's fixes merged in. Its Unitree-SDK I/O layer is replaced by rclpy.
- **Robot profiles:** each robot's data is defined once, as a YAML file in `golem_safety`'s share directory that both packages read. The controller adds its own per-robot module (URDFs, frames, named configurations).

**Tech Stack:** ROS 2 Humble, rclcpp (C++17), yaml-cpp, gtest (ament_cmake_gtest), rclpy, Pinocchio/Pink, OMPL (via the existing planner subprocess), pytest, the `unitree_hg` ROS messages.

**Spec:** `docs/superpowers/specs/2026-10-27-golem-upper-ik-and-safety-design.md`. Read it first; it is the authority. The audit behind it is in `docs/superpowers/specs/2026-10-27-golem-upper-ik-and-safety/redundancy-audit.md`: section A covers the safety layers and section B the controllers.

## Global Constraints

- **Branch** `golem_upper_ik_safety`. Package directories: `core_ws/src/golem_safety` (ament_cmake) and `core_ws/src/golem_upper_ik` (ament_python), in-tree, not submodules.
- **Old packages:** never edit `core_ws/src/h12_safety_layer`, `h12_ros2_controller`, `g1/g1_safety_layer` or `g1/g1_control`. Read them as the source of behavior. They are deleted later by the user, not in this plan.
- **ROS only:** no `unitree_sdk2py` import in either new package (enforced by a test in Tasks 7 and 11). ROS topic `/x` is DDS `rt/x`; configs name ROS topics (`/lowcmd`, `/safety/lowcmd_upper_in`, …).
- **Where conflicts are settled:**
  - Where the H1-2 and G1 behave differently and the spec does not settle it, the H1-2's behavior wins.
  - Every value or behavior that differs from what a robot does today is appended to the change report, `docs/superpowers/specs/2026-10-27-golem-upper-ik-and-safety/change-report.md`, in the task that makes the change. Each entry gives the robot, the old behavior, the new behavior and the reason.
- **Builds and tests run in containers:** `docker/scripts/docker_run.sh ros --restart --g1 bash -lc '<cmd>'`. G1 mode builds every package, `g1_*` included. Only `tests/docker` runs on the host.
  - Every pytest command starts with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (the image's anyio plugin breaks pytest).
  - `CT <pkg>` below means `export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 && cd /home/code/core_ws && colcon build --packages-select <pkg> && source install/setup.bash && colcon test --packages-select <pkg> --event-handlers console_direct+; colcon test-result --verbose`.
- **Test DDS domains:**
  - `golem_safety` integration tests: `ROS_DOMAIN_ID=91`.
  - `golem_upper_ik` tests: 92.
  - Never 0. Other suites own 97, 187, 193, 194 and 195-198.
- **The H1-2 build must not need `g1_*` or `third_party/leprechaun_g1`.**
  - The H1-2 builds with `--packages-ignore-regex '^g1_'`, and G1 data is imported only when `robot == 'g1'`.
  - Task 13 verifies the build with `docker/scripts/docker_run.sh ros --restart --h12 bash -lc 'true'`.
- **Rates:**
  - The safety relay runs at the config's `control.publish_hz` (500).
  - The controller publishes at `pub_hz` (500) and solves IK at `ctrl_hz` (30).
  - If a rate cannot be held, stop and report. Never lower it.
- **Code style:**
  - Comments describe the code as it is (CLAUDE.md §8): no change narration.
  - New pip dependencies go through `docker/locks`. None are expected.
- **Commits** end with the session's attribution trailer, and land only on this branch.
- **Robot facts** (spec, Robot profiles):
  - `mode_machine`: H1-2 6, G1 5. `mode_pr` 0 on both.
  - `LowCmd`/`LowState` motor arrays hold 35 slots.
  - H1-2: 27 body joints in slots 0-26; lower slots 0-11, upper slots 12-26; modes `full_body_mode` and `split_mode`.
  - G1: 29 body joints in slots 0-28; lower slots 0-11, upper slots 12-28; modes `arm_sdk_mode`, `split_mode` and `full_body_mode`; the arm_sdk weight is in slot 29.
- **H1-2 joint limits:** copied from `h12_safety_layer/h12_safety_layer/core/joint_limits.py`, which is identical to `h12_ros2_controller/utility/joint_limits.py`. They are not taken from the URDF: shoulder roll is ±0.38 in the table against ±0.19 in `h1_2_magpie.urdf`, and the table is what the robot runs.

## Review Focus

These five input classes are the most likely to bite in use. Each is pinned by a test in the task named:

1. **A controller is restarted while the robot is up.**
   - The new process is a new DDS writer on the same input.
   - Two writers that are live at once must latch the estop, as the G1 does today.
   - A new writer that appears after the old one has been quiet for `single_writer.handover_seconds` (default 0.5 s, both robots) is accepted and logged, as the H1-2 does today.
   - Tests: `SingleWriter.RestartAfterQuietIsAccepted` and `SingleWriter.TwoLiveWritersLatch` (Task 5).
2. **`/lowstate` absent, or from the other robot.** The node starts before the sim, sits on the wrong domain, or a second robot or sim shares the domain.
   - No input engages without a `/lowstate` carrying the profile's `mode_machine`.
   - `zero_stream` keeps streaming zero gains; `engage_gate` stays silent.
   - Either logs why, at most once per second.
   - After engaging, a foreign `mode_machine` latches the estop.
   - Tests: `Startup.NoLowstateNeverEngages` and `Identity.ForeignSampleAfterEngageLatches` (Tasks 5-6).
3. **Non-finite command values.** A broken controller sends NaN or inf in `q`, `dq`, `tau`, `kp` or `kd`.
   - The sample is dropped, the input's last good command keeps being relayed, and it is logged.
   - Before engaging, the sample does not count as the input's first command.
   - Test: `Clip.NonFiniteSampleIsDroppedAndHeld` (Task 5).
4. **A config that does not fit the robot.** For example, `robot:=h12` with `arm_sdk.yaml`, a G1 config path passed to the H1-2, or a mode the profile lacks.
   - The node exits non-zero at start with a message naming the robot, the config and the reason.
   - Test: `test_config_for_other_robot_is_refused` (Task 7).
5. **The hardware estop.** The estop node publishes "triggered" or "unplugged" in its first sample, or dies after engage.
   - The first sample latches at once if it is triggered or unplugged.
   - A sample older than `estop.timeout_seconds` (0.5 s) latches.
   - Without `use_estop`, the estop topics are ignored entirely.
   - Tests: `Estop.FirstSampleTriggeredLatches`, `Estop.StaleAfterEngageLatches` and `Estop.IgnoredWithoutUseEstop` (Task 6).

---

## Phase 0: The rate gate

### Task 1: `golem_upper_ik` skeleton and the 500 Hz rclpy publish gate

Python `rclpy` must publish `unitree_hg/msg/LowCmd` at 500 Hz while IK runs in the same process. If it can't, the spec says stop.

**Files:**
- Create: `core_ws/src/golem_upper_ik/{package.xml,setup.py,setup.cfg,resource/golem_upper_ik,golem_upper_ik/__init__.py,README.md}`.
  - `package.xml`: `exec_depend` on rclpy, unitree_hg, custom_ros_messages, std_msgs, geometry_msgs, sensor_msgs, h12_ros2_model, golem_safety, ament_index_python, `python3-yaml`.
  - No `g1_*` dependency (Global Constraints).
- Create: `core_ws/src/golem_upper_ik/tools/publish_rate_probe.py`. It runs a node with a background thread publishing a preallocated `LowCmd` on `/golem_probe/lowcmd` at a target rate, while the main thread burns CPU in a loop sized like one 30 Hz IK step (numpy matrix work, about 15 ms per 33 ms). A second node in the same process subscribes and records receive times.
  - Args: `--rate 500 --seconds 10`.
  - Prints JSON: `{"target_hz", "mean_hz", "p99_period_ms", "max_gap_ms"}`.
- Test: `core_ws/src/golem_upper_ik/test/test_publish_rate.py`.

**Interfaces:**
- Produces: package `golem_upper_ik` (built for both robots) and the probe tool.

- [ ] **Step 1: Write the test**
  `test_lowcmd_publish_holds_500hz` runs the probe as a subprocess with `ROS_DOMAIN_ID=92` for 10 s and asserts:
  - `mean_hz >= 495`;
  - `p99_period_ms <= 4.0`;
  - `max_gap_ms <= 20.0`.
- [ ] **Step 2: Run it** (`CT golem_upper_ik`, G1-mode container). Expected: FAIL, because the probe does not exist yet.
- [ ] **Step 3: Write the skeleton and the probe.** Publish from a plain `threading.Thread` with a monotonic deadline loop (`time.sleep(max(0, next - now))`), mutating one preallocated message.
- [ ] **Step 4: Run it.** Expected: PASS. Record the printed JSON in the task report.
  **If it fails, stop and report to the user with the numbers. Do not continue the plan.**
- [ ] **Step 5: Commit** `golem_upper_ik: package skeleton and the 500 Hz LowCmd publish probe`.

---

## Phase A: `golem_safety` (C++)

### Task 2: Package skeleton, robot data files, `RobotProfile`

**Files:**
- Create: `core_ws/src/golem_safety/CMakeLists.txt` (C++17; `find_package` rclcpp, std_msgs, unitree_hg, yaml-cpp, ament_index_cpp; tests under `if(BUILD_TESTING)` with ament_cmake_gtest and ament_cmake_pytest) and `package.xml`.
- Create: `core_ws/src/golem_safety/robots/h12.yaml` and `robots/g1.yaml`, installed to `share/golem_safety/robots/`. Schema:
  ```yaml
  robot: g1
  mode_machine: 5
  mode_pr: 0
  motor_slots: 35
  modes: [arm_sdk_mode, split_mode, full_body_mode]
  lower_slots: [0, 11]        # inclusive
  upper_slots: [12, 28]
  arm_sdk_weight_slot: 29     # g1 only
  joints:                     # body joints, slot order
    - {name: left_hip_pitch_joint, slot: 0, position: [low, high], velocity: v, effort: e}   # values from the sources below
  ```
  - **G1 values** come from `g1_description.joints.BODY`.
  - **H1-2 values** come from `h12_safety_layer/h12_safety_layer/core/joint_limits.py` (`JOINT_NAMES`, `URDF_POSITION_LIMITS`, `URDF_VELOCITY_LIMITS`, `URDF_TORQUE_LIMITS`).
  - Write a one-off generator `tools/generate_robot_data.py`, kept in the package, that builds both files from those two sources.
- Create: `include/golem_safety/robot_profile.hpp` and `src/robot_profile.cpp`.
- Test:
  - `test/test_robot_profile.cpp` (gtest);
  - `test/test_robot_data.py` (pytest);
  - `test/fixtures/h12_joint_limits.json`, a snapshot of today's H1-2 tables written by the generator.

**Interfaces:**
- Produces (namespace `golem_safety`):
  - `struct JointLimit { std::string name; int slot; double q_low, q_high, dq_max, tau_max; };`
  - `struct SlotRange { int first, last; bool contains(int) const; };`
  - `enum class Mode { FullBody, Split, ArmSdk };`, plus `Mode parse_mode(const std::string&)`, which accepts `full_body_mode|split_mode|arm_sdk_mode` and throws otherwise.
  - The profile struct:

    ```cpp
    struct RobotProfile {
      std::string robot; uint8_t mode_machine; uint8_t mode_pr; int motor_slots;
      std::vector<JointLimit> joints; std::vector<Mode> modes;
      SlotRange lower_slots, upper_slots; std::optional<int> arm_sdk_weight_slot;
    };
    ```
  - `RobotProfile load_robot_profile(const std::string& robot)`: reads `share/golem_safety/robots/<robot>.yaml` and throws `std::runtime_error` naming the robot when the file is missing.
  - The installed YAML files are read by `golem_upper_ik` (Task 8).

- [ ] **Step 1: Write the tests.**
  - gtest `RobotProfile.LoadsH12`: `mode_machine == 6`, `joints.size() == 27`, `upper_slots == {12, 26}`, no `arm_sdk_weight_slot`, `modes == {FullBody, Split}`.
  - gtest `RobotProfile.LoadsG1`: `mode_machine == 5`, 29 joints, `arm_sdk_weight_slot == 29`, `upper_slots == {12, 28}`.
  - gtest `RobotProfile.UnknownRobotThrows`.
  - pytest `test_g1_data_matches_g1_description` compares names, slots and limits with `g1_description.joints.BODY`, skipping when `g1_description` is not importable.
  - pytest `test_h12_data_matches_todays_table` compares against the fixture.
- [ ] **Step 2: Run** `CT golem_safety`. Expected: build or test FAIL.
- [ ] **Step 3: Implement** the generator, both YAML files, the fixture and the loader (yaml-cpp).
- [ ] **Step 4: Run** `CT golem_safety`. Expected: all PASS.
- [ ] **Step 5: Commit** `golem_safety: package, robot data files and RobotProfile`.

### Task 3: `SafetyConfig`, derived limits and the carried-over configs

**Files:**
- Create: `include/golem_safety/config.hpp`, `src/config.cpp`, and `src/dump_config_main.cpp` (executable `golem_safety_dump_config`).
- Create: `config/h12/{default_safety_full,default_safety_split,locomotion_safety_full,relax_safety_full,relax_safety_split,sim_safety_split,tight_safety_full,tight_safety_split}.yaml` and `config/g1/{arm_sdk,full,sim_arm_sdk,sim_split,split}.yaml`, installed to `share/golem_safety/config/<robot>/`.
- Create: `tools/snapshot_old_limits.py`. It runs in the container and writes `test/fixtures/limits/<robot>/<config>.json` from the old loaders: `h12_safety_layer.core.config` and `g1_safety_layer.core.config`.
- Test: `test/test_config.cpp` and `test/test_limit_parity.py`.

**Unified schema.** Port every key the two old loaders read (`h12_safety_layer/core/config.py`, `g1_safety_layer/core/config.py`), with these changes. Each change goes in the change report.

```yaml
mode: split_mode                       # full_body_mode | split_mode | arm_sdk_mode
topics: {low_cmd_in: /safety/lowcmd_in, low_cmd_lower_in: /safety/lowcmd_lower_in,
         low_cmd_upper_in: /safety/lowcmd_upper_in, low_cmd_out: /lowcmd, low_state: /lowstate}
control: {publish_hz: 500.0, state_lag_ms: 100.0}      # state_lag_ms was estop.state_lag_ms (H1-2)
stale: {upper_seconds: 0.0, lower_seconds: 0.2, state_seconds: 0.1}   # report-only; <= 0 disables the report
single_writer: {handover_seconds: 0.5}
startup: {rule: zero_stream}           # zero_stream (every h12 config) | engage_gate (every g1 config)
engage: {max_state_age_seconds: 0.1, max_position_error: 0.15}        # engage_gate only
arm_sdk: {engage_seconds: 1.0, release_seconds: 3.0, waist_kp_min: 40.0}  # arm_sdk_mode only
estop: {triggered_topic: /estop/triggered, plugged_in_topic: /estop/plugged_in, timeout_seconds: 0.5}
estop_response: {mode: zero, kd: 0.0, after: burst_then_silence, burst_count: 50}  # after: burst_then_silence | keep_writing
limits: {clip: {...}, estop: {...}}    # unchanged, incl. the kp_abs/kd_abs/enforce aliases
logging: {enabled: false, base_dir: /tmp/golem_safety, chunk_prefix: chunk_}
```

**Changes from today's schema:**
- **Removed:** `network.*` (ROS supplies the domain; the G1 interface pin is `cyclonedds_local.xml`), `robot: real|sim` (the bringup's `sim`), `engage.mode_machine` (now the profile's), `estop.enabled`/`estop_topic`/`poll_hz` (the bringup's `use_estop`).
- **Renamed:** `estop.upper_stale_seconds` becomes `stale.upper_seconds`. `control.lower_stale_estop_seconds` and `control.state_stale_estop_seconds` become `stale.lower_seconds` and `stale.state_seconds`.
- **Topic values** change from `rt/x` to `/x`, the same DDS topic.
- **New-key values** in the carried configs:
  - every H1-2 config: `startup.rule: zero_stream`, `estop_response.after: burst_then_silence`, `burst_count: 50`;
  - every G1 config: `engage_gate` and `keep_writing`, with its `estop_response.mode` and `kd`;
  - stale times without an old key: 0.2 (lower) and 0.1 (state);
  - `logging.base_dir` keeps each file's current value.

**Interfaces:**
- Consumes: `RobotProfile`, `Mode` (Task 2).
- Produces:
  - `struct DerivedLimits { std::vector<double> q_clip_low, q_clip_high, dq_clip, tau_clip, q_estop_low, q_estop_high, dq_estop, tau_estop; double kp_max, kd_max; };` Vectors are indexed by body slot. Derivation is exactly as `_derive_q_limits` / `_process_clip_limits` / `_process_estop_limits` today: scalar or per-joint lists, the same defaults 0.01/0.5/0.5, and kp/kd max 300/30.
  - `enum class StartupRule { ZeroStream, EngageGate };` and `enum class EstopAfter { BurstThenSilence, KeepWriting };`
  - `struct SafetyConfig` with one field per schema key (topics as `std::string`, times as `double` seconds, `Mode mode`, `DerivedLimits limits`, …).
  - `SafetyConfig load_safety_config(const std::string& name_or_path, const RobotProfile& profile)`:
    - a bare name (with or without `.yaml`) resolves against `share/golem_safety/config/<profile.robot>/`;
    - it throws `std::runtime_error` on unknown keys, a mode the profile lacks, or per-joint lists whose length is not the profile's joint count;
    - the message names the robot, the file and the key.
  - `golem_safety_dump_config <robot> <config>` prints `DerivedLimits` and the parsed scalars as JSON.

- [ ] **Step 1: Write the tests.**
  - gtest `Config.ResolvesBareNameUnderRobotDir`, `Config.UnknownKeyThrows`, `Config.ModeNotInProfileThrows` (h12 + `arm_sdk_mode`), `Config.ListLengthMustMatchJoints`, and `Config.H12DefaultsAreZeroStreamAndBurst`.
  - pytest `test_limits_match_old_loaders[<robot>/<config>]`, parametrised over all 13 configs: run `golem_safety_dump_config`, then compare every derived array and `kp_max`/`kd_max` to the fixture with `rtol=0, atol=1e-12`.
- [ ] **Step 2: Run** `CT golem_safety`. Expected: FAIL.
- [ ] **Step 3: Generate the fixtures** (`python3 tools/snapshot_old_limits.py`, in the G1-mode container, so both old packages import). Then implement the loader and write the 13 configs, carrying every value and comment that still applies.
- [ ] **Step 4: Run** `CT golem_safety`. Expected: PASS.
- [ ] **Step 5: Change report.** Append the schema changes above.
- [ ] **Step 6: Commit** `golem_safety: config schema, derived limits and the carried-over configs`.

### Task 4: CRC, command builders and the chunk logger

**Files:**
- Create: `include/golem_safety/lowcmd.hpp` and `src/lowcmd.cpp`. The CRC is adapted from `core_ws/src/unitree_ros2/example/src/src/common/motor_crc_hg.cpp` (BSD-3: keep its copyright notice in the file).
- Create: `include/golem_safety/chunk_logger.hpp`, `src/chunk_logger.cpp`, `scripts/combine_chunks.py` (copied from `g1_safety_layer/utility/combine_chunks.py`, with its default dir changed to `/tmp/golem_safety`).
- Create: `tools/crc_fixtures.py`. It runs in the container and uses `unitree_sdk2py.utils.crc.CRC` to write `test/fixtures/lowcmd_crc.json` (5 commands with known field values and their CRCs). This is the only SDK use, and it is test-fixture generation, not package code.
- Test: `test/test_lowcmd.cpp`, `test/test_chunk_logger.cpp` and `test/test_chunk_files.py`.

**Interfaces:**
- Produces:
  - `uint32_t lowcmd_crc(const unitree_hg::msg::LowCmd&)` and `void stamp_header_and_crc(unitree_hg::msg::LowCmd&, const RobotProfile&)`. The second sets `mode_pr`, `mode_machine`, then `crc`.
  - `unitree_hg::msg::LowCmd zero_command(const RobotProfile&)`: every slot mode 0 with q, dq, tau, kp and kd all 0, matching today's H1-2 boot and estop command (`make_estop_cmd(LowCmdDefault())`).
  - `unitree_hg::msg::LowCmd estop_command(const RobotProfile&, const SafetyConfig&, const unitree_hg::msg::LowCmd& last)`: `zero` → `zero_command`; `damp` → kp 0, the config's kd, as `g1_safety_layer/core/safety_checks.py` `make_estop_cmd`.
  - `class ChunkLogger { public: ChunkLogger(std::string base_dir, std::string chunk_prefix, size_t samples_per_chunk); void log(const LogSample&); void stop(); };`
    - It writes on a background thread.
    - Files are `.npz`, stored uncompressed, with the same keys as `g1_safety_layer/core/chunk_logger.py` (`time_stamp, mode, q, dq, ddq, tau_est, temperature, vol, sensor, motorstate, reserve, imu_quaternion, imu_gyroscope, imu_accelerometer, imu_rpy, imu_temperature, q_cmd, dq_cmd, tau_cmd, kp_cmd, kd_cmd`), dtypes and the per-run subdirectory naming.
    - `struct LogSample` has one field per key.

- [ ] **Step 1: Write the tests.**
  - gtest `LowCmd.CrcMatchesSdkFixtures`: all 5 CRCs equal the fixture.
  - gtest `LowCmd.StampSetsProfileHeader`: H1-2 gives `mode_machine == 6`, G1 gives 5, and `mode_pr == 0`.
  - gtest `LowCmd.DampKeepsKd`.
  - gtest `ChunkLogger.WritesChunkFiles`: log 2.5 chunks of samples, `stop()`, and 3 files exist.
  - pytest `test_chunks_load_in_numpy`: run a small logger executable, `np.load` each file, check the key set and shapes, and check that `combine_chunks.py` concatenates them.
- [ ] **Step 2: Run** `CT golem_safety`. Expected: FAIL.
- [ ] **Step 3: Implement.** The CRC packs into the `motor_crc_hg.h` struct layout. The npz writer writes the zip "stored" method by hand, with a CRC-32 per entry (zlib's `crc32`, already linked via zlib1g-dev).
- [ ] **Step 4: Run** `CT golem_safety`. Expected: PASS.
- [ ] **Step 5: Commit** `golem_safety: LowCmd CRC and builders, npz chunk logger`.

### Task 5: `SafetyCore`, part 1: merge, clip, limits, identity, writers, the estop latch

`SafetyCore` holds every decision, with no ROS: time is passed in, so tests drive it directly. Port the logic from `h12_safety_layer/core/safety_layer.py` + `safety_checks.py` and `g1_safety_layer/core/safety_layer.py` + `safety_checks.py` + `arm_sdk.py`, applying the spec's golem_safety items 1, 3, 5, 6 and 7.

**Files:**
- Create: `include/golem_safety/safety_core.hpp`, `src/safety_core.cpp` and `test/test_safety_core_relay.cpp`.

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces:
  ```cpp
  enum class Input { Full, Upper, Lower };
  using WriterId = std::string;          // the publisher GID bytes
  struct Event { enum Kind { Info, Warn, Error } kind; std::string text; };
  class SafetyCore {
   public:
    SafetyCore(RobotProfile, SafetyConfig, bool use_estop, int domain_id);
    void on_state(const unitree_hg::msg::LowState&, double now);
    void on_command(Input, const unitree_hg::msg::LowCmd&, const WriterId&, double now);
    void on_estop(bool triggered, bool plugged_in, double now);
    std::optional<unitree_hg::msg::LowCmd> step(double now);       // what to publish this cycle (header + CRC stamped)
    std::optional<unitree_hg::msg::LowCmd> exit_step(double now);  // during shutdown; nullopt = done
    bool estopped() const; const std::string& estop_reason() const;
    std::vector<Event> drain_events();
    std::optional<LogSample> take_log_sample();                     // one per new /lowstate when logging is on
  };
  ```
  - `now` is seconds on the boot-time clock.
  - Inputs per mode: `full_body_mode` uses `Full` (every body slot); `split_mode` uses `Lower` (lower slots) and `Upper` (upper slots); `arm_sdk_mode` uses `Upper` (upper slots).

**Behavior in this task:**
- **Merge.** The output takes each engaged input's newest command on that input's slots (Task 6 decides "engaged"). Slots no input owns get `zero_command`'s values. Header and CRC come from `stamp_header_and_crc`.
- **Clipping.** Clip as today: q to the clip range, `dq`/`tau` to their ratios, and kp/kd to their maxima, logging each clipped gain once per joint and gain.
- **Limit estops.** Latch on a measured `/lowstate` q, dq or tau past the estop range, on the checked slots (Task 6 decides which). The reason names the joint, the value, the limit and the multiple (H1-2 `safety_checks.py`, commit `5182fbe`).
- **Non-finite commands.** A command with any non-finite `q/dq/tau/kp/kd` on its slots is dropped and logged; the input's previous command stays.
- **Identity.** A `/lowstate` whose `mode_machine` is not the profile's latches once the core has engaged. Before that it is only recorded (Task 6 refuses engagement).
- **Single writer.** A command from a writer other than the input's current one:
  - is accepted, and becomes current, if the current writer's last command is older than `single_writer.handover_seconds`;
  - otherwise latches the estop with `second writer on <topic>`.
- **Estop latch.**
  - One-way. It records the reason and the decision time, and adds the timing report fields both old layers log (`action_ms`, `decision_to_action_ms`, `trigger_to_action_ms`).
  - After latching, `step` returns the `estop_command`.
  - With `burst_then_silence`, it returns that command `burst_count` times and then `nullopt` for ever. With `keep_writing`, it returns it every step.
  - `arm_sdk_mode` instead holds the last good command while the slot-29 weight ramps to 0 over `release_seconds`, then holds weight 0 for ever.

- [ ] **Step 1: Write the tests** (gtest, using the H1-2 profile with `sim_safety_split` and the G1 profile with `sim_split`/`sim_arm_sdk`, with fixed `now` values):
  ```
  Relay.SplitMergesUpperAndLowerSlots      q on slot 5 from Lower, slot 20 from Upper, header mode_machine == profile
  Relay.UnownedSlotsAreZeroCommand         full profile slots beyond the body (27..34 on h12) are zero_command
  Clip.PositionVelocityTorqueAndGains      each field clipped to DerivedLimits; kp above kp_max -> kp_max
  Clip.NonFiniteSampleIsDroppedAndHeld     NaN q on an upper slot: output keeps the previous upper command; one Warn event
  Limits.MeasuredPastEstopLatches          state q beyond q_estop_high on a checked slot -> estopped(), reason contains joint name and "x"
  Identity.ForeignSampleAfterEngageLatches engaged, then LowState mode_machine 5 on the h12 profile -> estopped()
  SingleWriter.TwoLiveWritersLatch         writer B within handover_seconds of writer A's last sample -> estopped()
  SingleWriter.RestartAfterQuietIsAccepted A quiet for 0.6 s, then B -> not estopped, B's command relayed, one Info event
  Estop.BurstThenSilence                   h12: after latch, exactly 50 estop commands, then nullopt
  Estop.KeepWriting                        g1 split: estop command every step for 1000 steps
  Estop.ArmSdkReleaseRamp                  g1 arm_sdk: weight slot 29 falls linearly to 0 over release_seconds, then stays 0
  ```
- [ ] **Step 2: Run** `CT golem_safety`. Expected: FAIL.
- [ ] **Step 3: Implement.** Engagement is stubbed so that every input that has sent a valid command counts as engaged; Task 6 replaces this.
- [ ] **Step 4: Run** `CT golem_safety`. Expected: PASS.
- [ ] **Step 5: Change report.** Add: single-writer handover (both robots); non-finite drop (the H1-2 relayed NaN; the G1 refused only at engage); identity latch (new on the H1-2).
- [ ] **Step 6: Commit** `golem_safety: SafetyCore relay, clipping, limits, identity, writers and the estop latch`.

### Task 6: `SafetyCore`, part 2: start-up rules, the estop input, quiet inputs, lag, arm_sdk, exit

Apply the spec's golem_safety items 2, 4, 8 and 9 and the estop parts of item 10.

**Files:**
- Modify: `src/safety_core.cpp` and `include/golem_safety/safety_core.hpp`.
- Test: `test/test_safety_core_startup.cpp`.

**Behavior:**
- **Prerequisite for any engagement:** a `/lowstate` carrying the profile's `mode_machine` has been seen. When `use_estop` is set, a first estop sample has also arrived.
- **`zero_stream` (H1-2):**
  - `step` returns `zero_command` from the first call, whether or not `/lowstate` has arrived, exactly as today.
  - Each input engages on its first valid command once the prerequisites hold.
  - Limit estops check every body slot from the first `/lowstate` on.
- **`engage_gate` (G1):**
  - `step` returns `nullopt` until every input of the mode has a command that passes the gate (port `g1_safety_layer` `engage_refusal`): `/lowstate` age at most `max_state_age_seconds`, and every one of the input's slots within `max_position_error` of measured q.
  - Limit estops start at engage and check only engaged slots.
  - `arm_sdk_mode` ramps the weight 0→1 over `engage_seconds`, and refuses (latching) an upper command whose waist kp is below `waist_kp_min` (slots 12-14).
- **Refusal logging:** a refusal or wait reason is logged at most once per second.
- **Estop input** (only with `use_estop`):
  - triggered, or not plugged in, latches;
  - no sample for `estop.timeout_seconds` after the first latches;
  - without `use_estop`, `on_estop` calls are ignored.
- **Quiet inputs:**
  - Once engaged, an input whose last command is older than `stale.<input>_seconds` (when > 0) keeps being relayed. A Warn event is raised when it goes quiet and an Info event when it resumes. It never latches.
  - `/lowstate` older than `stale.state_seconds` raises the same Warn/Info pair, and the relay keeps running.
- **Transport lag:** when `domain_id == 0`, port the H1-2 tick-lag estop (`h12_safety_layer/core/safety_layer.py` `_update_state_lag`, `STATE_LAG_DRIFT_MS_PER_S` 0.5, `TICK_WRAP` 1<<32) with `control.state_lag_ms`.
- **`exit_step`:**
  - `full_body_mode` and `split_mode` return `nullopt` at once, so the robot holds its last command.
  - `arm_sdk_mode` holds the last command while the weight ramps to 0 over `release_seconds`, then returns `nullopt`.
  - Once the core is estopped, `exit_step` continues the estop sequence, and a finished sequence returns `nullopt`.

- [ ] **Step 1: Write the tests** (gtest):
  ```
  Startup.ZeroStreamFromFirstStep          h12: step(0) before any input/state == zero_command
  Startup.NoLowstateNeverEngages           h12: commands but no LowState for 5 s -> output stays zero_command; g1: stays nullopt
  Startup.ZeroStreamEngagesPerInput        h12 split: only Upper has spoken -> upper slots relay it, lower slots zero
  Startup.EngageGateWaitsForEveryInput     g1 split: Upper valid, Lower absent -> nullopt; both valid -> output
  Startup.EngageGateRefusesFarCommand      g1: upper q 0.2 rad from measured -> nullopt and one Warn per second
  Startup.ArmSdkEngageRamp                 g1 arm_sdk: weight rises 0 -> 1 over engage_seconds
  Startup.WaistKpBelowMinLatches           g1 arm_sdk: waist kp 10 -> estopped()
  Estop.FirstSampleTriggeredLatches        use_estop: first on_estop(triggered=true) -> estopped()
  Estop.NoEngageBeforeFirstSample          use_estop, h12: inputs and state present, no estop sample -> output stays zero_command
  Estop.StaleAfterEngageLatches            use_estop: samples stop for 0.6 s -> estopped()
  Estop.IgnoredWithoutUseEstop             use_estop=false: on_estop(triggered=true) -> not estopped
  Quiet.InputHeldAndReported               engaged, Lower silent 1 s -> its last command still relayed, one Warn, never estopped
  Quiet.LowstateGapHolds                   no LowState 1 s -> relay continues, Warn; resumes -> Info
  Lag.Domain0LateTicksLatch                domain 0, ticks falling behind receive time by > state_lag_ms -> estopped(); domain 91 -> never
  Exit.SplitStopsAtOnce                    exit_step -> nullopt immediately
  Exit.ArmSdkRampsThenStops                exit_step weight ramps to 0 over release_seconds, then nullopt
  ```
- [ ] **Step 2: Run** `CT golem_safety`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `CT golem_safety`. Expected: PASS (Task 5's tests too).
- [ ] **Step 5: Change report.** Add:
  - the G1 stale-input and stale-state estops are now hold-and-report;
  - the G1 exit in split/full now stops publishing;
  - the H1-2 now needs a `/lowstate` with `mode_machine` 6 before relaying;
  - the estop-age latch and the first-sample rule;
  - the transport-lag estop now also covers the G1 on domain 0.
- [ ] **Step 6: Commit** `golem_safety: start-up rules, estop input, quiet-input hold, lag and exit`.

### Task 7: `safety_node` and its ROS tests

**Files:**
- Create: `src/safety_node.cpp`, `include/golem_safety/boot_clock.hpp` (`double boot_time()` via `clock_gettime(CLOCK_BOOTTIME)`), `README.md` and `launch/safety.launch.py`. The launch file runs one node with the parameters below, for manual runs.
- Test (pytest, rclpy, `ROS_DOMAIN_ID=91`):
  - `test/test_node_ros.py`;
  - `test/test_no_sdk.py`;
  - `test/conftest.py`, with a `fake_robot` fixture that publishes `/lowstate` and records `/lowcmd` and `/arm_sdk`.

**The node:**
- **Parameters:** `robot` (h12|g1, required), `config` (name or path, required), `sim` (bool, default false) and `use_estop` (bool, default false).
- **Start-up refusals.** It exits non-zero with a one-line reason when:
  - the profile or config fails to load;
  - `sim` is true and `ROS_DOMAIN_ID` is `0`.
- **Subscriptions:** `/lowstate` and the mode's inputs, with QoS keep-last 1, best effort.
- **Relay loop:**
  - It runs on its own thread at `publish_hz`, sleeping to absolute deadlines on the boot clock.
  - Each cycle it polls each subscription with `take()`, using `rclcpp::MessageInfo` for the publisher GID, then feeds `SafetyCore` and publishes `step()`'s result on `topics.low_cmd_out`.
  - The relay never runs in an executor callback.
- **Estop:** with `use_estop`, it subscribes to `estop.triggered_topic` and `estop.plugged_in_topic` and feeds `on_estop` (a sample means either topic arrived).
- **Heartbeat:** `/safety/heartbeat`, `std_msgs/Bool` from a 10 Hz wall timer (steady clock), true until estopped.
- **Events** go to the node logger at their level.
- **Logging:** with `logging.enabled`, a `ChunkLogger` receives `take_log_sample()`.
- **Shutdown:** SIGINT/SIGTERM/SIGHUP stop the relay loop, then run `exit_step` at `publish_hz` until `nullopt`. The process exits non-zero only if an arm_sdk ramp could not complete.

**Tests:** port each test of `g1_safety_layer/test/test_g1_safety.py`, `test_arm_sdk_mode_dds.py`, `test_lowcmd_modes_dds.py`, `test_hardening_dds.py` and `test_node_exit.py` whose behavior the spec keeps. Run each port against the node over ROS, for each profile it applies to.
- Tests whose behavior the spec changes are rewritten to the new behavior: stale estops become hold, exit stops publishing, and the `robot:` key goes.
- The task report lists every old test as ported, rewritten or dropped (with the reason).

New tests:
```
test_heartbeat_true_then_false_on_estop           both robots
test_split_relay_end_to_end[h12|g1]                publish lowstate + both inputs -> /lowcmd carries them, mode_machine == profile
test_arm_sdk_output_topic                           g1 arm_sdk -> /arm_sdk, never /lowcmd
test_newest_only_under_load                          publish 2000 Hz upper commands for 2 s -> relayed q equals the newest within 1 cycle; output rate >= 495 Hz
test_config_for_other_robot_is_refused              robot:=h12 config:=arm_sdk -> exit code != 0, stderr names h12 and arm_sdk
test_sim_refuses_domain_zero                         sim:=true ROS_DOMAIN_ID=0 -> exit != 0 (process never joins a domain)
test_estop_topics_gate_engagement                    use_estop:=true h12 -> zero stream until /estop/* published, then relay
test_sigint_split_stops_publishing                  SIGINT -> no /lowcmd within 50 ms after the signal
test_sigint_arm_sdk_ramps                            SIGINT -> /arm_sdk weight ramps to 0, then exit code 0
test_no_sdk_import (test_no_sdk.py)                  grep golem_safety sources: no "unitree_sdk2py"
```

- [ ] **Step 1: Write the tests.**
- [ ] **Step 2: Run** `CT golem_safety`. Expected: FAIL.
- [ ] **Step 3: Implement** the node and the launch file.
- [ ] **Step 4: Run** `CT golem_safety`. Expected: PASS.
- [ ] **Step 5: Change report.** Add the heartbeat on the G1, and the hardware estop moving from the raw DDS sample to the ROS topics.
- [ ] **Step 6: Commit** `golem_safety: safety_node over ROS, with the ported safety tests`.

---

## Phase B: `golem_upper_ik` (Python)

### Task 8: Robot profiles, controller config, pairing, carried configs

**Files:**
- Create: `golem_upper_ik/profile.py`, `golem_upper_ik/robots/{__init__,h12,g1}.py`, `golem_upper_ik/config.py` and `golem_upper_ik/pairing.py`.
- Create: `config/h12/{safety_split,sim_safety_split,safety_full,debug}.yaml` (from `h12_ros2_controller/config/`) and `config/g1/{g1,g1_split,sim,sim_split}.yaml` (from `g1_control/config/`).
- Test: `test/test_profile.py`, `test/test_config.py` and `test/test_pairing.py`.

**Interfaces:**
- `profile.load_profile(robot: str) -> RobotProfile`, a frozen dataclass:
  - **From `golem_safety/robots/<robot>.yaml`:** `robot`, `mode_machine`, `mode_pr`, `body_joints: tuple[str, ...]` (slot order), `q_low`, `q_high`, `dq_max` and `tau_max` (numpy arrays), `lower_slots` and `upper_slots` (`range`).
  - **From `robots/<robot>.py`:**
    - `urdf_path`, `urdf_ros_path` (H1-2 from `h12_ros2_model`'s share, as `h12_ros2_controller/utility/path_definition_ros.py`; G1 from `g1_description.paths.urdf_path('with_hand', ...)`, imported inside `robots/g1.py` only);
    - `collision: tuple[str, str] | None` (H1-2: `h1_2_magpie_sphere.urdf` + `h1_2_magpie_sphere_collision.srdf`; G1: `None`);
    - `ee_frames` (H1-2 `*_wrist_yaw_link`, G1 `*_hand_palm_link`) and `planner_frames` (H1-2 `*_grasp_frame`, G1 `*_hand_palm_link`);
    - `imu_frame`;
    - `held_joints: tuple[str, ...]` (H1-2 `torso_joint`, G1 the three waist joints);
    - `named_configs: dict`, from each package's `named_config.py`;
    - `enabled_joints`, from each `joint_definition.py`.
  - It raises `ValueError` for an unknown robot.
- `config.load_controller_config(robot: str, name_or_path: str) -> dict`: the merged loader of `h12_ros2_controller/utility/controller_config.py` (HEAD) and `g1_control/utility/controller_config.py`.
  - The union of keys, with clip-limit derivation as each does today (defaults 0.001/0.10/0.25 for clip).
  - `controller.hold_targets: {joint: rad}` replaces H1-2 `torso_target` and G1 `waist_target`, carrying their values.
  - `topics.low_cmd` must start with `/safety/`.
  - Topics are ROS names.
  - Unknown keys raise.
- `pairing.pairing_refusal(robot, controller_config, safety_config, sim=None, locomotion=False) -> str | None`:
  - **G1:** today's `g1_control/utility/config_pairing.py` rules, unchanged.
  - **H1-2:** pairs only, with no sim/real refusal by name, because `lmg_sim` runs `relax_safety_split`:
    - `{relax,default,tight}_safety_split` ↔ `safety_split`;
    - `sim_safety_split` ↔ `sim_safety_split`;
    - `{relax,default,tight,locomotion}_safety_full` ↔ `safety_full`.
  - Also `default_controller_config(robot, safety_config) -> str | None`.

- [ ] **Step 1: Write the tests:**
  ```
  test_profiles_load[h12|g1]                  joint count 27/29, mode_machine 6/5, ee_frames as above, collision None only for g1
  test_profile_body_joints_match_safety_data  names equal golem_safety/robots/<robot>.yaml
  test_h12_profile_needs_no_g1_package        importing golem_upper_ik.robots.h12 with g1_description absent from sys.modules does not import it
  test_configs_load[each of the 8]            load_controller_config succeeds; hold_targets == old torso_target / waist_target
  test_low_cmd_topic_must_be_safety_input     topics.low_cmd: /lowcmd -> ValueError
  test_g1_pairing_matches_old_rules           for every (controller, safety, sim, locomotion) combination the old module accepts or refuses, the new one agrees
  test_h12_pairs                              relax_safety_split+safety_split ok; relax_safety_split+sim_safety_split refused; relax_safety_split+safety_split with sim=True ok
  ```
- [ ] **Step 2: Run** `CT golem_upper_ik`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `CT golem_upper_ik`. Expected: PASS.
- [ ] **Step 5: Change report.** Add the `hold_targets` key and the H1-2 pairing rules.
- [ ] **Step 6: Commit** `golem_upper_ik: robot profiles, controller config and config pairing`.

### Task 9: Robot model, IK, dynamics and the planner

Start from the H1-2 HEAD files and merge in the G1 generalisations listed in the audit (section B, F1, F2, F3 and F6).

**Files:**
- Create in `golem_upper_ik/core/`: `robot_model.py`, `ik_solver.py`, `robot_dynamics.py` (the `RobotDynamics` base only: no `MomentumDDP`, `BalanceDDP` or `create_*_ddp`), and `planner/{__init__,reduced_joint_planner,planner_process}.py`.
- Create: `tools/snapshot_old_fk.py` (run once, in the container), which writes `test/fixtures/fk_h12.json`: the H1-2 FK at 50 seeded random configurations, from `h12_ros2_controller`.
- Test: `test/test_robot_model.py`, `test/test_ik_solver.py` and `test/test_planner.py`.

**Interfaces:**
- `RobotModel(profile: RobotProfile)`. Body joints and the frozen joints come from the profile. Frozen joints use the G1's `_keep_body_joints`, so every non-body joint is frozen in slot order. It keeps the H1-2's `rebuild_reduced_model(q_measured)` (re-lock at measured, `0159d45`), `state_staleness_ms` and `reduced_limit_violations`. The collision model is optional (G1 behavior when `profile.collision is None`).
- `IKSolver(model: RobotModel, named_configs: dict)`:
  - the H1-2 seed order, current pose first (`9eec384`);
  - the collision barrier only when the model has one;
  - `rebuild_reduced()` and `update_configurations(q)`, as the H1-2.
- `ReducedJointPlanner` / `PlannerClient`: the H1-2 HEAD versions, with `frame_names` from `profile.planner_frames`.
- The planner reports unavailable when `profile.collision is None`; that is what the G1 does today (`PLANNER_UNAVAILABLE`).

- [ ] **Step 1: Write the tests:**
  ```
  test_h12_fk_matches_old_controller        golem RobotModel FK at the 50 fixture q == fixture (atol 1e-9) for every ee/planner frame
  test_g1_model_builds_without_collision    RobotModel(load_profile('g1')).has_collision is False
  test_ik_seed_order_current_first          IKSolver default seed items start with 'current'
  test_rebuild_reduced_relocks_at_measured  frozen torso at 0.3 rad measured -> reduced-model FK uses 0.3
  test_planner_*                            port h12_ros2_controller/test/test_reduced_joint_planner.py (15 tests), h12 profile
  test_planner_unavailable_without_collision g1 profile -> PlannerClient reports unavailable
  ```
- [ ] **Step 2: Run** `CT golem_upper_ik`. Expected: FAIL.
- [ ] **Step 3: Implement.** Generate the FK fixture first with `snapshot_old_fk.py`.
- [ ] **Step 4: Run** `CT golem_upper_ik`. Expected: PASS.
- [ ] **Step 5: Commit** `golem_upper_ik: robot model, IK, dynamics and planner for both robots`.

### Task 10: ROS I/O, command handling and the follow gate

Replace the SDK I/O: `h12_ros2_controller/core/{channel_interface,latest_reader,low_cmd_handler}.py` and `g1_control/core/{channel_interface,low_cmd_handler,follow_gate}.py`.

**Files:**
- Create in `golem_upper_ik/core/`: `ros_io.py`, `low_cmd_handler.py` and `follow_gate.py`.
- Test: `test/test_ros_io.py` and `test/test_low_cmd_handler.py`.

**Interfaces:**
- `class StaleStateError(RuntimeError)`
- `StateSubscriber(node, topic='/lowstate', callback_group=None)`:
  - QoS keep-last 1, best effort, in its own `MutuallyExclusiveCallbackGroup`;
  - `latest() -> tuple[LowState, float] | None` (message, receive time on `time.monotonic()`);
  - tick-lag tracking ported from `TickLagTracker`;
  - `check_fresh(max_age_s: float) -> None` raises `StaleStateError`.
- `CommandPublisher(node, topic: str, pub_hz: float, profile)`:
  - a background thread publishing the latest command at `pub_hz` (Task 1's pattern);
  - `set(q, dq, tau, kp, kd)` for the upper slots;
  - the header is the profile's `mode_pr`/`mode_machine`, with no CRC (the safety layer stamps it);
  - `stop()`.
- `SafetyOutputSubscriber(node, topic)`: the newest relayed command, for the follow gate.
- `LowCmdHandler(node, profile, config)`: H1-2 HEAD behavior with no controller-side estop thread (`deacb9a`); command clipping kept.
  - `FollowGate`, from `g1_control/core/follow_gate.py`, runs on both robots and opens when the safety output carries the controller's command.
  - Held joints carry `hold_targets` in every command.

- [ ] **Step 1: Write the tests** (rclpy, `ROS_DOMAIN_ID=92`, a fake node publishing `/lowstate`):
  ```
  test_state_subscriber_keeps_newest        publish 100 states in a burst -> latest() is the last
  test_check_fresh_raises_when_stale        no state for 0.3 s with max_age 0.1 -> StaleStateError
  test_command_publisher_rate               500 Hz for 3 s measured by a subscriber: mean >= 495 Hz
  test_header_is_profile                    h12 -> mode_machine 6; g1 -> 5
  test_held_joints_carry_hold_target        g1 waist slots carry hold_targets, h12 torso slot carries hold_targets, while the measured value differs
  test_follow_gate_holds_until_relayed      commands stay within follow_tolerance of measured until the fake safety output echoes them; then follow freely
  test_no_controller_side_estop             measured dq past limits.estop -> handler keeps publishing (no estop thread exists)
  ```
- [ ] **Step 2: Run** `CT golem_upper_ik`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `CT golem_upper_ik`. Expected: PASS.
- [ ] **Step 5: Change report.** Add the G1 controller-side estop's removal, the follow gate on the H1-2, and held joints on the H1-2 (the torso no longer re-seeds from its measured value).
- [ ] **Step 6: Commit** `golem_upper_ik: rclpy state and command I/O, command handler and follow gate`.

### Task 11: Controllers, `frame_task_server` and `joint_state_publisher`

**Files:**
- Create in `golem_upper_ik/core/controller/`: `upper_controller.py`, `arm_controller.py` and `frame_controller.py`.
- Create in `golem_upper_ik/ros2/`: `frame_task_server.py`, `joint_state_publisher.py` and `utility.py`.
- Modify: `setup.py`, adding the console scripts `frame_task_server` and `joint_state_publisher`, plus the config and launch data files.
- Test: `test/test_controller_loop.py`, `test/test_frame_task_server.py` and `test/test_no_sdk.py`.

**Behavior:** the H1-2 HEAD `upper_controller.py` and `frame_task_server.py`, plus the G1 fixes (spec, golem_upper_ik):
- **From the G1:**
  - wait for a fresh `/lowstate` with finite q before enabling motors (raise, publish nothing);
  - start-up moves bounded by a deadline (raise on timeout);
  - the held joints' hold;
  - the follow gate.
- **From the H1-2:**
  - stale-state abort;
  - goals start from the last command;
  - `execute_path` paces at `dt − work`;
  - a hold pinned at a limit is not reported as converged;
  - `relative` and `slow_mode` (`SLOW_MODE_SCALE` 0.25) on both robots;
  - hold-blocked reporting.
- **Nodes:**
  - Node names `frame_task_server` and `joint_state_publisher`.
  - Each takes the parameters `robot` (required) and `config` (H1-2 default `debug` for `joint_state_publisher`).
  - The ROS surface is as the spec lists: `/frame_task`, `/named_config`, `frame_names`, `frame_targets`, `frame_poses`, `{left,right}_ee_{pose,target}`, `/joint_states` at 100 Hz.
- **Planned goals without a collision model:** on the G1 they abort with the message `PLANNER_UNAVAILABLE` gives today.

- [ ] **Step 1: Write the tests.** Port `g1_control/test/test_controller_loop_dds.py` (8 tests) to ROS against a fake robot, a node that integrates commanded q into `/lowstate` and echoes them as the safety output. Run them for both profiles. Port `g1_control/test/test_g1_controller.py`, apart from what tests the SDK, for both profiles. Then add:
  ```
  test_relative_frame_goal[h12|g1]          relative: true, +0.05 m x on left -> final ee pose within threshold of start + 0.05
  test_slow_mode_scales_speed[h12|g1]       slow_mode: true -> peak commanded ee speed <= 0.25 * normal + 1e-3
  test_named_config_slow_mode[h12|g1]       NamedConfig slow_mode accepted and reaches the config
  test_startup_waits_for_fresh_state        no /lowstate -> controller start raises and publishes nothing
  test_stale_state_aborts_goal              /lowstate stops mid-goal -> goal aborted with the stale-state message
  test_planned_goal_without_collision_g1    plan: true on g1 -> aborted, message names the missing collision model
  test_joint_state_publisher[h12|g1]        /joint_states has the profile's body joints, 100 Hz +- 5
  test_no_sdk_import                        no "unitree_sdk2py" in golem_upper_ik sources
  ```
- [ ] **Step 2: Run** `CT golem_upper_ik`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `CT golem_upper_ik`. Expected: PASS. Also re-run Task 1's rate test.
- [ ] **Step 5: Change report.** Add `relative`/`slow_mode` on the G1, the H1-2's fresh-state wait and deadline-bounded start-up, and the G1 goal-start seeding.
- [ ] **Step 6: Commit** `golem_upper_ik: controllers, frame_task_server and joint_state_publisher`.

---

## Phase C: Switching over

### Task 12: H1-2 sims report `mode_machine` 6

**Files:**
- Modify: `h1_robocasa/unitree_interface.py`. Set `self.low_state.mode_machine = 6` where `low_state` is created; one line with a short comment naming the real robot's value.
- Modify (submodule `CL_isaaclab_sim`): the H1-2 `LowState_` publisher in `src/python/dds/` (find it with `grep -rn "LowState_" CL_isaaclab_sim/src/python/dds`). Set `mode_machine` 6.
- Modify: `docker/scripts/wait_for_sim.py`. Set `MODE_MACHINE['h12'] = 6`, and rewrite the comment above it to say both robots' sims report their robot's value.
- Test: `tests/docker/test_wait_for_sim.py` (host) and `tests/robocasa/test_robot_select.py`-style CPU test `tests/robocasa/test_h12_mode_machine.py` (container: `golem_sim_robocasa`).

- [ ] **Step 1: Write the tests.**
  - Update `tests/docker/test_wait_for_sim.py`: `expected_mode_machine('h12') == 6`.
  - Add `test_h12_sim_lowstate_reports_6`, which builds the RoboCasa H1-2 `SimInterface` on a dummy model as the existing CPU tests do, and asserts `low_state.mode_machine == 6`.
- [ ] **Step 2: Run** `python3 -m unittest discover -s tests/docker` (host) and `docker run --rm --network none -v "$PWD:/repo:ro" -w /repo -e PYTHONPATH=h1_robocasa -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 golem_sim_robocasa:latest -m unittest tests/robocasa/test_h12_mode_machine.py`. Expected: FAIL.
- [ ] **Step 3: Implement the three edits.** In `CL_isaaclab_sim`:
  1. `git -C CL_isaaclab_sim checkout camera_ready && git -C CL_isaaclab_sim pull --ff-only` (its pin `6f89c64` is `origin/camera_ready`).
  2. Edit and commit there.
  3. Push: `git -C CL_isaaclab_sim push origin camera_ready`. The user authorised this push.
  4. Stage the new pin in GOLEM.
- [ ] **Step 4: Run both.** Expected: PASS. Also run `tests/robocasa/surface/run.sh h12`. Expected: matches the golden, which does not record `mode_machine`.
- [ ] **Step 5: Change report.** Add the H1-2 sims' `mode_machine`.
- [ ] **Step 6: Commit** `sims: the H1-2 sims report mode_machine 6, like the real robot` (GOLEM files and the `CL_isaaclab_sim` pin).

### Task 13: H1-2 bringups and the H1-2 test harness

**Files:**
- Modify:
  - `core_ws/src/h12_bringup/launch/h12_control.launch.py`: `golem_safety` `safety_node` with `robot: h12` and `config`/`sim`/`use_estop` as parameters; `golem_upper_ik` `frame_task_server` with `robot: h12`; a new `sim` launch argument; the pairing check through `golem_upper_ik.pairing`.
  - `h12_state.launch.py`: `golem_upper_ik` `joint_state_publisher`.
  - `h12_sim_bringup_mac.launch.py`.
  - `h12_sim_bringup.launch.py`, `lmg_sim_bringup.launch.py` (pass `sim: true`) and `h12_real_robot_bringup.launch.py`, `lmg_real_bringup.launch.py`, `h12_safety_grippers.launch.py` (pass `sim: false`).
  - `package.xml`: `exec_depend` `golem_safety` and `golem_upper_ik`, replacing `h12_safety_layer` and `h12_ros2_controller`.
- Modify:
  - `tests/stack/minimal_stack.launch.py`;
  - `tests/stack/launch_ros_stack.sh` (`--packages-up-to golem_safety golem_upper_ik …`);
  - `tests/stack/test_heartbeat_clock.py`, rewritten to check the `golem_safety` heartbeat over ROS, or dropped with the reason in the report if Task 7's heartbeat test covers it.
- Modify: `docker/mac/scripts/launch_ros_mac.sh` `PKGS`. Replace `h12_ros2_controller h12_safety_layer` with `golem_safety golem_upper_ik`, and pass `--cmake-args -DBUILD_TESTING=OFF`. Update the comment in `docker/mac/RosDockerfile.slim.arm64`.
- Test: `core_ws/src/h12_bringup/test/test_bringup_packages.py`.

- [ ] **Step 1: Write the test.** `test_no_launch_names_old_packages` parses every `h12_bringup/launch/*.py` and `tests/stack/*.launch.py` and asserts that none contains `h12_safety_layer`, `h12_ros2_controller`, `g1_safety_layer` or `g1_control`.
- [ ] **Step 2: Run** `CT h12_bringup`. Expected: FAIL.
- [ ] **Step 3: Implement the edits.**
- [ ] **Step 4: Verify.**
  1. `CT h12_bringup`: PASS.
  2. `docker/scripts/docker_run.sh ros --restart --h12 bash -lc 'true'`: exit 0, meaning `launch_ros.sh`'s build with `--packages-ignore-regex '^g1_'` succeeded.
  3. `ROS_DOMAIN_ID=193 tests/stack/run.sh robocasa arms`: `"passed": true`.
  4. `ROS_DOMAIN_ID=193 tests/stack/run.sh robocasa locomotion`: `LOWERBODY_STACK_PASS`.
  5. `ROS_DOMAIN_ID=193 tests/stack/run.sh isaac arms`: passes (the RTX 5070 Ti is on this host).
  6. **lmg sim smoke:** start `golem_sim_robocasa` with `docker_run.sh robocasa --headless` on domain 193, then in `golem_ros` run `ros2 launch h12_bringup lmg_sim_bringup.launch.py` for 60 s. The safety node logs an engaged input, `/lowcmd` flows at ≥ 495 Hz, and there is no estop. Stop both containers afterwards.
- [ ] **Step 5: Commit** `h12_bringup: run golem_safety and golem_upper_ik`.

### Task 14: G1 bringups, `g1_locomotion` and the G1 tests

**Files:**
- Modify:
  - `core_ws/src/g1/g1_bringup/launch/g1_core.launch.py`: `safety_node` and `frame_task_server`/`joint_state_publisher` from the new packages with `robot: g1`; `pairing_refusal` and `default_controller_config` from `golem_upper_ik.pairing`, called with `robot='g1'`.
  - `g1_bringup.launch.py` and `g1_sim_bringup.launch.py` (pass `sim`).
  - `package.xml`, and `test/test_locomotion_launch.py`/`test_nav_launch.py` if they name the old packages.
- Modify:
  - `core_ws/src/g1/g1_locomotion/package.xml` (`exec_depend golem_safety`);
  - its README and code comments that name `g1_safety_layer`;
  - `test/test_locomotion_dds.py`, which runs the `golem_safety` node (G1 profile, `sim_split`) as a subprocess instead of the in-process `g1_safety_layer`.
- Modify: `tests/g1/test_sim_bringup.py`. Imports come from `golem_upper_ik` (`load_profile('g1').named_configs`, `enabled_joints`), and the `pgrep` pattern becomes `lib/golem_upper_ik/frame_task_server`.
- Test: `core_ws/src/g1/g1_bringup/test/test_bringup_packages.py` (same assertion as Task 13's, over `g1_bringup/launch`).

- [ ] **Step 1: Write the test.**
- [ ] **Step 2: Run** `CT g1_bringup`. Expected: FAIL.
- [ ] **Step 3: Implement the edits.**
- [ ] **Step 4: Verify.**
  1. `CT g1_bringup` and `CT g1_locomotion`: PASS.
  2. `tests/g1/test_sim_bringup.sh`: PASS.
  3. `tests/g1/test_sim_bringup.sh --walk`: PASS.
  4. `tests/g1/test_sim_nav.sh`: PASS.
  5. `GOLEM_SIM_ODOM=1 tests/g1/test_sim_nav.sh`: PASS.
  6. `tests/g1/test_sim_explore.sh`: PASS.
  7. `tests/robocasa/surface/run.sh g1` matches the golden.

  If the known `t_pose` flake (0.000995 against 0.001) appears, re-run once and report it; do not retune.
- [ ] **Step 5: Commit** `g1_bringup, g1_locomotion: run golem_safety and golem_upper_ik`.

### Task 15: Rates and the change report

**Files:**
- Create: `core_ws/src/golem_safety/tools/relay_latency_probe.py`. With a running sim stack, it subscribes to `/safety/lowcmd_upper_in` and `/lowcmd` (or `/arm_sdk`) and reports the output rate and the input-to-output latency (median, p99), matching samples by their q values.
- Modify: `docs/superpowers/specs/2026-10-27-golem-upper-ik-and-safety/change-report.md`.

- [ ] **Step 1: Measure.** Bring up each sim stack in turn with `ROS_DOMAIN_ID=193 docker/scripts/docker_stack.sh up robocasa --headless`, then `... --g1 --headless`. Send one `/named_config` goal so the arms are commanded, then record:
  - the `golem_safety` publish rate and latency;
  - the `golem_upper_ik` `/safety/lowcmd_upper_in` rate (`ros2 topic hz`, 10 s).

  Expected: both ≥ 495 Hz. Stop the stacks afterwards.
- [ ] **Step 2: Finish the change report.**
  - Add a "Measured rates" section.
  - Add an "Unchanged on purpose" section: limits, gains, start-up postures and named configurations.
  - Check that the report covers, per robot, every change Tasks 3-14 made.
- [ ] **Step 3: Commit** `docs: golem change report and measured rates`.

### Task 16: Documentation

**Files:**
- Modify:
  - `CLAUDE.md` and `AGENTS.md` (keep them mirrored): §2 layout, §3 (the `golem_*` packages build for both robots), §4 package tables (add `golem_safety` and `golem_upper_ik`; mark the four old packages as unused until deleted), §6 bringup tables, §8 invariants (the safety layer is `golem_safety` for both robots; hold-on-quiet; identity check; H1-2 sims report 6);
  - `docs/g1/{index,packages,control-path,safety,first-session}.md`;
  - `docker/BUILD.md`, `tests/g1/README.md` and `core_ws/src/h12_bringup/common_cmds.md`.
- Create: `core_ws/src/golem_safety/README.md` and `core_ws/src/golem_upper_ik/README.md`, each covering the interface, parameters, configs per robot, the behaviors the spec lists, and the test commands.

- [ ] **Step 1: Grep check.** `grep -rn "h12_safety_layer\|h12_ros2_controller\|g1_safety_layer\|g1_control" CLAUDE.md AGENTS.md docs/g1 docker/BUILD.md tests/g1/README.md core_ws/src/h12_bringup/common_cmds.md`. Expected after the edits: only lines that say these packages are unused and awaiting deletion.
- [ ] **Step 2: Write the docs.** Verify each claim against the code, as the 2026-10-26 port's doc task did.
- [ ] **Step 3: Docs build.** `docker run --rm -v "$PWD:/repo" -w /repo python:3.11-slim sh -c "pip install -q -r docs/requirements-docs.txt && mkdocs build --strict"`. Expected: PASS.
- [ ] **Step 4: Commit** `docs: golem_safety and golem_upper_ik`.

---

## After the plan (for the user)

These are not plan tasks:
- **The Mac smoke test** in `docs/DOCKER_TESTING.md`, on an Apple-Silicon Mac. This host cannot run the arm64 images.
- **Real-robot runs** of the H1-2 and the G1.
- **Deleting the four old packages**, when the user says so.
