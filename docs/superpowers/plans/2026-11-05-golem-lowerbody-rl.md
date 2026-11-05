# golem_lowerbody_rl Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One leg package, `golem_lowerbody_rl`, runs both robots' walking policies with identical code; each robot is a config directory and each policy is a spec file that says how its observation is built and its action interpreted.

**Architecture:** `h12_lowerbody_rl` is renamed (`git mv`) and rebuilt around four ROS-free parts — a robot profile (joint names → slots from `golem_safety`'s table), a strict run config, a policy runner driven by a YAML spec (term registry, history groups, command processing, action mapping, TorchScript/ONNX backends) and one unified `LegSupervisor` — wrapped by one rclpy node whose `/lowstate` subscription runs in its own callback group and thread. Both bringups include `launch/lowerbody.launch.py`; `g1_homie` is deleted.

**Tech Stack:** ROS 2 Humble (rclpy), Python 3.10, numpy, PyYAML, torch (TorchScript), onnxruntime (lazy), pytest; Docker images `golem_ros:latest`, `golem_sim_robocasa:latest`.

**Spec:** `docs/superpowers/specs/2026-11-05-golem-lowerbody-rl-design.md` (approved). Evidence and line references: `docs/superpowers/specs/2026-11-05-golem-lowerbody-rl-audit.md` (`H/` = `core_ws/src/h12_lowerbody_rl/`, `G/` = `core_ws/src/g1_homie/`; line numbers predate the comment pass, facts stand).

## Global Constraints

- Identical code, different config: no robot or policy name as a branch in `golem_lowerbody_rl`'s code or launch file; `tests/docker/test_shared_packages.py` enforces it from Task 8 on (data dirs `config/<robot>/`, `policies/<name>/` are not code).
- No YAML flag that picks between two drifted code paths (no `ramp: cosine|linear`).
- Every policy's network input and leg targets match the golden traces of Task 1 within `atol=1e-6` for the same inputs; the only intended numeric change is gravity from a normalised quaternion.
- Transport is ROS only (rclpy): no `unitree_sdk2py` or `cyclonedds` import in the package.
- HOMIE's values are transcribed into `config/g1/policies/homie.yaml` (MIT). Its weights, `g1.yaml`, `homie_policy.py`, `mujoco_deploy_g1.py`, `test_homie_policy.py`, `LICENSE` and `README.md` live together in `core_ws/src/golem_lowerbody_rl/policies/homie/` (CC BY-NC-SA); `homie_policy.py` is used only as the golden oracle and by its MuJoCo script.
- onnxruntime and torch are imported only inside the backend that needs them; the Mac image (no onnxruntime) must import the package and run any H1-2 policy.
- Package licence `MIT` (`package.xml`, `setup.py`, `LICENSE` at the package root); weights keep their own licence files.
- CLAUDE.md applies, including §8 "Comments and docstrings": an imperative purpose line on every module/class/function/method and test, `SAFETY:` on safety invariants, `description=` on every launch argument, an annotated `config/example.<kind>.yaml` per schema that is not installed and is checked against the loader.
- Comments describe the code as it is; no history narration.
- Builds and tests only in disposable containers (`docker run --rm --name lb-<task>-... golem_ros:latest`, repo mounted, `colcon build --build-base /tmp/b --install-base /tmp/i`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), never in or replacing the user's `golem_ros` container, never `ROS_DOMAIN_ID` 0, never `--network none` for tests that start `safety_node` (CycloneDDS's warning then blocks a stalled-stdout test). Host-only: `/usr/bin/python3 -m pytest tests/docker -q`.
- Never `git stash`/`reset`/`clean`/`checkout -- <file>`/`add -A`; never touch `core_ws/src/golem_bringup/rviz/g1/g1.rviz`, the `livox_ros_driver2` submodule or `docker/.env`. Commits end with:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01MezUeauFS9tKMgZKJxo1HC`. No push.
- Values that move keep their value (gains, scales, defaults, timeouts) unless the spec's unified-behaviour table changes them.

## Review Focus

1. A `/lowstate` whose tick is not milliseconds of the node's clock (Isaac counts publishes): the legs never leave WAITING-for-ready and the node says why once every 5 s — Task 5 `test_tick_not_in_ms_never_becomes_ready`.
2. A policy spec whose observed joint name is missing from the robot's joint table, or whose input size does not match the network: refused at load with the joint/size named, before any command — Task 4 `test_unknown_joint_is_refused`, `test_input_size_mismatch_is_refused`.
3. `/cmd_vel` with NaN/inf or a stale command while running: zero velocity, never a non-finite observation — Task 5 `test_non_finite_cmd_is_zero`, Task 4 `test_observation_is_finite_or_raises`.
4. The band toggle reply arriving on the service thread while the state thread steps the policy: the reset happens once and never mid-step — Task 6 `test_band_reset_is_serialised_with_steps`.
5. Launching with `config:=real` in a sim bringup, or a policy the robot does not list: refused before any node starts — Task 6 `test_launch_refuses_*`.

---

### Task 1: Golden traces of today's four policies

**Files:**
- Create: `core_ws/src/h12_lowerbody_rl/test/golden/record_goldens.py`, `core_ws/src/h12_lowerbody_rl/test/golden/{walk,almi,fame,homie}.npz`, `core_ws/src/h12_lowerbody_rl/test/test_goldens_current.py`

**Interfaces:**
- Produces: `golden/<policy>.npz` with arrays `t` (s), `q` (N×body joints, slot order), `dq`, `quat_wxyz` (unit quaternions only), `gyro`, `cmd` (N×3, the command as the node hands it on: clipped to ±1 per axis on the H1-2; on the G1 the input of `g1_homie`'s `CommandFilter`, which the recorder runs with `G/config/sim.yaml`'s limits at dt 0.02 s to make HOMIE's input), `height` (N), `reset` (N bool: the policy was reset before this step), `net_input` (N×input size; FAME also `enc_input` N×21 and `z` N×8), `net_output` (N×12), `targets` (N×12), `kp` (12), `kd` (12). Later tasks compare against these.

- [ ] **Step 1:** Write `record_goldens.py` (not installed): drives `WalkPolicy`, `AlmiPolicy`, `FamePolicy` (from `h12_lowerbody_rl.policy`) and `HomiePolicy` (`g1_homie/policy/homie_policy.py`) through one scripted 120-step sequence per policy with `numpy.random.default_rng(7)`: joint positions = default pose ± 0.05 rad noise, dq ± 0.5 rad/s, gyro ± 0.3 rad/s, unit quaternions within 10° of upright; commands: 20 steps zero, then a ramp to (0.6, 0.0, 0.3) (ALMI onset rise at 0.1), hold, back below 0.05 (fall), rise again; a reset at step 60 (band release); height 1.0 (FAME) / 0.74→0.60 (HOMIE). Capture the network input and output by wrapping the loaded TorchScript/ONNX call (and FAME's encoder), and the targets the class returns.
- [ ] **Step 2:** Run it in a disposable `golem_ros` container from the repo root; commit the four `.npz` files.
- [ ] **Step 3:** Write `test_goldens_current.py::test_current_classes_reproduce_their_goldens[walk|almi|fame|homie]`: replay each golden's inputs through the current class and assert `net_input`, `net_output`, `targets` equal the file (`atol=1e-6`). Purpose: proves the recording is deterministic. Run: `colcon test --packages-select h12_lowerbody_rl` (needs `g1_homie/policy` on `sys.path` for HOMIE; skip HOMIE with a reason if onnxruntime is absent). Expected: PASS.
- [ ] **Step 4:** Commit `h12_lowerbody_rl: golden traces of WALK, ALMI, FAME and HOMIE`.

### Task 2: Rename `h12_lowerbody_rl` → `golem_lowerbody_rl`

**Files:**
- Move: `core_ws/src/h12_lowerbody_rl` → `core_ws/src/golem_lowerbody_rl` (module dir, `resource/`, `setup.py`, `setup.cfg`, `package.xml`: name, `MIT`, `LICENSE` file copied from `g1_homie/LICENSE` with the package name)
- Modify: `core_ws/src/golem_bringup/{package.xml,launch/h12/h12_lowerbody.launch.py,launch/h12/h12_sim_mac.launch.py,test/*}`, `docker/mac/scripts/launch_ros_mac.sh` (`PKGS`), `docker/mac/docker-compose.yml`, `docker/BUILD.md`, `tests/stack/{launch_ros_stack.sh,minimal_stack.launch.py,README.md}`, `tests/docker/*` that name it, `core_ws/src/lmg_deploy` comments that cite it, `README.md`, `docs/RUNNING.md`, `docs/NAVIGATION_DEMO.md`, CLAUDE.md and AGENTS.md (§1, §2, §4, §6, §9 and the §10 stale-install `rm -rf` list gains `h12_lowerbody_rl` and `g1_homie`)

**Interfaces:**
- Produces: Python package `golem_lowerbody_rl`, executable `lowerbody_controller_node`, node name `lowerbody_controller_node` (unchanged), share dir `share/golem_lowerbody_rl/policies/{almi,fame,walk}/`.

- [ ] **Step 1:** `git mv`, rename the module and every import; no behaviour change.
- [ ] **Step 2:** `git grep -n h12_lowerbody_rl -- ':!docs/superpowers'` must list only CLAUDE.md/AGENTS.md's stale-install text.
- [ ] **Step 3:** Build and test `golem_lowerbody_rl` and `golem_bringup` in a disposable container (all pass, Task 1 goldens included); run `tests/stack/run.sh` locomotion (README) on a non-zero domain. Expected: PASS.
- [ ] **Step 4:** Commit `golem_lowerbody_rl: rename h12_lowerbody_rl`.

### Task 3: Robot profile, run config, IMU and timing helpers

**Files:**
- Create: `golem_lowerbody_rl/profile.py`, `golem_lowerbody_rl/run_config.py`, `golem_lowerbody_rl/imu.py`, `golem_lowerbody_rl/timing.py`; `config/h12/{robot,sim,real}.yaml`, `config/g1/{robot,sim,real}.yaml`, `config/example.robot.yaml`, `config/example.run.yaml`; tests `test/test_profile.py`, `test/test_run_config.py`, `test/test_imu.py`, `test/test_timing.py`, `test/test_example_configs.py`
- Modify: `setup.py` (install `config/<robot>/**`, exclude `example.*`), `package.xml` (exec_depend `golem_safety`, `ament_index_python`)

**Interfaces:**
- Produces:
  - `RobotProfile` (frozen dataclass): `robot: str`, `mode_machine: int`, `mode_pr: int`, `body_joints: tuple[str, ...]` (slot order), `legs: tuple[str, ...]` (action order), `imu_frame: str`, `policies: tuple[str, ...]`, `default_policy: str`; method `slots(names: Sequence[str]) -> np.ndarray` (KeyError naming the joint). `load_profile(robot: str) -> RobotProfile` reads `share/golem_safety/robots/<robot>.yaml` (as `golem_upper_ik/golem_upper_ik/profile.py:_robot_data`) and `share/golem_lowerbody_rl/config/<robot>/robot.yaml` (keys exactly `legs, imu_frame, policies, default_policy`). `known_robots() -> tuple[str, ...]`.
  - `RunConfig` (frozen dataclass) and `load_run_config(robot: str, name_or_path: str) -> RunConfig`, strict (missing or unknown key → `ValueError` naming it). Keys:
    `sim: bool`; `startup: {wait_for_confirm: bool, ramp_seconds: 2.0, settle_tolerance: 0.15 (rad), settle_ticks: 10, settle_max_ticks: 400, steady_seconds: 1.0, tick_rate: [0.9, 1.1], upper_fresh_seconds: 0.5}`; `heartbeat: {timeout: 0.5}`; `band: {release: bool, max_wait: s}`; `commands: {timeout: 0.5, vx: [lo, hi], vy: [lo, hi], yaw: [lo, hi], accel: [ax, ay, ayaw] | null, height_rate: float | null}`; `faults: {max_state_gap: s, max_latency: s | null}`; `imu_trim_deg: [roll, pitch, yaw]`.
    Values: H1-2 sim — confirm false, band release true, max_wait 34.0, limits ±1.0 on each axis, accel null, height_rate null, gap 0.5, latency null, trim [0,0,0]; H1-2 real — confirm true, band release false, max_wait 34.0, limits ±1.0, gap 0.25, latency null, trim [0.5, -2.5, 0.0]; G1 sim — confirm false, band true, max_wait 30.0, vx [-0.8, 1.2], vy [-0.5, 0.5], yaw [-0.8, 0.8], accel [0.5, 0.5, 1.0], height_rate 0.1, gap 0.5, latency null, trim [0,0,0]; G1 real — confirm true, band false, max_wait 30.0, vx [-0.3, 0.5], vy [-0.3, 0.3], yaw [-0.6, 0.6], accel [0.5, 0.5, 1.0], height_rate 0.1, gap 0.25, latency null, trim [0,0,0]. All: tick_rate [0.9, 1.1], heartbeat 0.5, commands.timeout 0.5. Checks: every limit range contains 0; `sim` matches the file's role.
  - `imu.py`: `trim_quaternion(roll_deg, pitch_deg, yaw_deg) -> np.ndarray` (wxyz, the H1-2 node's current trim math moved unchanged), `apply_trim(q_wxyz, trim) -> np.ndarray`, `projected_gravity(q_wxyz) -> np.ndarray` (normalise, then `quat_rotate_inverse(q, [0, 0, -1])`), `gravity_z(q_wxyz) -> float`.
  - `timing.py`: `TICK_SECONDS = 0.001`; `tick_delta(prev: int, now: int) -> int` (uint32 wrap, as `G/g1_homie/locomotion.py:tick_delta`); `TickGrid(period_s=0.02)` with `due(tick: int) -> bool` (true once per 20 ms grid cell crossed, as `G/.../locomotion.py:_step_at`); `TickLag` with `update(tick: int, now: float) -> float` (ms; the approach of `golem_upper_ik/core/ros_io.py:TickLagTracker`).

- [ ] **Step 1: Failing tests** (assert the values above): `test_profile.py::test_slots_follow_the_safety_table[h12|g1]` (H1-2 `left_hip_yaw_joint` → 0; G1 `left_hip_pitch_joint` → 0), `test_unknown_joint_is_refused`, `test_mode_machine_from_the_safety_table` (6, 5); `test_run_config.py::test_values[h12-sim|h12-real|g1-sim|g1-real]`, `test_unknown_key_is_refused`, `test_missing_key_is_refused`, `test_limit_must_contain_zero`; `test_imu.py::test_projected_gravity_matches_both_old_formulas_for_unit_quaternions` (vs `policy.gravity_from_quat_walk` and `gravity_from_quat_fame` values copied into the test), `test_non_unit_quaternion_is_normalised`, `test_trim_matches_the_old_node`; `test_timing.py::test_grid_steps_every_20_ms`, `test_tick_wraps`, `test_tick_lag`; `test_example_configs.py::test_examples_hold_every_key_and_load` and `test_examples_are_not_installed`.
- [ ] **Step 2:** Run in a container; expect FAIL (modules missing).
- [ ] **Step 3:** Implement the interfaces above; configs and annotated examples per CLAUDE.md §8.
- [ ] **Step 4:** Run; expect PASS. Commit `golem_lowerbody_rl: robot profile, run config, IMU and tick helpers`.

### Task 4: Policy spec, term registry, runtimes and runner (WALK, ALMI, FAME, HOMIE)

**Files:**
- Create: `golem_lowerbody_rl/policy/{__init__,spec,observation,commands,action,runtime,encoder,runner}.py`; `config/h12/policies/{almi,walk,fame}.yaml`, `config/g1/policies/homie.yaml`, `config/example.policy.yaml`; tests `test/test_policy_spec.py`, `test/test_observation.py`, `test/test_runner_goldens.py`
- Move: `core_ws/src/g1_homie/policy/` → `core_ws/src/golem_lowerbody_rl/policies/homie/` (`git mv`); `rma/env_factor_encoder.py` → `policy/encoder.py` (delete `rma/`)
- Modify: `setup.py` (install `policies/homie/{deploy.onnx,LICENSE,README.md}`), `docker/docker-compose.yml` (the ros mount source becomes `core_ws/src/golem_lowerbody_rl/policies/homie`, target unchanged, so `g1_homie` keeps running until Task 7), `docker/scripts/docker_common.sh` (`require_g1_policy` path), `tests/docker/test_workflows.py`, `core_ws/src/g1_homie` tests that import `homie_policy` by path, `tests/robocasa/g1/test_g1_band.py` (`test_sim_reads_no_policy` path)

**Interfaces:**
- Consumes: Task 3 `RobotProfile`, `RunConfig`, `projected_gravity`.
- Produces:
  - `PolicySpec` and `load_policy_spec(robot: str, name_or_path: str) -> PolicySpec`, strict. Keys: `runtime: {kind: torchscript|onnx, file: <path relative to the package share>, reset: <method name> | null, threads: int, warmup: bool}`; `rate_hz: 50`; `observed_joints: [names]`; `default_pose: {name: rad}` (unlisted observed joints default 0; leg entries are the pre-pose); `groups: [{history: int, terms: [...]}]` (oldest first, zero-filled at reset); `command: {trained_range: {vx, vy, yaw} | null, gain: [3] | null, clip: float | null, smooth_alpha: float | null, onset_reset: {rise, fall} | null, height: {default: float, range: [lo, hi] | null}}`; `action: {scale: float, clip_raw: float | null, clip_targets: {lower: [12], upper: [12]} | null, kp: [12], kd: [12]}`; `reset_on: [engage, band_release]`.
  - Terms (registry in `observation.py`, `TERMS: dict[str, Callable]`): `ang_vel{scale}`, `projected_gravity`, `velocity_command{scale: [3]}`, `height_command{scale}`, `joint_pos_rel{scale}`, `joint_vel{scale}`, `last_action{clipped: bool}`, `gait_phase{period, form: sin_cos|left_right_sin, offset, stand_below: float|null}`, `latent{encoder: {file, hidden, activation, latent}, joints: [names], forces: [[3],[3]]}` (FAME; history order of its group oldest first, as `FamePolicy` feeds it).
  - `Observation` input: `PolicyInput(t: float, q: np.ndarray, dq: np.ndarray, quat_wxyz: np.ndarray, gyro: np.ndarray, cmd: np.ndarray, height: float)` (q/dq over all body slots).
  - `CommandProcessor(spec.command, run.commands)`: `process(raw_cmd, dt) -> np.ndarray` in order: run limits clip → trained_range clip → gain → clip → EMA → accel limit; `process_height(raw, dt)`; `reset(seed_cmd)`; onset detection returns `onset: bool` (ALMI's hysteresis 0.1 / 0.05).
  - `Runtime` protocol: `load(spec.runtime, share_dir) -> Runtime` with `__call__(x: np.ndarray) -> np.ndarray` and `reset() -> None`; `TorchScriptRuntime` (calls `torch.set_num_threads(threads)` and `torch.set_num_interop_threads` where allowed; every spec sets `threads: 1` — with torch's default pool the H1-2 leg node took 3.6–7.5 cores and slowed the sim from 0.79x to 0.65x real time), `OnnxRuntime` (onnxruntime imported in its `__init__`; `threads` intra-op, 1 inter-op, no spinning; warm-up step then reset when `warmup`). Input size checked against the network at load.
  - `PolicyRunner(profile, run, spec, share_dir)`: `reset(seed_cmd=None)`, `step(inp: PolicyInput) -> LegTargets`, `name: str`; `LegTargets(q: np.ndarray[12], kp, kd)` in `profile.legs` order. Raises `PolicyError` on a non-finite observation, network output or target.

- [ ] **Step 1: Failing tests:** `test_runner_goldens.py::test_runner_matches_golden[walk|almi|fame|homie]` — replay Task 1's inputs through `CommandProcessor` (the robot's `sim` run config, dt 0.02 s; an onset resets the runner with the seed command, as ALMI does today) and `PolicyRunner` (resetting where `reset` is set) and assert `net_input`, `net_output`, `targets`, `kp`, `kd` equal the golden (`atol=1e-6`; FAME also `enc_input`, `z`); `test_policy_spec.py::test_unknown_joint_is_refused`, `test_input_size_mismatch_is_refused`, `test_unknown_term_is_refused`, `test_example_policy_holds_every_key_and_loads`; `test_observation.py::test_observation_is_finite_or_raises`, `test_almi_stand_gate_zeroes_the_phase_below_0_1`, `test_onset_resets_once_per_rise`, `test_onnx_is_not_imported_for_torchscript_policies` (subprocess: import the package, load ALMI, assert `'onnxruntime' not in sys.modules`).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement; write the four specs from the policy YAMLs and the audit §2 (values unchanged; HOMIE's from `policies/homie/g1.yaml`); the old `policies/<name>/<name>.yaml` files stay until Task 6 (the node still reads them).
- [ ] **Step 4:** Run the package tests, `g1_homie`'s tests and `tests/docker`; expect PASS. Commit `golem_lowerbody_rl: config-driven policy runner; HOMIE's policy moves into the package`.

### Task 5: One `LegSupervisor`

**Files:**
- Rewrite: `golem_lowerbody_rl/supervisor.py`; `golem_lowerbody_rl/band.py` (from `G/g1_homie/band.py`)
- Test: `test/test_supervisor.py` (rewrite: keep the H1-2 cases, port `G/test/test_locomotion.py`'s waiting/readiness/heartbeat/ramp/confirm/start-block/band/fault/tick cases), `test/test_band.py` (port `G/test/test_band.py`)

**Interfaces:**
- Consumes: `RunConfig`, `RobotProfile`, `TickGrid`, `TickLag`, `gravity_z`, `PolicyRunner` (via a `step_policy(inp) -> LegTargets` callable and `reset_policy()`).
- Produces: `Phase` enum `WAITING, RAMPING, HOLDING, RUNNING, FAULTED`; `LegSupervisor(run, profile, prepose: np.ndarray, runner)` with `on_heartbeat(value: bool, mono: float)`, `on_cmd(vx, vy, yaw, mono)`, `on_height(h, mono)`, `on_upper_cmd(mono)`, `on_ee_pose(mono)`, `on_state(sample: StateSample, mono: float, node_t: float) -> LegTargets | None` (returns targets when the tick grid is due), `check(mono: float) -> None` (gap/heartbeat-silence faults between samples), `confirm(mono) -> tuple[bool, str]`, `band_due(mono) -> str | None` (the reason to call the toggle now, once), `on_band_reply(success: bool, message: str)`, `phase`, `fault: str | None`, `wait_reason: str | None`; `StateSample(tick: int, q, dq, quat_wxyz, gyro)`.
- Behaviour (spec table, both robots): cosine ramp over `ramp_seconds` from the measured pose to the pre-pose, then settle (max error < `settle_tolerance` for `settle_ticks`, or `settle_max_ticks` after the ramp → log "NOT SETTLED", hold anyway); readiness = `steady_seconds` of samples whose tick advanced at a rate within `tick_rate` of the node clock, no gap over `max_state_gap`, an upper command within `upper_fresh_seconds`; confirm accepted only in HOLDING and ready (real); without confirm the policy starts once ready (sim) and, with `band.release`, also once the first `/left_ee_pose` came or `max_wait` after HOLDING began; band toggle due once RUNNING and (`/left_ee_pose` seen or `max_wait` after RUNNING began); a successful reply resets the policy immediately, a failure is logged once as `elastic band release failed: <msg>` and nothing retries; faults latch (gap, non-finite, quaternion norm outside 1 ± 0.05, `gravity_z > -0.5` once commanded, policy error, tick going back, tick-lag over `max_latency` when set, false heartbeat or none for `heartbeat.timeout` after the first, in every phase).

- [ ] **Step 1: Failing tests** (fake clocks), including: `test_cosine_ramp_reaches_the_prepose_in_2_s`, `test_not_settled_holds_after_400_ticks`, `test_tick_not_in_ms_never_becomes_ready` (tick advancing 1 per 2 ms of node time), `test_ready_needs_fresh_upper_command`, `test_confirm_refused_until_ready`, `test_sim_policy_waits_for_ee_pose_or_max_wait`, `test_band_called_once_and_never_retried`, `test_band_success_resets_the_policy_immediately`, `test_non_finite_cmd_is_zero`, `test_stale_cmd_is_zero_after_0_5_s`, `test_false_heartbeat_latches_in_every_phase`, `test_silent_heartbeat_latches_after_0_5_s`, `test_gap_over_limit_faults_once_commanded`, `test_tick_back_faults`, `test_latency_off_when_null`, `test_fall_faults`, `test_policy_error_faults_and_stops_output`.
- [ ] **Step 2:** Run; expect FAIL. **Step 3:** Implement. **Step 4:** Run; expect PASS. Commit `golem_lowerbody_rl: one leg supervisor for both robots`.

### Task 6: The node, its launch file, and the H1-2 bringups on it

**Files:**
- Rewrite: `golem_lowerbody_rl/node.py` (replaces `scripts/lowerbody_controller_node.py`); delete `policy.py`, `policy_manager.py`, `scripts/`, `policies/<name>/<name>.yaml` (weights and READMEs stay)
- Create: `launch/lowerbody.launch.py`; tests `test/test_node.py`, `test/test_launch.py`
- Modify: `core_ws/src/golem_bringup/launch/h12/{h12_lowerbody,h12_sim,h12_sim_mac,h12_real_robot}.launch.py` (include `lowerbody.launch.py` with `robot: h12`; `lowerbody:=` → `policy:=`; keep `start_position_verified`, the `mjpc` exclusion and the MJPC estimator), delete `golem_bringup/config/h12/{lowerbody_sim,lowerbody_real,example.lowerbody}.yaml`, `golem_bringup/test/*` that assert them; `tests/stack/{minimal_stack.launch.py,check_lowerbody.py,run_checker.sh,README.md}`; `docker/mac/docker-compose.yml` and `launch_ros_mac.sh` (`GOLEM_LOWERBODY` → `policy:=`); `golem_bringup/common_cmds.md`, `docs/RUNNING.md`, CLAUDE.md/AGENTS.md §6 (the leg node paragraph, `/lowerbody/height`)

**Interfaces:**
- Consumes: Tasks 3-5.
- Produces: node `lowerbody_controller_node`, parameters `robot` (required), `config` (`sim`|`real`|path, required), `policy` (name|path, default the robot's `default_policy`), `imu_trim_{roll,pitch,yaw}_deg` (live, ±15°, seeded from the run config). Topics: sub `/lowstate` (best effort, keep-last 1, its own `MutuallyExclusiveCallbackGroup`, run by a `MultiThreadedExecutor(num_threads=2)`), `/cmd_vel`, `/lowerbody/height` (`std_msgs/Float32`, the policy's height command in its own units), `/safety/heartbeat`, `/safety/lowcmd_upper_in` (readiness only), `/left_ee_pose` (only with `band.release`; destroyed after the first); pub `/safety/lowcmd_lower_in` (slots from `profile.legs`, `mode=1`, `mode_pr`/`mode_machine` from the profile, dq = tau = 0), `/lowerbody/active_policy` (latched); srv `/lowerbody/confirm_engage`; cli `/elastic_band/toggle`; a 100 Hz steady timer for `check()`/band in the other group; one `threading.Lock` around every supervisor/runner call. Log lines (tests read them): `loaded run config <path> and policy <path>`, `legs pre-posed (...) — holding`, `policy started`, `releasing the elastic band: <reason>`, `elastic band released`, `elastic band release failed: <msg>`, a timing line every 10 s.
  `lowerbody.launch.py` arguments: `robot`, `config`, `policy`, `sim` (no default), `use_sim_time`, `start_delay`; an `OpaqueFunction` refuses before any node starts (`lowerbody: not starting: <reason>`) when the run config's `sim` ≠ `sim`, the policy is not in the robot's `policies`, or the robot is unknown.

- [ ] **Step 1: Failing tests:** `test_node.py` (in-process rclpy on a test domain with a fake `/lowstate` and fake heartbeat/upper publishers): `test_publishes_nothing_before_the_first_true_heartbeat`, `test_walks_on_cmd_vel_with_the_policy_targets` (ALMI), `test_band_reset_is_serialised_with_steps`, `test_active_policy_is_latched`, `test_confirm_service`; `test_launch.py::test_launch_refuses_real_config_in_sim`, `test_launch_refuses_unlisted_policy`, `test_every_argument_has_a_description`.
- [ ] **Step 2:** Run; expect FAIL. **Step 3:** Implement and rewire the H1-2 bringups, tests/stack and the Mac. **Step 4:** Run `golem_lowerbody_rl`, `golem_bringup` tests, `tests/docker`, `tests/stack/run.sh` (locomotion) and `tests/sim/test_sim_stack.sh --h12` (non-zero domain); expect PASS (14/14). Commit `golem_lowerbody_rl: one node and launch file; the H1-2 bringups run it`.

### Task 7: The G1 runs `golem_lowerbody_rl`

**Files:**
- Modify: `core_ws/src/golem_bringup/launch/g1/{g1_core,g1_sim,g1_real_robot}.launch.py` (include `lowerbody.launch.py` with `robot: g1`, `config` sim|real, `policy: homie`, inside the existing interlock and delays; `locomotion_config` and `LOCOMOTION_CONFIGS`/`locomotion_refusal` go — the include's refusal replaces them), `golem_bringup/test/test_g1_launch.py`, `test_nav_launch.py`; `tests/g1/{test_sim_bringup.py,test_sim_nav.py,README.md}` (logger `lowerbody_controller_node`, the Task 6 log lines, the run config's `band.max_wait`); `docs/g1/*`
- Test: port `G/test/test_locomotion_dds.py`'s start-up/fault/heartbeat cases that matter on ROS to `test/test_node_safety.py` (starts the installed `safety_node` with the G1 `sim_split` config on a test domain, default network)

**Interfaces:**
- Consumes: Task 6's launch file and node.

- [ ] **Step 1: Failing tests:** `test_g1_launch.py` asserts the include's arguments (`robot g1`, `config sim|real`, `policy homie`) and delays; `test_node_safety.py::test_g1_legs_start_behind_safety_node`, `test_heartbeat_loss_stops_the_legs`.
- [ ] **Step 2:** Run; expect FAIL. **Step 3:** Implement. **Step 4:** Run the package and bringup tests, `tests/sim/test_sim_stack.sh --g1` (14/14), `tests/g1/test_sim_bringup.sh --walk` (22/22) and `tests/g1/test_sim_nav.sh` (13/13); expect PASS. Commit `golem_bringup: the G1 runs golem_lowerbody_rl`.

### Task 8: Delete `g1_homie`, guard the package, update everything that named it

**Files:**
- Delete: `core_ws/src/g1_homie/`
- Modify: `docker/docker-compose.yml` (drop the HOMIE bind mount if nothing reads `/home/code/g1_homie`; the package installs the weights), `docker/scripts/docker_common.sh` (`require_g1_policy`: drop, or check `golem_lowerbody_rl/policies/homie/` if a mount stays), `docker/RosDockerfile` comment (onnxruntime is for `golem_lowerbody_rl`'s HOMIE), `docker/mac/scripts/launch_ros_mac.sh` (`BRINGUP_STUB_DEPS` drops `g1_homie`), `core_ws/src/golem_bringup/package.xml`, `tests/docker/{test_workflows.py,test_shared_packages.py}` (add `golem_lowerbody_rl` to `SCANNED`), `golem_lowerbody_rl/test/test_no_sdk.py` (new: no `unitree_sdk2py`/`cyclonedds` import), `tests/robocasa/g1/test_g1_band.py`, `README.md`, `docs/g1/*`, CLAUDE.md/AGENTS.md (§1, §2, §3 mounts, §4 tables: one row for `golem_lowerbody_rl`, the G1 table loses `g1_homie`; §6; §8 HOMIE licence rule → `golem_lowerbody_rl/policies/homie/`; §8 Isaac: its H1-2 legs never become ready, since its `rt/lowstate` tick counts publishes; §9 pre-flight: measure each robot's leg-node `/lowstate` gap and tick lag on the ROS path before setting `real.yaml`'s `max_state_gap`/`max_latency`; §10)

- [ ] **Step 1:** `test_shared_packages.py` with the package added and `test_no_sdk.py`: run; expect PASS (Tasks 3-6 kept robot names out of code) — fix any finding in code, not in the test.
- [ ] **Step 2:** Delete and update; `git grep -n 'g1_homie\|locomotion_node' -- ':!docs/superpowers'` lists only CLAUDE.md/AGENTS.md's stale-install text.
- [ ] **Step 3:** Run `tests/docker`, package and bringup tests, `tests/robocasa/g1`; expect PASS. Commit `golem_lowerbody_rl: delete g1_homie; both robots' legs are one package`.

### Task 9: Closed loop on both robots and the Mac launch

- [ ] **Step 1:** Run on non-zero domains: `tests/sim/test_sim_stack.sh --h12` and `--g1` (14/14 each), `tests/g1/test_sim_bringup.sh --walk` (22/22), `tests/g1/test_sim_nav.sh` (13/13), `tests/stack/run.sh` (arms and locomotion). Expected: all PASS.
- [ ] **Step 2:** In the x86 `golem_ros` image (disposable), launch `h12_sim_mac.launch.py` by path with `GOLEM_LOWERBODY` set to each of `almi`, `walk`, `fame` and with it unset, and check the graph builds (`ros2 launch --print-description` or a short run without a sim) — the Mac's build of the package uses no onnxruntime.
- [ ] **Step 3:** Fix anything found in the task that owns it (new commits), then report results.
