# G1 vs H1-2 redundancy audit (2026-10-27, camera_ready 4de2506): six agent reports

Controller note: F_crosscutting §6 item 1 (g1_control ignores relative/slow_mode) is WRONG: g1_control/utility/goal_checks.py aborts those goals in all three servers (verified).

---

# Redundancy audit A: the safety layers (h12_safety_layer vs g1_safety_layer, estop, `rt/safety/*` conventions)

Branch `camera_ready`, HEAD `4de2506`. Read-only analysis; nothing was run (no tests, no containers). All similarity
figures are `difflib.SequenceMatcher` ratios over source lines (package name normalised), computed by me.

## Summary: top consolidation candidates, ranked by (lines saved x confidence) / risk

1. **F1, the byte-identical spine (class A, about 660 duplicated Python lines per copy).** `chunk_logger.py`,
   `combine_chunks.py`, `record_limits.py`, the dummy `publisher.py`/`subscriber.py`, 99 lines of config helpers,
   `clone_cmd`/`_cmd_arrays`/`clip_low_cmd`, and the `--config` path helpers are the same text in both packages.
   Move into one new robot-agnostic pure-Python package. G1 adopts it first (no hardware history, 2.2k lines of tests);
   H1-2 follows through a submodule PR. Effort M, risk low if the functions move verbatim.
2. **F2, accidental divergence, not line savings (fix before or while consolidating).** The G1 layer was forked from the
   callback-era H1-2 layer and lacks the H1-2 fixes of `e0396fc`/`5182fbe`/`fb3f4d4` (newest-only relay, cheaper publish
   path, readable estop reasons, configurable upper watchdog). The H1-2 layer in turn lacks G1's lower/lowstate watchdogs,
   boot-time silence, CLOCK_BOOTTIME timing, single-writer latch and lowstate-stamped headers. Effort S-M each; the
   H1-2 direction touches hardware behaviour, so it needs the owner.
3. **F3, limit derivation and joint tables (class A/B, 4 copies of about 65 lines, 2 identical H1-2 joint tables).** The two
   safety layers AND the two controllers (`h12_ros2_controller`, `g1_control`) each re-implement
   `_load_joint_config`/`_derive_q_limits`/`_process_*_limits`, with different clip defaults in the controllers (0.001/0.10/0.25 vs
   0.01/0.5/0.5). One shared limits module fed by one joint table per robot (G1 already has `g1_description`; H1-2 has none).
   Effort M, risk low-medium.
4. **F4, `LatestReader` + tick-lag logic (class A/B).** Duplicated between `h12_safety_layer` and `h12_ros2_controller`,
   re-invented a third time in `g1_locomotion/driver.py`, and absent from `g1_safety_layer`/`g1_control`, which still use
   Unitree's listener+queue(10). Effort S, risk low for the move, medium for G1 behaviour change.
5. **F5, the `SafetyLayer` class itself (class B pieces, class C overall).** Share only the estop-latch/report/sample-logging
   pieces (about 150 lines); do NOT merge the relay loops now (callbacks vs polling, engage gate, arm_sdk hand-back).
   Effort L for a full merge, high risk; the H1-2 layer has almost no unit tests.
6. **F6-F9, smaller:** logging (`console.py` vs `safety_logger.py`, class C); the three `rt/safety/` prefix guards and the
   header-stamping protocol; three copies of the `EstopStatusRaw` type across `estop` and `h12_safety_layer`; package
   scaffolding/docs/lock files (class A, trivial).

Biggest test fact: the G1 layer has about 2,160 lines of pytest in 63 test functions (including in-process DDS tests); the H1-2 layer has 2 unit
tests on its logger plus `tests/stack/test_heartbeat_clock.py`. Any change to H1-2 safety code lacks a guard today.

---

## 1. Inventory

### 1.1 `core_ws/src/h12_safety_layer` (git submodule)

- Submodule: yes, `correlllab/h12_safety_layer`, pinned `a178f0b` (branch `camera_ready_main_merge`; also on `origin/camera_ready`).
  `git submodule status` shows `a178f0b... (heads/camera_ready_main_merge)`; submodule worktree clean.
- Executable (setup.py): `safety_node = h12_safety_layer.ros2.safety_node:main` (ROS node name `safety_node`).
- ROS interface: **publishes** `/safety/heartbeat` (`std_msgs/Bool`, 10 Hz, rclpy STEADY clock; True until estop latches)
  (`ros2/safety_node.py:47-70`). No ROS subscriptions, services, actions or parameters; CLI arg `--config`.
- Raw Unitree/Cyclone DDS (the real I/O):
  - sub `rt/lowstate` (`LowState_`, `LatestReader`: KEEP_LAST(1), best effort) (`core/safety_layer.py:195-197`);
  - sub `rt/safety/lowcmd_in` (full mode), or `rt/safety/lowcmd_lower_in` + `rt/safety/lowcmd_upper_in` (split) (`LowCmd_`) (`:198-209`);
  - pub `rt/lowcmd` (`LowCmd_`, `ChannelPublisher`, CRC recomputed per message) (`:97`, `:439-446`);
  - sub `h12/estop_status_raw` (`IdlStruct` typename `h12.estop.EstopStatusRaw`) when `estop.enabled` (`core/estop_subscriber.py`).
- Config/YAML (share dir `config/`): `default_safety_{full,split}`, `locomotion_safety_full`, `relax_safety_{full,split}`,
  `sim_safety_split`, `tight_safety_{full,split}` (8 files). Env: `ROS_DOMAIN_ID` (wins over `network.domain_id`).
- Other executables/scripts (not console-scripts): `script/safety_layer_main.py` (CLI with domain-0 y/N prompt),
  `unsafe/send_fixed_positions.py`, `unsafe/send_random_positions.py` (publish `rt/safety/lowcmd_in` only),
  `utility/{combine_chunks,record_limits,publisher,subscriber,estop_status}.py`.
- Output files: `.npz` chunks via `ChunkLogger` under `logging.base_dir` (default `/tmp/h12_safety_layer`); JSON event lines to the ROS logger.
- Size: `h12_safety_layer/` 2,227 Python lines; tests 111 lines (3 ament lint stubs + `test/test_safety_logger.py`, 38 lines).
- Consumers of its outputs: `/safety/heartbeat` is read by `h12_lowerbody_rl/.../lowerbody_controller_node.py:223`,
  `lmg_deploy/.../marl_deploy_node.py:253`, `h12_skills/h12_skills/base.py:393`.

### 1.2 `core_ws/src/g1/g1_safety_layer` (in-tree, not a submodule)

- Added in `a7fca29` (2026-10-26), ported from G1_GOLEM. Depends on `g1_description` (`package.xml:10`).
- Executable: `safety_node = g1_safety_layer.ros2.safety_node:main` (ROS node name `safety_node`).
- ROS interface: **none** (no publisher, subscriber, service, action or parameter). rclpy is used only for the node shell, the
  logger and to join the right domain (`ros2/safety_node.py:72-114`). There is no heartbeat.
- Raw DDS: sub `rt/lowstate`; sub `rt/safety/lowcmd_upper_in` (arm_sdk, split), `rt/safety/lowcmd_lower_in` (split),
  `rt/safety/lowcmd_in` (full), each via `ChannelSubscriber.Init(cb, 10)` (`core/safety_layer.py:182, 248-269`);
  pub `rt/arm_sdk` (arm_sdk mode) or `rt/lowcmd` (split/full) (`:159`).
- Config/YAML: `arm_sdk`, `split`, `full`, `sim_arm_sdk`, `sim_split` (5 files). Env: `ROS_DOMAIN_ID`, `G1_DDS_INTERFACE`, `G1_SAFETY_LOG_DIR`.
- Other: `script/safety_layer_main.py` (CLI), `utility/{combine_chunks,record_limits,publisher,subscriber}.py`.
- Size: `g1_safety_layer/` 2,281 Python lines; tests 2,231 lines, 63 test functions outside the lint stubs (`test_g1_safety.py` 546 lines/32 tests,
  `test_hardening_dds.py` 656/17, `test_lowcmd_modes_dds.py` 380/8, `test_node_exit.py` 383/4, `test_arm_sdk_mode_dds.py` 169/2, `conftest.py` 24, 3 lint stubs).

### 1.3 `core_ws/src/estop` (git submodule)

- Submodule: yes, `correlllab/estop`, pinned `c30fb50` (`main`).
- Executables: `estop_node`, `estop_raw_dds_subscriber`, `estop_serial_status_test`. Launch `launch/estop.launch.py`.
- `estop_node`: reads an Arduino over serial; publishes `estop/triggered` and `estop/plugged_in` (`std_msgs/Bool`, relative names,
  100 Hz) and, via CycloneDDS, `h12/estop_status_raw` (`h12.estop.EstopStatusRaw`: `triggered`, `plugged_in`, `sequence_id`,
  `stamp_ms`; `estop/estop_node.py:21-31, 130-185`). `triggered` is forced True while unplugged (`fail_safe_triggered_on_disconnect`).
- **How it plugs into the safety layers:** only into the H1-2 one. `h12_control.launch.py:23-58` starts `estop_node` when
  `use_estop:=true` (real bringup, lmg real bringup, `h12_safety_grippers`), and `h12_safety_layer`'s `EStopSubscriber` reads
  `h12/estop_status_raw` from `estop.estop_topic` (`core/estop_subscriber.py`, `core/safety_layer.py:119-126, 345-372`;
  `estop.enabled: false` in `sim_safety_split.yaml`). Nothing else consumes `/estop/*` (only lmg bag metadata mentions them).
  The G1 loader rejects an `estop:` section outright (`g1_safety_layer/core/config.py:216-219`, test
  `test_old_external_estop_section_is_rejected`), and `g1_core.launch.py` has no estop node.

---

## 2. Module-by-module comparison (file level)

| Module (H1-2 / G1) | Lines H / G | Similarity | Class | Note |
|---|---|---|---|---|
| `core/chunk_logger.py` | 144 / 146 | 0.986 | A | only diff: G1 imports `console` and uses `console.write_line` for one print (`:93` vs `:95`) |
| `utility/combine_chunks.py` | 57 / 57 | 1.000 | A | identical once the package name is normalised (raw diff: the two `/tmp/<pkg>` defaults at lines 53 and 55) |
| `utility/record_limits.py` | 210 / 210 | 1.000 | A | identical (imports each package's own `joint_limits`) |
| `utility/publisher.py`, `subscriber.py` | 34, 38 / 34, 38 | 0.971, 0.974 | A | only the default config path in the last line differs (`config/default_safety_full.yaml` vs `config/full.yaml`); they are SDK latency demos, nothing safety-specific |
| `core/config.py` | 234 / 331 | 0.729 | A + B | 8 helper functions identical (99 lines), see F1 |
| `core/safety_checks.py` | 122 / 189 | 0.572 | A + B | 3 functions identical (53 lines); `check_estop_limits`, `make_estop_cmd` diverge, see F2 |
| `core/joint_limits.py` | 128 / 39 | 0.048 | D | different data: 27 literal H1-2 joints vs 29 derived from `g1_description` |
| `core/safety_layer.py` | 567 / 821 | 0.392 | B/C | common ancestry, see F5; G1 best matches H1-2 revision `8ad5988` (0.567), current only 0.392 |
| `ros2/safety_node.py` | 108 / 118 | 0.558 | B/D | 2 helper functions identical; G1 adds signal hand-back, H1-2 adds the heartbeat |
| `script/safety_layer_main.py` | 55 / 80 | 0.563 | B/D | same helpers; G1 adds exit handling |
| `core/estop_subscriber.py`, `latest_reader.py`, `safety_logger.py`, `unsafe/*`, `utility/estop_status.py` | H1-2 only | - | - | see F4, F6, section 6 |
| `core/arm_sdk.py`, `clock.py`, `console.py`, `exit_signals.py` | G1 only | - | - | see F2, F6, section 6 |
| `docs/architecture.md`, `docs/config.md`, `README.md` | 227/227/158 vs 248/220/198 | 0.724 / 0.447 / 0.208 | B | prose forks, not worth consolidating |
| `setup.cfg`, `.gitignore`, `test_{copyright,flake8,pep257}.py` | - | 1.000 | A | scaffolding |
| `setup.py`, `package.xml`, `pyproject.toml`, `requirements.txt`, `uv.lock` | - | 0.97, 0.88, 0.90, 0.98, 0.99 | A | G1 `pyproject.toml:4` still says "Safety layer for h12 robot" |

Config YAML schema (the loaders return the same dict shape for the shared part):

- Shared: `mode`, `topics.{low_cmd_in,low_cmd_lower_in,low_cmd_upper_in,low_cmd_out,low_state}` (identical defaults, except G1's
  default `low_cmd_out` is `rt/arm_sdk` in arm_sdk mode), `network.{domain_id,interface}`, `control.publish_hz`,
  `limits.{clip,estop}.{position_offset,velocity_ratio,torque_ratio}`, `limits.clip.{kp_max,kd_max}` (with the legacy
  `kp_abs`/`kd_abs` and `enforce` aliases), `logging.{enabled,base_dir,chunk_prefix}`. Same defaults in both loaders
  (0.01 / 0.5 / 0.5 / 300 / 30), and the loaders emit the same flattened `limits` keys (`q_clip_limits`, ..., `tau_estop_limits`).
- H1-2 only: `estop.{enabled,estop_topic,poll_hz,upper_stale_seconds,state_lag_ms}`.
- G1 only: `robot` (real|sim, mandatory), `control.{lower_stale_estop_seconds,state_stale_estop_seconds}`,
  `engage.{max_state_age_seconds,mode_machine,max_position_error}`, `arm_sdk.{engage_seconds,release_seconds,waist_kp_min}`,
  `estop_response.{mode,kd}`, mode `arm_sdk_mode`.
- Per-joint lists are 27 (H1-2) vs 29 (G1) entries; those tuning values are robot data and stay separate (section 7).

---

## 3. Findings

### F1. The identical spine: logging tools, config helpers, command clipping, path helpers (class A)

- **Locations (H1-2 / G1):**
  - `h12_safety_layer/core/chunk_logger.py:10-144` / `g1_safety_layer/core/chunk_logger.py:12-146`
  - `utility/combine_chunks.py:1-57` / same; `utility/record_limits.py:1-210` / same; `utility/publisher.py`, `subscriber.py` (34, 38 lines) / same
  - `core/config.py` `_load_yaml` 21-27 / 30-36, `_load_joint_config` 30-41 / 39-50, `_derive_q_limits` 44-56 / 61-73,
    `_process_clip_limits` 59-86 / 76-103, `_process_estop_limits` 89-104 / 106-121, `init_channel_factory` 126-132 / 176-182,
    `init_channel_factory_guard` 135-150 / 185-200
  - `core/safety_checks.py` `clone_cmd` 16-18 / 18-20, `_cmd_arrays` 20-26 / 22-28, `clip_low_cmd` 28-70 / 30-72
  - `ros2/safety_node.py` `_ensure_yaml_extension`+`_resolve_config_path` 24-38 / 21-35 and `script/safety_layer_main.py` 16-29 / 21-34
    (the two path helpers are four copies in total)
- **Classification:** A.
- **Evidence:** normalised function-body diff gives ratio 1.00 for each function listed (an AST function diff with the package
  name replaced by `PKG`); `record_limits.py` differs only in the `from <pkg>.core.joint_limits import` line (`:18`), `combine_chunks.py` only in two `/tmp/<pkg>` default paths (`:53, :55`), `publisher.py`/`subscriber.py` only in the default config path on their last line.
  `chunk_logger.py` diff is exactly: `+from PKG.core import console` and `- print('ChunkLogger stopped')` / `+ console.write_line(...)`.
  Lines: 99 (config helpers) + 53 (checks) + 144 + 57 + 210 + 72 (publisher+subscriber) + 28 (path helpers x2 files) = about 663 duplicated lines per copy.
- **What differs:** nothing of substance. Both carry the same dead block (a commented-out `changed = np.logical_or.reduce(...)` in
  `clip_low_cmd`, which CLAUDE.md section 8 says to delete) and the same wrong return annotation (`-> tuple[LowCmd_, int]`,
  returns `LowCmd_`). `publisher.py`/`subscriber.py` are Unitree SDK latency demos on `demo/topic`, unrelated to safety.
- **Consolidation proposal:** a new pure-Python ament package, for example `core_ws/src/golem_safety_common/` (no ROS dependency;
  both safety layers already depend on `unitree_sdk2py`, `numpy`, `pyyaml`), with `chunk_logger`, `combine_chunks`, `record_limits`
  (taking the joint table as a parameter), `config_io` (`load_yaml`, `load_joint_config`, channel-factory/domain-guard,
  `resolve_config_path(package, name)`) and `checks` (`clone_cmd`, `cmd_arrays`, `clip_low_cmd`). The robot package keeps its
  joint table and the `load_config` that assembles its own YAML sections. Delete the two demo scripts from G1 (and from
  H1-2 if the owner agrees). Effort: S for G1 adoption, M including the H1-2 submodule PR.
- **Risk:**
  - Real robot: H1-2 runs these functions on hardware, but moving them verbatim is behaviour-neutral. Guard it with a
    characterisation test first: load all 8 H1-2 YAMLs before/after and compare every `limits` array, and run
    `clip_low_cmd`/`check_estop_limits` on randomised commands against the pre-move implementation. G1 has never run on its robot.
  - Boundaries: `h12_safety_layer` and `estop` are separate repos; H1-2 adoption means a commit+push in
    `correlllab/h12_safety_layer` plus a pin bump (CLAUDE.md section 5), and makes the submodule depend on a GOLEM-tree package,
    so its standalone `uv sync` flow (README: "place Unitree SDK at ../../../unitree_sdk2_python") changes.
    The G1-first path avoids this until the owner agrees.
  - Behaviour changes: none if verbatim. Folding G1's `slots` parameter into `check_estop_limits` is F2.
  - Tests: G1's `test_g1_safety.py` already covers config validation, `make_estop_cmd`, `clipped_gains` and `check_estop_limits`
    over slots; H1-2 has none.

### F2. Divergence between the two relay implementations (class B with accidental divergence; this finding is about bugs-in-waiting more than line count)

The G1 `core/safety_layer.py` is a fork of the callback-era H1-2 layer. H1-2 then evolved in `e0396fc`, `58d0f99`, `5182fbe`,
`fb3f4d4`, `5b08888` (all 2026-07 to 10) and G1 evolved on its own. Evidence of the fork point: the G1 file matches H1-2
revision `8ad5988` at 0.567 and the current `a178f0b` at only 0.392 (similarity vs revision: `5b08888` 0.557, `e0396fc` 0.401,
`ad06903` 0.551). Method-level similarity of the shared methods (same names): `_log_sample` 0.89, `_trigger_estop` 0.71,
`_validate_mode` 0.71, `_on_low_state` 0.61, `_clip_cmd_or_estop` 0.58, `_check_upper_watchdog_locked` 0.57, `_publish_checked` 0.40.
`_log_sample` differs only in where the q_cmd arrays come from and an early return (diffed). The estop report builder
(H1-2 `_get_estop_trigger_report` 468-486 / G1 `_complete_estop_report` 725-744) is the same computation
(`action_ms`, `decision_to_action_ms`, `trigger_to_action_ms`) with the lock moved to the caller.

**2a. Fixes the H1-2 side has and the G1 side lacks (flagged: G1 should probably get them).**

| # | H1-2 (`core/safety_layer.py` unless noted) | G1 | What the H1-2 side gained |
|---|---|---|---|
| 1 | Listener-free `LatestReader` polled by one relay loop, `core/latest_reader.py:18-46`, `safety_layer.py:195-209, 238-264, 293-323` | `ChannelSubscriber.Init(cb, 10)` for state and every input, `safety_layer.py:182, 254-269, 343-375, 626-673` | commit `e0396fc` message: under GIL contention the Cyclone receive thread waits on Python, "packets queue in the socket", causing a standing delay; a slow cycle lowers the publish rate instead of delaying commands. I confirmed the mechanism in `unitree_sdk2py/core/channel.py:56, 84-107`: the SDK listener `take(1)`s then `BQueue.Put(sample)` **without replace**, so with 10 queued samples a newer command is dropped while older ones are still processed. The commit's bench (400 Hz upper, 500 Hz lower and state): median latency 3.7 -> 2.6 ms, publish rate 342 -> 421 Hz, dropped upper commands 359/5515 -> 0. I did not re-run it. |
| 2 | Shallow-copy merge `:266-281`; `_publish_checked` stores the message, arrays built lazily (`:439-446`, `_cmd_arrays` only on estop/log); `_log_sample` returns early with logging off (`:495-498`) | deepcopy merge `:448-457, 468`; `_publish_checked` builds five 29-element numpy arrays per write `:681-695`; `_log_sample` builds about 12 arrays per state even with logging disabled `:751-821` | CPU per 500 Hz cycle on a laptop |
| 3 | dq/tau estop reason carries the limit and multiple, `safety_checks.py:72-76, 91-108` (commit `5182fbe`) | `safety_checks.py:142-173` prints only the value | operator can see how far past the limit it was; also `_limit_multiple` guards a zero limit |
| 4 | `estop.upper_stale_seconds`, `<= 0` disables, `config.py:169-171`, `safety_layer.py:84-88, 325-343` (commit `fb3f4d4`) | constant `UPPER_STALE_ESTOP_SECONDS = 2.0`, `safety_layer.py:47, 577-592`, not configurable | tuning for tasks that hold still |
| 5 | `/safety/heartbeat` Bool, `ros2/safety_node.py:47-70` | none | consumers on the H1-2 side only |
| 6 | Transport-lag estop against the robot tick on domain 0, `safety_layer.py:374-392, 400-407` | none; G1 has receive-time staleness only (2b) | different mechanism for a similar job (class C) |

Note on row 1: the G1 `rt/lowstate` watchdog stamps `received = watchdog_time()` at handler run time
(`safety_layer.py:628`), so a backlog in the 10-deep queue delays both the state and its staleness verdict. I have not measured this on a G1 stack.

**2b. Fixes the G1 side has and the H1-2 side lacks (flagged for the H1-2 owner; each is a behaviour change on hardware).**

| # | G1 | H1-2 |
|---|---|---|
| 1 | Lower-input watchdog `_check_lower_watchdog_locked` `:594-608`, `control.lower_stale_estop_seconds` 0.2 s | none: if the leg controller dies, its last command is relayed indefinitely (H1-2's heartbeat runs the other way, controller watches the safety layer, `lowerbody_controller_node.py:269-282`) |
| 2 | `rt/lowstate` staleness watchdog once engaged `:610-624`, 0.1 s | none: `_update_state_lag` needs a sample to arrive late; with no `rt/lowstate` at all, nothing latches (`_relay_loop` just skips `None`, `:302-304`) |
| 3 | `watchdog_time()` = CLOCK_BOOTTIME for every watchdog, ramp and loop sleep (`core/clock.py`, used `safety_layer.py:349, 358, 371, 529, 628`); test `test_wall_clock_steps_neither_fire_a_watchdog_nor_stall_the_loop` | `time.time()` for the relay timing and the upper watchdog (`safety_layer.py:301, 316, 320-323, 264`): a wall-clock step can fire or hide the 2 s watchdog or stall the sleep. Only the heartbeat timer is on STEADY time (`safety_node.py:56-60`, commit `5b08888`) |
| 4 | Single-writer latch `_check_single_writer_locked` `:304-321` (DDS publication handle per input) | none (and I did not check that `LatestReader.take` samples carry `sample_info`; G1 relies on cyclonedds attaching it to callback samples) |
| 5 | Silent until every input has sent a command and the command passes the engage gate `:377-434, 464-476` | publishes the zero-gain mode-0 boot command at 500 Hz from startup: `_desired_cmd_*` start as `make_estop_cmd(LowCmdDefault())` (`:78-82`) and `_relay_loop` always publishes when not estopped (`:316-323`). Partly robot-specific (G1 shares joints with Unitree through arm_sdk) |
| 6 | Header stamped by the safety layer: `mode_pr = MODE_PR`, `mode_machine` echoed from `rt/lowstate` (`:516-522`) | the merged header is copied from `_last_cmd`, the last message from any input (`:228`, `:266-281`), so every controller must hard-code the same header: `h12_ros2_controller/core/channel_interface.py:283-284` (mode_machine 6), `lowerbody_controller_node.py:73, 336-347`, `marl_deploy_node.py:439-446` ("the safety layer copies the header of whichever channel spoke last"). H1-2 sims report `mode_machine` 0 (not set), and the real unit's value is unverified, so echoing is not a drop-in change |
| 7 | Exit hand-back: `shutdown()` `:197-222`, `exit_signals.py`; writes the estop command or ramps arm_sdk weight to 0 for 0.1 s, then exits non-zero if it could not | `stop()` `:156-169` just stops publishing; the leg controller is the only other defence |
| 8 | Sim configs refuse DDS domain 0 inside the loader (`config.py:146-167, 269-275`, mandatory `robot:`) | the loader accepts anything; domain protection is the wrapper scripts only |
| 9 | Non-blocking stdout (`core/console.py`, 113 lines: queued writer thread, tolerates closed or stalled stdout; test `test_console_never_blocks_on_a_stalled_stdout`) | `PrintSafetyLogger.print(..., flush=True)` (`safety_logger.py:13-19`) can block on a stalled stdout; ROS logging in `safety_node` is injected (`safety_node.py:47`) |

**2c. Differences that are real robot differences (leave alone):** arm_sdk mode and `core/arm_sdk.py` (`with_weight`,
`zero_legs`, ramps), the engage-gate parameters (mode_machine 5, 29 slots, 0.15 rad), `check_waist_kp`, `estop_response`
(zero vs damp), the 27/29 joint split at slot 12, and the estop response timing: H1-2 writes the zero-gain command
`ESTOP_FINAL_CMD_COUNT = 50` times (about 0.1 s at 500 Hz) and then **goes silent** (`safety_layer.py:55, 306-314`, commit
`58d0f99`), G1 split/full keep writing the estop command for ever (`:478-488`) and arm_sdk keeps holding weight 0. The two
are consistent with each other's documentation, but nobody recorded that the difference is deliberate; it needs an owner decision
(see section 6) because `damp` mode on G1 plausibly needs a continuous command.

**Consolidation proposal for F2:** treat as a sequence, not a merge. (i) G1-only, no hardware history: port 2a rows 1-4
(relay loop on `LatestReader`, lazy arrays, readable estop reasons, configurable upper watchdog). Effort M (row 1 is the real work;
the engage gate and watchdog logic in `_publisher_loop` already ticks on its own thread, so the G1 loop mostly needs input
taking moved from callbacks to a poll at the top of each cycle). (ii) H1-2 side, owner decision: lower and lowstate watchdogs
(2b rows 1-3). Effort S each, hardware-affecting.
**Risk:** G1: low (not on hardware; 2,231 lines of tests, though `test_hardening_dds.py` timing assumptions would need a re-run
after the loop change). H1-2: high (the newest relay `e0396fc` says "Not yet run on the robot or in sim" in its own message; I
do not know whether it has since), crossing the submodule boundary, new estop conditions on a running robot, no unit tests to
guard them. **Tests that would guard it:** G1's `test_hardening_dds.py`, `test_lowcmd_modes_dds.py` and `test_node_exit.py` for the G1 loop change;
for H1-2, port their DDS fixtures (`_StateStream`, `_Robot`, `conftest.py` serialise-lock) first.

### F3. Limit derivation and joint tables: four copies of the derivation, at least two H1-2 joint tables (class A and B)

- **Locations:**
  - safety layers (class A, identical): `h12_safety_layer/core/config.py:30-104` / `g1_safety_layer/core/config.py:39-121`
  - controllers (class A with each other, class B vs the safety layers):
    `h12_ros2_controller/utility/controller_config.py:142-202` (`_load_joint_config` 142, `_derive_q_limits` 169, `_process_clip_limits` 180,
    `build_processed_limits` 191; no `_process_estop_limits` in this file) /
    `g1_control/utility/controller_config.py:57-132` (same functions, plus `_process_estop_limits` 106-115, `build_processed_limits` 117-132)
  - H1-2 joint tables: `h12_safety_layer/core/joint_limits.py:1-128` and `h12_ros2_controller/utility/joint_limits.py` (163 lines,
    `JOINT_POSITION_LIMITS`, `JOINT_VELOCITY_LIMITS`, `JOINT_TORQUE_LIMITS`, plus `joint_definition.py`, 189 lines, with the joint names)
  - G1 table: one source, `g1_description.joints.BODY` consumed by `g1_safety_layer/core/joint_limits.py:1-39`
- **Classification:** A (safety-layer pair; controller pair), B (controller vs safety layer: parameter names, simplified error messages).
- **Evidence:** between `h12_ros2_controller` and `g1_control`, `_load_joint_config`, `_derive_q_limits`, `_process_clip_limits`,
  `get_publisher_clip_limits` are ratio 1.00; against the safety layers, 0.40 / 0.43 / 0.26 (renamed arguments and bodies, same algorithm).
  I executed the H1-2 constants from both files: `JOINT_POSITION_LIMITS`, `JOINT_VELOCITY_LIMITS` and `JOINT_TORQUE_LIMITS` in
  `h12_ros2_controller` equal `URDF_POSITION_LIMITS`, `URDF_VELOCITY_LIMITS`, `URDF_TORQUE_LIMITS` in `h12_safety_layer` entry for entry.
- **What differs:** accidental divergence in the **defaults**: the safety layers default `position_offset 0.01`, `velocity_ratio 0.5`,
  `torque_ratio 0.5`; both controllers' `_process_clip_limits` default 0.001 / 0.10 / 0.25 (`h12_ros2_controller/utility/controller_config.py:180-189`,
  identical in `g1_control`).
  A controller config that omits `limits` therefore clips tighter than the safety layer would (the controllers' estop defaults in `g1_control` do match the layer's 0.01/0.5/0.5); I did not check whether any shipped controller YAML omits them.
  Robot-specific: joint counts and tables (27 vs 29).
- **Consolidation proposal:** in the F1 package add `limits.py` with a `JointTable` (names, position/velocity/torque limits) and
  `process_clip_limits(policy, table, defaults)`, `process_estop_limits(...)`, used by both safety layers and both controllers.
  Give H1-2 the same single-source pattern G1 has (`g1_description` generated the table from the URDF): one H1-2 joint table
  (natural home: the description package `h12_ros2_model`, which is itself a separate submodule) that the safety layer, the
  controller, and the lowerbody/lmg tests read; `lmg_deploy/test/test_controller.py`, `h12_lowerbody_rl/reference/*`, and
  `h12_ros2_controller/example/motor_record.py` also embed joint names. Effort M.
- **Risk:** real robot: pure refactor if the defaults stay as each caller passes them explicitly; changing the controller defaults to
  match is a behaviour change (tighter or looser clipping). Boundaries: crosses `h12_safety_layer`, `h12_ros2_controller`,
  `h12_ros2_model` (all separate repos) and `g1_control`; announce as an interface change. Tests: G1
  `test_config_limits_cover_the_29_g1_motors`; `h12_ros2_controller/test/test_controller_config.py` exists; add the before/after snapshot of derived limits.

### F4. `LatestReader` and tick-lag tracking (class A/B, plus a class C re-invention)

- **Locations:** `h12_safety_layer/core/latest_reader.py:18-46` (`LATEST_ONLY_QOS`, `TAKE_BATCH`, `LatestReader`) and the inlined lag
  math `core/safety_layer.py:47-48, 374-392` / `h12_ros2_controller/core/latest_reader.py:19-92` (`LatestReader`, `TickLagTracker`,
  `sdk_participant`); independent: `g1_locomotion/driver.py:26, 69-77` (`NEWEST_ONLY` QoS plus raw `DataReader`s and a `WaitSet`).
- **Classification:** A/B (the two `LatestReader` classes: ratio 0.89, only the `__init__` signature formatting differs; the lag
  math in `_update_state_lag` is the same algorithm as `TickLagTracker.update`, constants `STATE_LAG_DRIFT_MS_PER_S` = `LAG_DRIFT_MS_PER_S` = 0.5,
  `TICK_WRAP` 1<<32); C for `g1_locomotion`.
- **Evidence:** the two class bodies diff to a single 3-line signature difference; the QoS (`KeepLast(1)`, `BestEffort`) and comment rationale are the same.
  `g1_safety_layer` and `g1_control` still use `ChannelSubscriber.Init(cb, 10)` (`g1_control/core/channel_interface.py:42`), which
  `docs/superpowers/specs/2026-10-26-g1-port/1_g1_robot_packages.md:218` already notes is worth back-porting.
- **What differs:** the controller copy adds `sdk_participant()` (reuses the Unitree `ChannelFactory` participant so the interface/domain
  match); the safety layer creates its own `DomainParticipant(domain_id)` (`safety_layer.py:92`). That is an unexplained divergence in which participant
  carries the readers (the safety layer's reader would not honour `network.interface`; I did not verify whether that matters on the robot).
- **Consolidation proposal:** move `LatestReader`, `sdk_participant`, `TickLagTracker` into the F1 package (a `dds.py`), have
  `h12_safety_layer` use `TickLagTracker` instead of the inlined math, and let `g1_safety_layer`/`g1_control` adopt it as part of F2 row 1.
  `g1_locomotion` could adopt it too (class C -> shared), but its WaitSet-driven design is a deliberate wake-on-data loop; leave unless the owner wants it.
  Effort S (move) + M (G1 adoption).
- **Risk:** the move is behaviour-neutral. The G1 adoption changes G1 timing (F2). Crosses `h12_ros2_controller`
  and `h12_safety_layer` submodules. Test: needs DDS; G1 DDS tests on domain 97 cover the consumers.

### F5. The `SafetyLayer` class: shared skeleton, different runtime (class B pieces, class C overall)

- **Locations:** `h12_safety_layer/core/safety_layer.py:58-567` / `g1_safety_layer/core/safety_layer.py:71-821`.
- **Shared pieces (class B):** `_trigger_estop` (H1-2 448-466 / G1 698-723: diff is only the arm_sdk branch and `kd` argument),
  estop timing report (468-486 / 725-744), `_log_sample` (495-567 / 751-821), `_clip_cmd_or_estop` (226-236 / 283-302),
  `_on_low_state` diagnostics block (394-437 / 626-673), `_validate_mode`, the constants `LOG_*`, `UPPER_STALE_ESTOP_SECONDS`,
  the lock/latch invariants and the one-time `estop_command_written` event. About 150-200 lines are the same idea.
- **Different runtime (class C):** relay loop (H1-2 polls `LatestReader`s in `_relay_loop` 293-323; G1 handles DDS callbacks and
  runs a separate `_publisher_loop` 524-557), mode set (H1-2: full, split; G1: arm_sdk, split, full), engage/hand-back state machine (G1 only),
  external estop thread (H1-2 only, `_estop_monitor_loop` 345-372).
- **What differs:** see F2 and section 6.
- **Consolidation proposal:** do not unify yet. Extract a robot-agnostic `EstopLatch` (flag, reason, deferred timing report) and a
  `StateSampleLogger` (the `_log_sample` dict builder, parameterised by the last-command arrays) into the F1 package, ~150 lines.
  Revisit a full shared runtime only after (a) G1 has run on its robot, (b) H1-2 has a test suite, (c) F2 has converged the loops.
  A converged runtime would be H1-2's polling loop plus G1's engage gate, watchdogs, hand-back and header stamping, with mode strategy
  objects (`FullBody`, `Split`, `ArmSdk`) supplying `desired_command()`, `estop_command()` and the engage rule. Effort L.
- **Risk:** highest of all findings. H1-2 is the hardware default; submodule boundary; behaviour changes in estop and relay timing;
  only 2 H1-2 unit tests. Guard: port the G1 DDS tests so they run against both packages through a config fixture (`TEST_DOMAIN = 97`,
  `LowStateDefault`/`LowCmdDefault` fixtures are robot-neutral except `MOTOR_COUNT`).

### F6. Logging helpers (class C)

- **Locations:** `h12_safety_layer/core/safety_logger.py:1-19` (`SafetyLogger` Protocol + `PrintSafetyLogger`, injected so
  `safety_node` passes `get_logger()`) vs `g1_safety_layer/core/console.py:1-113` (queued, thread-written stdout, never blocks or raises).
- **Classification:** C. Both emit JSON event lines with `json.dumps(data, sort_keys=True)` with the same event names
  (`relay_started`, `relay_stopped`, `estop_command_written`) and `ChunkLogger` is shared (F1).
- **What differs:** G1's queue exists because a stalled stdout must not block the hand-back (test: `test_console_never_blocks_on_a_stalled_stdout`);
  H1-2's injection exists so ROS launch output is not delayed (commit `d3d2725`). Both goals are valid.
- **Consolidation proposal:** keep H1-2's `SafetyLogger` Protocol as the interface and ship `console.py` as the non-blocking implementation
  (`ConsoleSafetyLogger`) in the F1 package; H1-2's ROS node would wrap `get_logger()` behind the same queue if it wants the guarantee. Effort S.
- **Risk:** low; small behaviour change on H1-2 only if adopted. The existing H1-2 `test_safety_logger.py` (patches `builtins.print`)
  would need updating if `PrintSafetyLogger` changes. Submodule boundary applies.

### F7. Package scaffolding and docs (class A/B, low value)

- `setup.cfg`, `.gitignore` (207 lines), the three ament lint tests: byte-identical. `setup.py` 0.972, `package.xml` 0.878, `pyproject.toml`
  0.900, `requirements.txt` 0.977, `uv.lock` 0.994 (519 lines each, differ only in the package name), `AGENTS.md` (identical bodies; the H1-2
  title even says "h12_ros2_controller"). `docs/architecture.md` 0.724, `docs/config.md` 0.447.
- Proposal: nothing structural; the shared package owns its own docs once F1 happens, and each robot's docs link to it for the shared parts.
  Fix G1 `pyproject.toml:4`. Effort S. Not worth a dedicated change.

### F8. The `rt/safety/*` conventions each controller uses (class B/C: parallel guards, divergent header protocol)

Topic set (identical names, identical `unitree_hg` types): `rt/safety/lowcmd_in` (full), `rt/safety/lowcmd_upper_in`,
`rt/safety/lowcmd_lower_in`. In ROS-typed nodes `/safety/lowcmd_*` is the same DDS topic (`rt/` prefix).

| Safety input | H1-2 writers | G1 writers |
|---|---|---|
| `rt/safety/lowcmd_upper_in` | `h12_ros2_controller` `frame_task_server` (DDS `ChannelPublisher`, `core/channel_interface.py:268-284`); `lmg_deploy` `marl_deploy_node` (ROS `LowCmd`, rows 12-26, `:257, 443`) | `g1_control` `frame_task_server` (DDS, `core/channel_interface.py:237-262`, default `:21`). `g1_locomotion` also **reads** it (`driver.py:72-74`) to learn the arm controller has started |
| `rt/safety/lowcmd_lower_in` | `h12_lowerbody_rl` `walking_node.py:99`, `fame_node.py:89`, `lowerbody_controller_node.py:224` (ROS `LowCmd`); `marl_deploy_node.py:256` (rows 0-11) | `g1_locomotion` `locomotion_node` (DDS, `driver.py:68`) |
| `rt/safety/lowcmd_in` | `h12_deploy_mjpc` `mjpc_fullbody_core` (`CMakeLists.txt:146` comment: "needs safety in full mode"); `h12_ros2_controller` `safety_full.yaml`/`debug.yaml`; `unsafe/send_*` | none (G1 `full.yaml` exists, the bringup refuses full mode) |

- **Guards.** The controllers refuse any `topics.low_cmd` not under `rt/safety/` in three independent copies of the same rule:
  `h12_ros2_controller/utility/controller_config.py:49-61, 270-271` (plus a second check in `LowCmdPublisher.__init__`,
  `core/channel_interface.py:268-276`), `g1_control/utility/controller_config.py:212-216` (config check only, no publisher-level check),
  `g1_locomotion/config.py:63-67`. Class B/C, about 6 lines each. Proposal: one `require_safety_input(topic)` in the F1 package; effort S.
  Difference to flag: the H1-2 publisher class has the second, defence-in-depth check; the G1 publisher class does not.
- **Header protocol.** Described in F2b row 6: H1-2 controllers must stamp `mode_pr 0` / `mode_machine 6` identically because
  the safety layer relays the header of whichever input spoke last; G1 controllers send placeholders (`mode_machine 0`, comment at
  `g1_control/core/channel_interface.py:247-250`) and the G1 safety layer stamps the real values. This is the clearest convention
  difference and the G1 design is the less fragile one; adopting it for H1-2 needs the real H1-2 `rt/lowstate.mode_machine`
  confirmed (the sims leave it 0). Behaviour change, hardware-affecting.
- **CRC.** Controllers on both robots compute a CRC on their inputs (`h12_ros2_controller/core/channel_interface.py:206-207`,
  `g1_control/core/channel_interface.py:173-174`, `g1_locomotion/driver.py:142`), but both safety layers overwrite it
  in `_publish_checked` before writing `rt/lowcmd`/`rt/arm_sdk` and never validate an input's CRC. It is dead work on the input
  path (the ROS-typed H1-2 lower-body nodes already skip it). Not a duplicate to consolidate, but a convention to document:
  inputs need no CRC.
- **Liveness coupling (class C).** H1-2: safety layer publishes `/safety/heartbeat`; lower-body, lmg and skills latch on a False beat
  or silence (`lowerbody_controller_node.py:269-282, 399-410`, `marl_deploy_node.py:343-347`, `h12_skills/base.py:1038-1046`). G1: no
  heartbeat; the arm controller's follow gate watches the safety layer's *output* topic (`g1_control/core/follow_gate.py`), and the walking node
  stops commanding so the safety layer's lower watchdog estops. Two ways to do the same job. Do not unify before the F2 divergence
  is resolved; if H1-2 gains the lower/lowstate watchdogs, consider a common `/safety/heartbeat` on the G1 for skills and tooling.
- **Controller-side clipping is a third tier** (both controllers clamp in `_enforce_limits` before publishing; both safety layers then clip and estop).
  Out of scope here; noted because F3's table/limit code feeds all three.

### F9. External estop plumbing: the `EstopStatusRaw` type and the subscriber (class A within H1-2, none on G1)

- **Locations:** `estop/estop/estop_node.py:26-31`, `estop/estop/raw_dds_subscriber.py:35-40`,
  `h12_safety_layer/core/estop_subscriber.py:13-18`: three definitions of
  `IdlStruct(typename='h12.estop.EstopStatusRaw')` with the same four fields. The topic string `h12/estop_status_raw`
  appears in 5 Python files, 8 YAMLs, `estop.launch.py` and `config.py:172`.
- **Classification:** A (three verbatim dataclasses across two submodules).
- **What differs:** nothing, but the type name is the wire contract. If one copy drifts, `EStopSubscriber.latest` stays `None`
  and nothing latches (see section 8).
- **Consolidation proposal:** defer. The only fix that is not an ownership change is a checked-in comment and a test in the
  `estop` repo that asserts `h12_safety_layer`'s type name equals `estop`'s. If the G1 ever gets a physical estop box (today
  deliberately rejected), `estop` is reusable only after the `h12`-prefixed topic/type name becomes a parameter.
- **Risk:** crosses two submodules for a 6-line gain; not worth it now.

---

## 4. The `rt/safety/*` inputs from the safety layer's side (summary of who reads what)

- Both layers read only the topics named in their YAML `topics:` section, with the same default names; neither layer has a ROS subscription.
- H1-2 split mode merges `lower[:12] + upper[12:]` over all 35 `motor_cmd` entries (`safety_layer.py:277-280`); G1 split deep-copies the
  upper and overwrites slots 0-11 (`:453-457`). In both, `clip_low_cmd` only validates `MOTOR_COUNT` slots (27 H1-2, 29 G1), so slots 27-34
  (H1-2) or 30-34 (G1) of a 35-slot `LowCmd_` are relayed as received; G1 additionally zeroes legs in arm_sdk mode (`zero_legs`) and
  rewrites slot 29 with the weight (`with_weight`). A shared `clip_low_cmd` should zero or reject unused slots; neither layer does today.
  (I did not check what the robots do with those slots.)
- Neither layer has a rate limiter (no slew, delta or acceleration check in either `safety_checks.py`); "limits" means magnitude
  clamps on q/dq/tau/kp/kd and estop thresholds on measured q/dq/tau_est.

---

## 5. Tests

| | H1-2 `h12_safety_layer` | G1 `g1_safety_layer` |
|---|---|---|
| Unit | `test/test_safety_logger.py` (38 lines, 2 tests) | `test_g1_safety.py` (546 lines, 32 test functions, several parametrised) |
| In-process DDS (domain 97) | none | `test_arm_sdk_mode_dds.py` 169, `test_hardening_dds.py` 656, `test_lowcmd_modes_dds.py` 380 |
| Process/signal | none | `test_node_exit.py` 383 (subprocess, SIGINT/SIGTERM/SIGHUP, stalled stdout) |
| Lint | copyright, flake8, pep257 (identical stubs) | same |
| Repo-level | `tests/stack/test_heartbeat_clock.py` (needs ROS container; constructs `SafetyNode('sim_safety_split.yaml')`, so the constructor signature is pinned); `tests/stack/minimal_stack.launch.py` runs the node in the stack test | `tests/g1/test_sim_bringup.py` etc. (closed loop) |

Implications: (1) a shared core can inherit G1's config/limits/estop-command tests almost unchanged by parameterising the joint count;
(2) the G1 DDS tests are tied to G1 modes and 29 slots (`from g1_safety_layer...`, `ARM_SDK_WEIGHT_SLOT`, `g1_description.joints.BODY`), so they guard the shared
core only for G1 until a config fixture generalises them; (3) the conftest workaround (`IDL.serialize` lock, cyclonedds-python 0.10
serialises through one shared buffer) is robot-neutral and would be shared as-is. I did not run any of these tests (Docker only per the
repo's conventions).

---

## 6. Open decisions for the owners (found divergences that are not clearly bugs)

1. Is "go silent on `rt/lowcmd` after a 50-command burst" (H1-2, `58d0f99`) the policy for G1 split/full too, or does G1's `damp`
   mode need a continuous command? Today G1 split keeps writing (`safety_layer.py:478-488`).
2. Should the H1-2 layer latch on a dead leg controller and on a dead `rt/lowstate`? (F2b rows 1-2.)
3. Who stamps `mode_pr`/`mode_machine`, the safety layer from `rt/lowstate` (G1) or each controller (H1-2)? (F8.)
4. Is it intended that the H1-2 layer streams a zero-gain command from boot, before any controller has spoken? (F2b row 5.)
5. Should the sim safety configs refuse domain 0 inside the layer on the H1-2 side as the G1 loader does? (F2b row 8.)

---

## 7. Keep separate (looks duplicated, should stay apart)

- **`core/joint_limits.py` data and the YAML tuning lists.** 27 H1-2 joints vs 29 G1 joints, different URDF limits, different units of
  work per slot; the per-joint ratio lists (for example `default_safety_split.yaml:30-77`) are robot tuning, not code.
- **`core/arm_sdk.py`, `engage_refusal`, `check_waist_kp`, `estop_response`, the `arm_sdk_mode` branches.** The H1-2 has no `rt/arm_sdk`
  path (CLAUDE.md section 8); these are G1 control-sharing mechanics.
- **The `estop` package, `EStopSubscriber`, `h12/estop_status_raw`, `unsafe/` scripts, `utility/estop_status.py`.** Hardware of the H1-2 setup;
  the G1 loader rejects `estop:` on purpose.
- **`/safety/heartbeat` publisher.** The consumers are H1-2-only (`h12_lowerbody_rl`, `lmg_deploy`, `h12_skills`).
  If the G1 needs the same, add it to `g1_safety_layer`; do not move it.
- **The relay loops (F5).** Merging them is a rewrite of the hardware-facing path of one robot; wait for the F2 and test prerequisites.
- **G1 signal hand-back (`exit_signals.py`, `shutdown()`)** is tied to arm_sdk weight ramping; the H1-2 would need its own policy first (section 6, item 2 area).
- **G1 DDS tests and fixtures** until a config fixture makes them robot-neutral (F5).
- **Controller `rt/safety/` guards (F8)** may share one helper, but each controller keeps its own copy of the refusal at its own config and publisher boundary.
- **Docs prose** (`architecture.md`, `config.md`, README): robot-specific and drifting fast.

---

## 8. Incidental observations (not dedup, noticed while reading; unverified beyond the code)

- `h12_safety_layer/core/estop_subscriber.py` never checks the age of the last estop sample (`stamp_ms`/`sequence_id` are carried but unused
  for freshness; `safety_layer.py:345-372` acts only on `triggered` and `plugged_in`). If `estop_node` dies, the last good sample stays
  "not triggered, plugged in" and nothing latches. The node's own `fail_safe_triggered_on_disconnect` covers a serial drop, not a process
  death. Likewise a missing sample (type-name drift, F9) means "no estop". Worth an owner's look on the real robot.
- The H1-2 `EStopSubscriber` creates `DomainParticipant()` with no domain id (`estop_subscriber.py:53`), while the relay's readers use the
  config/`ROS_DOMAIN_ID` domain (`safety_layer.py:91-92`); I did not verify that both resolve to the same domain in the real bringup.
- `docs/superpowers/specs/2026-10-26-g1-port/5_golem_camera_ready_delta.md` finding 1 (the H1-2 layer pin regression) was fixed by the
  merge `a178f0b`; I verified that the current submodule contains `e0396fc`, `58d0f99`, `5182fbe` and `0eab20b` and is on `origin/camera_ready`.

## 9. What I did not verify

- No tests, containers or builds were run; every behavioural claim is from reading the code. Bench numbers in F2 are quoted from the H1-2 commit
  message `e0396fc`, not reproduced.
- Whether `LatestReader.take()` samples carry `sample_info` (needed to port the single-writer check to the H1-2 loop): cyclonedds is not installed on the host.
- The real H1-2 `rt/lowstate.mode_machine`.
- Whether any shipped controller YAML omits `limits` (relevant to the default mismatch in F3).
- Content of the sibling scopes (controllers, lower-body, bringup, sims) beyond the lines cited for the `rt/safety/*` conventions.

---

# B. Upper-body control and the hands: G1 vs H1-2 redundancy audit

## Summary

Candidates ranked by (lines saved × confidence) ÷ risk:

1. **Shared upper-body IK core (class B, about 5,000 lines net).** `g1_control` is a fork of `h12_ros2_controller` at `87bca34` (2026-07-15). Of the 6,514 lines in its `core/`, `ros2/` and `utility/`, 5,420 match H1-2 text (difflib). What is really G1-specific is small: the joint tables, the URDF paths, the waist hold (the H1-2 holds its torso instead), the follow gate and config pairing. The proposal is one robot-parameterised core, effort L. Risk is high: it touches the H1-2 hardware path and the `h12_ros2_controller` submodule owner. Lines are saved only once both robots use the core. Do item 2 first.
2. **Do first; saves no lines, fixes bugs: port six H1-2 fixes made after the fork into `g1_control` (M, G1 only).** The G1 still has bugs the H1-2 has fixed, and they matter before its first real-robot session:
   - `rt/lowstate` is read through a listener callback. On the H1-2 this was measured lagging about 4 s.
   - A duplicate estop thread on the controller side.
   - The IK tries `home` before the measured pose.
   - The reduced model's frozen joints are locked at 0.
   - Goals start from the measured pose, not from the last command.

   Fixes also go the other way. Port four G1 fixes back into the H1-2 (S-M, submodule owner):
   - Wait for a fresh state before enabling motors.
   - Bound the start-up moves by a deadline.
   - Carry the held joint's hold angle in every command. The H1-2 torso command re-seeds from the measured torso.
   - Load the frozen joints generically.
3. **Delete the H1-2 Inspire-hand path (about 320 lines, S, very low risk).** It covers `hand_controller_node`, `HandController`, `Hand{Publisher,Subscriber}` and `hand_cmd_gui`. Every command has raised `ValueError` since `NUM_HAND_DOF = 0` (2026-05-09), and no launch file starts it.
4. **`/named_config` handler duplicated within each package (class A, about 190 lines per package, S).** `frame_task_server` and `dual_arm_server` carry the same `/named_config` code: 0.99 similar in the G1, and already drifting apart in the H1-2.
5. **Dead code: `impedance_controller.py` in both packages (98 lines), and the package-local launch files that copy the bringups (S).**
6. **Hands: keep `g1_dex3` and the Magpie `gripper_node` separate (class D).** They share no helper code, at most 0.09 similarity. The only optional change is to align the per-side `Trigger` service names (`/<side>/hand/*` vs `/<side>/gripper/*`) so skills can be reused.

Abbreviations used below:
`H` = `/home/max/GOLEM/core_ws/src/h12_ros2_controller/h12_ros2_controller`,
`G` = `/home/max/GOLEM/core_ws/src/g1/g1_control/g1_control`,
`base` = `h12_ros2_controller@87bca34`, the fork point (per the earlier audit, re-verified here: the 3-way diffs below show G1 = base + G1 edits).

## 0. Method

- Similarity figures are `difflib.SequenceMatcher` ratios over whole lines. Package names are normalised first (`h12_ros2_controller`/`g1_control` → `PKG`).
- "Matching lines" is the sum of the matcher's matching blocks.
- I ran 3-way stats for every shared file: `base→H1-2 HEAD` and `base→G1`. They show which side changed what since the fork.
- Interfaces were taken from `create_publisher`/`create_service`/`ActionServer`/`ChannelPublisher`/`ChannelSubscriber`/`LatestReader` calls and `setup.py` entry points.
- Bringup usage comes from `h12_bringup/launch/h12_control.launch.py` and `h12_state.launch.py`, and from `g1/g1_bringup/launch/g1_core.launch.py`.
- Read-only. The scripts are in the scratchpad (`dedupB/`). Nothing in the repo was modified.

## 1. Inventory

### 1.1 Submodule status

| Package | Path | Submodule? |
|---|---|---|
| `h12_ros2_controller` | `core_ws/src/h12_ros2_controller` | **yes**, `b53a6b2`. Maintainer tonyzyt2000, 499 commits, active counter-balance research. |
| `g1_control` | `core_ws/src/g1/g1_control` | no (in-tree, MIT) |
| `g1_dex3` | `core_ws/src/g1/g1_dex3` | no (in-tree, MIT) |
| `magpie_control` | `core_ws/src/magpie_control` | **yes**, `f215020` |
| `custom_ros_messages` (interfaces used by all) | `core_ws/src/custom_ros_messages` | **yes**, `3f361bf` |

### 1.2 What the roughly 30k Python lines of `h12_ros2_controller` are

The package holds 36,650 Python lines in total. I classified every file: entry points, a static import closure of the nodes, the bringup launch files, and test imports.

| Category | Files | Lines | Notes |
|---|---:|---:|---|
| **Live in bringup**: closure of `frame_task_server` + `joint_state_publisher` | 20 | **6,654** | About 700 of these lines are `robot_dynamics.py:369-1069` (Crocoddyl `MomentumDDP`/`BalanceDDP`), imported but never called by the nodes. The truly live code is about **5,950** lines. |
| Live entry point, not in any bringup: `dual_arm_server` + `arm_controller` | 2 | 719 | |
| Dead: Inspire hand node, controller and GUI | 3 | 258 | Plus 59 lines in `channel_interface.py:305-363` (see F17) |
| Unregistered node: `gravity_comp_server` + `gravity_comp_controller` | 2 | 237 | No `setup.py` entry point |
| **Research library** | 36 | **11,958** | `core/controller/counter_balance/` (9,449), `zmp*` (2,216), `momentum_controller`, `support_region`, `impedance_controller` |
| **Research tests** | 26 | 6,191 | Counter-balance, ZMP, DDP |
| Tests of live code | 3 | 446 | `test_controller_config` (3 tests), `test_reduced_joint_planner` (15), `test_state_subscriber` (3) |
| Lint tests | 3 | 73 | |
| **Scripts and tools** | 43 | 6,224 | `example/` (3,674), `plot/` (1,463), `benchmark/` (405), action clients (430), `utility/` tools |
| **Archive** | 8 | 3,542 | `archive/ik_tests` |
| Launch files + `setup.py` | 6 | 348 | No package-local launch file is used by `h12_bringup` |

So about 18% of the package is code the H1-2 actually runs. About 49% is research library and research tests, and 27% is scripts, tools and archive. No line is generated or vendored, and no code is taken from upstream libraries.

### 1.3 Nodes and their interfaces

The topic and action names below are relative names in the root namespace. "Listener" means a `ChannelSubscriber` callback on the CycloneDDS receive thread. "LatestReader" means a polled `KEEP_LAST(1)` reader.

| Node (package) | Actions served | Topics published | Raw Unitree DDS | Config / env read | In bringup |
|---|---|---|---|---|---|
| `frame_task_server` (H1-2) | `/frame_task` (`custom_ros_messages/action/FrameTask`), `/named_config` (`NamedConfig`) | `frame_names` (`custom_ros_messages/msg/StringArray`), `frame_targets`, `frame_poses` (`geometry_msgs/PoseArray`), `{left,right}_ee_{pose,target}` (`geometry_msgs/PoseStamped`); 100 Hz timer | sub `rt/lowstate` (`unitree_hg LowState_`, LatestReader); pub `topics.low_cmd`, default `rt/safety/lowcmd_upper_in` (`LowCmd_`, 500 Hz) | `config/<--config>.yaml` (bringup: `safety_split.yaml`, or `sim_safety_split.yaml` in sim); URDF/sphere URDF/SRDF from `h12_ros2_model`; `ROS_DOMAIN_ID`. Spawns an OMPL planner subprocess (`PlannerClient`). | yes (`h12_control.launch.py:74-88`) |
| `frame_task_server` (G1) | same two actions. Goals with `relative`/`slow_mode` set are aborted (`G/utility/goal_checks.py`). | same 7 topics | sub `rt/lowstate` (**listener**); pub `rt/safety/lowcmd_upper_in` (500 Hz); sub (read only) `topics.safety_out` = `rt/arm_sdk` or `rt/lowcmd` (`LowCmd_` listener) until the follow gate opens | `config/{g1,g1_split,sim,sim_split}.yaml`; URDF `g1_description` `with_hand`; `ROS_DOMAIN_ID`, `G1_DDS_INTERFACE` | yes (`g1_core.launch.py:192-204`) |
| `dual_arm_server` (both) | `/dual_arm` (`DualArm`), `/named_config` | `{left,right}_ee_{pose,target}` | as their `frame_task_server` | as above | no (both). Serves the same `/named_config` name, so it cannot run beside `frame_task_server`. |
| `joint_state_publisher` (H1-2) | – | `joint_states` (`sensor_msgs/JointState`, 27 body joints, 100 Hz) | sub `rt/lowstate` (LatestReader) | default `debug.yaml` (DDS domain only); `h1_2_handless.urdf` | yes (`h12_state.launch.py:35-42`) |
| `joint_state_publisher` (G1) | – | `joint_states` (29 body joints, 100 Hz) | sub `rt/lowstate` (**listener**) | default `g1.yaml`; `G1_DDS_INTERFACE` | yes (`g1_core.launch.py:177-183`) |
| `frame_task_client`, `dual_arm_client` (both) | call `/frame_task`, `/named_config`, `/dual_arm` | – | – | – | CLI tools |
| `hand_controller_node` (H1-2) | – | `{left,right}_hand_state` (`std_msgs/Float64MultiArray`); sub `{left,right}_hand_cmd` (`Float64MultiArray`, 6 values in [0,1]) | pub `rt/inspire/cmd` (`unitree_go MotorCmds_`, 200 Hz `RecurrentThread`); sub `rt/inspire/state` (`MotorStates_`, LatestReader) | default `debug.yaml` | no. Commented out in `launch/robot_launch.py:60-61` and `robot_safety_launch.py:111-112`. **Broken**, see F17. |
| `hand_cmd_gui` (H1-2) | – | pub `{left,right}_hand_cmd` | – | – | no |
| `gravity_comp_server` (H1-2, unregistered) | – | `/h12/gravity_comp_active` (`std_msgs/Bool`); service `/h12/set_gravity_comp` (`std_srvs/SetBool`) | as controller | – | no |
| `dex3_hands` (`g1_dex3`) | services `/<side>/hand/set_pose` (`custom_ros_messages/srv/SetHandPose`), `/<side>/hand/{open,close,release}` (`std_srvs/Trigger`) | `/<side>/hand/pressure` (`std_msgs/Float32MultiArray`), `/joint_states` (14 finger joints, 50 Hz) | pub `rt/dex3/<side>/cmd` (`unitree_hg HandCmd_`, 100 Hz, only once commanding); sub `rt/dex3/<side>/state` (`HandState_`, listener) | ROS params `sides, kp, kd, rate_hz, pose_duration, joint_state_hz, network_interface, domain_id, on_shutdown, max_speed`; `ROS_DOMAIN_ID`, `G1_DDS_INTERFACE`; no YAML | yes (`g1_core.launch.py:205-212`, `use_hands`) |
| `gripper_node` (`magpie_control`, ns `left`/`right`) | action `gripper/deligrasp` (`magpie_msgs/action/DeliGrasp`); services `gripper/{open,close,calibrate,reset_parameters,reset_overload}` (`Trigger`), `gripper/set_position` (`magpie_msgs/srv/SetGripperPosition`), `gripper/set_force` (`SetGripperForce`) | `gripper/state` (`magpie_msgs/msg/GripperState`, 10 Hz) | none (Dynamixel serial) | params `auto_detect_port, port, use_eflesh, default_speed, default_torque, finger*theta_*` | real only (`h12_grippers.launch.py:18-50`) |

The H1-2 side has three Python node files that never reach a bringup: `dual_arm_server`, `hand_controller_node` and `gravity_comp_server`. `g1_control` dropped the last two at the fork.

### 1.4 Size of `g1_control` and `g1_dex3`

- **`g1_control`** has 9,892 Python lines:
  - `core/` 4,193, `ros2/` 1,589 and `utility/` 732;
  - `example/` 1,377;
  - `test/` 1,685;
  - launch files 273;
  - non-Python packaging files: `uv.lock` (4,031 lines) and `requirements.txt` (505).
  - The live closure of its three node entry points is 5,836 lines.
- **`g1_dex3`**: 511 lines of library and node code, and 555 lines of tests in 3 files (26 tests).

## 2. Overlap findings

### F1. Robot-model wrapper: Pinocchio models, reduced model, IMU→pelvis

- **Locations:** `H/core/robot_model.py:1-506` and `G/core/robot_model.py:1-394`.
- **Classification:** B.
- **Evidence:**
  - Similarity H1-2 HEAD vs G1 is 0.72 (322 matching lines). base→G1 is 0.87 (−45/+59); base→H1-2 is 0.83 (−11/+137).
  - 17 methods are byte-identical (101 lines), 7 are ≥0.8 similar, and 8 diverged.
- **What differs:**
  - **G1 generalisations, robot-agnostic and better than the base:**
    - `_keep_body_joints` (`G:99-113`) freezes every joint that is not in `BODY_JOINTS` and asserts the slot order. The H1-2 instead hand-lists `ALL_JOINTS`/`ALL_HANDLESS_JOINTS` (`H/utility/joint_definition.py:18-100`, about 80 lines of gripper joint names) and has a `handless` flag (`H:123-161`).
    - `imu_frame` parameter (`G:16-32`, `G:201-209`). The H1-2 hard-codes `torso_link` (`H:235`).
    - The collision model is optional: `check_collision_free*` returns True without one (`G:368-394`). The H1-2 asserts it exists (`H:486-506`).
  - **H1-2 changes since the fork:**
    - `rebuild_reduced_model` re-locks the frozen joints at the measured state (`H:76-92`, commit `0159d45`, which fixed a 14-21 mm end-effector error).
    - `state_staleness_ms` (`H:304`).
    - `reduced_limit_violations` (for the planner).
    - Thin `get_com/get_zmp/...` wrappers (research).
  - **Accidental divergence (G1 lacks the fix):**
    - `G:58-76` locks the frozen joints, the waist included, at `zero_q_body`, and never re-locks them.
    - The G1 commands its waist at `waist_target` (default 0), so the FK error is the waist's tracking error (sag).
    - This is the same bug class as `0159d45`, but smaller in magnitude.
- **Consolidation:** shared `RobotModel`. Construct it with:
  - `body_joints` (slot order), `imu_frame`, and an optional collision URDF/SRDF;
  - the G1 frozen-joint loader;
  - the H1-2 rebuild and staleness code.

  The research wrappers (`get_zmp`…) stay as H1-2 subclasses, or call `self.dynamics` directly. Effort M.
- **Risk:**
  - This is on the H1-2 real-robot path. The FK must stay bit-identical for the H1-2 URDF.
  - The guard is a new test: build both robots' models and compare FK at random q against the current H1-2 code.
  - Submodule: yes.

### F2. IK solver (Pink)

- **Locations:** `H/core/ik_solver.py:1-551` and `G/core/ik_solver.py:1-536`.
- **Classification:** B (near A).
- **Evidence:**
  - Similarity 0.92 (498 matching lines). base→G1 is 0.97 (−10/+17).
  - 29 functions are byte-identical (351 lines). Only 5 diverge: `__init__`, `update_configurations`, `_ik_reduced`, `_default_seed_items` and `ik_step_reduced`.
- **What differs:**
  - **G1, robot-agnostic generalisation:** the self-collision barrier is created only when a collision model exists (`G:65-79`, `G:251`).
  - **H1-2:**
    - `rebuild_reduced()` (`H:94-120`).
    - `update_configurations(q)`, which takes a seed (`H:205-211`).
    - A `constraints=` passthrough, which has no caller (research, `cd1429d`).
  - **Accidental divergence (G1 lacks the fix):**
    - Seed order: the H1-2 tries `('current', …)` first (`H:283-292`, `9eec384`: it "returns the solution nearest the arm").
    - The G1 tries `home` first and `current` last (`G:275-281`). Planned goals can swing the arm through `home`-like solutions.
- **Consolidation:** one `IKSolver` with an optional collision barrier, taking `NAMED_CONFIGS` for the seeds as a constructor argument instead of a module-level import (`G:10`, `H:10`). Effort S once F1 is done.
- **Risk:** low. The IK math is shared verbatim. The H1-2 behaviour is unchanged if the collision model is passed.

### F3. Dynamics: gravity compensation, frame twist, CoM

- **Locations:** `H/core/robot_dynamics.py:13-365` (class `RobotDynamics`) and `G/core/robot_dynamics.py:12-324`.
- **Classification:** A. The G1 copy is a strict subset of the H1-2 class.
- **Evidence:**
  - base→G1 has 0 changed lines. Class vs class similarity is 0.94, and all 313 G1 lines match.
  - The H1-2 file adds 704 lines of Crocoddyl `MomentumDDP`/`BalanceDDP` research (`H:369-1069`), plus wrench estimation, inside `RobotDynamics`.
- **What differs:** only what the H1-2 added. The live nodes call `get_gravity_compensation`, `compute_frame_twist` and `get_com` (`H/core/controller/upper_controller.py:652,672,694-695,728`; `H/core/low_cmd_handler.py:183`).
- **Consolidation:** the shared core keeps the `RobotDynamics` base class. The H1-2 research classes (`MomentumDDP`, `BalanceDDP`, `create_*_ddp`) move to an H1-2 module that subclasses it. Effort S.
- **Risk:** the research code imports these names from `h12_ros2_controller.core.robot_dynamics` (tests under `test/test_*ddp*`), so keep re-exports. No real-robot behaviour change.

### F4. `UpperController`, `ArmController`, `FrameController`: goal handling, interpolation, hold

- **Locations:**
  - `H/core/controller/upper_controller.py:1-768` and `G/core/controller/upper_controller.py:1-788`;
  - `arm_controller.py` (H 222, G 220, similarity 0.97);
  - `frame_controller.py` (H 94, G 92, 0.94).
- **Classification:** B.
- **Evidence:**
  - `upper_controller` similarity is 0.65 (502 matching lines). base→H1-2 is −63/+202; base→G1 is −67/+226.
  - 30 methods are byte-identical (145 lines: the EE pose properties, the path helpers, the sim steps).
  - 14 diverged and 15 are one-sided.
  - `arm_controller`/`frame_controller` differ from the base only in their constructor signature (collision paths renamed and optional, `handless` dropped).
- **Trajectory interpolation (shared design):**
  - **Direct mode:** a Pink IK velocity each `ctrl_hz` tick, capped in `_limit_joint_vel` by the end-effector linear and angular speed (`v_lim`, `w_lim`) and by `dq_lim`, then integrated. The H1-2 adds `speed_scale` for `slow_mode`.
  - **Planned mode:** an OMPL path that `execute_path` plays back one waypoint per `dt`.
  - **Start-up:** staged named postures, then the torso or waist is eased at 0.3 rad/s. `G:_move_waist_to_target` (`G:282-301`) is a 3-joint, deadline-bounded rewrite of `H:_move_torso_to_target` (`H:205-222`).
- **Robot-specific differences (legitimate):**
  - One torso joint (`torso_target`) vs three waist joints (`waist_target`).
  - The G1 follow-gate start-up (`_wait_until_following`, `_sync_to_command`, `G:184-217`, `741-754`), which exists because `rt/arm_sdk` shares the arms with Unitree's controller.
  - `PLANNER_UNAVAILABLE`, because the G1 has no collision model.
  - The EE frames: `*_wrist_yaw_link` vs `*_hand_palm_link`.
  - `damp_mode` keeps the waist holding (`G:769-776`).
- **Accidental divergence, where the H1-2 has the fix and the G1 lacks it:**

  | Fix in H1-2 | H1-2 | G1 now | Impact on G1 |
  |---|---|---|---|
  | `772f946`: a goal starts from the last command, not the measured pose (`COMMAND_SEED_MAX_GAP`); the torque bias carries over | `H:302-337`, `725-742` | `G:370-372` seeds from measured. `_sync_to_command` only applies before engagement. | The arm drops at every goal boundary (the H1-2 symptom). The tau-bias part does not apply while `ki: 0.0` (all G1 configs). |
  | `caf318e`: `check_state_fresh()` at goal start and in every step; `StaleStateError` → abort | `H:291-300` and callers | none | Stale state goes unseen (see F5) |
  | `3d87152`: a hold pinned at a q-limit is not reported as converged | `H:614-686` | `G:629-694` returns True | Does not apply while `ki: 0.0` |
  | `f3a9a55`: `execute_path` paces at `dt − work` (and plans slow moves at a lower speed) | `H:447-478`, `356-375` | `G:486` `time.sleep(self.dt)` | Does not apply while the planner is disabled |
  | `0159d45`: the reduced model is re-locked per goal | `H:302-316` | – | F1 |
- **Accidental divergence, where the G1 has the fix and the H1-2 lacks it (flag to the H1-2 owner):**

  | G1 fix | G1 | H1-2 now | Possible H1-2 bug |
  |---|---|---|---|
  | Wait for a fresh `rt/lowstate` with finite q before enabling motors; on failure, raise and stop publishing | `G:148-164`, `69-73` | `H:76-77` `time.sleep(0.5)`, then seeds `enable_motors` from whatever state exists. `check_state_fresh` is not called in `__init__`. | A late state stream makes the first targets `q=0`. Whether `h12_safety_layer` refuses them is outside my scope and was not checked. |
  | Start-up moves bounded by a deadline that raise on timeout, with the controller shut down | `G:166-235` | `H:128-147` runs a fixed `steps=50`; no convergence check, no error | A silently incomplete start-up |
  | Held joints carry their hold angle in every command | `G:730-739` (`q[WAIST_INDEX] = self._waist_hold`) | `H:318-337`: `_command_seed` takes the **measured** value for every non-enabled joint (torso included), and `_apply_joint_position(ik_solver.q)` commands it | **Probable:** the torso command is re-seeded to the measured, sagged torso at every goal, so it can ratchet. Verify on the H1-2. |
  | Index computed from the layout | `G:724` uses `LEFT_ARM_INDEX[0]` | `H:719` hard-codes `vel_scaled[:13]` | H1-2-only constant; harmless |
- **Consolidation:** one `UpperController` in the shared core, parameterised by a robot profile:
  - held joints + targets, enabled joints, EE frames, startup sequence, named configs, collision paths;
  - optional hooks: `pre_engage(controller)` for the G1 follow gate, and the planner on or off.

  Port the fixes in both directions first. `ArmController`/`FrameController` become shared unchanged. Effort L, the biggest piece.
- **Risk:**
  - High: the H1-2 `frame_task_server` drives the arms on the real robot.
  - Guards: `tests/stack/run.sh robocasa|isaac arms` (manual, needs a GPU host; checks `/frame_task` motion) and `tests/g1/test_sim_bringup.sh`. The DDS closed-loop tests in `g1_control/test/test_controller_loop_dds.py` (8 tests) would cover the shared code.
  - The H1-2 has **no unit test** of `UpperController` or the servers.

### F5. DDS I/O: `StateSubscriber`, `LowCmdPublisher`, `LowCmdHandler`

- **Locations:**
  - `H/core/channel_interface.py:32-303` and `G/core/channel_interface.py:24-269`;
  - `H/core/low_cmd_handler.py:1-241` and `G/core/low_cmd_handler.py:1-357`;
  - `H/core/latest_reader.py:1-92`, which exists only in the H1-2.
- **Classification:** B.
- **Evidence:**
  - `channel_interface` similarity 0.61 (193 matching lines); `low_cmd_handler` 0.79 (237).
  - 13 `LowCmdHandler` methods are byte-identical (134 lines).
  - The `CommandPublisher`/`LowCmdPublisher` bodies match line for line except the estop (`G:159-176` vs base).
- **What differs:**
  - **Robot-specific:**
    - the G1's `SafetyOutputSubscriber` (`G:81-100`) and `arm_follow_gate` (`G/core/low_cmd_handler.py:170-220`);
    - `mode_machine` 0 vs 6 in the default `LowCmd_`;
    - estop semantics: the G1 stops publishing so `g1_safety_layer` hands back; the H1-2 sends zero gains.
  - **Both sides independently enforce the same rule.** `topics.low_cmd` must start with `rt/safety/`:
    - `H/utility/controller_config.py:54-59` (`_validate_low_cmd_topic`);
    - `G/utility/controller_config.py:212-217` (inline).

    Same job, two implementations (C inside B).
  - **Accidental divergence, flagged as a bug for the G1:**
    1. **`rt/lowstate` is read through a listener callback** (`G:24-78`). The H1-2 replaced this in `caf318e` with `LatestReader` plus `TickLagTracker` (`H/core/latest_reader.py`, `H:32-136`). The reason was measured on the robot: a GIL-busy controller process lagged "~4 s". The G1 `frame_task_server` process has the same structure: a 500 Hz publisher thread, a 1 kHz checker thread, IK, and 100 Hz FK publishers. The G1's own freshness check (`G/core/controller/upper_controller.py:148-164`) compares against the time the callback ran, so it cannot see queued, late samples.
    2. **Controller-side estop thread.** `G/core/low_cmd_handler.py:59-66`, `230-232` and `266-294` still check q/dq/tau against `limits.estop`. The H1-2 removed this in `deacb9a` because "a trip here could drop the arms while the safety layer still considered the state safe". On the G1 a trip stops publishing. `g1_safety_layer` then estops on a stale upper channel after 2 s (`core_ws/src/g1/g1_safety_layer/g1_safety_layer/core/safety_layer.py:47,587`). In split mode that means "every motor off".

    Both are judgement calls for the G1 owner, but the H1-2 evidence argues for porting both before `docs/g1/first-session.md`.
- **Consolidation:**
  - Shared `StateSubscriber` (H1-2 version), `latest_reader.py`, `CommandPublisher`/`LowCmdPublisher` (`num_motor` and the default `mode_machine` as parameters), and `LowCmdHandler` without the estop thread.
  - The G1 keeps `SafetyOutputSubscriber` and `FollowGate` as a plug-in on the handler (one `gate` hook in `set_joint_commands`).
  - Effort M.
- **Risk:**
  - The H1-2 code is unchanged by this move: it is already the target version.
  - For the G1 it changes behaviour (estop and state reading); guarded by `test_controller_loop_dds.py` and the sim tests.
  - `H/core/channel_interface.py:305-363` (the Inspire hand) should be deleted rather than shared (F17).

### F6. Planner: OMPL reduced-joint planner and its subprocess

- **Locations:**
  - `H/core/planner/reduced_joint_planner.py:1-708` and `G/core/planner/reduced_joint_planner.py:1-573`;
  - `planner_process.py` (H 228, G 183);
  - `planner/__init__.py` (identical, 14 lines).
- **Classification:** A for the G1, which equals the base except for one line: `frame_names` defaults to `('left_hand_palm_link','right_hand_palm_link')`.
- **Evidence:**
  - base→G1 for `reduced_joint_planner` is −1/+1. H1-2 HEAD vs G1 is 0.88 (565 matching lines). `planner_process` is 0.82.
  - The H1-2 added full x/y/z frame limits (`7528f9d`), limit-violation reporting, and per-request `moving_speed`/`max_interpolation_steps` overrides (`f3a9a55`).
- **What differs:** default frame names and the collision path argument names. The G1 ships the planner disabled (`planner.enabled: false`, `G/core/controller/upper_controller.py:122-130`) because it has no self-collision model.
- **Consolidation:** shared, H1-2 HEAD version, with the frame names coming from the config (they already can: `planner.frame_names` in `g1.yaml`). Effort S.
- **Risk:**
  - None for the G1 today (disabled). For the H1-2: no change if the shared copy is its HEAD.
  - Guards: `test_reduced_joint_planner.py` (H1-2 has 15 tests, G1 has 10, and the G1 file is the base file minus the H1-2-asset smoke test).

### F7. Action servers: `/frame_task`, `/named_config`, `/dual_arm`

- **Locations:**
  - `H/ros2/frame_task_server.py:1-644` and `G/ros2/frame_task_server.py:1-534`;
  - `H/ros2/dual_arm_server.py:1-497` and `G/ros2/dual_arm_server.py:1-485`.
- **Classification:** B, close to A. For the G1, base→G1 is 0.97 for `frame_task_server` and 0.98 for `dual_arm_server`.
- **Evidence:**
  - H1-2 HEAD vs G1: `frame_task_server` 0.81 (476 matching lines), `dual_arm_server` 0.94 (460).
  - The G1 edits are the config/URDF names plus an abort for unsupported fields (`G/ros2/frame_task_server.py:154-161`, `345-352`; `G/ros2/dual_arm_server.py:299-307`, using `G/utility/goal_checks.py:1-15`).
- **Goal, feedback and result handling** (the same design on both sides, inherited from the base):
  - Each goal takes a `threading.Lock`, clears the tasks, sets the targets, calls `update_ik_solver()`, then runs the direct or planned branch.
  - The direct loop is IK until `threshold_ik`, then the steady-state hold until the linear/angular (or joint) thresholds, bounded by `duration` or the config `timeout`.
  - Feedback: `FrameTask.Feedback.errors_linear/errors_angular` per frame, `NamedConfig.Feedback.joint_error`, and `DualArm` per-arm errors.
  - Result: `success` only. A timeout still calls `goal_handle.succeed()` with `success=False` (`H:305`, `388`, `545`, `614`). A planning or execution exception calls `abort()`.
- **H1-2 additions since the fork:**
  - `slow_mode` (`SLOW_MODE_SCALE = 0.25`, `H/ros2/frame_task_server.py:32`, `208-220`).
  - `relative` targets (`H/ros2/frame_task_server.py:180-186`, `224-238`).
  - Stale-state abort (`H:161-172`).
  - Hold-blocked reporting (`H:393-408`).
  - Best-effort path visualisation.
- **Accidental divergence, a feature gap rather than a bug:**
  - The G1 rejects `relative` and `slow_mode`. This is a stopgap introduced because the G1 forked before those message fields existed.
  - Inside the H1-2, `dual_arm_server` got `slow_mode` but not the stale-state catch. `StaleStateError` from `check_state_fresh()` therefore propagates out of the `dual_arm` callbacks unhandled. That is an H1-2-internal issue, minor, and not launched.
- **Consolidation:**
  - Shared server classes, H1-2 HEAD version: the G1 then gains `slow_mode`/`relative` and `goal_checks.py` disappears.
  - The thin `main()` and console scripts stay in each robot package, so `ros2 run h12_ros2_controller frame_task_server` / `g1_control frame_task_server` and both bringups are unchanged.
  - Effort M.
- **Risk:**
  - Real-robot impact on the H1-2 (the server is the arm path).
  - Guards:
    - H1-2: `tests/stack` (`check_frame_task.py`).
    - G1: `test_server_goal_rejection.py`, which must be rewritten once the fields are supported, and `tests/g1/test_sim_bringup.sh`.

### F8. Within each package: `/named_config` handler duplicated between two servers

- **Locations:**
  - **G1:** `G/ros2/frame_task_server.py:334-511` vs `G/ros2/dual_arm_server.py:288-465`. Per-method similarity is 0.99-1.00 for `named_config_callback`, `_direct_named_config_callback`, `_plan_named_config_callback`, `_named_config_joint_error`, `_publish_named_config_feedback` and `cancel_callback`: 197 lines in methods ≥0.8 similar.
  - **H1-2:** `H/ros2/frame_task_server.py:426-621` vs `H/ros2/dual_arm_server.py:298-477`. `_direct_named_config_callback` is 0.92 and `_plan_named_config_callback` 0.88. `named_config_callback` is 0.04 because `frame_task_server` split it into a stale-abort wrapper. **The two copies have already drifted inside the H1-2.**
- **Classification:** A within a package.
- **Consolidation:** a `NamedConfigActionMixin` (or helper class) used by both servers. It goes into the shared core, or into each package if F7 is not done. Effort S.
- **Risk:** low. `dual_arm_server` is not in any bringup on either robot.

### F9. `joint_state_publisher`

- **Locations:** `H/ros2/joint_state_publisher.py:1-53` and `G/ros2/joint_state_publisher.py:1-54`.
- **Classification:** B (0.95; the diff is 3 lines: the URDF constant and the `handless` flag).
- **What differs:**
  - H1-2: 27 joints, `h1_2_handless.urdf`, LatestReader (through the shared `StateSubscriber`).
  - G1: 29 joints, `with_hand` URDF, listener.
  - Both build a full Pinocchio model and run `update_kinematics()` at 100 Hz just to republish `q`, which is unnecessary work in both.
- **Consolidation:** a shared node taking `body_joints` and a URDF path from the profile, or one that drops Pinocchio altogether: it only needs `rt/lowstate` q and the joint names. Effort S.
- **Risk:** low (read only). `robot_state_publisher` and RViz depend on the names.

### F10. YAML config loader and config schema

- **Locations:** `H/utility/controller_config.py:1-408` and `G/utility/controller_config.py:1-368`.
- **Classification:** B (0.70; 12 functions byte-identical, 138 lines).
- **Schema:** the H1-2 `config/safety_split.yaml` and G1 `config/g1.yaml` share 36 key paths:
  - `topics.*`, `network.*`;
  - `controller.{v_lim,w_lim,dq_lim,d_min,timeout,threshold_*}`;
  - `planner.*`, `frequency.*`;
  - `gains.{kp,kd,ki}`;
  - `limits.clip.*`, `logging.*`.
- **What differs:**

  | | Keys | Robot-specific? |
  |---|---|---|
  | H1-2 only | `mode`, `controller.torso_target`, and the extra `planner.*` (frame limits, speeds) | `torso_target` is robot-specific. The planner keys are not robot-specific. |
  | H1-2 research only | nine validators (`_validate_zmp`, `_validate_counter_ddp`, `iteration5*`…) | research |
  | G1 only | `controller.{waist_target, state_timeout, startup_timeout, startup_move_timeout, follow_tolerance, imu_frame}`, `topics.safety_out`, `planner.enabled`, `limits.estop`, `frequency.check_hz` | `waist_target`, `follow_tolerance` and `safety_out` are G1-specific. The `startup*`/`state_timeout`/`imu_frame` keys are robot-agnostic. `limits.estop`/`check_hz` are the divergence in F5. |

  The G1 also rejects a `mode` key (`G:178-181`). `_channel_factory_settings` in the G1 also reads `G1_DDS_INTERFACE` (`G:295-308`); the H1-2 version (`H:352-361`) does not.
- **Consolidation:**
  - A shared loader with the common schema and a hook for robot extras (`extra_validators`).
  - The DDS-interface variable renamed to something robot-agnostic. The earlier audit also suggested `GOLEM_DDS_INTERFACE`.
  - The H1-2 research keys are passed through as raw dicts, as they are today.
  - Effort M.
- **Risk:**
  - Medium: every H1-2 config must still load identically.
  - Guards: `test_controller_config.py` (H1-2 3 tests, G1 2), plus `test_g1_controller.py`'s config tests (about 9).

### F11. Named-configuration tables, joint definitions, joint limits

- **Locations:**
  - `named_config.py`: `H/utility/named_config.py:1-73` vs `G/utility/named_config.py:1-34` (0.19);
  - `joint_definition.py`: `H/utility/joint_definition.py:1-189` vs `G/utility/joint_definition.py:1-55` (0.16);
  - `joint_limits.py`: `H/utility/joint_limits.py:1-163` vs `G/utility/joint_limits.py:1-40` (0.26).
- **Classification:** D (robot data). The mechanism is shared: a dict from name to a 14-vector over `ENABLED_JOINTS`, `STARTUP_SEQUENCE` in the G1 vs a hard-coded `('init_1','init_2','init_3','home')` in `H/core/controller/upper_controller.py:128-135`. The lookup code in the servers is identical.
- **Note:**
  - The G1 derives joints, limits, effort and velocity from `g1_description` (single source).
  - The H1-2 hard-codes its lists, and `joint_limits.py` holds hand-typed limits.
  - The G1 pattern is worth adopting for the H1-2 (`h12_ros2_model` could expose a joint table). That is optional and outside this scope.
- **Consolidation:** stays per robot, as the "profile" data.

### F12. Frame and transform utilities, small helpers, action clients

- **Classification:** A (byte-identical after renames).
- **Locations and evidence:**
  - `H/ros2/utility.py` vs `G/ros2/utility.py` (`pose_to_matrix`, `matrix_to_pose`, `_project_to_so3`…): 89 vs 89 lines, ratio 1.00.
  - `utility/topic_subscriber.py`: 81 lines, 1.00.
  - `utility/view_urdf.py`: 27 lines, 1.00.
  - `ros2/frame_task_client.py`: 0.98.
  - `ros2/dual_arm_client.py`: 0.99.
  - `utility/path_definition*.py` are D (paths).
- **Consolidation:** move them with the shared core. For `pose_to_matrix`, a third copy exists in `h12_skills/h12_skills/perception_utils.py`, which is out of scope. Effort S.
- **Risk:** none.

### F13. Meshcat visualiser

- **Locations:** `H/core/robot_visualizer.py:1-377` and `G/core/robot_visualizer.py:1-317`.
- **Classification:** B (0.86). base→G1 changes 8 lines, all of them names: `unitree_h1_2`→`unitree_g1`, `livox_link`→`mid360_link`, `camera_link`→`d435_link`, and the grasp-frame names→palm links.
- **Consolidation:** shared, with the viewer name, sensor frames and EE frames coming from the profile. Effort S.
- **Risk:** none on the robot: visualisation only, off in `frame_task_server`.

### F14. Example scripts and package-local launch files

- **Examples:** six G1 files, 1,377 lines; 1,271 of them match the H1-2 (0.87-0.95):
  - `G/example/{arm_controller_goto, arm_controller_gui, frame_controller_goto, frame_task_trajectory, motor_viewer, motor_viewer_tui}.py`;
  - their counterparts in `H/example/`.

  The G1 edits are names plus a domain-0 refusal for the publishing tools (`G/utility/controller_config.py:338-352`, `init_channel_factory_for_commands`). That refusal is a robot-agnostic safety improvement the H1-2 examples lack: they only use the y/N prompt.
- **Launch files:**
  - `g1_control/launch/*.py` (5 files, 273 lines) vs `h12_ros2_controller/launch/*.py` (302 lines): 249 matching lines. `desktop_launch.py` is identical.
  - Neither bringup uses them. The H1-2 copies still list the dead `hand_controller_node`, commented out.
  - The G1 `robot_safety_launch.py` checks config pairing and is tested (`test_g1_controller.py::test_robot_safety_launch_checks_the_pairing`).
- **Classification:** B.
- **Consolidation:**
  - Examples become shared scripts that take a `--robot` profile, or the G1 keeps only the two motor viewers.
  - The package-local launch files are dropped in favour of `h12_bringup`/`g1_bringup`, after checking with the owners.
  - Effort S.
- **Risk:** none on the robot.

### F15. Tests

| Area | H1-2 | G1 |
|---|---|---|
| Config loader | `test/test_controller_config.py` (59 lines, 3 tests) | `test/test_controller_config.py` (79 lines, 2 tests; the base file, since base→G1 is −2), plus about 9 config tests in `test_g1_controller.py` |
| Planner | `test_reduced_joint_planner.py` (264 lines, 15 tests) | 161 lines, 10 tests (the base minus the H1-2-asset smoke test) |
| State subscriber | `test_state_subscriber.py` (123 lines, 3 tests, LatestReader) | – |
| Controller loop | – | `test_controller_loop_dds.py` (737 lines, 8 tests). Closed loop over DDS, with a fake robot and the real `g1_safety_layer`. |
| Server goal handling | – | `test_server_goal_rejection.py` (2), `test_goal_checks.py` (5) |
| Model/IK | – | `test_g1_controller.py` (26 tests: slot order, IMU frames, postures in limits, follow gate, pairing, domain-0 refusal) |
| Lint | 3 files, identical in both (A) | |
| Integration | `tests/stack/run.sh robocasa\|isaac arms` (GPU host, manual) | `tests/g1/test_sim_bringup.sh` |

- **Classification:** B for the copied tests (planner, config, lint). C for the rest: different coverage for the same code.
- **Observation:** most of the G1's controller-loop tests would test the shared core directly, and the H1-2 has nothing equivalent. Sharing the core gives the H1-2 its first closed-loop unit tests.

### F16. Copied packaging files

- **Locations:** `g1_control/{AGENTS.md (A, 30 lines), environment.yml (A), .gitignore (A), .python-version (A), requirements.txt (0.91), uv.lock (4,031 lines), pyproject.toml (0.90)}`.
- **Classification:** A/B.
- **Accidental leftovers:**
  - The G1's `pyproject.toml` still says `description = "Controller of the h12 robot with ROS2 services"`.
  - It pins `numpy>=2.2.6`, which contradicts the image clamp `numpy<2` (CLAUDE.md §8).
  - The G1 README documents `uv run` for the examples, so they are used for standalone runs, not by Docker or colcon.
- **Consolidation:** they follow the core. One set lives with the shared package. Effort S. No robot risk.

### F17. Dead code

1. **H1-2 Inspire hand path: broken, unlaunched.**
   - Code involved:
     - `H/ros2/hand_controller_node.py:1-97`;
     - `H/core/controller/hand_controller.py:1-79`;
     - `H/ros2/hand_cmd_gui.py:1-82`;
     - `H/core/channel_interface.py:305-363` (`HandSubscriber`, `HandPublisher`), plus `TOPIC_HAND*` (`:29-30`) and the `unitree_go`/`RecurrentThread` imports (`:11-13`);
     - `NUM_HAND_DOF` (`H/utility/joint_definition.py:14`);
     - two `setup.py` entry points.
   - It is broken: `NUM_HAND_DOF = 0` since `101a53a` (2026-05-09). `HandPublisher.q` is therefore `np.zeros(0)`, and `hand_controller.ctrl_right` does `q[0:6] = right_arr`, which raises `ValueError: could not broadcast input array from shape (6,) into shape (0,)` (reproduced with numpy). The state topics publish empty arrays.
   - Nothing launches it.
   - About 320 lines. It was still maintained in `caf318e`, which moved `HandSubscriber` to `LatestReader`.
   - **Proposal:** delete it (owner decision, in the submodule). The Isaac `--hand_type inspire` fixture still publishes `rt/inspire/state`. If Inspire hands come back, restore it from git with the correct DOF count.
2. **`impedance_controller.py`** in both packages (`H/core/controller/impedance_controller.py:1-50`, `G/...:1-48`, 0.88): nothing imports it. Delete it.
3. **`gravity_comp_server`/`gravity_comp_controller`** (H1-2, 237 lines): not a console script. Owner decision.

### F18. Hands: `g1_dex3` vs H1-2 Inspire vs Magpie `gripper_node`

- **Locations:**
  - `/home/max/GOLEM/core_ws/src/g1/g1_dex3/g1_dex3/{dex3.py:1-143, driver.py:1-179, hand_node.py:1-188}`;
  - the Inspire code in F17;
  - `/home/max/GOLEM/core_ws/src/magpie_control/src/magpie_control/gripper_node.py:1-341`.
- **Classification:** C against the Inspire driver: the same job (a Unitree-DDS dexterous-hand driver with a streaming command thread), implemented independently. D against Magpie.
- **Evidence:** difflib ratio 0.09 for each pair (Inspire-vs-Dex3: 33 matching lines; Magpie-vs-Dex3: 37). All of them are blank lines, `return response` and imports. No shared helper code exists. The Magpie hardware code (`gripper.py`, Dynamixel) has no counterpart.
- **Interface shapes:**

  | | Magpie (H1-2 real and RoboCasa `magpie_hand_bridge.py`) | Dex3 (G1) |
  |---|---|---|
  | Open/close | `/<side>/gripper/{open,close}` (`std_srvs/Trigger`) | `/<side>/hand/{open,close}` (`Trigger`) |
  | Position command | `/<side>/gripper/set_position` (`magpie_msgs/SetGripperPosition`: width in mm, speed 0-1 → `actual_position`) | `/<side>/hand/set_pose` (`custom_ros_messages/SetHandPose`: 7 joint angles in rad, duration) |
  | Force | `set_force` (N), action `deligrasp` | – (soft `kp`/`kd` parameters) |
  | Release/limp | – | `/<side>/hand/release` |
  | Maintenance | `calibrate`, `reset_parameters`, `reset_overload` | – |
  | State | `gripper/state` (`GripperState`, 10 Hz) | `/joint_states` (fingers) + `/<side>/hand/pressure` |
  | Transport | Dynamixel serial | `rt/dex3/<side>/{cmd,state}` DDS; bypasses the safety layer, as the Inspire did |

  `h12_skills` uses `/<side>/gripper/set_position` (`h12_skills/h12_skills/base.py:323`). Nothing uses the Dex3 services yet.
- **What differs:** hardware, units and transport. All of it is robot-specific.
- **Consolidation:** keep the drivers separate. If skills should run on both robots, define a minimal common hand interface: per-side `open`/`close` `Trigger` under one prefix. Options are `g1_dex3` also serving `/<side>/gripper/{open,close}`, or the skills taking the prefix as a parameter. That is an interface decision (CLAUDE.md §5). It saves no lines. Effort S.
- **Risk:** it adds names to the G1 surface. Watch for collisions in the one-stack-per-domain rule (both robots would expose `/<side>/gripper/open`).

### F19. DDS domain and interface resolution (cross-scope note)

- **Locations:**
  - `g1_dex3/dex3.py:118-143` (`dds_domain`): strict. No default; refuses a mismatch with the node's own ROS domain; refuses non-decimal values such as `010`.
  - `G/utility/controller_config.py:295-308` and `H/utility/controller_config.py:352-361` (`_channel_factory_settings`): `ROS_DOMAIN_ID`, else the YAML's `network.domain_id`. Both real configs default to 0, as do the H1-2 `debug.yaml` and `safety_split.yaml`.
  - `G1_DDS_INTERFACE` is read in `G/utility/controller_config.py:307` and `g1_dex3/hand_node.py:75`.
- **Classification:** C (same job, different strictness).
- **Proposal:** one helper for "resolve the Unitree DDS domain and interface", used by every DDS-speaking node of both robots. Per the earlier audit, `g1_safety_layer`, `g1_locomotion` and `h12_safety_layer` have their own versions; those are other auditors' scope. Effort M, cross-package.

## 3. Keep separate

| Item | Why |
|---|---|
| Joint tables, limits, default gains, named postures (F11), the controller YAML values, URDF/SRDF paths | Robot data. They become the per-robot profile. |
| G1 `follow_gate.py`, `SafetyOutputSubscriber`, `config_pairing.py`, waist-hold start-up semantics | Exist because `rt/arm_sdk` blends the arms with Unitree's controller and split mode waits for the walking node. The H1-2 has no equivalent bus. Keep them as a G1 plug-in on the shared core. |
| G1 `goal_checks.py` | A temporary policy; it disappears once `slow_mode`/`relative` are shared (F7). |
| H1-2 research: `counter_balance/`, `zmp*`, DDP classes in `robot_dynamics.py:369-1069`, `momentum_controller`, their 26 test files, `archive/`, `plot/`, `benchmark/` | H1-2-only research owned by the submodule maintainer. It must keep importing the core under its current names (re-exports). |
| `g1_dex3`, Magpie `gripper_node`, and the hand transports | Different hardware, units and transport (F18). No shared code to extract. |
| Estop semantics in `CommandPublisher.estop` (G1: stop publishing; H1-2: zero gains) | Tied to each safety layer's behaviour. Make it a constructor option, not shared behaviour. |

## 4. Suggested order

1. **G1-only bug-fix port (M, no ownership boundary crossed).** Do it before the G1's first real session.
   - `latest_reader.py` + `StateSubscriber` + `check_state_fresh` (F5).
   - Remove the controller estop thread (F5).
   - IK seed order (F2).
   - Re-lock the reduced model at the held or measured waist (F1).
   - Command-seeded goal start (F4).
   - Optional for now: `slow_mode`/`relative` (F7). Pacing, x/y/z limits and hold-limit only matter once the planner or `ki` is on.

   Guards: `g1_control/test/*`, `tests/g1/test_sim_bringup.sh [--walk]`.
2. **H1-2 owner: upstream the G1 generalisations and fixes (S-M, submodule).**
   - Generic frozen-joint loading, `imu_frame`, optional collision model (F1, F2).
   - Wait-for-state and deadline-bounded start-up (F4).
   - Held-joint command: verify the torso ratchet first (F4).
   - Domain-0 refusal in the examples (F14).
   - Delete the Inspire path and `impedance_controller` (F17).

   Guards: `tests/stack` arms (sim), then the real robot.
3. **Extract the shared core (L).** After steps 1-2 the two copies differ only in profile data and the G1 plug-ins, so extraction is mostly mechanical.
   - **Location:** a new in-tree robot-agnostic package (MIT, not a submodule). Making `g1_control` depend on the `h12_*` submodule would pull in `h12_ros2_model`/CL_Assets through `h12_ros2_controller`'s `package.xml`, and would put G1 changes under the H1-2 owner's review.
   - **Contents:**
     - `RobotModel`, `RobotDynamics` (base class), `IKSolver`;
     - `UpperController`/`ArmController`/`FrameController`;
     - `LowCmdHandler`, `StateSubscriber`/`LatestReader`, `LowCmdPublisher`;
     - planner; visualiser;
     - config loader; `ros2/utility.py`, `topic_subscriber`, `view_urdf`;
     - server classes + `NamedConfigActionMixin` (F8); the `joint_state_publisher` node class;
     - the clients.
   - **Robot plug-in:** a profile object passed to the constructors. It replaces the 19 module-level imports of `joint_definition`, `named_config`, `path_definition_ros` and `g1_description` counted in `G/core`, `G/ros2` and `G/utility`, with the same pattern in `H/`.
   - **Kept in each robot package:** thin `main()` console scripts, so `ros2 run h12_ros2_controller frame_task_server` and `g1_control frame_task_server` and both bringups stay unchanged.
   - **Order:** the G1 switches first, since it has never run on the real robot. The H1-2 switches when its owner is ready, with re-exports for the research code.
   - **Lines saved:** about 5,000-5,400 Python lines once both robots use it (G1 core/ros2/utility 5,420 matching, minus roughly 400 lines of profile and plug-in code). Plus about 190 from F8, and 1,000+ more if the examples are shared.

## 5. Caveats and what I did not verify

- I did not run any test or build. Every behaviour claim comes from reading the code, from commit messages (`git show` in the submodule) and from difflib.
- The H1-2 torso-ratchet finding (F4) is a code-path inference: `_command_seed` takes the measured value for non-enabled joints, and `_apply_joint_position` commands it. I did not confirm it on the robot or in sim.
- The G1 lag risk (F5-1) is inferred from the H1-2's measured lag in an identically structured process, not measured on the G1.
- I did not check how `h12_safety_layer` handles an all-zero first command from `frame_task_server` (F4, first H1-2 bug).
- For `magpie_control` I compared only `gripper_node.py`'s interfaces and the absence of shared helpers, as briefed. I did not audit the sensor nodes or `gripper.py`'s internals.
- Statements about the research code (what it imports from the core) come from test imports and a grep, not from a full read of the 12k lines.
- Scripts used are in `/tmp/claude-1000/-home-max-GOLEM/f8c735f2-cb0b-46cc-a631-673fdc69ab41/scratchpad/dedupB/` (`sim.py`, `three.py`, `closure.py`, `sum.py`).

---

# Redundancy audit C: lower-body and whole-body control

Scope: `core_ws/src/g1/g1_locomotion` against `h12_lowerbody_rl`, `lmg_deploy` and the Python
of `h12_deploy_mjpc`. Repo `/home/max/GOLEM`, branch `camera_ready`, HEAD `4de2506`.
Read-only analysis. Nothing in the repo was changed.

## Summary (ranked by lines saved x confidence / risk)

1. **There is no worthwhile G1-to-H1-2 code merge in this scope.** `g1_locomotion` is an independent
   design (raw cyclonedds, tick-grid pacing, fault latch). The only real overlaps are tiny
   (gravity projection, a 76-value observation layout, YAML key vocabulary), and the one file
   that would be shared, `policy/homie_policy.py`, is CC BY-NC-SA. Nearly all the savings are
   inside the H1-2 side.
2. **F1** Delete `h12_lowerbody_rl/h12_lowerbody_rl/new_policy_file/`. Its two `.pt` files are
   byte-identical (md5) to `policies/fame/`, and nothing references the directory. About 3.2 MB,
   risk none.
3. **F2** Retire or thin `fame_node.py` and `walking_node.py` (397 lines). They are older
   single-policy copies of `lowerbody_controller_node` and are used only by the Mac bringup.
   They also lack fixes the controller has (see §4). Risk is low (Mac sim only), effort S-M.
4. **F3** Extract the logic that `lowerbody_controller_node` and `marl_deploy_node` copy from each other
   (heartbeat watchdog, pre-pose ramp with identical constants, IMU-trim parameters, band-release
   client, LowCmd packing, confirm gate) into a shared H1-2 module. About 150-200 lines, effort M.
   Risk is medium: both are real-robot controllers and `h12_lowerbody_rl` has no behavioural tests.
5. **F4** Unify the H1-2 quaternion, gravity and IMU-trim math: 7 gravity-projection copies and
   2 trim implementations in scope, all numerically equal. About 60-80 lines, risk low with numeric
   tests. Best done together with F3.
6. **F5** `policy_manager.py` has about 90 lines of policy-switching handover gate that no caller reaches.
   Delete or keep, an owner call. **F6** `walk.yaml` and `almi.yaml` duplicate their leg limits and gains.
7. **Divergences worth a decision, not a merge (§4):** the H1-2 lower-body controller has no `/cmd_vel`
   timeout, no non-finite checks and no `/lowstate` staleness check, all of which the G1 node and
   `marl_deploy_node` have. `lowerbody_controller_node`'s pre-pose ramp has no peak-velocity limit,
   which `marl_deploy_node` added. Licence: any sharing that touches `homie_policy.py` pulls
   NC-SA code into the permissive packages (§6).

Method: `difflib.SequenceMatcher` over tokens (identifiers, numbers and punctuation, with comments and
docstrings stripped), called "token-sim" below. Numeric equivalence was checked by running both
functions on random inputs (host python with numpy and torch; onnxruntime stubbed). Work files are in
`.../scratchpad/dedup/c_work/`. I read every Python file of the four packages in full, except the C++ MJPC
cores (skipped as instructed) and the bodies of `reference/mujoco_deploy_h12_rma.py` and
`sim2real_gap_monitor.py`, which I only skimmed.

---

## 1. Inventory

None of the four packages is a git submodule (`git submodule status` lists neither; they are plain
tracked directories). `h12_deploy_mjpc` depends at runtime on the `mujoco_mpc` submodule's
`mjpc/deploy/helper_scripts/*.py`. Maintainers in `package.xml`: Niraj Pudasaini for `g1_locomotion`; `maxconway` for the three H1-2 packages.

### 1.1 `g1/g1_locomotion` (914 py lines in `g1_locomotion/`, 317 in `policy/`; MIT code + NC policy)

Entry point `locomotion_node` (`g1_locomotion/locomotion_node.py:1-94`), arguments
`--config g1.yaml|sim.yaml|path`, ROS param `policy_dir` (default `/home/code/g1_locomotion`).
`use_sim_time` is forced false (`locomotion_node.py:34-35`).

| Interface | Detail |
|---|---|
| ROS sub | `/cmd_vel` Twist, `/locomotion/height` Float32 (depth 10) |
| ROS pub / srv / action | none |
| DDS read (plain cyclonedds `DataReader`, keep-last-1 best-effort, WaitSet) | `rt/lowstate` (`LowState_`); `rt/safety/lowcmd_upper_in` (`LowCmd_`, only while WAITING, `driver.py:74-76,122-128`) |
| DDS write (SDK `ChannelPublisher`) | `rt/safety/lowcmd_lower_in` (`LowCmd_`): slots 0-11, mode 1, training kp/kd, `dq = tau = 0`, CRC computed; header `mode_pr` set, `mode_machine` stamped by g1_safety_layer |
| Config | `config/g1.yaml`, `config/sim.yaml` (checked by `config.py:check_config`, unknown keys rejected); `policy/g1.yaml`, `policy/deploy.onnx`, `policy/homie_policy.py` (loaded in place via `importlib`, `locomotion.py:55-78`) |
| Env | `ROS_DOMAIN_ID`, `G1_DDS_INTERFACE` (`config.py:155-178`) |
| Tests | `test_locomotion.py` (25), `test_locomotion_node.py` (3), `test_locomotion_dds.py` (4, in-process fake G1 plus the real `g1_safety_layer`), `policy/test_homie_policy.py` (9), plus the `tests/g1` closed-loop sim scripts |

### 1.2 `h12_lowerbody_rl` (2086 py lines in the package, 866 in `reference/`)

Entry points (`setup.py:35-45`): `walking_node`, `fame_node`, `lowerbody_controller_node`,
`sim2real_gap_monitor`. Only `lowerbody_controller_node` is launched by `h12_lowerbody.launch.py`.
`fame_node` and `walking_node` are launched only by `h12_sim_bringup_mac.launch.py:111-123`
(`GOLEM_LOWERBODY=fame|walk`).

`lowerbody_controller_node` (`scripts/lowerbody_controller_node.py`, 511 lines):

| Interface | Detail |
|---|---|
| Sub | `/lowstate` LowState (BE, depth 1); `/cmd_vel` Twist; `/lowerbody/squat_cmd` Float32; `/left_ee_pose` PoseStamped (param `band_release_topic`); `/safety/heartbeat` Bool |
| Pub | `/safety/lowcmd_lower_in` unitree_hg/LowCmd (ROS 2 topic, which is DDS `rt/safety/lowcmd_lower_in`); `/lowerbody/active_policy` String (latched) |
| Srv served | `/lowerbody/confirm_engage` Trigger |
| Srv called | `/elastic_band/toggle` Trigger |
| Params | `control_hz` 50, `active_policy`, `{walk,fame,almi}_config`, `imu_offset_{roll,pitch,yaw}_deg` (live, +-15), `default_height_cmd`, `prepose_ramp_s` 2.0, `engage_wait_for_confirm`, `disable_elastic_band`, `band_wait_for_frame_task`, `band_release_topic`, `band_max_wait`, `heartbeat_timeout` 0.5 |
| Config | `policies/{walk,fame,almi}/*.yaml` and TorchScript weights; `h12_bringup/config/lowerbody_{real,sim}.yaml` |

- `walking_node`: sub `/lowstate`, `/cmd_vel`; pub `/safety/lowcmd_lower_in`; client `/elastic_band/toggle`
  (blocking, called inside `__init__`).
- `fame_node`: as above plus `/lowerbody/squat_cmd` and, if band-waiting, `/left_ee_pose`; client `/elastic_band/toggle`.
- `sim2real_gap_monitor`: read-only. Sub `/safety/lowcmd_lower_in`, `/lowstate`, `/lowcmd`,
  `/lowerbody/active_policy`; argparse `--duration`, `--report-every`.
- Python modules: `policy.py` (`WalkPolicy`, `AlmiPolicy`, `FamePolicy`, quaternion helpers),
  `policy_manager.py`, `rma/` (FAME env-factor encoder).
- Tests: only the five ament lint boilerplate tests (copyright, flake8, mypy, pep257, xmllint).
  There are no behavioural tests in the package. The only functional coverage is
  `tests/stack/check_lowerbody.py` + `test_lowerbody_scoring.py` (a live ALMI balance/walk/stop run in sim,
  needs containers; the scoring unit tests are CPU-only).

### 1.3 `lmg_deploy` (1576 py lines)

Entry point `marl_deploy_node` (`lmg_deploy/marl_deploy_node.py`, 590 lines).

| Interface | Detail |
|---|---|
| Sub | `/lowstate` LowState (BE); `/livox/imu` Imu (sensor-data QoS); `/cmd_vel` Twist; `/safety/heartbeat` Bool; `/Odometry` nav_msgs/Odometry; `~/arm_goal` PoseArray |
| Pub | `/safety/lowcmd_lower_in` and `/safety/lowcmd_upper_in` (LowCmd); `~/state` String (latched); `~/wrist_error` Float32MultiArray |
| Srv served | `~/confirm_engage` Trigger; `~/end_arm_goal` Trigger; `~/set_mode` `LmgSetMode`; `~/set_arm_goals` `LmgSetArmGoals` (both `custom_ros_messages`) |
| Srv called | `/elastic_band/toggle` Trigger |
| Params | `config` (policy yaml path), `urdf_path`, `control_hz`, `prepose_ramp_s` 6.0, `prepose_max_vel` 0.25, `engage_wait_for_confirm`, `disable_elastic_band`, `band_release_delay_s`, `heartbeat_timeout`, `lowstate_timeout`, `livox_imu_timeout`, `livox_gravity_tau_s`, `livox_attitude`, `livox_odom_timeout`, `imu_offset_*_deg` |
| Config | `config/policy_template.yaml`, `policies/<name>/{<name>.yaml, export.yaml, legs.pt, arms.pt}`, the H1-2 URDF |
| Pure modules | `controller.py` (ROS-free MARL step), `kinematics.py` (Pinocchio IK), `networks.py`, `rotations.py`, `livox_imu.py`, `livox_attitude.py` |
| Tests | `test_controller.py` (25), `test_livox_imu.py` (10), `test_livox_attitude.py` (5), `test_imu_trim.py` (4); `sim_drive.py` is a helper |

### 1.4 `h12_deploy_mjpc` Python only (466 lines)

All three are rclpy parameter shells around work done outside the ROS process (CMake installs them as
`estimator_node`, `mjpc_deploy_lowerbody_controller`, `mjpc_debug_visualizer`; `CMakeLists.txt:36-50`).

- `estimator_node.py`: declares about 25 params, destroys the node, then `runpy`-runs the submodule's
  `base_estimator_node.py`. That worker reads `rt/lowstate` and writes `rt/sportmodestate_est` over
  unitree_sdk2py. No ROS pub/sub/srv.
- `controller_launcher.py`: about 25 params mapped to CLI flags of the C++ `mjpc_lowerbody_core` /
  `mjpc_fullbody_core`, run by `Popen`. It also has a thread that calls `/elastic_band/toggle` once after
  `band_drop_after_secs` of node-clock time (`controller_launcher.py:35-68`). The core does its own DDS I/O.
- `debug_visualizer_node.py`: the same shell shape around `plan_visualizer.py` (reads `rt/lowstate`,
  `rt/mjpc/plan`, `rt/sportmodestate_est`).
- Config: `config/controller.yaml`, `h12_bringup/config/mjpc_{sim,real}.yaml`. No tests.

---

## 2. Overlap findings

Format: locations, class, evidence, lines, differences, proposal, risk.

### F1. `new_policy_file/` is a stale copy of `policies/fame/` (class A, inside h12_lowerbody_rl)

- Locations: `h12_lowerbody_rl/h12_lowerbody_rl/new_policy_file/{policy_3600.pt, encoder_3600.pt,
  h1_2_rma_magpie_jun10_torso_neg10.yaml}` against `policies/fame/{policy_3600.pt, encoder_3600.pt, fame.yaml}`.
- Evidence: md5 `ee47793355c890b72142716ff662514e` (policy) and `02ce58ca8fb4af710d5356003cbc374a` (encoder)
  are identical on both sides. `grep -rn new_policy_file` over the repo returns no references. The yaml
  is the training-side config (`xml_path: ../HomieRL/...`, `kps_arms`), not the ROS deploy config.
- Size: 3.17 MB of weights plus an 82-line yaml. The directory sits inside the python package source tree (no `__init__.py`, so
  `find_packages` does not install it, but it is tracked in git).
- Differences: none in the weights. The yaml has `kds` 5.0 against `fame.yaml`'s 6.0, a different
  `default_angles` layout (`[-0.16, 0, 0, ...]`) and a -0.35 rad torso default. Dead either way.
- Proposal: delete the directory.
- Risk: none functional; the weights remain in `policies/fame/`. Check no external script uses the path (none in the repo).
- Effort: S.

### F2. `fame_node.py` and `walking_node.py` are older copies of `lowerbody_controller_node` (class B)

- Locations: `scripts/fame_node.py:1-195`, `scripts/walking_node.py:1-202` against
  `scripts/lowerbody_controller_node.py:1-511` plus `policy.py:220-285` (`WalkPolicy`) and `policy.py:439-568` (`FamePolicy`).
- Evidence (token-sim, method-level):
  - `_on_cmd_vel` (fame_node 114-118, walking_node 128-135) against controller 257-261: **1.000**.
  - `fame_node._release_band` 127-141 against controller `_release_band` 358-376: 0.657.
  - `walking_node._tick` 137-185 against `WalkPolicy.compute` 258-285: 0.623 (same obs layout, same gait
    clock, same `action * scale + default`).
  - `walking_node.get_gravity_orientation` 33-43 against `gravity_from_quat_walk` 69-79: 0.581 in text,
    the same closed form (float64 against float32).
  - The `/lowstate` to `RobotState` conversion in `fame_node._tick` is the same list comprehensions as
    `_state_from_lowstate` (controller 387-397).
  - The docstrings say so: `fame_node.py:9` ("Reuses `FamePolicy`...; the generic controller is
    `lowerbody_controller_node`"), `walking_node.py:10`.
- Lines: 397 on the duplicate side. The policy math is already shared (fame_node uses `FamePolicy`);
  `walking_node` keeps its own inline `WalkPolicy` copy (torch load, scales, obs, gait clock).
- Differences (accidental divergence, so possible bugs in the old nodes):
  1. `_publish`: `fame_node.py:169-178` and `walking_node.py:175-185` do not set `mode_pr` or
     `mode_machine`. The controller sets `MODE_PR = 0, MODE_MACHINE = 6` (`lowerbody_controller_node.py:339-356`) because
     the safety layer forwards the header of whichever channel spoke last
     (`h12_safety_layer/core/safety_layer.py:274`). Harmless in the sim (the sim never checks it:
     no `mode_machine` in `h1_robocasa/unitree_interface.py`), a real-robot hazard if used there.
  2. Band release: `fame_node._release_band` 127-141 sets `_band_released = True` first, then
     `wait_for_service(1.0)`; if the service is not up it never retries. The controller's comment
     (358-366) describes exactly this as the bug it fixed (non-blocking retry until the service is ready).
     `walking_node._disable_elastic_band_if_present` 110-121 calls `spin_until_future_complete` inside `__init__`.
  3. Neither has the heartbeat watchdog, confirm gate, pre-pose ramp or IMU trim. That is deliberate for a
     bench node, but the nodes are reachable from the Mac bringup.
- Consolidation proposal: make `GOLEM_LOWERBODY=fame|walk` in `h12_sim_bringup_mac.launch.py:111-123` launch
  `lowerbody_controller_node` with `active_policy:=fame|walk` (and `engage_wait_for_confirm:=false`, as
  `lowerbody_sim.yaml` does). Then delete both nodes, their `setup.py` entry points (`setup.py:38-41`), the
  `docker/mac/scripts/launch_ros_mac.sh:122` comment and the `CLAUDE.md` §4 entry-point list. Alternative: keep the
  names as 10-line wrappers that default `active_policy`.
- Risk: Mac sim only (not in the real bringups). Behaviour change: the controller pre-poses over
  2 s and waits for `/safety/heartbeat` before moving, whereas `fame_node` releases the band as soon as
  `/left_ee_pose` appears; the Mac docs say FAME "balances the robot standing UNSUPPORTED (verified)" with `fame_node`,
  so re-verify on Mac. The package is in the repo's own tree (not a submodule), owner `maxconway`.
- Tests that guard it: none (no unit tests for either). `tests/stack` covers only the ALMI controller path.
- Effort: S-M.

### F3. `lowerbody_controller_node` and `marl_deploy_node` share copied scaffolding (class B)

`lmg_deploy/README.md:12` says the node is "Modelled on `h12_lowerbody_rl`'s `lowerbody_controller_node`", and the
comments cite it (`marl_deploy_node.py:40,76`). The copies:

| Piece | lowerbody_controller_node | marl_deploy_node | token-sim | notes |
|---|---|---|---|---|
| Heartbeat watchdog (arm on first beat, latch on False or timeout) | `_on_heartbeat` 269-279, `_trip_heartbeat` 281-289, check at 398-410 | `_on_heartbeat` 343-349, `_latch` 394-399, check at 482-484 | **0.961** / 0.627 | same semantics, same default 0.5 s, same topic constant (`:80` / `:138`) |
| Pre-pose cosine ramp + settle/timeout | constants 99-101, logic 415-470 | constants 139-141, `_prepose` 519-573 | block 0.39, **constants identical** (0.15 / 10 / 400), blend `0.5*(1-cos(pi*frac))` identical | see differences |
| Confirm gate | `_on_confirm_engage` 291-303, `_clear_commands` 305-310 | `_on_confirm` 351-360 | 0.631 | both zero any pre-engage `/cmd_vel` |
| IMU trim params (3 live params, +-15 deg, finite check, merge-with-current in callback) | `IMU_OFFSET_*` 66-67, `_build_imu_corr` 312-315, `_on_set_parameters` 316-333 | `IMU_OFFSET_*` 143-144, `_build_imu_mount` 273-277, `_on_set_parameters` 279-289 | 0.78 | same declared range and description text |
| IMU trim maths | `policy.py:121-146` (`imu_offset_quat`, `apply_imu_offset`) | `rotations.py:61-70` (`imu_mount_quat`) + `quat_conjugate` use | not text-similar (0.277) | **numerically equal**: max diff 1.1e-7 over 500 random (quat, rpy); `lmg_deploy/test/test_imu_trim.py:44-49` pins the composition order to "as h12_lowerbody_rl" |
| `quat_mul` | `policy.py:106-118` | `rotations.py:12-20` | **0.975** | identical Hamilton product |
| Elastic-band client | `_release_band` 358-376, `_on_band_toggled` 378-385 | `_release_band` 457-464, `_on_band_toggled` 466-475 | 0.684 / 0.775 | same service, same retry-until-ready idea |
| LowCmd packing (`mode=1`, `dq = tau = 0`, kp/kd, header) | `_publish_leg` 339-356 | `_publish` 439-455 | 0.54 | see differences |
| Latched silent state, `/lowstate` to state conversion | `_state_from_lowstate` 387-397 | `_robot_state` 406-430 | low | differ (livox IMU in lmg) |

About 150-200 duplicated lines on each side. `controller_launcher.py:35-68` is a fifth band client (class C: a
thread that waits N node-clock seconds then toggles once).

Differences:
- **Accidental or refinement divergence (flag):**
  1. Pre-pose speed. `marl_deploy_node._prepose` 523-526 stretches the ramp so the peak target speed is at most
     `prepose_max_vel` = 0.25 rad/s (docstring 6-11: a target running ahead of the joint trips the safety layer's
     torque e-stop). `lowerbody_controller_node` uses a fixed 2.0 s ramp (`prepose_ramp_s`, `lowerbody_real.yaml`
     does not override it), so the peak speed is about 1.57 x distance / 2 s; for the ALMI deep-crouch nominal
     (hip -0.4, knee 0.8 from straight legs) that is about 0.6 rad/s. The controller's own comment (94-98)
     records a real-robot dq e-stop from a too-fast pre-pose, so this may be a latent issue.
  2. `/lowstate` health. `marl_deploy_node` skips stale `/lowstate` (`lowstate_timeout` 0.1 s, 485-490) and
     latches on non-finite state/observations/commands (496-498, 511-514, 440-442).
     `lowerbody_controller_node` does neither: it steps the policy from the last sample indefinitely and
     publishes at 50 Hz whatever comes out. (`h12_safety_layer/core/safety_checks.py:34-42,84` checks for non-finite values in the commands it relays, so a NaN
     would probably end in an estop; a stale state would not be caught by it. I did not trace the safety layer's response.)
  3. Header. The controller hard-codes `MODE_PR = 0, MODE_MACHINE = 6` (72-73); `marl_deploy_node` echoes the
     live `msg.mode_machine` (443-447, 515). The sim reports 0 and the real H1-2 6.
  4. Band release also resets the controller episode in lmg (`_on_band_toggled` 474-475) and resets the policy
     at release in the controller (`reset_active`, 455-470); different but both intended.
- **Robot/task-specific (keep):** all 27 joints and two channels in lmg against 12 legs; pre-pose over the
  policy's own default; `waiting_imu` and FAST-LIO levelling in lmg; policy objects.
- Consolidation proposal: a small pure-Python, rclpy-light module both packages depend on. Options, in order
  of preference: (a) a new ament_python package `h12_deploy_common` (both consumers are plain
  directories; neither is a submodule); (b) put it in `h12_lowerbody_rl` and make `lmg_deploy` depend on it.
  Contents: `HeartbeatWatchdog` (topic, timeout, `trip` state), `ImuTrim` (declare params, callback, quaternion
  correction), `PrePoseRamp` (cosine blend, optional `max_vel` stretch, settle and timeout counters),
  `ElasticBandReleaser` (non-blocking retry-until-ready, completion callback), `pack_lowcmd(q, kp, kd, rows,
  mode_pr, mode_machine)`. Each node keeps its orchestration. The confirm gate is 10 lines and can stay in the nodes.
- Risk: **real-robot impact, both controllers run on hardware today** (H1-2 real bringup and `lmg_real_bringup`).
  Moving the ramp to the lmg rule (max-vel stretch) changes the lowerbody pre-pose duration (longer), which is the
  intended fix but is a behaviour change to validate on a gantry. Ownership: same people, no submodule boundary,
  but `h12_bringup` params (`lowerbody_real.yaml`, lmg launch args) must keep working. Guarding tests:
  `lmg_deploy/test/test_imu_trim.py` and `test_controller.py` cover only the pure parts; there are **no tests for
  the node scaffolding on either side** (and none at all in `h12_lowerbody_rl`), so the refactor should start by
  writing tests against the extracted pure classes (the pre-pose and watchdog logic is easy to unit-test with a fake clock).
- Effort: M.

### F4. Quaternion, gravity and IMU-trim maths inside the H1-2 side (class B / C, numerically equal)

Gravity projection exists **seven times** in the scope; all give the same vector:

| Site | Lines |
|---|---|
| `h12_lowerbody_rl/policy.py` `gravity_from_quat_walk` (closed form) | 69-79 |
| `h12_lowerbody_rl/policy.py` `quat_rotate_inverse` + `gravity_from_quat_fame` | 82-103 |
| `h12_lowerbody_rl/scripts/walking_node.py` `get_gravity_orientation` (copy of the closed form) | 33-43 |
| `h12_lowerbody_rl/scripts/sim2real_gap_monitor.py` `_gravity_x` (one component) | 58-60 |
| `h12_lowerbody_rl/reference/deploy_mujoco.py` and `reference/mujoco_deploy_h12_rma.py` (provenance copies) | 11-23, 125-143 |
| `lmg_deploy/rotations.py` `projected_gravity` (`quat_apply_inverse`) | 92-95 |
| `g1_locomotion/locomotion.py` `gravity_z` (z only) and `policy/homie_policy.py` `quat_rotate_inverse`, `get_gravity_orientation` | 143-146; 48-75 |

Evidence: with 500 random unit quaternions, H1-2 `gravity_from_quat_walk` against `get_gravity_orientation`
(HOMIE) max abs diff 3.0e-8; lmg `projected_gravity` against HOMIE 5.0e-16; G1 `gravity_z` against HOMIE's z exactly 0.
`quat_rotate_inverse` HOMIE (`homie_policy.py:48-69`) against H1-2 (`policy.py:82-99`): token-sim **0.915**
after the `q_conj`/`qc` rename, numerically equal to 3.1e-7 (1000 samples); against the H1-2
`reference/mujoco_deploy_h12_rma.py:125-138`: **0.979**. So the helper has a shared upstream (legged_gym lineage).

Internal H1-2 redundancy: `policy.py` carries two implementations of one quantity ("each policy uses the
convention it was trained with", `policy.py:66`) that agree to 3e-8, so the split is not needed.

Proposal: one `rotations.py` (numpy, wxyz) for the H1-2 packages: `quat_mul`, `quat_conjugate`, `projected_gravity`,
`imu_mount_quat`. Natural home is the F3 `h12_deploy_common`; `lmg_deploy/rotations.py` is already the more complete
module (it also has `quat_apply`, `axis_angle`, `subtract_frame_transforms`, which stay lmg-only). Fold `walking_node`'s copy
into F2's deletion. Drop `gravity_from_quat_walk` (3e-8 change in float32 observations, still re-run `tests/stack`).
**Do not** make `homie_policy.py` import it (it must stay standalone, NC-licensed; see §6); `gravity_z` in the G1 package can
stay as the 4-line MIT copy it is.
Risk: low; pure functions; dtype differences (float32 in policy.py, float64 in lmg) need explicit casts. Tests:
`lmg_deploy/test/test_imu_trim.py` and `test_controller.py` already pin `rotations`; add a parity test against
the legged_gym formula. Effort: S. Lines saved about 60-80.

### F5. `policy_manager.py` handover gate is unused (not cross-robot; same package)

- `PolicyManager.update_switch`, `_gate_ok`, `GateConfig` fields and the `require_stop` plumbing
  (`policy_manager.py:30-47,100-110,114-135,138-153`, roughly 90 of 162 lines) have no caller:
  `grep -rn "update_switch\|_gate_ok\|require_stop\|GateConfig"` outside the file hits only the import and the
  `GateConfig()` construction at `lowerbody_controller_node.py:68,188`. The node's own docstring says "there is no
  switching between policies" (lines 14-15).
- Proposal: delete or leave documented as future work; a decision for the owner. Collapsing the manager to a
  dict plus an active name is about 70 lines smaller. Risk low, effort S.

### F6. Leg limits and gains duplicated across the policy YAMLs (class A inside the package)

- `policies/walk/walk.yaml:16-20` and `policies/almi/almi.yaml:14-22`: `kps`, `kds`,
  `legs_motor_pos_lower_limit_list`, `legs_motor_pos_upper_limit_list` and all scales are identical
  (checked with `yaml.safe_load`; the almi comment says "same values as walk.yaml").
  `fame.yaml:11-26` differs: hip-yaw limit 0.36 against 0.37, `kds` 6.0 against 2.5, ankle-pitch kp 60 against 40.
  The 0.36/0.37 difference is undocumented (may be intentional, the trained range).
- The H1-2 hardware limits also live in `h12_safety_layer/config/*.yaml` and the URDF. Proposal: a shared
  `leg_limits` file loaded by `_load_leg_position_limits`, or per-policy override only. About 8 lines, effort S,
  low value; do it with F3 if at all.

### F7. FAME single-step observation against HOMIE's observation (class C, numerically identical)

- Locations: `homie_policy.py:153-166` (`HomiePolicy.observation`, 76 values: cmd, height, gyro, gravity,
  27 q, 27 dq, 12 last action) against `policy.py:506-520` (`FamePolicy._single_obs`, 76 values, same order).
  History: `homie_policy.py:146-151,168-190` (deque of 6, zero-filled at reset) against `policy.py:533-566` (deque of 3,
  zero-filled at reset, plus `z_t`).
- Evidence: `FamePolicy._single_obs` and `HomiePolicy.observation` with the same scales and defaults: max abs diff
  **4.8e-7** over 200 random states (same shapes (76,)). Token-sim only 0.179 (independent code). The H1-2 FAME is trained in a
  HomieRL fork (`new_policy_file/*.yaml:3` `xml_path: ../HomieRL/legged_gym/...`), which is why the layouts match.
- Differences, all robot or policy specific: `ang_vel_scale` 0.25 (H1-2) against 0.5 (G1, documented as needed to turn
  and walk backwards, `policy/README.md` "Changes from HOMIE's MuJoCo script"), `cmd_scale` yaw 0.25 against 0.5, history length 3
  against 6, torch against onnxruntime, default pose of 27 joints (arms 0 in both), FAME's extra encoder `z_t`.
  **Accidental divergence (flag):** `HomiePolicy.step` checks the observation for shape and non-finite values before touching
  its history ("one NaN kept in the history would reach every later output", `homie_policy.py:178-185`);
  the H1-2 `WalkPolicy`, `AlmiPolicy` and `FamePolicy` have no such check and a NaN in `dq` or `quat` would
  enter `_action` and the history.
- Proposal: **do not merge.** About 40 lines would be shared, the cost is a dependency on NC-SA code (§6) or a rewrite of
  `HomiePolicy`, and the policies are trained separately. Better: copy the NaN-guard idea into the H1-2 policy
  `compute` methods (a behaviour change that fails loudly; the controller needs a place to catch it) and, if wanted,
  add a cross-check test that both layouts agree (the script above is 30 lines).
- Risk of merging: licence entanglement, real-robot behaviour change on the H1-2 path with no unit tests. Effort if
  done anyway: M.

### F8. Policy YAML vocabulary (class B at the key level, different loaders)

- `policy/g1.yaml` and `policies/walk/walk.yaml` share 12 keys from Unitree's deploy configs: `policy_path`, `xml_path`,
  `simulation_duration`, `simulation_dt`, `control_decimation`, `default_angles`, `ang_vel_scale`, `dof_pos_scale`,
  `dof_vel_scale`, `action_scale`, `cmd_scale`, `cmd_init`, `num_actions`. Different: the G1 uses substring-matched
  `stiffness`/`damping` by joint name and `policy_joints`, `obs_history_len`, `height_cmd`, `cmd_range`, `height_range`,
  `clip_actions`; the H1-2 uses per-index `kps`/`kds` and limit lists.
- Loaders: `homie_policy.py:25-45` (`load_config`, resolves paths relative to the file, falls back to the repo root)
  against `policy.py:149-151` (`_load_yaml`) with per-policy `cfg[...]` reads (`policy.py:224-252,328-372,454-500`).
- Proposal: leave. The G1 loader is entangled with the NC file; the H1-2 one is 3 lines. The G1 `config.py` strict key checker
  (`check_config`, 179 lines) is a nicer pattern the H1-2 packages could adopt, but that is a feature, not dedup.

### F9. `/cmd_vel` handling and command conditioning (class C, different on purpose; divergences flagged)

| | G1 `g1_locomotion` | H1-2 `lowerbody_controller_node` | lmg `marl_deploy_node` |
|---|---|---|---|
| Input clip | trained ranges (`clip_command`, `homie_policy.py:137-144`) then deployment limits (config, `commands.vx` etc. -0.3..0.5), `locomotion.py:149-172` | clip to +-1 (`:257-261`), then `cmd_scale` in the policy; ALMI adds `cmd_gain` and `cmd_smooth_alpha` (`policy.py:374-380`) | `np.clip` to `command_ranges` from the config (`controller.py:set_velocity_command`) |
| Rate limit | `commands.accel` (0.5, 0.5, 1.0) and `height_rate` 0.1 | exponential smoothing (ALMI only, alpha 0.15) | none |
| Timeout | `/cmd_vel` older than 0.5 s means zero (`locomotion.py:463-466`) | **none** (the comment at lines 134-136 notes a stale value keeps walking) | **none** |
| Non-finite | warn and stop (`set_cmd_vel` 256-262) | none (`np.clip` leaves NaN) | none (a NaN later latches the node via the observation check) |
| Height | `/locomotion/height`, rate-limited, default 0.74 | `/lowerbody/squat_cmd` (FAME), default 1.0, not rate-limited | n/a |

The same job done independently three ways, so class C. The G1 node's timeout and finite check are the pieces the H1-2 controller lacks.
Not a merge candidate (the H1-2 side is policy-specific, e.g. ALMI's cmd gain and onset reset, and the safety nets differ), but a
**port candidate**: add a `/cmd_vel` timeout and finite check to `lowerbody_controller_node` (a behaviour change on hardware;
nav2 stops by publishing zeros today, so a timeout only matters when the publisher dies).

### F10. Elastic-band release client x5 (class B / C)

Part of F2/F3: `fame_node.py:127-141`, `walking_node.py:110-121`, `lowerbody_controller_node.py:358-385`, `marl_deploy_node.py:457-475`,
`controller_launcher.py:35-68`. The H1-2 sim fixture only; the G1 sim has none (no `elastic`/`band` in `h1_robocasa/g1/*.py`).
One helper with the non-blocking retry semantics covers the first four; `controller_launcher.py` (sim-clock timer thread) can stay.
Effort S inside F3.

### F11. Pre-pose ramp (cosine, H1-2) against the G1's ramp (linear) and fault latches (class C / D)

`lowerbody_controller_node.py:415-470` and `marl_deploy_node.py:519-573` (cosine, from the measured pose to the policy's default, then a settle
check) do the same job as `locomotion.py:450-461` (`_ramp`, linear over `startup.ramp_seconds` 3.0 s real / 0.5 s sim, no settle check, the policy starts after the last
step). The G1's ramp is driven by the tick grid, not a timer, and followed by a start-state machine (`_wait`, 391-448) that has no counterpart in the
H1-2 nodes. Not worth sharing (about 10 lines of maths); keep the G1 design.

---

## 3. Keep separate (looks duplicated, should stay apart)

1. **The G1 node plumbing against the H1-2 node plumbing.** `g1_locomotion` reads `rt/lowstate` through a plain cyclonedds
   `DataReader`/`WaitSet`, paces the policy on the robot tick (20 ms grid, drops backlog), forces `use_sim_time` false and publishes through the SDK
   `ChannelPublisher` with a CRC (`driver.py`, `locomotion.py:117-141,324-337`). The H1-2 nodes are rclpy timer loops at 50 Hz on the newest
   ROS `LowState` and publish `unitree_hg/LowCmd` ROS messages (no CRC; the safety layer adds it). On the wire both end up on
   `rt/safety/lowcmd_lower_in`. The reasons are in the `driver.py:30-48` docstring and the G1 README ("Real time"): a latency budget of 25 ms from the DDS write. A common
   base class would have to abstract the step trigger, the clock and the transport, and would lose the G1's timing guarantees.
2. **Fault semantics.** The G1 node "latches silent" and relies on `g1_safety_layer`'s 0.2 s leg watchdog to estop
   (`locomotion.py:476-481`, README "Faults"). The H1-2 nodes latch silent too (`_hb_dead`, `_dead`) but `marl_deploy_node.py:49-50` states the H1-2
   safety layer holds the last targets and "has no staleness watchdog on these channels". Same word, different safety contract; do not unify.
3. **Start interlocks.** G1: no operator confirm; waits for 1 s of steady tick, upright pose, a unit quaternion, a commanding arm controller and a plausible tick rate
   (`locomotion.py:391-448`), then self-starts; real-robot gate is `start_position_verified` plus `startup_refusal` in `g1_core.launch.py`. H1-2: pre-pose, then a
   `confirm_engage` Trigger on the real robot, plus `start_position_verified` in `h12_bringup`. The launch-level `start_position_verified` pattern is shared
   (bringups are out of this scope); the node-level behaviour is not.
   Note the G1 has no operator confirm step: the walking node starts moving the legs by itself once the conditions hold. That follows CLAUDE.md §6 ("Engage gate,
   no engage service") and is a design difference, not drift, but is worth knowing when comparing the two arming procedures.
4. **Gait clocks.** `WalkPolicy`/`AlmiPolicy` use a wall/sim-clock phase (`state.t`, period 0.8 s, `policy.py:258-285,390-436`); lmg uses a step counter
   (`controller.py:_gait_phase`); HOMIE has no gait clock. Different policies.
5. **PD gains, default poses, joint order maps.** G1: names to slots through `g1_description.layout.SLOT_BY_NAME`, asserted to equal slots 0-12 and 15-28
   (`locomotion.py:81-89`), gains matched by joint-name substring. H1-2: implicit `/lowstate` index order, `OBS_JOINT_IDX = np.r_[0:17, 20:24]` for ALMI
   (`policy.py:326`), explicit `joint_names` in lmg yaml. Robot-specific (hip order differs: G1 pitch-roll-yaw, H1-2 yaw-pitch-roll). The H1-2 27-name list is repeated in
   `lmg_deploy` (5 files), `reference/` (2) and `h12_ros2_controller/utility/joint_definition.py`; that is a joint-table question for the H1-2 controller
   package, outside this scope.
6. **`policy_manager.py` registry and the RMA encoder** (`rma/`): H1-2-only (G1 runs one policy).
7. **`reference/` scripts against `policy/mujoco_deploy_g1.py`.** Both descend from Unitree's `deploy_mujoco.py` (`pd_control` is identical, token-sim 1.000); the H1-2 ones
   are provenance, "not maintained to run as-is" (`reference/README.md`), the G1's is a working sim2sim test. Leave.
8. **`h12_deploy_mjpc`.** Nothing on the G1 side. The estimator (leg-odometry EKF) and the MJPC controller launcher are H1-2 only; the G1 gets odometry from FAST-LIO or
   the sim. `estimator_node.py`, `debug_visualizer_node.py` share a header (`_MJPC_ROOT`, `_DEFAULT_SCENE`, domain/iface/lowstate params, about 12 lines, class B); not worth a helper.
   `controller_launcher.py` is a process launcher, unrelated to the policy runners.
9. **`sim2real_gap_monitor.py`** (H1-2 diagnostic) against `LocomotionController.log_timing` (G1 per-10 s timing log): similar goal, different mechanism (a ROS node across three topics against in-loop stats).
10. **lmg's `livox_attitude.py` and `livox_imu.py`.** The G1 uses the `rt/lowstate` IMU directly (config `imu_frame: pelvis`, no trim). lmg's FAST-LIO-levelled attitude is H1-2-specific
    (head-mounted MID-360). A cross-package neighbour in `h12_slam/level_odom.py` (accelerometer levelling of `camera_init`) and G1's `g1_navigation/level_frame.py` belongs to the navigation audit.

## 4. Divergences that may be bugs (one-line index)

| Where | Observation | Real-robot relevance |
|---|---|---|
| `lowerbody_controller_node.py:99-101,141` against `marl_deploy_node.py:154-155,523-526` | Pre-pose ramp has no peak-velocity limit (fixed 2.0 s); lmg stretches to 0.25 rad/s | H1-2 real: the controller's own comment records a dq estop from a fast pre-pose |
| `lowerbody_controller_node.py:398-413` against `marl_deploy_node.py:485-498` | no `/lowstate` staleness or non-finite check | H1-2 real |
| `lowerbody_controller_node.py:257-261` against `locomotion.py:256-262,463-466` | no `/cmd_vel` timeout or finite check | H1-2 real and sim |
| `policy.py:254-566` against `homie_policy.py:168-190` | no NaN guard on observations in `WalkPolicy`/`AlmiPolicy`/`FamePolicy` | H1-2 real |
| `fame_node.py:169-178`, `walking_node.py:175-185` against `lowerbody_controller_node.py:339-356` | LowCmd header (`mode_pr`, `mode_machine`) not set | only if these nodes run on hardware (they are launched in the Mac sim only) |
| `fame_node.py:127-141` against `lowerbody_controller_node.py:358-385` | band release gives up if the service is not ready | Mac sim |
| `lowerbody_controller_node.py:72-73` against `marl_deploy_node.py:443-447` | `mode_machine` hard-coded 6 against echoed | inconsistent; fine while the real H1-2 reports 6 |
| `fame.yaml:19-22` against `walk.yaml:16-20` | hip-yaw limit 0.36 against 0.37 | probably the trained range; undocumented |

## 5. What would test a consolidation

- `h12_lowerbody_rl`: nothing but lint tests. Any F3/F4 change needs new unit tests first; the closest live guard is `tests/stack/run.sh robocasa locomotion` (ALMI,
  needs images and a GPU worker) and its CPU-only `test_lowerbody_scoring.py`.
- `lmg_deploy`: `test_controller.py` (25 tests: observations, IK, SLAM odometry, modes), `test_imu_trim.py`, `test_livox_*`. Nothing for `marl_deploy_node` itself
  (`test/sim_drive.py` is a manual helper).
- `g1_locomotion`: strong coverage (41 tests incl. an in-process DDS test with the real safety layer) plus `tests/g1/test_sim_bringup.sh --walk`. Nothing here needs to change for the above.
- `h12_deploy_mjpc`: none.

## 6. Licence note (HOMIE is CC BY-NC)

- `policy/` (`deploy.onnx`, `g1.yaml`, `homie_policy.py`, `mujoco_deploy_g1.py`, `test_homie_policy.py`) is treated as CC BY-NC-SA 4.0 (`policy/README.md` "License",
  `policy/LICENSE` is CC BY-NC 4.0 text). The rest of `g1_locomotion` is MIT (`LICENSE`, `package.xml`) and deliberately does not copy `policy/`: it loads it by path with
  `importlib` (`locomotion.py:55-78`); `locomotion.py:143-146` re-derives `gravity_z` itself. `CLAUDE.md` §8 requires `homie_policy.py` to stay inside `policy/` and `LICENSE`/`README.md` beside it.
- The H1-2 packages carry no licence metadata (`h12_lowerbody_rl` and `lmg_deploy` both say `TODO: License declaration`; `h12_deploy_mjpc` says Apache-2.0). ALMI's weights are BSD-3
  (`policies/almi/README.md`).
- **Consolidations that would mix NC code into the permissive packages:** F7 (making the H1-2 `FamePolicy` import or copy from `homie_policy.py`, or moving the shared observation
  builder into it) and F4 if the shared `rotations` module were made to import from, or be merged into, `homie_policy.py`. Likewise any `h12_deploy_common` that imports `homie_policy`.
  The reverse (G1 code importing a permissive shared module) is allowed licence-wise, but `homie_policy.py` is meant to be standalone (numpy, onnxruntime, pyyaml), and the sim stand-in
  (`h1_robocasa/g1/stand_in.py`) and the tests load it directly, so adding a ROS-package dependency there would break them.
- Consolidations that stay clean: F1, F2, F3, F5, F6 (all inside the H1-2 side) and F4 as long as the new shared module is written from the H1-2/lmg code (or from legged_gym's formula) and
  is not derived from `homie_policy.py`.
- **Open question for the owners (not verified):** the H1-2 FAME policy was trained in a HomieRL fork (`new_policy_file/*.yaml:3`, and `reference/mujoco_deploy_h12_rma.py` has the same helper
  functions as HOMIE's deploy script, token-sim 0.979 for `quat_rotate_inverse`). If OpenHomie's CC BY-NC covers HomieRL, the FAME weights and reference script may already carry the NC
  condition on the H1-2 side. I did not check HomieRL's licence or the provenance of the helper code. `quat_rotate_inverse` is believed to originate from legged_gym (BSD-3), which I did not verify.

## 7. Not examined or only skimmed

- C++ MJPC cores (skipped as instructed), `reference/mujoco_deploy_h12_rma.py` and `sim2real_gap_monitor.py` bodies beyond headers and the gravity helpers (skimmed),
  `lmg_deploy/test/sim_drive.py`.
- The safety layers and bringups were read only where they touch the publish path and interlocks (`h12_safety_layer/core/safety_layer.py` header forwarding and staleness,
  `g1_core.launch.py` refusals, `h12_bringup` lowerbody/lmg launch files); their own overlaps belong to other audit scopes.
- Untracked, gitignored leftovers `core_ws/src/locomotion_game_deploy/` and `core_ws/src/h1_bringup/` hold only `__pycache__`/`.pytest_cache` from old package names; no source.
- Not run: any ROS node, container, build or test suite. All figures come from host-side `diff`/`difflib`/numpy over source text.

---

# Redundancy audit D: navigation, skills, bringup, description (G1 vs H1-2)

Scope: `h12_slam`, `h12_skills`, `g1_navigation`, `h12_bringup`, `g1_bringup`, `h12_ros2_model`, `g1_description`.
Repo `/home/max/GOLEM`, branch `camera_ready`, HEAD `4de2506`. Read-only analysis; nothing in the repo was changed.

## Summary: top consolidation candidates, ranked by (lines saved x confidence) / risk

1. **Reuse `livox_imu_upright` for the real G1 lidar** (F4). The G1 documents the exact problem the H1-2 solved (driver rotates points by `MID360_config.json` roll 180 but not the IMU) in `g1_fastlio.yaml`, `g1_navigation.launch.py` and its README (8 `TODO(lidar)` markers in all, 3 files). Effort S, low risk, unblocks the G1's `livox_to_pointcloud` deskew and FAST_LIO extrinsic. Saves a re-implementation (~120 lines) more than existing lines.
2. **Launch-file boilerplate** (F16, F17). `_include` is defined 6 times in H1-2, `ASSETS_DIR` 7 times with 3 values, the mac bringup copies the state/control nodes, the x86 and desktop bringups each carry a ~20-line commented-out slider block plus a dead `use_sliders` argument; G1 has `_as_bool` x2 and `to_matrix` x2. About 150 lines, launch-only, low risk, effort S.
3. **One frontier explorer for both robots** (F1-F3). `h12_skills`' mixin (427 lines) and `g1_navigation`'s node (537 lines) serve the same action with the same defaults; the G1 node already says "the H1-2 skill's policy" plus three additions. Putting G1's pure functions in a shared package and running one node on both robots deletes ~430 lines and gives the H1-2 the G1's guards (busy-goal rejection, input validation, sim-clock deadlines). Effort M-L, medium risk: the algorithms differ in goal choice (H1-2 "abyss" standoff vs G1 footprint-fit), and the H1-2 has no explorer test. It would also fix the mac stack, which has no server for `/skill/frontier_explore` (D10). A staged S version (shared constants, `_Run`, `frontier_mask`) changes no behaviour.
4. **URDF/transform helpers** (F7). `h12_slam.robot_geometry.fixed_transform` and `g1_navigation.frames.link_pose` agree to 1e-16 on the H1-2 URDF (measured); quaternion/matrix helpers exist in four more places. About 120 lines, effort S-M, low-medium risk (touches H1-2 real-robot launch).
5. **YAML overlays** (F10, F11). `g1_slam_toolbox.yaml` repeats 41 of 54 values of `slam_toolbox_h1.yaml` (and most of its mapper section is inert with scan matching off); `g1_fastlio_sim.yaml` vs `g1_fastlio.yaml` differ in 4 values. About 60-100 lines, effort S, low risk.
6. **Adopt G1 patterns on the H1-2 (F19, F20, F15, F6, F8; not dedup, but removes H1-2 duplication/latent faults)**: `g1_description`'s single generated joint table and `urdf_path()` (H1-2 keeps three byte-identical limit tables in two submodules, F20), `config_pairing` refusal (F15), the dynamic `body -> pelvis` (F6), the pointcloud_to_laserscan-without-TF workaround (F8). Crosses submodule boundaries and safety code: announce first.
7. **Leave alone**: Nav2 tuning and BT, `level_odom`/`level_frame` internals, scan bands, the two bringup shapes, description packaging (section 4).

Not verified at runtime: nothing here was run (no containers, no ROS on the host). `nav2_bringup/navigation_launch.py` is not on the host, so whether G1's hand-rolled Nav2 node list could use it is unchecked. `grasp.py`/`pick_place.py` were skimmed for their interfaces only.

---

## 1. Inventory

Submodule status (`git submodule status`): `h12_ros2_model` is a submodule (branch `unified_sim`, 0b46cfc) and `custom_ros_messages` is a submodule (branch `g1_interfaces`). `h12_slam`, `h12_skills`, `h12_bringup`, `g1_navigation`, `g1_bringup` and `g1_description` are ordinary directories of this repo. `g1_description/leprechaun_g1` is a symlink into the `third_party/leprechaun_g1` submodule. (`core_ws/src/h1_bringup` and `locomotion_game_deploy` contain only ignored `__pycache__`/`.pytest_cache`; harmless leftovers.)

### 1.1 `h12_slam` (`core_ws/src/h12_slam`, in-repo)

Console scripts (`setup.py`): `self_filter`, `level_odom`, `record_slam_bag`.

| Node / file | Subscribes | Publishes | TF | Params / config |
|---|---|---|---|---|
| `self_filter` (`h12_slam/self_filter.py`) | `cloud_in` PointCloud2 (sensor QoS), `/robot_description` String (transient local) | `cloud_out` PointCloud2 | reads TF to every collision link | `padding` 0.05, `exclude_links`, `max_tf_age` 0.2 |
| `level_odom` (`level_odom.py`, 239 lines) | `imu` Imu, `cloud` PointCloud2, `odometry` nav_msgs/Odometry | none | static `odom -> camera_init` (once, from IMU gravity + lidar floor plane); dynamic `odom -> base_footprint` per Odometry | `odom_frame`, `world_frame`, `body_frame`, `base_frame`, `footprint_frame`, `samples`, `max_accel_spread`, `floor_scans`, `min_floor_depth`, `min_floor_points` |
| `record_slam_bag` | none | none | `execvp ros2 bag record -a -x <pattern>` | `--name`, `--dir` |
| `robot_geometry.py` | library: URDF collision boxes, `fixed_transform`, quaternion helpers | | | |

Launch (`launch/h12_navigation.launch.py`, 225 lines): FAST-LIO (extrinsic and `body -> pelvis` computed from the URDF at launch), `level_odom` (remaps `imu`->`/livox/imu`, `cloud`->`/cloud_body_filtered`, `odometry`->`/Odometry`), `self_filter` (`/cloud_registered_body` -> `/cloud_body_filtered`), `octomap_server` (-> `/projected_map`), `pointcloud_to_laserscan` (`/cloud_body_filtered` -> `/converted_scan`, `target_frame base_footprint`), `slam_toolbox`, then `nav2_bringup/navigation_launch.py` with `RewrittenYaml` and `behavior_trees/navigate_to_pose_humanoid.xml`. `launch/h12_real_slam.launch.py`: real drivers + state + navigation, no actuation.

Configs: `mid360.yaml` (FAST-LIO, shared by real and sims; the test in `h12_bringup/test/test_real_lidar_config.py` asserts no `extrinsic_T/R` in it), `slam_toolbox_h1.yaml`, `nav2_config.yaml` (MPPI/Omni, Smac 2D), `nav2_config_mac.yaml` (mac), `octomap.yaml`, `MID360_config.json` (driver, real robot). `fastlio_nav2_config.yaml` and `test_nav2_config.yaml` are referenced by no launch file or code (grep). `rviz/slam.rviz` is neither installed (`setup.py` has no rviz data_files) nor referenced.

Tests: `test_level_odom.py` (40), `test_mid360_frames.py` (39), `test_robot_geometry.py` (70).

### 1.2 `g1_navigation` (`core_ws/src/g1/g1_navigation`, in-repo)

Console scripts (`setup.py`): `livox_to_pointcloud`, `level_frame`, `body_to_pelvis`, `frontier_explore`.

| Node | Subscribes | Publishes | TF | Params |
|---|---|---|---|---|
| `livox_to_pointcloud` (231 lines) | `livox/lidar` livox_ros_driver2/CustomMsg (sensor QoS), `livox/imu` Imu (best-effort 50, only with deskew) | `lidar_points` PointCloud2 (depth 5) | reads `target_frame <- livox_frame` at each scan stamp | `deskew`, `target_frame`, `max_tf_wait` 0.5 |
| `level_frame` (100 lines) | none (TF `odom -> pelvis`) | none | `pelvis -> pelvis_level`, `pelvis_level -> base_footprint` per new odom->pelvis stamp | `fixed_frame`, `base_frame`, `level_frame`, `floor_frame`, `rate` 100 |
| `body_to_pelvis` (83 lines) | none (TF `pelvis <- livox_frame`) | none | `body -> pelvis` at the stamp of each lidar TF | `fastlio_config` (required), `body_frame`, `base_frame`, `lidar_frame`, `rate` 200 |
| `frontier_explore` (537 lines) | `/map`, `/global_costmap/costmap` OccupancyGrid (transient local) | none | reads `map -> pelvis` | `map_frame`, `base_frame`, `footprint`, `default_timeout` 300, `map_topic`, `costmap_topic` |

`frontier_explore` serves action `/skill/frontier_explore` (`custom_ros_messages/action/SkillFrontierExplore`) and is a client of `/navigate_to_pose` (`nav2_msgs/NavigateToPose`). `frames.py` (147 lines) is a library (URDF chain, FAST_LIO extrinsic, static-transform argument formatting).

Launch (`launch/g1_navigation.launch.py`, 256 lines): args `odom_source` (`fastlio`|`sim_gt`), `sim`, `use_sim_time`, `fastlio_config`, `pelvis_height`, `nav2_params`, `slam_params`, `frontier_explore`. Nodes: static `mid360_link -> livox_frame`, `livox_to_pointcloud` (`target_frame pelvis_level`, deskew only in sim), `level_frame`, `pointcloud_to_laserscan` (no TF, `use_sim_time False`, `/lidar_points` -> `/scan`), `slam_toolbox`, [fastlio mode: `fast_lio`, static `odom -> camera_init`, `body_to_pelvis`], Nav2 as 5 explicit nodes (`controller_server`, `planner_server`, `behavior_server`, `bt_navigator`, `velocity_smoother`) + `lifecycle_manager_navigation`, [`frontier_explore`].

Configs: `g1_fastlio.yaml` (real, `TODO(lidar)` x4), `g1_fastlio_sim.yaml`, `g1_slam_toolbox.yaml` (106), `g1_nav2.yaml` (248, RPP + Smac 2D). Tests (2297 lines): `test_frontier_explore.py` (552), `test_nodes.py`, `test_frames.py`, `test_fastlio.py`, `test_launch.py`, `test_nav2_smoke.py`, `test_velocity_smoother.py`, fakes (`fake_g1.py`, `stack.py`); closed-loop `tests/g1/test_sim_{bringup,nav,explore}.sh`.

### 1.3 `h12_skills` (`core_ws/src/h12_skills`, in-repo, 1 console script `skills = h12_skills.node:main`)

One node `h12_skills` (`SkillsNode` = `SkillsBase` + 11 mixins; `MultiThreadedExecutor(8)`). The action table is `SKILL_ACTIONS` (`base.py:218`). Package metadata is still `TODO` (`setup.py`, `package.xml`: maintainer root, license TODO).

| Action server | Action type | Goal fields (besides `timeout`) | Status |
|---|---|---|---|
| `/skill/grasp` | `SkillGrasp` | `target_object`, `arm`, `top_down`, `min_downward_pitch_deg`, `visual_servo` | implemented (`skills/grasp.py`, 1752 lines) |
| `/skill/pick_place` | `SkillPickPlace` | `target_object`, `place_target`, `arm`, `top_down`, `visual_servo` | implemented (320 lines; calls the node's own `/skill/grasp`) |
| `/skill/frontier_explore` | `SkillFrontierExplore` | `min_frontier_cells`, `blacklist_radius`, `min_goal_distance`, `goal_timeout` | implemented (427 lines) |
| `/skill/open_door` | `SkillOpenDoor` | `target_object`, `arm`, `pull_distance` | stub (`NotImplementedError`) |
| `/skill/close_door` | `SkillCloseDoor` | `target_object`, `arm` | stub |
| `/skill/open_lid` | `SkillOpenLid` | `target_object`, `arm`, `open_angle` | stub |
| `/skill/close_lid` | `SkillCloseLid` | `target_object`, `arm` | stub |
| `/skill/navigate` | `SkillNavigate` | `target_object` | stub |
| `/skill/press` | `SkillPress` | `target_object`, `arm` | stub |
| `/skill/slide_rack` | `SkillSlideRack` | `target_object`, `direction`, `arm` | stub |
| `/skill/turn_lever` | `SkillTurnLever` | `target_object`, `direction`, `angle`, `arm` | stub (shared `skills/turn.py`) |
| `/skill/twist_knob` | `SkillTwistKnob` | `target_object`, `direction`, `angle`, `arm` | stub (shared `skills/turn.py`) |

Every result is `success` + `message` (`SkillFrontierExplore` adds `goals_reached`); feedback is `phase` + `progress` (`SkillFrontierExplore` adds `frontiers_remaining`). `SkillRetrieveMemory` is defined but served by nobody.

Clients/subscribers/publishers of `SkillsBase` (from `base.py`):
- Services: `/left|right/gripper/{set_position (SetGripperPosition), open, close, reset_overload (Trigger), set_force (SetGripperForce)}`, `gemini_query` (GeminiQuery), `sam_segment` (SamSegment), `graspgen` (GraspGen).
- Actions: `/frame_task` (FrameTask), `/named_config` (NamedConfig), `/skill/grasp` (SkillGrasp, in-process), `/navigate_to_pose` (NavigateToPose, frontier only).
- Topics in: `/left|right/gripper/state` (GripperState), `/safety/heartbeat` (Bool), `/realsense/head/color/image_raw/compressed`, `.../aligned_depth_to_color/image_raw/compressedDepth`, `.../color/camera_info`, head and per-hand `DetectionBundle` (`.../compressed/detections`), per-hand compressed colour, `/map` (frontier only, transient local).
- Topics out: `/skill/grasp/servo_error|servo_move|servo_move_residual` (Vector3Stamped), `/skill/grasp/servo_image` (Image), `graspgen_markers` (MarkerArray), `/frontier_explore/frontiers` (MarkerArray), TF broadcaster (grasp target frames).
- Params: `record_runs`, `grasp_method`, `contact_offset`, `servo_depth`, `enable_logging`, `enable_visualization`, `clear_logs`.
- Files: writes `<package>/logs/...` (git-ignored).
- Helper modules: `grasp_synthesis/` (7 methods), `perception_utils.py`, `model_logging.py`, `run_telemetry.py`. Test: `test/test_grasp_synthesis.py`.

Doc drift: `DEFAULT_SKILL_TIMEOUT = float('inf')` (`base.py:147`), so for the H1-2 skills a goal timeout of 0 means no deadline. CLAUDE.md section 7 says "0 -> 300 s default"; that holds only for the G1 explorer (`default_timeout` 300).

### 1.4 `h12_bringup` and `g1_bringup`

`h12_bringup` (in-repo; `livox_imu_upright` console script; `config/`: `lowerbody_{real,sim}.yaml`, `mjpc_{real,sim}.yaml`; `rviz/main.rviz` 1544 lines; `scripts/slider_debugger.py`; `common_cmds.md`):

| Launch file | Role |
|---|---|
| `h12_sim_bringup.launch.py` (129) | x86 sim: state, control (sim configs, delays 0), lowerbody (`start_position_verified:=true`), models, [nav], static `camera_link -> camera_color_optical_frame`, rviz |
| `h12_sim_bringup_mac.launch.py` (193) | standalone mac copy: state nodes, `frame_task_server`, `safety_node` inline; env-gated lowerbody, SLAM (own `pointcloud_to_laserscan` params), Nav2 |
| `h12_real_robot_bringup.launch.py` (50) | real onboard PC: real_drivers, navigation, state, control (`use_estop`), lowerbody (`start_position_verified` default false) |
| `h12_real_desktop_bringup.launch.py` (91) | companion desktop: hand cameras, models (`use_yolo`), rviz |
| `h12_safety_grippers.launch.py` (52) | estop + safety + grippers only |
| `lmg_sim_bringup.launch.py` (91), `lmg_real_bringup.launch.py` (87) | loco-manipulation game; `lmg_sim` refuses unset/0 `ROS_DOMAIN_ID` |
| blocks: `h12_state` (51), `h12_control` (89), `h12_lowerbody` (78), `h12_models` (110), `h12_real_drivers` (93), `h12_grippers` (69), `h12_locomanipulationgame` (65) | |

`g1_bringup` (in-repo, no console scripts, no config dir; `rviz/g1.rviz` 253 lines):

| Launch file | Role |
|---|---|
| `g1_core.launch.py` (302) | shared core, one `OpaqueFunction`: robot_state_publisher, `joint_state_publisher`, `safety_node`, `frame_task_server` after `controller_delay`, [`dex3_hands`], [rviz], [`locomotion_node` after `controller_delay + locomotion_delay`], [`g1_navigation` include]; raises `RuntimeError` on a bad combination (`startup_refusal`, `nav_refusal`, `locomotion_refusal`, `pairing_refusal` from `g1_control`) |
| `g1_sim_bringup.launch.py` (78), `g1_bringup.launch.py` (113) | thin sim/real wrappers; each re-declares ~8 arguments and forwards them to `g1_core` |

### 1.5 `h12_ros2_model` (submodule) and `g1_description` (in-repo)

| | `h12_ros2_model` | `g1_description` |
|---|---|---|
| Python modules | empty `__init__.py` | `joints.py` (generated, 29 body + 2x7 hand joints with limits), `layout.py` (DDS slots, arm_sdk weight slot 29, mode_pr/ab), `paths.py` (`urdf_path()`, `assets_dir()`), `tools/generate_joints.py` |
| Launch | `robot_description_launch.py` (`model:=magpie\|handless`): robot_state_publisher + joint_state_publisher_gui + rviz | `display.launch.py` (`hands`, `gui`, `rviz`): same three nodes |
| Install | `setup.py` stages `h1_2*.urdf/.srdf` from `CL_Assets/ros_assets` into `_install_staging/h1_2/`, rewrites `../meshes/h1_2/` -> `meshes/` in the Pinocchio variants, leaves `*_ros.urdf` unchanged | `setup.py` reads `leprechaun_g1` URDFs, installs them unchanged and writes `*_ros.urdf` copies with `package://g1_description/assets/meshes/` |
| rviz | `default.rviz` (469) | `g1.rviz` (85) |
| Tests | ament copyright/flake8/pep257 only | `test_joints.py` (generated table must match the URDF) |

---

## 2. Overlap findings

Similarity figures: "line ratio" = `difflib.SequenceMatcher(autojunk=False)` over non-blank, comment-stripped lines; "token ratio" = the same over Python tokens with comments/docstrings dropped. Script: `scratchpad/dedup/work/sim.py`.

### F1 Frontier explorers: the same action, B for the policy, C for the rest

- Locations: `core_ws/src/h12_skills/h12_skills/skills/frontier_explore.py:1-427`; `core_ws/src/g1/g1_navigation/g1_navigation/frontier_explore.py:1-537`.
- Classification: **B** for the constants, `_Run` and the frontier mask; **C** for goal choice and goal supervision. Whole-file line ratio 0.034 (the structure is rewritten: mixin on `self` vs module functions + node).
- Evidence of ancestry: G1 docstring: "That is the H1-2 skill's policy, with its defaults"; constants identical (`DEFAULT_MIN_CELLS = 6`, blacklist radius 0.6, min goal distance 0.7, goal timeout 120.0; H1-2 lines 46-49, G1 lines 81-84); same cell thresholds (free `< 25`, occupied `>= 65`); same 8-connected clustering; same "nearest by Euclid, not blacklisted, farther than `min_goal_distance`" rule (H1-2 inline at 303-327; G1 `choose` at 243-250).
  ```python
  # H1-2 _frontiers():                         # G1 frontier_mask():
  free = (g >= 0) & (g < 25)                    free = (g >= 0) & (g < FREE_BELOW)
  unknown_adj = binary_dilation(unknown, 1)     near_unknown = binary_dilation(g < 0, border_value=1)
  wall_adj = binary_dilation(occupied, 1)       near_wall = binary_dilation(g >= OCCUPIED_FROM)
  frontier = free & unknown_adj & ~wall_adj     return free & near_unknown & ~near_wall
  ```
- What differs:
  - **Goal point** (algorithmic, deliberate on both sides): H1-2 backs the goal off along the robot-to-frontier ray into known-free space, at least 0.35/0.25/0.18/0.12 m from unknown and 0.20 m from walls, so the base cannot walk off the RoboCasa void (`_safe_goal`, lines 213-258, `_STANDOFFS` line 73). G1 sends the frontier-cluster cell nearest the centroid where Nav2's footprint covers no lethal costmap cell (`find_frontiers`, 200-227). The G1 version has no standoff from unknown space; whether that matters at a RoboCasa doorway is **not verified** (the G1 README says only that the kitchen test passes).
  - **Reachability filter**: G1 keeps only frontier cells 8-connected through walkable cells (free in `/map`, below inscribed cost in the global costmap) to the robot (`walkable_mask`, `reachable_mask`, 143-164); the H1-2 has none and relies on Nav2 failing and a blacklist.
  - **Goal supervision**: G1 watches each new `/map` and costmap while Nav2 runs and cancels a goal whose frontier was "seen" (not blacklisted) or is now blocked (`outdated`, 230-240, `_navigate`, 385-416). H1-2 waits for Nav2's result (`_send_action`, `result_timeout=min(goal_timeout, run.remaining())`) and then decides using "base moved" and "newly known cells" heuristics (`_STUCK_MOVE`, `_PROGRESS_CELLS`, lines 96-103, 393-427).
  - **Stuck handling**: H1-2 defers frontiers with no safe standoff `>= _MIN_SAFE_ADVANCE` and aborts after `_MAX_STUCK_ROUNDS` 15; G1 aborts only if Nav2 refuses a goal, or the robot is off walkable space.
  - **Result**: `goals_reached` counts Nav2 successes with motion (H1-2) vs Nav2 successes + goals seen on the way (G1).
  - **Features only one side has**: H1-2 publishes RViz markers on `/frontier_explore/frontiers`; G1 has none. G1 rejects a second concurrent goal (`busy`, `_on_goal`, 347-359) and goals with negative or non-finite fields; the H1-2 server (default `ActionServer` accept-all, `node.py:78-86`) does neither, so two overlapping goals would both drive Nav2.
  - **Clock**: H1-2 deadlines and waits use `time.monotonic()` (`base.py:251,615`), G1 uses the node clock (sim time in the sim). At sim RTF < 1 the H1-2's goal timeout elapses in fewer simulated seconds; `docs/NAVIGATION_DEMO.md:114` already mentions 120 s goal timeouts being exceeded.
  - **Default overall timeout**: H1-2 infinite, G1 300 s.
  - **Doc inaccuracy in the H1-2 file**: line 152 says "8-adjacent to unknown", but `scipy.ndimage.binary_dilation` with the default structure is 4-connected (checked on a 5x5 grid). G1 says 4-adjacent and also treats off-map as unknown (`border_value=1`); the H1-2 does not.
- Robot-specific vs accidental: the footprint (`G1_FOOTPRINT` `[[0.44,0.24],...]`, parameterised via `footprint`), the abyss standoff (RoboCasa-driven, H1-2), and costmap sources are legitimate. The guards (busy rejection, input validation, sim-clock deadlines, "seen" cancellation) are **accidental divergence**: the G1 got fixes the H1-2 lacks.
- Consolidation proposal: a robot-agnostic package (name open, e.g. `golem_exploration`) holding G1's pure functions (`grid_of`, `sample`, `frontier_mask`, `walkable_mask`, `reachable_mask`, `footprint_fits`, `find_frontiers`, `outdated`, `choose`) and the node, with parameters `footprint`, `base_frame`, `costmap_topic` (all exist) plus a new `unknown_standoff` (0 = off, H1-2 sets 0.12-0.35) and an optional marker publisher. Each robot's navigation launch starts the node (`h12_navigation.launch.py` gets a `frontier_explore` argument as `g1_navigation.launch.py` has). Delete `FrontierExploreSkill`, its `SKILL_ACTIONS` entry and the `scipy`/`nav2_msgs` deps that only it uses in `h12_skills/package.xml`. Two side benefits: exploration stops depending on `use_skills:=true` (which starts the GPU `graspgen_server`); and the mac stack gets a server (D10). Effort **M-L**. Staged alternative **S**: only move the constants, `_Run` and `frontier_mask` into a shared module, leaving both algorithms.
- Risk: H1-2 real robot: indirect only (navigation runs there, the explorer is an operator-invoked skill); behaviour change for H1-2 users (new goal selection, new deadlines); crosses `h12_skills` ownership (CLAUDE.md section 5) and touches the shared `h12_navigation` launch; `SkillFrontierExplore.action` (submodule `custom_ros_messages`) needs no change. Tests: G1's `test_frontier_explore.py` (552 lines) and `tests/g1/test_sim_explore.sh` cover the node; **nothing tests the H1-2 explorer**, so an H1-2 sim exploration test (RoboCasa, void-edge scene) must be added first.
- Both nodes must never run on one domain (CLAUDE.md section 8); one shared node makes that rule moot.

### F2 `_Run` per-goal helper: B (copy with edits)

- Locations: `core_ws/src/h12_skills/h12_skills/base.py:234-305`; `core_ws/src/g1/g1_navigation/g1_navigation/frontier_explore.py:253-305` (docstring: "the H1-2 skills' _Run").
- Evidence (token ratio, `_node/_gh` renamed): `remaining` 0.86, `_finish_canceled`/`_canceled` 0.82, `abort` 0.86, `succeed` 0.86, `phase` 0.50 (G1 folds cancel + deadline into `check()`).
- Differences: G1 uses the node clock and takes no `action_type`/`label`; H1-2 `_Run.__init__` also resets `node._arm_halt_reason` (arm-specific) and logs every phase. Deadline source differs as in F1.
- Proposal: with F1, G1's `_Run` (action-type-parametrised) lives in the shared module; `h12_skills` keeps its own `_Run` subclass for the arm-halt reset, or the base class takes a `now()` callable. Effort S. Risk: `_Run` is used by all H1-2 skills including grasp (real-robot manipulation); do not change `h12_skills._Run` behaviour. Tests: none for H1-2.

### F3 Executor-safe waiting and action sending: C

- Locations: `base.py:606-625` (`_wait_future`) and `base.py:640-663` (`_send_action`); `skills/frontier_explore.py:260-267` (`_frontier_wait`); G1 `frontier_explore.py:368-383` (`_wait`, `_sleep`) and `385-416` (`_navigate`).
- Evidence: token ratio `_wait_future` vs `_wait` 0.59, `_frontier_wait` vs `_sleep` 0.57. Same job (block a worker thread on a future with cancel and timeout, then cancel the in-flight goal on the server), independent bodies; G1 on the node clock, H1-2 on wall clock and with a `stop_event`.
- Proposal: nothing separate; it moves with F1. Do not generalise `SkillsBase._wait_future` (used by every skill).

### F4 MID-360 IMU frame: the H1-2 solved it; the G1 has it as an open TODO

- Locations: `core_ws/src/h12_bringup/h12_bringup/livox_imu_upright.py:1-119`, launched in `h12_bringup/launch/h12_real_drivers.launch.py` (remaps the driver's `/livox/imu` to `/livox/imu_raw`, node outputs `/livox/imu`), guarded by `h12_bringup/test/test_real_lidar_config.py`. G1 side: `g1_navigation/config/g1_fastlio.yaml` (`TODO(lidar)` at `extrinsic_R`), `g1_navigation/launch/g1_navigation.launch.py` ("TODO(lidar): the deskew is the sim's only. On the robot, MID360_config.json's roll-180 extrinsic turns the points but not the IMU"), `g1_navigation/README.md:589-608` ("The real lidar").
- Classification: not a duplicate; a **gap with an existing robot-agnostic solution**. `livox_imu_upright` reads the driver's own `MID360_config.json` and applies the driver's rotation convention to the gyro and accelerometer, plus a gravity sanity check. Nothing in it is H1-2 specific (only the default log text mentions the torso pitch).
- The G1 needs exactly this for two consumers: FAST_LIO (`extrinsic_R` would otherwise need diag(1,-1,-1) and a roll-pi `LIDAR_FRAME`, per its README) and `livox_to_pointcloud`'s gyro deskew (which "would have the wrong sign" for y and z). Also reusable: the driver node block in `h12_real_drivers.launch.py` (G1 needs its own host IP and a G1 `MID360_config.json`; none exists, `README.md:589`).
- Proposal: move `livox_imu_upright` (+ its test) into a neutral package that both bringups depend on; the G1 real bringup (when the lidar is connected) launches it exactly as `h12_real_drivers` does. Effort **S**. Risk: moving it touches the real-robot H1-2 bringup (`h12_real_drivers`, `lmg_real_bringup` via `use_lidar`) and `h12_bringup` packaging; keep an entry point of the same name and the same test. G1 depending on `h12_bringup` instead would be wrong (couples G1 builds to the H1-2 launch package).

### F5 Level base frame: `level_odom` vs `level_frame`: C

- Locations: `core_ws/src/h12_slam/h12_slam/level_odom.py:1-239`; `core_ws/src/g1/g1_navigation/g1_navigation/level_frame.py:1-100` plus `frames.fastlio_transforms` (`frames.py:130-139`) for the static `odom -> camera_init`.
- Same job: give slam_toolbox/Nav2 a `base_footprint` that is level and on the floor in a gravity-aligned `odom`, with the cloud's height band in that frame. Line ratio 0.129; no shared code.
- Differences:
  - H1-2 **measures** the level: gravity from the IMU and the floor plane from self-filtered lidar scans at startup (robot must stand still), publishes static `odom -> camera_init` and per-odometry `odom -> base_footprint` (yaw only, pelvis xy).
  - G1 **assumes** it: `odom -> camera_init` computed from the URDF with a torso-level stance and `pelvis_height` 0.744 (README: HOMIE's stance is within 0.3 deg; "Unitree's own controller stands differently; not measured"), then `level_frame` publishes `pelvis -> pelvis_level` and `pelvis_level -> base_footprint` from any `odom -> pelvis` TF, so it also works in `sim_gt` mode where there is no FAST_LIO.
  - Tree topology differs: H1-2 `base_footprint` is a child of `odom`; G1's is a child of `pelvis_level` (child of `pelvis`).
- Robot-specific vs accidental: both are intentional. The G1's startup assumption is a known weakness on a real G1 whose stance is not HOMIE's; the H1-2's measured approach would remove it (needs `/cloud_body_filtered`-style input and the IMU in the point frame, i.e. F4).
- Proposal: no merge. Optional later: `level_frame` could replace the `_on_odometry` half of `level_odom` (lines 199-232) because the H1-2 TF tree already has `odom -> pelvis` (`odom -> camera_init -> body -> pelvis`); and a G1 real path could adopt the `find_floor_height` calibration. Effort M, **risk high for the H1-2**: this is the real-robot SLAM frame chain (memory notes record SLAM drift work there); only with bag replay. Leave.

### F6 `body -> pelvis`: static in H1-2 launch, dynamic node in G1 (accidental divergence)

- Locations: `h12_slam/launch/h12_navigation.launch.py:36-47` (static_transform_publisher built from `fixed_transform(urdf, 'livox_imu_link', 'pelvis')` with joints at zero); `g1_navigation/g1_navigation/body_to_pelvis.py:1-83` (+ `frames.body_to_pelvis`, `frames.py:125-127`).
- In the H1-2 URDF the lidar and IMU hang under `torso_link` via the revolute `torso_joint` (verified: `livox_imu_link <- livox_link <- torso_link <- pelvis (torso_joint)`), so a static `body -> pelvis` double-counts any torso angle. The G1 README documents the identical fault on the G1 waist (1.1-3 deg scan tilt with HOMIE's 1.5 deg waist) and solved it with a node reading TF at the measured waist angles.
- H1-2 impact today: the controller holds `torso_target: 0.0` (`h12_ros2_controller/config/sim_safety_split.yaml:18`), so the error is latent, not active.
- Opposite data flow worth noting: H1-2 writes FAST-LIO's `extrinsic_T/R` into the node parameters from the URDF at launch; G1 keeps them in `g1_fastlio.yaml` and derives the transforms from that file. Both rely on the URDF for the same chain.
- Proposal: a shared `body_to_pelvis` node parameterised by `base_frame`, `lidar_frame` and the extrinsic source (YAML or URDF) would serve both; effort S-M. Risk: H1-2 real SLAM frame chain (same caution as F5); behaviour change is nil while the torso is at 0. Priority low.

### F7 URDF kinematics and transform helpers: C, numerically interchangeable

- Locations: `h12_slam/h12_slam/robot_geometry.py:16-60,149-175` (`rpy_to_matrix`, `quat_to_matrix`, `matrix_to_quat`, `transform_to_rt`, `fixed_transform`); `g1_navigation/g1_navigation/frames.py:39-105` (`transform`, `invert`, `load_joints`, `link_pose`); scipy-based `to_matrix` in `body_to_pelvis.py:26-29` and `livox_to_pointcloud.py:138-144`. Other copies of the quaternion helpers: `h12_skills/perception_utils.py:58-110` (`transform_to_matrix`, `quat_to_rot`, `mat_to_quat`), `cl_realsense/.../point_cloud_accumulator.py:97`.
- Evidence (measured on `CL_Assets/ros_assets/h1_2_magpie_ros.urdf`, all joints at zero): `link_pose('pelvis', 'livox_imu_link')` vs `fixed_transform(...)`: max rotation difference 1.1e-16, translation 0.0; `pelvis -> left_wrist_yaw_link`: 2.2e-16 / 5.6e-17. Line ratio of the two files 0.022 (independent code).
- Differences: `link_pose` supports joint positions (needed for the waist) but only a tip below its root; `fixed_transform` handles any two links in one tree but only at zero joints. `robot_geometry` also holds the collision-box code used by `self_filter` (H1-2 only). H1-2 avoids scipy (no dependency in `h12_slam/package.xml`); G1 uses it.
- Within G1: `to_matrix` exists twice (same output, different construction) and the `_stamp`-style helpers are repeated; within H1-2/skills the quaternion helpers exist in three packages.
- Proposal: one pure-Python geometry module in a neutral package: `link_pose(urdf, root, tip, positions)` extended to ancestor/descendant pairs, `to_matrix(Transform)`, quaternion conversions. Both nav packages import it. Effort S-M, saves about 120 lines. Risk: low for G1; for H1-2 it changes the code that computes the FAST-LIO extrinsic on the real robot, so keep the equivalence check above as a unit test (`test_robot_geometry.py` exists) before swapping.

### F8 Scan pipeline: D for the architecture, with two cross-robot gaps

- H1-2: FAST-LIO's deskewed `/cloud_registered_body` -> `self_filter` (URDF collision boxes) -> `/cloud_body_filtered` -> `pointcloud_to_laserscan` (`target_frame base_footprint`, `min_height 0.25`, `max_height 1.7`, range 0.3-3.0) and `octomap_server`.
- G1: raw `/livox/lidar` CustomMsg -> `livox_to_pointcloud` (gyro deskew, placed in `pelvis_level` through TF) -> `pointcloud_to_laserscan` (no TF, `min_height -0.59`, `max_height 0.45`, `range_min 0.50`, range 8.0) -> `/scan`. It works without FAST-LIO (`sim_gt`).
- These are different designs for different sensor heights and odometry paths: keep separate (section 4). Shared fragments: `angle_min/max` and `angle_increment 0.0087` are equal in both parameter dicts, and a third copy of the H1-2 parameters sits in the mac launch (`h12_sim_bringup_mac.launch.py`, same file, "p2l params match h12_navigation.launch.py").
- Gap 1: `self_filter` is robot-agnostic (it needs only `/robot_description`, TF and collision geometry) and the G1 documents it cannot check its arm/hand exclusion in sim ("the real robot will", `g1_navigation.launch.py` SCAN comment). A real G1 may want it instead of the 0.5 m `range_min`; the G1 URDF would need collision geometry (not checked).
- Gap 2 (possible latent H1-2 fault, **unverified on the H1-2**): the G1 measured `pointcloud_to_laserscan` with its own TF message filter stalling for good in 7 of 27 runs (6 of 13 on sim time, 1 of 14 on wall clock) and removed the TF dependency by publishing the cloud already in the target frame (`livox_to_pointcloud` `target_frame`). The H1-2 launch uses the same node with `target_frame base_footprint`, `transform_tolerance 0.05`, and `use_sim_time` from the launch. No evidence that the H1-2 has hit it.

### F9 pointcloud_to_laserscan parameter blocks: D

Equal: `angle_min/max`, `angle_increment`, `use_inf True`. Different by design: band, ranges, `scan_time` (0.0333 vs 0.1), `queue_size`. Keep.

### F10 slam_toolbox configs: B (copy with edits)

- Locations: `h12_slam/config/slam_toolbox_h1.yaml` (83 lines); `g1_navigation/config/g1_slam_toolbox.yaml` (106 lines, "Started from the H1's slam_toolbox_h1.yaml").
- Evidence: 54 vs 52 keys (loaded with PyYAML): **41 identical values, 11 differ, 2 H1-2-only keys** (`occupancy_threshold`, `min_pass_through`). Line ratio 0.764 (comments stripped).
- Differing values (all intentional, tuned on each robot's sim/robot): `use_scan_matching` true/false, `do_loop_closing` true/false, `map_update_interval` 5.0/1.0, `minimum_travel_distance` and `_heading` 0.5/0.1, `minimum_time_interval` 0.5/0.2, `min/max_laser_range` 0.0-20.0 / 0.5-8.0, `transform_timeout` 0.2/0.3, `transform_publish_period` 0.02/0.05, `enable_interactive_mode` true/false.
- Observation: with `use_scan_matching: false` the G1 file says most of its scan-matcher, correlation and loop-closure parameters "do nothing", so about 25 of its 52 keys are inert copies.
- Proposal: one stock base file plus a small per-robot overlay passed as a second parameter file (`parameters=[base, overlay, {...}]`), or just delete the inert G1 keys. A shared base needs a home both robots can read (the G1 container already builds `h12_slam`, but a G1 -> h12_slam dependency is a layering choice for the owner). Effort S; saves about 60 lines. Risk: low; slam_toolbox merges parameter files in order. Tests: `g1_navigation/test/test_launch.py`/`test_nodes.py` load the G1 file; add an assertion that the merged result equals today's.

### F11 FAST-LIO configs: B

- Locations: `h12_slam/config/mid360.yaml` (64 lines); `g1_navigation/config/g1_fastlio.yaml` (72) and `g1_fastlio_sim.yaml` (63).
- Evidence (flattened PyYAML): H12 vs G1-real share 32 keys, 7 of them with different values (`filter_size_surf/map` 0.4/0.5, `blind` 0.5/0.7, `scan_line` 6/4, `extrinsic_est_en` true/false, `map_en` true/false, `map_file_path`), and `extrinsic_T/R` exist only in the G1 files. G1 sim vs G1 real share 34 keys, 4 with different values (`point_filter_num` 1/3, `filter_size_surf/map` 0.1/0.5, `scan_line` 6/4; line ratio 0.907). H12 vs G1 sim line ratio 0.634.
- The two robots disagree on principle: H1-2 forbids `extrinsic_*` in the YAML (the test asserts it; the launch derives them from the URDF), G1 requires them in the YAML (its transforms derive from the file).
- Proposal: G1 sim/real can be one file plus a 4-value sim overlay (S). A shared base across robots is possible only after settling where the extrinsic lives; not worth it for ~30 equal lines. Risk low-medium; the H1-2 file is exercised on hardware.

### F12 Nav2 configs and launch: D, with small B fragments

- Locations: `h12_slam/config/nav2_config.yaml` (259), `behavior_trees/navigate_to_pose_humanoid.xml`; `g1_navigation/config/g1_nav2.yaml` (248). Line ratio 0.211 (yaml), 0.112 (launch files).
- Identical fragments: Smac 2D `GridBased` block (tolerance differs 0.25/0.5, `allow_unknown true`, `max_iterations 1000000`, `max_planning_time 2.0`, `cost_travel_multiplier 2.0`, `smoother` `max_iterations 1000, w_smooth 0.3, w_data 0.2, tolerance 1.0e-10`), `SimpleGoalChecker` `xy_goal_tolerance 0.25`/`yaw_goal_tolerance 0.25`/`stateful`, velocity_smoother `OPEN_LOOP`/`scale_velocities false`/`deadband 0`, `controller_frequency 20.0`.
- Different by design: MPPI + Omni model, custom BT without Spin, `/projected_map` static layer, vy +-0.2 (H1-2/ALMI) vs Regulated Pure Pursuit, default BT, vy 0, `/scan` obstacle layers (G1/HOMIE). Footprints `[0.20,+-0.275]` vs `[0.44/-0.20, +-0.24]`.
- Launch: H1-2 includes `nav2_bringup/navigation_launch.py` with `RewrittenYaml`; G1 starts five Nav2 nodes plus a lifecycle manager and routes the cmd_vel remaps itself (`NAV2_NODES`, lines 65-72). Whether the stock launch could be reused with the G1's remappings is **not verified** (the file is not on the host). The G1 list omits `smoother_server` and `waypoint_follower`.
- Stale reference: `g1_nav2.yaml:16` and `g1_navigation/README.md:317` point at `g1_bringup/config/nav2_config.yaml`, which does not exist in the repo (it was the pre-port G1 file).
- Proposal: none for the files. Keep.

### F13 Navigation launch structure: D, with one small B

`h12_navigation.launch.py` (225) vs `g1_navigation.launch.py` (256) share only generic Node boilerplate (line ratio 0.112). `g1_navigation.launch.py:117-124` (`_static_tf`, `frames.static_transform_arguments`) and `h12_navigation.launch.py:37-47` build `static_transform_publisher` arguments from a rotation + translation; trivial (B, ~10 lines). Keep.

### F14 Bringup structure: D

- H1-2: sim and real are separate top-level files that each include shared building blocks (`h12_state`, `h12_control`, `h12_lowerbody`, `h12_models`, nav). G1: one `g1_core` (single `OpaqueFunction`) with `sim` and `use_sim_time` arguments, wrapped by two thin top-levels. Line ratios `g1_core` vs `h12_sim_bringup` 0.107; `g1_sim_bringup` vs `h12_sim_bringup` 0.147; `g1_bringup` vs `h12_real_robot_bringup` 0.175.
- Parallel nodes in both (robot_state_publisher, `joint_state_publisher`, `safety_node`, `frame_task_server`, rviz, staggered start with `TimerAction`) are 5-10 lines each with different packages and config names: **D/C**, not worth sharing. One mechanism differs: H1-2 reads the URDF with `Command(['cat ', path])`; G1 reads it in Python and passes a string.
- Config ownership convention differs: CLAUDE.md section 6 puts H1-2 node parameters in `h12_bringup/config/<file>.yaml`; the G1 selects `--config <name>.yaml` inside each owning package (`g1_control`, `g1_safety_layer`, `g1_locomotion`) and `g1_bringup` has no config dir. Keep (D), but note it for anyone writing shared tooling.
- `package.xml` and `setup.py` of `g1_bringup` are copies of `h12_bringup`'s (line ratios 0.737 and 0.831). `g1_bringup/package.xml` still lists `python3-opencv`, `python3-numpy`, `python3-scipy`, `std_srvs`, `custom_ros_messages`, `fast_lio`, `tf2_ros`, `pointcloud_to_laserscan`, `slam_toolbox`, `nav2_bt_navigator`, none used by its launch files (they belong to `g1_navigation`, which lists them itself). Harmless leftovers (S to trim).

### F15 Interlocks and pairing: D plus an adopt-from-G1 opportunity

| Pattern | H1-2 | G1 |
|---|---|---|
| `start_position_verified` | default false; gates the leg controller by `IfCondition`, so false silently starts everything except legs (`h12_lowerbody`, `h12_locomanipulationgame`); second gate `/lowerbody/confirm_engage` or `/marl_deploy_node/confirm_engage` | default false; only matters with `use_locomotion`, and then the launch **refuses to start** on the real robot unless true (`locomotion_refusal`, `g1_core.launch.py:102-113`); engage gate lives in the safety layer |
| Sim/real config mixups | not checked: real default `relax_safety_split.yaml` + `safety_split.yaml`; sim `sim_safety_split.yaml` for both; nothing refuses a sim config on hardware or an unpaired controller/safety config | `pairing_refusal` (`g1_control/utility/config_pairing.py`) refuses sim configs on the real robot and the reverse, requires the paired controller config; unit-tested (`g1_bringup/test/test_locomotion_launch.py`, 225 lines) |
| `ROS_DOMAIN_ID` guard in launch | only `lmg_sim_bringup` (refuses unset/0) | none in launch (the docker wrappers enforce it) |
| Refusals for unsupported combos | none | `nav_refusal` (no real nav, needs locomotion) |

The same argument name means different things on the two robots (false = "skip legs" vs "refuse"). Anyone writing shared tests/tooling (`docker_stack.sh`, `tests/stack`) must treat them separately. Proposal (adopt, not dedup): add a `pairing_refusal`-style check to `h12_control.launch.py` for the sim/real split; effort S-M; **risk for the H1-2 real robot** (a refusal could block an existing combination such as `tight_safety_split` or `balance_safety_split`), so enumerate the combinations in use first; the safety/controller packages are submodules (CLAUDE.md section 5).

### F16 H1-2 bringup duplicated inside itself: A/B

- `_include(...)` helper defined 6 times, each ~7 lines: `h12_sim_bringup.launch.py:16`, `h12_real_desktop_bringup.launch.py:18`, `lmg_sim_bringup.launch.py:29`, `lmg_real_bringup.launch.py:33`, `h12_real_robot_bringup.launch.py:20`, `h12_slam/launch/h12_real_slam.launch.py:19` (the last differs only because `h12_slam` cannot depend on `h12_bringup`, which already depends on it).
- `ASSETS_DIR` defined 7 times with 3 values (`'/home/code/CL_Assets'` x3, `os.environ.get('GOLEM_ASSETS_DIR', '/home/unitree/GOLEM/CL_Assets')` x3 at `h12_state.launch.py:13`, `lmg_real_bringup.launch.py:27`, `h12_navigation.launch.py:15`, plus `h12_skills/logs/sweep/sweep_bringup.launch.py:15`). The G1 resolves the URDF in one place (`g1_description.paths.urdf_path`). The H1-2 already installs the URDFs in `h12_ros2_model` (and `h12_ros2_controller` reads them from there, `utility/path_definition_ros.py`), so launch files could use that installed copy instead of repeating a host path; the two URDF sources (CL_Assets direct vs `h12_ros2_model` share) are a second duplication.
- `h12_sim_bringup_mac.launch.py:62-96` repeats the Node definitions of `h12_state.launch.py` and `h12_control.launch.py` (`joint_state_publisher`, `robot_state_publisher`, `frame_task_server --config sim_safety_split.yaml`, `safety_node --config sim_safety_split.yaml`) with the same arguments (B); the x86 sim bringup passes the same values to the blocks. Why the mac launch includes none of the blocks is not documented in the file beyond "standalone". The mac launch also holds a third copy of the pointcloud_to_laserscan parameters and the `camera_optical_frame_broadcaster` Node (A with `h12_sim_bringup.launch.py:85-96`).
- A ~20-line commented-out `slider_debugger` `TimerAction` block (and its explanatory comment) appears in both `h12_sim_bringup.launch.py` and `h12_real_desktop_bringup.launch.py`; the `use_sliders` argument in the sim bringup is used only inside that comment, so it is dead. CLAUDE.md section 8 forbids commented-out code.
- Proposal (all S, launch-only, low risk): one `launch_utils.py` in `h12_bringup` with `include()` and an assets-path helper (or reuse `h12_ros2_model`'s installed URDF); the mac launch includes the blocks; delete the dead block/argument. Saves about 150 lines. Risk: launch refactors have no unit tests (only `test_real_lidar_config.py`), so verify with `tests/stack` and a sim launch; mac cannot be run on x86.

### F17 G1 bringup/navigation duplicated inside itself: A/B

- `_as_bool` is defined identically in `g1_core.launch.py:94-95` and `g1_navigation.launch.py:103-104` (A, 2 lines).
- `to_matrix(Transform)` in `body_to_pelvis.py:26` and `livox_to_pointcloud.py:138` (B).
- `g1_sim_bringup.launch.py` and `g1_core.launch.py` both declare the same 8 arguments (`use_rviz`, `use_hands`, `use_sim_time`, `safety_config`, `use_locomotion`, `use_nav`, `odom_source`, `frontier_explore`) with similar descriptions, and `g1_bringup.launch.py` six of them; needed because Humble includes only see declared arguments. Keep; trivial.

### F18 RViz configs: D, plus a G1 gap

Line ratios: `g1_bringup/rviz/g1.rviz` (253) vs `g1_description/rviz/g1.rviz` (85) 0.444 (intra-G1: the description one is a subset: grid, robot model, TF); `h12_bringup/rviz/main.rviz` (1544) vs `h12_ros2_model/rviz/default.rviz` (469) 0.308; `h12_slam/rviz/slam.rviz` (792) vs `main.rviz` 0.678 (and `slam.rviz` is unused); H1-2 `main.rviz` vs G1 `g1.rviz` 0.206. The G1 bringup rviz has no map/costmap/path/scan displays, although `use_nav:=true` exists. Keep each; the viewer-vs-bringup split is the same on both sides.

### F19 Description packages: C for packaging, D for content

- Locations: `h12_ros2_model/setup.py:1-107`, `launch/robot_description_launch.py:1-66`; `g1_description/setup.py:1-84`, `launch/display.launch.py:1-63`.
- Evidence: line ratio `setup.py` 0.241, display launch 0.248 (rviz files 0.238). Same recipe in both setups (compute `HERE`, locate the assets dir, raise if missing, stage a URDF copy with rewritten mesh paths into `_install_staging/`, `data_files`), implemented with opposite rewrites: H1-2 rewrites the Pinocchio URDFs to share-relative `meshes/` and passes `*_ros.urdf` through; G1 leaves upstream URDFs untouched and writes the `_ros` copy with `package://` URIs. Launch: robot_state_publisher + `joint_state_publisher[_gui]` + rviz in both; G1 adds `hands`/`gui`/`rviz` switches, H1-2 a `model` choice.
- Path helpers: G1 has `paths.urdf_path(variant, ros)`; H1-2 has none in the description package. `h12_ros2_controller/utility/path_definition_ros.py` builds the installed paths by hand, `path_definition.py` points at `../../../CL_Assets`, and the launch files hard-code `ASSETS_DIR` (F16).
- Joint tables: G1 has **one** generated table (`joints.py`, regenerated by `tools/generate_joints.py --check`, imported by `g1_control`, `g1_safety_layer`, `g1_locomotion`, `g1_dex3`, `g1_navigation` tests). H1-2 hand-maintains the names and limits in `h12_ros2_controller/.../utility/joint_definition.py` (189), `.../joint_limits.py` (163) and `h12_safety_layer/.../core/joint_limits.py` (128), plus many YAML files.
- Proposal: keep the two description packages separate (section 4). Adopting the G1 pattern on the H1-2 side (a `paths` helper in `h12_ros2_model` or `h12_bringup`; a generated joint table) is real value but crosses two submodules (`h12_ros2_model`, and the controller and safety submodules). Effort M. Risk: the joint limits are safety-layer inputs on the real H1-2; a generated table would need a test that proves it equals today's values.

### F20 Intra-H1-2 joint-limit tables (cross-reference, other audits own the code)

`h12_ros2_controller/h12_ros2_controller/utility/joint_limits.py` and `h12_safety_layer/h12_safety_layer/core/joint_limits.py` hold the same three 27-entry tables: `JOINT_POSITION_LIMITS == URDF_POSITION_LIMITS`, `JOINT_VELOCITY_LIMITS == URDF_VELOCITY_LIMITS`, `JOINT_TORQUE_LIMITS == URDF_TORQUE_LIMITS` (compared with `ast.literal_eval`: all equal). File line ratio 0.654. Class **A** for the data. The G1 avoids this by importing `g1_description`. Listed here because it is the strongest argument for F19's "adopt the G1 pattern on the H1-2".

### F21 Skills vs the G1: no code overlap; interface touchpoints that block reuse

Only the frontier skill overlaps G1 code (F1). For anyone planning to run the grasp/pick-place skills on the G1 later, the touchpoints that differ:
- Arm IK: `/frame_task` and `/named_config` exist on both (`g1_control` serves the same action types), so `SkillsBase`'s clients work unchanged.
- `/safety/heartbeat` (Bool): published by `h12_safety_layer` only (`safety_node.py:21`); `SkillsBase._on_safety_heartbeat` uses it to halt arm moves on estop. No publisher in `g1_safety_layer` (grep).
- Grippers: Magpie services `/left|right/gripper/{set_position,open,close,set_force,reset_overload}` vs G1 Dex3 `/<side>/hand/{set_pose,open,close,release}` (`SetHandPose`, `Trigger`) with a different pose model (joint-space, 7 joints). `open`/`close` Trigger shapes are similar; `set_position`/`set_force` have no equivalent.
- `WORLD_FRAME = 'camera_init'` (FAST-LIO's frame) is correct on the H1-2 (level, upright) but upside down on the G1; it would be `odom` there. `GRASP_FRAMES` (`left_graspgenx_frame`) exist only in the H1-2 URDF.
- Head camera topics `/realsense/head/...` are named the same on both sims.

---

## 3. Divergences that may be bugs or stale (consolidated)

| # | Where | What | Evidence | Verified? |
|---|---|---|---|---|
| D1 | H1-2 explorer vs G1 | H1-2 lacks the busy-goal guard, input validation and sim-clock deadlines the G1 has; its comment says 8-adjacent where scipy gives 4-adjacent; off-map is not unknown | F1 | read from code |
| D2 | G1 explorer | no standoff from unknown space (H1-2 added it for the RoboCasa void) | F1 | behaviour not run |
| D3 | `h12_navigation.launch.py:37-47` | static `body -> pelvis` with torso at zero; the lidar is under `torso_link` | F6 | latent while `torso_target: 0.0` |
| D4 | `h12_navigation.launch.py` pointcloud_to_laserscan | TF-filtered target frame, the configuration the G1 measured stalling | F8 | not verified on H1-2 |
| D5 | G1 real lidar | points rotated, IMU not; H1-2 has the fix | F4 | by the G1's own TODOs |
| D6 | H1-2 control block | no sim/real config pairing refusal | F15 | read from code |
| D7 | `start_position_verified` | same name, different semantics | F15 | read from code |
| D8 | CLAUDE.md section 7 | "0 -> 300 s default" is true for the G1 explorer only; H1-2 skills default to infinite (`base.py:147`) | | read from code |
| D9 | `g1_nav2.yaml:16`, `g1_navigation/README.md:317` | reference a nonexistent `g1_bringup/config/nav2_config.yaml` | `ls g1_bringup` | verified |
| D10 | mac stack | `docs/NAVIGATION_DEMO.md:21` and `docker/mac/scripts/robot_cli.sh:104-111` (`rob_explore`) need `/skill/frontier_explore` from `h12_skills`, but `launch_ros_mac.sh:125` (`PKGS`) does not build `h12_skills` and `h12_sim_bringup_mac.launch.py` does not launch it; nothing in `docker/mac` mentions skills | grep | static only, not run |
| D11 | `h12_sim_bringup.launch.py:85-96` (and mac, line 38-48) | static `camera_link -> camera_color_optical_frame`: `camera_link` is not in the current URDF (head camera is `head_camera_link`), and no RoboCasa code stamps `camera_color_optical_frame`; only `CL_isaaclab_sim/.../ros_sensor_bridge.py:41` still does | grep | static only |
| D12 | `h12_slam` | `config/fastlio_nav2_config.yaml`, `config/test_nav2_config.yaml`, `rviz/slam.rviz` referenced by nothing; `h12_bringup` `use_sliders` argument dead; two commented-out slider blocks | grep | verified |
| D13 | `h12_skills` metadata | `TODO` description/license, maintainer `root` | `setup.py`, `package.xml` | verified |

---

## 4. Keep separate (looks duplicated, should stay apart)

- **Nav2 tuning, BT and costmap design** (F12): ALMI/H1-2 (MPPI, Omni, strafe, octomap `/projected_map`, no-Spin BT) vs HOMIE/G1 (RPP, no strafe, `/scan` layers). Only the Smac/goal-checker numbers are equal, and sharing 20 lines of YAML would couple two tuned stacks.
- **Scan band and ranges** (F8/F9): fixed by each lidar's mounting height and body (H1-2 lidar >= 1.4 m up with a floor band 0.25-1.7 m; G1 lidar 0.47 m above the pelvis, band -0.59..0.45 relative to the pelvis, hand reach 0.445 m).
- **`level_odom` and `level_frame` internals** (F5): measured vs assumed level frame; different TF topology; the H1-2 one runs on hardware.
- **`livox_to_pointcloud` vs FAST-LIO's registered cloud** (F8): the G1 needs a raw-cloud path for `sim_gt`; the H1-2 takes FAST-LIO's deskewed cloud. Deskewing in `livox_to_pointcloud` would be pointless on the H1-2 chain.
- **FAST-LIO values** (F11): `filter_size`, `blind`, `scan_line`, `point_filter_num` are sensor/sim-density tuning; keep per robot even if the file structure is shared.
- **Bringup shapes** (F14): H1-2 building blocks (several hardware variants, lmg, desktop) vs G1 single core + refusals. Different safety interlock designs (F15) track different safety layers.
- **BringUp nodes lists** (state/safety/frame_task/rviz): same roles, different packages and configs; the only shareable part is the trivial `_include`/path helpers (F16).
- **Description packages** (F19): thin, owner-specific, and the H1-2 one is a separate submodule with its own history (19 commits by another author). Merging would couple a submodule to the G1 layout for little gain. Their `setup.py` recipes look alike but use opposite rewrites and different upstream layouts.
- **`g1_description/leprechaun_g1` symlink and `_install_staging`**: needed by colcon `--symlink-install` (setup.py comment); do not "tidy".
- **`h12_skills` grasp/pick_place stack** (grasp_synthesis, perception, run_telemetry): H1-2/Magpie only; G1 has no counterpart.

---

## 5. Test coverage map for the proposed changes

| Change | Existing guard | Missing |
|---|---|---|
| F1/F2/F3 explorer | G1: `test_frontier_explore.py` (552 lines), `tests/g1/test_sim_explore.sh` | **none for the H1-2** (`h12_skills/test` has only `test_grasp_synthesis.py`); `tests/stack` covers `frame_task`/lowerbody only |
| F4 `livox_imu_upright` | `h12_bringup/test/test_real_lidar_config.py` (rotation, config, launch remap) | a G1 equivalent once its lidar exists |
| F5/F6 level frame / body->pelvis | H1-2: `test_level_odom.py`, `test_mid360_frames.py`; G1: `test_frames.py`, `test_nodes.py`, `test_fastlio.py` | bag-replay check on the real H1-2 chain |
| F7 geometry | `test_robot_geometry.py` (70), `test_frames.py` (235); my zero-joint equivalence check is not in the repo | a cross-implementation test |
| F10/F11 YAML | G1 `test_launch.py`, H1-2 `test_real_lidar_config.py` (`scan_line >= 6`, no extrinsic in YAML) | merged-parameter equality test |
| F14-F17 bringup | G1: `test_locomotion_launch.py`, `test_nav_launch.py` (397 lines) | **H1-2 launch files are untested**; mac cannot be run on x86 |
| F19 descriptions | `g1_description/test/test_joints.py` | H1-2 joint-table/URDF consistency test |

## 6. What I did not do

- Nothing was executed except host-side Python (`difflib`, `ast`, PyYAML, numpy/scipy) on source text and the H1-2 URDF; no ROS, no Docker.
- `grasp.py` and `pick_place.py` were read for their docstrings and interfaces, not line by line; `model_logging.py`, `run_telemetry.py`, `perception_utils.py` only for names.
- The G1 explorer's behaviour at the RoboCasa void edge, the H1-2 pointcloud_to_laserscan stall, and the mac `rob_explore` failure are inferred from code, not reproduced.
- `nav2_bringup` is not on the host, so the claim that G1 could (or could not) reuse `navigation_launch.py` is open.
- Scratch script: `/tmp/claude-1000/-home-max-GOLEM/f8c735f2-cb0b-46cc-a631-673fdc69ab41/scratchpad/dedup/work/sim.py`.

---

# E: RoboCasa simulator and test harnesses, redundancy map

Scope: `h1_robocasa/*.py` against `h1_robocasa/g1/*.py`; `tests/g1/`, `tests/stack/`,
`tests/robocasa/` (with `g1/` and `surface/`), and the `docker/scripts/` helpers those tests call.
Repo `/home/max/GOLEM`, branch `camera_ready`, HEAD `4de2506`. Read-only analysis.

## Summary: top candidates, ranked by (lines saved x confidence) / risk

1. **Spawn placement and RoboCasa registration (S1 + S2).** `g1/g1_robosuite.py` repeats about
   280 lines of `h1_2_robosuite.py`. Six functions match at difflib 0.90-1.00, `install_robocasa_patches`
   at 0.86 and `make_kitchen_env` at 0.95. The two spawn-collision tests match at 0.72, and the H1-2
   copy is run by no runner or CI job. Move them into one robot-agnostic placement module. Effort M,
   sim-only. The G1 lacks the H1-2's "never back through a fixture behind the robot" fix.
2. **Test-harness shell library (T1-T4, T7, T8).** The same host preflight appears in 6 scripts,
   with 3 sim-knob blanking lists, 3 result-dir + `run.json` + cleanup traps and 2 log-line readiness
   polls. That is about 120-150 lines, test-only, no real-robot path. `tests/stack/run.sh` (H1-2) blanks
   none of the sim knobs, so its results depend on `docker/.env`.
3. **Sensor scheduling and message builders (S5, S6, S10).** These are independent rewrites (C). The
   H1-2's `sim_t - last >= period` scheduling runs a 200 Hz IMU at a measured 166.7 Hz (`h12.json`). The
   G1's `_Schedule` gets 199.98 Hz. Sharing `_Schedule` plus one message-builder module saves about 100
   lines. It also fixes that rate, but the H1-2 golden must be re-recorded.
4. **Runner scaffolding (S3, S4, S16).** `main`, env build, the odom flag, the executor, the fall
   logger, shutdown and env-flag parsing are duplicated (difflib 0.4-0.78 per block). Moving them into
   `sim_common` saves about 70 lines at low risk. The H1-2 loop body stays as it is.
5. **Unitree DDS helpers (S11).** Three Python `ROS_DOMAIN_ID` validators exist, and only the G1
   validates `GOLEM_CMD_TIMEOUT`. An empty value crashes the H1-2 sim, which compose and
   `tests/docker` currently work around. Sharing the helpers saves about 30 lines (S/M). Keep the two
   `SimInterface` cores separate.

The biggest non-dedup finding is that **the H1-2 sim's `/livox/imu` is in m/s² with orientation
filled** (golden mean 9.81), while its real driver publishes in g with no orientation (S8).
`lmg_deploy` auto-detects the unit to cope. The G1 sim gets this right.

---

## 0. Method and coverage

- **Similarity figures.** `difflib.SequenceMatcher` ratio over code lines, with blank lines,
  comment lines and docstrings removed. The scripts are in
  `/tmp/claude-1000/-home-max-GOLEM/f8c735f2-cb0b-46cc-a631-673fdc69ab41/scratchpad/dedup/e_work/`:
  - `sim.py`: whole files;
  - `funcs.py`: per-function, via AST;
  - `ranges.py`: line ranges, giving `line_ratio` and `char_ratio`.
- **Read fully:** both runners, both robosuite modules, both `unitree_interface`, both `sim_names`,
  `h12_sensors`, all four G1 sensor modules, `realsense_profile.py`, `sim_common`, `mujoco_ros_bridge`,
  `measurement_bridge`, `mujoco_env`, `sim_timing`, `battery_env`, both spawn tests, `tests/g1/common.sh`
  and its 3 `.sh`, all `tests/stack/*.sh`, `minimal_stack.launch.py`, `observe_simulator.py`,
  `surface/run.sh` + `README`, the 3 `run_in_container.sh`, `wait_for_sim.py`, `docker_common.sh`,
  `docker_stack.sh`, `docker_run.sh`, `launch_robocasa.sh`, `runtime_common.sh`.
- **Skimmed** (structure and interfaces only):
  - `magpie_hand_bridge.py`;
  - `tests/g1/test_sim_*.py`: classes, `Driver`, `run_sim`;
  - `surface/probe.py`: first 140 lines;
  - the H1-2 unit tests: test names plus `test_sensor_bridge.py:207-246`;
  - `tests/g1/README.md`, `tests/README.md`: domain table only.
- `h1_robocasa/`, `tests/` and `docker/` are **not** submodules. The sims consume the submodules
  `CL_Assets` (H1-2 robot/gripper MJCF, battery assets), `third_party/leprechaun_g1` (G1 MJCF/URDF) and
  `unitree_sdk2_python`, and the H1-2 sim also needs `magpie_msgs`/`custom_ros_messages` built into
  `msgs_ws`. Under CLAUDE.md §5/§8 the simulators are shared infra whose default behaviour must be preserved.

## 1. Inventory

### 1.1 Simulator process (`h1_robocasa/h12_mujoco.py`, the one entry point for both robots)

No `setup.py`. `docker/scripts/launch_robocasa.sh:128` and `docker/mac/scripts/launch_robocasa_mac.sh:119`
start it, and `h12_mujoco.py:8-10` hands `--g1`/`GOLEM_ROBOT=g1` to `g1/g1_mujoco.main`. The Mac also
runs the H1-2 path, with FastDDS on arm64 and no golden.

| Node / participant | Class (file:line) | Robot | Publishes | Subscribes / serves |
|---|---|---|---|---|
| `/mujoco_sensors` | `RosSensorBridge` (`mujoco_ros_bridge.py:55`) | both | `/clock` rosgraph_msgs/Clock (every step); opt-in `/odom` nav_msgs/Odometry + `/tf` odom→pelvis (`GOLEM_SIM_ODOM`) | none |
| (suite on that node) | `H12Sensors` (`h12_sensors.py:153`) | H1-2 | `/livox/lidar` livox_ros_driver2/CustomMsg (RELIABLE, depth 20), `/livox/pointcloud` sensor_msgs/PointCloud2, `/livox/imu` sensor_msgs/Imu; per ns ∈ {head, left_hand, right_hand}: `color/image_raw` Image, `color/image_raw/compressed` CompressedImage, `color/camera_info` CameraInfo, `aligned_depth_to_color/{image_raw, image_raw/compressedDepth, camera_info}`, `depth/{image_rect_raw, image_rect_raw/compressedDepth, camera_info}` (`GOLEM_CAMERA_DEPTH`), `depth/color/points` PointCloud2 (`GOLEM_CAMERA_POINTCLOUD`), all sensor-data QoS; `/tf_static` + `/tf` (camera driver edges, 10 Hz) | none |
| (suite on that node) | `G1Sensors` (`g1/g1_sensors.py:398`) | G1 | `/livox/lidar` CustomMsg + `/livox/imu` Imu (RELIABLE, depth 256); `/realsense/head/{color/image_raw, color/image_raw/compressed, color/camera_info, aligned_depth_to_color/image_raw, aligned_depth_to_color/image_raw/compressedDepth}`; `/tf_static` d435_link→head_color_optical_frame | none |
| `/robocasa_measurement` | `MeasurementBridge` (`measurement_bridge.py:40`) | both (shared) | `/robocasa/task_name`, `/robocasa/task_goal` std_msgs/String (transient local), `/robocasa/success` Bool, `/robocasa/reward` Float32; opt-in `/sim/ground_truth` String (`GOLEM_SIM_EXPERIMENT`) | srv `/elastic_band/toggle` std_srvs/Trigger (H1-2 only, when a band is passed); opt-in srv `/sim/reset_objects` Trigger |
| `/magpie_hand_{left,right}` | `MagpieHandBridge` (`magpie_hand_bridge.py:101`) | H1-2 | `/<side>/gripper/state` magpie_msgs/GripperState (10 Hz wall timer) | srv `open`, `close`, `calibrate`, `reset_parameters` (Trigger), `set_position` (SetGripperPosition), `set_force` (SetGripperForce); action `/<side>/gripper/deligrasp` (DeliGrasp) |
| bare Unitree DDS | `SimInterface` (`unitree_interface.py:23`) | H1-2 | `rt/lowstate` unitree_hg LowState_ (wall-clock `RecurrentThread` at dt, no CRC, `mode_machine` 0) | `rt/lowcmd` LowCmd_ |
| bare Unitree DDS | `SimInterface` (`g1/unitree_interface.py:211`) | G1 | `rt/lowstate` (after every physics step, CRC, `mode_machine` 5), `rt/dex3/{left,right}/state` HandState_ (100 Hz sim) | `rt/lowcmd`, `rt/arm_sdk` LowCmd_, `rt/dex3/{left,right}/cmd` HandCmd_ |

**Files and env read.**

- **H1-2.**
  - Files:
    - `CL_Assets/robosuite_assets/robots/h1_2/robot.xml` and `grippers/magpie_{right,left}_gripper.xml`
      (`h1_2_robosuite.py:32-37`);
    - `realsense_profile.yaml`;
    - `CL_Assets/mujoco_assets/battery/**` (`battery_env.py`).
  - Env:
    - placement and tether: `GOLEM_SPAWN_BACKOFF`, `GOLEM_SPAWN_MARGIN`, `GOLEM_SPAWN_OFFSET`,
      `GOLEM_SIM_TETHER_ROT_{STIFFNESS,BODY}`;
    - sensors: `GOLEM_CAMERAS`, `GOLEM_CAMERA_SCALE`, `GOLEM_HAND_CAMERA_RES`,
      `GOLEM_CAMERA_{DEPTH,POINTCLOUD}`, `GOLEM_LIDAR_SELF_FILTER`;
    - odom and experiment: `GOLEM_SIM_ODOM`, `GOLEM_SIM_EXPERIMENT`;
    - DDS: `GOLEM_CMD_TIMEOUT`, `ROS_DOMAIN_ID`;
    - rendering: `MUJOCO_GL`.
- **G1.**
  - Files:
    - `third_party/leprechaun_g1/assets/g1_description/g1_29dof_with_hand_rev_1_0.{xml,urdf}`
      (`g1/g1_robosuite.py:41-43`);
    - `core_ws/src/g1/g1_locomotion/policy/g1.yaml` (`g1_robosuite.py:48-57`);
    - `homie_policy.py` + the ONNX (`g1/stand_in.py:20-25`, mounted at `/home/code/g1_locomotion`).
  - Env:
    - `GOLEM_SPAWN_BACKOFF`, `GOLEM_SIM_ODOM`, `GOLEM_SIM_EXPERIMENT`;
    - lidar: `GOLEM_LIDAR`, `GOLEM_LIDAR_POINTS`, `GOLEM_LIDAR_HZ`;
    - camera: `GOLEM_CAMERAS`, `GOLEM_CAMERA_HZ`, `GOLEM_CAMERA_SCALE`;
    - DDS: `GOLEM_CMD_TIMEOUT`, `GOLEM_ARM_SDK_TIMEOUT`, `G1_DDS_INTERFACE`, `ROS_DOMAIN_ID`.

**Module map** (code lines = non-blank, non-comment):

| H1-2 / shared | G1 | whole-file ratio |
|---|---|---|
| `h1_2_robosuite.py` (581) | `g1/g1_robosuite.py` (588) | **0.496** (220 matching lines) |
| `h12_mujoco.py` (531) | `g1/g1_mujoco.py` (340) | 0.218 |
| `sim_names.py` (130) | `g1/sim_names.py` (126) | 0.289 (tables differ, helpers match) |
| `unitree_interface.py` (175) | `g1/unitree_interface.py` (476) | 0.047 |
| `h12_sensors.py` (737) | `g1/g1_sensors.py` (435) / `g1_lidar.py` (249) / `g1_head_camera.py` (130) / `g1_sensor_msgs.py` (194) | 0.034 / 0.018 / 0.027 / 0.042 |
| `realsense_profile.py` (262) | `g1/g1_head_camera.py` | 0.013 |
| `test_spawn_collision.py` (304, in `h1_robocasa/`) | `tests/robocasa/g1/test_g1_spawn_collision.py` (211) | **0.717** |
| `mujoco_env.py` (ElasticBand, 77) | `g1/stand_in.py` (77) | n/a: different mechanism |
| `battery_env.py`, `magpie_hand_bridge.py`, `sim_timing.py` | none | H1-2 only |
| `sim_common.py`, `measurement_bridge.py`, `mujoco_ros_bridge.py` | (used by both) | already shared |

### 1.2 Test harnesses

| Harness | Robot | How the sim starts | DDS isolation | Domain | Results |
|---|---|---|---|---|---|
| `tests/robocasa/run_in_container.sh` | H1-2 | none: `unittest discover -s tests/robocasa` in a robocasa container | host network; per its README the tests create no ROS nodes or DDS participants | shell's (`load_config` default 1) | stdout |
| `tests/robocasa/g1/run_in_container.sh` | G1 | none: runs `test_g1_*.py` scripts (own `main()`) | `network_mode: none` overlay (`:44-47`) | 97 | stdout |
| `tests/robocasa/surface/run.sh` + `probe.py` | both | the real `launch_robocasa.sh` under `probe.py` | host network + `flock /tmp/golem-dds-domain-194.lock` (`:24-28`) | 194 | `tests/results/surface-<robot>-<time>-<pid>/` |
| `tests/stack/run.sh` (+ `launch_ros_stack.sh`, `run_checker.sh`, `minimal_stack.launch.py`, `observe_simulator.py`) | H1-2 only (`:18`) | a rewritten copy of `launch_robocasa.sh` with `h12_mujoco.py` replaced by `observe_simulator.py` (`:67-76`) | host network, no lock | 193 | `tests/results/stack-...` |
| `tests/g1/common.sh` + `test_sim_{bringup,nav,explore}.sh/.py` | G1 only | `python3 tests/g1/<py> sim` directly, without the launcher (`common.sh:193-208`) | `network_mode: none`; ros joins the sim's netns; loopback multicast (`:29-31`, `:121-140`) | 195-198 | `tests/results/g1-...`, with a build lock `$GOLEM_CACHE_DIR/g1_tests.lock` |
| `docker/scripts/wait_for_sim.py` | both | n/a: waits for one `rt/lowstate` sample and checks `mode_machine` (0 for H1-2, 5 for G1) | n/a | 1..232 | n/a |

**CI** (`.github/workflows/simulator-tests.yml`): the CPU job runs only `test_loop_timing.py` and
`test_robot_select.py` from `tests/robocasa`. It runs no G1 unit test, no spawn-collision test and no
surface probe. The manual GPU job runs `tests/stack/run_all.sh` (H1-2 only). The goldens are
therefore guarded only when someone runs `surface/run.sh` by hand.

**What the goldens check** (`tests/robocasa/surface/README.md`):

- node surfaces (pubs, subs, services, actions, with types);
- bare DDS topics;
- per topic: type, QoS, `frame_id`, `encoding`, `width`/`height` and PointCloud2 fields, exact match;
- `rate_hz` and `livox_imu_mean_acc` within 5 %;
- TF edges;
- that `rt/lowstate` ticks advance.

They do **not** check message payload bytes (the CustomMsg `lidar_id`/`reflectivity`/`line`, the
compressedDepth header, the depth rounding), spawn pose, physics behaviour or log lines. Changes there
are guarded only by unit tests, where they exist.

---

## 2. Overlap findings: simulator modules

### S1. RoboCasa registration and spawn placement: `h1_2_robosuite.py` vs `g1/g1_robosuite.py`. Class A/B

**Per-function matches** (AST, docstrings and comments stripped):

| Function | H1-2 lines | G1 lines | ratio | What differs |
|---|---|---|---|---|
| `_euler_to_wxyz` | `h1_2_robosuite.py:439-459` | `g1_robosuite.py:94-115` | 1.00 | docstring only |
| `place_robot_freejoint` | `219-231` | `326-338` | 1.00 | docstring robot name |
| `place_robot_clear` | `256-267` | `362-373` | 1.00 | nothing |
| `_backward_dir` | `270-282` | `376-386` | 0.93 | H1-2 imports mujoco inside the function |
| `_robot_env_contacts` | `285-303` | `389-405` | 0.96 | same |
| `lowest_robot_point_z` | `234-253` | `341-359` | 0.90 | the G1 skips non-colliding (visual) geoms (`:354`) |
| `install_robocasa_patches` | `462-537` | `465-544` | 0.86 | robot predicate, offsets, log tag |
| `make_kitchen_env` | `543-581` | `550-588` | 0.95 | `robots=` name only |
| `place_robot_collision_free` | `377-433` (+ `_robot_env_gap` `306-344`, `_overlaps_behind` `347-351`, `_placed_gap` `354-374`) | `408-459` | 0.46 | see "Accidental" below |

**Robot-model classes.** `H1_2` (`139-212`) and `G1Dex3` (`249-319`) have identical
`default_base`, `base_xpos_offset`, `top_offset`, `_horizontal_radius` and `arm_type` (1.00 each).
Each class is about 0.81 overall.

**Robot-specific (keep as parameters):**

- robot XML (CL_Assets file vs the G1 MJCF generated at import, `build_robot_xml` `133-208`);
- grippers (Magpie vs `MasslessNullGripper`);
- controller-config name, `init_qpos`, `_eef_name`;
- the `_ROBOT_POS_OFFSETS` entry: H1_2 `[0,0,1.05]` (`:492`), G1Dex3 `[0,-0.30,0.80]` (`:499`);
- the robot predicate: `type(...).__name__ == "H1_2"` (`:496`) vs `isinstance(..., G1Dex3)` (`:503`);
- the G1's `clearance=SPAWN_CLEARANCE` 1 mm (`g1_mujoco.py:36`).

**Accidental divergence (flag):**

- **The G1 lacks the H1-2's spawn-search fixes.** The H1-2 measures a real gap with
  `mj_geomDistance` and excludes the floor. Through `_overlaps_behind`/`blocked` it stops
  when it backs into something behind the robot ("never ends up on the far side of a fixture",
  `:394-398`). It checks extra joint poses and searches up to 50 steps.
  - The G1's version is the older contact-count loop with 25 steps.
  - It can keep backing through a fixture behind the robot, since nothing stops it once a new
    contact appears behind.
  - The H1-2 test `test_margin_blocked_behind_keeps_widest_gap` (`test_spawn_collision.py:251-270`)
    covers exactly this. The G1 has no equivalent.
- **The H1-2 may lack the G1's floor-fit fix.** `lowest_robot_point_z` counts visual geoms on the
  H1-2. The G1 found that its foot meshes reach 1.7 cm below its collision spheres
  (`g1_robosuite.py:344-345`). Whether the H1-2 spawns higher than intended because of this was
  **not measured**.
- **`LEG_JOINT_RENAME` is defined twice inside the H1-2:** `h1_2_robosuite.py:42-46` (unused
  anywhere, checked by grep) and `sim_names.py:33-37` (used). There is a third copy in
  `CL_Assets/robosuite_assets/build_assets.py:58-61`, a submodule. `NOMINAL_STANCE` and
  `nominal_stance_vector` (`h1_2_robosuite.py:67-83`) are also unused.

**Consolidation.** Create `h1_robocasa/robocasa_humanoid.py` (robot-agnostic, numpy + mujoco only at
import) holding:

- the placement primitives (`_euler_to_wxyz`, `place_robot_freejoint`,
  `lowest_robot_point_z(colliding_only=...)`, `place_robot_clear`, `_backward_dir`,
  `_robot_env_contacts`, `_robot_env_gap`, `_overlaps_behind`, `_placed_gap`,
  `place_robot_collision_free`);
- one `install_robocasa_patches(registry)`, where each robot module registers `(class, offsets)`
  and one monkeypatch of `EU.set_robot_base`/`set_robot_to_position` serves both;
- `make_kitchen_env(env_name, robot_name, ...)`;
- optionally a `FloatingBaseHumanoid(LeggedManipulatorModel)` mixin carrying the five identical
  properties.

`h1_2_robosuite.py` and `g1/g1_robosuite.py` keep only assets, grippers, the model class and their
offsets, and re-export the old names (`hr._robot_env_contacts` etc. are used by tests). The H1-2
passes exactly today's arguments, so its default is unchanged. The G1 can either:

- keep its current contact semantics through a mode flag, which is pure dedup with no behaviour
  change; or
- move to the margin algorithm with `margin=0`, which brings it the "blocked behind" fix but
  changes the G1 spawn.

**Lines and effort.** About 280 lines removed from `g1_robosuite.py`, about 30 dead lines from
`h1_2_robosuite.py`. **Effort M.**

**Risk.**

- No real-robot impact, sim only.
- Crosses no submodule boundary, but the simulators are shared infra (§8: preserve defaults).
- Behaviour: none for the H1-2 if its parameters are kept.
- **Guards:**
  - the surface goldens do not see the spawn pose;
  - the guards are the spawn unit tests (S2), the G1 acceptance tests (`tests/g1/*.sh`, which assert
    standing and nav from the spawn), and `tests/stack/run.sh robocasa locomotion` for the H1-2.
  - A shared intermediate class under robosuite's registering metaclass may register an extra
    "robot" name; check that `REGISTERED_ROBOTS` stays clean.
- The port design (`.../audit/6_unified_sim_design.md:225,239,501`) deliberately left placement
  per robot. This proposal revisits that now that both copies are in one repo.

### S2. Spawn-collision tests: `h1_robocasa/test_spawn_collision.py` vs `tests/robocasa/g1/test_g1_spawn_collision.py`. Class B

**Evidence.**

- Whole-file ratio 0.717, with 137 matching code lines.
- The robosuite stubs (`h1:21-44` / `g1:19-42`), the scene (`50-65` / `48-63`), the fake env
  (`86-127` / `84-125`), the 7 shared tests (`130-194` / `129-193`) and `main()` (`289-304` / `196-211`)
  match line for line apart from `hr` vs `gr`.
- The H1-2 adds 7 margin tests (`197-286`).

**What differs.** Only the module under test and the stub set (GripperModel vs NullGripper).

**Accidental.**

- **The H1-2 file is run by nothing:**
  - it sits in `h1_robocasa/`, not `tests/robocasa/`;
  - it is script-style, so `unittest discover` would not collect it anyway;
  - it is absent from CI and from `tests/robocasa/run_in_container.sh`.
- The G1 file needs the leprechaun MJCF and `g1.yaml` at import time, because `g1_robosuite` builds
  its MJCF at module load. It is therefore container-only, while the logic it tests is pure
  numpy + mujoco.

**Consolidation.** After S1, write one `tests/robocasa/test_spawn_placement.py` (unittest) against
the shared module. It imports no robot module, so it is host-runnable and can join the CI CPU job.
It covers the margin tests and the old contact semantics if those are kept as a mode. **Effort S**
once S1 lands. About 140 lines removed.

**Risk.** Test-only. It raises coverage, since the H1-2 spawn tests have been unexercised.

### S3. Runner scaffolding: `h12_mujoco.py` vs `g1/g1_mujoco.py`. Class B

| Block | H1-2 | G1 | ratio (line / char) |
|---|---|---|---|
| `create_kwargs`, make, `env.reset()`, model/data | `130-146` | `111-128` | 0.77 / 0.78 |
| CLI `main` (parse, random/resolve task, `sim_loop`) | `511-531` | `324-340` | 0.41 / 0.73 |
| executor thread for bridge nodes | `346-356` | `207-218` | 0.38 / 0.66 |
| ground-truth odom flag + base body | `238-252` | `166-173` | 0.15 / 0.64 |
| viewer framing on `torso_link` | `371-387` | `220-231` | 0.14 / 0.49 |
| `finally:` shutdown | `498-508` | `313-321` | 0.70 / 0.73 |
| control-step cadence | `398` `round(control_timestep/timestep)` | `240` `round(1/(control_freq*timestep))` | same value |
| `GOLEM_SPAWN_BACKOFF` parse | `164` | `148` | identical text |

**Robot-specific.** H1-2: band, Magpie nodes, cameras, the viewer thread, the timings. G1: the
stand-in settle, physics-reset detection, inline pacing.

**Consolidation.** Add to `sim_common.py`:

- `build_env(make_env, task, layout, style, seed)`;
- `sim_odom_enabled()` and `free_joint_body(model, prefix, name)`;
- `start_executor(nodes)`;
- `main(robot, sim_loop, tag, robot_task=None, task_help=...)`;
- `shutdown(handle, executor, ros_bridge, nodes, extra)`.

`sim_common` stays stdlib-only at import because these functions import lazily. About 60 lines
saved. **Effort S.**

**Risk.**

- Touches the H1-2 default path but no loop logic. Guards: `tests/robocasa/test_sim_common_tasks.py`
  and `test_robot_select.py` (CI CPU), the surface goldens (node and topic surface), and the stack
  tests.
- The readiness string `"ROS bridges up"` (`h12_mujoco.py:354`, `g1_mujoco.py:215`) is an implicit
  contract read by `surface/probe.py:48` and `tests/g1/common.sh:203-207`. Give it one constant in
  `sim_common` that both runners print, and keep the text unchanged.

### S4. Env-knob parsing. Class B/C

| Copy | Location | What it does |
|---|---|---|
| `GOLEM_SIM_ODOM` test | `h12_mujoco.py:244`, `g1_mujoco.py:171` | identical expression |
| `GOLEM_SIM_EXPERIMENT` | `measurement_bridge.py:68-69` | the same expression |
| `_env_on` | `h12_mujoco.py:275-276` | empty string counts as on |
| `GOLEM_LIDAR_SELF_FILTER` | `h12_mujoco.py:326-327` | inline flag |
| `_env_flag` / `_env_number` | `g1_sensors.py:102-115` | empty = default; numbers validated finite and positive |
| `env_seconds` | `g1/unitree_interface.py:70-81` | validated sim seconds |
| unvalidated `float(...)` | `h12_mujoco.py:164, 165, 217, 278`, `unitree_interface.py:84` | no validation |

**Accidental.** The H1-2 accepts NaN or negative values and crashes on an empty
`GOLEM_CMD_TIMEOUT`. That crash is known and worked around in two places:

- `docker/docker-compose.g1.yml:12-14` passes the knob only in the G1 overlay;
- `tests/docker/test_workflows.py:779-783` asserts that the base compose lacks it.

**Consolidation.** Add `env_flag(name, default)` and `env_number(name, default, kind, positive=True)`
to `sim_common`, used by both runners, `measurement_bridge` and both `unitree_interface`s. **Effort S.**

**Risk.** Behaviour changes only for invalid input. It would let the compose workaround go.

### S5. Sensor scheduling. Class C, with an **H1-2 rate bug**

**Locations.**

- H1-2: `h12_sensors.py:422-446` (`if sim_t - self._last_* >= period: self._last_* = sim_t`), and the
  same pattern for odom in `mujoco_ros_bridge.py:113`, which both robots use.
- G1: `_Schedule` (`g1_sensors.py:171-186`), which accumulates `next += period` and restarts after a
  sim-time reset.

**Evidence** (goldens):

| Topic | H1-2 configured | H1-2 golden | G1 golden |
|---|---|---|---|
| `/livox/imu` | 200 Hz (`h12_mujoco.py:314`) | **166.667** (`tests/robocasa/surface/h12.json`) | 199.98 |
| cameras | 15 Hz | 14.706 | 6.0 (configured 6) |
| `/livox/lidar` | 10 Hz | 9.936 | 10.0 |

With 2 ms steps, firing on "≥ period since the last fire" rounds every period up to a whole number of
steps (5 ms becomes 6 ms).

**Classification.** Accidental: it is a bug in the H1-2 and in the shared odom.

**Consolidation.** Move `_Schedule` into `mujoco_ros_bridge.py` (shared) and use it in `H12Sensors`
and the odom path. **Effort S.**

**Risk.**

- Behaviour change: the H1-2 IMU goes from 166.7 to 200 Hz, closer to the real MID-360.
  - `h12.json` must be re-recorded: the IMU rate changes by 20 %, outside the 5 % tolerance. The
    camera (+2 %) and lidar (+0.6 %) changes are inside it.
  - `test_sensor_bridge.py:207-227` pins tick order and schedule; I traced it by hand and it should
    still pass.
- Real-robot impact: none, but sim FAST-LIO gets 20 % more IMU samples.

### S6. Sensor-message construction. Class C

| Job | H1-2 | G1 | ratio (char) | Differences |
|---|---|---|---|---|
| CameraInfo (plumb_bob, K/R/P) | `h12_sensors.py:139-150` | `g1_sensor_msgs.py:183-194` | 0.62 | equivalent output |
| Image from array (`array('B')` fast path) | `533-547` | `132-142` | 0.77 | G1 derives the step from the dtype and forces little-endian |
| JPEG CompressedImage, q95, "rgb8; jpeg compressed bgr8" | inline `601-609` | `159-163` | n/a | **H1-2 encodes only when subscribed** (`601, 613, 619`; tested by `test_compressed_topics_encode_only_for_subscribers`); **the G1 always encodes JPEG + PNG** (`g1_sensors.py:376-377`) |
| compressedDepth | `550-560`, header `struct.pack("<III",0,9,0)` (`:127`, "captured from the robot") | `171-180`, header `struct.pack("<iff",0,0,0)`, depths > 10 m zeroed, PNG level 3 | 0.41 | **the two header choices conflict**; at most one matches a real driver |
| CustomMsg CDR fast path | `_serialize_livox_scan` `97-116` (rclpy-serialized header + packed points) | `livox_custom_msg_cdr` `93-117` (hand-written CDR) | 0.29 | `lidar_id` 0 vs 192, reflectivity 0 vs 100, frame id; same 20-byte point layout |
| Imu message | `448-469` (orientation filled, m/s²) | `livox_imu_msg` `120-128` (no orientation, g) | 0.35 | see S8 |
| depth m to mm, invalid → 0 | `524-530` (truncate, clip at 65535) | `g1_head_camera.py:121-125` (round, zero above 65535) | 0.49 | truncation vs rounding |

**Consolidation.** Promote `g1_sensor_msgs.py`, which is already ROS-message-only and has no node,
to a shared `sensor_msgs_sim.py`, with every robot-specific constant as an argument (`frame_id`,
`lidar_id`, `reflectivity`, compressedDepth header bytes, `depth_max`, PNG level). `H12Sensors`
calls it with today's H1-2 values. About 80 lines saved. **Effort M.**

**Risk.**

- The goldens do not see payload bytes. Guards are `test_sensor_bridge.py` (incl.
  `test_packed_livox_scan_matches_rclpy_serialization`, `test_point_cloud_matches_the_real_driver_layout`)
  and `test_g1_sensor_msgs.py`.
- Port the lazy-encode rule to the G1 camera thread separately; it is a G1 behaviour change.

### S7. Lidar ray-casting. Class C: share the fix, not the model

**Locations.**

- H1-2: a uniform 360x56 grid with 6 lines on a worker thread from an MjData snapshot
  (`h12_sensors.py:241-276`, `642-737`); torso excluded plus a prefix self-filter; all geom groups.
- G1: a golden-ratio pattern of 5000 points per frame, 4 lines, cast every 5 steps from a ring of
  poses on a model copy (`g1_lidar.py:105-236`, `g1_sensors.py:189-261`); robot excluded by root
  body; visual group only.

**Accidental divergences:**

- **The H1-2 may lack the G1's `mj_multiRay` BVH-culling workaround** (`g1_lidar.py:177-190`).
  Without it, the G1 measured 3-5 % of rays passing through walls in a RoboCasa kitchen. The H1-2
  calls the same `mj_multiRay` on the same MuJoCo 3.3.1 (`h12_sensors.py:681-690`). This is
  inferred, **not measured on the H1-2**.
- **The H1-2 emits lines 0-5** (`h12_sensors.py:261`). The real MID-360 driver emits 0-3
  (`livox_ros_driver2/src/comm/comm.h:82` `kLineNumberMid360 = 4`, `pub_handler.cpp:386`).
  `h12_slam/config/mid360.yaml:37-38` acknowledges "the real MID360 emits lines 0-3, the sims 0-5".

**Robot-specific (keep):** scan pattern and density, mount body, frame name. These are tuning that
H1-2 SLAM has been validated against.

**Consolidation.** Extract only `cast(model, data, origin, rot, dirs, exclude_body, robot_geom_mask,
groups, min, max)`, with the BVH workaround, into a shared `livox_sim.py`. Both suites call it.
**Effort S-M.**

**Risk.** H1-2 scan contents change (fewer phantom pass-throughs, and a different group set if
unified). The goldens are not sensitive to that; the guards are `test_sensor_bridge.py` lidar tests
and the stack tests. Keep the patterns, threading and line numbering separate unless someone decides
to match the real driver.

### S8. Livox IMU model. Class C; mostly robot-agnostic, with an H1-2 fidelity divergence

**Locations.**

- H1-2: reads MJCF framequat/gyro/accelerometer sensors on the livox site (`h12_sensors.py:448-469`).
  It publishes orientation and acceleration in **m/s²** (golden `livox_imu_mean_acc` 9.8141).
- G1: `Mid360.imu` (`g1_lidar.py:239-249`) computes the specific force at the IMU's offset in **g**,
  with no orientation (golden 1.0003).

**Accidental.** The real H1-2's `/livox/imu` comes from `livox_imu_upright`, which states that the
values are "units of g" and that "the driver never fills orientation"
(`core_ws/src/h12_bringup/h12_bringup/livox_imu_upright.py:72-93`). `lmg_deploy` works around the sim
with unit auto-detection: "The MID360 reports acceleration in g, the simulators' /livox/imu in m/s^2"
(`lmg_deploy/livox_imu.py:18-28`, `livox_attitude.py:14`). The H1-2 sim therefore deviates from its
own robot where the G1 sim does not.

**Consolidation.** Reuse the G1's `imu()` math (site velocity/acceleration plus an offset) for the
H1-2, scaled to g with orientation zeroed. **Effort S.**

**Risk.**

- Re-record `h12.json`: `livox_imu_mean_acc` 9.81 → about 1.0.
- Check every H1-2 sim consumer of `/livox/imu`. FAST-LIO normalises to the mean, and `lmg_deploy`
  auto-detects; I did not check the others.
- This is a behaviour change on the default H1-2 path. It needs an announcement, not just a dedup
  commit.

### S9. Head-camera rendering and intrinsics. Class C

**Locations.**

- Intrinsics: `realsense_profile.fovy_intrinsics` (`realsense_profile.py:125-128`) and
  `g1_head_camera.intrinsics` (`g1_head_camera.py:57-60`). Same formula (fy = h / (2 tan(fovy/2)),
  cx = w/2 - 0.5); char ratio 0.75.
- Renderer setup (raise offwidth/offheight, build the Renderer): `h12_sensors.py:497-506` vs
  `g1_head_camera.py:80-88` (0.58).
- Scene options (geomgroup, sitegroup 0): `h12_sensors.py:513-522` vs `g1_head_camera.py:76-78`.
  The groups shown are robot-specific (the H1-2 shows group 2 for the Magpie visuals).

**Divergences.**

- The G1 omits `aligned_depth_to_color/camera_info`, which `realsense2_camera` publishes with
  `align_depth` and the H1-2 sim does publish. It is absent from `g1.json`. The real G1's surface has
  not been captured.
- Threading differs: the H1-2 waits for each frame and never drops one (snapshot copy); the G1 hands
  the scene to a thread and drops a frame when busy.

**Consolidation.** The G1 should call `realsense_profile.fovy_intrinsics`. **Effort S.**

Describing the G1 head camera as a `realsense_profile.yaml` entry (no captured intrinsics, the
fallback path in `camera_specs` `204-235`) would let one camera publisher serve both. That is
**L** effort, because the threading models differ.

**Risk.** The G1 golden would change if `aligned_depth_to_color/camera_info` were added.

### S10. Sim time to `builtin_interfaces/Time`. Class A/C

**Locations.** `mujoco_ros_bridge._sim_time_to_msg` (`mujoco_ros_bridge.py:31-38`, truncates) and
`g1_sensor_msgs.time_msg` (`g1_sensor_msgs.py:63-66`, rounds to the nearest ns). The G1 process
uses both: `/clock` and the static TF truncate, its sensor stamps round.

**Evidence.** At t = 0.006 s the first gives 5,999,999 ns; the second gives 6,000,000.

**Consolidation.** Keep one rounding function in `mujoco_ros_bridge`. **Effort S.**

**Risk.** Stamps shift by at most 1 ns; the goldens do not see it.

### S11. Unitree DDS sim interfaces: `unitree_interface.py` vs `g1/unitree_interface.py`. Class C for the cores, A/C for the helpers

**Evidence.** Whole-file ratio 0.047. The cores are different designs:

- the H1-2 has 27 motors, a wall-clock lowstate thread, no CRC and no header checks;
- the G1 has 29 + 2x7 motors, a stand-in blend through `rt/arm_sdk`, CRC and `mode_machine`
  checks, per-step lowstate and the Dex3 hands.

**The same job done three times:**

- **`ROS_DOMAIN_ID` validation:**
  - `unitree_interface.py:21-22`: `int(os.getenv(...))` at import, so a `TypeError` when unset, plus
    an `assert`, which `-O` strips;
  - `g1/unitree_interface.py:198-208`;
  - `mujoco_ros_bridge.init_ros` `159-180`.
- **Timeout knob:** `unitree_interface.py:84` vs `env_seconds` (`g1:70-81`); see S4.
- **Latched-command PD per step:** H1-2 `low_cmd_handler` + `write_ctrl` (`124-158`, per-motor loop,
  `mode == 1`) vs G1 `MotorCommand.torques` / `parse_motor_cmds` (`96-118`, vectorised,
  `mode != 0`); char ratio 0.22.

**Accidental (flag):**

- **H1-2 torn command latch.** `low_cmd_handler` writes six arrays from the DDS thread without the
  sim lock (`124-137`). `write_ctrl` reads them under the lock, so one physics step can mix two
  frames. The G1 swaps one immutable tuple (`:304`).
- **No non-finite rejection** in the H1-2.
- **No handling of MuJoCo's auto-reset** in the H1-2. After `data.time` jumps back,
  `check_cmd_timeout` compares a negative silence and the watchdog cannot fire until sim time catches
  up. The G1 has `physics_reset` (`369-379`) and detection in its runner (`g1_mujoco.py:47-49`,
  `259-269`).

**Consolidation.** Create a small shared `unitree_sim_dds.py` with:

- `sim_domain_id()`, used by both interfaces and by `init_ros`;
- `env_seconds`;
- `MotorCommand` / `parse_motor_cmds`;
- optionally `fast_crc`.

The H1-2 `SimInterface` adopts the domain and timeout helpers (S), and optionally `MotorCommand` for
the atomic latch (M). **Keep the two classes separate.**

**Risk.**

- The H1-2 controllers (FAME, ALMI) were tuned against this sim. The PD math stays the same, but
  the latch and NaN changes alter edge-case behaviour.
- Guards: `tests/stack/run.sh robocasa locomotion|arms`; the goldens cover only bare DDS types and
  tick advance.
- `wait_for_sim.py:13` relies on the H1-2 sim leaving `mode_machine` at 0. Do not set it.

### S12. Name resolvers: `sim_names.py` vs `g1/sim_names.py`. Helpers A, tables D

**Evidence.** `_id`, `site_id`, `body_id` and `strip` match: `sim_names.py:45-46, 100-110` vs
`g1/sim_names.py:55-56, 116-126` (ratio 0.78). The motor-resolution loops (`59-92` vs `95-114`) share
a shape only (0.05 / 0.36).

**Robot-specific.** Tables, leg renames, torque sensors vs `actfrcrange`, hands, base indices.

**Consolidation.** At most a 15-line `PrefixedNames` base. Low value; do it only alongside S1/S11.
Within the G1, `FREE_JOINT` is defined in `g1/sim_names.py:44` and `g1_robosuite.py:54`
(and in `h1_2_robosuite.py:49`).

### S13. Real-time pacing and RTF logging. Class C

**Locations.**

- H1-2: `sim_timing.RealtimePacer` (`sim_timing.py:7-30`, unit-tested in `test_loop_timing.py` in
  CI) and `LoopTimings` (`33-80`).
- G1: inline pacing (`g1_mujoco.py:296-312`) with the same `MAX_LAG` 0.1 s, plus its own RTF log.

**What differs.** On a lag above `max_lag`, the H1-2 resyncs the deadline to `now - max_lag`; the G1
resyncs to zero lag.

**Consolidation.** The G1 uses `RealtimePacer`, with an optional slept-time counter, and `LoopTimings`
for its RTF line. `sim_common._UNSUPPORTED` (`sim_common.py:19`) then drops `--timing` for the G1, and
`--duration` could be added to the H1-2 the same way. **Effort S.**

**Risk.** G1 only. The CLAUDE.md line "`sim_timing.py` is the H1-2's only" would change.

### S14. One side got a loop fix the other lacks (not duplication; recorded so a merge carries them)

| Fix | Has it | Missing | Evidence |
|---|---|---|---|
| Viewer on its own thread with its own MjData at 60 Hz (`handle.sync()` costs about 11 ms) | H1-2 `h12_mujoco.py:82-95, 358-426` | **G1 calls `handle.sync()` every physics step** (`g1_mujoco.py:294-295`); GUI mode not measured, the tests are headless | read |
| MuJoCo unstable-reset detection + DDS restamp | G1 `g1_mujoco.py:47-49, 257-269`; `unitree_interface.py:369-379` | H1-2 | read |
| GIL switch interval 0.5 ms (rt/lowstate jitter from sensor threads) | G1 `g1_mujoco.py:41-46, 190` | H1-2 (its lowstate thread also competes with the camera and lidar workers) | read; H1-2 jitter not measured |
| Sensor stamps = time of the measured kinematics (t − dt) | G1 `g1_sensors.py:411-418` | H1-2 stamps with post-step `data.time` while snapshotting pre-step kinematics | read |

### S15. G1-internal duplication

- **Policy-dir lookup:** `g1/stand_in.py:20-23` and `g1/g1_robosuite.py:40, 48-51`, the same
  `_POLICY_DIRS` tuple and `next(...)` (A). Put it in one helper in `g1/__init__.py`. **Effort S.**
- **Fall logger:** H1-2 `h12_mujoco.py:388-392, 464-479` (torso, 0.5 / 0.5 m, band release) and G1
  `g1_mujoco.py:232-236, 272-285` (pelvis, 0.5 / 0.2 m, stand-in release) (B). A shared
  `FallLogger(body, min_upz, min_h, release_label)` in `sim_common` saves about 20 lines. **Effort S.**
  The `[fall-logger] *** ROBOT FELL` line is grepped by the G1 nav test's `fall_log` check
  (`tests/g1/test_sim_nav.py:982-990` greps `ROBOT FELL`), so keep the text.

---

## 3. Overlap findings: test harnesses

### T1. Host preflight preamble. Class B, 6 copies

Each copy runs `source docker_common.sh`, `export GOLEM_ROBOT=X`, `load_config`,
`[configure_robot]`, `check_domain`, `GOLEM_DISPLAY=headless`, `configure_display` and
`prepare_mounts`:

- `tests/robocasa/run_in_container.sh:5-12`
- `tests/robocasa/g1/run_in_container.sh:35-43`
- `tests/robocasa/surface/run.sh:42-50`
- `tests/stack/run.sh:16-24`
- `tests/g1/common.sh:81-88`
- `tests/isaaclab/run_in_container.sh:71-76`

**Divergence.** `tests/isaaclab/run_in_container.sh` does not pin `GOLEM_ROBOT=h12` as its RoboCasa
siblings do. Harmless today, since `configure_robot` is never called and `prepare_mounts isaac`
ignores the robot.

**Consolidation.** Add `test_preflight ROBOT SERVICE...` to `docker_common.sh`. About 40 lines saved.
**Effort S.**

**Risk.** Test-only. `tests/docker/test_workflows.py` already exercises `docker_common.sh` and would
cover the new function.

### T2. Sim-knob blanking. Class B, 3 lists plus one missing

**Copies.**

- `surface/run.sh:31-41`: `pinned` + `allowed`, 19 knobs, with a self-check that every
  `GOLEM_*`/`G1_*` the robocasa service interpolates is listed (`:51-67`).
- `tests/robocasa/g1/run_in_container.sh:30-34`: 12 knobs.
- `tests/g1/common.sh`: explicit `-e` per knob (`:196-198`) plus each test's `SIM_ENV` and
  `g1_test_shell_knobs`.

**Accidental.** `tests/stack/run.sh` sets only `GOLEM_CAMERAS` (`:56`). These knobs from the shell or
`docker/.env` reach the H1-2 stack acceptance sim through `docker/docker-compose.yml`'s robocasa
service:

- `GOLEM_SPAWN_MARGIN`, `GOLEM_SPAWN_OFFSET`;
- `GOLEM_SIM_TETHER_ROT_*`;
- `GOLEM_SIM_ODOM`, `GOLEM_SIM_EXPERIMENT`;
- `GOLEM_CAMERA_*`, `GOLEM_HAND_CAMERA_RES`.

`docker/docker-compose.yml:90-113`, robocasa service from line 64.

**Consolidation.** Put one `SIM_KNOBS` array and `clear_sim_knobs` in `docker_common.sh` (or a
`tests/lib/sim_knobs.sh`). The surface self-check then validates the shared list, and all four
harnesses use it. **Effort S.**

**Risk.** Test-only. The stack results become independent of `docker/.env`, which is a fix.

### T3. Result directory, `run.json` provenance and cleanup trap. Class C, 3 implementations

| Harness | Directory | `run.json` | Cleanup |
|---|---|---|---|
| `surface/run.sh:70-90` | `tests/results/surface-<robot>-<time>-<pid>`, `chmod 777` | printf JSON: `tree_dirty` bool, one image ID, **no submodules** | `docker rm -f`, `exit "$code"` |
| `tests/stack/run.sh:25-55` | same pattern, `chmod 777` | python heredoc: `working_tree`, `submodules`, `images` | logs saved, `docker rm -f`, `exit "$code"` |
| `tests/g1/common.sh:97-103, 145-170` | `tests/results/g1-<name>-<time>-<pid>`, refuses a non-empty `OUT` | printf + `g1_json_string` | `chown` through a container instead of `chmod 777` |

**Divergence.** The surface runs omit the submodule state, yet the H1-2 surface depends on
`CL_Assets` (a submodule) and the G1's on `leprechaun_g1`.

**Consolidation.** A `tests/lib/run_record.py` (stdlib) that takes key=value pairs and always adds
the commit, dirty tree, submodules and image IDs, plus a shell `test_result_dir NAME` and a
`test_cleanup CONTAINERS...` hook. About 60 lines saved. **Effort M**, because the schemas must be
unified; check whether any reader parses `run.json`.

**Risk.** Test-only.

### T4. Readiness waits. Class C, 4 mechanisms

| Mechanism | Location |
|---|---|
| Grep `"ROS bridges up"` in the sim log | `tests/g1/common.sh:201-207` (300 x 1 s, checks the container is still running) |
| Regex on the same line | `surface/probe.py:48` |
| `docker logs` grep for `STACK_BUILD_READY` | `tests/stack/run.sh:59-64` (600 x 1 s, same liveness check) |
| DDS sample + `mode_machine` | `wait_for_sim.py` (`launch_ros.sh:114-115` with `GOLEM_WAIT_FOR_SIM=1`, and `tests/stack/launch_ros_stack.sh:39`) |

**Consolidation.** A shell `wait_for_log CONTAINER|FILE PATTERN TIMEOUT` for the two poll loops, and
the S3 readiness-line constant. **Effort S.**

**Risk.** Low. The G1 acceptance tests skip `wait_for_sim.py`'s robot-identity check, but they run in
an isolated netns, so no foreign sim can be present (D).

### T5. `ROS_DOMAIN_ID` validation. Class C, 9 copies with different bounds

| Copy | Bounds |
|---|---|
| `docker_common.sh:31-35` | 0..232 |
| `runtime_common.sh:2-11` | 1..232 |
| `tests/stack/launch_ros_stack.sh:4-10` | 1..232 |
| `tests/stack/minimal_stack.launch.py:13-17` | 1..232 |
| `wait_for_sim.py:26-29` | 1..232 |
| `tests/g1/common.sh:64-68` | > 0, no upper bound |
| `mujoco_ros_bridge.py:159-180` | > 0, no upper bound |
| `g1/unitree_interface.py:198-208` | > 0, no upper bound |
| `unitree_interface.py:21-22` | `assert > 0`; `TypeError` when unset |

**Consolidation.**

- Python: one `sim_domain_id()` (S11) for the sim. `wait_for_sim.py` and `minimal_stack.launch.py`
  live in other containers, so leave them, or source `runtime_common.sh` from `launch_ros_stack.sh`,
  where `/home/code/h12_sim_scripts/` is mounted.
- Shell: `g1_test_setup` can call `validate_sim_domain`.

**Effort S.** Low risk.

### T6. DDS isolation and domain locking. Class C, plus a divergence

**Copies.**

- The G1 harnesses isolate with `network_mode: none`, written as an overlay in two places:
  `tests/robocasa/g1/run_in_container.sh:44-47` and `tests/g1/common.sh:121-140`. `common.sh` also
  sets `network_mode: container:<sim>` and turns on loopback multicast (`:29-31`).
- The surface probe uses the host network plus `flock` on the domain (`surface/run.sh:24-28`).
- The H1-2 stack tests use the host network with **no lock and no isolation**, on domain 193, with
  the base compose's `network_mode: host`. A concurrent run, or a user's sim on 193, would join
  their graph, and their DDS leaves the host.

**Consolidation.** Add `isolate_dds` (no-network overlay + loopback multicast + ros-in-sim netns) and
`lock_domain DOMAIN` to the shared test library. `tests/stack/run.sh` adopts them. **Effort M**,
since the stack test's three containers must share one netns.

**Risk.** Test-only. Behaviour change for the stack tests; re-validate on the GPU worker.

### T7. CycloneDDS local pin. Class B

`write_g1_dds_pin` (`docker_common.sh:110-126`) writes `<CycloneDDS/>` or a pin into the shared
`$GOLEM_CACHE_DIR/cyclonedds_local.xml`. `tests/g1/common.sh:104, 138` writes its own `<CycloneDDS/>`
into `$OUT` so it never touches a running container's file. This is deliberate.

**Consolidation.** `write_g1_dds_pin [PATH]` with an optional target. **Effort S.** Low risk.

### T8. In-container ROS sourcing + `compose run ... robocasa bash -lc '...'`. Class A, 4-5 copies

- `tests/robocasa/run_in_container.sh:13-14`
- `tests/robocasa/g1/run_in_container.sh:49-66`
- `surface/run.sh:106-111`
- `tests/g1/common.sh:196-200`
- `launch_robocasa.sh:89-93`

Each repeats `source /opt/ros/humble/setup.bash && source /opt/livox_ws/install/setup.bash`.

**Divergence.** The G1 acceptance tests bypass `launch_robocasa.sh` entirely (no
`validate_sim_domain`, no `msgs_ws` build, their own `MUJOCO_GL`). A launcher regression would
therefore reach the G1 only through the surface probe.

**Consolidation.** A `runtime_common.sh` `source_sim_env` function used by the launcher and every
test. **Effort S.**

### T9. In-process sim telemetry recorders. Class C

**Locations.**

- H1-2: `tests/stack/observe_simulator.py:56-78` wraps `mujoco.mj_step` and finds `sim_loop`'s
  locals `band` and `resolver` through `sys._getframe(1)`. It records nothing, silently, if those
  local names change.
- G1: `tests/g1/test_sim_bringup.py:162-196` and `test_sim_nav.py:308-348` monkeypatch
  `g1.unitree_interface.SimInterface.after_step` / `__init__`, a stable seam.
- Both run the runner with `runpy`.

**Consolidation.** Add an explicit opt-in step hook to both runners, for example
`sim_common.STEP_HOOKS`, called after each physics step with (model, data, interface, extras), and
give the H1-2 stack recorder that seam in place of frame introspection. **Effort S-M.**

**Risk.** It touches the H1-2 loop (one call per step, a no-op by default). The hook must never
publish, to keep the ground-truth rule (§8).

### T10. G1 acceptance drivers: internal duplication. Class B

**Evidence.**

- `Driver.mark` / `spin_for` / `wait_future`: `test_sim_bringup.py:257-287` vs `test_sim_nav.py:537-585`
  (0.54 / 0.67).
- The `/clock` subscription and `_on_clock`: `236-255` vs `385-432`.
- `run_sim`: `162-196` vs `308-348`, the same runpy/monkeypatch/SIGTERM skeleton.
- `test_sim_nav.Driver` does not subclass the bringup `Driver`, although `Recorder` and `Report` are
  already inherited.

**Consolidation.** A `SimClockDriver(tag, spin=...)` base and a
`run_sim(out, sim_args, recorder, extra_patches)` helper in `test_sim_bringup.py`. About 50 lines.
**Effort S.** G1 test-only.

### T11. Test conventions and runners. Class C

The H1-2 unit tests are `unittest` classes, discovered. The G1 unit tests are scripts with their own
`main()`, run one by one (`tests/robocasa/g1/run_in_container.sh:56-66`). So:

- `python -m unittest discover` cannot collect the G1 tests, or the H1-2 `test_spawn_collision.py`;
- CI runs none of them.

**Consolidation.** Convert the G1 unit tests to `unittest`, or run both runners under pytest.
Keep the separate container overlay (DDS domain 97, no network) for `test_g1_unitree_interface.py`,
and add the host-runnable ones (spawn placement after S1/S2, `test_g1_sensor_msgs.py` if it has no
GL) to the CI CPU job. **Effort S-M.**

---

## 4. Keep separate

| What | Where | Why |
|---|---|---|
| `SimInterface` cores | `unitree_interface.py:23-175`, `g1/unitree_interface.py:211-476` | Different DDS surfaces (27 motors vs 29 + 2x7 + `rt/arm_sdk`), header semantics (`mode_machine` 0 vs 5, which `wait_for_sim.py` relies on), and publish timing (wall vs per step); H1-2 controllers are tuned on this sim. Share helpers only (S11). |
| Robot tables | `sim_names.py:18-42`, `g1/sim_names.py:23-52` | DDS slot order, leg renames, torque sensors vs `actfrcrange`: robot-specific (D). |
| Elastic band vs stand-in | `mujoco_env.py`, `h12_mujoco.py:190-227, 439-456` vs `g1/stand_in.py`, `g1_mujoco.py:52-87` | Each emulates its own real support: a gantry tether for the H1-2, Unitree's built-in controller handed over through `rt/arm_sdk`/`rt/lowcmd` for the G1 (D). |
| Lidar scan patterns, densities, threading | `h12_sensors.py:241-276, 642-737` vs `g1_lidar.py`, `g1_sensors.py:189-291` | Tuned against each robot's SLAM; share only the cast core and the BVH fix (S7). |
| Magpie gripper bridge vs Dex3 over DDS | `magpie_hand_bridge.py` vs `g1/unitree_interface.py` hand paths | Different real transports: ROS services/action vs Unitree DDS (D). |
| Robot MJCF sources | CL_Assets prebuilt `robot.xml` vs `g1_robosuite.build_robot_xml` (`133-224`) | Different asset pipelines and submodules (D). |
| `battery_env.py`, `realsense_profile.py/.yaml` camera captures | n/a | H1-2-only task and captured robot data. |
| `wait_for_sim.MODE_MACHINE` table | `wait_for_sim.py:13` vs `g1/unitree_interface.py:59` | Different containers; `tests/docker/test_wait_for_sim.py:66-69` pins the values. |
| Stack checkers vs G1 reports | `tests/stack/check_*.py`, `tests/g1/test_sim_*.py` Report | Different controllers and pass criteria. Harness plumbing can be shared (T1-T4); the scoring cannot. |

## 5. Accidental divergences that may be bugs

H1-2 side; these change the default path, so announce them and re-record goldens where noted:

1. **IMU and camera rates.** The `/livox/imu` rate is 166.7 Hz instead of the configured 200 Hz, and
   the cameras run at 14.7 Hz instead of 15 (S5; `h12.json`). The scheduler is shared with odom.
2. **`/livox/imu` fidelity.** It is in m/s² with orientation filled; the real driver gives g with no
   orientation (S8; `livox_imu_upright.py`, `lmg_deploy/livox_imu.py:18`).
3. **Lidar line numbers.** Lines 0-5; the real MID-360 gives 0-3 (S7; `comm.h:82`).
4. **Possible wall pass-through rays.** `mj_multiRay` BVH culling may let 3-5 % of rays through walls
   (S7; inferred from the G1's measurement).
5. **`SimInterface` robustness.**
   - an empty `GOLEM_CMD_TIMEOUT` crashes the sim (worked around in compose);
   - an unset `ROS_DOMAIN_ID` gives a `TypeError` at import;
   - the command latch is non-atomic;
   - non-finite commands are not rejected;
   - the watchdog is dead after a MuJoCo auto-reset (S11, S14).
6. **Floor fit.** It includes visual geoms (S1; effect unmeasured).

G1 side:

7. **Spawn search.** No "blocked behind" stop and no real-gap measurement; it can back through a
   fixture behind it (S1).
8. **GUI-mode pacing.** `handle.sync()` runs every physics step (S14; unmeasured).
9. **Encoding cost.** JPEG and PNG are encoded on every head-camera frame even with no subscriber
   (S6).
10. **Missing topic.** `aligned_depth_to_color/camera_info` is not published (S9).

Both sides:

11. **compressedDepth ConfigHeader.** `(0,9,0)` vs all zeros; at most one matches the real driver
    (S6).
12. **Two sim-time-to-stamp conversions** inside one G1 process (S10).

Test side:

13. **`tests/stack/run.sh` is not insulated.** It does not blank sim knobs from `docker/.env` and
    has no DDS isolation or domain lock (T2, T6).
14. **The H1-2 `test_spawn_collision.py` is run by nothing** (S2, T11).

**Noticed in passing; outside this audit, verify before acting.** `h12_real_drivers.launch.py:24-29`
runs the real H1-2 driver with `xfer_format` 1 (CustomMsg only), `publish_freq` 30 Hz and frame
`livox_frame`. The H1-2 sim publishes `/livox/lidar` at 10 Hz in `lidar_link` and also
`/livox/pointcloud` (`h12.json`). That would put a topic in the H1-2 sim surface that the real H1-2
does not publish, contrary to CLAUDE.md §8.

## 6. Suggested order

1. S2 + T11 (spawn tests into `tests/robocasa`, unittest, CI) and T1/T2/T8 (shell preflight,
   knob list, sourcing). These are test-only and make the later steps safer.
2. S1 (shared placement module). The H1-2 keeps its exact arguments. Decide separately whether the
   G1 adopts the margin search.
3. S3/S4/S15 (runner scaffolding and env helpers into `sim_common`).
4. S5 + S10 (scheduler and time), with an `h12.json` re-record in the same commit.
5. S6/S7 shared cores. Then S8 and S11 as announced H1-2 behaviour changes, run against
   `tests/stack/run.sh robocasa arms|locomotion` and the surface probe.

---

# F — Cross-cutting helper duplication, and inventory of the packages no other agent covers

Repo `/home/max/GOLEM`, branch `camera_ready`, HEAD `4de2506`. Read-only analysis (host-side grep, `diff`, `difflib`, AST scans, and a numeric check of the quaternion helpers).

## Summary: top consolidation candidates, ranked by (lines saved x confidence) / risk

1. **`h12_skills/h12_skills/model_logging.py` == `model_server/model_server/model_logging.py`**: 223 lines, `diff` shows 0 differing lines (class A). H1-2 only, off the robot command path, risk ~0 (S). The file header says the copy is deliberate ("no shared dependency, keep in sync"); `mat_to_quat` is likewise byte-identical in `h12_skills/perception_utils.py:81-99` and `model_server/graspgen_server.py:169-187`.
2. **DDS init and domain guard** (`_channel_factory_settings`, `init_channel_factory`, `init_channel_factory_guard`): `init_channel_factory` and `init_channel_factory_guard` are byte-identical (ratio 1.00) in all four of `h12_ros2_controller`, `g1_control`, `h12_safety_layer`, `g1_safety_layer`. `g1_locomotion` and `g1_dex3` have two more variants, 4 inline `ChannelFactoryInitialize` call sites repeat the `if interface:` branch, and about 15 more places parse `ROS_DOMAIN_ID`. Class A/B, ~150 lines. Home: a new in-tree package (not a submodule). Effort S for the package, M for adoption. Risk low to medium (runs before the real-robot DDS join; two submodules involved).
3. **YAML-path/loader helpers**: `_ensure_yaml_extension` x6 (identical), `_resolve_config_path` x7 (3 flavours), `_load_yaml` x4, positive-number validators x4 inside the G1 alone. ~110 lines, S, low risk.
4. **H1-2 joint tables**: `h12_ros2_controller/utility/joint_limits.py` and `h12_safety_layer/core/joint_limits.py` carry the same position/velocity/torque limit tables (verified element-for-element equal by executing both). Six more 27-name lists sit elsewhere. The G1 already has a single generated table (`g1_description.joints`). A shared H1-2 table would be ~130+ lines saved, M, medium risk (safety limits; two submodules).
5. **Latest-only DDS reader**: this is a divergence fix more than a line-count win. The H1-2 controller and safety layer read `rt/lowstate`/commands with a listener-free newest-sample reader (`LatestReader`, introduced 2026-10-12), `g1_locomotion` wrote its own, and `g1_control` and `g1_safety_layer` still use SDK callback subscribers, which `g1_locomotion/driver.py:43-47` itself documents as a latency hazard. Three implementations, ~70 lines. Risk medium (G1 control path; G1 has never been on hardware).
6. **Quaternion/rotation math**: ~25 hand-rolled implementations across 9 H1-2 packages plus both sims, almost none on the G1 side (it uses `scipy`). Mostly an H1-2-internal cleanup (~250 lines, M); low payoff for G1.

Also found while reading (details in section 6): `g1_control` silently ignores `FrameTask.relative`, `FrameTask.slow_mode` and `NamedConfig.slow_mode`, which the shared actions define and `h12_ros2_controller` honours. `custom_ros_messages` is pinned to a commit that is on no remote-tracking branch locally. `cl_realsense/pc_acc` cannot import (`arpa_control` is absent).

Abbreviations used below (all relative to `/home/max/GOLEM`):
`H12C` = `core_ws/src/h12_ros2_controller/h12_ros2_controller`, `G1C` = `core_ws/src/g1/g1_control/g1_control`, `H12S` = `core_ws/src/h12_safety_layer/h12_safety_layer`, `G1S` = `core_ws/src/g1/g1_safety_layer/g1_safety_layer`, `H12L` = `core_ws/src/h12_lowerbody_rl/h12_lowerbody_rl`, `G1L` = `core_ws/src/g1/g1_locomotion/g1_locomotion`, `SIM` = `h1_robocasa`, `ISAAC` = `CL_isaaclab_sim/src/python`.

## 0. Method and limits

- Scanned 622 first-party Python files (tracked files of the main repo plus tracked files of the first-party submodules `cl_realsense`, `custom_ros_messages`, `estop`, `h12_ros2_controller`, `h12_ros2_model`, `h12_safety_layer`, `magpie_control`, `magpie_msgs`, plus `CL_isaaclab_sim`). Excluded: `FAST_LIO`, `livox_ros_driver2`, `unitree_ros2`, `unitree_sdk2_python`, `mujoco_mpc`, `third_party`, `CL_Assets`. The scratch scripts are in `.../scratchpad/dedup/` (`filepairs.py`, `funcdups.py`, `funcdups2.py`, `fcmp.py`).
- "ratio" below is `difflib.SequenceMatcher` over lines with blanks and `#` lines stripped, unless I say "raw diff" (plain `diff`, counting differing lines). "Jaccard" is over the set of stripped lines.
- `core_ws/src/h1_bringup/` and `core_ws/src/locomotion_game_deploy/` are leftover `__pycache__`/`.pytest_cache` directories with no tracked files (`git ls-files` is empty); ignored.
- I did not read the two safety layers, the two controllers, nor the lower-body packages in full (other agents have them). Where they appear, it is only as call sites of a helper.
- Context for ownership: of the packages I inventory, `cl_realsense`, `custom_ros_messages`, `estop`, `magpie_control`, `magpie_msgs` are submodules; `model_server` is in-tree. `h12_ros2_controller`, `h12_safety_layer`, `h12_ros2_model` are submodules too, so any consolidation touching H1-2 controller/safety code crosses CLAUDE.md §5. Every `g1_*` package is in-tree.
- Container boundary that limits sharing: the `robocasa` container mounts `h1_robocasa`, `CL_Assets`, `docker/scripts` and the two message packages only (`docker/docker-compose.yml`, robocasa `volumes`), not `core_ws`; the G1 overlay adds `leprechaun_g1` and `g1_locomotion/policy` (`docker/docker-compose.g1.yml`). So simulator code cannot import `g1_description` or a new shared package without a compose change. The Isaac container is separate again.

## 1. Helper families

### 1.1 DDS channel init and domain handling

| Where | path:line | What |
|---|---|---|
| H1-2 controller | `H12C/utility/controller_config.py:352-361`, `364-366` (`dds_domain_id`), `369-375`, `378-393` | `_channel_factory_settings`, `init_channel_factory`, `init_channel_factory_guard` (interactive `input()` confirm on domain 0) |
| G1 controller | `G1C/utility/controller_config.py:295-308`, `311-317`, `320-335` | same three; settings also read `G1_DDS_INTERFACE` |
| H1-2 safety | `H12S/core/config.py:107-118`, `121-123`, `126-132`, `135-150` | same |
| G1 safety | `G1S/core/config.py:146-167`, `170-173`, `176-182`, `185-200` | same; settings also read `G1_DDS_INTERFACE` and refuse a `robot: sim` config on domain 0 |
| G1 locomotion | `G1L/config.py:155-179` | own `init_channel_factory(config)`: `ROS_DOMAIN_ID` + `G1_DDS_INTERFACE`; refuses domain 0 unless the config is for it; returns the domain; no prompt |
| G1 hands | `core_ws/src/g1/g1_dex3/g1_dex3/hand_node.py:67-79`, `dex3.py:117-146` (`dds_domain`) | domain from env or parameter, must equal the node's ROS domain; same `if interface:` branch |
| inline copies | `G1S/utility/record_limits.py:138-140`, `H12S/utility/record_limits.py:138-140`, `H12S/unsafe/send_fixed_positions.py:103-105`, `H12S/unsafe/send_random_positions.py:101-103` | the 4-line `if interface: ... else: ...` repeated |
| sims / infra | `docker/scripts/wait_for_sim.py:26-33`, `SIM/unitree_interface.py:21-22,59`, `SIM/g1/unitree_interface.py:198-210,252-258`, `SIM/mujoco_ros_bridge.py:159-179`, `CL_isaaclab_sim/sim_main.py:45-48`, `ISAAC/dds/common/dds_master.py:60-62`, `ISAAC/sensors/magpie_bridge.py:62-64`, `ISAAC/sensors/ros_sensor_bridge.py:110-113`, `ISAAC/dds/common/transport.py:18-30`, `docker/scripts/docker_common.sh:31-35`, `docker/scripts/runtime_common.sh:4-9`, `core_ws/src/h12_bringup/launch/lmg_sim_bringup.launch.py:39-42` | each re-parses `ROS_DOMAIN_ID` and re-checks "domain 0 is the real robot" |

Evidence (computed): `init_channel_factory` (7 loc) and `init_channel_factory_guard` (16 loc) ratio **1.00** between every pair of the four. `_channel_factory_settings`: `H12C` vs `H12S` 0.82, vs `G1C` 0.75, vs `G1S` 0.50. The two controllers' guard has 31 call sites (25 in `h12_ros2_controller` examples/benchmarks/plots, 6 in `g1_control`; `grep init_channel_factory_guard(`), so every change to its signature fans out.

Class: **A** for the two identical functions, **B** for `_channel_factory_settings` and `G1L/config.py`, **C** for `dex3.dds_domain`, and the sim/infra parsers are **C/D** (different languages, different containers).

What differs:
- Robot-specific: `G1_DDS_INTERFACE` env override (G1 laptop, wired port); H1-2 takes `network.interface` from YAML only.
- Accidental divergence: G1 safety refuses a sim config on domain 0 (`G1S/core/config.py:156-162`); the H1-2 safety layer has relaxed sim configs and no such refusal. The G1 has five different "domain 0 is the real robot" policies (controller prompt, safety refusal + prompt, locomotion refusal, dex3 must-equal-ROS, `wait_for_sim` 1..232); the H1-2 RoboCasa sim only checks `> 0` while Isaac, `wait_for_sim` and the shell wrappers check `1..232`. `SIM/unitree_interface.py:21` does `int(os.getenv("ROS_DOMAIN_ID"))` before its own assert, so an unset variable raises `TypeError`, not the friendly message.

Consolidation: new in-tree ament_python package (e.g. `golem_unitree_common`) with `channel_settings(config, interface_env=None)`, `init_channel_factory`, `guard_domain_zero`, and the `if interface` branch. Each robot passes its env var name; keep the sim/infra parsers as they are (they run in containers without `core_ws`). Effort S + M adoption. Risk: both H1-2 modules are submodules (`h12_ros2_controller`, `h12_safety_layer`): do the G1 side first, then PRs to the submodules. The guard runs at start-up before a real-robot DDS join, so a bug there is visible but not unsafe; behaviour must stay identical. Guarded by `h12_ros2_controller/test/test_controller_config.py`, `g1_safety_layer/test/test_g1_safety.py`, `g1_dex3/test/test_dex3.py`, `g1_locomotion/test/test_locomotion.py:233` (monkeypatches the factory), `tests/docker/test_wait_for_sim.py`.

### 1.2 YAML/path/config helpers

| Helper | Locations | Evidence |
|---|---|---|
| `_ensure_yaml_extension` (5 loc) | `G1C/utility/controller_config.py:24-28`, `H12C/utility/controller_config.py:19-23`, `G1S/ros2/safety_node.py:21-25`, `G1S/script/safety_layer_main.py:21-25`, `H12S/ros2/safety_node.py:24-28`, `H12S/script/safety_layer_main.py:16-20` | all six ratio 1.00 |
| `_resolve_config_path` | the same six files (`controller_config.py:30-38`/`25-33`, `safety_node.py:27-35`/`30-38`, `safety_layer_main.py:27-34`/`22-29`) plus `G1L/config.py:141-152` (`resolve_config_path`) | three flavours: base dir = `REPO_ROOT` (controllers), `get_package_share_directory` (safety nodes), `PROJECT_ROOT` (safety `*_main`), and G1L's share-dir variant; ratio 0.57-0.62 between flavours |
| `_load_yaml` | `G1C/utility/controller_config.py:40-47`, `H12C/utility/controller_config.py:35-42` (ratio 1.00, `None` -> `{}`); `G1S/core/config.py:30-36`, `H12S/core/config.py:21-27` (ratio 1.00 with each other, `None` -> `ValueError`) | the controller and safety loaders differ on empty files (intentional or not, not stated) |
| `_load_joint_config`, `_derive_q_limits`, `_process_clip_limits` | `G1S/core/config.py:39-50,61-73,76-103` vs `H12S/core/config.py:30-41,44-56,59-86` | ratio 1.00 each |
| "positive finite number" validators | `G1L/config.py:35-40` (`_positive`, rejects bool), `G1S/core/config.py:53-58` (`_positive_finite`), `core_ws/src/g1/g1_dex3/g1_dex3/driver.py:34` (`_check_value`), `SIM/g1/unitree_interface.py:70-81` (`env_seconds`) | four independent validators inside the G1 stack/sim alone; strictness differs (bool, NaN) |

Class: A for the first row and the three safety-config functions; B for `_resolve_config_path`; C for the validators.
Proposal: same shared package as 1.1 (`config_paths.py`: `ensure_yaml`, `resolve(name, base)`, `load_mapping(path, allow_empty)`), the validators as `require_positive(value, name)`. S. The safety-config functions belong to the safety-layer agent's finding.
Risk: low; the loaders run at start-up and fail loudly. Guarded by the per-package config tests.

### 1.3 Unitree CRC, `LowCmd_`/`LowState_` construction and parsing

CRC set/verify sites (`from unitree_sdk2py.utils.crc import CRC`; all call the SDK's per-process `CRC.Crc`):
- Sign on publish: `H12C/core/channel_interface.py:206-208` (obj at `:281`), `G1C/core/channel_interface.py:173-175` (obj at `:245`), `H12S/core/safety_layer.py:439-442`, `G1S/core/safety_layer.py:675-678`, `G1L/driver.py:66,141-142`, `H12S/unsafe/send_fixed_positions.py:110,119-120`, `H12S/unsafe/send_random_positions.py:108,118-119`, `SIM/g1/unitree_interface.py:455` (signs `rt/lowstate`), `ISAAC/dds/specialized/h12_robot_dds.py:107` (signs `rt/lowstate`).
- Verify on receive: `SIM/g1/unitree_interface.py:162-170,297,314` and `ISAAC/dds/specialized/h12_robot_dds.py:132-134` only. **`SIM/unitree_interface.py` (H1-2 RoboCasa) neither signs nor verifies** (no `crc` anywhere in the 175-line file). No safety layer verifies inbound CRC either (`grep "\.crc\|Crc("` finds only outbound signing).
- Speed-up: `SIM/g1/unitree_interface.py:127-159` `fast_crc()` replaces the SDK's Python word loop (`__Trans`) with a numpy view; the docstring measures 33 us instead of 145 us per frame, and the result is self-checked against the SDK on sample frames; tested by `tests/robocasa/g1/test_g1_unitree_interface.py:120-145`. The safety layers sign every outbound frame at `publish_hz: 500` (`core_ws/src/g1/g1_safety_layer/config/*.yaml`, `core_ws/src/h12_safety_layer/config/default_safety_split.yaml:14`) with the slow path.

`LowCmd_` defaults and header:
| Where | Header handling |
|---|---|
| `H12C/core/channel_interface.py:279-285` | `mode_pr = 0`, `mode_machine = 6` hard-coded |
| `H12L/scripts/lowerbody_controller_node.py:70-73,338-349` | `MODE_PR = 0`, `MODE_MACHINE = 6` hard-coded; comment: the robot drops any `LowCmd` whose `mode_machine` differs |
| `G1C/core/channel_interface.py:243-251` | `mode_pr = 0`, `mode_machine = 0`; "g1_safety_layer stamps" them |
| `G1S/core/safety_layer.py:516-522,634` | stamps `MODE_PR` and echoes the measured `rt/lowstate` `mode_machine` |
| `G1L/driver.py:57-60` | `mode_pr = MODE_PR` from `g1_description.layout` |
| `H12S/core/safety_layer.py:268-280` | merged command copies the last command's header |

Differences: the H1-2 stamps a constant (6), the G1 echoes the measured value; robot-specific, keep. The controller-side CRC (`H12C:206`, `G1C:173`, `G1L/driver.py:142`) is dead work: the safety layers recompute it at `H12S:441`/`G1S:677`, and the H1-2 lower-body nodes (`unitree_hg.msg.LowCmd` over ROS, no CRC at all) show the safety layer does not need it.

`LowState_` -> numpy parsing (same field walk, different array sets):
`H12C/core/channel_interface.py:86-94` vs `G1C/core/channel_interface.py:49-57` (the same nine assignments); `H12S/core/safety_layer.py:412-420,502-533` vs `G1S/core/safety_layer.py:630-657,756-787`; `H12S/core/safety_checks.py:80-82` vs `G1S/core/safety_checks.py:149-151`; `H12S/utility/record_limits.py:33-39` vs `G1S/utility/record_limits.py:33-39` (ratio 0.99); lower body `H12L/scripts/fame_node.py:153-154`, `lowerbody_controller_node.py:388-389`, `walking_node.py:141-143`, `sim2real_gap_monitor.py:207`, `core_ws/src/lmg_deploy/lmg_deploy/marl_deploy_node.py:407`; `G1L/locomotion.py:100-115` (`read_state`); `core_ws/src/g1/g1_dex3/g1_dex3/driver.py:73-75` (hands).
Note the H1-2 lower-body nodes read the ROS type `unitree_hg.msg.LowState`, the G1 side reads the SDK type `unitree_hg.msg.dds_.LowState_`. Both expose the same field names (`motor_state[i].q`, ...), so a duck-typed `motor_arrays(msg, slots)` helper would serve both.

Class: C for the fast-CRC vs plain CRC (same job, one optimised), A/B for the two `channel_interface` parsers, D for the headers.
Proposal: put `fast_crc()` (with its self-check) and a `lowstate_arrays(msg, slots)` in the shared package of 1.1; it is a performance change for the safety layers, not a deduplication, and monkeypatches SDK-private names (`_CRC__Trans`, `_crc_ctypes`), so pin the SDK version and keep the self-check. M. Risk: medium for the H1-2 real robot (500 Hz command path), so do it G1-first.

### 1.4 Latest-only DDS readers (potential bug)

- `H12C/core/latest_reader.py:23-65` (`LatestReader`, `sdk_participant` at `:34-44` reaching into `ChannelFactory()._ChannelFactory__participant`, `TickLagTracker` at `:67`); used at `H12C/core/channel_interface.py:56-62,310-317`.
- `H12S/core/latest_reader.py:11-46` (class ratio 0.88 vs the controller's: same `LATEST_ONLY_QOS`, `TAKE_BATCH`, `take_latest`); used at `H12S/core/safety_layer.py:92,195-210,244-258,302`.
- `G1L/driver.py:25-27,69-77,102-130`: a third version (`NEWEST_ONLY` QoS + `WaitSet`) with the rationale spelled out at `:43-47`: "The SDK's subscribers decode every sample in the DDS receive thread, under the GIL, so a node that falls behind would first have to decode its backlog".
- **Not applied** to `G1C/core/channel_interface.py:40-42` (`ChannelSubscriber(...).Init(handler, 10)`) nor to `G1S/core/safety_layer.py:156-158,182,254-269` (callback subscribers for the state and all command inputs).

Why it matters: the comment at `H12C/core/latest_reader.py:13-18` (and `H12S/core/latest_reader.py:12-17`) says a Python listener runs on the DDS receive thread and needs the GIL, and a receive thread waiting on the GIL leaves packets queued in the socket, which "shows up as state that is seconds old" / "a standing command delay of seconds". The commits that introduced the H1-2 readers are `caf318e` (`h12_ros2_controller`, "Read rt/lowstate latest-only; stop motion on stale state") and `e0396fc` (`h12_safety_layer`, "Relay newest-only inputs from one loop; estop on late low_state"), both 2026-10-12. The G1 files were added on 2026-10-26 (`G1S` in `a7fca29`, `G1C` in `ccec4c0`) and lack it; my inference is that they forked from a baseline older than 2026-10-12 (the same cause as section 6.1). `G1C` also has no counterpart of the H1-2 `StaleStateError`/`check_state_fresh` guard (`H12C/core/controller/upper_controller.py:34,291`, caught at `H12C/ros2/frame_task_server.py:168-174`); `G1S` does have its own state-staleness estop (`state_stale_estop_seconds`). This is accidental divergence in the G1's real-time path; the G1 has not run on hardware, so nothing has hurt yet.

Proposal: one `LatestReader` (+ `sdk_participant`) in the shared package; the G1 controller and safety layer adopt it; `G1L/driver.py` keeps its `WaitSet` variant or moves to the shared one. M. Risk: medium (changes G1 control-loop timing semantics; for H1-2 only the import path changes). Tests: `core_ws/src/h12_ros2_controller/test/test_state_subscriber.py:39,78,107` (mocks `LatestReader`), `g1_safety_layer/test/test_*_dds.py`, `g1_locomotion/test/test_locomotion_dds.py`.

### 1.5 Motor-slot and joint-name tables, header constants

G1 (centralised): `core_ws/src/g1/g1_description/g1_description/joints.py:19-60` (generated, `Joint(name, slot, lower, upper, velocity, effort)`), `layout.py` (slots, `MODE_PR`, `ARM_SDK_*`), consumed by `G1C/utility/joint_definition.py:1-13`, `G1C/utility/joint_limits.py:3-11`, `G1S/core/joint_limits.py:3-20`, `G1L/locomotion.py:14,82-85`, `g1_dex3/dex3.py:10-11`, `g1_dex3/driver.py:13`, `G1L/driver.py:18`.
G1 leftovers: `SIM/g1/sim_names.py:23-35` hard-codes `BODY_JOINTS`/`HAND_JOINTS` (comment: "match g1_description"; the robocasa container cannot import it); `SIM/g1/unitree_interface.py:58-67` re-declares `MODE_MACHINE = 5`, `MODE_PR`, `ARM_SDK_SLOTS = range(12, 29)`, `ARM_SDK_WEIGHT_SLOT = 29`; `G1S/core/config.py:239` defaults `engage.mode_machine` to 5; `docker/scripts/wait_for_sim.py:13` has `{'h12': 0, 'g1': 5}`; `core_ws/src/g1/g1_locomotion/policy/g1.yaml:16` lists names. I checked by executing: `SIM/g1/sim_names.BODY_JOINTS` equals `g1_description.joints.BODY` order, and the hand order matches; **no test compares them** (the only test using `sim_names` is `tests/robocasa/g1/test_g1_unitree_interface.py`, which does not import `g1_description`).
H1-2 (not centralised):
- `H12C/utility/joint_definition.py:19,71,102,132` (four lists of names), `H12C/utility/joint_limits.py:9-163` (limit tables), `H12S/core/joint_limits.py:6-120`: **the position (27 dicts), velocity (27) and torque (27) limit tables are element-for-element equal** (executed both; `H12S` uses `URDF_*_LIMITS`, `H12C` `JOINT_*_LIMITS`).
- `SIM/sim_names.py:19-21`, `ISAAC/control/motor_contract.py:5-9` (names built from a comprehension; `HAND_JOINT_NAMES` too), `ISAAC/envs/common/robots.py:40`, `CL_isaaclab_sim/tasks/common_config/robot_configs.py:29`, `h12_lowerbody_rl/rma/env_factor_spec.py`, `core_ws/src/lmg_deploy/config/policy_template.yaml:14` and the policy YAMLs, `H12C/core/controller/upper_controller.py`, ~17 YAML configs under `core_ws/src/h12_ros2_controller/config/` and `core_ws/src/h12_safety_layer/config/` (names in comments and `joint_names`-like keys).
- `MODE_MACHINE = 6` appears twice (`H12C/core/channel_interface.py:284`, `H12L/scripts/lowerbody_controller_node.py:73`), `MOTOR_MODE_PR = 1` four times (`fame_node.py:42`, `lowerbody_controller_node.py:70`, `walking_node.py:30`, `core_ws/src/lmg_deploy/lmg_deploy/marl_deploy_node.py:137`).

Class: A for the two limit tables; C for the name lists; D for G1 `mode_machine` 5 vs H1-2 6.
Proposal: an H1-2 analogue of `g1_description` (names + limits + `MODE_*`), generated from the H1-2 URDF as the G1's is. Candidate home: `h12_ros2_model` (submodule, description package) or a new in-tree package; `H12C` and `H12S` would import the limits. For the sim, keep a copy and add a sync test (as the sim cannot import it). M. Risk: medium, the numbers are the H1-2 safety limits; a test that asserts the new table equals the old two protects the migration. Crosses two submodules.

### 1.6 Quaternion / rotation math

H1-2 side (all independent hand-rolled code, listed by job):
- projected gravity: `H12L/policy.py:69-79` (`gravity_from_quat_walk`, closed form), `:82-99` (`quat_rotate_inverse`), `:102-103` (`gravity_from_quat_fame`); `H12L/scripts/walking_node.py:33-43` (`get_gravity_orientation`, ratio 1.00 with `H12L/reference/deploy_mujoco.py:11-23`, same formula as `gravity_from_quat_walk`); `H12L/scripts/sim2real_gap_monitor.py:58-60` (`_gravity_x`, one term of it); `core_ws/src/lmg_deploy/lmg_deploy/rotations.py:92-94` (`projected_gravity` via `quat_apply_inverse :39-41`); reference scripts `H12L/reference/mujoco_deploy_h12_rma.py:125-142,245-262`, `reference/standalone_fame_test.py:32-35`.
- Hamilton product: `H12L/policy.py:106-118` and `lmg_deploy/rotations.py:12-20` are the same four-line expression (the first adds `dtype=np.float32`); also `imu_offset_quat` `H12L/policy.py:121-139` vs `imu_mount_quat` `lmg_deploy/rotations.py:61-70` (same mounting-bias rotation; the latter's docstring points at the former).
- matrix <-> quaternion: `h12_skills/perception_utils.py:69-78,81-99,58-67,110-127`, `model_server/graspgen_server.py:169-187` (`mat_to_quat`, identical to `perception_utils`), `h12_slam/robot_geometry.py:16-76` (`rpy_to_matrix`, `quat_to_matrix`, `matrix_to_quat`, `transform_to_rt`, `rotation_aligning`), `lmg_deploy/rotations.py:73-89` (`quat_from_matrix`), `lmg_deploy/livox_imu.py:88-100` (`_rotation_aligning`, code-identical to `h12_slam/robot_geometry.py:62-76` apart from entry normalisation, funcdups ratio 0.91), `lmg_deploy/livox_attitude.py:37-46` (quaternion version), `cl_realsense/.../point_cloud_accumulator.py:97-113` (inline matrix, in a file that already imports scipy `Rotation` at `:2`), `h12_bringup/h12_bringup/livox_imu_upright.py:30-60`, `H12C/ros2/utility.py:18-27` (via `pyquaternion`), `h12_skills/h12_skills/run_telemetry.py:82-86` (yaw), `h12_skills/skills/grasp.py:1658-1680` (`_pose_quat`, `_quat_angle`, `_quat_slerp`), `H12C/plot/repeatability_report_viewer.py:72-80`, four inline roll terms in `H12C/core/controller/counter_balance/*.py` and `zmp_controller.py:352`.
- sims: `SIM/realsense_profile.py:43-60`, `SIM/sim_common.py:187-193` (inline yaw), `ISAAC/sensors/ros_sensor_bridge.py:71-82` (inline matrix), `CL_isaaclab_sim/tasks/common_observations/h12_27dof_state.py:71-111` (torch batch), `SIM/h1_2_robosuite.py:439-459` vs `SIM/g1/g1_robosuite.py:94-115` (`_euler_to_wxyz`, ratio 0.93, differs only in a docstring sentence).
- `tools/ros_mcp_server.py:92` (the upright term).
G1 side: `G1L/locomotion.py:143-146` (`gravity_z`, a one-line restatement of `homie_policy.get_gravity_orientation` z; tested by `g1_locomotion/test/test_locomotion.py:289`), `SIM/g1/unitree_interface.py:190-195` (`rpy_from_quat`), and `scipy.spatial.transform.Rotation` in `g1_navigation` (`frames.py:74,98,143`, `level_frame.py:33-41`, `body_to_pelvis.py:20-29,65`, `livox_to_pointcloud.py:44,142`). `homie_policy.py:48-75` is vendored; keep it in `policy/`.

Numeric check (1000 random unit quaternions, executed): `homie_policy.quat_rotate_inverse` vs `H12L.policy.quat_rotate_inverse` max abs difference 1.1e-7 (the float32 cast), `gravity_from_quat_walk` vs HOMIE's `get_gravity_orientation` 3.0e-8, `gravity_z` vs the HOMIE z component 5.6e-17. So all of these projected-gravity forms are interchangeable for unit quaternions. The textual match of `homie_policy.quat_rotate_inverse` (`:48-69`) with `H12L/policy.py:82-99` is a rename (`q_conj` -> `qc`) of the same legged_gym-style expressions (class B, token ratio 0.89 after renaming; origin shared upstream, not the G1 port).

Transform/matrix helpers for ROS messages (same job, 7 implementations): `core_ws/src/g1/g1_navigation/g1_navigation/body_to_pelvis.py:24-29` and `livox_to_pointcloud.py:136-144` (two `to_matrix` copies inside the G1 package, ratio 0.55 only because one calls `frames.transform`), `h12_slam/robot_geometry.py:55-59`, `cl_realsense:97-113`, `h12_skills/perception_utils.py:58-67,110-127`, `G1C/ros2/utility.py:18-45` == `H12C/ros2/utility.py:18-45` (raw diff 0 lines of 89).

Class: A/B within H1-2 (`mat_to_quat` x2, `quat_mul` x2, `_rotation_aligning` x2, gravity x4), C across the rest, D for the G1/H1-2 split (the G1 side has almost nothing to merge).
Proposal: no cross-robot move. If the H1-2 maintainers want it, a small `quat.py` (wxyz and xyzw variants, `gravity_in_body`, `mat_to_quat`, `rotation_aligning`) in the shared package, built on `scipy` as the G1 does. M. Risk: medium (convention mix: `wxyz` in policies, `xyzw` in ROS), covered by `lmg_deploy/test/test_controller.py`, `h12_slam/test/test_robot_geometry.py`, `h12_lowerbody_rl` policy tests.

### 1.7 PointCloud2 / Livox CustomMsg builders, TF broadcast helpers

- `PointCloud2` hand-built from a packed array: `SIM/h12_sensors.py:629-641` (RealSense cloud; fields `:132-135`) and `:723-738` (lidar; fields `:78-84`), `ISAAC/sensors/ros_sensor_bridge.py:182-190`, `core_ws/src/g1/g1_navigation/g1_navigation/livox_to_pointcloud.py:48-49,106-143`, `core_ws/src/h12_slam/h12_slam/self_filter.py:123-132`, `cl_realsense/.../point_cloud_accumulator.py:188,200-206`, `core_ws/src/h12_skills/h12_skills/base.py:978-983`. Three of them use the `array.array('B', ...)` trick to avoid per-byte validation (`h12_sensors.py:640,737`, `livox_to_pointcloud.py:134`). Class C; each has different fields and a different container.
- Livox `CustomMsg` serialisation: `SIM/h12_sensors.py:88-124` (rclpy prefix + packed points), `SIM/g1/g1_sensor_msgs.py:70-122` (hand-rolled CDR), `ISAAC/sensors/livox_dds.py:18-60` (CycloneDDS IDL). Class C (three simulators, three approaches to the same 20-byte-per-point layout); all three live in containers that cannot share code.
- `TransformStamped` field fills with no helper: `SIM/mujoco_ros_bridge.py:132-143`, `SIM/h12_sensors.py:365-376`, `SIM/g1/g1_sensors.py:326-333`, `h12_skills/base.py:571,588` + `skills/grasp.py:1744-1755`, `h12_slam/level_odom.py:184-227`, `g1_navigation/body_to_pelvis.py:59-66`, `level_frame.py:69-83`. Class D: 8 sites of 5-12 lines, each different.
- Sim time -> `builtin_interfaces/Time`: `SIM/mujoco_ros_bridge.py:31-39` (truncates), `SIM/g1/g1_sensor_msgs.py:63-67` (rounds to ns), `ISAAC/sensors/ros_sensor_bridge.py:47-51` (`stamp_parts`, rounds and validates). The G1 sim uses both of the first two (`g1_sensors.py:81,327` for the static TF and `g1_sensor_msgs.time_msg` for payloads).
Proposal: none across robots (container boundaries). Optional: one `make_transform(parent, child, xyz, quat_xyzw, stamp)` in the shared package for the in-ROS-stack sites. S, negligible risk, saves ~40 lines.

### 1.8 Logging throttles, pacing, clocks, executors

- Throttled logging: no shared helper anywhere. Nodes use rclpy's `throttle_duration_sec` directly (`grep`: 25 uses in 9 files). Hand-rolled wall-clock throttles are all in the G1 sim: `SIM/g1/unitree_interface.py:93,283-296` (`_warn`, with repeat counts), `SIM/g1/g1_sensors.py:94,213,225,383-387` (`WARN_PERIOD_S`, `_warn_once`), plus a `_log` printer in each (`unitree_interface.py:121`, `g1_sensors.py:98`). `h12_lowerbody_rl/policy_manager.py:42,150` counts ticks. Class C, D.
- Real-time pacing: `ISAAC/control/pacing.py:6-29` `RealtimePacer` is **byte-identical** to `SIM/sim_timing.py:7-30` (class text compared, equal; the only file-level difference is the extra `import threading`). The G1 sim re-implements pacing inline (`SIM/g1/g1_mujoco.py:37,296-305`) and does not use `RealtimePacer` (`sim_timing` is the H1-2's only, CLAUDE.md §8). Class A (Isaac vs RoboCasa) and C (G1). Kept separate for the container reason; the simulator agent owns it.
- Watchdog clock: `G1S/core/clock.py:1-26` (`CLOCK_BOOTTIME` with fallback), used by the G1 safety layer only; `H12S` and `H12C` use plain wall/monotonic time (not read in detail). Related: `G1L/locomotion.py:117-` `SourceClock` and `H12C/core/latest_reader.py:67-` `TickLagTracker` both estimate DDS delivery lag, one from the DDS source timestamp, one from the robot tick (class C).
- `rclpy.init/spin/KeyboardInterrupt/destroy_node/shutdown` `main()` boilerplate: 48 files call `rclpy.init(` (20 packages incl. both sims and `tools/`); `grep "except KeyboardInterrupt"` finds 58. Five lines each; keep (section 5).
- Launch helpers: `_include()` is defined 6 times in `h12_bringup/launch/*.launch.py` and `h12_slam/launch/h12_real_slam.launch.py:19-24` (ratios 0.91-1.00 on 5-7 loc); `_as_bool` twice in G1 (`g1_bringup/launch/g1_core.launch.py:94`, `g1_navigation/launch/g1_navigation.launch.py:103`, identical). Trivial; the bringup agent owns this.

### 1.9 Near-identical files and functions, for the other agents (not analysed further here)

Pairs found by the whole-tree file/function scan (line-normalised ratio; "raw" = plain `diff`):

| G1 file | H1-2 file | ratio / raw diff |
|---|---|---|
| `G1C/ros2/utility.py` | `H12C/ros2/utility.py` | 1.00, raw 0 of 89 |
| `G1C/utility/view_urdf.py` | `H12C/utility/view_urdf.py` | raw 0 |
| `G1S/utility/record_limits.py` | `H12S/utility/record_limits.py` | 0.99, raw 2 (import path) |
| `G1S/core/chunk_logger.py` | `H12S/core/chunk_logger.py` | 0.99, raw 4 (`console` vs `print`) |
| `G1C/utility/topic_subscriber.py` | `H12C/utility/topic_subscriber.py` | 0.98, raw 2 (import path) |
| `G1C/ros2/dual_arm_client.py` / `frame_task_client.py` | H12C same names | 0.98 / 0.97, raw 7 / 10 (H1-2 has `slow_mode`) |
| `G1S/utility/combine_chunks.py`, `publisher.py`, `subscriber.py` | H12S same names | 0.95, 0.92, 0.92 (raw 4 each) |
| `G1C/core/controller/arm_controller.py`, `ros2/dual_arm_server.py`, `core/ik_solver.py`, `core/controller/frame_controller.py` | H12C same | 0.96, 0.95, 0.91, 0.91 |
| `G1C/core/planner/reduced_joint_planner.py`, `ros2/frame_task_server.py`, `core/robot_model.py`, `utility/controller_config.py`, `core/channel_interface.py`, `core/controller/upper_controller.py` | H12C same | 0.87, 0.81, 0.70, 0.72, 0.68, 0.62 |
| `G1S/core/config.py` | `H12S/core/config.py` | 0.73 |
| `SIM/g1/g1_robosuite.py` | `SIM/h1_2_robosuite.py` | 0.50 as files; function pairs `place_robot_freejoint`, `place_robot_clear` 1.00, `make_kitchen_env` 0.94, `_robot_env_contacts` 0.97, `_backward_dir` 0.95, `install_robocasa_patches` 0.84 |
| `tests/robocasa/g1/test_g1_spawn_collision.py` | `SIM/test_spawn_collision.py` | 0.72 |
| `core_ws/src/h12_skills/h12_skills/model_logging.py` | `core_ws/src/model_server/model_server/model_logging.py` | raw 0 of 223 |

Aggregate for the controller pair (same-name functions of at least 5 lines with ratio >= 0.6): **270 pairs, 202 byte-identical, 21 between 0.90 and 0.99** (`G1C`: 42 files, 6,884 non-blank lines; `H12C`: 110 files, 23,381). For the safety pair: 36 pairs, 27 identical. `G1C` is a fork of `H12C`, not an independent implementation.

### 1.10 Test-harness helpers inside the G1 stack

`TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)` is declared separately in nine G1 test modules (`g1_dex3/test/test_driver_dds.py:22`, `test_hand_node.py:19`, `g1_locomotion/test/test_locomotion_node.py:19`, `test_locomotion_dds.py:46`, `g1_control/test/test_controller_loop_dds.py:55`, `g1_navigation/test/stack.py:18`, `g1_safety_layer/test/test_hardening_dds.py:31`, `test_arm_sdk_mode_dds.py:22`, `test_node_exit.py:34`, `test_lowcmd_modes_dds.py:29`); `_wait_for`/`_wait_until` is rewritten in `g1_dex3/test/test_driver_dds.py:104`, `g1_locomotion/test/test_locomotion_dds.py:249`, `g1_safety_layer/test/test_node_exit.py:119,191`. A shared test fixture (or the shared package's `TEST_DOMAIN`) would remove them; S, no runtime risk.

## 2. Inventory of the packages no other agent covers

### 2.1 `model_server` (in-tree, not a submodule; ament_python; `setup.py` entry points `gemini_server`, `sam_server`, `graspgen_server`, `yolo_server`)

| Node (entry) | Serves / publishes / subscribes | Params / env / files |
|---|---|---|
| `gemini_server` (`gemini_server.py:122`, `RunGeminiServer :288`) | service `gemini_query` (`custom_ros_messages/srv/GeminiQuery`: `CompressedImage image`, `string prompt` -> `response`, `success`, `message`); reentrant group, `MultiThreadedExecutor` | `enable_logging`, `enable_visualization`, `clear_logs` (from `model_logging`); env `GEMINI_API_KEY` or `GOOGLE_API_KEY`; constants for model/timeout/retries in the file |
| `sam_server` (`sam_server.py:140`, `RunSamServer :248`) | service `sam_segment` (`SamSegment`: image + text and/or xyxy box exemplars -> mono8 `mask`, `score`); SAM3 loaded eagerly | weights `model_server/weights/sam3.pt` (or `MODEL_SERVER_WEIGHTS_DIR`); logging params |
| `yolo_server` (`yolo_server.py:243`, `RunYoloServer :515`) | subscribes `CompressedImage` on each of `image_topics` (default `/realsense/{head,left_hand,right_hand}/color/image_raw/compressed`), `.../aligned_depth_to_color/image_raw/compressedDepth`, `.../color/camera_info`; publishes `custom_ros_messages/DetectionBundle` on `<image_topic>/detections` and an annotated `Image` (5 Hz default) | params `image_topics`, `detection_topics`, `queries` (live), `conf_threshold` (0.90), `nms_iou` (0.01), `publish_rate_hz`; weights `weights/yolo_world_battery_best.pt` |
| `graspgen_server` (`graspgen_server.py:207`, `main :701`) | service `graspgen` (`GraspGen`: object + scene `PointCloud2`, `gripper_name`, `arm` -> ranked `PoseStamped[]`, `scores`, `gripper_width`); publishes latched `grasp_cloud`, `scene_cloud` (`PointCloud2`); viser web GUI on port 8080 | env `GRASPGENX_CHECKPOINT_DIR`, `GRASPGENX_ASSETS_DIR`, `GRASPGEN_GRASPS_FILE`; `DEFAULT_GRIPPER = 'magpie'` (`:58`) |

Config: no YAML; all defaults are module constants plus ROS params. Launched only from `core_ws/src/h12_bringup/launch/h12_models.launch.py` (gemini/sam always; `yolo_server` behind `use_yolo`; `graspgen_server` and `h12_skills` behind `use_skills`). Mac builds stub it out (`docker/mac/scripts/launch_ros_mac.sh:126`). Consumers: `h12_skills/base.py:41-44` (`GeminiQuery`, `SamSegment`, `GraspGen`, `DetectionBundle`), `h12_skills/skills/grasp.py:53` (`Detection`). `model_logging.py` is the duplicate in 1.9. G1: no G1 node, bringup or test uses it.

### 2.2 `magpie_control` (submodule, `core_ws/src/magpie_control`, branch `main`; gitlink `f215020`)

Entry points (`setup.py`): `gripper_node`, `ft_sensor_node`, `tactile_sensor_node`. The package also contains a large lab library (`gripper.py` 1,204 lines, `ur5.py`, `poses.py`, `BT.py`, `realsense_*.py`, `utils.py`, ...) that GOLEM's runtime does not import other than `gripper.py` and `ft_sensor.py`.

| Node | Interfaces | Params |
|---|---|---|
| `gripper_node` (`src/magpie_control/gripper_node.py`) | services `gripper/open`, `close`, `calibrate`, `reset_parameters`, `reset_overload` (`std_srvs/Trigger`), `gripper/set_position` (`magpie_msgs/SetGripperPosition`), `gripper/set_force` (`SetGripperForce`); action server `gripper/deligrasp` (`DeliGrasp`); publisher `gripper/state` (`GripperState`, 10 Hz timer) | `auto_detect_port`, `port`, `use_eflesh`, `default_speed`, `default_torque`, `finger{1,2}theta_{min,max,90}` (`:33-47`); namespaced `left`/`right` by `h12_bringup/launch/h12_grippers.launch.py` (calibration constants live there) |
| `ft_sensor_node` (`ft_sensor_node.py`) | publisher `ft_sensor/wrench` (`geometry_msgs/WrenchStamped`), service `ft_sensor/zero` (`Trigger`); OptoForce over TCP | `ip_address` (192.168.0.5), `port`, `poll_rate` |
| `tactile_sensor_node` (`tactile_sensor_node.py`) | `tactile/pressure_map` (`Float32MultiArray`) created only when `enable`; the body is a stub ("not yet implemented") | `enable`, `auto_detect_ports` |

Used by the H1-2 only: the H1-2 real bringup (`h12_grippers.launch.py` via `h12_real_drivers` and `h12_safety_grippers`), `h12_skills/base.py:49-50`, `h12_bringup/scripts/slider_debugger.py:38`, and both H1-2 simulators emulate its surface (`SIM/magpie_hand_bridge.py`, `ISAAC/sensors/magpie_ros.py`, class C: ratio 0.06, 325 vs 229 stripped lines, same services and action).
Stale: `launch/gripper_control.launch.py` points at `config/gripper_config.yaml`, but the package has no `config/` directory (`ls` fails; `setup.py` globs `config/*.yaml` into nothing).
G1: not used. The G1's Dex3 hands are a different hardware path (`g1_dex3`, DDS `rt/dex3/*`, `SetHandPose`); see section 4.

### 2.3 `estop` (submodule, branch `main`; gitlink `c30fb50`; H1-2 hardware e-stop button over serial)

Entry points: `estop_node`, `estop_raw_dds_subscriber`, `estop_serial_status_test`; Arduino sketch in `arduino/`.
`estop_node` (`estop/estop_node.py:30-190`): publishes `estop/triggered` and `estop/plugged_in` (`std_msgs/Bool`, 100 Hz by default) and also a raw CycloneDDS sample on `h12/estop_status_raw` (IDL type name `h12.estop.EstopStatusRaw`: `triggered`, `plugged_in`, `sequence_id`, `stamp_ms`) written with `DomainParticipant()` (default domain). ~20 params (serial port/VID/PID candidates, `baud_rate`, debounce, stale timeouts, `fail_safe_triggered_on_disconnect`, `raw_dds_enabled`, `raw_dds_topic`). Launch: `launch/estop.launch.py`.
Consumers: `h12_safety_layer` reads only the raw DDS topic (`H12S/core/estop_subscriber.py:15-80`, which re-declares the same IDL struct; config key `estop_topic: h12/estop_status_raw`); per CLAUDE.md §6 the H1-2 real bringup starts it through `h12_control.launch.py` (`use_estop:=true`; I did not read that file). Not started in sim.
G1: `g1_safety_layer` has only a latched software estop and watchdogs (`config/*.yaml` `estop_response`, no hardware input). The package name, topic and IDL name are H1-2-flavoured but the content is robot-agnostic; a G1 hardware button would reuse the node unchanged and need an `EStopSubscriber` in `g1_safety_layer` (copy of `H12S/core/estop_subscriber.py`, or the shared package).

### 2.4 `cl_realsense` (submodule, branch `main`; gitlink `1f5d921`)

Entry points: `pc_acc` (`scripts/point_cloud_accumulator.py`) and `another_node` (`scripts/another_node.py`, a hello-world stub). Launch files: `h12_headcamera.launch.py`, `h12_hand_cameras.launch.py`, `h12_rs_cams.launch.py`, `ur16e_rs_cams.launch.py` (wrappers around `realsense2_camera` plus static TFs). `scripts/list_realsense_devices.py` is a helper.
`pc_acc` interfaces: subscribes `PointCloud2` on `/realsense/ee_cam/depth/color/points` (TF to `floor_link`, Open3D voxel/outlier filters), publishes `/realsense/accumulated_point_cloud`, serves `pointcloud_accumulator/{clear,get,save,load}_arm_pointcloud` and `start_acc`/`stop_acc` (`std_srvs/Trigger`, and `arpa_control/srv/GetPointCloud` for `get_arm_pointcloud`). **`from arpa_control.srv import GetPointCloud` (`:15`) cannot be satisfied: `arpa_control` is nowhere in the repo**, and no bringup launches `pc_acc` (`grep pc_acc` outside the package finds nothing). The earlier port spec also records it ("the G1 stack does not use cl_realsense; pc_acc imports the missing arpa_control", `docs/superpowers/specs/2026-10-26-g1-port/4_docker_infra_shared.md:323`).
Used by the H1-2 real bringup only (`h12_real_drivers.launch.py:60-64`, `h12_real_desktop_bringup.launch.py:47`). The G1 sim publishes `/realsense/head/*` from the simulator instead.

### 2.5 `custom_ros_messages` (submodule, gitlink `3f361bf` on local branch `g1_interfaces`; `CMakeLists.txt` lists every file below)

Every interface, who uses it (imports found by grep; `rosidl` code-generated). "Both" = used by G1 and H1-2 code.

Actions (19):
| Action | Users |
|---|---|
| `FrameTask` | server+clients: `H12C/ros2/frame_task_*`, `G1C/ros2/frame_task_*` (both); `h12_skills/base.py`, `h12_bringup/scripts/slider_debugger.py`; tests `tests/stack/check_frame_task.py`, `g1_control/test` |
| `NamedConfig` | `H12C`, `G1C` (both), `h12_skills/base.py`; tests `tests/g1/test_sim_bringup.py` |
| `DualArm` | `H12C/ros2/dual_arm_*`, `G1C/ros2/dual_arm_*` (both) |
| `SkillFrontierExplore` | `h12_skills` (served by the `SkillsNode`), `g1_navigation/frontier_explore.py:68` (both, "one robot stack per domain"); tests `tests/g1/test_sim_explore.py` |
| `SkillGrasp`, `SkillPickPlace` | `h12_skills` (implemented) |
| `SkillOpenDoor`, `SkillCloseDoor`, `SkillOpenLid`, `SkillCloseLid`, `SkillNavigate`, `SkillPress`, `SkillSlideRack`, `SkillTurnLever`, `SkillTwistKnob` | `h12_skills` (stubs) |
| `SkillRetrieveMemory` | nobody serves it |
| `HandAdaptiveForce`, `ParkerGotoPose`, `ScanBattery` | no user in this repo |

Messages (8): `DetectionBundle` (`model_server/yolo_server`, `h12_skills/base.py`), `Detection` (`yolo_server`, `grasp.py`), `StringArray` (`frame_task_server` of both controllers, topic `frame_names`; both); `Dummy`, `MotorCmd`, `MotorCmds`, `MotorState`, `MotorStates`: no user (the tests use `unitree_go` types of the same names).
Services (16): `GeminiQuery`, `SamSegment`, `GraspGen` (`model_server` + `h12_skills`), `LmgSetMode`, `LmgSetArmGoals` (`lmg_deploy/marl_deploy_node.py:113`), `SetHandPose` (`g1_dex3/hand_node.py:16`, G1 only); no user: `Query`, `ResetBeliefs`, `UpdateBeliefs`, `UpdateTrackedObject`, `SetHandAngles`, `UR16BehaviorTrigger`, `EthernetMotor`, `ParkerGetPoseTrigger`, `RemovePart`, `GoHome` (legacy interfaces of other lab projects).

Both robots use 5 interfaces: `FrameTask`, `NamedConfig`, `DualArm`, `StringArray`, `SkillFrontierExplore`. The G1-only addition is `SetHandPose`. Of 43 interfaces, 18 are used by H1-2 code only (7 msg/srv + 11 `Skill*`), 1 by G1 only, 5 by both, and 19 by nobody (5 msgs, 10 srvs, 4 actions). Gitlink note: see section 6, item 6.

### 2.6 `magpie_msgs` (submodule, gitlink `75a5267` on `origin/main`)

Used (all H1-2 only; the G1 uses none): `GripperState`, `SetGripperPosition`, `SetGripperForce`, `DeliGrasp` (`magpie_control/gripper_node.py:10-12`; `h12_skills/base.py:49-50`; both simulators' Magpie bridges, `SIM/magpie_hand_bridge.py:34-36`, `ISAAC/sensors/magpie_ros.py:16-18`; tests `tests/isaaclab/*`), `DeliGraspParams` (`gripper_node.py:11`). `h12_bringup/scripts/slider_debugger.py:38` uses `SetGripperPosition`. No user for: `ExecutePolicy`, `ExecuteTrajectory`, `ForceControl` (actions); `Episode`, `Observation`, `PolicyAction` (msgs); `CaptureFrame`, `DetectObjects`, `GetPose`, `MoveJoint`, `MoveLinear`, `SegmentObject`, `SetSpeed`, `StartEpisode`, `StartRecording`, `StopRecording` (srvs): 16 of 21.

### 2.7 Vendored third-party (inventory only, no findings)

| Package (path) | Submodule? | Package name(s) | What uses it |
|---|---|---|---|
| `core_ws/src/FAST_LIO` | no (vendored in the main repo; contains its own `.gitmodules`, `Log/`, `PCD/`) | `fast_lio` (`fastlio_mapping`) | H1-2: `h12_slam/launch/h12_navigation.launch.py:26-29` + `h12_slam/config/mid360.yaml`; G1: `g1_navigation/launch/g1_navigation.launch.py:186-188` + `config/g1_fastlio{,_sim}.yaml`; `exec_depend` in `h12_bringup`, `h12_slam`, `g1_bringup`, `g1_navigation`. Both robots keep their own derived parameter files (differences in `point_filter_num`, `filter_size_*`, `blind`, `extrinsic_*`: robot/sim-specific) |
| `core_ws/src/livox_ros_driver2` | yes, pinned `13eb05e` (CLAUDE.md §8: do not bump) | `livox_ros_driver2` (`livox_ros_driver2_node`; msg `CustomMsg`, `CustomPoint`) | H1-2 real: `h12_real_drivers.launch.py:36-37`, `h12_bringup/livox_imu_upright.py`; both sims publish `/livox/lidar` as `CustomMsg` (`SIM/h12_sensors.py:49`, `SIM/g1/g1_sensors.py:73`, Isaac via its own IDL `ISAAC/sensors/livox_dds.py`); G1 `g1_navigation` declares `<depend>livox_ros_driver2` and converts `CustomMsg` to `PointCloud2` (`livox_to_pointcloud.py:40`) |
| `core_ws/src/unitree_ros2` | yes (`v0.3.0-19`) | `unitree_hg`, `unitree_go`, `unitree_api`, `unitree_ros2_example` | ROS-typed `unitree_hg.msg.LowCmd/LowState`: `H12L/scripts/{fame_node,lowerbody_controller_node,walking_node,sim2real_gap_monitor}.py`, `lmg_deploy/marl_deploy_node.py:114`; `unitree_go` `MotorCmds/MotorStates` in `tests/isaaclab` and the Inspire-hand path; `h12_lowerbody_rl/package.xml` `exec_depend`. **No `g1_*` package imports any ROS `unitree_*` message**: the G1 talks DDS only through the SDK |
| `unitree_sdk2_python` | yes (`c158c8d`, `heads/master`) | `unitree_sdk2py` | see 2.8 |
| `mujoco_mpc`, `third_party/leprechaun_g1` | yes | not in scope | `h12_deploy_mjpc` (MJPC cores); `g1_description` and the G1 sim |

### 2.8 `unitree_sdk2_python` as used by both robots (first-party import survey, excluding tests)

Distinct API surface used: `core.channel` (`ChannelFactoryInitialize`, `ChannelPublisher`, `ChannelSubscriber`, plus the private `ChannelFactory()._ChannelFactory__participant` at `H12C/core/latest_reader.py:41`), `utils.crc.CRC` (9 import sites; the G1 sim patches its private `_CRC__Trans`), `idl.unitree_hg.msg.dds_` (`LowCmd_`, `LowState_`, `HandCmd_`, `HandState_`), `idl.default` constructors, `idl.unitree_go.msg.dds_` (`MotorCmds_`, `MotorStates_`: H1-2 Inspire hands, `H12C/core/channel_interface.py:11`, `ISAAC/dds/specialized/inspire_dds.py`), `idl.std_msgs`/`builtin_interfaces` (Isaac `commands_dds.py`, the safety-layer demo publisher/subscriber), `utils.thread.RecurrentThread` (H1-2 RoboCasa sim only, `SIM/unitree_interface.py:9`).
Files importing it (non-test): H1-2 side `H12C` 4, `H12S` 8, both sims (`SIM/unitree_interface.py`, `ISAAC/dds/**`, `ISAAC/sensors/livox_dds.py`); G1 side `G1C` 3, `G1S` 7, `G1L` 2, `g1_dex3` 2, `SIM/g1/unitree_interface.py`; shared: `docker/scripts/wait_for_sim.py`.
Unused by GOLEM although present in the SDK: `unitree_sdk2py.comm.motion_switcher` (`MotionSwitcherClient`), `g1.loco`, `g1.arm`, `g1.audio`, `rpc`. The G1's "release Unitree's controller" step is documented as a manual operator action that "nothing in this stack performs" (CLAUDE.md §6); the SDK's motion switcher is the obvious building block if that ever changes. Not a recommendation: it is a safety decision for the owner.
Pinning note: `docker/BaseDockerfile:81-84` installs the in-repo submodule (`c158c8d`); `docker/IsaacDockerfile:104-106` instead clones upstream and checks out `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5`, which is not an object in the local submodule. So the Isaac container may run a different SDK revision (CRC/IDL) than the other images.

### 2.9 `tools/` and `docker/scripts/*.py`

- `tools/ros_mcp_server.py` (316 lines): an HTTP MCP server run inside `golem_ros` for the Mac H1 sim. One warm `rclpy` node subscribing `/odom`, `/converted_scan`, `/clock`, `/global_costmap/costmap`, `/plan`, publishing `/cmd_vel`; tools `robot_status`, `costmap_summary`, `nav_status`, `scan_status`, `drive`, `wait_for`, `list_topics`, `echo_topic`, `topic_hz`, `call_service`, `node_list`. Env `GOLEM_ROS_MCP_PORT`, `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`. Robot-specific bits: the `uprightness`/`z` posture thresholds (`:92-97`) and `/odom` (ground-truth, opt-in in both sims). The rest is generic Nav2/ROS and would work against the G1 stack with different thresholds.
- `docker/scripts/wait_for_sim.py` (domain + `mode_machine` check, see 1.1/1.5) and `docker/scripts/package_humble_runtime.py` (packages a private Python/Humble runtime; no overlap).

## 3. What the G1 stack uses, and could use instead of its own code

Uses today: `custom_ros_messages` (5 shared interfaces + `SetHandPose`), `unitree_sdk2_python`, `FAST_LIO`, `livox_ros_driver2` (message type), `third_party/leprechaun_g1`. Uses none of: `model_server`, `magpie_control`, `magpie_msgs`, `estop`, `cl_realsense`, `unitree_ros2`.

Could use or reuse:
- **From the H1-2 code (not packages in this inventory):** the latest-only reader + staleness handling (1.4), the `slow_mode`/`relative` action fields (section 6.1), `NamedConfig`/`FrameTask` features.
- **`model_server`:** `gemini_server` and `sam_server` are robot-agnostic image-in/answer-out services (`GeminiQuery`, `SamSegment`). The G1 sim already publishes `/realsense/head/*` (CLAUDE.md §8); a G1 models launch would need its own file because the only models launch is `h12_models.launch.py`, coupled to `use_skills`/`h12_skills`. `graspgen_server` defaults to the Magpie gripper (`DEFAULT_GRIPPER = 'magpie'`), so it needs a Dex3 gripper definition first. `yolo_server` defaults to the H1-2's three cameras and a battery-workcell vocabulary.
- **`estop`:** a hardware button for the G1 would reuse the node as is; the G1 safety layer would need a subscriber (see 2.3).
- **`custom_ros_messages` interfaces:** `Skill*` actions are robot-agnostic; the G1 serves only `SkillFrontierExplore`. A G1 `grasp`/`pick_place` would reuse the same action types.
- **`tools/ros_mcp_server.py`:** usable on the G1 sim after changing the posture thresholds and the `/odom` assumption (the G1 has `sim_gt` odom opt-in).
- **`g1_description` pattern for the H1-2:** the single generated table is the model for 1.5.

Does the G1 already duplicate something these packages offer? Only (a) the unused `custom_ros_messages` `MotorCmd/MotorState` messages vs. SDK IDL (nobody uses either), and (b) the `Livox CustomMsg -> PointCloud2` conversion in `g1_navigation/livox_to_pointcloud.py`, which is not offered elsewhere (the H1-2 sim publishes `/livox/pointcloud` directly instead, `SIM/h12_sensors.py:723-738`; the real H1-2 driver emits both).

## 4. Keep separate (looks duplicated, should stay)

- `rclpy.init/spin/destroy_node/shutdown` in 48 `main()`s: 5 lines of idiom each; a helper would add a dependency edge for nothing.
- `homie_policy.py` and its `quat_rotate_inverse`/`get_gravity_orientation`: vendored non-commercial policy (CLAUDE.md §8), must stay in `g1_locomotion/policy/`. `G1L/locomotion.gravity_z` is guarded by a test against it.
- `SIM/g1/sim_names.py` and `SIM/g1/unitree_interface.py` slot/`MODE_*` constants vs `g1_description`: the robocasa container does not mount `core_ws`. Keep the copy, add a sync test (see 5, item 4).
- `RealtimePacer` in `ISAAC/control/pacing.py` vs `SIM/sim_timing.py`: byte-identical, but two containers; keep with a "copy of" note or a CI diff check (simulator agent).
- Simulator-side `PointCloud2` / `CustomMsg` / `TransformStamped` builders (1.7): three simulators in three containers, different fast paths.
- `ISAAC/dds/common/transport.py`: a deliberate CycloneDDS participant that joins ROS's domain (avoids an RMW conflict); not the SDK factory.
- `G1_DDS_INTERFACE` handling vs the H1-2's YAML `network.interface`, `MODE_MACHINE` 5 (G1) vs 6 (H1-2): robot-specific.
- `g1_dex3` vs `magpie_control` grippers: different hardware (Dynamixel serial vs Unitree DDS Dex3), different services (`SetHandPose` vs `SetGripperPosition`). A skill layer that wants both would need a gripper abstraction first; do not merge the drivers.
- The G1 software latched estop vs the H1-2 hardware button (`estop`): different safety concepts.
- `h12_skills/model_logging.py` copy: documented as intentional; moving it is candidate 1 only if the owners agree to a `model_server` dependency from `h12_skills`.

## 5. Consolidation plan, with ownership and tests

Common recommendation: one new in-tree ament_python package (working name `golem_unitree_common`, not a submodule) for `dds.py` (init/guard/`LatestReader`/`fast_crc`), `config_paths.py`, and optionally `quat.py`. G1 packages (in-tree) adopt it first; `h12_ros2_controller` and `h12_safety_layer` (submodules) adopt through their own repos, after which the gitlinks are bumped, per CLAUDE.md §5. It would be new shared infra: announce it (§5), and add it to `docker/mac/scripts/launch_ros_mac.sh` `PKGS` only if the mac build needs it.

| # | Change | Effort | Real-robot impact | Ownership boundary | Behaviour change | Tests that guard it |
|---|---|---|---|---|---|---|
| 1 | `model_logging.py` once (import from `model_server`, or the shared package) | S | none (perception side only) | in-tree on both sides | none | `h12_skills`/`model_server` tests exist only per package; add one that imports both |
| 2 | DDS init/guard shared | S + M | H1-2 start-up path (guard runs before DDS joins) | `h12_ros2_controller`, `h12_safety_layer` are submodules | none intended; G1 may adopt the H1-2 "no env interface" default only through an `interface_env` argument | `h12_ros2_controller/test/test_controller_config.py`, `g1_safety_layer/test/test_g1_safety.py`, `g1_dex3/test/test_dex3.py`, `tests/docker/test_wait_for_sim.py` |
| 3 | YAML path/loader/validator helpers | S | none | same two submodules | `None` handling must be a parameter (controllers allow an empty file, safety layers reject it) | per-package config tests |
| 4 | H1-2 joint table (names, limits, `MODE_*`), generated from the URDF | M | the H1-2 safety limits move; keep the numbers byte-equal | `h12_ros2_controller`, `h12_safety_layer`, possibly `h12_ros2_model` | none if equal | add a test that the new table equals the two existing ones; add a sim-copy sync test for the G1 (`sim_names` vs `g1_description`, which does not exist today) |
| 5 | One `LatestReader`; G1 control and safety adopt it | M | G1 only (H1-2 import path changes) | `g1_control`, `g1_safety_layer` in-tree; H1-2 submodules for the import | **yes**, on the G1: latest-only, polled reads instead of callbacks; timing and watchdog semantics change | `h12_ros2_controller/test/test_state_subscriber.py`, `g1_safety_layer/test/*_dds.py`, `g1_control/test/test_controller_loop_dds.py` |
| 6 | Quaternion helpers (H1-2 internal) | M | `h12_lowerbody_rl`, `lmg_deploy` policy observations | `h12_lowerbody_rl`, `h12_slam`, `lmg_deploy`, `h12_skills`, `model_server` | none for unit quaternions (checked numerically); watch `wxyz`/`xyzw` | `lmg_deploy/test/test_controller.py`, `h12_slam/test/test_robot_geometry.py`, lower-body policy tests |

## 6. Accidental divergences and possible bugs found (flag to the owners)

1. **`g1_control` ignores shared action fields.** `custom_ros_messages/action/FrameTask.action` has `slow_mode` and `relative`, `NamedConfig.action` has `slow_mode`. `H12C/ros2/frame_task_server.py:178-196` and `dual_arm_server.py:334-336` honour them (`goal.relative`, `goal.slow_mode`; H1-2 commits `6c75106` 2026-07-21 and `99662cc` 2026-10-12). `G1C/ros2/*` contains none of `slow_mode`/`relative` (`grep -c` = 0 in every file) and uses `goal.frame_targets` directly (`frame_task_server.py:169-174`). A `relative: true` goal is therefore executed as an absolute target on the G1, and `slow_mode: true` at full speed. The G1 clients also lack the `slow_mode` argument (`G1C/ros2/dual_arm_client.py:121` vs `H12C/ros2/dual_arm_client.py:121-125`). Probably because `g1_control` forked from a baseline older than those commits.
2. **No latest-only reads in `g1_control` and `g1_safety_layer`** (1.4).
3. **H1-2 RoboCasa sim accepts any `rt/lowcmd`**: `SIM/unitree_interface.py:125-137` has no CRC or header check, while the G1 sim rejects bad CRC / `mode_machine` / `mode_pr` (`SIM/g1/unitree_interface.py:162-170`) and Isaac verifies CRC (`ISAAC/dds/specialized/h12_robot_dds.py:132-134`). CLAUDE.md §8 says sims should behave like the real robot.
4. **`GOLEM_CMD_TIMEOUT` parsing**: `SIM/unitree_interface.py:84` (`float(os.environ.get('GOLEM_CMD_TIMEOUT', '0.5'))`) fails on an empty value and accepts NaN; the G1 sim's `env_seconds` (`SIM/g1/unitree_interface.py:70-81`) handles both. The G1 compose overlay's comment confirms "The H1-2 sim fails on an empty GOLEM_CMD_TIMEOUT" (`docker/docker-compose.g1.yml`).
5. **Dead CRC work** in `H12C/core/channel_interface.py:206`, `G1C/core/channel_interface.py:173`, `G1L/driver.py:142`: the safety layers recompute it and nothing verifies inbound CRC.
6. **Submodule pin**: `core_ws/src/custom_ros_messages` gitlink `3f361bf` ("Add SetHandPose for the G1's Dex3-1 hands") is on local branch `g1_interfaces`; `git branch -r --contains 3f361bf` lists no remote branch (as of the last fetch; I did not fetch). CLAUDE.md §5 forbids pins on unpushed commits. Verify with `git ls-remote`.
7. **`cl_realsense/pc_acc` import failure** (`arpa_control`), `another_node` placeholder, and `magpie_control/launch/gripper_control.launch.py` referencing a non-existent `config/gripper_config.yaml` (2.2, 2.4).
8. **Two SDK revisions across containers** (2.8).
9. **No sync test between `SIM/g1/sim_names.py` / `SIM/g1/unitree_interface.py` constants and `g1_description`** (1.5); I verified they match today by executing both.
10. **Domain validation is inconsistent** (`>0` in the RoboCasa sims, `1..232` in Isaac/`wait_for_sim`/shell wrappers) and the G1 has five distinct "domain 0" policies (1.1).
11. **Sim time conversions differ** (truncate vs round) and the G1 sim mixes both (1.7).

## 7. Not checked / limits

- I did not diff the two safety layers' `safety_layer.py` or `safety_checks.py`, the lower-body packages, the navigation stacks or the simulators beyond the helper-level items above; the figures in 1.9 are from the whole-tree scan only.
- "No user" for an interface means no import or mention in first-party Python/launch/YAML/C++ outside the message packages and `docs/`; external repos (lab code, bags) could still use them. `lmg_deploy/bags/*` contains recorded `/estop/*` topics only.
- The `ratio` figures are line-based; they under-count similarity when code was reformatted or variables renamed (e.g. `quat_rotate_inverse` in HOMIE vs H1-2 shows 0.00 line ratio but 0.89 token ratio and numeric equivalence).
- I did not run any tests, builds or containers.
