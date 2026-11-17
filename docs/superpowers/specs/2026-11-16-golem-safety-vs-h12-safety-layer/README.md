# golem_safety against h12_safety_layer: what is the same and what differs

This compares GOLEM's safety layer on the H1-2, `core_ws/src/golem_safety` at GOLEM `main`
`d1d790e`, with [correlllab/h12_safety_layer](https://github.com/correlllab/h12_safety_layer)
at its `main`, `ad06903` ("Sync dev to main", 2026-10-20). No other upstream branch has a
commit past GOLEM's last pin of the package (`camera_ready` `a178f0b`, which is `ad06903`
plus six commits); those six are differences from `main` too and are marked below.

The existing change report,
[`../2026-10-27-golem-upper-ik-and-safety/change-report.md`](../2026-10-27-golem-upper-ik-and-safety/change-report.md),
lists every difference from the pinned `a178f0b`, with its reason. This page checks the two
layers against each other directly and says, for each row, how it was checked:

- **measured**: both layers were run on the same scripted inputs (below), and the row is what
  they wrote on `rt/lowcmd`;
- **loader**: each layer's own config loader, on each config, compared value by value;
- **inspection**: read in the code only, not run. The file and line are given.

## Summary

The derived limits, and the clipping of a valid command on the body slots, are identical:
the clipped values are bit-for-bit the same in both modes, and the estop limits trip at the
same values. Six of the eight upstream configs give identical limits; `relax_safety_split` differs in
one value (ankle pitch torque estop), and upstream's `sim_safety_split` has no counterpart. The behaviour around the clipping differs: when inputs engage, what a
command's header and spare slots carry, what happens after an estop, and five triggers
(two that only `golem_safety` latches on, one, the upper watchdog, that only
`h12_safety_layer` does, and two hardware-estop rules).

## How it was tested

`golem_safety` was built from `main` `d1d790e` into its own workspace in the `golem_ros`
container, and its suite passed: **233 tests, 0 failures, 0 skipped** (`colcon test
--packages-select golem_safety`, including the 84 ROS-level tests of `test_node_ros.py`).

The differential harness is in [`harness/`](harness/), its outputs in [`results/`](results/):

- `compare_limits.py` runs `h12_safety_layer`'s own `load_config` on each of its 8 configs and
  `golem_safety_dump_config h12 <name>` on the config of the same name, and compares every
  derived array (q, dq, tau clip and estop limits, kp and kd maxima) joint by joint, and the
  two joint tables. Output: `results/limits.json`. `compare_settings.py` prints the other
  settings side by side (`results/settings.txt`).
- `driver.py` starts one layer (`h12_safety_layer`'s `SafetyLayer` through `old_runner.py`
  on DDS domain 92, `golem_safety`'s `safety_node` with `sim:=true` on domain 93), waits
  2.5 s, then plays a scenario on Unitree DDS: `rt/lowstate` at 500 Hz, the safety inputs at
  200 Hz, the hardware estop at 50 Hz in each layer's own form (the old layer's raw
  `h12/estop_status_raw` struct, the new layer's ROS Bools), the same values to both. It
  records every `rt/lowcmd` frame. `analyze.py` summarises and compares them
  (`results/report.txt`). Both layers ran `default_safety_split` (`default_safety_full` for
  the full-body scenario), whose limits are identical, with the hardware estop off except
  in the two estop scenarios.
- The old layer ran as its `SafetyLayer` class, not its ROS node, so the node-level rows
  (heartbeat, exit) are by inspection. Its runner does not stop on SIGINT, so the recordings
  are cut at each scenario's end, and its log lines after that end (its upper watchdog
  firing once the driver stopped publishing) are not results.

Reproduce: copy `harness/` and a checkout of `h12_safety_layer` at `ad06903` (as
`harness/h12_safety_layer`) to `/tmp/safety_diff` in the `golem_ros` container, build
`golem_safety` into `/tmp/gs` (`colcon build --packages-select golem_safety` there, with
`core_ws/install` sourced), then `bash /tmp/safety_diff/run_all.sh` and
`python3 analyze.py out`. Domains 92 and 93 must be free; the harness refuses domain 0.

## The real-robot bags, replayed

`core_ws/src/lmg_deploy/bags/policy_bag_walk` (65 s, walking to Nav2 goals) and `reach`
(31 s) were recorded on the real H1-2 with `h12_safety_layer` `camera_ready` `a178f0b`
relaying `lmg_deploy`: both safety inputs, `/lowstate`, the estop topics and `/lowcmd`. The
scripts are in [`harness/bags/`](harness/bags/).

- **Headers on the real robot.** Every `/lowstate` (47,962 samples) reports `mode_machine` 6
  and `mode_pr` 0; every input and every `/lowcmd` carries (6, 0); slots 27-34 are zero
  throughout (`headers.py`).
- **The recorded `/lowcmd` is the inputs, unchanged.** Every recorded frame with gains is
  bit-identical, on all 27 body slots and every field, to the lower input (slots 0-11) and
  the upper input (12-26) recorded within 200 ms before or 60 ms after it: 27,203 of 27,203
  in `policy_bag_walk`, 11,294 of 11,294 in `reach` (`compare.py`). The 58 walk frames before
  the first recorded input are not counted.
- **golem_safety on the same inputs.** `replay.sh` plays each bag's inputs, `/lowstate` and
  estop into `safety_node` (`relax_safety_split`, `use_estop:=true`, domain 93) and records
  its `/lowcmd`. Every relayed frame is bit-identical to its inputs, header (6, 0) and zero
  slots 27-34 included: 11,813 of 11,813 in `reach`, 28,556 of 28,562 in the walk; the 6
  others are the first 11 ms, when the upper input had engaged and the lower's first command
  had not arrived (its slots carry zero gains until then, as in `h12_safety_layer`).
- **The estops happen at the same moment, for the same reason** (`timeline.py`): in the walk
  both layers latch at 57.11 s on the right ankle roll measured at -0.2623 rad, past its
  -0.2617 estop limit; in the reach both latch on the hardware estop press at 23.63 s. Both
  write 50 zero commands after the latch, then nothing.

The bags stay inside the clip limits and carry only (6, 0) headers, so they do not exercise
clipping or a foreign header; the scenarios below do.

## Identical

| Behaviour | Evidence |
|---|---|
| Joint table: names, slot order, position, velocity and torque limits of the 27 joints (`h12_safety_layer/core/joint_limits.py`, `golem_safety/tools/h12_joint_limits.py`) | loader (`limits.json`: `_joint_tables` empty) |
| Derived clip and estop limits of `default_safety_full`, `default_safety_split`, `locomotion_safety_full`, `relax_safety_full`, `tight_safety_full`, `tight_safety_split` | loader (`limits.json`) |
| Relay rate target 500 Hz (`control.publish_hz`) | loader; measured 500 frames/s new, 457-484/s old (its Python loop sleeps on wall time) |
| Clipping of a valid command on the body slots: q to the clip range, dq and tau to ± their limits, kp to [0, `kp_max`], kd to [0, `kd_max`], negative gains to 0, values exactly at and one float32 step past the limits | measured (`relay_clip_split`, `relay_clip_full`: the body slots are bit-identical in both phases) |
| Split mode: legs (slots 0-11) from the lower input, torso and arms (12-26) from the upper | measured (`relay_clip_split`, `upper_silent`, `lower_silent`) |
| A non-finite value in any body slot of a command, the input's own or not, latches the estop | measured (`nan_unowned_slot`: NaN in slot 20 of the lower input; both latch within 3 ms) |
| Estop limits are strict: a measured q, \|dq\| or \|tau_est\| past its limit latches, one at it does not; comparison in double of the float32 wire value (so a knee at float32(3.2) = 3.2000000477 latches, one float32 step below does not) | measured (`estop_q_edge`, `estop_q_below`, `estop_dq`) |
| A non-finite `/lowstate` q, dq or tau_est latches | measured (`state_nan`) |
| `/lowstate` checks run on every body slot in both modes: any non-finite value first, then q, then dq, then tau, each reporting the lowest bad slot | inspection (`h12_safety_layer/core/safety_checks.py:72-96`, `golem_safety/src/safety_core.cpp` `limit_violation`); the measured reasons name the same joint |
| A quiet lower input keeps its last command relayed (and, by inspection, a quiet full-body input) | measured (`lower_silent`); inspection for full body |
| A hardware estop sample that says triggered latches, within one cycle | measured (`hw_estop_trigger`: old t = 1.006 s, new 1.003 s) |
| The estop is one way until restart | measured (every latch above) |

## Different

| Behaviour | `h12_safety_layer` `ad06903` | `golem_safety` `d1d790e` | Evidence |
|---|---|---|---|
| `relax_safety_split` ankle pitch torque estop | 60 N·m | 95 N·m (`camera_ready` `0eab20b`, for the walking policy) | loader |
| `sim_safety_split` (domain 1, no estop, upper watchdog off) | a config | no H1-2 sim config: the H1-2 sims run the real `relax_safety_split` (GOLEM `61a0af1`) | loader |
| Choosing the hardware estop | `estop.enabled` in the config (true in all but `sim_safety_split`) | the node's `use_estop` parameter; the bringups pass true | loader; inspection |
| Hardware estop input | the raw DDS struct `h12/estop_status_raw`, read on the default domain 0 whatever `ROS_DOMAIN_ID` says (`estop_subscriber.py:53`) | the ROS Bools `/estop/triggered` and `/estop/plugged_in` from `estop_node`, on `ROS_DOMAIN_ID` | inspection; measured (each layer fed its own form) |
| Engaging before the hardware estop has spoken | relays from the first command; no estop sample is needed | no input engages before the first estop sample (`engage_refused`) | measured (`hw_estop_silent`, `hw_estop_trigger`: at t = 0.2 s the old layer relays, the new writes zero gains) |
| A hardware estop that falls silent | nothing happens: the last sample stands | latches after `estop.timeout_seconds` (0.5 s) | measured (`hw_estop_silent`: samples stop at t = 1.0; old relays on, new latches at 1.485) |
| Engaging before `/lowstate` reports the robot | relays from the first command, whatever `/lowstate` says, or before any | each input engages on its first command once `/lowstate` reports `mode_machine` 6 | measured (`engage_mm0`: with `mode_machine` 0 the old layer relays, the new writes zero gains until it turns 6) |
| `/lowstate` from another robot (`mode_machine` 5) once engaged | ignored | latches | measured (`foreign_mm`) |
| A second publisher on one input | both accepted, the newest wins each cycle | latches unless the current writer has been quiet for `single_writer.handover_seconds` (0.5 s) | measured (`second_writer`) |
| Upper input quiet in split mode | past `estop.upper_stale_seconds` (default 2.0 s; set to 0.0, off, only in `relax_safety_split` and `sim_safety_split`) the whole robot estops | its last command keeps being relayed and `input_quiet` is logged (`stale.upper_seconds`: 2 s in `default`/`tight_safety_split`; 0, no report, in `relax_safety_split`); never latches | measured (`upper_silent`: old latches at 2.495 s, new still relays the arms at 3.4 s). With `relax_safety_split`, the bringups' config, neither latches (loader: upper stale 0.0, off, in the old config; 0, no report, in the new) |
| Command header (`mode_machine`, `mode_pr`) | copied from the last command received on either input, `LowCmdDefault` (0, 0) before any (`safety_layer.py:202,240`) | always the profile's (6, 0) | measured (`startup_idle`: old (0, 0); `split_header_source`: lower (8, 2) and upper (7, 1) both appear, alternating) |
| Slots 27-34 and `reserve` | split mode: copied from the upper input's command; full body: from the full input's | always zero | measured (`split_header_source`: only the upper input's 9s, never the lower's 5s; `relay_clip_full`) |
| Output before any command | `make_estop_cmd(LowCmdDefault())` at 500 Hz: every motor off, header (0, 0) | `zero_command` at 500 Hz: every motor off, header (6, 0) | measured (`startup_idle`) |
| After an estop | the estop command (body slots mode 0, all values 0) every cycle until the process stops; its header and slots 27-34 are the last command's (`safety_layer.py:262`, `make_estop_cmd`) | 50 zero commands, then silence until restart (`camera_ready` `58d0f99`) | measured (every latch: old about 480 frames/s after it, new exactly 50); the header and spare slots by inspection |
| Estop reasons | `motor 16 (left_elbow_joint) dq out of estop range: 50.0`; `Command validation failed on split_mode_lower: q contains non-finite values` | adds the limit and its multiple and names the motor: `... 50 (limit 6.000, 8.33x)`, `... motor 20 (right_shoulder_pitch_joint) q is nan` (`camera_ready` `5182fbe`) | measured |
| Late `/lowstate` | no check | on domain 0, a `/lowstate` later than `control.state_lag_ms` (100 ms) past its tick latches (`camera_ready` `e0396fc`); not checked off domain 0, so not measured here | inspection |
| Quiet `/lowstate` and inputs | not reported | `state_quiet`/`input_quiet`/`input_resumed` logged; relay unchanged | measured (logs) |
| Relay structure | input callbacks clip into per-input slots; a publisher thread sends at 500 Hz of wall time | newest-only keep-last-1 inputs, one relay loop on robot time (the sim's `/clock` under `use_sim_time`) (`camera_ready` `e0396fc`) | inspection; rates measured above |
| Heartbeat `/safety/heartbeat` | true at 10 Hz from start-up until the latch, then false (`ros2/safety_node.py:63`) | none until relaying can begin (`/lowstate` reports 6 and, with `use_estop`, a first estop sample), then true at 10 Hz on the steady clock, false once latched (`camera_ready` `5b08888` for the steady clock) | inspection; `golem_safety`'s `test_node_ros.py` |
| Domain 0 | every config but `sim_safety_split` sets domain 0, and the node initialises it without asking (`ros2/safety_node.py:45`); `ROS_DOMAIN_ID` wins when set | uses `ROS_DOMAIN_ID`; with `sim:=true` refuses domain 0 | inspection; `test_sim_refuses_domain_zero` |
| Exit | the node's `destroy_node` stops the publisher thread: output stops, no estop command | output stops at once; an estop burst in progress is finished first | inspection; `golem_safety` tests |
| Implementation | Python, Unitree SDK DDS (`unitree_sdk2py`) | C++ `rclcpp`, ROS topics only (`/x` is DDS `rt/x`, same `unitree_hg` types); CRC computed as the SDK does (`test_lowcmd` checks SDK values) | inspection; test suite |
| Logging | events printed; with `logging.enabled`, chunks under `/tmp/h12_safety_layer` | events through the ROS logger from their own thread; chunks per run under `/tmp/golem_safety/<run>/` | inspection |

## Found along the way

- `h12_safety_layer`'s raw estop reader opens its DDS participant on the default domain (0),
  not `ROS_DOMAIN_ID` (`core/estop_subscriber.py:53`), so a sim run of it on another domain
  still reads, and can be latched by, an estop box publishing on the real robots' domain
  (CycloneDDS's Python `DomainParticipant.__init__` defaults to `domain_id: int = 0`, checked in
  the `golem_ros` container; GOLEM's `cyclonedds.xml` sets `Domain Id="any"`). The harness pins it
  to its own domain.
- A workspace built before GOLEM `61a0af1` keeps a dangling
  `install/golem_safety/share/golem_safety/config/h12/sim_safety_split.yaml` link
  (`--symlink-install`); `safety_node` refuses that config as unreadable.
