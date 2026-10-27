# Change report: golem_safety and golem_upper_ik

Every value or behavior that differs from what the H1-2 or the G1 does today, for each robot. Each task appends a section for the changes it makes. An entry gives the robot, the old behavior, the new behavior and the reason. A value that is carried over unchanged is not listed, except where the file it lives in changed shape.

## Task 3: safety config schema and the carried-over configs

The 8 H1-2 and 5 G1 safety configs now live in `core_ws/src/golem_safety/config/<robot>/` under the same names. Their limits derive to the same numbers as before: `test_limits_match_old_loaders` compares all 13 against fixtures written from the old loaders, with `atol=1e-12`. The rows below are the differences.

### Keys removed

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2, G1 | `network.domain_id` and `network.interface` chose the DDS domain and interface for the SDK channel factory (`ROS_DOMAIN_ID` won over the file; the G1's `G1_DDS_INTERFACE` won over `network.interface`). The files held domain 0 (real) or 1 (sim) and no interface. | The key is refused with a message that names its replacement. ROS supplies the domain; the G1's interface pin is `cyclonedds_local.xml`, which the docker wrappers write from `G1_DDS_INTERFACE`. | The node is a ROS node and no longer opens an SDK channel. |
| G1 | `robot: real\|sim` was mandatory. A `sim` config refused DDS domain 0, and could not set `network.domain_id: 0`. | The key is refused. The bringup's `sim` argument says whether the robot is simulated. | One config serves sim and real (spec, Robot profiles). |
| G1 | `engage.mode_machine` came from the config, default 5; all five configs held 5. | The key is refused. The robot profile holds it: G1 5, H1-2 6. | `mode_machine` is a fact about the robot, not a tuning value. The G1's value is unchanged. |
| H1-2 | `estop.enabled` (true in the real configs, false in `sim_safety_split`), `estop.estop_topic` (`h12/estop_status_raw`, a raw DDS sample) and `estop.poll_hz` (500) configured the hardware estop. | The three keys are refused. The bringup's `use_estop` turns the hardware estop on. The new `estop:` section holds the ROS topics (`/estop/triggered`, `/estop/plugged_in`) and `timeout_seconds` (0.5), all with those defaults and unset in every carried config. | Whether an estop box is present comes from the bringup. The layer reads the `estop` node's ROS topics (spec item 10). |
| G1 | Any `estop:` section was an error ("the external estop section is no longer supported"). | An `estop:` section is the hardware-estop section above. | The G1 has the same optional hardware estop as the H1-2, through `use_estop`. |

### Keys renamed or moved

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2 | `estop.upper_stale_seconds`. `default_safety_split` and `tight_safety_split` had no key, so the loader's default of 2.0 applied; `relax_safety_split` and `sim_safety_split` set 0.0 (off). In `split_mode`, an upper input quiet for longer than this latched the estop. | `stale.upper_seconds`: 2.0 in `default_safety_split` and `tight_safety_split`, 0.0 in `relax_safety_split` and `sim_safety_split`. The full-body configs leave it out (0.0): they have no upper input. The loader default for the key is now 0.0, where the H1-2 loader's was 2.0. | Renamed. The threshold now only reports and the relay keeps going (Task 6 lists the behavior change). The 2.0 is written into the two configs that used to take it from the loader's default. |
| G1 | The upper input's watchdog threshold was the constant `UPPER_STALE_ESTOP_SECONDS` = 2.0 in the node, used in `split_mode` and `arm_sdk_mode`. An upper input quiet for longer latched the estop. | `stale.upper_seconds: 2.0` in `arm_sdk`, `sim_arm_sdk`, `split` and `sim_split`; `full` leaves it out (0.0). | The constant becomes a configured reporting threshold. |
| G1 | `control.lower_stale_estop_seconds`: 0.2 (default and `full`, `split`), 1.0 in `sim_split`. A leg input (`split` lower, or `full`) quiet for longer latched the estop. | `stale.lower_seconds`: the same values. The two `arm_sdk` configs leave it out (they have no lower input). | Renamed. The threshold now only reports and the relay keeps going (Task 6). |
| G1 | `control.state_stale_estop_seconds`: 0.1 in `arm_sdk`, `full` and `split`, 1.0 in `sim_arm_sdk` and `sim_split`. Once engaged, a `/lowstate` gap longer than this latched the estop. | `stale.state_seconds`: the same values. | Renamed. The threshold now only reports and the relay keeps going (Task 6). |
| H1-2 | `lower` and `state` stale times did not exist. | `stale.lower_seconds: 0.2` and `stale.state_seconds: 0.1` in all 8 configs. | The brief's values for a time without an old key. `sim_safety_split` takes the same 0.2 and 0.1; the G1's sim configs relax these to 1.0 because the sim can run slower than real time, so the H1-2 sim may report quiet inputs it never reported before. |
| H1-2 | `estop.state_lag_ms`: no config set it, so the loader's 100.0 applied. | `control.state_lag_ms: 100.0` in all 8 configs. | Moved; the lag check is not an estop-section setting. |
| G1 | No transport-lag check. | `control.state_lag_ms: 0.0` in all 5 configs, so the G1's lag check stays off, as today. A value <= 0 disables the check, as in `h12_safety_layer`. | The key and the mechanism are shared with the H1-2. The G1's `/lowstate` tick has not been measured on the real robot, so the check is not turned on for it. |

### Values that changed shape

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2, G1 | Topic values were DDS names: `rt/safety/lowcmd_in`, `rt/safety/lowcmd_lower_in`, `rt/safety/lowcmd_upper_in`, `rt/lowcmd`, `rt/arm_sdk`, `rt/lowstate`. | `/safety/lowcmd_in`, `/safety/lowcmd_lower_in`, `/safety/lowcmd_upper_in`, `/lowcmd`, `/arm_sdk`, `/lowstate`. A value without a leading `/` is refused. | The same DDS topics: ROS topic `/x` is DDS `rt/x`. |
| H1-2 | Hard-coded: from boot the layer streams zero-gain commands; after an estop it writes 50 zero-gain commands and then goes silent. | `startup.rule: zero_stream`, `estop_response: {mode: zero, after: burst_then_silence, burst_count: 50}` in all 8 configs. `startup.rule` and `estop_response.after` are required keys with no default: a config without either is refused. | The H1-2's behavior as configuration, so both robots share one layer. A default would give a G1 config that left a key out the H1-2's behavior. |
| G1 | Hard-coded: nothing is written until the engage gate passes; `split` and `full` keep writing the estop command; `arm_sdk` hands the arms back by ramping the weight. | `startup.rule: engage_gate` and `estop_response.after: keep_writing` in all 5 configs, both required keys with no default. `full`, `split` and `sim_split` carry `mode: zero` and `kd: 8.0`; the two `arm_sdk` configs carry `mode: zero` and no `kd` (the weight ramp replaces the estop command). In `arm_sdk_mode`, `after` must be `keep_writing`: any other value is refused. | The G1's behavior as configuration. The weight ramp keeps writing after an estop, so a burst-then-silence response cannot describe `arm_sdk_mode`. |
| G1 | `arm_sdk.release_seconds` loader default 1.0. | Loader default 3.0. `arm_sdk` (3.0) and `sim_arm_sdk` (1.0) write their own values, as before. | The schema's default. It matters only for a new arm_sdk config that omits the key. |
| H1-2, G1 | `logging.enabled` defaulted to true and `logging.base_dir` to `/tmp/h12_safety_layer` or `/tmp/g1_safety_layer`. | Loader defaults `false` and `/tmp/golem_safety`. Every carried config writes its own `enabled` (H1-2 false, G1 true) and `base_dir` (its old value), so no carried value changed. | One layer, one default. |
| G1 | `G1_SAFETY_LOG_DIR`, when set, replaced `logging.base_dir` with a per-run subdirectory. | The config loader does not read it. Task 7 decides what the node does with it. | It is node behavior, not config. |

### Validation that differs

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2, G1 | `kp_max` and `kd_max` (and `kp_abs`, `kd_abs`) could be a number or a per-joint list. All 13 configs use one number. | A single number, which is what `DerivedLimits.kp_max` and `kd_max` hold. A list is refused. | The relay clips every joint's gain against one limit. A per-joint gain limit would be a new interface. |
| H1-2 | Unknown keys were ignored, for example `kp_max` under `limits.estop`. | Every unknown key, at every level, is refused with the robot, the file and the key. | Spec item 11. The G1 loader ignored them too, apart from refusing `estop`. |
| H1-2, G1 | `nan` and `inf` were accepted as limits and ratios. | Any non-finite number is refused. | A NaN limit would never trip a comparison. |
| H1-2, G1 | With both `limits.clip` and `limits.enforce`, or both `kp_max` and `kp_abs`, the first silently won. | Both together are refused. | Two spellings of one setting can disagree. |
| G1 | Every G1 config had to have `arm_sdk.waist_kp_min` at most the waist's `kp_max`, whatever its mode. | Checked in `arm_sdk_mode` only. | Only `arm_sdk_mode` uses the value. |
| G1 | `lower_stale_estop_seconds` and `state_stale_estop_seconds` had to be positive. | Any finite number; a value <= 0 disables the report. | The stale times only report now. |

## Task 4: LowCmd CRC, command builders and the chunk logger

The CRC is the one the Unitree SDK computes: `test_lowcmd` compares `lowcmd_crc` with SDK values for five commands and for the two old estop commands (`test/fixtures/lowcmd_crc.json`, written by `tools/crc_fixtures.py`). The chunk files keep the old keys, key order, dtypes and shapes, and `combine_chunks.py` reads them. The rows below are the differences.

### Commands

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2 | The boot command and the estop command were `make_estop_cmd` of the newest input command, or of `LowCmdDefault()` before any input arrived: `mode_machine` 0 and `mode_pr` 0 until a controller command carried its own header. Every one of the 35 slots had mode 0 and zero q, dq, tau, kp, kd (`make_estop_cmd` zeroes slots 0-26, and `LowCmdDefault` has zeros in 27-34). | `zero_command` is the same 35 slots, with `mode_machine` 6 and `mode_pr` 0 from the profile, stamped with its CRC. | Outgoing commands carry the profile's header (spec, item 5). Every slot value is unchanged. |
| G1 | The `damp` estop command copied the newest input command and set slots 0-28 to mode 1, kp 0, kd from the config. Its header and slots 29-34 came from that input command. | `estop_command` starts from `zero_command`: the header is the profile's (5, 0) and slots 29-34 are all-zero with mode 0. Slots 0-28 are as before. | Nothing of the input commands reaches the estop command. Slots 29-34 differ only if an input had set them, and the arm_sdk weight in slot 29 is written by the weight ramp, not by the estop command. |

### Chunk logging

| Robot | Old | New | Reason |
|---|---|---|---|
| G1 | A sample whose arrays had another length than the first sample's raised when the writer thread stacked a chunk. | The logger takes the motor count from the first sample and drops a sample with another count; `samples_dropped` counts it. | A bad sample must not stop the log. The node fills one entry per body slot, so the chunk shapes are the old ones (29 motors for the G1). |
| G1 | A failed chunk write raised in the writer thread, which stopped logging with no message. | A failed chunk is counted (`write_failures`), reported on stderr and leaves a gap in the chunk numbers; later chunks still write. | A full disk must not end the log silently. |
| G1 | A chunk was written to its final name directly, so `combine_chunks` could read half a file. | A chunk is written to `<name>.npz.tmp` and renamed; at start the logger also removes the prefix's leftover `.npz.tmp` files. | A reader never sees a partial chunk. |
| G1 | `combine_chunks.py` defaulted to `/tmp/g1_safety_layer`. | `scripts/combine_chunks.py`, installed to `lib/golem_safety`, defaults to `/tmp/golem_safety`. Its code is otherwise the old script's. | The logging default is `/tmp/golem_safety`. The G1 configs still set their own `base_dir`. |
| G1 | `G1_SAFETY_LOG_DIR` gave a per-run directory `<root>/<YYYYmmdd-HHMMSS>-<pid>`. | `run_log_directory(root)` returns that name and the logger clears its directory's chunks when it starts, as before. The node decides whether to use it (Task 7). | The logger takes a directory; the choice is the node's. |

## Task 5: SafetyCore relay, clipping, limits, identity, writers and the estop latch

`SafetyCore` (`include/golem_safety/safety_core.hpp`) holds the layer's decisions; the node that runs it is Task 7, so none of these rows reaches a robot before then. Ported unchanged, and not listed: the clip and estop-limit arithmetic (numpy's `clip` order included), the newest-valid-command-per-input relay, the arm_sdk engage and release ramps (the release advancing only while `/lowstate` is at most `stale.state_seconds` old), the 50-command H1-2 burst and the G1 keep-writing response, silence after a latch before engaging (G1), the estop diagnostic arrays, logging every `/lowstate` up to and including the one that latches, and the G1's JSON log events `gain_clipped` (`commanded`, `event`, `gain`, `joint`, `max`, `source`) and `arm_sdk_release_ramping`/`arm_sdk_release_held` (`event`, `state_age`, `weight`, rounded to 4 decimals), with the same keys, sorted, and numbers as Python's `json` writes them.

### Inputs and writers

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2 | No writer check: every publisher on an input was relayed, the newest sample of any of them each cycle. | A command from a writer other than the input's current one latches the estop (`second writer on <topic>: ...`) when the current writer's last command is at most `single_writer.handover_seconds` (0.5 s) old. After a longer quiet gap the new writer is accepted, logged (Info) and becomes current. `handover_seconds` must be positive. | Review Focus 1: two live writers interleave their commands. A controller restarted after its old process is gone is still accepted, as today. |
| G1 | The first writer of each input (its DDS publication handle) was the only one ever accepted. Any other writer latched the estop, even after the first had stopped, so a restarted controller latched it (`test_a_restarted_writer_also_latches_the_estop`). A sample without a publication handle skipped the check. The reason was `Second writer on <topic> (<source>): two publishers`. | As the H1-2 row: accepted after a quiet gap over 0.5 s, a latch within it. The writer is the publisher GID that the node passes with every command (Task 7). | Review Focus 1. |
| H1-2, G1 | A command with a non-finite q, dq, tau, kp or kd on any body slot (all 27 or 29, not only the input's own) latched the estop, before and after engaging: `clip_low_cmd` raised and `_clip_cmd_or_estop` latched with `Command validation failed on <source>: q contains non-finite values` (`<source>` e.g. `split_mode_upper`). On the G1 before engaging the latch was silent. | Unchanged: it still latches on both robots, on any body slot, before and after engaging, and silently on the G1 before engaging. Only the reason text differs: it names the input's topic, the motor and the field, as `Command validation failed on /safety/lowcmd_upper_in: motor 20 (right_shoulder_pitch_joint) q is nan`. | The plan's Review Focus 3 (drop the sample and hold the last valid command) rested on a wrong reading of the old code. The controller corrected it (Ruling 20): the H1-2's behavior wins, and both layers latch. The reason text names the joint and field the old text left out. |
| H1-2 | A kp or kd clipped to `kp_max`/`kd_max` was not reported. | Each is reported once per joint and gain as the G1's `gain_clipped` JSON event (Warn), with `source` `split_mode_lower`, `split_mode_upper` or `full_body_mode`. | The G1's report, with no H1-2 behavior against it. |

### Robot identity

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2 | `/lowstate`'s `mode_machine` was never read. | Once engaged (from the first valid command of any input), a `/lowstate` whose `mode_machine` is not 6 latches the estop: `/lowstate mode_machine 5 is not the h12's 6: another robot or sim is on this domain`. Before that the sample is only recorded. A foreign sample is never limit-checked. | Spec item 5: another robot or sim on the domain. The H1-2 sims report 0 today, so they need Task 12 (report 6) before the H1-2 bringups switch over (Task 13). |
| G1 | The engage gate required `mode_machine` 5. After engaging nothing checked it, and the `rt/lowcmd` header echoed whatever the newest `/lowstate` carried. | After engaging, a `/lowstate` whose `mode_machine` is not 5 latches the estop. Before engaging it is recorded (Task 6 refuses engagement on it). | Spec item 5. |

### What the relayed command carries

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2 | `split`: the relayed command was a copy of the upper input's newest command with slots 0-11 replaced by the lower input's. Its header (`mode_pr`, `mode_machine`) was then overwritten from `_last_cmd`, the newest command of *either* input (`safety_layer.py:228, 274-275`), and its `reserve` words and slots 27-34 came unclipped from the upper input. `full`: the full input's command, header, `reserve` and slots 27-34 included. The arm controller sends `mode_machine` 6 and leaves slots 27-34 and `reserve` at zero (`channel_interface.py`, 27 motors); other publishers were not checked. | The header is the profile's (`mode_machine` 6, `mode_pr` 0) and the CRC is computed. Slots 27-34 and the `reserve` words are `zero_command`'s: mode 0 and all zero. | Spec item 5. Only an input's own slots are relayed. |
| G1 | `split` and `full`: `mode_pr` 0, `mode_machine` echoed from `/lowstate`, and slots 29-34 and `reserve` from the upper or full input. `arm_sdk`: the header was not stamped, so the upper input's header went out; slots 30-34 and `reserve` came from the input; the legs were zeroed and slot 29 carried the weight. | The profile's header (5, 0) in every mode, `/arm_sdk` included. Slots 29-34 (30-34 in `arm_sdk`, where slot 29 is the weight) and `reserve` are zero. The legs and the weight slot are as before. | Spec item 5. |

### Estop reasons and the report

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2 | The q reason was `motor i (name) q out of estop range: v`; the dq and tau reasons added `(limit L, Mx)`. A non-finite `/lowstate` gave `low_state contains non-finite values`. | The q reason also gives the limit crossed and the multiple, for example `(limit 1.670, 1.12x)`, with the multiple \|q\|/\|limit\| (`inf x` for a zero limit). A non-finite `/lowstate` names the motor and the field (`motor 0 (left_hip_yaw_joint) q is nan in /lowstate`). | Spec item 6: every limit reason names the joint, the value, the limit and the multiple. |
| G1 | No reason carried the limit or the multiple. Non-finite as on the H1-2. | As the H1-2 row. | Spec item 6 (H1-2 `5182fbe`). |
| H1-2, G1 | The estop report (`estop_command_written`, or the G1's `estop_latched_silent`) held wall-clock epoch milliseconds: `decision_ms` at the latch, and `action_ms` right after the DDS write. The H1-2's `trigger_stamp_ms` was the estop board's own stamp. | The same keys, in the same JSON, as an Error event on the boot-time clock. `decision_ms` is the core's `now` at the latch and `action_ms` the `now` of the first step after it, whose command the node then publishes. `trigger_stamp_ms` is the arrival of the hardware estop sample: the `std_msgs/Bool` topics carry no stamp. As before, the report is formatted only after that step's command has been handed back: the core builds it in `drain_events()`. | The core takes its time from the node's boot clock (spec item 2) and does not publish. |

## Task 6: SafetyCore start-up rules, the estop input, quiet inputs, lag, arm_sdk and exit

These rows describe `SafetyCore`; the node that runs it is Task 7. Ported unchanged, and not listed: the G1 engage gate (fresh `/lowstate`, `mode_machine` 5, every slot of the mode's inputs within `max_position_error` of the measured q, the furthest slot named), its refusal logged at most once a second, the `arm_sdk_engage` and `lowcmd_engage` events, the arm_sdk engage ramp, the `waist_kp_min` latch before and after engaging, the H1-2 zero-gain stream from boot, and the H1-2 transport-lag arithmetic (`STATE_LAG_DRIFT_MS_PER_S` 0.5, a uint32 tick that wraps at 2^32, the baseline at the earliest offset seen).

### Start-up and engaging

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2 | Each input's commands were relayed on its slots from its first command, whatever `/lowstate` said or whether one had arrived. | An input engages on its first valid command that arrives once the newest `/lowstate` carries `mode_machine` 6 (and, with `use_estop`, a hardware estop sample has arrived). Until then the zero-gain stream goes on. A command from before does not count: the input engages on its next one. A `/lowstate` from another robot before engaging only delays it. | Spec items 4 and 5: no input engages without the robot's own `/lowstate`. The H1-2 sims need Task 12 (report 6) before the H1-2 bringups switch over. |
| H1-2 | Nothing was logged while waiting. | A command that cannot engage raises `{"event": "engage_refused", "mode": ..., "reason": ...}` (Warn), at most once a second: `no /lowstate received yet`, `/lowstate mode_machine is 5, expected 6`, or `no hardware estop sample received yet on /estop/triggered or /estop/plugged_in`. Each input's engagement raises `{"event": "input_engaged", "source": ..., "topic": ...}` (Info). | Review Focus 2: the layer says why it is not relaying. |
| G1 | `split` and `full` waited silently until every input had sent a command. Refusal reasons named `rt/lowstate`: `no rt/lowstate received yet`, `rt/lowstate is 0.200 s old (> 0.1 s)`, `rt/lowstate mode_machine is 6, expected 5`, `motor 15 (...) command q ... rad from rt/lowstate q ...`, `motor 3 (...) has a non-finite q`. | A missing input is a refusal reason too (`no command received yet on /safety/lowcmd_lower_in`), and with `use_estop` so is a missing estop sample. The reasons name the ROS topic (`/lowstate`), and the non-finite one ends `has a non-finite q in /lowstate`. The order is: no `/lowstate`, its age, its `mode_machine`, the estop sample, a missing input, a non-finite measured q, the furthest slot. | One gate for both robots; the configs name ROS topics. |
| G1 | The `waist_kp_min` reason was `Command validation failed on arm_sdk_mode: motor 12 (waist_yaw_joint) kp 0.0 is below arm_sdk.waist_kp_min 40.0`. | `Command validation failed on /safety/lowcmd_upper_in: motor 12 (waist_yaw_joint) kp 0.0 is below arm_sdk.waist_kp_min 40.0`. The waist slots are the profile's `waist_*` joints (12-14). | The reason names the input's topic, as every command reason now does (Task 5). |

### Quiet inputs and `/lowstate`

| Robot | Old | New | Reason |
|---|---|---|---|
| G1 | An input quiet for longer than its stale time latched the estop: the upper input after `UPPER_STALE_ESTOP_SECONDS` (2.0 s, `split` and `arm_sdk`), the leg input (`split` lower, or `full`) after `lower_stale_estop_seconds` (0.2 s; 1.0 s in `sim_split`), both from the input's first command. | Once the input has engaged, a gap longer than `stale.upper_seconds` or `stale.lower_seconds` (the same values) raises `{"age": ..., "event": "input_quiet", "seconds": ..., "source": ..., "topic": ...}` (Warn) and its last command keeps being relayed (in `arm_sdk`, at full weight); its next command raises `{"event": "input_resumed", "gap": ..., "source": ..., "topic": ...}` (Info). It never latches. | Spec item 2 (user's decision): quiet inputs are held and reported. |
| G1 | Once engaged, a `/lowstate` older than `state_stale_estop_seconds` (0.1 s; 1.0 s in the sim configs), or none at all, latched the estop (`rt/lowstate stale for ...`). | Once engaged, the gap raises `{"age": ..., "event": "state_quiet", "seconds": ..., "topic": "/lowstate"}` (Warn) and the relay keeps running; the next `/lowstate` raises `{"event": "state_resumed", "gap": ..., "topic": "/lowstate"}` (Info). It never latches. The arm_sdk release ramp still waits while `/lowstate` is older than `stale.state_seconds`. | Spec item 2: a quiet `/lowstate` leaves the relay running, as on the H1-2. |
| H1-2 | In `split_mode`, an upper input quiet for longer than `upper_stale_seconds` (2.0 s in `default_safety_split` and `tight_safety_split`; off in `relax_safety_split` and `sim_safety_split`) latched the estop (`Upper-body command stale for ...`). | `input_quiet` and `input_resumed` as in the G1 row, with `stale.upper_seconds`; never a latch. | Spec item 2. |
| H1-2 | The lower and full inputs and `/lowstate` had no stale report. | `input_quiet`/`input_resumed` for the lower and full inputs after `stale.lower_seconds` (0.2 s), and `state_quiet`/`state_resumed` after `stale.state_seconds` (0.1 s), in all 8 configs, `sim_safety_split` included. | The configs carry these times (Task 3). The H1-2 sim may now report gaps it never reported. |

### The hardware estop input

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2 | The relay engaged with no estop sample. A triggered or unplugged sample latched; after the first sample nothing checked its age, so a dead `estop_node` left its last "not triggered" sample in force for ever. | With `use_estop`: no input engages until the first estop sample has arrived (the zero-gain stream goes on, and `engage_refused` says why). From the first sample on, a `step()` more than `estop.timeout_seconds` (0.5 s) after the newest sample latches: `External estop silent: no sample on /estop/triggered or /estop/plugged_in for 0.510 s (timeout 0.5 s)`. A first sample that is triggered or unplugged latches at once, as any sample did. The age latch runs before engaging too. | Spec item 4 (user's decision) and Review Focus 5. `estop_node` publishes both topics at 100 Hz, 50 samples per timeout. |
| G1 | No hardware estop. | The same rules when the bringup passes `use_estop:=true`; without it the estop topics are ignored, as on the H1-2. | The G1 has the same optional hardware estop (Task 3). |

### Transport lag

| Robot | Old | New | Reason |
|---|---|---|---|
| H1-2 | On domain 0, `low_state arriving N ms late (> 100 ms) against the robot tick` latched from the first sample, timed on `time.time()`. | The same check, on domain 0 with `control.state_lag_ms` (100), timed on the boot-time `now`. It runs only on samples carrying `mode_machine` 6, and its reason names the topic: `/lowstate arriving 101 ms late (> 100 ms) against the robot tick`. | A wall-clock step must not fake or hide lag (spec item 2), and a foreign sample says nothing about this robot's tick. |
| G1 | No transport-lag check. | The G1's lag check stays off, as today: every G1 config carries `control.state_lag_ms: 0.0`, and a value <= 0 disables it. The mechanism is shared with the H1-2. With `engage_gate` the check would run from engagement, as the limit checks do. | Ruling 14: the G1's `/lowstate` tick has not been measured on the real robot. |

### Exit

| Robot | Old | New | Reason |
|---|---|---|---|
| G1 | `split` and `full`: on SIGINT/SIGTERM/SIGHUP, once engaged, the exit latched the estop with the exit reason (an `estop_command_written` report), wrote the estop command (`zero` or `damp`) for `EXIT_HOLD_SECONDS` (0.1 s) and then stopped. | `exit_step` returns nothing at once, so the robot holds the last command written. Neither the estop command on exit nor the 0.1 s exit hold is ported for these modes. An exit after an estop finishes that estop's sequence: with `keep_writing`, the estop command once if none has been written since the latch. | Spec item 8: the H1-2's exit behavior wins. |
| G1 | `arm_sdk`: the exit latched the estop with the exit reason, then held the last good command while the weight ramped to 0 over `release_seconds` (whatever the `/lowstate` age), wrote weight 0 for `EXIT_HOLD_SECONDS` (0.1 s), and stopped. | The ramp, which runs whatever the `/lowstate` age (an estop's release included), and the 0.1 s hold at weight 0 are kept. The exit no longer latches the estop: it writes no estop report and `estopped()` stays false. | Spec item 8 keeps the arm_sdk hand-back. The 0.1 s hold keeps one lost packet from leaving the robot on a higher weight. An exit is not an estop. |
| H1-2 | The relay stopped; the robot held its last command. With an estop latched, the burst was cut short. | Unchanged when not estopped. Once estopped, `exit_step` finishes the 50-command burst first. | A shutdown mid-burst should not leave the motors on the last live command. |
