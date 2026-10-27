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
