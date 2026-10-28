# golem_safety

The safety layer of the Unitree H1-2 and G1. Its node, `safety_node`, is the only writer of
the robots' body command topic, `/lowcmd`. It relays the controllers' commands from their
`/safety/*` inputs, clips them to the config's limits, checks every `/lowstate` against the
robot's identity and the estop limits, and latches an estop until restart. It is C++
(`rclcpp`) and talks to the robot only through ROS topics: ROS topic `/x` is DDS topic
`rt/x`, with the same `unitree_hg` wire type, so the robots, the sims and the Unitree SDK's
publishers all see the same topics.

The design is `docs/superpowers/specs/2026-10-27-golem-upper-ik-and-safety-design.md`
(section golem_safety). Every value and behavior that differs from the packages this one
replaces is in the change report beside it,
`docs/superpowers/specs/2026-10-27-golem-upper-ik-and-safety/change-report.md`.

## Running it

The bringups start the node (`golem_bringup`'s `h12_control.launch.py` and
`g1_core.launch.py`). On its own:

```bash
ros2 launch golem_safety safety.launch.py robot:=h12 config:=sim_safety_split sim:=true
ros2 run golem_safety safety_node --ros-args -p robot:=g1 -p config:=sim_split -p sim:=true
```

| Parameter | Default | Meaning |
|---|---|---|
| `robot` | required | `h12` or `g1`: the profile in `robots/<robot>.yaml` |
| `config` | required | a config name in `config/<robot>/` (with or without `.yaml`), or a path; a path whose directory is named after another robot with a profile (`.../g1/split.yaml` for the `h12`) is refused |
| `sim` | `false` | `true` for a simulator: the node then refuses ROS domain 0, the real robots' command bus, an unset `ROS_DOMAIN_ID` included |
| `use_estop` | `false` | `true` to read the hardware estop on the config's `estop` topics; `false` ignores them |

The node refuses to start, with one `FATAL` line and exit code 1, when a parameter is
missing, the profile or config does not load (`safety_node refuses to start: robot <robot>,
config <config>: <reason>`), or `sim:=true` meets domain 0 (`safety_node refuses to start:
sim:=true on ROS domain 0, ...`). It exits 1 as well when its relay loop
fails, when its executor thread (the heartbeat's) fails, after running the exit sequence,
or when its exit sequence overruns its length by more than 1 s
(`exit_hand_back_timeout`); 0 otherwise.

## ROS interface

| Direction | Topic | Type | QoS | When |
|---|---|---|---|---|
| sub | `/lowstate` | `unitree_hg/msg/LowState` | keep-last 1, best effort | always |
| sub | `/safety/lowcmd_in` | `unitree_hg/msg/LowCmd` | keep-last 1, best effort | `full_body_mode` |
| sub | `/safety/lowcmd_lower_in`, `/safety/lowcmd_upper_in` | `LowCmd` | keep-last 1, best effort | `split_mode` |
| sub | `/estop/triggered`, `/estop/plugged_in` | `std_msgs/Bool` | keep-last 1, best effort | `use_estop:=true` |
| pub | `/lowcmd` | `LowCmd` | reliable, keep-last 1 | always |
| pub | `/safety/heartbeat` | `std_msgs/Bool` | reliable, depth 10 | 10 Hz on the steady clock once relaying can begin: true, then false once the estop latches |

The topic names are the config's `topics` (and `estop`) values; the table gives the
defaults, which every carried config uses. `h12_lowerbody_rl`, `lmg_deploy` and `h12_skills`
read `/safety/heartbeat` on the H1-2, and `h12_lowerbody_rl` starts pre-posing the legs on
the first beat, so a beat means relaying can begin. The node publishes none until the
engage conditions that do not depend on the inputs hold: the newest `/lowstate` reports the
profile's `mode_machine` and, with `use_estop`, a first hardware estop sample has arrived
(`SafetyCore::relay_ready`). From then on it beats true, and false once the estop latches.
An estop that latches before then (a non-finite command with no `/lowstate` yet, a first
estop sample that says triggered) makes the beats false from the latch on, never true.
Before engaging, a `/lowstate` from another robot stops the beats again, which the readers
time out on.

## Robot profiles

`robots/<robot>.yaml` is the robot's data, written by `tools/generate_robot_data.py` (do not
edit it by hand). `golem_upper_ik` reads the same file, so each robot's joint table is
defined once.

| | `h12` | `g1` |
|---|---|---|
| `mode_machine` (what `/lowstate` reports and every command carries) | 6 | 5 |
| `mode_pr` | 0 | 0 |
| Body joints | 27, slots 0-26 | 29, slots 0-28 |
| Lower / upper slots | 0-11 / 12-26 (torso 12, arms 13-26) | 0-11 / 12-28 (waist 12-14, arms 15-28) |
| Modes | `full_body_mode`, `split_mode` | `split_mode`, `full_body_mode` |
| Joint limits | the H1-2's joint-limit table (shoulder roll ±0.38 rad, not the URDF's ±0.19) | `g1_description.joints.BODY`, from the URDF |

Both robots' `LowCmd`/`LowState` motor arrays hold 35 slots.

## Configs

| Robot | Config | Mode | Run by |
|---|---|---|---|
| H1-2 | `relax_safety_split` | split | `h12_real_robot` (the `h12_control.launch.py` default), `lmg_sim`, `lmg_real`, `h12_safety_grippers` (their defaults) |
| H1-2 | `default_safety_split`, `tight_safety_split` | split | by name (`safety_config:=`) |
| H1-2 | `sim_safety_split` | split | `h12_sim`, the Mac bringup, `tests/stack` |
| H1-2 | `default_safety_full`, `relax_safety_full`, `tight_safety_full`, `locomotion_safety_full` | full body | by name |
| G1 | `split` | split | `g1_real_robot.launch.py` (default) |
| G1 | `sim_split` | split | `g1_sim.launch.py` (default) |
| G1 | `full` | full body | refused by `g1_core.launch.py` (it must relay the upper-body input) |

What the robots' configs share:

- **H1-2:** `estop_response` `zero`, `burst_then_silence`, 50 commands;
  `control.state_lag_ms: 100`; `stale.lower_seconds` 0.2 and `state_seconds` 0.1;
  `stale.upper_seconds` 2.0 in `default_safety_split` and `tight_safety_split`, 0 (off) in
  `relax_safety_split` and `sim_safety_split`; logging off.
- **G1:** `estop_response` `zero`, `keep_writing`; `control.state_lag_ms: 0` (off: the real
  G1's tick has not been measured); logging on, to `/tmp/golem_safety`. `sim_split` relaxes
  the stale times and the estop limits, since the sim can run slower than real time.

Each safety config runs with one `golem_upper_ik` controller config; the bringups refuse
another pair (`golem_upper_ik.pairing`).

### Keys

| Key | Default | Meaning |
|---|---|---|
| `mode` | the profile's first mode | `full_body_mode` or `split_mode`; a mode the profile lacks is refused, naming the profile's modes |
| `topics.low_cmd_in`, `low_cmd_lower_in`, `low_cmd_upper_in` | `/safety/lowcmd_in`, `/safety/lowcmd_lower_in`, `/safety/lowcmd_upper_in` | the inputs; absolute ROS names (`rt/...` is refused) |
| `topics.low_cmd_out` | `/lowcmd` | the output |
| `topics.low_state` | `/lowstate` | the robot's state |
| `control.publish_hz` | 500 | the relay rate |
| `control.state_lag_ms` | required | the transport-lag estop, ms; on ROS domain 0 only; <= 0 turns it off |
| `stale.upper_seconds`, `lower_seconds`, `state_seconds` | 0, 0.2, 0.1 | quiet times of the upper input, the lower or full input and `/lowstate`, s; <= 0 turns the report off |
| `single_writer.handover_seconds` | 0.5 | how long the current writer must be quiet before another is accepted, s |
| `estop.triggered_topic`, `plugged_in_topic` | `/estop/triggered`, `/estop/plugged_in` | the hardware estop's topics |
| `estop.timeout_seconds` | 0.5 | the oldest a hardware estop sample may be, s |
| `estop_response.mode` | `zero` | `zero`: every motor off (mode 0, every value 0); `damp`: every body motor on with kp 0 and `kd` |
| `estop_response.kd` | none | `damp` only (required there): one number or one per body joint, at most `limits.clip.kd_max` |
| `estop_response.after` | required | `burst_then_silence` or `keep_writing` |
| `estop_response.burst_count` | 50 | the commands of `burst_then_silence` |
| `limits.clip.position_offset`, `velocity_ratio`, `torque_ratio` | 0.01, 0.5, 0.5 | one number, or one per body joint in slot order |
| `limits.clip.kp_max`, `kd_max` | 300, 30 | one number for every joint (`kp_abs`, `kd_abs` are aliases) |
| `limits.estop.position_offset`, `velocity_ratio`, `torque_ratio` | 0.01, 0.5, 0.5 | as `limits.clip`, for the measured state |
| `logging.enabled`, `base_dir`, `chunk_prefix` | `false`, `/tmp/golem_safety`, `chunk_` | the `.npz` sample log |

`limits.enforce` is an alias of `limits.clip`. The limits derive from the profile's joint
table: q within [low + `position_offset`, high - `position_offset`] (a negative offset widens
the range), |dq| up to velocity limit x `velocity_ratio`, |tau| up to effort limit x
`torque_ratio`. `golem_safety_dump_config <robot> <config>` (installed in
`lib/golem_safety`) prints every derived and parsed value as JSON.

The loader refuses, naming the robot, the file and the key: an unknown key at any level; a
key of an older schema (`network`, `robot`, `startup`, `engage`, `estop.enabled`, ...),
saying why it was removed or which key replaces it; a non-finite or out-of-range number; a
list whose length is not the profile's joint count; an alias given with the key it aliases;
a missing required key.

## Behavior

**Relay, newest only.** One relay thread runs at `control.publish_hz`, sleeping to absolute
deadlines on the boot-time clock. Each cycle it takes the newest `/lowstate`, the newest
hardware estop values and the newest command of each input (keep-last 1, so no queue builds
up), and publishes one command. A cycle that ends more than a period late restarts the
schedule: a slow cycle lowers the rate, and nothing is replayed to catch up. Every timer in
the node runs on the boot-time or steady clock, never on the wall clock or on `/clock`.

**What it writes.** Each engaged input's newest valid command on that input's own slots
(`full`: every body slot; `split`: the lower and upper slots). Every other slot, the slots
past the body and the `reserve` words carry `zero_command` (mode 0, every value 0). Every
command carries the profile's `mode_machine` and `mode_pr` and its own Unitree CRC (CRC-32
over the message body, as the SDK computes it); nothing of an input's header is copied.

**Clipping.** A command is valid when q, dq, tau, kp and kd are finite on every body slot,
the input's own or not; any other command latches the estop. A valid command is clipped on
its input's slots: q, dq and tau to `limits.clip`, kp and kd to [0, `kp_max`] and [0,
`kd_max`]. A gain above its maximum raises `gain_clipped` (Warn) once per joint and gain.

**Start-up**, the same on both robots (no config key chooses it): from boot the layer
streams `zero_command` on every slot, so every motor is off. Each input engages on its own
first valid command that arrives once the newest `/lowstate` reports the profile's
`mode_machine` (and, with `use_estop`, a first hardware estop sample has arrived); its slots
carry its commands from then on (`input_engaged`). The slots of an input that has not
engaged stay at zero gain: in split mode the robot runs arms-only, with its leg motors off,
until the leg node commands. The `/lowstate` limit checks run from boot.

A command that cannot engage raises `engage_refused` (Warn) with the first reason, at most
once a second: `no /lowstate received yet`, `/lowstate mode_machine is 0, expected 6`, or
`no hardware estop sample received yet on /estop/triggered or /estop/plugged_in`.

**Robot identity.** Every `/lowstate` is checked against the profile's `mode_machine`. Before
engaging, another value only refuses (another robot or sim delays engaging); once engaged, it
latches the estop (`/lowstate mode_machine 5 is not the h12's 6: another robot or sim is on
this domain`). A foreign sample is never limit-checked. `docker/scripts/wait_for_sim.py`
checks the same values before a paired bringup starts, and each robot's sims report them.

**Quiet inputs are held and reported, not estopped.** Once an input has engaged, a gap in
its commands longer than its stale time (`stale.upper_seconds` for the upper input,
`stale.lower_seconds` for the lower and full inputs) raises `input_quiet` (Warn) while its
last command keeps being relayed, and its next command raises `input_resumed` (Info). Once
engaged, a `/lowstate` gap longer than `stale.state_seconds` raises `state_quiet` and the
relay keeps running; the next sample raises `state_resumed`. Neither latches. So:

- a controller that dies, or a walking node that faults and goes silent, leaves its last
  command in force until the bringup stops. On the G1 in split mode the legs stay on a
  frozen target that does not balance the robot; the layer has no tilt or fall check and
  does not read the IMU;
- `frame_task_server`'s last command at exit is the one held: on the H1-2 every upper slot
  is at mode 0 and zero gains, so the torso and the arms go limp; in the G1's split mode the
  waist keeps its hold angle and gains and only the arms go limp.

**Single writer.** Each input's current writer is the DDS publisher of its first command. A
command from another publisher is accepted, and its publisher becomes current (logged, Info),
when the current writer's last command is older than `single_writer.handover_seconds`; a
restarted controller is accepted that way. Within that time it latches the estop (`second
writer on <topic>: ...`).

**Hardware estop** (`use_estop:=true`). The layer reads the `estop` node's two topics; a
sample is a value on each, taken every cycle. No input engages before the first sample. A
sample that says triggered (`External estop triggered`) or not plugged in (`External estop
unplugged`) latches, the first one included. From the first sample on, no new sample for
`estop.timeout_seconds` latches too (`External estop silent: ...`), before engaging as well.
On the H1-2 the bringups start `estop_node` `safety_delay` (3 s) before this node, and a
`safety_node` running while `estop_node` reports a failed connection latches
(`h12_control.launch.py`'s comment). Without `use_estop` the topics are ignored.

**Transport lag.** On ROS domain 0 with `control.state_lag_ms` > 0, a `/lowstate` (with the
profile's `mode_machine`) that arrives more than `state_lag_ms` later than the robot's
millisecond tick implies latches the estop. The baseline is the smallest arrival-minus-tick
offset seen, and may rise by 0.5 ms/s to follow clock drift. The sims stamp the tick in sim
steps, so the check runs only on domain 0.

**The estop latch.** One way until restart. Its triggers: a non-finite command value; a
second writer; once engaged, another robot's `mode_machine`; the transport lag; a measured q
past [`q_estop_low`, `q_estop_high`], |dq| or |tau_est| past its estop limit, or any of them
non-finite, on the slots of the mode's inputs (every body slot in both modes); and the
hardware estop. Each limit reason names the joint, the value, the limit and the multiple
(`motor 3 (left_knee_joint) dq out of estop range: 15 (limit 12.000, 1.25x)`), and a limit
latch also logs the commanded and measured arrays (`estop diagnostic`, Error). After the
latch:

- `burst_then_silence` (the H1-2): `burst_count` estop commands, then no output until
  restart;
- `keep_writing` (the G1): the estop command every cycle.

The estop report (Error) is `estop_command_written`: the reason, `decision_ms` (the latch),
`action_ms` (the first step after it) and, for the hardware estop, `trigger_stamp_ms`, all on
the boot-time clock. The heartbeat turns false, or starts false when the relay was not ready
yet.

**Exit** (SIGINT, SIGTERM, SIGHUP; `main()` takes them, and a closed stderr cannot stop the
exit). The layer stops publishing at once; an estop burst in progress is finished first, and
with `keep_writing` the estop command is written once if none has been since the latch. The
sims then turn every body motor off after `GOLEM_CMD_TIMEOUT` (0.5 s of sim time) of
`rt/lowcmd` silence; what the real robots do when it stops has not been measured.

## Logging

Every event goes to the node's ROS logger at its level, from a thread of its own (a stalled
stderr cannot stall the relay): `[INFO] [<time>] [safety_node]: <event>`, and `/rosout`.
Most events are one JSON object with sorted keys after the `]: `: `relay_started` (the
robot, the config file, the mode, `sim`, `use_estop` and the log directory),
`engage_refused`, `input_engaged`, `input_quiet`, `input_resumed`, `state_quiet`,
`state_resumed`, `gain_clipped`, `estop_command_written`, `exit_requested`,
`exit_hand_back_timeout` and `relay_stopped`.

With `logging.enabled`, each run logs to its own directory,
`<logging.base_dir>/<YYYYmmdd-HHMMSS>-<pid>/`, in chunks of 1000 samples
(`<chunk_prefix>000000.npz`, ...): one sample per `/lowstate` the relay took before the
latch (the latching one included), with its motor fields and IMU, and the last command
written. `combine_chunks.py` (installed in `lib/golem_safety`) merges one run's chunks:

```bash
ros2 run golem_safety combine_chunks.py --load /tmp/golem_safety/<run> --save /tmp/combined.npz
```

## Tests

In the G1-mode container (the H1-2 and the G1 tests both run there; `--restart` replaces a
running `golem_ros` container):

```bash
docker/scripts/docker_run.sh ros --restart --g1 /home/code/h12_sim_scripts/launch_ros.sh bash -lc \
  'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 && cd /home/code/core_ws && colcon build --packages-select golem_safety && source install/setup.bash && colcon test --packages-select golem_safety --event-handlers console_direct+; colcon test-result --verbose'
```

- gtests: `SafetyCore` (`test_safety_core_relay`, `test_safety_core_startup`: the relay,
  clipping, identity, single writer, quiet inputs, the start-up, the hardware estop,
  the latch responses, exit), the config loader (`test_config`), the profile loader, the
  `LowCmd` builders and CRC (`test_lowcmd`, against SDK-computed values), the chunk logger
  and the boot clock.
- `test/test_node_ros.py` runs the installed `safety_node` against a fake robot over ROS on
  `ROS_DOMAIN_ID` 91 (`test/conftest.py`), for both robots. `test_spinner` (the executor
  thread) runs on 91 too.
- `test_limit_parity.py` compares every config's derived limits with fixtures from the
  previous safety layers' loaders, `test_robot_data.py` the profiles with their sources,
  `test_chunk_files.py` the chunks with numpy and `combine_chunks.py`, and `test_no_sdk.py`
  that nothing imports `unitree_sdk2py`.

A pytest reported as skipped means pytest was not usable when CMake configured the package
(export `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` before building), not a pass.

Do not run the suite on a host whose robot stack is up: `test_sim_refuses_domain_zero`
starts the node with `sim:=true` on ROS domain 0, the real robots' command bus, on the
loopback interface only. It refuses before it creates any command publisher, so nothing is
commanded, but a stack on the same host (the G1 laptop's `golem_ros`, for one) can be
discovered over loopback and fail the test.

## Tools

- `tools/relay_latency_probe.py --robot h12|g1`: on a running stack, the relay's output
  rate and input-to-output latency, as JSON (it refuses domain 0).
- `tools/generate_robot_data.py` writes `robots/h12.yaml` from `tools/h12_joint_limits.py`
  and `robots/g1.yaml` from `g1_description`, so it runs only where `g1_description`
  imports. The test fixtures are recorded data (`test/fixtures/README.md`).
