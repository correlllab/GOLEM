# Safety layer

`golem_safety` (node `safety_node`) is the only node that writes the robot's
body command topics, `rt/arm_sdk` and `rt/lowcmd`. The H1-2 runs the same
node, with its own robot profile and configs. Every controller publishes to
one of its `rt/safety/*` input topics instead, and `golem_upper_ik` and
`g1_locomotion` refuse any other command topic. It relays the newest command of
each input, clips it, decides when to take control, checks every `rt/lowstate`,
and latches an estop when something goes wrong.

Every parameter, config key and event is in the
[package README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/golem_safety/README.md); the G1's values are in
[`config/g1/`](https://github.com/correlllab/GOLEM/tree/main/core_ws/src/golem_safety/config/g1). This page names the
keys rather than copying their values.

> **Not covered:** the Dex3 hand driver (`g1_dex3`) writes
> `rt/dex3/<side>/cmd` directly and bypasses the safety layer.

## Modes

The safety config picks the mode. Each one names its input and output topics
under `topics:` (ROS names: ROS topic `/x` is DDS topic `rt/x`).

| Mode | Configs | Input topics | Output | Who holds the legs |
|---|---|---|---|---|
| arm_sdk (default) | `arm_sdk.yaml`, `sim_arm_sdk.yaml` | `rt/safety/lowcmd_upper_in` (arm controller) | `rt/arm_sdk` | Unitree's controller (in the sim, the HOMIE stand-in) |
| split (walking) | `split.yaml`, `sim_split.yaml` | `rt/safety/lowcmd_upper_in` and `rt/safety/lowcmd_lower_in` (walking node) | `rt/lowcmd` | the walking node |
| full body | `full.yaml` | `rt/safety/lowcmd_in` | `rt/lowcmd` | one upstream controller; the bringup refuses this mode |

In arm_sdk mode, slot 29 of `rt/arm_sdk` is a weight: 1 means our commands,
0 means Unitree's. The safety layer ramps it in over `arm_sdk.engage_seconds`
and back out over `arm_sdk.release_seconds` on an estop and when it exits.

`golem_bringup`'s `g1_core.launch.py` runs each safety config with exactly one
arm-controller config
(`G1_SAFETY_CONTROLLER_CONFIGS` in
`core_ws/src/golem_upper_ik/golem_upper_ik/pairing.py`) and refuses the real
configs in the sim and the `sim_*` ones on the robot. With `sim:=true`, which
the sim bringup passes, `safety_node` refuses ROS domain 0, an unset
`ROS_DOMAIN_ID` included.

## Taking control (the engage gate)

The layer publishes nothing until a command passes these checks:

1. `rt/lowstate` is fresh: no older than `engage.max_state_age_seconds`
   (looser in the sim configs, since the sim can run slower than real time).
2. `rt/lowstate` reports `mode_machine` 5, the G1 profile's.
3. Every input of the mode has sent a command, and none is quiet for longer
   than its `stale.*_seconds`. In split mode that means the walking node too.
4. Every command is within `engage.max_position_error` of the measured joints.

Until then it logs `engage_refused` with the first reason, at most once a
second. In arm_sdk mode the weight then ramps in from 0.

## Quiet inputs are held, not estopped

Once engaged, an input with no new command for its `stale.upper_seconds` or
`stale.lower_seconds` keeps its last command relayed and is reported
(`input_quiet`, then `input_resumed` when it sends again). A quiet
`rt/lowstate` is reported the same way (`state_quiet`, `state_resumed`), and
the relay keeps running. Nothing latches on either.

- **A controller that dies without its last command** leaves the waist and
  arms held on its last command, at weight 1 in arm_sdk mode, until the
  bringup stops.
- **Split mode: a walking-node fault holds the legs, it does not turn the
  motors off.** A fault (a stale `rt/lowstate`, a non-finite IMU value, a fall,
  a command later than `faults.max_latency`) silences `locomotion_node`.
  `golem_safety` then keeps relaying its last leg command at 500 Hz, with the
  waist and arms, and reports `input_quiet` after `stale.lower_seconds`. That
  frozen leg target does not balance the robot. `golem_safety` has no tilt or
  fall check of its own, and does not check the IMU: the command stands until a
  `rt/lowstate` latches the estop (below), for example a fall that drives a
  joint past `limits.estop`, or the bringup stops. A non-finite joint q, dq or
  tau_est is different: `golem_safety` latches on that sample, and `split.yaml`'s
  estop turns every motor off. Keep the robot on a gantry.

## What latches an estop

- a command with a non-finite q, dq, tau, kp or kd on any body slot;
- in arm_sdk mode, a command whose waist kp is below `arm_sdk.waist_kp_min`
  (the controller's own exit command is one, below);
- a second live DDS writer on an input topic (a new writer is accepted once the
  old one has been quiet for `single_writer.handover_seconds`, 0.5 s, so a
  restarted controller is not refused);
- once engaged, a `rt/lowstate` whose `mode_machine` is not 5: another robot or
  sim is on the domain;
- once engaged, a measured joint position, speed or torque past its estop limit
  (`limits.estop`), or not finite, on the mode's slots (in arm_sdk mode the
  waist and arms only). In split mode `split.yaml`'s estop then turns every
  motor off;
- with `use_estop:=true`, a hardware estop that is triggered, unplugged or
  silent. The G1 bringup passes `use_estop: false`: the G1 has no estop node.

The transport-lag check (`control.state_lag_ms`) is off in every G1 config.

| Mode | What an estop does |
|---|---|
| arm_sdk | Hold the last good command and ramp the weight to 0 over `arm_sdk.release_seconds`. The waist and arms go back to Unitree. The ramp advances only while `rt/lowstate` is no older than `stale.state_seconds`. |
| split, full body | Write the estop command on `rt/lowcmd` every cycle (`estop_response.mode`). Every G1 config has `zero`, which turns every motor off, so the robot drops and must be on a gantry. |

A latch before the engage gate has passed writes nothing
(`estop_latched_silent`).

The estop stays latched. **After any estop, restart the whole bringup**: the
arm controller's follow gate opens for good once the safety layer relays its
commands, so a restarted safety layer alone would meet a command held wherever
the robot used to be.

## When the arm controller stops

`frame_task_server` (`golem_upper_ik`) ends by sending one last command, then
stops publishing. It does so on Ctrl+C or SIGTERM, and when its start-up fails
after its first command. The arms always get mode 0 and zero kp, kd and tau;
the waist gets them too in arm_sdk mode, and keeps its hold in split mode. What
follows depends on the mode:

- **arm_sdk mode:** `golem_safety` refuses that command, because its waist kp
  is below `arm_sdk.waist_kp_min`. It **latches its estop**, logged as one:
  `estop_command_written` with the reason `Command validation failed on
  /safety/lowcmd_upper_in: motor 12 (waist_yaw_joint) kp 0.0 is below
  arm_sdk.waist_kp_min 40.0`. It then holds the last good command and ramps
  the weight to 0, handing the waist and arms back to Unitree. Restart the
  bringup before commanding the arms again.
- **split mode:** the waist keeps its hold angle, gains and torque, because
  the walking policy was trained with a stiff waist and a limp one slumps onto
  it. `golem_safety` relays the command, so **the arms go limp** (mode 0, zero
  gains) while the waist holds and the walking node keeps the legs. That stays
  so: the quiet upper input keeps that command relayed.
  `tests/g1/test_sim_bringup.sh --walk` stops `frame_task_server` while the
  walking node runs and checks that the waist roll and pitch stay at their
  hold.

## Stopping

Ctrl+C, SIGTERM and SIGHUP stop the layer the same way. In arm_sdk mode it
first ramps the weight to 0 over `arm_sdk.release_seconds` (whatever the
`rt/lowstate` age) and writes weight 0 for 0.1 s more. In split and full-body
mode it stops publishing `rt/lowcmd` at once. The sim then turns every motor off
after `GOLEM_CMD_TIMEOUT` (0.5 s of sim time); what the real G1's motors do when
`rt/lowcmd` stops has not been measured ([Control in sim and on the
robot](control-path.md)). `docker stop` can kill the node before it finishes,
so Ctrl+C the launch first.

## Logs

Events go to the node's ROS logger at their level, so they print as
`[INFO] [<time>] [safety_node]: {...}` (or `[WARN]`, `[ERROR]`) and reach
`/rosout`. Most are one JSON object after the `]: `. The main ones:

| Event | Meaning |
|---|---|
| `relay_started` | The layer is up; it names the robot, the config, the mode and the log directory. |
| `engage_refused` | The engage gate refused a command; carries the reason. |
| `arm_sdk_engage`, `lowcmd_engage` | The layer took control. |
| `input_quiet`, `input_resumed` | An input went quiet (its last command keeps being relayed), or sent again. |
| `state_quiet`, `state_resumed` | `rt/lowstate` went quiet, or came back. |
| `estop_command_written` | An estop latched and its first command went out; carries the reason. |
| `estop_latched_silent` | An estop latched before engaging; nothing was written. |
| `arm_sdk_release_ramping`, `arm_sdk_release_held` | The arm_sdk hand-back ramp runs, or waits for `rt/lowstate`. |

State and commands are also logged to `.npz` chunks: each run writes its own
`<logging.base_dir>/<YYYYmmdd-HHMMSS>-<pid>/` (`/tmp/golem_safety` in the
G1 configs, inside the container), named in `relay_started`. Copy them out
before the container exits; `combine_chunks.py` (installed in
`lib/golem_safety`) merges one run's chunks.

## Run it alone

```bash
ros2 launch golem_safety safety.launch.py robot:=g1 config:=sim_arm_sdk sim:=true
```

Its tests are in the README's Tests section; they run the installed node
against a fake robot over ROS on domain 91.
