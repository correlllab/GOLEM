# Safety layer

`golem_safety` (node `safety_node`) is the only node that writes the robot's
body command topic, `rt/lowcmd`. The H1-2 runs the same node, with its own
robot profile and configs. Every controller publishes to one of its
`rt/safety/*` input topics instead, and `golem_upper_ik` and `g1_homie`
refuse any other command topic. It relays the newest command of each input,
clips it, checks every `rt/lowstate`, and latches an estop when something goes
wrong.

Every parameter, config key and event is in the
[package README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/golem_safety/README.md); the G1's values are in
[`config/g1/`](https://github.com/correlllab/GOLEM/tree/main/core_ws/src/golem_safety/config/g1). This page names the
keys rather than copying their values.

> **Not covered:** the Dex3 hand driver (`dex_3`) writes
> `rt/dex3/<side>/cmd` directly and bypasses the safety layer.

## Modes

The safety config picks the mode. Each one names its input and output topics
under `topics:` (ROS names: ROS topic `/x` is DDS topic `rt/x`). Both write
every body motor on `rt/lowcmd`: the G1 runs in Unitree's debug mode, where
Unitree's own controller is released and nothing but `rt/lowcmd` drives the
motors ([Control in sim and on the robot](control-path.md)).

| Mode | Configs | Input topics | Who commands the legs |
|---|---|---|---|
| split | `split.yaml` (real robot), `sim_split.yaml` (sim) | `rt/safety/lowcmd_upper_in` (arm controller: waist and arms) and `rt/safety/lowcmd_lower_in` (walking node: legs) | the walking node |
| full body | `full.yaml` | `rt/safety/lowcmd_in` | one upstream controller; the bringup refuses this mode |

`golem_bringup`'s `g1_core.launch.py` runs each safety config with exactly one
arm-controller config (`G1_SAFETY_CONTROLLER_CONFIGS` in
`core_ws/src/golem_upper_ik/golem_upper_ik/pairing.py`) and refuses the real
configs in the sim and the `sim_*` ones on the robot. With `sim:=true`, which
the sim bringup passes, `safety_node` refuses ROS domain 0, an unset
`ROS_DOMAIN_ID` included.

## Start-up

The G1 starts as the H1-2 does; no config key chooses it.

1. **Zero gains from boot.** From its start the layer writes `rt/lowcmd` at
   `control.publish_hz` with mode 0 and every value 0 on every slot, stamped
   with `mode_machine` 5: every motor is off. It checks every `rt/lowstate`
   against the estop limits from the first sample (below).
2. **The heartbeat.** Once the newest `rt/lowstate` reports `mode_machine` 5,
   the layer beats true on `/safety/heartbeat` at 10 Hz: relaying can begin.
   It beats false once its estop latches, and never true if the estop latched
   first. HOMIE's walking node starts only on the first true beat.
3. **Each input engages on its own first valid command** from then on, and
   logs `input_engaged` with its `source` (`split_mode_upper` for the arm
   controller, `split_mode_lower` for the walking node). Its slots carry its
   commands from then on; the slots of an input that has not engaged stay at
   zero gain. So the arms run on the controller's first command, and the leg
   motors stay off, the legs hanging limp, until the walking node commands
   them.

There is no distance check: an input's first command is relayed wherever it
points. The arm controller's first command is the measured pose, held until
its follow gate sees the layer relay it, and the walking node moves the legs
from their measured positions to its default pose over 2 s.

A command that cannot engage logs `engage_refused` with the first reason, at
most once a second: `no /lowstate received yet`, or `/lowstate mode_machine
is <n>, expected 5`.

## Quiet inputs are held, not estopped

Once engaged, an input with no new command for its `stale.upper_seconds` or
`stale.lower_seconds` keeps its last command relayed and is reported
(`input_quiet`, then `input_resumed` when it sends again). A quiet
`rt/lowstate` is reported the same way (`state_quiet`, `state_resumed`), and
the relay keeps running. Nothing latches on either.

- **A controller that dies without its last command** leaves the waist and
  arms held on its last command until the bringup stops.
- **A walking-node fault holds the legs, it does not turn the motors off.** A
  fault (a stale `rt/lowstate`, a non-finite IMU value, a fall, a command later
  than `faults.max_latency`) silences `locomotion_node`.
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
- a second live DDS writer on an input topic (a new writer is accepted once the
  old one has been quiet for `single_writer.handover_seconds`, 0.5 s, so a
  restarted controller is not refused);
- once an input has engaged, a `rt/lowstate` whose `mode_machine` is not 5:
  another robot or sim is on the domain (before that, such a sample only
  refuses);
- a measured joint position, speed or torque past its estop limit
  (`limits.estop`), or not finite, on any body slot, from the first
  `rt/lowstate` on. A limp leg hanging past its limit at start-up latches too:
  `split.yaml`'s position limits are the URDF's ranges exactly
  (`limits.estop.position_offset: 0.0`), while `sim_split.yaml`'s are relaxed;
- with `use_estop:=true`, a hardware estop that is triggered, unplugged or
  silent. The G1 bringup passes `use_estop: false`: the G1 has no estop node.

The transport-lag check (`control.state_lag_ms`) is off in every G1 config.

An estop writes the estop command on `rt/lowcmd` every cycle until restart
(`estop_response`: `mode: zero`, `after: keep_writing` in every G1 config).
`zero` turns every motor off, so the robot drops and must be on a gantry. The
heartbeat turns false, which stops the walking node for good.

The estop stays latched. **After any estop, restart the whole bringup**: the
arm controller's follow gate opens for good once the safety layer relays its
commands, so a restarted safety layer alone would meet a command held wherever
the robot used to be.

## When the arm controller stops

`frame_task_server` (`golem_upper_ik`) ends by sending one last command, then
stops publishing. It does so on Ctrl+C or SIGTERM, and when its start-up fails
after its first command. The arms get mode 0 and zero kp, kd and tau; the
waist keeps its hold angle, gains and torque (the G1 profile's
`shutdown_keeps_held_joints_on_lowcmd`), because the walking policy was
trained with a stiff waist and a limp one slumps onto it. `golem_safety`
relays the command, so **the arms go limp** while the waist holds and the
walking node keeps the legs. That stays so: the quiet upper input keeps that
command relayed. No estop latches. `tests/g1/test_sim_bringup.sh` (with and
without `--walk`) stops `frame_task_server` while the walking node runs and
checks that the waist roll and pitch stay at their hold.

## Stopping

Ctrl+C, SIGTERM and SIGHUP stop the layer the same way: it stops publishing
`rt/lowcmd` at once (after a latch, it first writes the estop command once if
none has gone out since the latch). The sim then turns every motor off after
`GOLEM_CMD_TIMEOUT` (0.5 s of sim time); what the real G1's motors do when
`rt/lowcmd` stops has not been measured ([Control in sim and on the
robot](control-path.md)). `docker stop` can kill the node before it finishes,
so Ctrl+C the launch first.

## Logs

Events go to the node's ROS logger at their level, so they print as
`[INFO] [<time>] [safety_node]: {...}` (or `[WARN]`, `[ERROR]`) and reach
`/rosout`. Most are one JSON object after the `]: `. The main ones:

| Event | Meaning |
|---|---|
| `relay_started` | The layer is up and streams zero gains; it names the robot, the config, the mode and the log directory. |
| `engage_refused` | A command could not engage; carries the reason. |
| `input_engaged` | An input engaged; `source` names it (`split_mode_upper`, `split_mode_lower`, `full_body_mode`). |
| `input_quiet`, `input_resumed` | An input went quiet (its last command keeps being relayed), or sent again. |
| `state_quiet`, `state_resumed` | `rt/lowstate` went quiet, or came back. |
| `gain_clipped` | A commanded kp or kd above `limits.clip.kp_max` or `kd_max` (once per joint and gain). |
| `estop_command_written` | An estop latched and its first command went out; carries the reason. A limit latch also logs `estop diagnostic` with the commanded and measured joints. |

State and commands are also logged to `.npz` chunks: each run writes its own
`<logging.base_dir>/<YYYYmmdd-HHMMSS>-<pid>/` (`/tmp/golem_safety` in the
G1 configs, inside the container), named in `relay_started`. Copy them out
before the container exits; `combine_chunks.py` (installed in
`lib/golem_safety`) merges one run's chunks.

## Run it alone

```bash
ros2 launch golem_safety safety.launch.py robot:=g1 config:=sim_split sim:=true
```

Its tests are in the README's Tests section; they run the installed node
against a fake robot over ROS on domain 91.
