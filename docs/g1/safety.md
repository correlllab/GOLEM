# Safety layer

`g1_safety_layer` (node `safety_node`) is the only node that writes the
robot's body command topics, `rt/arm_sdk` and `rt/lowcmd`. Every controller
publishes to one of its `rt/safety/*` input topics instead, and `g1_control`
and `g1_locomotion` refuse any other command topic. It clips each command,
decides when to take control, and latches an estop when something goes wrong.

Every config key, limit and event is in the
[package README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_safety_layer/README.md) and
[`docs/config.md`](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_safety_layer/docs/config.md) there; the values are in
[`config/`](https://github.com/correlllab/GOLEM/tree/main/core_ws/src/g1/g1_safety_layer/config). This page names the
keys rather than copying their values.

> **Not covered:** the Dex3 hand driver (`g1_dex3`) writes
> `rt/dex3/<side>/cmd` directly and bypasses the safety layer.

## Modes

The safety config picks the mode. Each one names its input and output topics
under `topics:`.

| Mode | Configs | Input topics | Output | Who holds the legs |
|---|---|---|---|---|
| arm_sdk (default) | `arm_sdk.yaml`, `sim_arm_sdk.yaml` | `rt/safety/lowcmd_upper_in` (arm controller) | `rt/arm_sdk` | Unitree's controller (in the sim, the HOMIE stand-in) |
| split (walking) | `split.yaml`, `sim_split.yaml` | `rt/safety/lowcmd_upper_in` and `rt/safety/lowcmd_lower_in` (walking node) | `rt/lowcmd` | the walking node |
| full body | `full.yaml` | `rt/safety/lowcmd_in` | `rt/lowcmd` | one upstream controller; the bringup refuses this mode |

In arm_sdk mode, slot 29 of `rt/arm_sdk` is a weight: 1 means our commands,
0 means Unitree's. The safety layer ramps it in over `arm_sdk.engage_seconds`
and back out over `arm_sdk.release_seconds` on any fault.

`g1_bringup` runs each safety config with exactly one arm-controller config
(`SAFETY_CONTROLLER_CONFIGS` in
`core_ws/src/g1/g1_control/g1_control/utility/config_pairing.py`) and refuses
the real configs in the sim and the `sim_*` ones on the robot. The `sim_*`
configs also refuse DDS domain 0.

## Taking control (the engage gate)

The layer publishes nothing until a command passes three checks:

1. `rt/lowstate` is fresh: no older than `engage.max_state_age_seconds`
   (looser in the sim configs, since the sim can run slower than real time).
2. `rt/lowstate` reports `mode_machine` `engage.mode_machine` (5, our unit's).
3. The command is within `engage.max_position_error` of the measured joints.

Until then it logs `engage_refused` with the reason. In split mode both inputs
must have sent a command first.

## What latches an estop

- a non-finite command, or a waist kp below `arm_sdk.waist_kp_min` (arm_sdk
  mode);
- a second DDS writer on an input topic;
- the arm controller silent for 2 s, or the walking node silent for
  `control.lower_stale_estop_seconds`;
- `rt/lowstate` stale once engaged (`control.state_stale_estop_seconds`);
- a measured joint position, speed or torque past its estop limit
  (`limits.estop`).

| Mode | What an estop does |
|---|---|
| arm_sdk | Hold the last good command and ramp the weight to 0 over `arm_sdk.release_seconds`. The waist and arms go back to Unitree. The ramp advances only while `rt/lowstate` is fresh. |
| split, full body | Write the estop command on `rt/lowcmd` (`estop_response.mode`). The default, `zero`, turns every motor off, so the robot drops and must be on a gantry. |

The estop stays latched. **After any estop, restart the whole bringup**: the
arm controller's follow gate opens for good once the safety layer relays its
commands, so a restarted safety layer alone would meet a command held wherever
the robot used to be. Restarting the controller alone is a new DDS writer,
which latches the estop too.

## Stopping

Ctrl+C, SIGTERM and SIGHUP give the same hand-back before the node exits.
`docker stop` can kill the node before it finishes, so Ctrl+C the launch first.
The package README's Modes section has the timings.

## Logs

Events print as JSON lines on stdout. The main ones:

| Event | Meaning |
|---|---|
| `relay_started` | The layer is up. |
| `engage_refused` | The engage gate refused a command. |
| `arm_sdk_engage`, `lowcmd_engage` | The layer took control. |
| `estop_command_written` | An estop fired; carries the reason. |
| `estop_latched_silent` | An estop latched before engaging; nothing was written. |
| `arm_sdk_release_ramping`, `arm_sdk_release_held` | The arm_sdk hand-back ramp runs, or waits for `rt/lowstate`. |

State and commands are also logged to `.npz` chunks under `logging.base_dir`
(`/tmp/g1_safety_layer` inside the container; `G1_SAFETY_LOG_DIR` overrides
it). Copy them out before the container exits.

## Run it alone

```bash
ros2 run g1_safety_layer safety_node --config sim_arm_sdk.yaml
```

Its tests are in the README's Tests section; the DDS tests use domain 97 and
fake robot peers.
