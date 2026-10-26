# YAML Configuration Definition

[Back to README](../README.md)

## Overview

The YAML configuration files define how the safety layer operates,
including DDS topic names, network settings, control parameters, safety limits, and logging behavior.
All configuration files are located in the `config/` directory.

- [arm_sdk mode, real robot (default)](../config/arm_sdk.yaml).
- [Split mode, real robot](../config/split.yaml).
- [Full-body mode, real robot](../config/full.yaml).
- [arm_sdk mode, simulation](../config/sim_arm_sdk.yaml).
- [Split mode, simulation](../config/sim_split.yaml).

The limits in these files are starting values, not yet tuned on our G1.
`arm_sdk.yaml` carries deliberately conservative first-session values; its
comments derive them from the URDF limits in `g1_description`.

## Configuration Settings

### `robot`

What the config is written for.

- **Type:** string, required (no default).
- **Options:**
    - `real`: the real G1 (`arm_sdk.yaml`, `split.yaml`, `full.yaml`).
    - `sim`: the simulated G1 (`sim_arm_sdk.yaml`, `sim_split.yaml`). A sim config refuses DDS domain 0, the real robot: `network.domain_id: 0` fails to load, and `ROS_DOMAIN_ID=0` fails before DDS or rclpy starts.

### `mode`

Specifies the operational mode of the safety layer.

- **Type:** string.
- **Options:**
    - `arm_sdk_mode`: One upper-body `low_cmd` input, published on `rt/arm_sdk` with the weight slot managed here.
    - `split_mode`: Two separate `low_cmd` inputs for lower-body and upper-body control.
    - `full_body_mode`: Single `low_cmd` input controlling the entire robot.
- **Default:** `arm_sdk_mode`.

### `topics`

Defines the DDS topic names used by the safety layer for communication.

- **`low_cmd_out`** (string): Output topic where filtered commands are published.
    - Default: `rt/arm_sdk` in `arm_sdk_mode`, `rt/lowcmd` otherwise.
- **`low_state`** (string): Input topic for monitoring current robot state.
    - Default: `rt/lowstate`.
- **`low_cmd_upper_in`** (string): Upper-body input (`arm_sdk_mode`, `split_mode`).
    - Default: `rt/safety/lowcmd_upper_in`.
- **`low_cmd_lower_in`** (string): Lower-body input (`split_mode`).
    - Default: `rt/safety/lowcmd_lower_in`.
- **`low_cmd_in`** (string): Single input (`full_body_mode`).
    - Default: `rt/safety/lowcmd_in`.

### `network`

Configures DDS Domain ID and network interface settings.

- **`domain_id`** (integer): DDS Domain ID for inter-process communication.
    - Default: `0`.
    - Effective value: `ROS_DOMAIN_ID` when that environment variable is set; otherwise `network.domain_id` from the YAML config.
    - When `ROS_DOMAIN_ID` is unset, this value must match the robot's domain ID.
    - A `robot: sim` config refuses an effective domain of `0`.
    - `safety_node` runs its rclpy node on the effective domain too.
- **`interface`** (string): Network interface name (empty string uses default interface).
    - Default: `''` (auto-detect).
    - Example: `enp130s0`.

### `control`

Specifies control loop parameters.

- **`publish_hz`** (float): Frequency (Hz) at which filtered commands are published.
    - Default: `500.0`.
- **`lower_stale_estop_seconds`** (float): Longest gap allowed on the leg input (`low_cmd_lower_in` in `split_mode`, `low_cmd_in` in `full_body_mode`) once it has sent a command; a longer gap latches the estop.
    - Default: `0.2` (`sim_split.yaml` uses `1.0`, since the sim can run slower than real time). Must be positive and finite.
    - Ignored in `arm_sdk_mode`. The upper-body input's limit is fixed at 2 s.
- **`state_stale_estop_seconds`** (float): Once the layer has engaged, the longest gap allowed on `rt/lowstate`; a longer gap latches the estop.
    - Default: `0.1` (`sim_*.yaml` use `1.0`). Must be positive and finite.
    - In `arm_sdk_mode` the release ramp then waits until `rt/lowstate` is newer than this again (see `arm_sdk`).

All watchdogs and ramps are timed on CLOCK_BOOTTIME (`core/clock.py`), which keeps counting through a suspend and is never stepped.

### `engage`

The engage gate: the layer takes control (starts the `rt/arm_sdk` weight ramp, or starts publishing `rt/lowcmd`) only with a command that passes all three checks.
Until then it publishes nothing and logs `engage_refused` with the reason, at most once a second.
In `split_mode` the merged command is checked, once both inputs have sent.

- **`max_state_age_seconds`** (float): `rt/lowstate` must be newer than this.
    - Default: `0.1` (`sim_*.yaml` use `1.0`). Must be positive and finite.
- **`mode_machine`** (integer): `rt/lowstate`'s `mode_machine` must equal this.
    - Default: `5`, what our unit reports (and the sim).
- **`max_position_error`** (float): The command's q must be within this of `rt/lowstate`'s q on every checked slot: the waist and arms (12-28) in `arm_sdk_mode`, all 29 otherwise.
    - Units: radians.
    - Default: `0.15`. Must be positive and finite.

### `arm_sdk`

Weight-slot ramps and the waist floor for `arm_sdk_mode` (ignored in other modes).

- **`engage_seconds`** (float): Time to ramp the weight from 0 to 1 once a command passes the engage gate.
    - Default: `1.0`. `0` engages immediately.
- **`release_seconds`** (float): Time to ramp the weight back to 0 on estop or on SIGINT/SIGTERM/SIGHUP, holding the last good command.
    - Default: `1.0` (`arm_sdk.yaml` sets `3.0`, as Unitree's example releases).
    - On estop the ramp advances only while `rt/lowstate` is newer than `control.state_stale_estop_seconds`; until then the weight reached is held. On SIGINT/SIGTERM/SIGHUP it runs regardless, since nothing is published after the exit.
- **`waist_kp_min`** (float): A command with a kp below this on a waist slot (12-14) fails validation and latches the estop.
    - Units: N·m/rad.
    - Default: `40.0`. Must be nonnegative, finite and at most the waist `limits.clip.kp_max`.
    - From the URDF, gravity pulls the torso over with 22.9 N·m/rad (arms at `home`) to 33.8 N·m/rad (arms straight forward) about waist roll and pitch; the controller sends 60.

### `estop_response`

The command the estop writes on `rt/lowcmd` in `split_mode` and `full_body_mode`.
Ignored in `arm_sdk_mode`, whose estop always hands the waist and arms back to Unitree.

- **`mode`** (string): The estop command.
    - `zero`: every motor off (`mode = 0`, `kp = kd = 0`), so the robot drops and must be supported (e.g. on a gantry).
    - `damp`: every motor stays enabled (`mode = 1`) with `kp = 0`, `q = dq = tau = 0` and the `kd` below, so the robot sinks instead of dropping.
    - Default: `zero`.
- **`kd`** (float or list): Damping gain of the `damp` command, a scalar or a 29-entry list in `rt/lowcmd` slot order.
    - Units: N·m·s/rad.
    - Required when `mode` is `damp`; no default. The shipped configs set `8.0`, a starting value not yet tuned on our G1.
    - Must be nonnegative, finite and at most `limits.clip.kd_max`; checked whenever it is set, even with `mode: zero`.

### `limits`

Defines safety thresholds for command clipping and emergency stop logic.
All fields under `limits` can be either:

- **Scalar**: Single value applied to all 29 joints.
- **List**: 29-entry list in `rt/lowcmd` slot order for per-joint customization.

Ratios scale the URDF limits in `g1_description`.

#### `clip` Section

Used to constrain commands within safe operating bounds without triggering emergency stop.

- **`position_offset`** (float or list): Margin kept inside each URDF position limit.
    - Units: radians.
    - Default: `0.01`.
- **`velocity_ratio`** (float or list): Maximum commanded velocity as a ratio of the joint's URDF velocity.
    - Default: `0.5`.
- **`torque_ratio`** (float or list): Maximum commanded feedforward torque as a ratio of the joint's URDF effort.
    - Default: `0.5`.
- **`kp_max`** (float or list): Maximum proportional gain.
    - Units: N·m/rad.
    - Default: `300.0`.
- **`kd_max`** (float or list): Maximum derivative gain.
    - Units: N·m·s/rad.
    - Default: `30.0`.

A kp or kd clipped to `kp_max`/`kd_max` is logged as a `gain_clipped` event, once per joint and gain, on the slots the input drives (12-28 in `arm_sdk_mode`).
`estop_response.kd` may not exceed `kd_max` either.

#### `estop` Section

Checked against `rt/lowstate` once the layer has engaged, that is once a command has passed the engage gate (see `engage`).
When violated, the safety layer latches estop.
In `arm_sdk_mode` only the waist and arm slots (12-28) are checked.

- **`position_offset`** (float or list): Margin inside each URDF position limit.
    - Units: radians. A negative value widens the limit.
    - Default: `0.01`.
- **`velocity_ratio`** (float or list): Maximum measured velocity as a ratio of the joint's URDF velocity.
    - Default: `0.5`.
- **`torque_ratio`** (float or list): Maximum estimated torque as a ratio of the joint's URDF effort.
    - Default: `0.5`.

A top-level `estop:` section (the H1-2's external estop) is rejected with an error.

### `logging`

Controls logging behavior for recording robot states and commands.

- **`enabled`** (boolean): Whether to enable logging.
    - Default: `true`.
- **`base_dir`** (string): Directory path where log chunks are stored.
    - Default: `/tmp/g1_safety_layer`.
    - Must have write permissions.
    - `G1_SAFETY_LOG_DIR`, when set (e.g. a mounted host path), replaces it: each run then logs to `$G1_SAFETY_LOG_DIR/<YYYYmmdd-HHMMSS>-<pid>/`, because the logger clears its directory's chunks when it starts.
- **`chunk_prefix`** (string): Prefix for chunked log filenames.
    - Default: `chunk_`.
    - Logs are named as `{chunk_prefix}{timestamp}.npz`.

## Per-Joint Configuration Example

When using per-joint lists, the order is the G1 `rt/lowcmd` slot order
(`g1_description.layout`):

```text
Indices 0-5:   Left leg (hip pitch, roll, yaw, knee, ankle pitch, roll)
Indices 6-11:  Right leg (same order)
Indices 12-14: Waist (yaw, roll, pitch)
Indices 15-21: Left arm (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw)
Indices 22-28: Right arm (same order)
```

Example of per-joint `velocity_ratio` in `estop`:

```yaml
estop:
  velocity_ratio:
    - 0.50 # left_hip_pitch_joint
    - 0.50 # left_hip_roll_joint
    - 0.50 # left_hip_yaw_joint
    # ... (continue for all 29 joints)
```

## Tips for Tuning

1. **Start Conservative**: Begin with tight safety limits and relax gradually as needed.
2. **Joint-Specific Tuning**: Different joints have different safety requirements; use per-joint lists for critical joints.
3. **Test Incrementally**: Make small changes and test thoroughly between adjustments.
4. **Monitor Logs**: Check logged data to understand clipping and estop triggers.
5. **Document Changes**: Keep notes on why specific limits were chosen for your use case.
