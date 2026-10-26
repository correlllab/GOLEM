# Safety Layer Architecture

[Back to README](../README.md)

## Overview

The safety layer is a relay that intercepts command messages to the G1 robot, applies safety checks and clipping, monitors the robot's state for dangerous conditions, and forwards filtered commands to the robot. It operates in three modes: **arm_sdk Mode** (upper-body input shared with Unitree's controller on `rt/arm_sdk`, the default), **Split Mode** (separate lower/upper body inputs on `rt/lowcmd`) and **Full-Body Mode** (single command input on `rt/lowcmd`). The split and full-body data flows are drawn below; arm_sdk mode uses the upper-body half of split mode with no lower input.

## Core Components

### Command Subscriptions

- **arm_sdk Mode**: Single subscriber on `low_cmd_upper_in` topic.
- **Full-Body Mode**: Single subscriber on `low_cmd_in` topic.
- **Split Mode**: Two subscribers on `low_cmd_lower_in` and `low_cmd_upper_in` topics.

### Safety Checks Layer

- **Clipping**: Constrains commands within safe bounds without halting.
- **Engage gate**: The layer takes control only with a command that passes it: `rt/lowstate` newer than `engage.max_state_age_seconds`, its `mode_machine` equal to `engage.mode_machine`, and the command within `engage.max_position_error` of the measured q on every checked slot (12-28 in arm_sdk mode, all 29 otherwise). Refusals are logged as `engage_refused`, at most once a second.
- **Estop (Emergency Stop)**: Monitors robot state and incoming commands for dangerous conditions.
    - Triggered on: extreme position deviation, excessive velocity, unrealistic torque commands, non-finite commands, and in arm_sdk mode a waist kp below `arm_sdk.waist_kp_min`.
    - Also triggered when an input goes quiet after its first command: the lower-body or full-body input for longer than `lower_stale_estop_seconds`, the upper-body input for longer than 2 s.
    - Also triggered, once engaged, when `rt/lowstate` is older than `state_stale_estop_seconds`.
    - Also triggered by a sample from a second DDS writer on an input (its `sample_info.publication_handle` differs from the first writer's); that sample is not used.
    - State limits are checked only once the layer has engaged (see [State Subscriber](#state-subscriber)).
    - Effect on `rt/lowcmd`, set by `estop_response.mode`: `zero` (default) turns every motor off (mode 0, `kp = kd = tau = 0`); `damp` keeps every motor enabled (mode 1) with `kp = 0`, `q = dq = tau = 0` and `estop_response.kd`.
    - Effect on `rt/arm_sdk`: the last good command is held while the weight slot ramps to 0 over `release_seconds`, handing the waist and arms back to Unitree's controller. The ramp advances only while `rt/lowstate` is fresh; while it is stale the weight reached is held.
    - Before the layer engages nothing is written in any mode, and the estop is logged as `estop_latched_silent` instead of `estop_command_written`.
    - The estop is latched until restart.

### Publisher Loop

- Runs at configurable frequency (default 500 Hz).
- Publishes filtered commands to `low_cmd_out` topic.
- Runs the upper-body and lower-body input watchdogs, each from its input's first command, and once engaged the `rt/lowstate` watchdog.
- Times every watchdog and ramp, and its own period, on CLOCK_BOOTTIME (`core/clock.py`): it keeps counting through a suspend and is never stepped, unlike `time.time()`.
- In `rt/lowcmd` modes, stamps `mode_pr = 0` and the `mode_machine` last read from `low_state`.
- In arm_sdk mode, stays silent until a command passes the engage gate, then ramps the weight slot (motor 29) from 0 to 1 over `engage_seconds`; the leg slots (0-11) are written as zeros.
- In split and full-body mode, stays silent until every input has sent a command and the command passes the engage gate, so the zeroed boot command never reaches `rt/lowcmd`.
- Is the only writer of `LowCmd_` in the process, and must stay so: cyclonedds-python 0.10 serializes every sample of a type through one buffer of that type without a lock, so two threads writing `LowCmd_` at once can mix their samples. The in-process DDS tests, which write the inputs from the same process, serialize one sample at a time (`test/conftest.py`).

### State Subscriber

- Monitors `low_state` topic for robot joint/IMU data.
- Triggers estop if state exceeds safety thresholds, once the layer has engaged (a command has passed the engage gate). Until then there is nothing of ours to guard, and a joint resting past its limit must not latch the estop.
- Records each message's receive time and q, for the engage gate and the `rt/lowstate` watchdog, also while estopped.
- Provides data for logging.

### Exit Path

- `safety_node` and `safety_layer_main.py` handle SIGINT, SIGTERM and SIGHUP themselves (rclpy's handlers are off) and call `SafetyLayer.shutdown()`; later signals change nothing.
- In arm_sdk mode, `shutdown()` holds the last good command while the weight ramps to 0 over `release_seconds`, even with `rt/lowstate` stale, since nothing is published after the exit.
- In split and full-body mode, it writes the estop command (`estop_response`).
- The final command is written for 0.1 s, then the layer stops; a layer that never engaged stops at once, writing nothing.
- `SafetyLayer.stop()` alone stops at once, without handing back.
- The node exits non-zero when the safety layer fails to start, its publisher loop dies, or `shutdown()` does not finish within `release_seconds` + 1.1 s.
- Before `rclpy.init`, `safety_node` resolves the DDS domain from the config (a sim config refuses domain 0 there) and passes it to `rclpy.init`, so rclpy and the Unitree channels share one domain.

### Console

- Events and messages are queued for stdout and written by a daemon thread (`core/console.py`); nothing writes stdout under `_lock`, in the publisher loop or on the exit path.
- Once a write fails (EPIPE from a killed `ros2 launch`, EIO from a closed terminal), stdout's file descriptor is pointed at `/dev/null`, so no later write, nor Python's exit flush, can fail on it, and every later line is dropped.
- While stdout is stalled, up to 10000 lines wait; later ones are dropped and counted in a `stdout_lines_dropped` event.
- `stop()` waits up to 1 s for the queued lines; the entry points wait up to 1 s more and, if stdout is still stalled, exit with `os._exit`, since Python's exit flush would wait on stdout forever.

### Async Logger

- Records robot state and command data to HDF5 chunks.
- Runs in separate thread to avoid blocking control loop.
- Configurable chunk size and write frequency.

## Data Flow Diagrams

### Full-Body Mode

```
┌─────────────────┐
│  low_cmd_in     │  External command source
│    topic        │
└────────┬────────┘
         │
         ↓
    ┌─────────────────────────────────┐
    │  Command Handler (_on_low_cmd   │
    │       _full)                    │
    │                                 │
    │  • Receive command              │
    │  • Validate format              │
    │  • Apply clipping               │
    │  • Update desired_cmd_full      │
    └──────────┬──────────────────────┘
               │
               |  Clipped command
               │  stored
               │
    ┌──────────▼──────────┐         ┌─────────────────┐
    │  Publisher Loop     │◄───────►│  low_state      │
    │  (500 Hz)           │         │    topic        │
    │                     │         └────────┬────────┘
    │  • Get desired cmd  │                  │
    │  • Check estop      │         State monitoring
    │  • Publish          │         • Position limits
    │  • Update log cache │         • Velocity limits
    └──────────┬──────────┘         • Torque limits
               │
               ↓
    ┌──────────────────────┐
    │  low_cmd_out topic   │
    │                      │
    │ (to robot)           │
    └──────────────────────┘
               │
               ↓
    ┌──────────────────────-┐
    │  G1 (rt/lowcmd)       |
    └──────────────────────-┘
```

### Split Mode

```
┌──────────────────────┐              ┌──────────────────────┐
│ low_cmd_lower_in     │              │ low_cmd_upper_in     │
│   topic              │              │   topic              │
│ (legs + lower body)  │              │ (torso + arms)       │
└──────────┬───────────┘              └──────────┬───────────┘
           │                                     │
           ↓                                     ↓
    ┌────────────────────┐           ┌────────────────────┐
    │ Lower Handler      │           │ Upper Handler      │
    │ (_on_low_cmd       │           │ (_on_low_cmd_upper)│
    │  _lower)           │           │                    │
    │                    │           │ • Validate         │
    │ • Validate         │           │ • Clip             │
    │ • Clip             │           │ • Update           │
    │ • Update           │           │   desired_cmd_upper│
    │   desired_cmd_lower│           │                    │
    └──────────┬─────────┘           └──────────-┬────────┘
               │                                 │
               │    Clipped commands             │
               │    stored separately            │
               │                                 │
               └────────────┬────────────────────┘
                            │
                            ↓
                    ┌───────────────────┐
                    │  Merge Step       │
                    │                   │
                    │ Combine lower     │
                    │ (indices 0-11)    │
                    │ with upper        │
                    │ (indices 12-28)   │
                    │ into single       │
                    │ LowCmd message    │
                    └─────────┬─────────┘
                              │
              | Merged command
              │
    ┌─────────▼────────────┐         ┌─────────────────┐
    │  Publisher Loop      │◄───────►│  low_state      │
    │  (500 Hz)            │         │    topic        │
    │                      │         └────────┬────────┘
    │  • Merge split cmds  │                  │
    │  • Check estop       │         State monitoring
    │  • Publish           │         • Position limits
    │  • Update log cache  │         • Velocity limits
    └─────────┬────────────┘         • Torque limits
              │
              ↓
    ┌──────────────────────┐
    │  low_cmd_out topic   │
    │                      │
    │ (single merged cmd   │
    │  to robot)           │
    └──────────────────────┘
              │
              ↓
    ┌──────────────────────┐
    │ G1 (rt/lowcmd)       |
    └──────────────────────┘
```

## Safety Check Execution

### Command Validation (Clipping)

1. Receives incoming `LowCmd` message on callback thread
2. Calls `clip_low_cmd()` to constrain values:
    - Position: clipped to the URDF limits shrunk by `position_offset`
    - Velocity: scaled to max `velocity_ratio × joint_max_velocity`
    - Torque: scaled to max `torque_ratio × joint_max_torque`
    - Gains (Kp, Kd): clamped to specified maximums
3. Stores clipped command as `_desired_cmd_*` (protected by lock)
4. Returns early if validation fails → triggers estop

### State Monitoring (Estop)

1. Receives `LowState` message on callback thread
2. Once the layer has engaged, calls `check_estop_limits()` with stricter thresholds
3. If any joint violates limits:
    - Sets `_estop = True` and queues the estop event with the violation details
    - Queues debug info for stdout (position, velocity, acceleration commands vs actual)
4. Publisher loop detects `_estop`, publishes the estop command (see above) and logs the event

### Publisher Loop

1. Every `dt = 1/publish_hz` seconds, under `_lock`:
    - If not estopped: checks the upper-body watchdog (2 s, `split_mode` and `arm_sdk_mode`) and the lower-body watchdog (`lower_stale_estop_seconds`, `split_mode` and `full_body_mode`), each from its input's first command, and once engaged the `rt/lowstate` watchdog (`state_stale_estop_seconds`)
    - If estopped: builds `make_estop_cmd()` (`rt/lowcmd`, zero or damp per `estop_response`) or the release ramp (`rt/arm_sdk`), which advances only while `rt/lowstate` is fresh, or always while exiting
    - Else: builds the stored `_desired_cmd` (merged in split mode, with the engage ramp in arm_sdk mode)
    - Before the layer engages there is nothing to build, estop or not, and the loop stays silent
2. Stamps the `rt/lowcmd` header, calculates the CRC checksum, publishes, and caches the published command for logging
3. Logs a new estop once: `estop_command_written` with its timing, or `estop_latched_silent` if nothing was written, then the events decided under the lock (`arm_sdk_release_held`, `arm_sdk_release_ramping`)
4. While exiting, signals `shutdown()` once the final command has been written for 0.1 s

## Synchronization

The safety layer uses a single lock (`_lock`) to protect shared state:

- `_desired_cmd_full` / `_desired_cmd_lower` / `_desired_cmd_upper`.
- `_estop` and `_estop_reason`.
- `_last_cmd` and command component caches (`_last_q_cmd`, `_last_dq_cmd`, etc.).
- The engage and release state, the input and `rt/lowstate` receive times, the latest measured q, each input's first writer, and the exit state.

Lock is acquired:

- In command callbacks when updating desired commands.
- In state callback when checking limits.
- In publisher loop when reading desired command and estop state.

This ensures no race conditions between state monitoring, command updates, and publishing.

## Data Logging

When enabled, the logger collects per-cycle:

- **Timestamp**: Unix time when state message arrived.
- **Motor State**: Mode, position (q), velocity (dq), acceleration (ddq), torque, temperature, voltage.
- **IMU State**: Quaternion, gyroscope, accelerometer, roll-pitch-yaw, temperature.
- **Command**: Last published q_cmd, dq_cmd, tau_cmd, kp_cmd, kd_cmd.

Data is:

- Buffered in-memory (up to `LOG_MAX_QUEUE_SIZE` samples).
- Written to disk asynchronously at `LOG_WRITE_HZ` (5 Hz default).
- Stored as npz "chunks" (e.g., `chunk_1234.npz`) containing up to `LOG_SAMPLES_PER_CHUNK` (1000) samples.
- Combined post-run using `combine_chunks.py` utility.
