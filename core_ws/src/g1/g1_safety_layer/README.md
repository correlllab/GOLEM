# g1_safety_layer

Safety layer for the Unitree G1 (29 DoF). It filters low-level commands, monitors
joint states, and enforces clipping and estop logic before forwarding commands
to the robot on `rt/arm_sdk` or `rt/lowcmd`.

Derived from [correlllab/h12_safety_layer](https://github.com/correlllab/h12_safety_layer)
(H1-2).
The joint names, slot order and URDF limits come from `g1_description`.

## Modes

- **arm_sdk mode** (default, `config/arm_sdk.yaml`): our controller drives the
    waist and arms over `rt/arm_sdk` while Unitree's controller keeps the legs
    balanced.
    - Nothing is published until an upper-body command passes the engage gate
      (below).
    - The weight slot (motor 29) then ramps from 0 to 1 over `engage_seconds`.
    - The leg slots (0-11) are written as zeros, as in Unitree's example.
    - Only the waist and arm slots (12-28) are checked, and only once engaged.
    - A command with a waist kp (slots 12-14) below `arm_sdk.waist_kp_min` (40)
      latches the estop: a stray publisher's default command has kp 0 there.
    - On estop the last good command is held while the weight ramps back to 0
      over `release_seconds` (3 s in `arm_sdk.yaml`), handing the waist and arms
      back to Unitree.
    - The release ramp only advances while `rt/lowstate` is fresh: with the
      link down nothing reaches the robot, which would see the weight jump when
      the link returned. Until then the held command keeps the weight reached.
- **Split mode** (`config/split.yaml`): legs (slots 0-11) from a lower-body
    controller and waist + arms (12-28) from the upper-body controller, merged
    and published on `rt/lowcmd`.
    - Nothing is published until both inputs have sent a command and the
      merged command passes the engage gate; the layer then logs
      `lowcmd_engage` and starts checking `rt/lowstate`.
    - Once the lower input has sent a command, a gap longer than
      `control.lower_stale_estop_seconds` (0.2 s) triggers the estop.
    - Once the upper input has sent a command, a gap longer than 2 s triggers
      the estop.
- **Full-body mode** (`config/full.yaml`): a single upstream `low_cmd` published
    on `rt/lowcmd`.
    - Nothing is published until a command passes the engage gate; the layer
      then logs `lowcmd_engage` and starts checking `rt/lowstate`.
    - After that, a gap longer than `control.lower_stale_estop_seconds` (0.2 s)
      triggers the estop.
- The engage gate: the layer takes control only with a command that passes
    three checks.
    - The last `rt/lowstate` is newer than `engage.max_state_age_seconds` (0.1 s).
    - Its `mode_machine` is `engage.mode_machine` (5, our unit's).
    - The command is within `engage.max_position_error` (0.15 rad) of the
      measured q on every checked slot (12-28 in arm_sdk mode, all 29 otherwise).
    - Until then the layer logs `engage_refused` with the reason, at most once a
      second.
- Once engaged, an `rt/lowstate` older than `control.state_stale_estop_seconds`
    (0.1 s) latches the estop.
- A sample from a second DDS writer on an input topic latches the estop before
    it is used: two publishers would interleave their commands.
    - Each sample's `sample_info.publication_handle` names its writer.
    - A matched-writer count is not used: cyclonedds-python 0.10.2 has no
      matched-publication query on the reader, the count only comes through an
      `on_subscription_matched` listener that unitree_sdk2py does not install,
      and that listener runs on another thread than the SDK's data handler, so
      nothing orders it before the second writer's first sample.
    - A writer that is matched but never writes is not detected; it commands
      nothing.
    - Restarting the controller while the safety layer runs is a new writer
      too, so it latches the estop: restart both.
- In split and full-body mode Unitree's controller must be released first
    (`motion_switcher`).
- In split and full-body mode `estop_response.mode` sets the estop command
    (arm_sdk mode ignores it):
    - `zero` (default): every motor off, so the robot drops and must be
      supported (e.g. on a gantry).
    - `damp`: every motor stays enabled with `kp = 0`, `q = dq = tau = 0` and
      `estop_response.kd` (at most `kd_max`), so the robot sinks instead of
      dropping.
- An estop is logged as `estop_command_written` once its command is written.
    An estop before the layer engages writes nothing, keeping the layer silent,
    and is logged as `estop_latched_silent` instead.
- A kp or kd above `kp_max`/`kd_max` is clipped and logged as a `gain_clipped`
    event, once per joint and gain, on the slots the input drives.
- In every `rt/lowcmd` mode the header gets `mode_pr = 0` (pitch/roll ankles) and
    the `mode_machine` read from `rt/lowstate`.
- The estop is latched: restart the safety layer to clear it.
- SIGINT, SIGTERM and SIGHUP (Ctrl-C, `ros2 launch` shutting down, `docker stop`,
    a closed terminal or SSH session) hand control back before the process exits:
    - arm_sdk mode holds the last good command while the weight ramps to 0 over
      `release_seconds`, even with `rt/lowstate` stale: nothing is published
      after the exit, and a ramp that might be lost is no worse than stopping
      at the current weight.
    - Split and full-body mode write the estop command.
    - The final command is written for 0.1 s; a layer that never engaged exits
      at once. Further signals change nothing; SIGKILL cannot be caught.
    - Ctrl-C on `ros2 launch` reaches the node directly (the terminal signals
      the whole process group), so the 3 s ramp ends well before launch's
      SIGTERM (5 s after SIGINT).
    - `docker stop` sends SIGTERM to the container's PID 1 only, and SIGKILL
      10 s later. It hands back when PID 1 is the node itself (`exec`). An
      interactive bash (`launch_ros.sh` ends in `exec bash`) ignores SIGTERM,
      and Humble's `ros2 launch` as PID 1 catches it but signals none of its
      nodes: the node is then killed after 10 s at weight 1. Ctrl-C the launch
      before stopping such a container.
    - With `--stop-signal SIGINT`, `ros2 launch` as PID 1 passes the SIGINT on
      at once when its stdin is not a terminal, so the hand-back ends about
      3.1 s after `docker stop`. With a terminal (`tty: true`, as the compose
      `ros` service has) it expects the terminal to have signalled its nodes
      and only sends SIGTERM 5 s later, so the hand-back ends about 8.1 s
      after `docker stop`, inside its 10 s.
- The node exits non-zero when the safety layer fails to start, stops by
    itself, or does not finish handing control back within `release_seconds`
    plus 1.1 s.
- Events (JSON lines) and messages go to stdout through a thread of their own
    (`core/console.py`), so stdout can never stop the publisher loop or the
    hand-back:
    - Once a write fails (a killed `ros2 launch`, a closed terminal), stdout is
      pointed at `/dev/null` and every later line is dropped.
    - While stdout is stalled (a reader that stopped reading), up to 10000
      lines wait; later ones are dropped and then counted in a
      `stdout_lines_dropped` event.
    - On exit the node waits up to 1 s for the last lines (twice: when the
      layer stops, and before exiting). If stdout is still stalled it then
      exits at once, since Python's own exit would wait on stdout forever.
- Every watchdog and weight ramp is timed on CLOCK_BOOTTIME: it keeps counting
    through a laptop suspend, so a gap spanning one is seen, and NTP or `date`
    never step it. `time.time()` can be stepped; `time.monotonic()` stops while
    suspended.
- Each config declares `robot: real` or `robot: sim`. `sim_arm_sdk.yaml` and
    `sim_split.yaml` are the same modes on DDS domain 1 with relaxed estop limits,
    and refuse DDS domain 0 (`ROS_DOMAIN_ID=0` included).
    - The node checks this before rclpy starts, and gives rclpy the domain the
      Unitree channels use: with `ROS_DOMAIN_ID` unset, rclpy alone would join
      domain 0.
- Both sim configs allow a 1 s gap on `rt/lowstate` (engage gate and watchdog),
    and `sim_split.yaml` a 1 s gap on the lower input: the sim can run slower
    than real time, while the watchdogs measure real time.
- `G1_SAFETY_LOG_DIR`, when set (e.g. a mounted host path), replaces
    `logging.base_dir`; each run logs to its own subdirectory there.

## Files

- `config/`: YAML configuration files.
- `docs/`: detailed documentation.
- `g1_safety_layer/`: source code.
    - `core/`: core implementation of the safety layer.
        - `arm_sdk.py` computes the `rt/arm_sdk` engage and release weight ramps.
        - `chunk_logger.py` provides logger of robot states and commands.
        - `clock.py` is the clock every watchdog and ramp is timed with.
        - `config.py` parses and loads YAML configuration files.
        - `console.py` writes stdout lines from a thread of its own.
        - `exit_signals.py` turns SIGINT, SIGTERM and SIGHUP into an exit request.
        - `joint_limits.py` exposes joint names and URDF limits from `g1_description`.
        - `safety_checks.py` defines safety checks of robot states and commands.
        - `safety_layer.py` implements the safety layer that relays commands, checks safety and logs in the background.
    - `script/`: runnable scripts.
        - `safety_layer_main.py` launches the safety layer without ROS.
    - `utility/`: utility scripts.
        - `combine_chunks.py` combines chunked logs.
        - `record_limits.py` records joint limits to tweak YAML configuration files.
        - `publisher.py` is a dummy publisher.
        - `subscriber.py` is a dummy subscriber.
- `test/`: unit tests plus end-to-end DDS tests of every mode.

## Usage

- ROS 2 entry point, after building and sourcing the workspace:

    ```bash
    ros2 run g1_safety_layer safety_node --config arm_sdk.yaml
    ros2 run g1_safety_layer safety_node --config split.yaml
    ```

- Without ROS, from this directory (needs `unitree_sdk2py`, numpy and PyYAML):

    ```bash
    python3 g1_safety_layer/script/safety_layer_main.py --config arm_sdk.yaml
    ```

- `--config` can be a bare config name (resolved from `config/`) or an absolute path.
- The script asks for confirmation before using DDS domain 0, the real robot.
- `uv` users: `pyproject.toml` and `uv.lock` describe this package (`g1-safety-layer`)
    with the repo's `unitree_sdk2_python` as an editable workspace member.

## Tests

```bash
PYTHONPATH=.:../g1_description python3 -m pytest test/test_g1_safety.py \
    test/test_arm_sdk_mode_dds.py test/test_lowcmd_modes_dds.py \
    test/test_hardening_dds.py test/test_node_exit.py
```

The DDS tests run on domain 97; they never touch the robot.
`test_node_exit.py` runs the node and the script as subprocesses and signals
them, also with their stdout closed or stalled; its node cases need ROS
(`rclpy`) and are skipped without it.

## Detailed Documentation

- [YAML Configuration Definition](docs/config.md)
- [Safety Layer Architecture](docs/architecture.md)
