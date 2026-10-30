# golem_upper_ik and golem_safety: one arm controller and one safety layer for both robots

Status: design, 2026-10-27. Base: `camera_ready` @ `4de2506`.

## Intent

GOLEM runs two humanoids, the H1-2 and the G1. Each has its own safety layer and arm controller:

| Today | Replaced by |
|---|---|
| `h12_safety_layer` (submodule, Python) + `g1_safety_layer` (in-tree, Python) | `golem_safety` (in-tree, **C++/rclcpp**) |
| `h12_ros2_controller` (submodule, Python) + `g1_control` (in-tree, Python) | `golem_upper_ik` (in-tree, Python) |

Why:
- **The G1 packages are forks of older H1-2 code.** `g1_control` forked from `h12_ros2_controller` at `87bca34`; `g1_safety_layer` from an earlier H1-2 safety layer.
- **The forks have drifted, and each lacks fixes the other has.**
  - The G1 safety layer can relay stale commands: its 10-deep SDK queue drops the *newest* sample when full.
  - The G1 controller keeps a controller-side estop the H1-2 removed as harmful (`deacb9a`).
  - The H1-2 controller never waits for a fresh state before enabling motors.

  With one implementation, every fix lands on both robots.

The redundancy audit behind this is in `2026-10-27-golem-upper-ik-and-safety/redundancy-audit.md`, next to this file:
- Section A covers the safety layers. Its tables 2a and 2b list each side's fixes the other lacks.
- Section B covers the controllers (F1-F7).

The audit describes the code before this work. One claim in its section F is wrong, as its header notes. Its facts are restated here where they matter.

The end state:
- Both real robots, and both sims, run `golem_safety` and `golem_upper_ik`.
- The four old packages are deleted later, when the user says so.

**Where the two robots' behavior conflicts, the H1-2's wins.** It is the robot that runs on hardware today (user's decision).

## Goals

1. **One behavior, both robots.** Each package behaves the same on both robots. Everything that differs between robots is data in a **robot profile**:
   - joint names, motor slots and limits;
   - URDF, optional collision model and frames;
   - the modes the robot supports;
   - output topics and header values.

   Where a robot needs its own start-up rule (see golem_safety, item 4), that rule is a config choice, and each robot's configs choose today's behavior.
2. **ROS only.** Both packages talk to the robot and to each other only through ROS 2 topics, services and actions, using the `unitree_hg` messages built in `core_ws/src/unitree_ros2`.
   - No `unitree_sdk2py`: no `ChannelFactoryInitialize`, `ChannelPublisher`/`ChannelSubscriber`, SDK CRC or SDK IDL types.
   - ROS topic `/lowcmd` is DDS topic `rt/lowcmd` with the same wire type, so the sims (Unitree DDS) and the other publishers keep working unchanged.
3. **Safety behavior: H1-2 first, plus the G1's checks where they don't conflict.**
   - Relay only the newest command.
   - Watch every input and `/lowstate` for staleness, and report it.
   - When an input goes quiet after any input has engaged, **hold the last command** instead of going limp (user's decision).
4. **Controller behavior: the better of both controllers**, on both robots, as listed in "golem_upper_ik". Both support `FrameTask.relative`, `FrameTask.slow_mode` and `NamedConfig.slow_mode`.
5. **Collision-aware IK and OMPL planning wherever the profile has a self-collision model.** The H1-2 has one. The G1 has none and runs without both, as it does today; building a G1 model is later work (user's decision).
6. **Bringups switch to the new packages with no fallback.** If the new packages fail, the bringup fails loudly. No silent fallback to the old packages, and no quietly reduced rates or features.

## Non-goals

- **Deleting the four old packages.** That is a later step, on the user's word. Until then they stay in the tree, unchanged and unused. They still build, but nothing launches them.
- **The parts of `h12_ros2_controller` that no bringup runs.** These are not carried over:
  - gravity compensation (`gravity_comp_server`);
  - `dual_arm_server` and `/dual_arm`;
  - the counter-balance, ZMP and Crocoddyl DDP research code and its tests;
  - `scripts/`, `archive/` and `example/`;
  - the Inspire-hand path (`hand_controller_node`, broken since `NUM_HAND_DOF = 0`);
  - the meshcat visualiser;
  - the research configs (`balance_*`, `counter_*`, `real_benchmark_*`, `legacy/`).

  `g1_control`'s `example/` and its `dual_arm_*` are dropped the same way.
- **Changes to the lower body, locomotion, hands or skills packages**, beyond what keeps them working with the new packages: `h12_lowerbody_rl`, `lmg_deploy`, `h12_deploy_mjpc`, `g1_locomotion`, `g1_dex3`, `h12_skills`.
  - `g1_locomotion` keeps its SDK publisher on `rt/safety/lowcmd_lower_in`; it is wire-compatible.
  - `lmg_deploy` keeps reading `/safety/heartbeat` and knows nothing about the hardware estop (user's decision).
- **Real-robot runs.** The user runs the real H1-2 and the real G1.
- **A G1 self-collision model.** Deferred by the user. `golem_upper_ik` turns on the collision barrier and the planner for any profile that names a sphere URDF and SRDF, so a later G1 model is a data change.

## Anti-goals

These would make the work a failure even if every test passes:

- **An unlisted change to a value the H1-2 runs on hardware today.** That covers limits, ratios, gains, rates, timeouts, start-up postures and named configurations. Every value or behavior that has to change is listed in the change report for the user.
- **A second writer of the body command topics.** `golem_safety` stays the only publisher of `/lowcmd` and `/arm_sdk`. `golem_upper_ik` refuses any command topic outside `/safety/`.
- **A silent degradation.** For example, publishing commands below the configured rate, or running without a check the config asks for. Fail and say why.
- **Robot data defined twice.** The joint tables and limits are defined once per robot. Today the H1-2's are copied, identical, in two submodules.

## Constraints

- **Location:** `core_ws/src/golem_safety` and `core_ws/src/golem_upper_ik`, plain directories in this repo, not submodules. Both are built for both robots. `docker/scripts/launch_ros.sh` filters only `^g1_`.
- **The H1-2 build and run must not need `third_party/leprechaun_g1` or any `g1_*` package.** The H1-2 builds with `--packages-ignore-regex '^g1_'`, and `g1_description` symlinks into leprechaun, which is mounted only for the G1. G1 data (for example `g1_description`'s URDF path) is reached at run time, and only when the G1 profile is selected.
- **The real H1-2 builds natively** on its onboard PC (ROS Humble, no Docker), which already builds C++ packages (`fast_lio`, `livox_ros_driver2`). The real G1 runs in the laptop's `golem_ros` container. The Mac (`docker/mac`, arm64, FastDDS, H1-2 only) builds an explicit package list in `docker/mac/scripts/launch_ros_mac.sh`, and must build and run both new packages.
- **Dependencies:** new pip dependencies go through `docker/locks` (CLAUDE.md §8). Prefer libraries already in the ROS Humble images (rclcpp, yaml-cpp, …) over new third-party code.
- **Rates:**
  - The safety layer publishes at the configured `publish_hz` (500 today, both robots).
  - The arm controller publishes at the configured `pub_hz` (500) and solves IK at `ctrl_hz` (30).
  - These are measured, not assumed. If `golem_upper_ik` cannot sustain 500 Hz through rclpy in the `golem_ros` container, the builder stops and reports to the user rather than lowering the rate.
- **DDS:**
  - `ROS_DOMAIN_ID=0` is the real robots' bus.
  - The G1's network-interface pin reaches ROS processes through `cyclonedds_local.xml` (CLAUDE.md §3).
  - One robot stack per domain.
- **Submodule changes:** the `CL_isaaclab_sim` change is committed in that repo, on its `camera_ready` branch (the pin, `6f89c64`, is `origin/camera_ready`). It is pushed there before GOLEM's pin moves (CLAUDE.md §5); the user authorised this push.
- **Process rules:**
  - Builds and tests run inside the containers (only `tests/docker` runs on the host).
  - Comments describe the code as it is (CLAUDE.md §8).
  - Changes to the simulators are minimal, and their default behavior is preserved.

## Robot profiles

A profile is data. Each node takes `robot:=h12|g1` and `sim:=true|false` from its bringup. Configs live under `config/<robot>/`.

| Field | H1-2 | G1 |
|---|---|---|
| `mode_machine` the robot reports and the commands must carry | 6 (verified: all 47,962 `/lowstate` and 38,655 `/lowcmd` samples in the real-robot bags `lmg_deploy/bags/{reach,policy_bag_walk}`) | 5 (`g1_description/layout.py`: "our unit reports 5"; the G1 sim reports 5; not yet seen on the real G1) |
| `mode_pr` | 0 | 0 (`MODE_PR`) |
| Body motors | 27 joints, slots 0-26, legs 0-11, torso 12, arms 13-26 | 29 joints, slots 0-28, legs 0-11, waist 12-14, arms 15-28 (`g1_description`) |
| `LowCmd` motor array | 35 | 35 |
| URDF limits | the table now duplicated, identical, in `h12_safety_layer/core/joint_limits.py` and `h12_ros2_controller/utility/joint_limits.py` | `g1_description.joints.BODY` (generated from the URDF) |
| Safety modes | `full`, `split` | `arm_sdk`, `split`, `full` |
| Output topic | `/lowcmd` | `/arm_sdk` (arm_sdk mode) or `/lowcmd` |
| Controller URDF / collision model | `h1_2_magpie` URDF + `h1_2_magpie_sphere.urdf` + `h1_2_magpie_sphere_collision.srdf` (`CL_Assets/ros_assets`) | `g1_29dof_with_hand_rev_1_0` URDF; no self-collision model, so the IK collision barrier and the planner stay off, as today |
| IK end effectors | `left/right_wrist_yaw_link` | `left/right_hand_palm_link` |
| Planner frames | `left/right_grasp_frame` | `left/right_hand_palm_link` |
| IMU frame for the model | `torso_link` | from `g1_control` (`imu_frame`) |
| Held (non-arm) joints | `torso_joint` | `waist_yaw/roll/pitch_joint` |
| Named configurations, start-up sequence | `h12_ros2_controller` today | `g1_control` today |

**Robot data is defined once.** *(my call)* Each robot has one data file (joint names, slots, URDF limits) that both packages read. The C++ safety layer cannot import Python, so the data lives as YAML in `golem_safety`'s share directory, and `golem_upper_ik` reads the same file. Tests check:
- the G1 file against `g1_description`'s generated table;
- the H1-2 file against today's H1-2 tables, value for value.

**Configs carry today's values.** Configs are grouped by robot. The names stay the same as today, so bringup arguments keep their values:
- `golem_safety`: the H1-2's 8 safety configs and the G1's 5;
- `golem_upper_ik`: the H1-2 controller configs the bringups use (`safety_split`, `sim_safety_split`, `safety_full`, and `debug` for `joint_state_publisher`), and the G1's 4.

Values are carried over unchanged. Keys whose meaning changes under this design go in the change report, for example the stale-input estop timeouts, which now only report (item 2).

**One config serves sim and real.** Whether the robot is simulated, and whether a hardware estop is present, come from the bringup (`sim`, `use_estop`), not from the config (user's decision). For example, `lmg_sim_bringup` keeps using `relax_safety_split.yaml`, the same file the real bringup uses. Consequences:
- The G1 configs' mandatory `robot: real|sim` key goes.
- The G1 bringup's existing refusals stay in `g1_core`: real refuses `sim_*` configs, sim refuses real ones.

**The safety and controller configs are paired.** `config_pairing.py` (from `g1_control`) moves into `golem_upper_ik` and gains the H1-2's pairs, taken from what the H1-2 bringups launch together today. Both robots' bringups refuse a mismatched pair, as `g1_core.launch.py` does now.

## golem_safety (C++, rclcpp)

One node, `safety_node`. It is the only writer of `/lowcmd` and `/arm_sdk`.

**ROS interface**

| Direction | Topic | Type | When |
|---|---|---|---|
| sub | `/lowstate` | `unitree_hg/msg/LowState` | always |
| sub | `/safety/lowcmd_in` | `unitree_hg/msg/LowCmd` | `full` |
| sub | `/safety/lowcmd_upper_in`, `/safety/lowcmd_lower_in` | `LowCmd` | `split` |
| sub | `/safety/lowcmd_upper_in` | `LowCmd` | `arm_sdk` (G1) |
| pub | `/lowcmd` or `/arm_sdk` | `LowCmd` | per profile and mode |
| pub | `/safety/heartbeat` | `std_msgs/Bool`, 10 Hz on steady time, true until estop | always, both robots |
| sub | `/estop/triggered`, `/estop/plugged_in` | `std_msgs/Bool` | when the bringup passes `use_estop:=true` |

Today's consumers of `/safety/heartbeat` keep working: `h12_lowerbody_rl`, `lmg_deploy` and `h12_skills`.

**Behavior**

1. **Newest only.** Every input and `/lowstate` keeps only its newest sample (keep-last 1). One relay loop runs at `publish_hz`; a slow cycle lowers the rate, it never queues commands [H1-2, `e0396fc`].
2. **Quiet inputs are held and reported, not estopped** (user's decision). Once any input has engaged:
   - an input with no new command for its stale time keeps having its last command relayed;
   - a quiet `/lowstate` leaves the relay running (the H1-2 today).

   Each case is logged when it starts and when it ends. The stale times are configured; today's `upper_stale_seconds`, `lower_stale_estop_seconds` and `state_stale_estop_seconds` become these reporting thresholds. The G1's stale-input and stale-state estops go (change report).

   The H1-2's transport-lag estop against the robot's tick counter on domain 0 (`state_lag_ms`) stays: it fires on `/lowstate` samples arriving late, not on silence.

   All timers run on a monotonic boot-time clock, so a wall-clock step can neither fire nor hide one [G1].
3. **Single writer per input** [G1]. A second publisher on an input, identified by its publisher GID, latches the estop.
4. **Start-up and engaging: each robot keeps today's rule** (user's decision for the H1-2), chosen in its configs.

   **H1-2 rule:**
   - From boot the layer streams a zero-gain command on every slot.
   - An input engages when its first command arrives, and from then on its slots carry its commands.
   - The slots of inputs that have not engaged stay at zero gain. The real H1-2 runs arms-only until `start_position_verified:=true` starts the leg node.
   - There is no position-tolerance gate.
   - Limit estops watch all 27 slots from boot.

   **G1 rule** (needed, not just inherited):
   - The layer writes nothing until its engage gate passes: a fresh `/lowstate`, `mode_machine` 5, and every required input's command within `max_position_error` of the measured joints.
   - In `split` that means both inputs: the G1 sim's stand-in (`h1_robocasa/g1/unitree_interface.py`) releases the robot on the first valid `rt/lowcmd`, so arms-first would drop the robot before `locomotion_node` starts.
   - Limit estops start at engage, so a limp joint past its limit does not latch.

   **On both robots, when the bringup passes `use_estop:=true`:**
   - No input engages until the first estop sample has arrived (user's decision).
   - After that, an estop sample older than its configured timeout latches the estop. Today a dead `estop_node` leaves the last "not triggered" sample in force forever.
5. **Robot identity on every sample.** Every `/lowstate` is checked against the profile's `mode_machine`. A mismatch before engaging refuses; after engaging it latches the estop, because it means another robot or sim is on the domain. Outgoing commands carry the profile's `mode_machine` and `mode_pr`, not a copy from the inputs or from `/lowstate`. This keeps the robot's own firmware check (the H1-2 drops a `LowCmd` whose `mode_machine` is not its own) as a second, independent guard.
6. **Clipping and limit estops.**
   - Commands are clipped to `limits.clip` (position offset, velocity and torque ratios, kp/kd maxima).
   - A measured state beyond `limits.estop` latches the estop.
   - Estop reasons name the joint, the value, the limit and the multiple [H1-2 `5182fbe`].
   - Today's limit derivation from the URDF table is kept exactly.
7. **Estop latch.** One-way until restart, with the reason and the timing report both layers produce today. Its triggers:
   - a limit violation;
   - a robot-identity mismatch after engaging;
   - a second writer;
   - a triggered, unplugged or stale hardware estop;
   - the transport-lag check.

   The response is configured: `estop_response.mode` `zero` or `damp` (with `kd`), plus what happens after it. Each robot's configs reproduce today's behavior *(my call)*:
   - H1-2: 50 zero-gain commands, then silence (`58d0f99`);
   - G1 `split`/`full`: keep writing the estop command;
   - G1 `arm_sdk`: ramp the weight to 0 and hold it there.
8. **Exit (SIGINT/SIGTERM).**
   - In `full` and `split` the layer stops publishing, and the robot holds its last command, as the H1-2 does today (H1-2 first). The G1's write-the-estop-command-on-exit goes (change report).
   - In `arm_sdk` mode the layer ramps the weight to 0 before exiting, as the G1 does today *(my call: there is no H1-2 counterpart, and the ramp hands the arms back to Unitree's controller instead of leaving them on a weight nobody updates)*.
9. **`arm_sdk` mode** (G1 profile only) [G1]:
   - the weight in slot 29;
   - `engage_seconds` and `release_seconds` ramps;
   - the `waist_kp_min` check.
10. **Hardware estop over ROS** *(my call)*. When `use_estop:=true`, the layer reads the `estop` node's ROS topics instead of the raw DDS `h12/estop_status_raw` sample. Triggered or unplugged latches the estop, as today; staleness and the first-sample rule are in item 4.
11. **Config validation** [G1]. Unknown keys are rejected, and `sim:=true` refuses `ROS_DOMAIN_ID=0`.
12. **Logging.** Sample logging in chunks, enabled per config as today (on for the G1 configs, off for the H1-2's). The `.npz` chunk layout stays as today, so `combine_chunks` (kept as a script) still reads it *(my call)*.
13. **CRC.** The layer computes the Unitree `LowCmd` CRC itself (CRC-32 over the message body, as the SDK does), with a test against known SDK-computed values.

## golem_upper_ik (Python, rclpy)

Two nodes, with the same names and ROS surface as today:

- **`frame_task_server`**
  - Serves the actions `/frame_task` (`FrameTask`) and `/named_config` (`NamedConfig`).
  - Publishes `frame_names`, `frame_targets`, `frame_poses` and `{left,right}_ee_{pose,target}`.
  - Reads `/lowstate` (newest only).
  - Writes `/safety/lowcmd_upper_in`, or the configured `topics.low_cmd`, which must be under `/safety/`.
  - Reads the safety layer's output (`/arm_sdk` or `/lowcmd`) only for the follow gate.
- **`joint_state_publisher`:** `/joint_states` for the profile's body joints at 100 Hz, from `/lowstate`.

**Behavior.** The H1-2's current code is the base; the G1's fixes are added.

From the H1-2:
- **State reads:** newest state, with tick-lag tracking. A goal aborts with an error when the state is stale (`caf318e`).
- **No controller-side estop.** Command clipping stays (`deacb9a`).
- **IK seeds** try the current pose first (`9eec384`).
- **Reduced model:** the frozen joints are re-locked at the measured state for each goal (`0159d45`).
- **Goal start:** a goal starts from the last command, not the measured pose (`772f946`).
- **Planner:**
  - `execute_path` paces at `dt − work` (`f3a9a55`);
  - a hold pinned at a limit is not reported as converged (`3d87152`);
  - full x/y/z frame limits, and per-request speed and interpolation overrides.
- **Goal options:** `relative` targets and `slow_mode` (`SLOW_MODE_SCALE` 0.25), on both robots.

From the G1:
- **Start-up:** wait for a fresh `/lowstate` with finite joint positions before enabling motors; on failure raise and publish nothing.
- **Start-up moves** are deadline-bounded and raise on timeout.
- **Held joints** (H1-2 torso, G1 waist) carry their hold angle in every command, never the measured, sagging value.
- **Follow gate:** commands stay within `controller.follow_tolerance` of the measured joints until the safety layer relays them. On the G1 this matters because of the engage gate and the arm_sdk handover. On the H1-2 it opens at the first relay.
- **Frozen joints:** generic frozen-joint model loading, so every non-body joint is frozen in slot order.

**Collision-aware IK and OMPL.** The self-collision barrier in the IK and the OMPL planner run whenever the profile names a sphere URDF and SRDF:
- The H1-2 uses its existing ones (`CL_Assets/ros_assets`).
- The G1 profile names none, so both stay off on the G1, and the G1 `/frame_task` keeps refusing planned goals as `g1_control` does today (`PLANNER_UNAVAILABLE`).

## Switching the bringups and other consumers

Everything that starts, depends on or tests the old packages moves to the new ones:

- **`h12_bringup`:**
  - `h12_control.launch.py` (`safety_node`, `frame_task_server`; it already has `use_estop`);
  - `h12_state.launch.py` (`joint_state_publisher`);
  - `h12_sim_bringup_mac.launch.py`;
  - `package.xml`.

  The `lmg_*` and `h12_safety_grippers` bringups go through `h12_control`. Each launch passes `robot:=h12` and its `sim` value.
- **`g1_bringup`:** `g1_core.launch.py` (its startup refusal imports `pairing_refusal` from `golem_upper_ik`), `g1_bringup.launch.py` and `package.xml`. Each passes `robot:=g1`.
- **`g1_locomotion`:** its `package.xml` `exec_depend` changes, and its README and comments name `golem_safety`. `test_locomotion_dds.py` runs against the `golem_safety` node, not the in-process `g1_safety_layer`.
- **Tests:**
  - `tests/stack/`: `minimal_stack.launch.py`, `launch_ros_stack.sh` and `test_heartbeat_clock.py`;
  - `tests/g1/`: `test_sim_bringup.py`, which imports `g1_control.utility.*` and matches the `lib/g1_control/frame_task_server` process name.
- **Mac:** `docker/mac/scripts/launch_ros_mac.sh` `PKGS`, and the comment in `docker/mac/RosDockerfile.slim.arm64`.
- **Simulators.** The H1-2 sims report `mode_machine` 6 like the real robot, instead of the IDL default 0 (user's decision). That touches:
  - `h1_robocasa/unitree_interface.py`;
  - `CL_isaaclab_sim`'s H1-2 lowstate DDS (committed and pushed in that repo, then the pin moves);
  - the H1-2 entry in `docker/scripts/wait_for_sim.py`'s `MODE_MACHINE`.

  If `tests/robocasa/surface/h12.json` records the field, re-record it in the same commit.
- **Docs:**
  - `CLAUDE.md` and `AGENTS.md` (§2, §4, §6, §8 invariants);
  - `docs/g1/*.md`, `docker/BUILD.md`, `tests/g1/README.md` and `h12_bringup/common_cmds.md`;
  - a README for each new package.

## Done when

1. **Package tests pass in the containers.** Every behavior listed above has a test on both profiles, over ROS:
   - `golem_safety`: the G1's 63 safety tests ported to ROS and adjusted to the decisions above, plus tests for:
     - hold-on-quiet;
     - each start-up rule;
     - the estop first-sample and age rules;
     - heartbeat, single-writer, identity and CRC.
   - `golem_upper_ik`: the planner, config, controller-loop, goal-handling (`relative`, `slow_mode`), start-up and follow-gate tests. The H1-2 profile runs with its collision model; the G1 profile runs without one.
2. **Limit parity.** For every carried-over config, the derived clip and estop limits equal what the old package derives, except values listed in the change report.
3. **H1-2 sim stacks:**
   - `tests/stack/run.sh robocasa arms` and `robocasa locomotion` pass;
   - `tests/stack/run.sh isaac arms` passes (this host has an RTX 5070 Ti and the `golem_sim_isaac` image);
   - `lmg_sim_bringup` launches and its safety layer engages *(my call: a launch smoke check, since no test covers it today)*.
4. **G1 sim acceptance:** `tests/g1/test_sim_bringup.sh` (and `--walk`), `test_sim_nav.sh` (both odometry modes) and `test_sim_explore.sh` pass.
5. **Surface goldens:** `tests/robocasa/surface/run.sh h12|g1` matches the goldens, apart from a re-recorded `mode_machine` if the golden holds it.
6. **Mac:** the Mac H1-2 smoke test in `docs/DOCKER_TESTING.md` passes.
7. **Rates:** both robots' sim stacks sustain the configured rates, measured.
   - `golem_safety`'s relay: publish rate and input-to-output latency are reported.
   - `golem_upper_ik`: its 500 Hz publish is reported.
8. **A change report for the user:** every value or behavior that differs from what the H1-2 or the G1 does today, for each robot. It includes at least:
   - the G1's stale-input estops becoming hold-and-report;
   - the G1's exit behavior in `split`/`full`;
   - the new estop-age latch;
   - the H1-2 sims' `mode_machine`;
   - the `robot: real|sim` key's removal.

## Left to the builder

- The internal structure of both packages: classes, files, how a profile is loaded.
- The C++ test framework (gtest, launch_testing, or rclpy-driven integration tests).
- QoS details beyond keep-last 1, as long as they interoperate with the Unitree DDS publishers in the sims.
- How the `.npz` chunks are written.
- Whether the shared robot data file is YAML or generated.
