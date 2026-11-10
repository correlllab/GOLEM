# One ROS stack for both robots, and the audit fixes — design

Status: approved for implementation by the user on 2026-11-05 ("Implement those").
Branch: `shared_robot_stack`, on top of `dead_code_cleanup`.

## Intent

The H1-2 and the G1 should run the same ROS code: the same packages, nodes
and node graph, with only configs and policies changing between deployments.
A change to a shared package is made once and holds for both robots, and a
robot is described by data, not by branches in code. Alongside, the
code-quality audit (artifact https://claude.ai/artifact/V57Sxq8tf4k5kVTNZbhM7D)
found bugs and safety gaps; the user chose which to fix.

## What is shared and what is per-robot (user decisions)

| Layer | Shared code, robot picked by config | Per robot |
|---|---|---|
| Safety | `golem_safety` (already config-only), with its own launch file | its configs |
| Arms | `golem_upper_ik`, with its own launch files | its robot config and controller configs |
| SLAM / navigation | `golem_slam`: one launch file, one node graph | its config directory |
| Bringup | — | `golem_bringup/launch/h12/` and `launch/g1/` stay separate; they only choose configs, drivers, delays and start-up order, and include the shared packages' launch files |
| Legs | — | `h12_lowerbody_rl`, `g1_homie` stay separate packages and nodes |
| Hands | — | each robot's driver, started by its bringup or provided by its simulator |
| Skills | — | `golem_skills` is H1-2-owned |

`golem_safety` stays the only writer of `rt/lowcmd` (already true; checked on
the live graphs of `h12_sim`, `g1_sim`, `lmg_sim` and the Mac launch on
2026-11-05).

## Goals

### Same code, configs only

1. **`golem_upper_ik` holds no robot data in code.** Everything in
   `golem_upper_ik/robots/h12.py` and `robots/g1.py` moves into a per-robot
   config file: display name, IK joints and held joints, end-effector, planner
   and IMU frames, named postures, start-up sequence,
   `shutdown_keeps_held_joints`, the per-robot config defaults and the default
   config. The safety/controller pairing tables in `pairing.py` become per-robot
   data with one rule for both robots. The joint table stays in `golem_safety`'s
   robot profile.
2. **Each shared package owns its launch files, and both bringups include
   them.** `golem_safety`: `safety.launch.py` (safety_node). `golem_upper_ik`:
   the arm controller (`frame_task_server`, with the pairing check) and the
   robot state (`robot_state_publisher` plus `joint_state_publisher`).
   `golem_slam`: one `navigation.launch.py`. The bringups keep their own
   delays, start-up order, drivers and config choices.
3. **`golem_slam` runs one node graph on both robots.** No robot branches:
   - one dynamic `body → pelvis` node on both, generalised from
     `g1_body_to_pelvis` (FAST-LIO's lidar-to-IMU extrinsic composed with the
     live TF from `pelvis` to the lidar), replacing the H1-2's static one;
   - a static identity `<lidar link> → livox_frame` on both (the driver's frame);
   - `level_odom` with a moving base transform on both;
   - `level_cloud` on both; the scan converter reads `/cloud_base_filtered` on both;
   - the behaviour server's `cmd_vel` goes through the velocity smoother on both;
   - each robot's differences live in its config directory
     (`golem_slam/config/<robot>/`): FAST-LIO (real and sim), Nav2, OctoMap,
     lidar driver config, and a pipeline profile (lidar link name, scan band,
     `level_odom` values).
   The G1's navigation is the H1-2's code, so a fix in one holds for the other.
4. **Shared packages carry no robot-name branches.** Inside `golem_safety`,
   `golem_upper_ik` and `golem_slam` (code and launch files; tests and tools
   excluded), a robot name appears only as a value that selects a config. A
   test enforces it.

### Raw-frame lidar on both robots

5. The H1-2 moves to the G1's handling of its upside-down MID-360:
   - the driver applies no rotation (`MID360_config.json` roll 0);
   - every H1-2 model in `CL_Assets` that has the lidar (all URDF variants,
     the MJCF and the robosuite copy built from it) carries the upside-down
     mount: `livox_link` and `livox_imu_link` are the sensor's raw frames,
     physically where they were, rotated 180° about their x axis. The G1's
     models already carry it and stay byte-for-byte upstream;
   - the H1-2 RoboCasa sim publishes `/livox/lidar`, `/livox/pointcloud` and
     `/livox/imu` in the raw frame, stamped `livox_frame` like the real
     driver; the RoboCasa surface golden is re-recorded in the same commit;
   - FAST-LIO's H1-2 extrinsic is the raw-frame one;
   - `livox_imu_upright` and its wiring are deleted.

### Fixes from the audit

6. **G1 navigation with no hand node.** `self_filter` filters the links it has
   TF for and skips links that have never had a TF (reported once), so a robot
   without a hand driver still navigates. A link whose TF goes stale after it
   was seen still drops the cloud.
7. **The velocity smoother** sees every Nav2 command on both robots (goal 3).
8. **Skills halt on a silent heartbeat.** `golem_skills` treats the arms as
   halted until the first true `/safety/heartbeat`, after a false one (for
   good: the safety latch is permanent until restart), and after 0.5 s with no
   beat; a halt also cancels the arm goal in flight.
9. **The LMG pre-pose waits for the first true heartbeat**, as the H1-2 leg
   node does. Nothing else in `lmg_deploy` changes.
10. **The H1-2 leg node gets `g1_homie`'s runtime checks:** stale or
    non-finite `/lowstate`, upright before the pre-pose and a fall latch after,
    a `/cmd_vel` timeout (zero velocity) with non-finite commands treated as
    zero, a fault latch on a non-finite policy output or a policy exception
    (the node stops publishing and stays up), and the operator confirm accepted
    only while holding the pre-pose. Unit tests cover each.
11. **The elastic band release is one call, never retried, on both leg
    nodes.** The band is sim-only and should never fail: the node calls the
    toggle once, when the service is up; a failure is logged as an error and
    the band is not marked released. The policy history is reset when the
    release succeeds, not when the call is sent.
12. **The H1-2 sim's command path matches the G1 sim's:** each `rt/lowcmd`
    is validated (CRC, `mode_machine` 6, `mode_pr` 0, finite values) and
    swapped in whole (no torn reads); every motor is off until the first valid
    command; the command timeout is checked in sim time inside the physics
    step; `rt/lowstate` is published after every physics step with a
    millisecond sim-time tick, a CRC and `rpy`. The motor-on rule (mode 1) and
    the default behaviour otherwise stay.

### The Mac

13. The Mac moves onto the current launch architecture: its top-level launch
    includes the same blocks as `h12_sim.launch.py` (state, control, legs,
    navigation), with its `GOLEM_*` toggles choosing what is included. Its
    navigation is the shared `golem_slam` pipeline (FAST-LIO, `level_odom`,
    OctoMap, slam_toolbox, Nav2); `nav2_config_mac.yaml` and the Mac's own
    scan/odom path are deleted. The Mac image gains what that pipeline needs
    (PCL, Livox-SDK2, `livox_ros_driver2`, FAST-LIO, OctoMap).

## Non-goals (user decisions)

- Merging the bringups or the leg packages; a shared hand interface;
  robot-agnostic skills.
- Isaac, and the loco-manipulation game (`lmg_*`), apart from goal 9.
- The `-1.01` position offset in the unused "default" safety config.
- The gripper hold thread in `golem_skills` (it prevents gripper overload crashes).
- `deploy_mjpc`, IMU/lidar rates, FAST-LIO's map size.
- Changing `livox_ros_driver2` (pin `13eb05e` stands) or Livox-SDK2.
- Real-robot sessions: this work ends at the sim.

## Anti-goals

- Robot branches moved from code into launch-file `if`s or YAML that encodes
  logic.
- Any change to `golem_safety`'s runtime behaviour: configs, limits, estop
  responses and start-up are unchanged on both robots.
- A moved value that changes: tuning moved from code to config stays the same
  value.
- Ground truth on anywhere (CLAUDE.md §8).

## Constraints

- CLAUDE.md applies: comments describe the code as it is; build and test in
  disposable containers, never on `ROS_DOMAIN_ID` 0; never stop, replace or
  exec into the user's `golem_ros` container; the simulators keep their
  default behaviour apart from goals 5 and 12.
- `CL_Assets` is a submodule: its changes are committed there on a branch
  named `shared_robot_stack`. Pushing (it, then the pin) waits for the user.
- Don't touch the user's uncommitted `core_ws/src/golem_bringup/rviz/g1/g1.rviz`
  or the `livox_ros_driver2` submodule change.

## Known consequences the user accepted by scoping

- `lmg_sim` and `lmg_real` read `/livox/imu` in `livox_imu_link`; after goal 5
  that frame is the raw one, so the LMG policy gets an upside-down IMU until
  `lmg_deploy` is updated.
- The H1-2 Isaac sim and its USD stay upright, so Isaac navigation no longer
  matches the H1-2 configs.
- `mujoco_mpc`'s `lio_bridge_node.py` constants cite the old MJCF lines.

## Done when

- Every package test, `tests/robocasa` (with the re-recorded H1-2 golden),
  `tests/robocasa/g1`, `tests/docker`, `tests/stack` and `tests/g1` pass.
- One closed-loop sim test, the same script for both robots (`--h12`/`--g1`),
  runs the full bringup with navigation in RoboCasa and requires, at the same
  time: the robot standing unsupported, a Nav2 goal reached by walking, an arm
  move through `/named_config` that completes, and SLAM healthy throughout
  (FAST-LIO odometry, `level_odom`'s `odom`, a growing `/map`, `/converted_scan`
  at rate). `tests/g1/test_sim_nav.sh` is the model.
- On every bringup, `safety_node` is the only publisher of `/lowcmd`.
- The Mac launch file starts and builds its graph in the x86 `ros` image (the
  arm64 image cannot be built on the x86 host, which has no arm64 emulation).

## Before the real robot (not part of done)

The H1-2's raw-frame FAST-LIO is replayed on recorded real-robot bags (at
least 3 repeats; FAST-LIO is not deterministic) before the stack runs on the
real H1-2. Existing bags hold the driver's roll-180 points and the upright
IMU, so the replay rotates both back to the raw frame.

## Left to the builder

- File names inside the per-robot config directories, and the pairing data's shape.
- How the H1-2 leg node's checks are structured (a ROS-free supervisor class
  like `g1_homie`'s controller is the suggested shape, for testability).
- Whether the two RoboCasa DDS interfaces share a helper module.
