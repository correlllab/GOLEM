# golem_upper_ik

The upper-body (torso or waist, and arms) IK controller of the Unitree H1-2 and G1. Two
nodes, both robots, each robot described by a profile:

- `frame_task_server` solves Pink IK on the 14 arm joints, plans with OMPL where the robot
  has a self-collision model, and serves `/frame_task` and `/named_config`. It writes its
  commands only to a `golem_safety` input (`/safety/...`), never to `/lowcmd`.
- `joint_state_publisher` publishes `/joint_states` for the robot's body joints from
  `/lowstate`.

Both are Python (`rclpy`) and talk to the robot only through ROS topics: ROS topic `/x` is
DDS topic `rt/x`, with the same `unitree_hg` wire type. The design is
`docs/superpowers/specs/2026-10-27-golem-upper-ik-and-safety-design.md` (section
golem_upper_ik). Every value and behavior that differs from the controllers this package
replaces is in the change report beside it,
`docs/superpowers/specs/2026-10-27-golem-upper-ik-and-safety/change-report.md`.

## Running it

The bringups start both nodes (`golem_bringup`'s `h12_control.launch.py` and
`h12_state.launch.py`, and its `g1_core.launch.py`), after `golem_safety`'s
`safety_node`. On their own:

```bash
ros2 launch golem_upper_ik upper_ik.launch.py robot:=h12 config:=sim_safety_split use_sim_time:=true
ros2 run golem_upper_ik frame_task_server --ros-args -p robot:=g1 -p config:=sim_split
ros2 run golem_upper_ik joint_state_publisher --ros-args -p robot:=h12
```

| Parameter | Default | Meaning |
|---|---|---|
| `robot` | required | `h12` or `g1` |
| `config` | the robot's `DEFAULT_CONFIG`: `debug` (H1-2), `g1_split` (G1) | a controller config name under `share/golem_upper_ik/config/<robot>/` (with or without `.yaml`), or a path; a path in another robot's config directory is refused. `joint_state_publisher` reads only its `topics.low_state` |

A node whose robot is missing or unknown, whose config does not load, or whose start-up
fails logs why (`frame_task_server did not start: ...`, naming the robot and the config once
they are read) and exits 1. On SIGINT or SIGTERM it shuts down (below) and exits 0.

## ROS interface

`frame_task_server`:

| Direction | Name | Type | Notes |
|---|---|---|---|
| action | `/frame_task` | `custom_ros_messages/action/FrameTask` | pull named frames to target poses in the `pelvis` frame |
| action | `/named_config` | `custom_ros_messages/action/NamedConfig` | move the IK joints to a named configuration |
| pub | `frame_names` | `custom_ros_messages/msg/StringArray` | the current goal's frames, 100 Hz |
| pub | `frame_targets`, `frame_poses` | `geometry_msgs/PoseArray` | their targets and measured poses, `pelvis` frame, 100 Hz |
| pub | `left_ee_pose`, `right_ee_pose` | `geometry_msgs/PoseStamped` | the end effectors' measured poses, 100 Hz; their first message tells other nodes the controller is up |
| pub | `left_ee_target`, `right_ee_target` | `geometry_msgs/PoseStamped` | the current goal's targets for the end-effector frames, when it names them |
| pub | `topics.low_cmd` | `unitree_hg/msg/LowCmd` | reliable, keep-last 1, at `frequency.pub_hz`; must be under `/safety/` |
| sub | `topics.low_state` (`/lowstate`) | `unitree_hg/msg/LowState` | keep-last 1, best effort |
| sub | `topics.safety_out` | `LowCmd` | `golem_safety`'s output, read (never written) only until the follow gate opens |

The actions, publishers and the `/frame_task` goal check exist once the start-up is done.

`joint_state_publisher`: `joint_states` (`sensor_msgs/JointState`) at 100 Hz, the profile's
body joints in slot order (27 or 29) with the newest `/lowstate` positions, all 0 before the
first sample. The hands' and grippers' joints come from their own nodes.

## Robot profiles

`golem_upper_ik.profile.load_profile(robot)` joins two sources: `golem_safety`'s
`robots/<robot>.yaml` (joint names in slot order, limits, slot ranges, `mode_machine`), the
one joint table both packages read, and `golem_upper_ik/robots/<robot>.py` (model files,
frames, IK and held joints, named configurations, start-up sequence, per-robot config
defaults). Both robots' model files come from `golem_ros2_models`. A robot module imports
nothing of another robot's packages: the H1-2 builds and runs without the `g1_*` packages.

| | H1-2 (`robots/h12.py`) | G1 (`robots/g1.py`) |
|---|---|---|
| IK joints | the 14 arm joints | the 14 arm joints |
| Held joints | `torso_joint` | `waist_yaw_joint`, `waist_roll_joint`, `waist_pitch_joint` |
| Held joints at shutdown on `/lowcmd` (Shutdown, below) | off, with the arms | kept at their hold |
| End-effector frames | `left/right_wrist_yaw_link` | `left/right_hand_palm_link` |
| Planner frames | `left/right_grasp_frame` | `left/right_hand_palm_link` |
| IMU frame | `torso_link` | `imu_in_pelvis` (`controller.imu_frame` overrides) |
| Model (`golem_ros2_models`) | `h1_2_magpie.urdf` | `g1_29dof_with_hand_rev_1_0.urdf` |
| Self-collision model | `h1_2_magpie_sphere.urdf` + `h1_2_magpie_sphere_collision.srdf` | none: no collision barrier, no planner |
| Start-up postures | `init_1`, `init_2`, `init_3`, `home` | `home` |
| Named configurations | `NAMED_CONFIGS` in `robots/h12.py` (`home`, `t_pose`, `prep`, `staging_left`, ...) | `NAMED_CONFIGS` in `robots/g1.py` (`home`, `arms_down`, `t_pose`, `arms_front_45`) |
| `topics.safety_out` default | `/lowcmd` | `/lowcmd` |
| `controller.max_state_lag_ms` default | 50 | 0 (off: the real G1's tick has not been measured) |

## Configs

| Robot | Config | Writes | Runs with `golem_safety`'s |
|---|---|---|---|
| H1-2 | `safety_split` | `/safety/lowcmd_upper_in` | `relax_safety_split`, `default_safety_split`, `tight_safety_split` |
| H1-2 | `sim_safety_split` | `/safety/lowcmd_upper_in` | `sim_safety_split` |
| H1-2 | `safety_full` | `/safety/lowcmd_in` | `relax_safety_full`, `default_safety_full`, `tight_safety_full`, `locomotion_safety_full` |
| H1-2 | `debug` | `/safety/lowcmd_in` | none: the default of both nodes, read by `joint_state_publisher` |
| G1 | `g1_split` | `/safety/lowcmd_upper_in` | `split` (a stiff waist, for walking) |
| G1 | `sim_split` | `/safety/lowcmd_upper_in` | `sim_split` |

`golem_upper_ik.pairing` holds these pairs (`H12_SAFETY_CONTROLLER_CONFIGS`,
`G1_SAFETY_CONTROLLER_CONFIGS`) and the rules the bringups check before they start a node:
`pairing_refusal(robot, controller, safety, sim)` gives the reason a pair must not run, and
`default_controller_config(robot, safety)` the controller config a safety config runs with.
On the G1 it refuses a safety config that does not relay the upper-body input (any config
but the two that do, `full` included) and sim and real configs mixed (told apart by the
`sim` prefix of their names); as one of its pairs is the real robot's and the other the
sim's, that also keeps each known controller config to its own safety config. On the H1-2 a
config outside the tables is not refused.

### Keys

| Key | Default | Meaning |
|---|---|---|
| `topics.low_cmd` | `/safety/lowcmd_upper_in` | the `golem_safety` input written; a topic outside `/safety/` is refused |
| `topics.low_state` | `/lowstate` | the robot's state |
| `topics.safety_out` | per robot (above) | `golem_safety`'s output for the follow gate: `/lowcmd`, the output of every `golem_safety` mode; another topic is refused |
| `controller.v_lim`, `w_lim` | 1.0, 2.0 | end-effector linear (m/s) and angular (rad/s) speed limits |
| `controller.dq_lim` | 1.0 | IK joint speed limit, rad/s |
| `controller.d_min` | 0.02 | self-collision distance the IK keeps, m |
| `controller.timeout` | 10.0 | s per goal sent with a zero `duration` |
| `controller.threshold_ik`, `threshold_joint`, `threshold_linear`, `threshold_angular` | 0.001, 0.01, 0.002, 0.01 | convergence: IK velocity, rad, m, rad |
| `controller.hold_targets` | 0 for each held joint | the held joints' angles, rad, a mapping by joint name |
| `controller.state_timeout` | 5.0 | s to wait for a fresh `/lowstate` at start |
| `controller.startup_timeout` | 30.0 | s for `golem_safety` to be seen relaying the commands |
| `controller.startup_move_timeout` | 30.0 | s for the start-up moves, from the follow gate opening |
| `controller.follow_tolerance` | 0.05 | rad; 0 turns the follow gate off |
| `controller.imu_frame` | the profile's | the frame of `/lowstate`'s IMU in the model |
| `controller.max_state_lag_ms` | per robot (above) | the stale-state limit on domain 0, ms; <= 0 turns it off |
| `frequency.ctrl_hz`, `pub_hz`, `check_hz` | 50, 500, 1000 | IK steps, command publishing, the follow gate's checks between steps, Hz |
| `gains.kp`, `kd`, `ki` | | per upper-body joint |
| `limits.clip.position_offset`, `velocity_ratio`, `torque_ratio` | 0.001, 0.10, 0.25 | the command clip, as in `golem_safety` (`limits.enforce` is an alias) |
| `planner.*` | `enabled: true`, `frame_names` the profile's | the OMPL planner (`planner`, `timeout`, `moving_speed`, the frame bounds, ...); the rest take `PlannerConfig`'s defaults |

Every config sets `ctrl_hz` 30 and `pub_hz` 500, except the H1-2's `sim_safety_split`, which
leaves `frequency` out and so solves IK at 50 Hz. The loader refuses, naming the robot, the
file and the key: an unknown key at any level; a key of an older schema (`mode`, `network`,
`controller.torso_target`, `controller.waist_target`, `limits.estop`,
`planner.frame_z_min_margin`), saying why it was removed or which key replaces it; a
non-finite number, a number
out of its range or a value of the wrong kind; an alias given with its key.

## Behavior

**Start-up**, before any goal:

1. Wait up to `controller.state_timeout` for a `/lowstate` younger than 0.1 s with finite
   joint positions. Otherwise stop with an error and publish nothing.
2. The first command is every upper-body slot at the measured pose, with the gravity torque
   there.
3. Hold that pose until `golem_safety` is seen relaying the commands (the follow gate opens)
   within `controller.startup_timeout`. The error otherwise names what `golem_safety` waits
   for: a `/lowstate` reporting the profile's `mode_machine` (6 or 5), a first estop sample
   with `use_estop:=true`, the paired safety config, and no estop latched at this start-up (its
   limit checks run from its first `/lowstate`, so a joint already past `limits.estop` latches
   it) or in an earlier run.
4. Move through the start-up postures, each until the IK's command is within 1e-3 rad of it,
   then step the held joints to `controller.hold_targets` at min(0.3, `dq_lim`) rad/s, all
   within `controller.startup_move_timeout` of the gate opening.

A start-up that fails after step 2 sends the shutdown command (Shutdown, below) and the
node exits 1. `golem_safety` streams zero gains from boot and engages the upper-body input on
its first command once `/lowstate` reports the profile's `mode_machine` (and, on the real
H1-2, `estop_node` has sent a first sample), so the gate opens at the first relay, on both
robots whether or not the leg node commands.

**The follow gate.** Until a `topics.safety_out` sample carries the controller's command on
every upper slot (kp > 0, each q within `controller.follow_tolerance`), every upper-body
command is kept within `follow_tolerance` of the measured joints, and a thread re-checks at
`frequency.check_hz` between control steps. Nothing executes the commands before
`golem_safety` relays them, so an open-loop move would run away from the robot and the robot
would jump at the relay. Once open, the gate never moves a command again and the
subscription is dropped. A planned path is not executed while the gate is shut.

**Held joints.** Every command carries the held joints' hold angle (the H1-2's torso, the G1's
waist), never the q the IK passes for them, so a sagging torso is pulled back. The hold
starts at the measured angle, follows the gated command while the follow gate is shut, and
then moves only in the start-up step 4.

**Goals.**

- One goal at a time drives the controller. A `/frame_task` goal naming a frame the model
  does not have is refused before it is accepted; an unknown `/named_config` name aborts.
- Each goal re-locks the reduced model's frozen joints (the torso or waist, and the legs) at
  their measured angles, so the IK's forward kinematics match the robot, and starts the IK
  from the last command when it is within 0.1 rad of the measured pose on every IK joint;
  otherwise from the measured pose, dropping the held torque bias.
- A direct goal (`plan: false`) steps the IK at `ctrl_hz`, its arm velocities limited to
  `dq_lim` per joint and `v_lim`/`w_lim` at the end effectors. Once the IK converges it holds
  the target with integral control (`gains.ki`, fading over the next move). It succeeds when
  the frames (or joints) are within the thresholds before the goal's `duration` (or
  `controller.timeout`) runs out; a hold pinned at a joint position limit stops with
  `success: false` (`Hold blocked at a joint position limit: ...`).
- `relative: true` (`FrameTask`): each target's translation is added to the frame's measured
  position, and its rotation is applied in pelvis axes about the frame's origin.
- `slow_mode: true` (both actions): a quarter of the speed (`SLOW_MODE_SCALE` 0.25), applied
  after the speed limits; a planned goal is planned at a quarter of `planner.moving_speed`.
- `plan: true`: the OMPL planner (in a subprocess of its own) plans in the IK joints, with
  the self-collision checks and the planner frames' x/y/z bounds, and the path is executed one
  waypoint per control period. On a robot whose profile has no self-collision model (the G1)
  every planned goal is aborted before any IK solve: `Planning is unavailable: the G1 has no
  self-collision model ...; send the goal with plan=false`. `planner.enabled: false` refuses
  them as well.
- On ROS domain 0 every control step first checks that `/lowstate` is no staler than
  `controller.max_state_lag_ms` (the larger of its tick lag and its age); a stale state aborts
  the goal and the last command keeps going out. The sims stamp the tick in sim steps, so the
  check runs only on domain 0.

**Commands.** A publisher thread sends the newest command at `frequency.pub_hz` on absolute
deadlines of the monotonic clock. Every q, dq and tau is clipped to `limits.clip`; a
non-finite value is refused (`ValueError`) and the last good command keeps going out, since
`golem_safety` latches its estop on a non-finite command. Only the upper slots are written:
the lower slots and the slots past the body are mode 0 with every value 0. The header
carries the profile's `mode_machine` and `mode_pr`; the CRC field is 0, since
`golem_safety` computes the CRC of what it writes. Each command carries the gravity
feed-forward torque. There is no controller-side estop: `golem_safety`'s limits are the
check. Both nodes set `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` to 1 before numpy
loads: OpenBLAS's busy-waiting threads would stall the 500 Hz publisher.

**Shutdown.** rclpy's signal handlers are off. The first SIGINT or SIGTERM ends the node; a
later one is ignored. `frame_task_server` first sends its upper slots one command and stops
publishing, then stops the planner and its `/lowstate` subscription, the last two even if an
earlier step fails; the whole shutdown fits in launch's 5 s `sigterm_timeout`. The command
gives the IK joints mode 0 and zero kp, kd and tau (q and dq as they were). The held joints
get the same, unless the profile's `shutdown_keeps_held_joints_on_lowcmd` is set (the G1):
then they keep their hold angle, gains and torque, since a walking policy on the legs needs
a stiff waist. What `golem_safety` does with that command:

- H1-2 (both modes): it relays it, so the torso and the arms go limp, and as a quiet input
  it stays relayed;
- G1: it relays it, so the arms go limp while the waist holds, and as a quiet input it stays
  relayed.

A controller killed without that command (SIGKILL, a crash) leaves its last command held by
`golem_safety` until the bringup stops.

## Tests

In either container mode (`--restart` replaces a running `golem_ros` container):

```bash
docker/scripts/docker_run.sh ros --restart --g1 /home/code/h12_sim_scripts/launch_ros.sh bash -lc \
  'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 && cd /home/code/core_ws && colcon build --symlink-install --packages-select golem_upper_ik && source install/setup.bash && colcon test --packages-select golem_upper_ik --event-handlers console_direct+; colcon test-result --verbose'
```

The ROS tests run on `ROS_DOMAIN_ID` 92 (`test/conftest.py`, `test/ros_harness.py`), some
against `golem_safety`'s installed `safety_node`. They cover the profiles
(`test_profile.py`), the configs and pairing (`test_config.py`, `test_pairing.py`), the robot
model, IK and planner (`test_robot_model.py`, `test_ik_solver.py`, `test_planner.py`; the
H1-2 with its collision model, the G1 without), the ROS I/O, command handler and follow gate
(`test_ros_io.py`, `test_low_cmd_handler.py`), the controller and the nodes
(`test_controller_loop.py`, `test_frame_task_server.py`), the publish rate
(`test_publish_rate.py`) and that nothing imports `unitree_sdk2py` (`test_no_sdk.py`). The
fixtures under `test/fixtures/` are recorded data: what the previous arm controllers
returned for the same configs and configurations (`test/fixtures/README.md` says which
package and commit); no test imports those controllers.

## Tools

- `tools/publish_rate_probe.py`, `tools/command_publisher_rate.py`,
  `tools/frame_task_server_rate.py`: measure the 500 Hz command rate of a bare `rclpy`
  publisher, of `CommandPublisher`, and of the installed `frame_task_server` with its IK.
