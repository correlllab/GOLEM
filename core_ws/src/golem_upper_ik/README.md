# golem_upper_ik

The upper-body (torso or waist, and arms) IK controller of the Unitree H1-2 and G1. Two
nodes, both robots, each robot described by a profile:

- `frame_task_server` solves Pink IK on the 14 arm joints with a self-collision barrier,
  plans with OMPL, and serves `/frame_task` and `/named_config`. It writes its
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

The package's launch files start both nodes, and both robots' bringups include them
(`golem_bringup`'s `h12_control.launch.py` and `h12_state.launch.py`, and its
`g1_core.launch.py`):

- `upper_ik.launch.py` (`robot`, `config`, `safety_config` and `sim` required; `use_sim_time`
  and `start_delay`, s of wall clock, default `false` and `0.0`) refuses a `config` that does
  not run with `safety_config` (Safety pairing, below), raising `upper_ik: not starting:
  <reason>` before it starts anything, then starts `frame_task_server` after `start_delay`.
  The bringups include it before any other node, so a refused pair stops the launch first.
- `robot_state.launch.py` (`robot` required; `use_sim_time`; `urdf_file`, empty for the
  robot's ROS URDF from `golem_ros2_models`) starts `robot_state_publisher` on that URDF and
  `joint_state_publisher`.

On their own:

```bash
ros2 launch golem_upper_ik upper_ik.launch.py robot:=h12 config:=sim_safety_split \
    safety_config:=sim_safety_split sim:=true use_sim_time:=true
ros2 launch golem_upper_ik robot_state.launch.py robot:=h12 use_sim_time:=true
ros2 run golem_upper_ik frame_task_server --ros-args -p robot:=g1 -p config:=sim_split
ros2 run golem_upper_ik joint_state_publisher --ros-args -p robot:=h12
```

| Parameter | Default | Meaning |
|---|---|---|
| `robot` | required | a robot with data in both packages: `h12` or `g1` (Robot profiles, below) |
| `config` | the robot's `default_config` in `config/<robot>/robot.yaml`: `debug` (H1-2), `g1_split` (G1) | a controller config name under `share/golem_upper_ik/config/<robot>/` (with or without `.yaml`), or a path; a path in another robot's config directory is refused. `joint_state_publisher` reads only its `topics.low_state` |

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
| sub | `/lowcmd` | `LowCmd` | `golem_safety`'s output, read (never written) only until the follow gate opens |

The actions, publishers and the `/frame_task` goal check exist once the start-up is done.

`joint_state_publisher`: `joint_states` (`sensor_msgs/JointState`) at 100 Hz, the profile's
body joints in slot order (27 or 29) with the newest `/lowstate` positions, all 0 before the
first sample. The hands' and grippers' joints come from their own nodes.

## Robot profiles

`golem_upper_ik.profile.load_profile(robot)` joins three sources: `golem_safety`'s
`robots/<robot>.yaml` (joint names in slot order, limits, slot ranges, `mode_machine`), the
one joint table both packages read; `config/<robot>/robot.yaml` in this package (display
name, IK and held joints, frames, named configurations, start-up sequence, per-robot config
defaults, default config), which `profile.robot_config(robot)` returns checked (every key
required, no others); and the model files, from `golem_ros2_models`. A robot is known
(`profile.known_robots()`) when `robots/<robot>.yaml` and `config/<robot>/robot.yaml` both
exist, so a new robot is two data files and its configs, with no Python change. An unknown
robot is refused with `ValueError`, naming the known ones.

| | H1-2 (`config/h12/robot.yaml`) | G1 (`config/g1/robot.yaml`) |
|---|---|---|
| IK joints | the 14 arm joints | the 14 arm joints |
| Held joints | `torso_joint` | `waist_yaw_joint`, `waist_roll_joint`, `waist_pitch_joint` |
| Held joints at shutdown (Shutdown, below) | off, with the arms | kept at their hold |
| End-effector frames | `left/right_wrist_yaw_link` | `left/right_hand_palm_link` |
| Planner frames | `left/right_grasp_frame` | `left/right_hand_palm_link` |
| IMU frame | `torso_link` | `imu_in_pelvis` (`controller.imu_frame` overrides) |
| Model (`golem_ros2_models`) | `h1_2_magpie.urdf` | `g1_29dof_with_hand_rev_1_0.urdf` |
| Self-collision model | `h1_2_magpie_sphere.urdf` + `h1_2_magpie_sphere_collision.srdf` | `g1_sphere.urdf` + `g1_sphere_collision.srdf`, packed on the rubber-hand G1: two spheres per hand at the Dex3-1 palm, none on the fingers, so the barrier and the planner do not see the Dex3-1 fingers. Its torso spheres stay inside the torso's flanks, where the hanging upper arms are 9 mm from the torso (`CL_Assets/README.md`) |
| Start-up postures | `init_1`, `init_2`, `init_3`, `home` | `home` |
| Named configurations | `named_configs` in `config/h12/robot.yaml` (`home`, `t_pose`, `prep`, `staging_left`, ...) | `named_configs` in `config/g1/robot.yaml` (`home`, `arms_down`, `t_pose`, `arms_front_45`) |
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

Each robot's `config/<robot>/pairing.yaml` holds these pairs (`pairs:`, safety config name to
controller config name), and `golem_upper_ik.pairing` applies one rule to every robot, which
`upper_ik.launch.py` checks before the bringups start a node. `pairing_refusal(robot, controller, safety, sim)`
gives the first reason a pair must not run, or `None`: the safety config is not in the
robot's pairs; the controller config is not the one paired with it; or, with `sim` given
(`True` in the sim, `False` on the real robot), a safety or controller config whose name
starts with `sim` differs from it. `default_controller_config(robot, safety)` is the
controller config a safety config runs with (`ValueError` for a safety config in no pair). A
config outside the pairs, the H1-2's `debug` included, is refused; a name is taken without
its directory or `.yaml`.

### Keys

| Key | Default | Meaning |
|---|---|---|
| `topics.low_cmd` | `/safety/lowcmd_upper_in` | the `golem_safety` input written; a topic outside `/safety/` is refused |
| `topics.low_state` | `/lowstate` | the robot's state |
| `controller.v_lim`, `w_lim` | 1.0, 2.0 | end-effector linear (m/s) and angular (rad/s) speed limits |
| `controller.dq_lim` | 1.0 | IK joint speed limit, rad/s |
| `controller.d_min` | 0.02 | self-collision distance the IK keeps between sphere pairs, m; every config of both robots keeps it (the G1's closest pairs at `home`, the upper arms against the torso, are 0.025 m apart) |
| `controller.timeout` | 10.0 | s of robot time per goal sent with a zero `duration` |
| `controller.threshold_ik`, `threshold_joint`, `threshold_linear`, `threshold_angular` | 0.001, 0.01, 0.002, 0.01 | convergence: IK velocity, rad, m, rad |
| `controller.hold_targets` | 0 for each held joint | the held joints' angles, rad, a mapping by joint name |
| `controller.state_timeout` | 5.0 | s of robot time to wait for a fresh `/lowstate` at start (under `use_sim_time`, first as many s of steady time for the first `/clock`) |
| `controller.startup_timeout` | 30.0 | s of robot time for `golem_safety` to be seen relaying the commands |
| `controller.startup_move_timeout` | 30.0 | s of robot time for the start-up moves, from the follow gate opening |
| `controller.follow_tolerance` | 0.05 | rad; 0 turns the follow gate off |
| `controller.imu_frame` | the profile's | the frame of `/lowstate`'s IMU in the model |
| `controller.max_state_lag_ms` | per robot (above) | the stale-state limit on domain 0, ms; <= 0 turns it off |
| `frequency.ctrl_hz`, `pub_hz`, `check_hz` | 50, 500, 1000 | IK steps, command publishing, the follow gate's checks between steps, per s of robot time |
| `gains.kp`, `kd`, `ki` | | per upper-body joint |
| `limits.clip.position_offset`, `velocity_ratio`, `torque_ratio` | 0.001, 0.10, 0.25 | the command clip, as in `golem_safety` |
| `planner.*` | `enabled: true`, `frame_names` the profile's | the OMPL planner (`planner`, `timeout`, `moving_speed`, the frame bounds, ...); the rest take `PlannerConfig`'s defaults |

Every config sets `ctrl_hz` 30 and `pub_hz` 500, except the H1-2's `sim_safety_split`, which
leaves `frequency` out and so solves IK at 50 Hz. The loader refuses, naming the robot, the
file and the key: an unknown key at any level; a non-finite number, a number out of its
range or a value of the wrong kind.

## Behavior

**Robot time.** Every rate, wait and timeout of `frame_task_server` (the start-up, the goals,
the publisher, the follow gate) counts robot time: the sim's `/clock` with `use_sim_time:=true`,
so the controller keeps its timing in sim seconds whatever the sim's speed, and the steady
(monotonic) clock otherwise, on the real robots (`golem_upper_ik.core.robot_clock`). Under
`use_sim_time` robot time stands still while `/clock` does, and never steps back. The IK's
and OMPL's time limits are host compute budgets, in s of steady time.

**Start-up**, before any goal:

1. Under `use_sim_time`, wait up to `controller.state_timeout` s of steady time for the first
   `/clock`. Then wait up to `controller.state_timeout` for a `/lowstate` younger than 0.1 s
   with finite joint positions. Otherwise stop with an error and publish nothing.
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

**The follow gate.** Until a `/lowcmd` sample (`golem_safety`'s output) carries the controller's command on
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
- `relative: true` (`FrameTask`): each target is an offset from the frame's measured pose in
  the `pelvis` frame, the frame of absolute targets and of `/left_ee_pose`: its translation is
  added to the frame's position in pelvis axes, and its rotation pre-multiplies the frame's
  orientation (pelvis axes, about the frame's origin). The IMU orientation plays no part, so
  `{position: {z: 0.05}}` moves the frame 5 cm up the pelvis's z axis however the robot is
  turned or tilted.
- `slow_mode: true` (both actions): a quarter of the speed (`SLOW_MODE_SCALE` 0.25), applied
  after the speed limits; a planned goal is planned at a quarter of `planner.moving_speed`.
- `plan: true`: the OMPL planner (in a subprocess of its own) plans in the IK joints, with
  the self-collision checks and the planner frames' x/y/z bounds, and the path is executed one
  waypoint per control period. `planner.enabled: false` aborts every planned goal before any
  IK solve: `Planning is turned off: ...; send the goal with plan=false`. A profile without a
  self-collision model could not plan either (`Planning is unavailable`); both robots' have
  one.
- The G1's `arms_down` is in self-collision: the real wrists and hands rest on the hips, and
  each hand and wrist sphere sits 2 to 5 cm inside its hip-pitch sphere. A goal to it stops
  short, held off by the barrier, and a planned goal to it fails (`goal is in
  self-collision`). Every other named configuration of both robots is collision-free.
- The barrier pushes a sphere pair that starts a few millimetres inside `d_min` back out, but
  not one much further in: from a measured pose with a pair overlapping, or on the G1 about
  10 mm apart or closer (the start-up from the limp arms rolled in toward the torso, offline),
  the IK's quadratic program has no solution, and the move fails at its first step. A
  start-up move that fails so stops the controller with an error that names the closest
  enabled sphere pairs and their distances against `d_min` (logged first as `<move>: the IK
  has no solution; closest self-collision pairs (controller.d_min 20 mm): ...`), so a start
  inside the keep-out reads as self-collision. On the G1 the limp arms settle with the wrists
  resting on the hips (elbows about 1.25 rad), their spheres 20 mm from the hip spheres, and
  the start-up reaches `home` from there; from `arms_down` it could not.
- On ROS domain 0 every control step first checks that `/lowstate` is no staler than
  `controller.max_state_lag_ms` (the larger of its tick lag and its age); a stale state aborts
  the goal and the last command keeps going out. The sims stamp the tick in sim steps, so the
  check runs only on domain 0.

**Commands.** A publisher thread sends the newest command at `frequency.pub_hz` on absolute
deadlines of robot time. Every q, dq and tau is clipped to `limits.clip`; a
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
get the same, unless the profile's `shutdown_keeps_held_joints` is set (the G1):
then they keep their hold angle, gains and torque, since a walking policy on the legs needs
a stiff waist. What `golem_safety` does with that command:

- H1-2 (both modes): it relays it, so the torso and the arms go limp, and as a quiet input
  it stays relayed;
- G1: it relays it, so the arms go limp while the waist holds, and as a quiet input it stays
  relayed.

A controller killed without that command (SIGKILL, a crash) leaves its last command held by
`golem_safety` until the bringup stops.

## Tests

In the `ros` container (`--restart` replaces a running `golem_ros` container):

```bash
docker/scripts/docker_run.sh ros --restart /home/code/h12_sim_scripts/launch_ros.sh bash -lc \
  'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 && cd /home/code/core_ws && colcon build --symlink-install --packages-select golem_upper_ik && source install/setup.bash && colcon test --packages-select golem_upper_ik --event-handlers console_direct+; colcon test-result --verbose'
```

The ROS tests run on `ROS_DOMAIN_ID` 92 (`test/conftest.py`, `test/ros_harness.py`), some
against `golem_safety`'s installed `safety_node`. They cover the profiles
(`test_profile.py`), the configs and pairing (`test_config.py`, `test_pairing.py`), the robot
model, IK and planner (`test_robot_model.py`, `test_ik_solver.py`, `test_planner.py`; both
robots with their collision models), the ROS I/O, command handler and follow gate
(`test_ros_io.py`, `test_low_cmd_handler.py`), the controller and the nodes
(`test_controller_loop.py`, `test_frame_task_server.py`), a start-up IK step with no
solution (`test_startup_no_solution.py`), the publish rate
(`test_publish_rate.py`) and that nothing imports `unitree_sdk2py` (`test_no_sdk.py`).

## Tools

- `tools/publish_rate_probe.py`, `tools/command_publisher_rate.py`,
  `tools/frame_task_server_rate.py`: measure the 500 Hz command rate of a bare `rclpy`
  publisher, of `CommandPublisher`, and of the installed `frame_task_server` with its IK.
