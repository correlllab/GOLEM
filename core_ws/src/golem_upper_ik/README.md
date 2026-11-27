# golem_upper_ik

The upper-body (torso or waist, and arms) IK controller of the Unitree H1-2 and G1. Two
nodes, both robots, each robot described by a profile:

- `frame_task_server` solves Pink IK on the 14 arm joints with a self-collision barrier,
  plans with OMPL, and serves `/grasp_frame_target`. It
  writes its commands only to a `golem_safety` input (`/safety/...`), never to `/lowcmd`.
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
ros2 launch golem_upper_ik upper_ik.launch.py robot:=h12 config:=safety_split \
    safety_config:=relax_safety_split sim:=true use_sim_time:=true
ros2 launch golem_upper_ik robot_state.launch.py robot:=h12 use_sim_time:=true
ros2 run golem_upper_ik frame_task_server --ros-args -p robot:=g1 -p config:=sim_split
ros2 run golem_upper_ik joint_state_publisher --ros-args -p robot:=h12
```

| Parameter | Node | Default | Meaning |
|---|---|---|---|
| `robot` | both | required | a robot with data in both packages: `h12` or `g1` (Robot profiles, below) |
| `config` | `frame_task_server` | required | a controller config name under `share/golem_upper_ik/config/<robot>/` (with or without `.yaml`), or a path; a path in another robot's config directory is refused |
| `low_state_topic` | `joint_state_publisher` | `/lowstate` | the robot's state |

A node whose robot is missing or unknown, whose config is missing or does not load, or whose
start-up fails logs why (`frame_task_server did not start: ...`, naming the robot and the
config once they are read) and exits 1. On SIGINT or SIGTERM it shuts down (below) and exits 0.

## ROS interface

`frame_task_server`:

| Direction | Name | Type | Notes |
|---|---|---|---|
| action | `/grasp_frame_target` | `custom_ros_messages/action/GraspFrameTarget` | move each arm's grasp frame to a target in the `pelvis` frame or a world frame (Grasp-frame targets, below) |
| pub | `frame_names` | `custom_ros_messages/msg/StringArray` | the current goal's frames, 25 Hz |
| pub | `frame_targets`, `frame_poses` | `geometry_msgs/PoseArray` | their targets and measured poses, `pelvis` frame, 25 Hz |
| pub | `/left_grasp_frame_pose`, `/right_grasp_frame_pose` | `geometry_msgs/PoseStamped` | the grasp frames' measured poses in the `pelvis` frame, 25 Hz, from every measured body joint (as TF has the frames: the H1-2's are URDF frames, the G1's sit at their offsets from the palm links); their first message tells other nodes the controller is up |
| pub | `topics.low_cmd` | `unitree_hg/msg/LowCmd` | reliable, keep-last 1, at `frequency.pub_hz`; must be under `/safety/` |
| sub | `topics.low_state` (`/lowstate`) | `unitree_hg/msg/LowState` | keep-last 1, best effort |
| sub | `/lowcmd` | `LowCmd` | `golem_safety`'s output, read (never written) only until the follow gate opens |

The action and the publishers exist once the start-up is done.
`/grasp_frame_target` and the grasp pose topics have global names: they are the interface every
arm controller of the robot serves alike.

`joint_state_publisher`: `joint_states` (`sensor_msgs/JointState`) at 100 Hz, the profile's
body joints in slot order (27 or 29) with the newest `low_state_topic` (`/lowstate`)
positions, all 0 before the first sample. The hands' and grippers' joints come from their
own nodes.

## Robot profiles

`golem_upper_ik.profile.load_profile(robot)` joins three sources: `golem_safety`'s
`robots/<robot>.yaml` (joint names in slot order, limits, slot ranges, `mode_machine`), the
one joint table both packages read; `config/<robot>/robot.yaml` in this package (display
name, IK and held joints, frames, grasp frames, named configurations, start-up sequence,
per-robot config defaults), which `profile.robot_config(robot)` returns checked (every key
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
| Grasp frames (`grasp_frames`) | `left/right_grasp_frame`, the Magpie grippers' pinch points in the URDF | `left/right_grasp_frame`, the Dex3-1 pinch points, which the URDF lacks: `RobotModel` adds each at its fixed offset (`xyz`, `rot`) from `left/right_hand_palm_link`, between the thumb and the fingers (from the URDF's joint origins, unmeasured) |
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
| H1-2 | `safety_full` | `/safety/lowcmd_in` | `relax_safety_full`, `default_safety_full`, `tight_safety_full`, `locomotion_safety_full` |
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
config outside the pairs is refused; a name is taken without its directory or `.yaml`.

### Keys

Every key of a controller config, with its meaning, units, default and allowed values, is in
`config/<robot>/example.controller.yaml`. The loader refuses, naming the robot, the file and
the key: an unknown key at any level; a non-finite number, a number out of its range or a
value of the wrong kind.

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
4. Move through the start-up postures, each until the IK's command is within 1e-3 rad of it
   (the IK's steps within the acceleration limit, Goals below, and each IK joint at most
   `JOINT_SPEED_CAP`, 0.83 rad/s, on its own: a joint with less to go arrives first), then step
   the held joints to `controller.hold_targets` at min(0.3, `dq_lim`) rad/s, all within
   `controller.startup_move_timeout` of the gate opening.

A start-up that fails after step 2 sends the shutdown command (Shutdown, below) and the
node exits 1. `golem_safety` streams zero gains from the robot's first `/lowstate` and engages the upper-body input on
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

- One goal at a time drives the controller: a goal waits for the one in flight to end.
- Each goal re-locks the reduced model's frozen joints (the torso or waist, and the legs) at
  their measured angles, so the IK's forward kinematics match the robot, and starts the IK
  from the last command when it is within 0.1 rad of the measured pose on every IK joint;
  otherwise from the measured pose, dropping the held torque bias.
- The IK, the planner's bounds and the path checks keep every IK joint within the clip range,
  `limits.clip.position_offset` inside the joint table (and within the URDF's limits): the
  controller narrows its robot model's position limits to it (`RobotModel.set_position_limits`)
  and hands the same range to the planner's subprocess. No solution lies where the clip would
  cut the command, and a joint the IK parks at a limit sits `position_offset` inside the table,
  so its overshoot past the command stays clear of `golem_safety`'s estop range. A seed, a
  measured pose or an integrated step outside the range is clamped into it. A direct goal's
  and a start-up move's IK steps brake into the range's limits: a joint closes on its limit at
  no more than its distance over `LIMIT_APPROACH_TIME` (0.2 s, `core/ik_solver.py`), so it
  arrives slowly instead of at `dq_lim`; moves that stay farther from the limits are not
  slowed, and the IK solves of planned goals do not brake. Each config's
  `limits.clip.position_offset` comment says how its margin covers the overshoot measured
  against the estop ranges of the `golem_safety` configs it pairs with.
- A direct goal (`plan: false`) steps the IK at `ctrl_hz`, its arm velocities limited to
  `dq_lim` per joint and `v_lim`/`w_lim` at the end effectors, and their changes to the
  acceleration limit (below). An iteration that takes longer than the control period (a loaded
  host) runs the IK steps it owes next time, one per period it took and at most
  `CATCH_UP_STEPS` (3), each solved over one period and each within the acceleration limit, so
  the arm keeps its speed in robot time. The self-collision barrier constrains the
  `BARRIER_PAIRS` (32) closest sphere pairs of each step (`core/ik_solver.py`). Once the IK's
  step is slower than `threshold_ik` it holds the target with integral control (`gains.ki`,
  fading over the next move); the targets are checked against the thresholds only from then on.
- **The torque bias near a clip.** On every command (a hold, a move, a stop) each joint's
  torque bias toward a position clip limit is bounded at kp times the command's distance inside
  that limit plus `BIAS_PAST_CLIP` (5 mrad, `core/controller/upper_controller.py`), so the
  bias alone holds a joint at most 5 mrad past its clip, whatever a hold integrated (the hand
  pressing on a table) or a move carried in; per joint, the other joints integrate on. A hold
  whose bias stops at the bound short of the clip is neither converged nor pinned at a limit
  (the pinned report needs the joint within 1 mrad of its clip): the goal ends at its timeout
  or the caller's cancel. In the H1-2 sim's grasps, 21 of 83 holds pinned at a limit had needed
  more bias than the bound to get there.
- **The acceleration limit.** Every IK-stepped command (the start-up moves and direct goals)
  changes each IK joint's velocity by at most `ACCEL_LIMIT` (5 rad/s² of robot time,
  `core/controller/upper_controller.py`), from rest at its start to rest in the hold. Each
  control step takes the step nearest the IK's (within the speed limits) that changes no
  joint's step by more than `ACCEL_LIMIT * dt^2` from its last, lets each joint still stop
  within the IK's distance left to the goal, and keeps the IK's position-limit brake, velocity
  limits and self-collision barrier (`IKSolver.limited_step`). Where those need a harder change
  (the brake, the barrier or a goal behind a fast joint), the step keeps them and leaves the
  acceleration limit by the least it can; only when the QP solver finds no step at all is each
  arm's step scaled along the IK's instead (`accel_scale`). The step is then scaled under
  `dq_lim`, `v_lim` and `w_lim` once more, so the speed limits win too. The IK's own step is
  otherwise its whole step to the goal, so the joints cruise at the speed limits (the H1-2's
  `dq_lim` is 0.83 rad/s, the top speed they had before the acceleration limit). A direct
  goal's IK has converged once both its own and the commanded joint speeds are under
  `threshold_ik`. Two commands jump on purpose, outside the IK steps: a goal that starts from
  the measured pose (more than 0.1 rad off the command, above) moves the command onto the arm,
  and a cancel or a timeout moves it to where the arm will rest (`stop_joints`, below).
  5 rad/s² covers the position-limit brake's own deceleration, `dq_lim / LIMIT_APPROACH_TIME`
  (4.2 rad/s² at the H1-2's 0.83 rad/s, 5 at the G1 sim's 1.0); the constant's comment has the
  measurements.
- `plan: true`: the OMPL planner (in a subprocess of its own) plans in the IK joints, with the
  self-collision checks and the planner frames' x/y/z bounds, and the path is executed one
  waypoint per control period. Its smoothstep timing gets more waypoints until no joint
  accelerates faster than `ACCEL_LIMIT` along it, from rest to rest (up to
  `planner.max_interpolation_steps`). `planner.enabled: false` aborts every planned goal before
  any IK solve: `Planning is turned off: ...; send the goal with plan=false`. A profile without
  a self-collision model could not plan either (`Planning is unavailable`); both robots' have
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
- **Grasp-frame targets** (`/grasp_frame_target`). Each arm's target is a `PoseStamped` for
  its grasp frame (`grasp_frames`); an empty `header.frame_id` leaves that arm where it is, and
  a goal without any target aborts. A target in `pelvis` is taken as it is. A target in any
  other frame (`map`, `odom`) is resolved into the `pelvis` frame once, when the goal starts,
  before it waits for a goal in flight: the node listens to TF (`/tf`, `/tf_static`) for up
  to `TF_WAIT_S` (1 s), takes the newest transform from that frame to `pelvis` and stops
  listening; a transform TF does not give, or whose newest stamp is more than `TF_MAX_AGE_S`
  (0.5 s) of robot time old (a chain of static transforms only is always taken), aborts the
  goal, naming the frame. The controller then holds that `pelvis`-frame target: the arms ride
  on the pelvis, so when the robot walks, sways or turns after the goal starts, the grasp frame
  moves with the pelvis and leaves the world pose; send a new goal to aim at the world pose
  again. A target with a non-finite value or a zero quaternion aborts the goal; the quaternion
  is normalised.
  The goal replaces the frame tasks with one on each targeted grasp frame and runs direct, or
  along a plan with `plan: true`. Only the targeted arms' joints change their torque bias: the hold's integral term
  (`gains.ki`) and its fading over a move act on them alone, so the arm without a target and
  the torso or waist keep their commands, torque bias included, and stay where they are (both
  robots' holds integrate on the arms, and an arm sagging below its command would otherwise
  be lifted onto it, about 11 mm in the H1-2 sim). Its `timeout` (0: `controller.timeout`) counts
  robot time from its start, planning and the path included. It succeeds once every target has
  stayed within `threshold_linear` and `threshold_angular` for `GRASP_SETTLE_S` (0.2 s of robot
  time); the hold step's own convergence ends nothing. Both robots' configs set the thresholds
  to 2 mm and 10 mrad: in the sims the hold settles within about 0.6 mm and 2 mrad, and a
  planned goal holds the IK solve's solution, which stops within 1 mm and 10 mrad. It aborts at the timeout, on a planning
  or execution failure, on a hold pinned at a joint limit and on a stale `/lowstate`. A cancel,
  or the timeout, during a planned path stops it before its next waypoint; the IK solve and the
  planning before the path are not cut short (the planner's first request also builds its
  robot model, seconds). Every end sets the
  result's `outcome`: `REACHED` (succeeded), `NOT_REACHED` (aborted: the timeout, before the IK
  converged, during the path or in the hold, or a hold pinned at a joint limit), `CANCELED`, or
  `FAILED` (aborted for anything else: a bad target, TF, planning or execution, a stale
  `/lowstate`); and its `message` (`reached (...)`, or why it was aborted or canceled, with the
  errors). A
  cancel or the timeout, direct or during the path, or a path that fails while it runs, stops the targeted arms where they are, since a moving
  arm lags its command (about 20 mm at the grasp frame at 0.3 m/s on the H1-2 sim) and would
  keep going that far: each of their joints is commanded at its measured position plus its
  resting offset (`stop_joints`), how far it sat off its command when the goal started (the
  gravity error its torque bias leaves), adjusted by the bias faded since; a joint that was
  moving or more than 0.1 rad off then gets no offset. The arm then stops within its braking
  distance and settles near the pose (in the sims, cancelled at 0.13 to 0.33 m/s: at most 6 mm
  away on the H1-2 and 14 mm on the soft-armed G1 within 0.1 s, settling within 2.5 mm and
  1.5 mm). After any other end (reached, a hold pinned at a limit, a plan
  that fails before the path, a stale `/lowstate`) the arms hold the last command, where the goal left them. The feedback, every
  control step and before every waypoint, and the errors in the message are each grasp frame's
  distance from its target (m) and the angle between their rotations (rad), measured as the IK
  sees the frame (the reduced model at the measured arm joints, the torso or waist at the angle
  the goal locked it at); 0 for an arm without a target. The pelvis frame is the robot's own:
  the IMU orientation plays no part, so a target 5 cm above the measured pose is 5 cm up the
  pelvis's z axis however the robot is turned or tilted.
- On ROS domain 0 every control step first checks that `/lowstate` is no staler than
  `controller.max_state_lag_ms` (the larger of its tick lag and its age); a stale state aborts
  the goal and the last command keeps going out. The sims stamp the tick in sim steps, so the
  check runs only on domain 0.

**Commands.** A publisher thread sends the newest command at `frequency.pub_hz` on absolute
deadlines of robot time. Under `use_sim_time` a robot-time wait ends only when `/clock` moves,
so every rate (`pub_hz`, `check_hz`, `ctrl_hz`) follows `/clock`'s steps (RoboCasa's every
2 ms of sim time; Isaac's once per control step): about one cycle per step while a step is no
longer than the period. The publisher makes up a longer step, or a short lag, with a burst of
the newest command (up to `MAX_CATCH_UP_S` of robot time, `core/ros_io.py`), and a control
step whose work outlasts its period starts the next at once. Every q, dq and tau is clipped to `limits.clip`; a
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
(`test_controller_loop.py`, `test_frame_task_server.py`, `/grasp_frame_target` and the grasp
pose topics included), a start-up IK step with no solution (`test_startup_no_solution.py`),
the goal loop of `/grasp_frame_target` against a scripted controller (`test_goal_loops.py`) and
that nothing imports `unitree_sdk2py` (`test_no_sdk.py`).

## Tools

- `tools/command_publisher_rate.py`, `tools/frame_task_server_rate.py`: measure the 500 Hz
  command rate of `CommandPublisher` and of the installed `frame_task_server` with its IK.
