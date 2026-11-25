# lmg_deploy (LocoMotionGameDeploy)

Runs a two-agent MARL policy from Locomanipulation_game (branch `marl-direct`,
`LocoManip-Marl-Direct-v0`: LSTM actors, explicit PD at 500 Hz, a 0-20 ms target
delay in training) on the H1-2. The policies live one per robot, in `policies/h12/` and
`policies/g1/` (empty so far), **one iteration at a time: a new export replaces the robot's
folder, and old ones are deleted, not kept beside it** (`policies/README.md`). `policies/h12/` holds
Coop300k_s42 at 150k steps (LocoManip-Marl-Direct-v0: legs and arms trained together, the sport-mode
recipe at 100 Hz; it takes arm goals in the pelvis frame and keeps walking during them; the arms run the
IK trajectory from `arm_seed_table.npz` plus the learned residual; the /lowstate torso IMU with the
calibrated orientation, so it needs h12's entry in `config/imu_calibration.yaml`, and its legs actor is
not shown the wrist targets); its `export.yaml`'s `source` names the checkpoint. A
**legs** agent that walks on `/cmd_vel`, and an **arms** agent that reaches
wrist-pose goals through a damped-least-squares IK step plus a learned
residual. One node, `marl_deploy_node`, owns all 27 joints. It publishes
through the split safety layer and obeys `/safety/heartbeat`.

Modelled on `golem_lowerbody_rl`'s `lowerbody_controller_node`. The node goes
through the same steps (pre-pose ramp, operator confirm, elastic-band
release) and uses the same heartbeat gate and watchdog: it publishes nothing, and
does not start its pre-pose ramp, until `golem_safety`'s first true `/safety/heartbeat`
(state `waiting_heartbeat`), and a false beat or none for `heartbeat_timeout` (0.5 s) after
the first latches it silent.

## Layout

| File | What it is |
|---|---|
| `lmg_deploy/controller.py` | ROS-free env step: SLAM odometry → mode → observations (each actor's terms as the export's `legs_obs_terms` / `arms_obs_terms` list them, in the controller's order, any left out; all of them without the lists: legs 95, arms 123, 97 and 125 with `gait_phase_cos`) → joint targets |
| `lmg_deploy/kinematics.py` | Pinocchio wrist FK, Jacobian and IK step, in the pelvis frame; the arm joints' gravity torques and `arm_ik`'s seed solve |
| `lmg_deploy/rotations.py` | numpy versions of the isaaclab.utils.math functions training used |
| `lmg_deploy/networks.py` | TorchScript loader for the exported actors |
| `lmg_deploy/gyro_lowpass.py` | `GyroLowpass`, the 2nd-order Butterworth training ran on the torso IMU's angular velocity |
| `lmg_deploy/imu_calibration.py` | the torso IMU's orientation calibration: the correction (`correct_imu`), the export contract (`resolve_calibration`), the kinematic tilt and the estimate, the runtime check (README "IMU calibration") |
| `lmg_deploy/calibrate_imu.py` | `ros2 run lmg_deploy calibrate_imu <bag>`: the calibration from a bag of the robot standing still on both feet |
| `config/imu_calibration.yaml` | each robot's torso-IMU calibration (`h12`, `g1`), with how and when it was measured |
| `lmg_deploy/marl_deploy_node.py` | the ROS node: safety channels, heartbeat, pre-pose, band, operator interface |
| `config/example.yaml`, `config/example.export.yaml` | every key of a deploy config and of `export.yaml`, with its meaning and units (nothing loads them; `test/test_example_configs.py` keeps their keys in step) |
| `config/policy_template.yaml` | the deploy config a new policy starts from: joint order, defaults, gains, scales, bounds, command ranges (each from the training env), and the SLAM body frame |
| `policies/<robot>/<robot>.yaml`, `{legs,arms}.pt`, `export.yaml` | the robot's one policy (`h12/`; `g1/` empty): its deploy config and what the export writes, below; the `.pt` files in Git LFS (`policies/.gitattributes`). One iteration at a time, old ones deleted (`policies/README.md`) |
| `bags/<name>/` | rosbag2 recordings of Explicit500 on the real robot (`policy_bag_walk`: walking to nav2 goals; `reach`: an arm goal, ended by the hardware e-stop), the `.db3` files in Git LFS |
| `test/test_controller.py` | unit tests (rotations vs Isaac Lab, Jacobian, IK, observation layout, SLAM odometry, modes, `arm_ik`) on the shipped policy's deploy config (`test/shipped_policy.py`: `policies/h12/h12.yaml` with its `export.yaml` over it, as the node loads them), and the export formats it does not use on `test/data/explicit500.yaml`, Explicit500's |
| `test/sim_drive.py` | a scripted drive in the Robocasa sim (stand, walk, turn, side-step, arm goal), measured on FAST-LIO's `/Odometry` |
| `test/test_heartbeat_gate.py` | a real node with the shipped policy on a non-zero DDS domain: silent, in `waiting_heartbeat`, until the first beat, then it pre-poses from the pose at the beat |
| `test/test_imu_calibration.py` | the calibration's math, the export contract in the controller, the estimator on synthetic kinematics |
| `test/test_gyro_lowpass.py` | the gyro low-pass: a Butterworth at any sample rate, its step response, its restart after a dropout |

## Exporting a checkpoint

A policy's networks are Git LFS files: after cloning, `git lfs install` once and `git lfs pull` fetch them.

These commands are run in Locomanipulation_game. Exporting needs only torch, no simulator:

```bash
P=<GOLEM>/core_ws/src/lmg_deploy/policies/h12
rm -f $P/legs.pt $P/arms.pt $P/export.yaml $P/h12.yaml                          # the old iteration goes
python scripts/skrl/export_marl.py --checkpoint logs/skrl/locomanip_marl/<run>/checkpoints/agent_<step>.pt \
  --out $P                                                                       # writes h12.yaml too
rm -f $P/estimator.pt
```

Then rebuild the package (`colcon build --symlink-install --packages-select lmg_deploy`). Replace, never add:
the launches run `policies/h12/h12.yaml` and nothing else, and an older iteration comes back from git history or
by re-exporting the checkpoint its `export.yaml` names.

It writes:
- `legs.pt` and `arms.pt`: the observation scaler, the network and the ±10 clamp. An LSTM actor exports as
  `forward(obs, h, c) -> (action, h, c)`; the node carries `h, c` from step to step and zeroes them at every reset
  (`networks.RecurrentTorchScriptNetwork`).
- `export.yaml`: the run's navigation rest pose, its `target_limits` (runs trained inside GOLEM's position e-stop
  range: joint -> [low, high], narrowing the target bounds), `gait_phase_cos` (the actors observe each leg's gait
  phase cos after its sin), `imu` (the frame the run's IMU sat in, its offset, and `gyro_lowpass_hz` when the gyro
  was low-passed) and the source files. The sport-mode runs' exports also carry their control rate (`control_dt`
  0.01), gains, stance, per-joint action scales, `max_ik_step`, gait period, standing threshold, command ranges and
  low-pass (`command_filter_tau`), their observation term lists, which the controller checks against its own (the
legs' list may leave out `ee_targets`: LocoManip-LegsOnly-v0 runs from 2026-11-16, 83 inputs), and
  `arm_goals` (false: the run never trained an arm goal, so an arm goal is refused). The node
  loads it over the policy's yaml and runs at 1 / `control_dt`. A new policy's yaml is the template with
  `legs.num_obs` / `arms.num_obs` set to the networks' sizes (export_marl.py writes them from 2026-11-20).
- `imu.source: lowstate` (runs from 2026-11-20, trained on the torso IMU as /lowstate reports it): the node reads
  `LowState.imu_state` (quaternion, accelerometer; the gyroscope through `imu.gyro_lowpass_hz`; marl_deploy_node.py's
  IMU section). Exports on the `/livox/imu` attitude path (an `imu.source` other than `lowstate`, runs before
  2026-11-20) are refused at start. Those runs' actors drop
  `base_lin_acc` (`imu.acc: dropped`). `imu.orientation` says how its quaternion was trained: `calibrated` (the
  robot's static offset removed, only `imu.residual_deg` drawn): the deploy applies the robot's calibration and
  refuses to start without one; `raw`, or absent (the exports with `imu.quat_bias_deg`, trained on the bias itself):
  the quaternion as reported, the robot's calibration ignored. Anything else is refused ("IMU calibration" below).
- `engage_blend_s` (runs from 2026-11-22, trained with it): over the first `engage_blend_s` after the engage the legs'
  targets go linearly from the pre-pose's last command to the policy's (the legs observe the blended targets, as in
  training), so a policy engaged from a sagged pose does not jump its targets. Only the engage blends: the sim's
  restart at the band's release does not. Absent or 0: none.
- `leg_action_lowpass_hz` (runs from 2026-11-24, trained with it): the legs' targets go through a one-pole low-pass at
  this cutoff each control step, after the engage blend and before the bounds (the legs observe the filtered targets,
  as in training). It starts at the pre-pose's last command at the engage, else at the measured legs. Absent or 0:
  none. A policy trained with it must not run on an lmg_deploy without it.
- `joint_vel_lowpass` (runs from 2026-11-24): the motor drivers' 2nd-order low-pass at ~21 Hz on the hip, knee and
  torso velocities /lowstate reports, which those runs trained with. The robot does it itself, so the node reads
  nothing of it; it is there for the MuJoCo evals.
- `arm_goal_frame: pelvis` and `arm_goal_keeps_walking: true` (LocoManip-Marl-Direct-v0 from 2026-11-19): an arm goal
  rides on the pelvis (the SLAM odometry doesn't move it) and the velocity command stays live through it.
- With an `arm_ik` block in `export.yaml` (runs whose arms trained with `mdp.IKPostureArmAction`:
  LocoManip-ArmsOnly-v0, LocoManip-Marl-Direct-v0 from 2026-11-19), the arms run as that action term does, not by the
  IK step above: with `trajectory`, the arm target moves from its own last value to a posture (each arm's joints
  together, at `trajectory_max_vel` / `trajectory_max_acc`), the posture being the default angles or, with
  `posture_from_command`, the joint angles of `arm_seed_table.npz`'s row nearest the arm goal (solved for the goal
  itself with `seed_solve`; set once per goal); plus the arms' gravity torque at the target over kp
  (`gravity_offset`; gravity in the pelvis frame from the IMU attitude), the pose-error integral (`integral`) and the
  residual (integrated with `residual_integrate`, latched on the goal with `freeze`). `reset` (engage, band release)
  starts the target at the measured arm angles. `config/example.export.yaml` documents every key;
  `arm_seed_table` names a file in the export's folder (`policies/<robot>/`, copied with the networks). Locomanipulation_game's
  `mujoco_evals/mujoco_sim.py` (`MarlMujoco.process_actions`) is the same arm path in numpy.

The export also writes `estimator.pt` and an `odometry` block in `export.yaml`; the deploy doesn't use them (its
odometry is the SLAM pipeline's, below), so leave them out of `policies/`.

Locomanipulation_game's `deploy_parity.py` checks the deploy controller against the training env, step by step, on
the run's physics: observations term by term, the networks against skrl (LSTM actors stepped alongside), the joint
targets and the rest pose. Against this package (SLAM odometry, no estimator) it skips the odometry comparison; with
`imu.gyro_lowpass_hz` it feeds the controller this package's own `GyroLowpass` (`lmg_deploy.gyro_lowpass`) and
compares it with training's:

```bash
python scripts/skrl/deploy_parity.py --headless --checkpoint <same> \
  --export $P --deploy_pkg <GOLEM>/core_ws/src/lmg_deploy --arm_goal_prob 1.0
```

## Running in the sim

`golem_bringup/launch/h12/lmg_sim.launch.py` starts everything (the policy node comes from the shared
block `h12_locomanipulationgame.launch.py`):
- the state publishers;
- the split safety layer with the real robot's preset (`relax_safety_split`; ankle pitch torque e-stop 95 Nm, above sport mode's 93 Nm peak);
- `marl_deploy_node` with `policies/h12/`, behind `lmg_real`'s operator gates: `start_position_verified:=true`
  (default `false`) starts it, and it holds the pre-pose under the sim's elastic band until `~/confirm_engage`,
  then releases the band;
- SLAM: `golem_slam/launch/navigation.launch.py` with `robot: h12` (FAST-LIO, the floor-level laser scan, slam_toolbox and nav2).
  FAST-LIO's `/Odometry` is what keeps the arm goals fixed in the world.

`frame_task_server` and the lower-body RL node stay off, so this node is the only publisher on each safety channel.

Use a domain other than 0; domain 0 is the real robot. The launch refuses to start when `ROS_DOMAIN_ID` is unset or 0.

```bash
export ROS_DOMAIN_ID=42
docker/scripts/docker_run.sh robocasa env GOLEM_SPAWN_BACKOFF=0.3 \
    /home/code/h12_sim_scripts/launch_robocasa.sh --headless --task AirDryFruit --layout 1 --seed 42   # terminal 1
docker/scripts/docker_run.sh ros                                                                     # terminal 2, then:
ros2 launch golem_bringup lmg_sim.launch.py start_position_verified:=true [use_skills:=true]
ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger   # terminal 3: engage, as on the robot
```

**Launch arguments:** `start_position_verified` (default `false`, as in `lmg_real`), `use_slam` (default `true`;
with `false` the arm goals ride with the pelvis and the node warns), `use_rviz`, `prepose_ramp_s`, `prepose_max_vel`,
`safety_config`, `use_skills` (default `false`; `true` starts `golem_skills`, whose `/skill/frontier_explore` explores
the kitchen through Nav2: `tests/sim/test_lmg_explore.sh` runs it closed loop).

**Driving it** (from `docker exec -it golem_ros bash`). The node walks on `/cmd_vel` once engaged, the arms at the
rest pose:

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"   # walk (clipped to the export's command_ranges)
ros2 topic echo /marl_deploy_node/state         # idle, waiting_heartbeat, preposing, waiting_confirm, nav, arm_goal, fault: ...
```

**Arm goals.**
- An arm goal takes over at once (`settle_to_arm_s` and `settle_to_nav_s` must be 0: a settle at the switch is refused).
- The goal is a `left_wrist_yaw_link` and `right_wrist_yaw_link` pose in the pelvis frame, taken when it starts.
  The SLAM odometry keeps it fixed in the world from then on: each FAST-LIO pose (`/Odometry`, camera_init -> body,
  ~10 Hz) is carried from FAST-LIO's body, the IMU it integrates (`slam_frame: livox_imu_link`), to the pelvis by forward kinematics, through the
  torso joint, and the goal moves by the pelvis's motion since the previous one. Training used a learned estimator
  instead; with no fresh `/Odometry` while reaching the node warns and the goal stays fixed to the pelvis.
- `/cmd_vel` is ignored during an arm goal; the policy only ever saw a zero command there.

## Real robot

On the onboard PC, run `golem_bringup/launch/h12/lmg_real.launch.py` **instead of**
`h12_real_robot.launch.py`, never beside it. It starts the same drivers, state publishers, estop, safety
layer and `golem_slam` navigation stack, without `frame_task_server`, plus the shared game block on wall-clock time. Keep the lower-body RL controller
off.

```bash
export ROS_DOMAIN_ID=0
ros2 launch golem_bringup lmg_real.launch.py                                # sensors, safety, SLAM
ros2 launch golem_bringup lmg_real.launch.py start_position_verified:=true  # + the policy node
ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger          # engage
```

- `start_position_verified:=true` (default `false`) starts `marl_deploy_node`. It pre-poses all 27 joints, then holds.
- Engaging zeroes the velocity command: a `/cmd_vel` sent before the confirm is dropped, so only a new message
  after `nav` appears moves the robot. A publisher still streaming (e.g. `ros2 topic pub -r`) sets it again on its
  next message.
- The node always waits for `~/confirm_engage`, here as in `lmg_sim`, and never calls the sim's elastic band; neither is an argument.
- Drive with `/cmd_vel` as above.
- Same `safety_config` and `use_slam` arguments as the sim launch; keep `use_slam` on for arm goals.

**Before the first real run:**
- **IMU.** The node reads the torso IMU as `/lowstate` reports it (`imu.source: lowstate`, `imu.frame: imu_link`;
  an export with another source is refused at start): the quaternion, corrected by the robot's calibration for a
  calibrated policy, and the gyroscope through the low-pass the run trained with, `imu.gyro_lowpass_hz` in
  `export.yaml` (`gyro_lowpass.GyroLowpass`, a 2nd-order Butterworth discretized exactly over each sample's dt from
  `LowState.tick`, so training's physics rate and `/lowstate`'s realize the same filter).
- **IMU calibration** for torso-IMU policies (`imu.source: lowstate`): below.
- **Safety-layer gaps.** The safety layer holds the last command on a silent channel. Its stale times only report a quiet input (`input_quiet`): in `relax_safety_split`, the lmg bringups' preset, `stale.lower_seconds` is 0.2 and `stale.upper_seconds` 0 (off), and neither stops the relay. If this node latches silent, the safety layer keeps sending its last targets until it is stopped.

## IMU calibration

The H1-2's torso IMU (`/lowstate` `imu_state`) reads its quaternion about 2-3 deg nose up with the torso level.
Policies trained before the calibration existed draw that bias in training (`export.yaml` `imu.quat_bias_deg`,
pitch -3.6..-2.0 deg) and run on the quaternion as reported. Newer ones train on a level IMU with a small residual
(`imu.orientation: calibrated`, `imu.residual_deg` +-1 deg), so the deploy must remove the robot's offset first.

**Convention** (training's, Locomanipulation_game's `lowstate_imu.py`): the calibration is the attitude the quaternion
reads with `imu_link` level, (roll, pitch, yaw) deg; reported = true * R_bias in the body frame, R_bias = Rz(yaw)
Ry(pitch) Rx(roll) (Isaac's `quat_from_euler_xyz`), pitch positive nose down, roll positive tilting right. The
correction is corrected = reported * R_bias^-1 (`imu_calibration.correct_imu(quat, gyro, calibration)`). Training
biases the quaternion only, so the gyro is left as reported; `rotate_gyro: true` re-expresses it (and the specific
force) in the level frame, for an offset that is a physical mounting tilt (gyro_level = R_bias gyro_imu).

**Where it lives:** `config/imu_calibration.yaml`, one entry per robot named as its policy folder (`h12`, `g1`),
and one for its simulators (`h12_sim`: all zero, the sim's torso IMU being MuJoCo's exact attitude; `lmg_sim` runs
the node with `robot: h12_sim` through `h12_locomanipulationgame`'s `imu_calibration_robot`, and `lmg_real` keeps
the default `h12`):
`roll_deg`, `pitch_deg`, `yaw_deg` (0), `rotate_gyro` (false), or `null` while unmeasured. `marl_deploy_node` reads
its robot's entry (parameters `robot`, default the policy folder's name, and `imu_calibration_file`, default the
installed `config/imu_calibration.yaml`) and passes it to `MarlController(cfg, kin, legs, arms,
imu_calibration={...})`, which applies it inside the controller, so the observations, the arms' gravity offset
(`gravity_b`) and `step()` all see the corrected attitude. The rule, in the constructor
(`imu_calibration.resolve_calibration`):

| `export.yaml` `imu.orientation` | robot calibration set | robot calibration `null` |
|---|---|---|
| `calibrated` (needs `imu.source: lowstate`) | applied, logged | **refused** at start (ValueError) |
| `raw`, or absent (older exports) | ignored, with a warning | nothing applied |
| anything else | refused | refused |

**Measuring it.** Lower the robot onto level floor in the pre-pose, both soles flat, the hoist slack, and record a few
seconds still (`waiting_confirm` is enough): `ros2 bag record /lowstate`. Then, offline (inside the ROS image, or
anywhere with pinocchio and the `rosbags` library: `--msg-dir` unitree_hg's msg folder):

```bash
ros2 run lmg_deploy calibrate_imu <bag> [<bag> ...] --urdf <CL_Assets>/ros_assets/h1_2_magpie.urdf
```

It picks the still samples (every leg joint under 0.2 rad/s, the gyro under 0.1 rad/s, both knees at or below -8 Nm
`tau_est`: carrying weight, in runs of at least 1 s), takes `imu_link`'s tilt from each sole by forward kinematics
(the sole level, yaw unobservable: it stays 0), and fits the quaternion's tilt to it. Each foot gives its own estimate;
their disagreement is the quality check (over 0.5 deg it warns: a foot on its edge, uneven floor, the hoist carrying
weight, a joint zero offset). Write the value in `config/imu_calibration.yaml` with the date, the method and each
bag's numbers, only when two recordings agree within 0.3 deg in roll and pitch. On floor that is not level within
about 0.2 deg, record pairs on one spot, facing forward and turned 180 deg, and compare the pairs' means: a floor
slope flips sign with the turn. A taut hoist is fine when the pre-pose does not stand alone; a sole it lifts shows
in the feet's disagreement.

The H1-2's entry (roll +1.0, pitch -1.7, 2026-11-24) holds only with the right ankle zero offsets set on its motor
controller (motor 10 -3.4, motor 11 4.7 in the Unitree app). Without them the right sole read ~3.4 deg toe-up and
~1.2 deg rolled against the left on flat feet, so the feet disagreed by ~3 deg in every earlier recording. It is the
mean of two pairs of recordings on one spot, facing forward and turned 180 deg (a floor slope flips sign with the
turn, so a pair's mean cancels it), in the pre-pose with the hoist taut (the pre-pose does not balance alone); the
pairs agree within 0.29 roll, 0.04 pitch, and the feet within 0.41 in every recording. The torso accelerometer reads
1.0 roll, 0.65 pitch away from the leg kinematics, floor-free (the quaternion against it: roll -0.2, pitch -2.35):
most likely an IMU board tilted on `torso_link` (else an offset shared by both legs, which a flat-footed stand
cannot see); the kinematic value removes the tilt with the estimator's offset.
`config/imu_calibration.yaml` has the numbers.

**At run time,** once the policy is engaged on the torso IMU and has stood still on both loaded feet at zero command
for 5 s, the node compares the IMU tilt the policy sees with the kinematic one (the estimator's function) and warns
once if they differ by more than 1 deg (a raw policy logs it: its offset is expected).
