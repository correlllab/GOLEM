# locomotion_game_deploy (LocoMotionGameDeploy)

Runs the two-agent MARL policy from Locomanipulation_game (branch
`marl-direct`) on the H1-2: `policies/marl_golem` for run R of
`LocoManip-Marl-Direct-v0`, trained against h12_safety_layer's e-stops, and
`policies/marl_ik` for `golemlim120k` (agent_120000, a partly trained run of the
same task whose joint targets were clipped into GOLEM's position e-stop range;
see its `export.yaml`). A
**legs** agent that walks on `/cmd_vel`, and an **arms** agent that reaches
wrist-pose goals through a damped-least-squares IK step plus a learned
residual. One node, `marl_deploy_node`, owns all 27 joints. It publishes
through the split safety layer and obeys `/safety/heartbeat`.

Modelled on `h12_lowerbody_rl`'s `lowerbody_controller_node`. The node goes
through the same steps (pre-pose ramp, operator confirm, elastic-band
release) and uses the same heartbeat watchdog.

## Layout

| File | What it is |
|---|---|
| `locomotion_game_deploy/controller.py` | ROS-free env step: odometry → mode → observations (legs 96, arms 124; 95 and 123 with `height_drop_obs: false`) → joint targets |
| `locomotion_game_deploy/kinematics.py` | Pinocchio wrist FK, Jacobian and IK step, in the pelvis frame |
| `locomotion_game_deploy/rotations.py` | numpy versions of the isaaclab.utils.math functions training used |
| `locomotion_game_deploy/networks.py` | TorchScript loader, and the `zero` stand-ins for plumbing tests |
| `locomotion_game_deploy/marl_deploy_node.py` | the ROS node: safety channels, heartbeat, pre-pose, band, operator interface |
| `locomotion_game_deploy/game_commander_node.py` | sends walking and arm-goal commands on training's schedule; lidar obstacle guard, SLAM geofence, event log |
| `policies/<name>/<name>.yaml` | deploy config: joint order, defaults, gains, scales, odometry, bounds, command ranges (each from the training env) |
| `policies/<name>/{legs,arms,estimator}.pt`, `export.yaml`, `arm_goals_*.npz` | written by the export, below; `marl_golem` holds run R's and `marl_ik` golemlim120k's (no arm goals), the `.pt` and `.npz` files in Git LFS (`policies/.gitattributes`) |
| `test/test_controller.py` | unit tests (rotations vs Isaac Lab, Jacobian, IK, observation layout, odometry) |

## Exporting a checkpoint

`policies/marl_golem` already holds run R's export (agent_24000). Its networks and arm goals are Git LFS files:
after cloning, `git lfs install` once and `git lfs pull` fetch them.

These commands are run in Locomanipulation_game. Exporting needs only torch, no simulator:

```bash
P=<GOLEM>/core_ws/src/locomotion_game_deploy/policies/marl_golem
python scripts/skrl/export_marl.py --checkpoint logs/skrl/locomanip_marl/<run>/checkpoints/agent_<step>.pt \
  --out $P
python scripts/skrl/export_arm_goals.py --checkpoint <same> --levels 4:10,7:10,10:10 --out $P --headless
```

It writes:
- `legs.pt` and `arms.pt`: the observation scaler, the MLP's mean and the ±10 clamp.
- `estimator.pt`: the input norm and MLP.
- `export.yaml`: the run's navigation rest pose, its `target_limits` (runs trained inside GOLEM's position e-stop
  range: joint -> [low, high], narrowing the target bounds) and the source files. The node loads it over the policy's
  yaml.
- `arm_goals_<spread>_<drop>.npz`: goals drawn by the training command, for the commander.

Then check the deploy controller against the training env, step by step. It compares observations term by term, the networks against skrl, the joint targets, and leg odometry against the true pelvis motion:

```bash
python scripts/skrl/deploy_parity.py --headless --checkpoint <same> \
  --export $P --deploy_pkg <GOLEM>/core_ws/src/locomotion_game_deploy --arm_goal_prob 1.0
```

## Running in the sim

`h1_bringup/launch/locomanipulationgame_sim_bringup.launch.py` starts everything (the policy node, commander and
SLAM come from the shared block `h1_locomanipulationgame.launch.py`):
- the state publishers;
- the split safety layer with the real robot's preset (`relax_safety_split`);
- `marl_deploy_node` with `policies/<policy>/` (default `marl_golem`);
- `game_commander_node`, which sends walking and arm-goal commands on training's schedule;
- SLAM: FAST-LIO, slam_toolbox and the pelvis-height laser scan. nav2 isn't started.

`frame_task_server` and the lower-body RL node stay off, so this node is the only publisher on each safety channel.

Use a domain other than 0; domain 0 is the real robot. The launch refuses to start when `ROS_DOMAIN_ID` is unset or 0.

```bash
export ROS_DOMAIN_ID=42
docker/scripts/docker_run.sh robocasa env GOLEM_SPAWN_BACKOFF=0.3 \
    /home/code/h12_sim_scripts/launch_robocasa.sh --headless --task AirDryFruit --layout 1 --seed 42   # terminal 1
docker/scripts/docker_run.sh ros                                                                     # terminal 2, then:
ros2 launch h1_bringup locomanipulationgame_sim_bringup.launch.py
```

**Launch arguments:**
- `commander_mode`: `training` (the default), or `sequence` for one scripted pass: back off, turn, walk, a standing reach, a crouch reach, walk.
- `use_commander:=false`: drive the robot yourself, as below.
- `goals:=arm_goals_<spread>_<drop>.npz`: the arm-goal sample set.
- `commander_log:=<file>`: a JSON-lines event log.
- `use_slam`, `use_rviz`, `policy`. `policy:=marl_ik` has no arm-goal samples, so run it with `use_commander:=false`.
- `policy_mode:=zero`: run without exported networks. The legs hold the default pose and the arms follow pure IK, which checks the plumbing only; the robot falls once the band releases.

`tests/locomanipulation_game/run.sh` runs the same launch in disposable containers. It records the sim without modifying it, renders a follow-camera video, and scores the run with `analyze_run.py`.

**Driving it yourself** (`use_commander:=false`, from `docker exec -it golem_ros bash`):

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"   # walk (clipped to vx ±0.7, vy ±0.3, wz ±0.5)
ros2 topic pub --once /marl_deploy_node/arm_goal geometry_msgs/msg/PoseArray \
  "{poses: [{position: {x: 0.30, y: 0.25, z: 0.05}, orientation: {w: 1.0}},
            {position: {x: 0.30, y: -0.25, z: 0.05}, orientation: {w: 1.0}}]}"  # left, right; pelvis frame
ros2 service call /marl_deploy_node/end_arm_goal std_srvs/srv/Trigger
ros2 topic echo /marl_deploy_node/state         # idle, preposing, waiting_confirm, nav, settle_to_arm, arm_goal, settle_to_nav, fault: ...
ros2 topic echo /marl_deploy_node/wrist_error   # m, left and right
```

**Arm goals.**
- An arm goal stops the robot, waits 0.75 s, then reaches.
- The goal is a `left_wrist_yaw_link` and `right_wrist_yaw_link` pose in the pelvis frame, taken when it arrives. The odometry keeps it fixed in the world from then on: the learned estimator for `marl_ik`, and for `marl_golem` the IMU attitude plus the planted feet.
- `/cmd_vel` is ignored during an arm goal; the policy only ever saw a zero command there.
- Ending a goal settles for 1.5 s, then walking resumes.

## Real robot

On the onboard PC, run `h1_bringup/launch/locomanipulationgame_real_bringup.launch.py` **instead of**
`h1_real_robot_bringup.launch.py`, never beside it. It starts the same drivers, state publishers, estop and safety
layer, without `frame_task_server`, plus the shared game block on wall-clock time. Keep the lower-body RL and MJPC
controllers off.

```bash
export ROS_DOMAIN_ID=0
ros2 launch h1_bringup locomanipulationgame_real_bringup.launch.py                                # sensors, safety, SLAM
ros2 launch h1_bringup locomanipulationgame_real_bringup.launch.py start_position_verified:=true  # + the policy node
ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger                           # engage
```

- `start_position_verified:=true` (default `false`) starts `marl_deploy_node`. It pre-poses all 27 joints, then holds.
- Engaging zeroes the velocity command: a `/cmd_vel` sent before the confirm is dropped, so only a new message
  after `nav` appears moves the robot. A publisher still streaming (e.g. `ros2 topic pub -r`) sets it again on its
  next message.
- The node always waits for `~/confirm_engage` here, and never calls the sim's elastic band; neither is an argument.
- The commander is off by default (`use_commander:=false`): drive with `/cmd_vel` and `~/arm_goal` as above.
  `use_commander:=true` starts it alongside the node.
- Same `policy`, `policy_mode`, `safety_config`, `use_slam` and `commander_*` arguments as the sim launch.

**Before the first real run:**
- **IMU frame.** Training read the IMU on `torso_link` (`imu_link`, offset (-0.0445, -0.0189, 0.2776), no rotation). The MuJoCo sim's IMU is in the same place. Confirm that the real `LowState.imu_state` comes from that same IMU.
- **Safety-layer gaps.** The safety layer holds the last command on a silent channel: neither input channel has a staleness watchdog in the sim or real configs (`upper_stale_seconds: 0.0`). If this node latches silent, the safety layer keeps sending its last targets until it is stopped.
