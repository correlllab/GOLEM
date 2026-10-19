# lmg_deploy (LocoMotionGameDeploy)

Runs the two-agent MARL policy from Locomanipulation_game (branch
`marl-direct`) on the H1-2: `policies/explicit500`, `Explicit500` of
`LocoManip-Marl-Direct-v0` (agent_196800, 200k steps: LSTM actors, explicit PD
at 500 Hz, a 0-20 ms target delay, pushes and rough terrain in training; the
legs crouch for low goals). A
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
| `lmg_deploy/controller.py` | ROS-free env step: SLAM odometry → mode → observations (legs 95, arms 123) → joint targets |
| `lmg_deploy/kinematics.py` | Pinocchio wrist FK, Jacobian and IK step, in the pelvis frame |
| `lmg_deploy/rotations.py` | numpy versions of the isaaclab.utils.math functions training used |
| `lmg_deploy/networks.py` | TorchScript loader, and the zero stand-ins the unit tests use |
| `lmg_deploy/marl_deploy_node.py` | the ROS node: safety channels, heartbeat, pre-pose, band, operator interface |
| `policies/explicit500/explicit500.yaml` | deploy config: joint order, defaults, gains, scales, bounds, command ranges (each from the training env), and the SLAM body frame |
| `policies/explicit500/{legs,arms}.pt`, `export.yaml` | written by the export, below; the `.pt` files in Git LFS (`policies/.gitattributes`) |
| `bags/<name>/` | rosbag2 recordings of Explicit500 on the real robot (`policy_bag_walk`: walking to nav2 goals; `reach`: an arm goal, ended by the hardware e-stop), the `.db3` files in Git LFS |
| `test/test_controller.py` | unit tests (rotations vs Isaac Lab, Jacobian, IK, observation layout, SLAM odometry, modes) |
| `custom_ros_messages/srv/Lmg*.srv` | the node's services: `LmgSetMode` (walk / arm), `LmgSetArmGoals` (left, right pose) |

## Exporting a checkpoint

`policies/explicit500` holds Explicit500's export. Its networks are Git LFS files: after cloning, `git lfs install`
once and `git lfs pull` fetch them.

These commands are run in Locomanipulation_game. Exporting needs only torch, no simulator:

```bash
P=<GOLEM>/core_ws/src/lmg_deploy/policies/explicit500
python scripts/skrl/export_marl.py --checkpoint logs/skrl/locomanip_marl/<run>/checkpoints/agent_<step>.pt \
  --out $P
```

It writes:
- `legs.pt` and `arms.pt`: the observation scaler, the network and the ±10 clamp. An LSTM actor exports as
  `forward(obs, h, c) -> (action, h, c)`; the node carries `h, c` from step to step and zeroes them at every reset
  (`networks.RecurrentTorchScriptNetwork`).
- `export.yaml`: the run's navigation rest pose, its `target_limits` (runs trained inside GOLEM's position e-stop
  range: joint -> [low, high], narrowing the target bounds) and the source files. The node loads it over the
  policy's yaml.

The export also writes `estimator.pt` and an `odometry` block in `export.yaml`; the deploy doesn't use them (its
odometry is the SLAM pipeline's, below), so leave them out of `policies/`.

Locomanipulation_game's `deploy_parity.py` checks the deploy controller against the training env, step by step, on
the run's physics: observations term by term, the networks against skrl (LSTM actors stepped alongside), the joint
targets and the rest pose. It was written for the estimator odometry and passes the controller an estimator, which
`MarlController` no longer takes, so it needs that argument dropped before it runs against this package:

```bash
python scripts/skrl/deploy_parity.py --headless --checkpoint <same> \
  --export $P --deploy_pkg <GOLEM>/core_ws/src/lmg_deploy --arm_goal_prob 1.0
```

## Running in the sim

`h12_bringup/launch/lmg_sim_bringup.launch.py` starts everything (the policy node comes from the shared
block `h12_locomanipulationgame.launch.py`):
- the state publishers;
- the split safety layer with the real robot's preset (`relax_safety_split`);
- `marl_deploy_node` with `policies/explicit500/`;
- SLAM: `h12_slam/launch/h12_navigation.launch.py` (FAST-LIO, the pelvis-height laser scan, slam_toolbox and nav2).
  FAST-LIO's `/Odometry` is what keeps the arm goals fixed in the world.

`frame_task_server` and the lower-body RL node stay off, so this node is the only publisher on each safety channel.

Use a domain other than 0; domain 0 is the real robot. The launch refuses to start when `ROS_DOMAIN_ID` is unset or 0.

```bash
export ROS_DOMAIN_ID=42
docker/scripts/docker_run.sh robocasa env GOLEM_SPAWN_BACKOFF=0.3 \
    /home/code/h12_sim_scripts/launch_robocasa.sh --headless --task AirDryFruit --layout 1 --seed 42   # terminal 1
docker/scripts/docker_run.sh ros                                                                     # terminal 2, then:
ros2 launch h12_bringup lmg_sim_bringup.launch.py
```

**Launch arguments:** `use_slam` (default `true`; with `false` the arm goals ride with the pelvis and the node warns),
`use_rviz`, `policy` (a directory under `policies/`, default `explicit500`), `prepose_ramp_s`, `prepose_max_vel`,
`safety_config`.

**Driving it** (from `docker exec -it golem_ros bash`). The node starts in walking mode once engaged:

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"   # walk (clipped to vx ±0.7, vy ±0.3, wz ±0.5)
ros2 service call /marl_deploy_node/set_arm_goals custom_ros_messages/srv/LmgSetArmGoals \
  "{left: {position: {x: 0.30, y: 0.22, z: -0.15}, orientation: {w: 1.0}},
    right: {position: {x: 0.30, y: -0.22, z: -0.15}, orientation: {w: 1.0}}}"  # pelvis frame; kept until arm mode
ros2 service call /marl_deploy_node/set_mode custom_ros_messages/srv/LmgSetMode "{mode: arm}"    # stop and reach (crouch)
ros2 service call /marl_deploy_node/set_mode custom_ros_messages/srv/LmgSetMode "{mode: walk}"   # back to /cmd_vel
```

`set_arm_goals` in arm mode moves the goals at once; while walking it keeps them for the next `set_mode arm`, and
they are taken in the pelvis frame of that switch. Arm mode without goals holds the rest pose (standing still).
`set_mode` is refused until the node is engaged. A PoseArray sets the goals and switches to arm mode in one message:

```bash
ros2 topic pub --once /marl_deploy_node/arm_goal geometry_msgs/msg/PoseArray \
  "{poses: [{position: {x: 0.30, y: 0.25, z: 0.05}, orientation: {w: 1.0}},
            {position: {x: 0.30, y: -0.25, z: 0.05}, orientation: {w: 1.0}}]}"  # left, right; pelvis frame
ros2 service call /marl_deploy_node/end_arm_goal std_srvs/srv/Trigger
ros2 topic echo /marl_deploy_node/state         # idle, preposing, waiting_confirm, nav, settle_to_arm, arm_goal, settle_to_nav, fault: ...
ros2 topic echo /marl_deploy_node/wrist_error   # m, left and right
```

**Arm goals.**
- An arm goal stops the robot and reaches at once (`settle_to_arm_s` and `settle_to_nav_s` are 0 for Explicit500).
- The goal is a `left_wrist_yaw_link` and `right_wrist_yaw_link` pose in the pelvis frame, taken when it arrives.
  The SLAM odometry keeps it fixed in the world from then on: each FAST-LIO pose (`/Odometry`, camera_init -> body,
  ~10 Hz) is carried from the lidar (`slam_frame: livox_link`) to the pelvis by forward kinematics, through the
  torso joint, and the goal moves by the pelvis's motion since the previous one. Training used a learned estimator
  instead; with no fresh `/Odometry` while reaching the node warns and the goal stays fixed to the pelvis.
- `/cmd_vel` is ignored during an arm goal; the policy only ever saw a zero command there.

## Real robot

On the onboard PC, run `h12_bringup/launch/lmg_real_bringup.launch.py` **instead of**
`h12_real_robot_bringup.launch.py`, never beside it. It starts the same drivers, state publishers, estop, safety
layer and `h12_slam` navigation stack, without `frame_task_server`, plus the shared game block on wall-clock time. Keep the lower-body RL and MJPC
controllers off.

```bash
export ROS_DOMAIN_ID=0
ros2 launch h12_bringup lmg_real_bringup.launch.py                                # sensors, safety, SLAM
ros2 launch h12_bringup lmg_real_bringup.launch.py start_position_verified:=true  # + the policy node
ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger          # engage
```

- `start_position_verified:=true` (default `false`) starts `marl_deploy_node`. It pre-poses all 27 joints, then holds.
- Engaging zeroes the velocity command: a `/cmd_vel` sent before the confirm is dropped, so only a new message
  after `nav` appears moves the robot. A publisher still streaming (e.g. `ros2 topic pub -r`) sets it again on its
  next message.
- The node always waits for `~/confirm_engage` here, and never calls the sim's elastic band; neither is an argument.
- Drive with `/cmd_vel`, `~/set_mode` and `~/set_arm_goals` as above.
- Same `policy`, `safety_config` and `use_slam` arguments as the sim launch; keep `use_slam` on for arm goals.

**Before the first real run:**
- **IMU.** The node reads only the MID360's `/livox/imu` (`livox_imu_upright`'s output), never `LowState.imu_state`, whose attitude carries a ~2.5 deg calibration error. `/livox/imu` is in the URDF's `livox_imu_link` (the IMU chip). The node carries the readings from there into the frame the policy trained with, `export.yaml`'s `imu.frame`, through the URDF:
  - `livox_imu_link`: as is;
  - `livox_link` (the point frame, 0.7 deg from the chip);
  - `imu_link` on `torso_link` for Explicit500 and older (the default when the export has no `imu` block).

  The attitude comes from a complementary filter (`lmg_deploy/livox_imu.py`). The angular velocity passes the gyro low-pass the run trained with, `imu.gyro_lowpass_hz` in `export.yaml` (absent: none, as for Explicit500). `livox_imu.GyroLowpass` is a 2nd-order Butterworth, discretized exactly over each sample's dt, so training's physics rate and the MID360's 200 Hz realize the same filter. At 8 Hz it takes the head's ~20 Hz pitch resonance while walking from 0.5-0.9 to ~0.1 rad/s rms. `livox_joint` and `livox_imu_joint` carry the real robot's measured mount (point frame 8.77 deg, chip 9.45 deg pitch); `imu_offset_{roll,pitch,yaw}_deg` (default 0) trims what is left, live.
- **Safety-layer gaps.** The safety layer holds the last command on a silent channel: neither input channel has a staleness watchdog in the sim or real configs (`upper_stale_seconds: 0.0`). If this node latches silent, the safety layer keeps sending its last targets until it is stopped.
