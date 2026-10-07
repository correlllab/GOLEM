# locomotion_game_deploy (LocoMotionGameDeploy)

Runs the two-agent MARL policy from Locomanipulation_game
(`LocoManip-Marl-Flat-IK-Direct-v0`, branch `marl-direct`) on the H1-2: a
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
| `locomotion_game_deploy/controller.py` | ROS-free env step: odometry → mode → observations (legs 96, arms 124) → joint targets |
| `locomotion_game_deploy/kinematics.py` | Pinocchio wrist FK, Jacobian and IK step, in the pelvis frame |
| `locomotion_game_deploy/rotations.py` | numpy versions of the isaaclab.utils.math functions training used |
| `locomotion_game_deploy/networks.py` | TorchScript loader, and the `zero` stand-ins for plumbing tests |
| `locomotion_game_deploy/marl_deploy_node.py` | the ROS node: safety channels, heartbeat, pre-pose, band, operator interface |
| `policies/marl_ik/marl_ik.yaml` | deploy config: joint order, defaults, gains, scales, command ranges (each from the training env) |
| `policies/marl_ik/{legs,arms,estimator}.pt`, `export.yaml` | written by the export, below |
| `test/test_controller.py` | unit tests (rotations vs Isaac Lab, Jacobian, IK, observation layout, odometry) |

## Exporting a checkpoint

These commands are run in Locomanipulation_game. Exporting needs only torch, no simulator:

```bash
python scripts/skrl/export_marl.py \
  --checkpoint logs/skrl/locomanip_marl/<run>/checkpoints/agent_<step>.pt \
  --template  <GOLEM>/core_ws/src/locomotion_game_deploy/policies/marl_ik/marl_ik.yaml \
  --out       <GOLEM>/core_ws/src/locomotion_game_deploy/policies/marl_ik
```

It writes:
- `legs.pt` and `arms.pt`: observation scaler, MLP mean, ±10 clamp.
- `estimator.pt`: input norm and MLP.
- `export.yaml`: the run's navigation rest pose and the source files. The node loads it over `marl_ik.yaml`.

Then check the deploy controller against the training env, step by step:

```bash
python scripts/skrl/deploy_parity.py --headless \
  --export <GOLEM>/core_ws/src/locomotion_game_deploy/policies/marl_ik \
  --deploy_pkg <GOLEM>/core_ws/src/locomotion_game_deploy \
  --estimator logs/skrl/locomanip_marl/<run>/estimator/estimator_<step>.pt \
  --log_dir logs/skrl/locomanip_marl/<run>
```

## Running in the sim

Use a domain other than 0; domain 0 is the real robot.

```bash
export ROS_DOMAIN_ID=42
docker/scripts/docker_run.sh robocasa --headless           # terminal 1
docker/scripts/docker_run.sh ros                            # terminal 2, then:
ros2 launch h1_bringup h1_sim_bringup.launch.py lowerbody:=marl_ik use_skills:=false use_nav:=false
```

`lowerbody:=marl_ik` gates `frame_task_server`, `lowerbody_controller_node`
and the skills off. Only one publisher may feed each safety channel.

`marl_policy_mode:=zero` runs without exported networks: the legs hold the
default pose and the arms follow pure IK. That checks the plumbing only. The
robot falls once the band releases.

Drive it from a third shell (`docker exec -it golem_ros bash`):

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
- The goal is a `left_wrist_yaw_link` and `right_wrist_yaw_link` pose in the pelvis frame, taken when it arrives. The learned odometry keeps it fixed in the world from then on.
- `/cmd_vel` is ignored during an arm goal; the policy only ever saw a zero command there.
- Ending a goal settles for 1.5 s, then walking resumes.

## Real robot

1. Onboard PC: `ros2 launch h1_bringup h1_real_robot_bringup.launch.py use_frame_task:=false`
2. Desktop: `ros2 launch h1_bringup h1_real_desktop_bringup.launch.py lowerbody:=marl_ik start_position_verified:=true use_skills:=false`
3. The node pre-poses all 27 joints, then holds. Engage it with `ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger`.

**Before the first real run:**
- **IMU frame.** Training read the IMU on `torso_link` (`imu_link`, offset (-0.0445, -0.0189, 0.2776), no rotation). The MuJoCo sim's IMU is in the same place. Confirm that the real `LowState.imu_state` comes from that same IMU.
- **Safety-layer gaps.** The safety layer holds the last command on a silent channel: neither input channel has a staleness watchdog in the sim or real configs (`upper_stale_seconds: 0.0`). If this node latches silent, the safety layer keeps sending its last targets until it is stopped.
