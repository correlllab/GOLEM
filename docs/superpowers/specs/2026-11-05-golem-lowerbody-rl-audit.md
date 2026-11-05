# Audit: merging `h12_lowerbody_rl` and `g1_homie` into `golem_lowerbody_rl`

Read-only audit, 2026-11-05, branch `dead_code_cleanup` at `5dba200`. Nothing was edited, built or
run. Paths are relative to `/home/max/GOLEM`; `H/` = `core_ws/src/h12_lowerbody_rl/`,
`G/` = `core_ws/src/g1_homie/`. Line numbers are as of this commit (another agent is deleting dead
code on the same branch; small drifts are expected).

Target, in the user's words: rename `h12_lowerbody_rl` to `golem_lowerbody_rl` and merge
`g1_homie` into it, "following the same pattern as the safety layer and upper body controller.
Identical code, different config. The config should determine how the observations and actions
are made and interpreted."

The pattern, as it exists today:

- `golem_safety`: one C++ node, the joint table per robot in `robots/<robot>.yaml`
  (generated, read by both golem packages), run configs in `config/<robot>/*.yaml`, its own
  `launch/safety.launch.py` (`robot`, `config`, `sim`, `use_estop`, `use_sim_time`,
  `start_delay`) included by both bringups.
- `golem_upper_ik`: one Python node, `config/<robot>/robot.yaml` (robot data, strict key set,
  `profile.py:22-26`), `config/<robot>/<config>.yaml` (run configs), `config/<robot>/pairing.yaml`,
  `launch/upper_ik.launch.py` (refuses a bad pairing before any node starts,
  `upper_ik.launch.py:98-104`), joint names → slots from `golem_safety/robots/<robot>.yaml`
  (`profile.py:67-73`), `known_robots()` = robots with data in both packages (`profile.py:76-85`).
  ROS only: `test/test_no_sdk.py` forbids `unitree_sdk2py` in the package's code.
- `tests/docker/test_shared_packages.py:16-26` scans the shared packages' code (not `config/`,
  `robots/`, tests or tools) and fails on a robot-name string literal, a comparison of a
  `*robot*` identifier against a literal, or a code file/dir named for a robot (`:95-98`).

---

## 1. Inventory

### 1.1 `h12_lowerbody_rl` (tracked in the main repo; not a submodule, no LFS)

| File | Lines | What it is |
|---|---|---|
| `H/h12_lowerbody_rl/scripts/lowerbody_controller_node.py` | 495 | The ROS node `lowerbody_controller_node` (class `LowerBodyControllerNode`): params, subs/pubs, IMU trim, pre-pose ramp + settle, hold, band release, 50 Hz tick. |
| `H/h12_lowerbody_rl/supervisor.py` | 265 | `LegSupervisor`, ROS-free phases and faults (waiting/ramping/holding/running/faulted). Added 2026-11-05 (commit `505016c`), modelled on `g1_homie`'s controller. |
| `H/h12_lowerbody_rl/policy.py` | 575 | `RobotState`, `LegCommand`, gravity helpers, IMU-trim quaternions, `Policy` ABC, `WalkPolicy`, `AlmiPolicy`, `FamePolicy`. All observation/action math. |
| `H/h12_lowerbody_rl/policy_manager.py` | 76 | `PolicyManager`: registry + desired/active/commit. Vestigial since policies never switch (only `request` once, `commit`, `reset_active`, `run`, `desired_policy`, `names`, `active_name`, `desired_name` are used; `is_pending` only internally). |
| `H/h12_lowerbody_rl/rma/env_factor_encoder.py` | 61 | `EnvFactorEncoder` MLP (in 21 → 256 → 128 → 8, ELU), FAME's z_t encoder; loads a `state_dict`. |
| `H/h12_lowerbody_rl/rma/env_factor_spec.py` | 93 | `RmaEtSpec`, `UPPER_BODY_JOINT_NAMES`, force ranges: documentation only; nothing in the node uses them (re-exported by `rma/__init__.py:16-24`). Dead-code candidate. |
| `H/h12_lowerbody_rl/rma/__init__.py` | 24 | Re-exports. |
| `H/h12_lowerbody_rl/{__init__,py.typed,scripts/__init__}.py` | 0 | Empty. |
| `H/policies/almi/{almi.yaml,policy_lstm_12800.pt,README.md}` | 47 / 183 KB / 93 | ALMI (TeleHuman/ALMI-Open, BSD-3) TorchScript LSTM + config + provenance. |
| `H/policies/fame/{fame.yaml,policy_3600.pt,encoder_3600.pt}` | 51 / 3.0 MB / 161 KB | FAME RMA policy (TorchScript) + encoder (`state_dict`). Lab-trained (no licence file). |
| `H/policies/walk/{walk.yaml,walkingPolicy.pt}` | 25 / 146 KB | Walking MLP (TorchScript). Provenance not recorded (legged_gym/unitree_rl_gym-style deploy). |
| `H/reference/{deploy_mujoco.py,mujoco_deploy_h12_rma.py,README.md}` | 130 / 622 / 17 | Original standalone deploy scripts (not built, not linted): source of truth for walk/FAME obs math. |
| `H/test/test_supervisor.py` | 264 | Phases/faults on a fake clock (19 tests). |
| `H/test/test_band_node.py` | 98 | Band release against a fake toggle on domain 97 (3 tests). |
| `H/test/test_{copyright,flake8,mypy,pep257,xmllint}.py` | ~25 each | Linters (mypy skipped without `ament_mypy`). |
| `H/setup.py` | 42 | Installs `policies/{walk,fame,almi}/*` to `share/` (`:12-23` of the file body), entry point `lowerbody_controller_node`. `license='TODO'`. |
| `H/package.xml` | 29 | exec_depend rclpy, geometry_msgs, std_msgs, std_srvs, unitree_hg, ament_index_python, numpy, yaml. **No torch** declared. Licence TODO. |
| `H/README.md` | 90 | Behaviour + param table. |

No tests cover any policy's observation or action math (WALK/ALMI/FAME): nothing pins the
obs vectors, scales or the ALMI onset logic.

### 1.2 `g1_homie` (tracked in the main repo)

| File | Lines | What it is |
|---|---|---|
| `G/g1_homie/locomotion_node.py` | 199 | ROS node `locomotion_node` (class `LocomotionNode`): `--config` CLI arg, `policy_dir` param, subs, confirm service, band timer/client. |
| `G/g1_homie/locomotion.py` | 657 | ROS/DDS-free logic: G1 body joint table, `load_policy`, `policy_slots`, `State`, `read_state`, `tick_delta`, `SourceClock`, `gravity_z`, `CommandFilter`, `LocomotionController` (phases, tick grid, readiness, latency, faults). |
| `G/g1_homie/driver.py` | 194 | `LocomotionDriver`: Unitree-SDK publisher + raw cyclonedds readers in a worker thread, CRC, lock around the controller. |
| `G/g1_homie/config.py` | 200 | Strict YAML schema check (`check_config`), `resolve_config_path`, `init_channel_factory` (domain/interface, domain-0 guard). |
| `G/g1_homie/band.py` | 58 | `BandRelease`, ROS-free one-call band logic. |
| `G/config/sim.yaml`, `G/config/g1.yaml` | 78 / 71 | Run configs (sim / real). Installed (`setup.py` data_files). |
| `G/policy/deploy.onnx` | 1.8 MB | HOMIE's policy (CC BY-NC(-SA) 4.0). |
| `G/policy/g1.yaml` | 61 | HOMIE's policy config (joints, scales, gains, ranges, MuJoCo knobs). |
| `G/policy/homie_policy.py` | 190 | HOMIE's obs/history/action code (`HomiePolicy`), onnxruntime one-thread session. CC BY-NC-SA. |
| `G/policy/mujoco_deploy_g1.py` | 127 | Standalone MuJoCo sim2sim (`--cmd`, `--height`, `--headless`). |
| `G/policy/test_homie_policy.py` | 201 | Tests of `homie_policy.py` (9 tests: layout, history order, action→target, input size, one-thread session, clip, gains). |
| `G/policy/{LICENSE,README.md}` | 80 / 180 | CC BY-NC 4.0 text; README with credits and "treat as CC BY-NC-SA". |
| `G/LICENSE` | 17 | MIT (the package code). |
| `G/test/test_locomotion.py` | 1046 | ~45 unit tests (config schema, slot mapping, obs vs `homie_policy`, command filter, waiting/readiness, heartbeat, ramp, confirm, start block, band reset, faults, tick grid/wrap, latency, `SourceClock`). |
| `G/test/test_locomotion_dds.py` | 748 | DDS integration on domain 97 with a fake G1, fake arms and the installed `golem_safety` `safety_node` (G1 `split.yaml` copy) as a subprocess: start-up, faults, backlog/cadence, latency, heartbeat loss. |
| `G/test/test_locomotion_node.py` | 298 | Node-level tests with a `FakeDriver` (topics reach the driver, `use_sim_time` ignored, confirm, band client, failed/`enabled` toggle replies). |
| `G/test/test_band.py` | 79 | `BandRelease` unit tests (7). |
| `G/test/conftest.py` | 10 | `gc.disable()` (cyclonedds-python channel-crash workaround). |
| `G/setup.py`, `G/package.xml`, `G/setup.cfg`, `G/.gitignore` | 33 / 23 / 4 / 4 | MIT; depend rclpy, geometry/std msgs, std_srvs; exec_depend golem_safety (tests only), numpy, yaml. **No onnxruntime, unitree_sdk2py or cyclonedds** declared. `policy/` is **not installed**: the node reads it from a bind mount. |
| `G/README.md` | 390 | Interfaces, behaviour, real-time findings, config table, tests. |

Stale spot found: `G/README.md:299` gives the sim `startup.tick_rate` as `[0.25, 1.25]`;
`G/config/sim.yaml:40` has `[0.1, 1.25]`. `G/README.md:95-98` calls the G1's linear ramp "the
H1-2's `prepose_ramp_s`" (the H1-2's is a cosine blend plus a settle check).

### 1.3 The ROS nodes

**H1-2: `lowerbody_controller_node`** (`H/.../lowerbody_controller_node.py`)

| Kind | Name | Type | QoS / rate | Where |
|---|---|---|---|---|
| sub | `/lowstate` | `unitree_hg/LowState` | best effort, keep-last 1 | `:233-238` |
| sub | `/cmd_vel` | `geometry_msgs/Twist` | depth 10 (reliable) | `:239` |
| sub | `/lowerbody/squat_cmd` | `std_msgs/Float32` | depth 10 | `:240` |
| sub | `band_release_topic` (`/left_ee_pose`) | `geometry_msgs/PoseStamped` | depth 10 (never destroyed) | `:241-243` |
| sub | `/safety/heartbeat` | `std_msgs/Bool` | depth 10 | `:244-245` |
| pub | `/safety/lowcmd_lower_in` | `unitree_hg/LowCmd` | depth 10, reliable | `:246` |
| pub | `/lowerbody/active_policy` | `std_msgs/String` | reliable, transient-local, depth 1 (latched) | `:235-237,247` |
| srv | `/lowerbody/confirm_engage` | `std_srvs/Trigger` | | `:248-249` |
| cli | `/elastic_band/toggle` | `std_srvs/Trigger` | | `:225` |
| timer | `_tick` | | `1/control_hz` on the **node clock** (sim time with `use_sim_time:=true`) | `:252` |

Parameters (`:122-177`): `control_hz` 50.0; `active_policy` "fame"; `walk_config`,
`fame_config`, `almi_config` (share paths); `imu_offset_{roll,pitch,yaw}_deg` 0.0 (dynamic,
±15°, slider range, `:137-142`, callback `:307-324`); `default_height_cmd` 1.0;
`prepose_ramp_s` 2.0; `engage_wait_for_confirm` true; `disable_elastic_band` true;
`band_wait_for_frame_task` true; `band_release_topic` "/left_ee_pose"; `band_max_wait` 30.0;
`heartbeat_timeout` 0.5; `max_state_gap` 0.25; `cmd_vel_timeout` 0.5.
Constants in code: `MOTOR_MODE_PR=1`, `MODE_PR=0`, `MODE_MACHINE=6` (`:83-86`),
`CMD_VEL_LIMIT=1.0` (`:90`), `BAND_ON="enabled"` (`:93`), `PREPOSE_TOL=0.15`,
`PREPOSE_SETTLE_TICKS=10`, `PREPOSE_MAX_TICKS=400`, `CONFIRM_REMINDER_SECONDS=5.0`
(`:105-109`). Topic names are all literals in code (only the band topic is a param).

Transport: rclpy, single-threaded executor (`rclpy.spin`, `:485`): `/lowstate` at 500 Hz, the
policy tick and every service share one thread, hence `max_state_gap` 0.25 s on the real robot
(commit `0064221`).

Output message (`_publish_leg`, `:330-344`): `mode_pr=0`, `mode_machine=6`; slots 0-11:
`mode=1`, `q=target`, `dq=0`, `tau=0`, `kp`, `kd` from the policy YAML; slots 12-34 default
(zero). No CRC (golem_safety writes `/lowcmd`). State read (`_read_state`, `:346-354`): slots
0-26 `q`, `dq` (all 27 H1-2 body joints, `/lowstate` order), `imu_state.quaternion` (w first,
then the IMU trim), `imu_state.gyroscope`.

**G1: `locomotion_node`** (`G/g1_homie/locomotion_node.py` + `driver.py`)

| Kind | Name | Type | QoS / rate | Where |
|---|---|---|---|---|
| ROS sub | `/cmd_vel` | `Twist` | depth 10 | `locomotion_node.py:85` |
| ROS sub | `/locomotion/height` | `Float32` | depth 10 | `:86` |
| ROS sub | `/safety/heartbeat` | `Bool` | depth 10 | `:87` |
| ROS srv | `/lowerbody/confirm_engage` | `Trigger` | | `:88` |
| ROS sub (band only) | `/left_ee_pose` | `PoseStamped` | depth 10, destroyed after the first | `:95-96, 129-133` |
| ROS cli (band only) | `/elastic_band/toggle` | `Trigger` | | `:97` |
| ROS timer (band only) | `_on_band_timer` | | 10 Hz (`BAND_PERIOD`) wall | `:21, 98` |
| DDS reader | `rt/lowstate` (`topics.low_state`) | `LowState_` | keep-last 1, best effort, plain cyclonedds reader, no listener | `driver.py:29, 80-84` |
| DDS reader | `rt/safety/lowcmd_upper_in` (`topics.upper_cmd`) | `LowCmd_` | same; detached once walking/faulted | `driver.py:81-88, 158-164` |
| DDS writer | `rt/safety/lowcmd_lower_in` (`topics.low_cmd`) | `LowCmd_` | Unitree SDK `ChannelPublisher` | `driver.py:77-78` |
| thread | worker `locomotion` | | waits on a WaitSet, 10 ms watchdog | `driver.py:26, 89-90, 142-153` |

Parameters: `policy_dir` (default `/home/code/g1_homie`, `:19, 62`), `use_sim_time` forced
false (`:60-61`); the run config is a CLI argument `--config g1.yaml|sim.yaml|path`
(`:180-184`), not a ROS parameter. Env: `ROS_DOMAIN_ID` overrides `network.domain_id`,
`G1_DDS_INTERFACE` overrides `network.interface` (`config.py:184-199`).

Output message (`driver.py:66-78, 175-180`): `LowCmd_` default, `mode_pr=0` (mode_machine left
0; golem_safety restamps 5), slots 0-11 `mode=1`, `q`, `kp`/`kd` from HOMIE's substring gains,
`dq=tau=0`; every other slot zero; CRC computed. Written only if `latency_ok` (`:179-182`).
State read: `read_state` (`locomotion.py:119-127`): `quat_wxyz`, `gyro` from `imu_state`; `q`,
`dq` at the 27 policy slots (0-12, 15-28), float64.

---

## 2. Policies

Common to all: 12 leg actions, slots 0-11 of the lower-body input (golem_safety's split
input), `target_q = action * action_scale + default_leg_angles`, `dq = tau = 0`, PD in the
motor (`mode = 1`), no torque limit in the node (golem_safety clips to `limits.clip`). 50 Hz.

Robot joint orders (`core_ws/src/golem_safety/robots/{h12,g1}.yaml`): the H1-2's hips are
**yaw, pitch, roll** (slots 0-2 / 6-8), torso 12, arms 13-26; the G1's hips are **pitch, roll,
yaw**, waist yaw/roll/pitch 12-14, arms 15-28. Every policy's "leg" order equals its robot's
slot order; joint naming must therefore go through the robot table, not an index convention.

### 2.1 H1-2 WALK — `WalkPolicy` (`H/.../policy.py:224-286`, `H/policies/walk/walk.yaml`)

- Format/runtime: TorchScript MLP, `torch.jit.load` (`:247`), default torch threads.
- Observation, 47-d, one step, no history (`:263-280`), float32:

| idx | term | detail | where defined |
|---|---|---|---|
| 0:3 | gyro × `ang_vel_scale` (0.25) | `imu_state.gyroscope`, raw IMU frame (torso IMU on the H1-2, not verified as pelvis) | scale yaml `:19`; layout code |
| 3:6 | projected gravity | `gravity_from_quat_walk` closed form (`:69-79`) after the IMU trim | code |
| 6:9 | cmd × `cmd_scale` [2, 2, 0.25] | /cmd_vel clipped ±1 by the node (`:284` of the node), zero after `cmd_vel_timeout` | yaml `:23` |
| 9:21 | (q[0:12] − default) × `dof_pos_scale` 1.0 | legs, slot order | yaml `:16-17,20` |
| 21:33 | dq[0:12] × `dof_vel_scale` 0.05 | | yaml |
| 33:45 | previous raw action (12) | | code |
| 45:47 | sin, cos of the gait phase | period `GAIT_PERIOD=0.8` s (`:226`), phase from `state.t - t_start`, `t` = node clock (sim time in sim), `t_start` = first compute / reset | code |

- Action: `target = a*0.25 + default` (`:284`), then clipped to
  `legs_motor_pos_{lower,upper}_limit_list` (`walk.yaml:7-10`, `:285`). kp
  `[200,200,200,300,40,40]×2`, kd `[2.5,2.5,2.5,4,2,2]×2` (yaml `:4-5`).
- Default/nominal: `[0, -0.16, 0, 0.36, -0.2, 0]×2` (yaml `:14-15`).
- Reset: action zeroed, `t_start = state.t` (`:255-257`).
- `num_obs`, `num_actions` in yaml (`:24-25`) but layout is code.

### 2.2 H1-2 ALMI — `AlmiPolicy` (`H/.../policy.py:292-435`, `H/policies/almi/almi.yaml`)

- Format/runtime: TorchScript `PolicyExporterLSTM` (legged_gym export) with hidden/cell state as
  module buffers; `reset_memory()` method (`:367, 386`). Inference cadence matters (trained 50 Hz).
- Observation, 65-d, one step (recurrence inside the module) (`:409-429`):

| idx | term | detail |
|---|---|---|
| 0:3 | gyro × 0.25 | |
| 3:6 | projected gravity | `gravity_from_quat_walk` |
| 6:9 | cmd_eff × [2, 2, 0.25] | `cmd_eff = EMA_α(clip(cmd × cmd_gain, ±1))`, `cmd_gain` [1.5, 1.0, 2.0], α 0.15 (`almi.yaml:42-44`, `_effective_cmd :376-381`); the filter is reseeded on reset (`:389`) |
| 9:30 | (q21 − default21) × 1.0 | `OBS_JOINT_IDX = np.r_[0:17, 20:24]` (`:328`): legs 0-11, torso 12, L shoulder p/r/y + elbow 13-16, R shoulder p/r/y + elbow 20-23; wrists unobserved |
| 30:51 | dq21 × 0.05 | |
| 51:63 | previous raw action (12) | |
| 63:65 | left_sin, right_sin | period 0.8 s, right = phase + 0.5 (`:320-321, 419-421`); **both 0 when ‖cmd_eff‖ < 0.1** (`STAND_CMD_EPS`, `:416-417`) |

- Command onset: `ONSET_RISE=0.1` rising (from not walking) calls `reset()` (LSTM memory zeroed,
  action zero, t_start, filter reseeded to the raw clipped command) and uses that full command on
  the same tick; `ONSET_FALL=0.05` re-arms (`:323-324, 396-407`). Hard-coded.
- Action: `a*0.25 + default21[:12]`, clipped to the position limits (`almi.yaml:16-19`). kp/kd as
  WALK (`:11-12`). Default 21 (`:26-30`): legs `[0,-0.4,0,0.8,-0.4,0]×2`, torso 0, arms
  `[0.4,0.2,0,0.3]`, `[0.4,-0.2,0,0.3]`.
- Config holds scales, gains, defaults, gain/α; code holds layout, joint index set, gait/onset
  constants.

### 2.3 H1-2 FAME (RMA) — `FamePolicy` (`H/.../policy.py:441-575`, `H/policies/fame/fame.yaml`, `H/h12_lowerbody_rl/rma/`)

- Format/runtime: actor TorchScript (`:487`); encoder `EnvFactorEncoder` rebuilt in Python and
  loaded from a `state_dict` with `torch.load(weights_only=True)` (`:490-499`); if the encoder
  file is missing z_t = 0 (node warns, `lowerbody_controller_node.py:203-207`).
- Single observation, 76-d (`_single_obs :512-527`):

| idx | term | detail |
|---|---|---|
| 0:3 | cmd × [2, 2, 0.25] | cmd clipped ±1 by the node, no gain/filter |
| 3 | height_cmd (unscaled) | `/lowerbody/squat_cmd`, default `default_height_cmd` 1.0 (node param), reset to default at engage; not clipped |
| 4:7 | gyro × 0.25 | |
| 7:10 | projected gravity | `gravity_from_quat_fame` = `quat_rotate_inverse(q, [0,0,-1])` (`:82-103`) |
| 10:37 | (q27 − padded defaults) × 1.0 | all 27 slots; defaults = legs `[0,-0.16,0,0.36,-0.2,0]×2` + `default_angles_arms` (15 zeros) (`fame.yaml:29-34`, `:483-485`) |
| 37:64 | dq27 × 0.05 | |
| 64:76 | previous raw action (12) | |

- Actor input 252 = proprio history 3 × 76 (deque, oldest first; zero-filled at reset, then
  the newest appended each step) + z history 3 × 8 flipped to oldest first (`:554-569`).
- z_t = encoder(e_t), e_t (21) = raw q of slots 12-26 (torso + both arms, unscaled) + left
  force (3) + right force (3) from yaml (zeros) (`:529-538`). `LATENT_DIM=8`, `Z_HISTORY=3`,
  `ET_DIM=21` and the encoder's hidden dims (256, 128, ELU) are code.
- Action: `a*0.25 + default12`, clipped (`fame.yaml:22-25`); kp
  `[200,200,200,300,60,40]×2`, kd `[6,6,6,7.5,1,0.3]×2` (`:16-19`).
- Note: FAME's per-step layout is the **same term list as HOMIE's** (cmd, height, gyro,
  gravity, q, dq, last action) with different scales and history length, plus an appended
  latent group.
- `hand_force_provider` hook (`:457, 480, 533-534`) is never passed by the node (dead hook).

### 2.4 G1 HOMIE — `HomiePolicy` (`G/policy/homie_policy.py`, `G/policy/g1.yaml`)

- Format/runtime: ONNX via onnxruntime, session with 1 intra- and 1 inter-op thread, no
  spinning, CPU EP (`homie_policy.py:78-98`); loaded from `policy_dir` by importing
  `homie_policy.py` with `importlib` (`locomotion.py:74-97`); one warm-up step then `reset()`
  at load (`:92-96`).
- Single observation, 76-d (`observation :153-166`), state in float64 then cast to float32:

| idx | term | detail |
|---|---|---|
| 0:3 | cmd × `cmd_scale` [2, 2, 0.5] | after `CommandFilter`: clip to `cmd_range` (trained) then to the run config's limits, then rate limit by `commands.accel` × 0.02 (`locomotion.py:168-191`); zero when /cmd_vel older than `commands.timeout` |
| 3 | height (m, unscaled) | `/locomotion/height`, default `height_cmd` 0.74 (`g1.yaml:58`), clipped to `height_range` [0.24, 0.74] and rate-limited by `commands.height_rate` |
| 4:7 | gyro × 0.5 | pelvis IMU assumed (`config.py:25`, `imu_frame: pelvis`) |
| 7:10 | projected gravity | `quat_rotate_inverse(q, [0,0,-1])` (`:48-75`) |
| 10:37 | (q27 − default_q) × 1.0 | `policy_joints` by name (`g1.yaml:15-25`): legs, waist yaw, 14 arm joints (waist roll/pitch dropped); default_q = legs `[-0.1,0,0,0.3,-0.2,0]×2`, 0 elsewhere |
| 37:64 | dq27 × 0.05 | |
| 64:76 | previous action (12) | the **clipped** raw action (±`clip_actions` 100) |

- Input 456 = history 6 × 76 (deque, oldest first, zero-filled at reset) (`:146-151, 180-181`);
  input size checked against the ONNX input at load (`:124-128`); the observation is checked
  finite before it enters the history (`:178-185`).
- Action: `clip(a, ±100) * 0.25 + default_angles` (`:189-190`); no position clip.
- Gains: substring match in order (`joint_gains :101-106`, `g1.yaml:36-39`): hips 100/2,
  knee 150/4, ankle 40/2.
- Rate: `POLICY_HZ=50` (`locomotion.py:34`), stepped on a 20 ms grid of `rt/lowstate`'s tick
  (`tick_seconds` 0.001, `_step_at :442-455`).
- Config (licensed `g1.yaml`) holds joints, history length, defaults, gains, scales, clip,
  height default, ranges; layout is code in the licensed `homie_policy.py`; slots
  (`POLICY_SLOTS`, `BODY_JOINT_NAMES`, `locomotion.py:16-40`) are a hand copy of
  `golem_safety/robots/g1.yaml`.

### 2.5 Where each piece lives today

| Piece | WALK | ALMI | FAME | HOMIE |
|---|---|---|---|---|
| obs term order | code | code | code | code (licensed) |
| scales | yaml | yaml | yaml | yaml (licensed) |
| observed joints | code (`[:12]`) | code (`OBS_JOINT_IDX`) | code (all 27) | yaml names (licensed) + code slot check |
| defaults | yaml | yaml | yaml (+ arms) | yaml (legs; rest 0 in code) |
| history | none | in model (LSTM) | code len from yaml (3) + z code (3) | yaml (6) |
| gravity formula | code (closed form) | code (closed form) | code (rotate inverse) | code (rotate inverse) |
| gait phase | code (0.8 s, sin/cos) | code (0.8 s, L/R sin, stand gate) | — | — |
| cmd processing | node ±1 | node ±1 + yaml gain/α + code onset | node ±1 | yaml ranges + run-config limits/accel |
| height | — | — | node param 1.0, topic squat_cmd | yaml 0.74, range, rate; topic /locomotion/height |
| action clip | yaml pos limits | yaml pos limits | yaml pos limits | yaml raw ±100 |
| gains | yaml per joint | yaml per joint | yaml per joint | yaml substring |
| rate | node param 50 Hz, node clock | same | same | code 50 Hz on tick |

Gravity formulas: the closed form (`policy.py:69-79`, WALK/ALMI) and `quat_rotate_inverse`
(FAME/HOMIE) are identical for a unit quaternion; they differ for a non-unit one, which the
supervisors admit up to ‖q‖ = 1 ± 0.05. A bit-exact merge keeps both as named variants (or
proves the difference irrelevant). `gravity_z` in both supervisors is a third copy
(`supervisor.py:53-56`, `locomotion.py:162-165`).

---

## 3. Supervisors / state machines

`LegSupervisor` (`H/.../supervisor.py`) + node vs `LocomotionController` (`G/.../locomotion.py:194-657`)
+ driver + node. Same phases with one rename (`running` vs `walking`).

| Aspect | H1-2 | G1 | Real robot difference or divergence? |
|---|---|---|---|
| Waiting: finite, ‖q‖ within 0.05, gravity z < −0.9, first true beat | `supervisor.py:224-238` | `locomotion.py:510-530` | identical; unify (thresholds identical constants in both files) |
| Wait-reason logging (kind change, ≤ every 5 s) | `:240-247` | `:585-596` | identical; unify |
| Ignore a sample whose tick did not advance; fault on a tick going back | no | `:403-417` | divergence (needs the tick-grid timebase) |
| Policy clock | ROS timer 50 Hz on node clock (sim time in sim) (`node :252`); supervisor times are monotonic | 20 ms grid of `rt/lowstate` tick; monotonic for gaps; `use_sim_time` forced false | divergence; the tick grid works on both (both robots' `tick` are ms counters per `golem_upper_ik/core/ros_io.py:47`, the H1-2 sim's is ms of sim time per CLAUDE.md §8) |
| Ramp shape | cosine blend from measured over `prepose_ramp_s` (2 s), ticks at `control_hz` (`node :454-465`) | linear over `ramp_seconds` (2 s) × 50 steps (`:598-614`) | divergence; the cosine came from a real-robot hip-yaw dq estop on the H1-2 (`node :95-104`) |
| End of ramp | settle check: max err < 0.15 rad for 10 ticks, or `PREPOSE_MAX_TICKS` 400 after the ramp; "NOT SETTLED" posed anyway (`node :466-478`) | straight to HOLDING after the last ramp step | divergence |
| Holding output | nominal command each tick, reminder every 5 s (`node :429-437`) | default pose each step (`:439`) | divergence (logging only) |
| Policy-start readiness | none | steady 1 s of tick (`STEADY_SECONDS`), no gap > `max_state_gap`, sample age < `max_latency/2`, tick rate within `startup.tick_rate`, arm controller commanding (`upper_cmd` within 0.5 s), no start block (`_track_ready :532-583`) | G1-only, born from CPU-starvation findings (`G/README.md` "Real time"); candidate to unify as config |
| Confirm | accepted in HOLDING (`supervisor.py:159-165`) | accepted in HOLDING **and** ready (`:312-324`) | divergence |
| Without confirm (sim) | runs right after posed (`supervisor.py:152-157`) | runs once ready (`:436-438`) | divergence |
| Engage | zero cmd, default height, `PolicyManager.commit` → `reset` (`node :439-443`) | zero cmd, standing height, `policy.reset()` (`:616-628`) | same intent; unify |
| Band: when the policy may start | at once (policy runs under the band until `/left_ee_pose`, ~24-29 s in the sim per `lowerbody_sim.yaml:14-18`) | blocked until `/left_ee_pose` or `band.max_wait` after reaching the pose (`block_start`, node `:82-84`, `:557-564`) because HOMIE stumbles after > ~1.5 s under the band (`G/config/sim.yaml:47-53`) | policy-specific today; make it config (and decide whether ALMI should also wait) |
| Band: when the toggle is called | in `_tick` at 50 Hz, once engaged and (`/left_ee_pose` or `band_max_wait` since **node start**) (`node :364-383`) | 10 Hz timer, once walking and (`/left_ee_pose` or `band.max_wait` since **policy start**) (`band.py:31-51`, node `:129-143`) | divergence; unify on `BandRelease` |
| Band reply handling | one call, failure logged, "enabled" warns, success → reset next tick (`node :385-409`) | identical texts; success → `policy.reset()` immediately (`node :145-171`, `:326-337`) | unify (test `test_band_node.py` asserts reset on next tick) |
| /left_ee_pose sub | kept for life | destroyed after the first (`node :129-133`) | divergence (CPU: 100 Hz topic) |
| cmd_vel | clipped ±1 at receive, non-finite → zero + warn, zero after `cmd_vel_timeout` (`node :283-285`, `supervisor.py:167-179`) | non-finite → zero + warn (`:355-361`), zero after `commands.timeout`, then `CommandFilter` (trained range, limits, accel) (`:630-633`) | the per-policy processing is policy config; the timeout/finite checks are identical |
| Height | `/lowerbody/squat_cmd`, non-finite ignored, no clip, no rate limit | `/locomotion/height`, non-finite ignored, clipped to `height_range`, rate-limited | divergence in topic name (user decision) and processing (policy config) |
| Faults once commanded | gap > `max_state_gap`, non-finite, bad quat, fall (gz > −0.5), policy raise / non-finite / wrong shape, unexpected tick error | same + tick went back + latency > `max_latency` while walking (`:457-472`) | latency needs the DDS source timestamp; see §6 |
| Heartbeat | first true beat starts; false or silent > 0.5 s latches, every phase (`supervisor.py:100-108, 131-140`) | identical (`:298-310, 370-383`) | identical |
| Stale check cadence | every 50 Hz tick | every worker wake (≤ 10 ms) | divergence |
| Fault message | "golem_safety keeps relaying..." (`:256-262`) | same plus the estop cases (`:643-652`) | text only |
| `max_state_gap` | real 0.25, sim 0.5 | real 0.05, sim 0.5 | real values differ because of the executor design, not the robot |
| IMU trim | live `imu_offset_*` params (real 0.5/−2.5/0) | none; `imu_frame` must be `pelvis` | robot difference (H1-2 IMU bias measured); config for both |
| Active policy topic | `/lowerbody/active_policy` latched | none | unify (tests/stack reads it) |
| Timing log | none | latency/step spacing every 10 s (`:474-493`) | unify (cheap) |
| Domain 0 guard | none (golem_safety refuses sim on 0) | refuses a non-0 config on domain 0 (`config.py:190-194`) | tied to the SDK transport |
| Config style | ROS params with defaults, bringup YAML | strict YAML, every key required, unknown key error (`config.py:11-34`) | the golem pattern is strict (golem_upper_ik `robot.yaml`) |
| Policies loaded | all three at start (`node :198-202`) | one | divergence (start-up time, torch memory) |

Real robot differences that must become config: joint names/slots, `mode_machine`
(from `golem_safety/robots/<robot>.yaml`), IMU trim and IMU frame, sim vs real
(`wait_for_confirm`, band, `max_state_gap`, tick-rate window, command limits). Everything else
in the table is accidental divergence or a policy property.

---

## 4. Configs and who reads them

### 4.1 `core_ws/src/golem_bringup/config/h12/lowerbody_sim.yaml` (node params, key `lowerbody_controller_node`)
`engage_wait_for_confirm: false`, `imu_offset_{roll,pitch,yaw}_deg: 0.0`, `band_max_wait: 34.0`,
`max_state_gap: 0.5`, `cmd_vel_timeout: 0.5`. Read by the node through
`h12_lowerbody.launch.py:54-56`; included by `h12_sim.launch.py:60-64` and
`h12_sim_mac.launch.py:73-78`; asserted by `golem_bringup/test/test_mac_launch.py:77`.

### 4.2 `.../config/h12/lowerbody_real.yaml`
`disable_elastic_band: false`, `imu_offset_roll_deg: 0.50`, `imu_offset_pitch_deg: -2.5`,
`imu_offset_yaw_deg: 0.0`, `max_state_gap: 0.25`, `cmd_vel_timeout: 0.5`. Default of
`lowerbody_params` (`h12_lowerbody.launch.py:33-36`), passed by `h12_real_robot.launch.py:57-62`.

Params never set by any bringup (defaults only): `control_hz`, `walk_config`, `fame_config`,
`almi_config`, `default_height_cmd`, `prepose_ramp_s`, `band_wait_for_frame_task`,
`band_release_topic`, `heartbeat_timeout`. `active_policy` comes from `lowerbody:=`
(`h12_lowerbody.launch.py:56`). `tests/stack/minimal_stack.launch.py:33-40` sets its own copy
(almi, no confirm, zero trims, gap 0.5, timeout 0.5).

### 4.3 H1-2 policy YAMLs (`H/policies/*/*.yaml`), read by the policy classes
- walk: `policy_path`, `kps`, `kds`, `legs_motor_pos_{lower,upper}_limit_list`,
  `default_angles`, `ang_vel_scale`, `dof_pos_scale`, `dof_vel_scale`, `action_scale`,
  `cmd_scale`, `num_actions`, `num_obs`.
- almi: walk's keys + `cmd_gain`, `cmd_smooth_alpha` (21-entry `default_angles`).
- fame: walk's keys + `encoder_path`, `default_angles_arms`, `obs_history_len`,
  `left_hand_force`, `right_hand_force`.
- `num_obs` is checked only by FAME (`policy.py:568-569`); WALK/ALMI just size a buffer.

### 4.4 `G/config/{sim,g1}.yaml`, read by `G/g1_homie/config.py:check_config` (strict)
`topics.{low_state, low_cmd, upper_cmd}`; `network.{domain_id (1|0), interface}`; `imu_frame`;
`tick_seconds` (0.001); `startup.{ramp_seconds 2.0, wait_for_confirm false|true, tick_rate
[0.1,1.25]|[0.9,1.1]}`; `heartbeat.timeout 0.5`; `band.{release true|false, max_wait 30}`;
`commands.{vx [-0.8,1.2]|[-0.3,0.5], vy [-0.5,0.5]|[-0.3,0.3], yaw [-0.8,0.8]|[-0.6,0.6],
accel [0.5,0.5,1.0], timeout 0.5, height_rate 0.1}`; `faults.{max_state_gap 0.5|0.05,
max_latency 0.025}`. Checks: `low_cmd` must start `rt/safety/`, differ from `upper_cmd`;
limits include 0 and lie inside `cmd_range`; `tick_seconds` divides 20 ms; `tick_rate` brackets
1; heartbeat timeout > 0.1 s. Also read by `tests/g1/test_sim_bringup.py:433-438` (it parses
the `Loaded <path>` log line and reads `band.max_wait`) and by `G/test/*`.
`g1_core.launch.py:110-130` refuses `sim` on the real robot and `g1` in the sim.

### 4.5 `G/policy/g1.yaml` (licensed), read by `homie_policy.load_config`
`policy_path`, `xml_path` (both resolved relative, with a repo-root fallback,
`homie_policy.py:25-45`), `simulation_duration`, `simulation_dt`, `control_decimation`
(MuJoCo script only), `policy_joints`, `num_actions`, `obs_history_len`, `default_angles`,
`stiffness`, `damping`, `joint_damping`/`armature`/`frictionloss` (MuJoCo only),
`ang_vel_scale`, `dof_pos_scale`, `dof_vel_scale`, `action_scale`, `cmd_scale`, `clip_actions`,
`cmd_init` (script), `height_cmd`, `cmd_range`, `height_range`. `cmd_range` is also passed into
the run-config check (`locomotion_node.py:66`).

---

## 5. Everything outside the two packages that names them

### 5.1 Bringup (`core_ws/src/golem_bringup/`)
- `package.xml:17` exec_depend `h12_lowerbody_rl`; `:21` exec_depend `g1_homie`.
- `launch/h12/h12_lowerbody.launch.py:12-15, 33-36, 50-64` (package, executable, node name
  "MUST match the yaml key", `lowerbody`, `lowerbody_params`, `start_position_verified`, `mjpc`
  exclusion; also the MJPC estimator, which stays in bringup).
- `launch/h12/h12_sim.launch.py:38-41, 60-64`; `h12_sim_mac.launch.py:5, 9, 48-51, 73-78`
  (`POLICIES` list, `GOLEM_LOWERBODY`); `h12_real_robot.launch.py:14-21, 46-47, 57-62`.
- `launch/g1/g1_core.launch.py:8, 36-52, 109-130, 156-161, 212-227, 272-276`
  (`LOCOMOTION_CONFIGS`, `locomotion_refusal`, `locomotion_config`, `locomotion_delay`, the
  `Node(package='g1_homie', executable='locomotion_node', arguments=['--config', ...])`);
  `g1_sim.launch.py:13`; `g1_real_robot.launch.py:5, 30-39, 115, 131`.
- `config/h12/lowerbody_{sim,real}.yaml` (move into the package).
- Tests: `test/test_g1_launch.py:4, 176-215` (asserts `locomotion_node` arguments `--config
  sim.yaml|g1.yaml` and delays); `test/test_nav_launch.py:136, 184, 193`;
  `test/test_mac_launch.py:5, 54-81` (`h12_lowerbody.launch.py` include and its arguments).
- `common_cmds.md:44, 82` (log line `legs pre-posed (...) — holding`, confirm command).

### 5.2 Docker
- `docker/docker-compose.yml:250-257`: bind mount `../core_ws/src/g1_homie/policy` →
  `/home/code/g1_homie` (read-only, ros service).
- `docker/scripts/docker_common.sh:169-189`: `require_g1_policy` checks `deploy.onnx`,
  `g1.yaml`, `homie_policy.py` under `core_ws/src/g1_homie/policy` for every x86 ros container.
- `docker/RosDockerfile:600-611`: onnxruntime 1.23.2 installed "for g1_homie" (x86 ros only;
  torch 2.12.1+cu130 is in the same image).
- `docker/mac/scripts/launch_ros_mac.sh:116-134`: `PKGS` includes `h12_lowerbody_rl`;
  `BRINGUP_STUB_DEPS` includes `g1_homie`. The Mac image has torch (base) but **no
  onnxruntime**; its ROS uses FastDDS.
- `docker/mac/docker-compose.yml:138-145` (comment about `h12_lowerbody`/`lowerbody_controller_node`).
- `docker/mac/RosDockerfile.slim.arm64:127-134` (comment: unitree_hg for the lower-body stack).
- `docker/BUILD.md:42-43, 280`.
- `docker/scripts/launch_ros.sh:97`: exits on a stale install with no source (a rename
  triggers it on every host that built the workspace).

### 5.3 Tests outside the packages
- `tests/stack/launch_ros_stack.sh:32` (`--packages-up-to ... h12_lowerbody_rl`),
  `tests/stack/minimal_stack.launch.py:32-40` (package, executable, node name, params),
  `tests/stack/check_lowerbody.py:72, 137-140` (`/lowerbody/active_policy`,
  `/lowerbody/confirm_engage`), `tests/stack/run_checker.sh:8` (`--expected-walk-policy almi`),
  `tests/stack/README.md:35`.
- `tests/g1/test_sim_bringup.py:109-121, 425-438, 834-890` (logger `locomotion_node`; log
  lines `Loaded (\S+\.yaml) and the policy`, `walking: policy started`, `releasing the elastic
  band: `, `the policy runs and frame_task_server is up (/left_ee_pose)`, `elastic band
  released`; reads the config's `band.max_wait`); `tests/g1/test_sim_nav.py:706-709, 791`
  (`walking: policy started`, process name `locomotion_node` for CPU); `tests/g1/README.md:27,
  99-102`.
- `tests/sim/test_sim_stack.py`: no names, but checks `safety_node` is the only `/lowcmd`
  writer and walking behaviour for both robots (closed-loop).
- `tests/docker/test_workflows.py:249-254, 499-517, 668-700` (G1 policy files and the mount).
- `tests/docker/test_shared_packages.py:16-26` (SCANNED; the new package must be added) and
  `:141` (a literal `"g1_homie"` in a scanner test string; harmless).
- `tests/robocasa/g1/test_g1_band.py:128-140` (`test_sim_reads_no_policy`: the sim touches
  nothing of HOMIE's policy dir).

### 5.4 Other packages
- `core_ws/src/lmg_deploy/lmg_deploy/marl_deploy_node.py:36, 41, 78`,
  `lmg_deploy/rotations.py:64-65`, `lmg_deploy/test/test_imu_trim.py:46`, `lmg_deploy/README.md:13`
  (comments citing `h12_lowerbody_rl`'s IMU trim and heartbeat gate; no import).
- `core_ws/src/golem_safety/README.md:55-56`; `core_ws/src/golem_upper_ik/config/g1/
  {g1_split,sim_split}.yaml:2` (comments).
- `golem_robocasa/`: no reference (the sims read no policy file; `test_sim_reads_no_policy`
  enforces it for the G1).
- `tools/`: no reference.

### 5.5 Docs
`README.md:49, 72-83`; `CLAUDE.md` and `AGENTS.md` (§1, §2, §4 rows `h12_lowerbody_rl` and the
G1 table `g1_homie`, §6 leg-node start-up, §8 "HOMIE's policy is non-commercial" (`:688-692`),
§9, §10 stale-install list); `docs/RUNNING.md:191, 238`; `docs/NAVIGATION_DEMO.md:96`;
`docs/g1/{index.md:4,142-143; packages.md:3,17,37,46-49,72; walking.md:3,7,51,74-76;
first-session.md:123,388-392; control-path.md:13,26,37,76; dds.md:6; safety.md:6,80}`;
historical specs/plans under `docs/superpowers/` (64 hits in
`specs/2026-10-27-golem-upper-ik-and-safety/redundancy-audit.md`; the shared-stack design
`specs/2026-10-31-shared-robot-stack-design.md:23` says "legs stay separate", superseded).

### 5.6 Licence constraints (CLAUDE.md §8, `:686-692`)
"Keep its `LICENSE` and `README.md` beside the files, and keep `homie_policy.py` inside
`policy/`." The policy files are CC BY-NC 4.0 by `LICENSE`, treated as CC BY-NC-SA 4.0; the
G1 package code is MIT; the H1-2 package declares no licence (`TODO`); ALMI weights are BSD-3;
FAME and WALK weights have no recorded licence.

---

## 6. Hard parts and risks

1. **No golden coverage of the H1-2 policies.** WALK/ALMI/FAME observation and action math has
   no unit test. A config-driven rewrite can silently reorder a term, drop the ALMI stand gate
   or flip the FAME z-history order, and only a closed-loop run (or a fall) would show it.
   Record golden traces from the current classes first (scripted states/commands incl. reset,
   band reset, ALMI onset rise/fall, FAME warm-up, non-unit quaternion) and require the new
   code to reproduce them exactly. HOMIE already has `test_observation_from_lowstate_matches_homie_policy`
   (`G/test/test_locomotion.py:321`) and `policy/test_homie_policy.py`.
2. **Two runtimes.** TorchScript (3 H1-2 policies, ALMI with in-module LSTM state and a
   `reset_memory()` method, FAME's encoder as a Python `nn.Module` + `state_dict`) and ONNX
   (HOMIE). torch is in both x86 and Mac images; onnxruntime only in the x86 ros image. Backends
   must import lazily (the Mac must never import onnxruntime). Converting the H1-2 policies to
   ONNX is possible (MLPs trivially; the LSTM needs explicit h/c I/O) but changes artefacts.
   torch runs with its default thread pool today (no `set_num_threads(1)`): the same CPU issue
   HOMIE hit with onnxruntime may apply.
3. **Transport and timebase.** g1_homie uses the Unitree SDK + raw cyclonedds readers in a
   worker thread for measured reasons (`G/README.md` "Real time": SDK subscribers decode every
   sample under the GIL; 0.05 s real `max_state_gap`; 25 ms latency fault). `golem_safety` and
   `golem_upper_ik` are ROS-only (`golem_upper_ik/test/test_no_sdk.py`), with keep-last-1
   best-effort subscriptions in their own callback groups (`golem_upper_ik/core/ros_io.py`).
   Moving the G1 onto rclpy: (a) a single-threaded executor would bring back the stalls the
   H1-2 already had to raise `max_state_gap` to 0.25 s for; (b) Humble rclpy does not pass
   `MessageInfo` (source timestamp) to subscription callbacks, so `SourceClock`/`max_latency`
   need either a manual `take` or golem_upper_ik's tick-lag approach (`ros_io.py:65-92`).
   Keeping the SDK path for one robot is a code branch by transport, not by robot, but it
   splits "identical code". Decision needed (below).
4. **Behaviour that must not change while values move.** Every number must land in config
   unchanged: the H1-2's cosine ramp + settle (0.15 rad / 10 ticks / 400 ticks), the 5 s
   reminder, `band_max_wait` 34 s from node start, reset-on-next-tick, `CMD_VEL_LIMIT` ±1, the
   IMU trims (0.50/−2.5/0), `default_height_cmd` 1.0, `max_state_gap` 0.25/0.5; the G1's
   linear ramp, readiness gates, `STEADY_SECONDS` 1.0, `UPPER_FRESH_SECONDS` 0.5, start block,
   band timing from the policy start, tick grid, `max_latency` 0.025, 0.05/0.5 gaps, the
   command filter. Unifying any of these is a behaviour change to make on purpose, after the
   move, with the closed-loop tests.
5. **HOMIE's licence placement.** "Identical code" means the merged node builds HOMIE's
   observation from the shared term registry, not from `homie_policy.py`. Then the licensed
   module is used only by `mujoco_deploy_g1.py`, its own tests and (as the oracle) the golden
   test. Whether values from the licensed `g1.yaml` may be transcribed into an MIT spec file
   (as `almi.yaml` transcribes ALMI's BSD config) or must be read from the licensed file at
   run time is a user call. Installing `policy/` into the package share instead of bind-mounting
   it changes `docker-compose.yml`, `require_g1_policy`, `test_workflows.py` and the CLAUDE.md
   rule's wording.
6. **Robot-name branches the current code would need** and their config form:
   - leg/observed joint sets (`OBS_JOINT_IDX`, `[:12]`, `POLICY_SLOTS`) → joint **names** in the
     policy spec, slots from `golem_safety/robots/<robot>.yaml` (as `golem_upper_ik.profile`);
   - `MODE_MACHINE=6` literal → the robot profile's `mode_machine`/`mode_pr`;
   - `imu_frame: pelvis` check and the H1-2 IMU trim → robot config `imu: {frame, trim_deg}`;
   - FAME's e_t "slots 12-26" → `latent.inputs.joints: [torso_joint, left_shoulder_...]`;
   - HOMIE's "0 for non-leg defaults" → `default` per observed joint in the spec;
   - sim vs real → run configs per robot (`config/<robot>/{sim,real}.yaml`) plus a refusal like
     `locomotion_refusal`/`pairing_refusal`.
   A small term registry (`ang_vel`, `projected_gravity[variant]`, `velocity_command`,
   `height_command`, `joint_pos_rel`, `joint_vel`, `last_action`, `gait_phase[sin_cos |
   left_right_sin, stand_gate]`, `latent[encoder, inputs, history]`) covers all four policies;
   no term names a robot.
7. **Stateful policy behaviours** that are not observation terms: ALMI's LSTM reset on engage,
   on band release and on command onset (hysteresis); WALK's phase origin at reset; FAME's
   zero-filled history; HOMIE's history reset on band release; ALMI's command gain + EMA filter
   vs HOMIE's range clip + accel limit. They need a `reset_on` / `command` section in the spec.
8. **External contracts.** Node names and log lines are read by tests: `locomotion_node` and its
   lines (`tests/g1`), `lowerbody_controller_node` + `/lowerbody/active_policy`
   (`tests/stack`), `golem_bringup` launch tests (argument shapes `--config sim.yaml`), the G1
   test reading `band.max_wait` out of the logged config path. Topic names `/lowerbody/squat_cmd`
   vs `/locomotion/height` are user-visible.
9. **Stale installs.** Renaming `h12_lowerbody_rl` and deleting `g1_homie` makes
   `launch_ros.sh:97` refuse to build on every host until their `build`/`install` dirs (and
   `build_stack_tests`/`install_stack_tests`, `container_cache/g1_tests/{build,install}`) are
   removed; the H1-2 onboard PC keeps answering to the old names. Add both to CLAUDE.md's
   pitfall `rm -rf` list.
10. **The shared-code test.** Adding `golem_lowerbody_rl` to `test_shared_packages.py` SCANNED
    forbids `"h12"`/`"g1"` literals and robot-named code files: e.g. a `policies/h12/` data dir is
    fine (not code), but module names like `homie.py` are fine and `g1_driver.py` is not.
11. **Test coverage per behaviour (today).**

| Behaviour | H1-2 tests | G1 tests |
|---|---|---|
| waiting / heartbeat / upright | `test_supervisor.py` | `test_locomotion.py`, DDS tests |
| faults (gap, non-finite, fall, policy) | `test_supervisor.py` | `test_locomotion.py`, DDS tests (with real `safety_node`) |
| ramp shape / settle | none | `test_ramp_takes_two_seconds`, `test_ramp_is_linear...` |
| readiness gates, tick grid, latency | n/a | `test_locomotion.py`, DDS tests |
| band release | `test_band_node.py` | `test_band.py`, `test_locomotion_node.py` |
| obs/action math | **none** | `test_observation_from_lowstate_matches_homie_policy`, `policy/test_homie_policy.py` |
| ALMI onset / cmd filter | **none** | command filter tested |
| IMU trim | none (lmg_deploy has a mirror test) | n/a |
| closed loop | `tests/stack` (ALMI), `tests/sim --h12` | `tests/g1 --walk`, `test_sim_nav`, `tests/sim --g1` |

---

## 7. Proposed shape (proposals, not decisions)

### 7.1 Package layout

```
core_ws/src/golem_lowerbody_rl/          (git mv of h12_lowerbody_rl; keeps history)
  golem_lowerbody_rl/
    node.py              ROS node (one executable)
    supervisor.py        one state machine: LegSupervisor + LocomotionController features
    profile.py           robot data: golem_safety robots/<robot>.yaml + config/<robot>/robot.yaml
    run_config.py        strict schema of config/<robot>/<run>.yaml (g1_homie/config.py style)
    commands.py          velocity/height processing (timeout, clip, gain, EMA, accel, rate)
    band.py              BandRelease (from g1_homie)
    timing.py            tick grid, tick_delta, SourceClock or tick-lag
    imu.py               trim quaternions, projected gravity variants, gravity_z
    policy/
      spec.py            strict schema of a policy spec
      observation.py     term registry + history groups
      action.py          scale/offset/clip/gains → leg targets
      runtime.py         backends: torchscript, onnx (lazy imports); reset hooks
      encoder.py         FAME's env-factor MLP (from rma/)
  config/
    h12/robot.yaml  h12/sim.yaml  h12/real.yaml
    h12/policies/{almi,walk,fame}.yaml
    g1/robot.yaml   g1/sim.yaml   g1/real.yaml
    g1/policies/homie.yaml
  policies/                              weights (data, not scanned)
    almi/ fame/ walk/                    as today
    homie/ (or keep core_ws/src/.../policy/ with LICENSE+README; see decisions)
  launch/lowerbody.launch.py
  reference/                             the H1-2 standalone scripts, as today
  test/
```

### 7.2 Per-robot config schema (strict: every key required, unknown keys refused)

`config/<robot>/robot.yaml` (robot facts for the lower body):
```yaml
legs: [left_hip_yaw_joint, ..., right_ankle_roll_joint]   # action order = slot order today
imu: {frame: <name>, trim_deg: [roll, pitch, yaw]}       # live-tunable params seeded from here
policies: [almi, walk, fame]                             # the specs in config/<robot>/policies/
default_policy: almi
```

`config/<robot>/<run>.yaml` (sim / real; the G1's current sections, plus the H1-2's knobs):
```yaml
sim: true|false
topics: {low_state: /lowstate, low_cmd: /safety/lowcmd_lower_in, upper_cmd: /safety/lowcmd_upper_in}
timing: {tick_seconds: 0.001}
startup:
  ramp: {seconds: 2.0, shape: cosine|linear, settle: {tolerance: 0.15, ticks: 10, max_ticks: 400} | null}
  wait_for_confirm: bool
  ready: {steady_seconds: 1.0, tick_rate: [lo, hi], wait_for_upper_cmd: bool, upper_fresh_seconds: 0.5} | null
heartbeat: {timeout: 0.5}
band: {release: bool, policy_waits_for_frame_task: bool, max_wait: 30.0, max_wait_from: node_start|policy_start}
commands: {timeout: 0.5, vx: [lo, hi], vy: [...], yaw: [...], accel: [...] | null, height_rate: 0.1 | null}
faults: {max_state_gap: 0.25, max_latency: 0.025 | null}
```
Each H1-2 value is set to today's H1-2 behaviour and each G1 value to today's G1 behaviour, so
the move changes nothing; unification is a later, separate change.

`config/<robot>/policies/<name>.yaml` (how observations and actions are made and interpreted):
```yaml
runtime: {kind: torchscript|onnx, file: ../../../policies/almi/policy_lstm_12800.pt,
          reset: reset_memory|null, threads: 1}
rate_hz: 50
observed_joints: [<names>]           # q/dq order; defaults below are per observed joint
default_pose: {<joint>: rad, ...}    # legs = the pre-pose; others 0 unless given
groups:
  - history: 1|3|6                   # oldest first, zero-filled at reset
    terms:
      - {term: ang_vel, scale: 0.25}
      - {term: projected_gravity, variant: closed_form|rotate_inverse}
      - {term: velocity_command, scale: [2, 2, 0.25]}
      - {term: height_command, scale: 1.0}
      - {term: joint_pos_rel, scale: 1.0}
      - {term: joint_vel, scale: 0.05}
      - {term: last_action}
      - {term: gait_phase, period: 0.8, form: sin_cos|left_right_sin, offset: 0.5, stand_below: 0.1}
  - history: 3                       # FAME only
    order: oldest_first
    terms:
      - {term: latent, encoder: {file: ..., hidden: [256, 128], activation: elu, latent: 8},
         inputs: {joints: [torso_joint, ...], forces: [[0,0,0],[0,0,0]]}}
command:
  trained_range: {vx: [...], vy: [...], yaw: [...]} | null
  clip: 1.0 | null; gain: [1.5, 1.0, 2.0] | null; smooth_alpha: 0.15 | null
  onset_reset: {rise: 0.1, fall: 0.05} | null
  height: {default: 1.0|0.74, range: [lo, hi] | null}
action:
  joints: <robot legs>
  scale: 0.25
  clip_raw: 100 | null
  clip_targets: {lower: [...], upper: [...]} | null
  gains: {kp: [...], kd: [...]}      # or {match: substring, stiffness: {...}, damping: {...}}
reset_on: [engage, band_release]     # + onset via command.onset_reset
```

How each policy maps:
- **WALK**: observed = 12 legs; one group, history 1: ang_vel 0.25, gravity closed_form, cmd
  [2,2,0.25], joint_pos_rel, joint_vel 0.05, last_action, gait_phase sin_cos 0.8;
  command.clip 1; action scale 0.25, clip_targets = `walk.yaml` limits, gains from `walk.yaml`.
- **ALMI**: observed = the 21 names of `OBS_JOINT_IDX`; same terms, gait_phase left_right_sin
  offset 0.5 stand_below 0.1; command clip 1 + gain + EMA 0.15 + onset_reset 0.1/0.05; runtime
  reset `reset_memory`.
- **FAME**: observed = all 27; group 1 history 3: cmd, height, ang_vel, gravity rotate_inverse,
  joint_pos_rel, joint_vel, last_action; group 2 history 3: latent; height default 1.0.
- **HOMIE**: observed = 27 names (no waist roll/pitch); group history 6: cmd [2,2,0.5], height,
  ang_vel 0.5, gravity rotate_inverse, joint_pos_rel, joint_vel, last_action; command
  trained_range = `cmd_range`, height default 0.74 range [0.24, 0.74]; action clip_raw 100, gains
  by substring; runtime onnx threads 1, warm-up step at load.

### 7.3 Node and launch

- One executable (name: user decision), parameters `robot`, `config` (run config name or path),
  `policy` (spec name or path), `use_sim_time`, `imu_trim_*` (live). Robots discovered like
  `golem_upper_ik.profile.known_robots()`.
- `launch/lowerbody.launch.py` with `robot`, `config`, `policy`, `sim`, `use_sim_time`,
  `start_delay`, and an `OpaqueFunction` refusal (the G1's `locomotion_refusal` generalised: a
  run config's `sim` must match `sim:=`; the policy must be listed in the robot's `robot.yaml`).
  `h12_lowerbody.launch.py` keeps `start_position_verified`, the `mjpc` exclusion and the MJPC
  estimator, and includes it; `g1_core.launch.py` includes it inside its existing interlock and
  `TimerAction(controller_delay + locomotion_delay)` (or via `start_delay`).
- Mac: `PKGS` gets `golem_lowerbody_rl`; `GOLEM_LOWERBODY` maps to `policy:=`; stubs drop
  `g1_homie`.

### 7.4 Migration order (proposal)

1. **Golden traces first** (no behaviour change): a test per H1-2 policy that drives the current
   class with a scripted sequence and stores obs vectors + targets; HOMIE's already exists.
   Also scenario traces for both supervisors (phase sequence, outputs per tick).
2. **Rename** `h12_lowerbody_rl` → `golem_lowerbody_rl` (`git mv`, module, entry point,
   `setup.cfg`, `resource/`), and update `golem_bringup/package.xml` + `h12_lowerbody.launch.py`,
   Mac `PKGS`, `tests/stack/{launch_ros_stack.sh,minimal_stack.launch.py}`, docs, CLAUDE/AGENTS
   (incl. the stale-install list). Run tests/stack locomotion.
3. **Move the H1-2's params** from `golem_bringup/config/h12/lowerbody_*.yaml` into
   `config/h12/{sim,real}.yaml` + add `launch/lowerbody.launch.py`; bringup includes it.
4. **Robot profile**: names → slots from `golem_safety/robots/<robot>.yaml`; `mode_machine` from
   the profile; drop `NUM_*`/`OBS_JOINT_IDX` literals.
5. **Spec + term registry + runtimes**: express WALK/ALMI/FAME in YAML; golden traces must
   match; delete `WalkPolicy`/`AlmiPolicy`/`FamePolicy`/`PolicyManager`; load only the active
   policy.
6. **One supervisor** with the G1's features as config (readiness gates, tick grid, latency,
   start block, ramp shape/settle); port `G/test/test_locomotion.py` cases onto it; keep
   `test_supervisor.py`.
7. **Transport decision applied** (rclpy with a dedicated state thread/callback group, or the
   SDK path) and the band unified on `BandRelease`.
8. **HOMIE in**: `config/g1/{robot,sim,real}.yaml`, `config/g1/policies/homie.yaml`, policy files
   placed per the licence decision; `g1_core.launch.py` includes `lowerbody.launch.py robot:=g1`;
   port `test_locomotion_dds.py` (it starts the real `safety_node`), `test_locomotion_node.py`,
   `test_band.py`; keep the log lines `tests/g1` reads (or update the tests in the same commit).
9. **Delete `g1_homie`**; update `docker-compose.yml` mount, `docker_common.sh`
   `require_g1_policy`, `tests/docker/test_workflows.py`, `RosDockerfile` comment, Mac
   `BRINGUP_STUB_DEPS`, `golem_bringup` tests, docs (`docs/g1/*`, `README.md`), CLAUDE.md/AGENTS.md
   §§1, 2, 3, 4, 6, 8, 9, 10.
10. **Guard it**: add `golem_lowerbody_rl` to `tests/docker/test_shared_packages.py` SCANNED (and a
    `test_no_sdk.py` if the transport is ROS-only).
11. **Closed loop**: `tests/stack` (locomotion, robocasa), `tests/sim/test_sim_stack.sh --h12`
    and `--g1`, `tests/g1/test_sim_bringup.sh --walk`, `tests/g1/test_sim_nav.sh`; then, as
    separate deliberate changes, unify the divergent behaviours (ramp shape, band timing,
    readiness gates, `max_state_gap`) one at a time.

---

## 8. Decisions only the user can make

1. **Transport for the G1 leg node**: rclpy like `golem_safety`/`golem_upper_ik` (and accept
   re-measuring the G1's latency/gap budget, replacing `SourceClock` with tick lag), or keep the
   Unitree-SDK/raw-cyclonedds path for both robots, or keep it as a configurable transport.
2. **Runtimes**: keep TorchScript + ONNX backends side by side, or export the H1-2 policies to
   ONNX (one runtime, onnxruntime on the Mac too; ALMI's LSTM state made explicit).
3. **HOMIE licence placement**: reimplement HOMIE's observation in MIT shared code (homie_policy.py
   kept only for its MuJoCo script and as a test oracle) vs load `homie_policy.py` as a plug-in;
   transcribe `g1.yaml` values into an MIT spec vs read the licensed file; install `policy/` into
   the package share vs keep the bind mount at `/home/code/g1_homie`; where the directory lives.
4. **Package licence** for `golem_lowerbody_rl` (g1_homie is MIT; h12_lowerbody_rl says TODO).
5. **Which divergent behaviours to unify, and to which side**: ramp (cosine + settle vs linear),
   G1 readiness gates on the H1-2 (steady timing, tick rate, upper_cmd), latency fault on the
   H1-2, band timing (policy waits for `/left_ee_pose`? `max_wait` from node or policy start),
   confirm while not ready, real `max_state_gap` (0.25 vs 0.05). Or move first and unify later.
6. **Names**: executable/node name (`lowerbody_controller_node` vs `locomotion_node` vs new), the
   height topic (`/lowerbody/squat_cmd` vs `/locomotion/height`), whether the G1 publishes
   `/lowerbody/active_policy`, and the launch argument name (`lowerbody:=`/`policy:=`).
7. **Whether ALMI, WALK and FAME all stay** (every bringup runs ALMI; WALK and FAME are reachable
   only via `lowerbody:=` and the Mac's `GOLEM_LOWERBODY`), and whether `reference/` and
   `rma/env_factor_spec.py` stay.
8. **Robot facts to verify on hardware** before the configs claim them: the H1-2's `/lowstate`
   tick unit (if the tick grid becomes shared), the G1 IMU frame and tick unit (already open in
   `G/README.md:340-345`).
