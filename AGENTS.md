# AGENTS.md — GOLEM developer guide for coding agents

The safety layer and the arm controller of both robots (`golem_safety`,
`golem_upper_ik`), the bringups that start them, the `mode_machine` identity
check and the package tables were checked against the code on 2026-10-28; the
robot selection, the rest of the G1 text, the DDS-domain rules and the H1-2
claims beside them on 2026-10-26; the rest was verified on 2026-07-12. Where
older docs or script comments disagree with the code, the code wins (known
stale spots are flagged inline).

## 1. Project overview

GOLEM (Generalized Open Layered Embodied Modules) is the software stack for two
Unitree humanoids, the H1-2 and the G1 (29 DoF, two Dex3-1 hands). The H1-2 has
MuJoCo/RoboCasa and Isaac Lab simulators and a ROS 2 Humble workspace
(drivers, upper-body IK control, lower-body MPC/RL locomotion, safety layer,
perception model servers, and an LLM-callable skills layer), all running in
separate Docker containers that share one DDS domain. The G1 has its own
packages in the same workspace (`core_ws/src/g1/`: HOMIE walking policy,
Dex3-1 hands, navigation, description) and a RoboCasa sim (no Isaac).
Both robots run one safety layer, `golem_safety`, and one arm IK controller,
`golem_upper_ik`, each with a profile per robot, and start from one bringup
package, `golem_bringup`. Each robot speaks its own Unitree DDS interface
(`rt/lowcmd` / `rt/lowstate`, and on the G1 also `rt/arm_sdk` and `rt/dex3/*`)
identically against its sim and the real robot, so code moves between them
unchanged.

The H1-2 is the default everywhere. `GOLEM_ROBOT=g1` or `--g1` on a wrapper
selects the G1 (§3). The G1 pages are in `docs/g1/` (start at
`docs/g1/index.md`).

## 2. Repo layout

| Path | What it is |
|---|---|
| `docker/` | x86 setup — the fully-featured, preferred platform (Dockerfiles, `docker-compose.yml`, the G1 overlay `docker-compose.g1.yml`, `scripts/`) |
| `docker/mac/` | self-contained Apple-Silicon (arm64, CPU-only) port — limited: H1-2 only, no Isaac, no ML/vision stack, FastDDS instead of CycloneDDS |
| `core_ws/` | the ROS 2 workspace — where the bulk of development happens (see §4); the G1 packages are in `core_ws/src/g1/`, the safety layer and arm controller both robots run in `core_ws/src/golem_safety/` and `core_ws/src/golem_upper_ik/` |
| `golem_robocasa/` | RoboCasa/MuJoCo simulator entry point for both robots (`h12_mujoco.py`; `--g1` hands over to the G1 sim package `golem_robocasa/g1/`) and the ROS/DDS bridges |
| `CL_isaaclab_sim/` | Isaac Lab simulator, H1-2 only (`sim_main.py`, DDS bridges, tasks) |
| `CL_Assets/` | URDF meshes, MuJoCo XML, Isaac USD (Git-LFS) |
| `third_party/leprechaun_g1` | G1 robot description submodule (URDF, MJCF, meshes); `g1_description` and the G1 sim build on it |
| `mujoco_mpc/` | MuJoCo-MPC fork submodule (built standalone, outside colcon — see §8) |
| `unitree_sdk2_python/` | vendored Unitree DDS SDK (Python) |
| `tests/` | simulator regression and acceptance tests (`tests/README.md`); the G1 acceptance tests are in `tests/g1/` |
| `tools/` | standalone debug tools (ROS MCP server for Codex) |
| `docs/` | user-facing docs: `SETUP.md`, `RUNNING.md`, `MACOS.md`, `DDS_TUNING.md`, `DOCKER_TESTING.md`, `NAVIGATION_DEMO.md`, `ROS_MCP_DEBUG.md`, and the G1's in `docs/g1/` |
| `docker/BUILD.md` | deep dive on the image/build system — read it before touching any Dockerfile |

Most of the tree is **git submodules** (`git submodule update --init --recursive`
is mandatory) and large assets are **Git-LFS** (`git lfs install`).

## 3. Containers: always use the provided scripts

Never `docker run` the images by hand — the scripts wire up `docker/.env`, DDS
domain safety checks, bind-mount pre-creation, robot selection and stable
container names. They can be invoked from any directory.

### x86 (preferred platform)

| Script | Verified arguments | Notes |
|---|---|---|
| `docker/scripts/docker_build.sh [profile ...]` | `isaac`, `robocasa`, `ros` (no args = all three) | `golem_base` is built first automatically for `robocasa`/`ros`; `isaac` is self-contained. Pins the MJPC build to the `mujoco_mpc` submodule SHA (`MJPC_REF`). The images serve both robots. |
| `docker/scripts/docker_run.sh <profile> [--restart] [cmd...]` | `isaac`, `robocasa`, `ros`; `--restart` right after the profile replaces an existing container; then optionally `bash` (shell instead of default launcher) or any command; a leading `-flag` (e.g. `--headless`) is forwarded to the default launcher; `--g1` / `--h12` anywhere in the arguments select the robot | Container names: `golem_ros`, `golem_sim_robocasa`, `golem_sim_isaac`, for either robot. Prints `robot: h12` or `robot: g1` before the domain check. |

Paired startup: `docker/scripts/docker_stack.sh up robocasa|isaac [--headless|--gui] [--g1|--h12]`.
Use `restart`, `stop`, or `logs` with the same simulator name. It prepares the
workspaces before physics and gates bringup on a fresh DDS state sample from a
simulator of the selected robot (`wait_for_sim.py`). The bringup it starts is
`golem_bringup h12_sim.launch.py`, or `golem_bringup
g1_sim.launch.py` for the G1 (with only `use_rviz` set).
`GOLEM_ASSETS_DIR` and `GOLEM_CACHE_DIR` relocate host mounts. Shell exports
win over `docker/.env`. Existing single-service containers require explicit
`--restart`. See `docs/DOCKER_TESTING.md` for both platform smoke tests.

Default launchers (compose `command`): `docker/scripts/launch_isaac.sh`,
`launch_robocasa.sh`, `launch_ros.sh`. The `ros` launcher colcon-builds
`core_ws` (incrementally, on every start) and **drops to a shell — it does not
launch bringup**.

### Robot selection (x86 only)

`docker/scripts/docker_common.sh` owns this. The robot is `--g1`/`--h12` on the
wrapper, else `GOLEM_ROBOT` from the shell, else from `docker/.env`, else
`h12`. For `g1` the wrappers:

- add `docker/docker-compose.g1.yml` to both services: it sets `GOLEM_ROBOT=g1`,
  passes the G1 sim knobs (§8) and `G1_DDS_INTERFACE`, and bind-mounts
  `third_party/leprechaun_g1` and `core_ws/src/g1/g1_locomotion/policy` (at
  `/home/code/g1_locomotion`) read-only;
- write `$GOLEM_CACHE_DIR/cyclonedds_local.xml` from `G1_DDS_INTERFACE` just
  before they start a `ros` container (`docker_run.sh ros`, `docker_stack.sh
  up|restart`; never for a sim-only run), which the `ros` container loads after
  `core_ws/cyclonedds.xml` to pin CycloneDDS to that host interface (empty =
  CycloneDDS chooses). It is a single-file bind mount that CycloneDDS rereads in
  every new process, so the wrappers replace it by rename (a running container
  keeps the pin it started with) and refuse a value that is not an interface
  name. The pin is for the real robot on domain 0: on a sim domain the wrappers
  warn when it is set, since the sim's ROS nodes are never pinned;
- check in `prepare_mounts` that the leprechaun URDF/MJCF and the policy files
  exist and are not LFS pointers, before any container starts;
- refuse the Mac wrappers and the Isaac simulator.

`launch_ros.sh` builds the `g1_*` packages only when `GOLEM_ROBOT=g1`
(`--packages-ignore-regex '^g1_'` for the H1-2), so the H1-2 never depends on
`third_party/leprechaun_g1`. `golem_safety`, `golem_upper_ik` and
`golem_bringup` build for both robots; `golem_upper_ik` imports the G1's model
(`g1_description`) only when a node runs with `robot:=g1`. Both robots use the
same images and container names, so the wrappers run one robot at a time on a
host.

### mac (`docker/mac`, limited)

| Script | Verified arguments | Notes |
|---|---|---|
| `docker/mac/scripts/docker_build_mac.sh [service ...]` | `robocasa`, `ros` (no args = both) | **No `isaac`** — Isaac Sim needs an NVIDIA GPU. arm64 base built from `docker/mac/BaseDockerfile.arm64`. |
| `docker/mac/scripts/docker_run_mac.sh <service> [cmd...]` | `robocasa`, `ros` | Starts ONE service. For paired startup use `docker/mac/scripts/docker_stack_mac.sh up robocasa` (RoboCasa running alone for a few seconds lets the robot collapse — motor command timeout). |

The Mac wrappers run the H1-2 only; they refuse `--g1` and `GOLEM_ROBOT=g1`.
Mac feature toggles are env vars read by the compose file: `GOLEM_DISPLAY=vnc`
(MuJoCo viewer → noVNC :6080), `GOLEM_RVIZ=vnc` (RViz → :6081), `GOLEM_LOWERBODY=fame|walk|switch`,
`GOLEM_SLAM=1`, `GOLEM_NAV2=1`, `GOLEM_SIM_ODOM=1`, `GOLEM_CAMERAS=0`,
`GOLEM_SPAWN_BACKOFF=<m>`, `GOLEM_CMD_TIMEOUT=<sim-s>`, `GOLEM_ROS_MCP=1`.
`GOLEM_SPAWN_MARGIN=<m>` (default 0.20; a task env can set its own via
`spawn_margin`, as `battery_env.py` does) sets the H1-2's minimum spawn
clearance from non-floor geoms on both platforms. `GOLEM_CAMERA_SCALE=<f>`
(H1-2 default 0.5 of the real 1280x720), `GOLEM_CAMERA_DEPTH=0` and
`GOLEM_CAMERA_POINTCLOUD=0` tune the H1-2's RoboCasa cameras on both platforms.
GUIs stream over noVNC via `docker/mac/scripts/mac_vnc_tunnel.sh` (no XQuartz).

### DDS domain safety (both platforms)

`ROS_DOMAIN_ID=0` is the **real robots' command bus**. The run scripts refuse to
start a sim on it (the x86 `ros` profile asks for interactive confirmation; mac
always rejects). Unset/empty defaults to `1` (the sim domain). Set it in
`docker/.env` (copy from `docker/.env.example`; also holds `GEMINI_API_KEY`,
which the `ros` container needs for the vision servers and `h12_skills`, and
`GOLEM_ROBOT` and `G1_DDS_INTERFACE`).

The real G1 is commanded from the laptop wired to it, through
`docker/scripts/docker_run.sh ros --g1` on domain 0 (the H1-2 runs natively on
its onboard PC). For the G1 on domain 0 the wrapper first refuses a
`G1_DDS_INTERFACE` that is empty, names no host interface, is not `UP` or has
no IPv4 address, and only then asks for confirmation; the prompt names the
robot, the interface and its IPv4 addresses. It cannot tell whether that port is
the robot's; check it yourself (`docs/g1/first-session.md`, step 6).

**One robot stack per DDS domain.** The two stacks share node, action and DDS
topic names, with identical `unitree_hg` types. Each robot's sims report its
real robot's `rt/lowstate` `mode_machine`: 6 for the H1-2, 5 for the G1.
`wait_for_sim.py` refuses a simulator of the other robot by that value, but it
runs only in `docker_stack.sh` and the mac stack (`GOLEM_WAIT_FOR_SIM=1`) and in
`tests/stack`. `docker_run.sh ros` followed by a manual `ros2 launch` is not
guarded by the wrappers. `golem_safety` checks `rt/lowstate`'s value too (§8),
but that does not replace one stack per domain: the stacks share the
`rt/safety/*` input topics and the node and action names, `golem_safety` does
not check which robot an input command was written for, and `golem_upper_ik`
does not check `rt/lowstate`'s `mode_machine`.

## 4. `core_ws/src` packages

25 ROS packages: 21 (the H1-2's, the shared ones and the three `golem_*`
packages both robots run) in 18 top-level directories, and the 4 G1 packages in
`core_ws/src/g1/` (below). Note two name mismatches: dir
`FAST_LIO` → package `fast_lio`; dir `unitree_ros2` is not itself a package (it
nests `unitree_api`, `unitree_go`, `unitree_hg`, `unitree_ros2_example`).

| Directory | Role | What it does | Key entry points |
|---|---|---|---|
| `golem_safety` | **safety (both robots)** | C++. The only writer of `/lowcmd` and `/arm_sdk` (§8): relays the controllers' `/safety/*` inputs, clips them, checks every `/lowstate` (robot identity, estop limits) and latches an estop until restart. A robot profile (`robots/<robot>.yaml`, the joint table both `golem_*` packages read) and configs per robot (`config/h12/`, `config/g1/`) | `safety_node` (`robot`, `config`, `sim`, `use_estop` parameters); `launch/safety.launch.py` |
| `golem_upper_ik` | **controller (upper body, both robots)** | Pink IK on the 14 arm joints with the torso (H1-2) or waist (G1) held; OMPL planning and a self-collision barrier where the robot profile has a self-collision model (the H1-2); the follow gate; the safety/controller config pairing (`golem_upper_ik.pairing`) the bringups check | `frame_task_server` (`/frame_task`, `/named_config`), `joint_state_publisher`; `launch/upper_ik.launch.py` |
| `golem_bringup` | **bringup (both robots)** | Launch-only package that starts the whole H1-2 (`launch/h12/`) or G1 (`launch/g1/`), sim or real. See §6. | top-level + building-block launch files, `config/h12/*.yaml` |
| `h12_ros2_model` | description | H1-2 robot description (URDF, meshes, RViz launch) | `launch/`, `assets/` |
| `h12_deploy_mjpc` | controller (lower body, MPC) | MJPC balance/locomotion controller (C++ cores from the `mujoco_mpc` fork, run as separate processes) + RW-EKF base estimator | `mjpc_lowerbody_core`, `controller_launcher.py`, `estimator_node.py` |
| `h12_lowerbody_rl` | controller (lower body, RL) | TorchScript walking, FAME RMA stand/squat and ALMI stand/walk policies; controller runs one pinned policy; consumes `/lowstate` + `/cmd_vel`, emits 12-joint PD setpoints | `walking_node`, `fame_node`, `lowerbody_controller_node` |
| `lmg_deploy` | controller (whole body, MARL) | The loco-manipulation game's two-agent (legs + IK arms) MARL policy, run by the `lmg_*` bringups | `marl_deploy_node` |
| `h12_slam` | SLAM + navigation | The shared nav launch (FAST-LIO → pointcloud_to_laserscan → slam_toolbox → nav2) and its configs: Livox MID360 driver config (`MID360_config.json`, real robot, read by `h12_real_drivers.launch.py`), FAST-LIO (`mid360.yaml`), slam_toolbox (`slam_toolbox_h1.yaml`), Nav2 (`nav2_config.yaml`; mac bringup: `nav2_config_mac.yaml`) | `launch/h12_navigation.launch.py`, `config/` |
| `h12_skills` | **skills** | Action servers for the 12 `/skill/*` atomic skills an LLM can call (frontier exploration lives here). See §7. | `skills` (SkillsNode) |
| `model_server` | model server | ROS service servers wrapping ML models: Gemini VLM, SAM3 segmentation, GraspGenX grasp generation, YOLO detection | `gemini_server`, `sam_server`, `graspgen_server`, `yolo_server` |
| `custom_ros_messages` | interfaces (both robots) | 19 actions (13 `Skill*`, `FrameTask`, `NamedConfig`, `DualArm`, …), msgs, srvs (`GeminiQuery`, `SamSegment`, `GraspGen`, `SetHandPose`, …) | rosidl only |
| `magpie_control` | driver (gripper/sensors) | Dynamixel gripper node (`/left|right/gripper/*` services, DeliGrasp action), F/T + tactile sensor nodes | `gripper_node`, `ft_sensor_node` |
| `magpie_msgs` | interfaces | Gripper/manipulation interfaces (`GripperState`, `SetGripperPosition/Force`, `DeliGrasp`, …) | rosidl only |
| `estop` | safety (hardware) | Serial e-stop bridge → ROS Bool (real robot only; launched by real bringup, not sim) | `estop_node` |
| `cl_realsense` | perception util | Point-cloud accumulator (tf2 + Open3D) for the RealSense cams; ships the real-camera launch file | `pc_acc`; `launch/h12_rs_cams.launch.py` |
| `FAST_LIO` (`fast_lio`) | SLAM (vendored) | LiDAR-inertial odometry/mapping from the Livox cloud (used by `h12_slam` and `g1_navigation`) | `fastlio_mapping` |
| `livox_ros_driver2` | driver (vendored) | Livox MID360 lidar driver (builds via its own `build.sh humble`) | `livox_ros_driver2_node` |
| `unitree_ros2/*` | interfaces (vendored) | Unitree IDL packages — `unitree_hg` `LowCmd_`/`LowState_` are the stack's spine | rosidl only |

### The G1 packages (`core_ws/src/g1/`)

Built only for `GOLEM_ROBOT=g1` (§3). Their interfaces are in
`custom_ros_messages`; each package has a README with the
details, and `docs/g1/packages.md` indexes them.

| Directory | Role | What it does | Key entry points |
|---|---|---|---|
| `g1_locomotion` | controller (lower body) | HOMIE's walking policy (onnxruntime) on the 12 leg joints, driven by `/cmd_vel` through `golem_safety`'s split mode | `locomotion_node` |
| `g1_dex3` | driver (hands) | Dex3-1 hands on `rt/dex3/<side>/*`: `/<side>/hand/{set_pose,open,close,release}`; bypasses the safety layer | `dex3_hands`; `launch/hands.launch.py` |
| `g1_navigation` | SLAM + navigation | FAST_LIO (or the sim's ground truth) → slam_toolbox map → Nav2 on `/cmd_vel`; the G1's own frontier explorer, serving `/skill/frontier_explore` | `launch/g1_navigation.launch.py`, `frontier_explore`, `livox_to_pointcloud`, `level_frame`, `body_to_pelvis` |
| `g1_description` | description | G1 URDFs and meshes (from `third_party/leprechaun_g1`, through a symlink), joint table, DDS slot layout | `launch/display.launch.py`, `g1_description.paths` |

Third-party/vendored (do not refactor): `FAST_LIO`, `livox_ros_driver2`,
`unitree_ros2`, plus the top-level `unitree_sdk2_python`, `mujoco_mpc` and
`third_party/leprechaun_g1`, and HOMIE's policy in
`core_ws/src/g1/g1_locomotion/policy/` (non-commercial licence, §8).

## 5. Ownership & scoping convention

Development is **scoped to a single package**. Each package (the ROS 2
controller, the MJPC deploy package, the lower-body RL package, the skills
package, the model servers, the simulators, the G1 packages, …) has its own
owner; several are separate git submodules with their own history. Rules of
thumb:

- Confine a change to the package it belongs to. If a task seems to require
  edits across packages, stop and coordinate first.
- Shared infra — `golem_bringup`, `custom_ros_messages`/`magpie_msgs`
  (both robots use `custom_ros_messages`), `docker/`, `golem_safety` and
  `golem_upper_ik` (both robots run them), the simulators — affects everyone.
  Treat changes there as interface changes: minimal, deliberate, announced.
- Submodule packages must be committed and pushed in the submodule repo, then
  the pin bumped here. Never leave a submodule pointing at an unpushed commit.

## 6. Bringup: how the robot starts

### The H1-2 (`golem_bringup`, `launch/h12/`)

`golem_bringup` is launch-only. Sim vs real is **separate top-level launch files**
(no top-level file takes a `sim:=` argument); each one only includes the shared
building blocks below and sets `use_sim_time` plus its platform's arguments,
including `h12_control.launch.py`'s `sim`, which has no default.

| Top-level launch file | Scenario | Includes |
|---|---|---|
| `h12_sim.launch.py` | **x86 sim** (`use_sim_time=true`), plus the camera optical-frame static TF and rviz (`use_rviz`) | state, control (`sim:=true`, `sim_safety_split` for both `safety_node` and `frame_task_server`, no stagger), lowerbody (`lowerbody_sim.yaml`), models (`use_skills`), navigation (`use_nav`) |
| `h12_sim_mac.launch.py` | **mac sim** — standalone, includes none of the blocks: `robot_state_publisher` and `golem_upper_ik`'s `joint_state_publisher` and `frame_task_server`, `golem_safety`'s `safety_node` (`sim: true`, both on `sim_safety_split`); lower body/SLAM/nav2 gated by `GOLEM_*` env vars, not launch args | — |
| `h12_real_robot.launch.py` | **real robot, onboard PC** (native, `ROS_DOMAIN_ID=0`) | real_drivers, navigation, state, control (`sim:=false`, `use_estop:=true`, the defaults `relax_safety_split` and `safety_split`), lowerbody (`lowerbody_real.yaml`, no estimator). Leg control is interlocked behind `start_position_verified:=true` (default **false**) |
| `h12_real_desktop.launch.py` | **real: companion x86 desktop** — rviz, plus the hand cameras with `use_hand_cameras:=true` (default **false**) | models (`use_yolo:=true`) |
| `h12_safety_grippers.launch.py` | real: actuation-only debug bringup | control (`sim:=false`, estop + safety, no IK), grippers |
| `lmg_sim.launch.py` | **x86 sim**, loco-manipulation game: the `lmg_deploy` MARL policy (Explicit500) owns all 27 joints, no operator confirm. Refuses an unset or 0 `ROS_DOMAIN_ID` | state, control (`sim:=true`, no IK, no stagger), locomanipulationgame, navigation (`use_slam`) |
| `lmg_real.launch.py` | **real, onboard PC**, loco-manipulation game, run *instead of* `h12_real_robot`. Policy behind `start_position_verified:=true` + `/marl_deploy_node/confirm_engage` | real_drivers, state, control (`sim:=false`, `use_estop:=true`, no IK), locomanipulationgame, navigation (`use_slam`) |

| Building block | Starts |
|---|---|
| `h12_state.launch.py` | `golem_upper_ik`'s `joint_state_publisher` (`robot: h12`, the 27 body joints from `/lowstate`) + `robot_state_publisher` (`urdf_file` arg) |
| `h12_control.launch.py` | [`estop_node`] → `golem_safety`'s `safety_node` (`robot: h12`, `safety_config`, `sim`, `use_estop`) → `golem_upper_ik`'s `frame_task_server` (`robot: h12`, `frame_task_config`; `use_frame_task`), staggered by `safety_delay` (3 s) / `frame_task_delay` (5 s). `sim` has no default: a launch without it fails before any node starts. With `use_frame_task` it refuses a `frame_task_config` that does not run with `safety_config` (`golem_upper_ik.pairing`) before starting anything |
| `h12_lowerbody.launch.py` | RL `lowerbody_controller_node` (`lowerbody:=`, params from `lowerbody_params`) and optional MJPC `estimator_node`, both behind `start_position_verified` |
| `h12_models.launch.py` | gemini + sam servers, [yolo], graspgen + `h12_skills` (`use_skills`) |
| `h12_real_drivers.launch.py` | Livox MID360 (+ `livox_imu_upright`, which rotates `/livox/imu_raw` into the point frame — the stock driver rotates only points), RealSense head camera, `h12_grippers.launch.py` (left+right `gripper_node`) |
| `h12_locomanipulationgame.launch.py` | `marl_deploy_node` (behind `start_position_verified`; holds arm goals in the world by the SLAM `/Odometry`) |
| `h12_slam/launch/h12_navigation.launch.py` | shared nav stack (FAST-LIO → pointcloud_to_laserscan → slam_toolbox → nav2) |

`h12_slam` also ships one top-level file of its own, `h12_real_slam.launch.py`
(`ros2 launch h12_slam h12_real_slam.launch.py`): real robot sensors + SLAM/navigation
only, on wall time — real_drivers, state, navigation; no estop, safety or IK, so
no joint is commanded. Record its runs with `ros2 run h12_slam record_slam_bag --name <name>`.

Humble includes are not scoped: an argument passed to one include stays set
for the includes after it, so pass consistent values (all four blocks share
`use_sim_time`).

**The H1-2's start-up** (`h12_control.launch.py`; `golem_safety`'s and
`golem_upper_ik`'s READMEs have the details):

- `safety_node` streams zero-gain `/lowcmd` from boot. It engages each input on
  that input's first valid command once `/lowstate` reports `mode_machine` 6,
  and, with `use_estop:=true`, once a first hardware estop sample has arrived;
  until then it logs `engage_refused` with the reason, at most once a second.
- **`/safety/heartbeat` means relaying can begin.** `safety_node` publishes no
  beat until those two conditions hold; then it beats true at 10 Hz, and false
  once its estop latches (false from the start if it latched first).
  `h12_lowerbody_rl` pre-poses the legs only from the first beat, and
  `lmg_deploy` and `h12_skills` read it too.
- **`frame_task_server` needs `golem_safety` relaying.** It waits up to
  `controller.state_timeout` (5 s) for a fresh `/lowstate`, then holds the
  measured pose until it sees its commands on `/lowcmd` (the follow gate), for
  up to `controller.startup_timeout` (30 s). If `golem_safety` has not relayed
  them by then, it sends one zero-gain command and exits 1 with an error that
  lists the causes: no `/lowstate` reporting `mode_machine` 6 (an H1-2 sim
  started before the sims reported 6), no first estop sample with
  `use_estop:=true`, a safety config that does not pair with its controller
  config, or an estop latched in an earlier run. The other nodes keep running,
  without `/frame_task` and `/named_config`.
- **`use_estop`.** `use_estop:=true` (the real bringups and
  `h12_safety_grippers`) starts `estop_node` and has `safety_node` read
  `/estop/triggered` and `/estop/plugged_in`: no input engages before the first
  sample, a sample that says triggered or unplugged latches the estop until
  restart, and so does no new sample for `estop.timeout_seconds` (0.5 s) once
  the first has arrived. With `use_estop:=false` (the sim bringups) those topics
  are ignored.
- **The real robot's estop start-up.** `estop_node` publishes nothing until its
  first attempt to connect ends; that attempt probes every serial port it finds
  (up to about 3.25 s for each port that opens, so another serial device such as
  a gripper board lengthens it). When an attempt fails, it publishes unplugged
  and triggered until a retry, once a second, connects. **A `safety_node` that
  is running while `estop_node` reports a failed connection latches until
  restart**: when the first attempt fails and no retry has connected by
  `safety_delay` (3 s), or the first attempt outlasts `safety_delay` and then
  fails. A first attempt that outlasts `safety_delay` and then connects only
  delays the start: no input engages and no heartbeat goes out until its first
  sample, so the legs pre-pose and the arms start late. `safety_delay` must
  cover the time `estop_node` takes to connect (the comment in
  `h12_control.launch.py`).

**Wiring a new package into bringup:** add it as `exec_depend` in
`golem_bringup/package.xml`, add a `launch_ros.actions.Node(...)` entry to the
relevant launch file (put node params in `golem_bringup/config/h12/<file>.yaml` — the
top-level YAML key must match the node `name`), and gate optional nodes with
`IfCondition(LaunchConfiguration(...))`. Add the node to the building block it
belongs to (or a new block included by the top-level files), not to a top-level
file; see `h12_lowerbody.launch.py` as the model. **Mac only:** also add the package to
the `PKGS` list in `docker/mac/scripts/launch_ros_mac.sh` or the slim image
won't build it; a package the Mac does not build goes into `BRINGUP_STUB_DEPS`
there instead, or the Mac build of `golem_bringup` fails. For the G1 the node
goes into `launch/g1/g1_core.launch.py`, with the same `exec_depend`.

### The G1 (`golem_bringup`, `launch/g1/`)

Two top-level files include one shared core, `g1_core.launch.py`, with `sim`
and `use_sim_time` set for their platform. The docstring of each file is its
reference.

| Launch file | Scenario | Starts |
|---|---|---|
| `g1_sim.launch.py` | **G1 sim** (sim time, the `sim_*` configs). `use_locomotion:=true` adds walking (switching the safety layer to `sim_split`), `use_nav:=true` navigation (needs `use_locomotion:=true`; `odom_source:=fastlio` or `sim_gt`), `frontier_explore:=true` the explorer | `g1_core` |
| `g1_real_robot.launch.py` | **real G1**, every node in the laptop's `golem_ros` container on domain 0. Arms only by default: `safety_config:=arm_sdk`, Unitree's controller keeps the legs. No sensors or skills yet | `g1_core` |
| `g1_core.launch.py` | shared core: `robot_state_publisher`, `golem_upper_ik`'s `joint_state_publisher` (`robot: g1`), `golem_safety`'s `safety_node` (`robot: g1`, `safety_config`, `sim`, `use_estop: false`: the G1 has no estop node), then `golem_upper_ik`'s `frame_task_server` (`robot: g1`, `controller_config`) after `controller_delay` (2 s); [`dex3_hands`] (`use_hands`), [rviz] (`use_rviz`), [`locomotion_node`] (`use_locomotion`), [`g1_navigation`] (`use_nav`) | — |

The core refuses to start on a bad combination; `startup_refusal` in
`g1_core.launch.py` and `pairing_refusal` in
`core_ws/src/golem_upper_ik/golem_upper_ik/pairing.py` are the rules:

- **Config pairing.** Each safety config runs with exactly one controller
  config (`G1_SAFETY_CONTROLLER_CONFIGS` in `pairing.py`), and
  `controller_config` defaults to it. The real robot refuses every `sim_*`
  config and the sim refuses the real ones.
- **Walking interlock.** On the real G1, `use_locomotion:=true` needs
  `safety_config:=split` and `start_position_verified:=true` (default
  **false**), set only after checking the robot is in a safe start pose. Split
  mode writes `rt/lowcmd`, so Unitree's controller must already be released, a
  manual operator step that nothing in this stack performs; support the robot
  on a gantry (the split estop turns every motor off).
- **No real navigation.** `use_nav:=true` is refused on the real G1 until the
  real MID-360 is connected to the bringup.
- **Engage gate, no engage service.** `golem_safety` publishes nothing until a
  command passes its engage gate (a fresh `rt/lowstate` reporting
  `mode_machine` 5, every input of the mode commanding and none quiet, each
  command within `engage.max_position_error` of the measured joints), and
  `golem_upper_ik`'s follow gate keeps the controller's commands within
  `controller.follow_tolerance` of the measured joints until `golem_safety`
  relays them. `frame_task_server` stops with an error if that has not happened
  within `controller.startup_timeout` (30 s). After any estop, restart the whole
  bringup.
- **The controller's exit.** When `frame_task_server` stops, it sends one last
  command: mode 0 and zero kp, kd and tau on the arms, and in arm_sdk mode on
  the waist too. `golem_safety` refuses that one (its waist kp is below
  `arm_sdk.waist_kp_min`), latches its estop, logged as one, and hands the
  waist and arms back to Unitree. In split (and full-body) mode the waist keeps
  its hold angle and gains, since a walking policy needs it stiff; `golem_safety`
  relays the command, so only the arms go limp while the legs keep their
  commands. `docs/g1/safety.md` has the per-mode details.

## 7. Skills (`h12_skills`) — the LLM-callable layer

Each skill is a mixin class in `h12_skills/skills/` with one
`_exec_<name>()` method; `SkillsNode` registers one action server per entry in
the `SKILL_ACTIONS` table (`base.py`). All types are
`custom_ros_messages/action/Skill*`; every goal has a `timeout` Duration
(0 → 300 s default); every result is `success` + `message`; feedback is
`phase` + `progress`.

| Action server | Status |
|---|---|
| `/skill/grasp`, `/skill/pick_place`, `/skill/frontier_explore` | **implemented** |
| `/skill/open_door`, `/skill/close_door`, `/skill/open_lid`, `/skill/close_lid`, `/skill/navigate`, `/skill/press`, `/skill/slide_rack`, `/skill/turn_lever`, `/skill/twist_knob` | stubs — goal accepted, then aborted with "not implemented" |

`custom_ros_messages` also defines `SkillRetrieveMemory`; no node in this
repo serves it yet. On the G1, `h12_skills` does not run: `/skill/frontier_explore`
is served by `g1_navigation`'s `frontier_explore` (`g1_sim.launch.py
... frontier_explore:=true`), with the same action type.

Skills consume the `frame_task` IK action, the gripper services, and the
`model_server` perception services (`gemini_query`, `sam_segment`, `graspgen` —
this is why the `ros` container needs `GEMINI_API_KEY`). There is currently no
autonomous LLM orchestrator in-repo that calls them; they are invoked over ROS
(by a human, an external agent, or the `tools/ros_mcp_server.py` debug MCP).

Verified invocation examples (from a shell inside `golem_ros`):

```bash
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" \
  --feedback

ros2 action send_goal /skill/frontier_explore custom_ros_messages/action/SkillFrontierExplore \
  "{min_frontier_cells: 6, blacklist_radius: 0.6, min_goal_distance: 0.7, goal_timeout: 120.0, timeout: {sec: 1800, nanosec: 0}}" \
  --feedback

# non-skill but useful: move arms to a named posture first
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'home', duration: {sec: 0, nanosec: 0}}"
```

## 8. Do-not-break rules

### The simulators are shared infra — don't modify them casually

`CL_isaaclab_sim` and `golem_robocasa` should generally not be modified unless
necessary, and their **default behaviors must be preserved**. Everyone's
sim-vs-real workflow depends on them behaving like the real robot.
`golem_robocasa/h12_mujoco.py` is the one RoboCasa entry point for both robots: it
picks the robot (`--h12`/`--g1`, else `GOLEM_ROBOT`, else `h12`) before its
H1-2 imports and hands the G1 over to `golem_robocasa/g1/g1_mujoco.py`. The H1-2
stays the default, and nothing on the G1 path may change its behaviour. The robots share
`sim_common.py`, `measurement_bridge.py` and the `mujoco_sensors` node in
`mujoco_ros_bridge.py` (`sim_timing.py` is the H1-2's only); each has its own sensor suite
(`h12_sensors.py`, `g1/g1_sensors.py`) and Unitree DDS interface
(`unitree_interface.py`, `g1/unitree_interface.py`).

### Simulators mimic the real robot's topic surface

They should publish only what the real robot publishes and consume only what
it consumes. Verified current surface:

**RoboCasa, H1-2 (`golem_robocasa/h12_mujoco.py`)** — publishes `/clock`, the real
RealSense drivers' topics, frame names and camera TF (`/tf_static` + `/tf`) for
`/realsense/{head,left_hand,right_hand}` — colour, depth and aligned depth with
their camera_info, and `depth/color/points` — as captured from the robot in
`golem_robocasa/realsense_profile.yaml`, `/livox/lidar` (CustomMsg), `/livox/pointcloud`, `/livox/imu`,
`/{left,right}/gripper/state`, and `rt/lowstate` over Unitree DDS (with the
real robot's `mode_machine`, 6); consumes `rt/lowcmd` and the
`/{left,right}/gripper/*` services/DeliGrasp action.
(Trust the code over any topic list repeated elsewhere: the sensor suite in
`h12_sensors.py`, the grippers in `magpie_hand_bridge.py`, `/clock` and the
opt-in odom in `mujoco_ros_bridge.py`, Unitree DDS in `unitree_interface.py`.)

**RoboCasa, G1 (`h12_mujoco.py --g1`, `golem_robocasa/g1/`)** — publishes `/clock`,
the Livox MID-360 as its driver does in CustomMsg mode (`/livox/lidar`,
`/livox/imu`, no `/livox/pointcloud`), the head RealSense D435
(`/realsense/head/*` and its `/tf_static`), and over Unitree DDS `rt/lowstate`
and `rt/dex3/{left,right}/state`; consumes `rt/lowcmd`, `rt/arm_sdk` and
`rt/dex3/{left,right}/cmd`. A HOMIE stand-in plays Unitree's controller and
holds the robot up until the first valid `rt/lowcmd`; `rt/arm_sdk` blends the
waist and arms with it by the weight in slot 29. (Trust `g1/g1_sensors.py` for
the ROS side and `g1/unitree_interface.py` for the DDS side.) Its knobs are
`GOLEM_LIDAR`, `GOLEM_LIDAR_POINTS`, `GOLEM_LIDAR_HZ`, `GOLEM_CAMERAS`,
`GOLEM_CAMERA_HZ`, `GOLEM_CAMERA_SCALE` (G1 default 1.0), `GOLEM_CMD_TIMEOUT`,
`GOLEM_ARM_SDK_TIMEOUT`, `GOLEM_SPAWN_BACKOFF`, `GOLEM_SIM_ODOM`,
`GOLEM_SIM_EXPERIMENT` and `G1_DDS_INTERFACE`. Their defaults are in the code
that reads them: `SensorConfig` in `g1/g1_sensors.py` (sensors),
`g1/unitree_interface.py` (timeouts, interface), `g1/g1_mujoco.py` (spawn,
odom) and `measurement_bridge.py` (experiment); the compose files pass them
(`docker/docker-compose.g1.yml` the G1-only ones). The G1 sim ignores the H1-2-only
knobs (`GOLEM_SPAWN_MARGIN`, `GOLEM_SPAWN_OFFSET`, `GOLEM_HAND_CAMERA_RES`,
`GOLEM_CAMERA_DEPTH`, `GOLEM_CAMERA_POINTCLOUD`, the tether ones). It refuses
`--timing`, and the H1-2 sim refuses `--duration` (`sim_common.py`).

The measured surface of each RoboCasa robot is checked in as a golden:
`tests/robocasa/surface/{h12,g1}.json`, compared by
`tests/robocasa/surface/run.sh h12|g1` (see its README). A change that alters
what the sim publishes re-records the golden in the same commit.

**Isaac (`CL_isaaclab_sim/sim_main.py`, H1-2 only)** — publishes `rt/lowstate`
(`mode_machine` 6) and optional `rt/inspire/state` over CycloneDDS, `/clock`,
three RealSense RGBD camera streams with camera info, and
`/livox/{lidar,pointcloud,imu}`. Consumes
`rt/lowcmd` and optional `rt/inspire/cmd`. The default asset is the floating-base Magpie H1-2; `--hand_type inspire`
selects the Inspire fixture. Magpie exposes the real gripper state, services,
and DeliGrasp action through a native ROS process inside the Isaac container.
See `tests/README.md` for saved CPU and cross-container integration tests.

### Ground truth must NEVER be on by default

Sims may *optionally* publish ground-truth data (true base/object poses), but
it must be opt-in. Rationale: any node that consumes ground truth "cheats" —
it works in sim and silently fails on the real robot, where no such topic
exists. Keeping it off by default keeps the sim honest and protects
sim-to-real transfer.

- **RoboCasa complies, for both robots:** ground-truth `/odom` + `odom→pelvis`
  TF is gated by `GOLEM_SIM_ODOM` (default `'0'` = off, read in `h12_mujoco.py`
  and `g1/g1_mujoco.py`). Only the mac nav demo, the G1's `odom_source:=sim_gt`
  and the G1 nav/explore tests' `sim_gt` variants opt in. Keep it that way. The
  experiment surface (`/sim/ground_truth`, `/sim/reset_objects`) is gated by
  `GOLEM_SIM_EXPERIMENT` (`measurement_bridge.py`). The G1 acceptance tests
  record the robot's state inside the sim process instead of publishing it.
- **Isaac:** no full-scene ground-truth publisher or per-step scene JSON conversion.
- **Known deviations (do not make worse):** RoboCasa always publishes privileged task signals
  (`/robocasa/success`, `/robocasa/reward`, `/robocasa/task_goal`), for both
  robots. Never make robot-stack code *depend* on any of these, and don't add
  new ungated ground-truth publishers.

### Do NOT bump the `livox_ros_driver2` submodule

Pinned at **`13eb05e`** ("support Mid-360s Lidar, Ubuntu 24.04 and ROS2 Jazzy").
Leave it there. A bump to `4a1def9` on 2026-08-05 broke both sim images and was
reverted; the same trap is still armed.

Why it breaks: the driver and Livox-SDK2 are **separate upstream repos that ship
paired halves of the same feature**, but only the driver is pinned here — both
Dockerfiles install the SDK with an unpinned `git clone --depth 1 …master`
(`RosDockerfile` §2, `RobocasaDockerfile` near the end). That clone sits in a
layer *above* the driver's `COPY`/build, and Docker invalidates downward only,
so bumping the driver rebuilds it against whatever SDK headers your cache
happens to hold. `4a1def9` (Avia2) calls `kLivoxLidarDoubleEchoData` /
`LivoxLidarDoubleEchoRawPoint`, added in SDK commit `08f523c` — against an
older cached SDK it fails with "was not declared in this scope".
**Stale, not re-verified:** both Dockerfiles check out Livox-SDK2 at `08f523c`
(`RosDockerfile` §2, `RobocasaDockerfile`); the unpinned `master` clone
described in this paragraph is gone. The pin rule above stands.

The failure is silent. Upstream's `build.sh` ends in `popd`, so it exits 0 even
when its colcon build fails, and `launch_ros.sh` doesn't check either. The image
builds "successfully" with `/opt/livox_ws/install/livox_ros_driver2/` holding
only `package.*` stubs — no `local_setup.bash`, no Python module — and you find
out at runtime:

```
ModuleNotFoundError: No module named 'livox_ros_driver2'   # h12_sensors.py, or g1/g1_sensors.py for the G1
```

Whether it reproduces depends on how old your Docker cache is, so "it built on
my machine" proves nothing.

If you genuinely need a newer driver (Avia2/Mid-360s hardware), do it in one
change: add `ARG LIVOX_SDK2_REF=<matching SDK sha>` to **both** Dockerfiles with
a pinned `git fetch --depth 1 origin "$LIVOX_SDK2_REF"`, keep the two SHAs
identical (the images must agree on the `CustomMsg` wire format), and assert the
build worked — `python3 -c 'from livox_ros_driver2.msg import CustomMsg'` — so a
mismatch fails the image build instead of shipping broken.

### Comments describe the code as it is, not how it got there

A comment is scoped to what the code does **now**, written for someone who has
never seen an earlier version. Change history belongs in the commit message and
the PR, not in the source — git already has it, and a comment about a past edit
goes stale the moment the next edit lands.

Don't write:

- change narration — `# now uses X instead of Y`, `# switched to CycloneDDS`,
  `# was 0.5, bumped to 0.8`, `# fixed the TF extrapolation bug`
- dated or attributed notes — `# 2026-07-12 (max): ...`, `# per review`
- status chatter about the edit itself — `# new`, `# updated`, `# temporary`,
  `# deprecated, keep for now`
- commented-out code kept "just in case" — delete it; git remembers
- a restatement of the identifier the comment sits above

Do write what the code can't say for itself: units and frames (`# rad, pelvis
frame`), why a non-obvious constant, ordering or sleep is required, invariants
a caller must uphold, and a link to the upstream issue that forces a workaround.

Rationale from a past change is welcome when the rationale is still
load-bearing — but phrase it as a present-tense constraint, not as a diff:
`# 3.2.3 matches the MJPC ABI` beats `# downgraded from 3.3.1, MJPC broke`.

This applies to edits too: when you change behavior, rewrite the nearby comment
and docstring to describe the new behavior instead of appending a note about the
change. A comment that duplicates a list the code owns (topics, package names,
paths, tuning values) is a stale comment waiting to happen — point at the
authoritative source instead of copying it.

### Other invariants

- **`golem_safety` is the only writer of the body command topics, on both
  robots.** Its `safety_node` is the only publisher of `rt/lowcmd` (ROS
  `/lowcmd`) and, on the G1 in arm_sdk mode, of `rt/arm_sdk` (the waist and
  arms, with the weight in slot 29, while Unitree's controller keeps the legs).
  Controllers publish only to its `rt/safety/*` inputs: `golem_upper_ik` and
  `g1_locomotion` refuse a command topic outside them. The exception is
  `g1_dex3`, which writes `rt/dex3/<side>/cmd` directly, with no safety layer.
- **Quiet inputs are held and reported, not estopped.** Once an input has
  engaged, an input with no new command for its `stale.*_seconds` keeps its
  last command relayed (`input_quiet`, then `input_resumed`), and a quiet
  `rt/lowstate` leaves the relay running (`state_quiet`, `state_resumed`). This
  includes the G1's split mode: a walking-node fault silences
  `locomotion_node`, and `golem_safety` keeps relaying its last leg command,
  which does not balance the robot, instead of turning every motor off.
- **The estop latch** is one way until restart. It latches on a command with a
  non-finite value, a second live writer on an input (a new writer is accepted
  after `single_writer.handover_seconds`, 0.5 s, of quiet), a measured joint
  past `limits.estop` or not finite, another robot's `mode_machine` once
  engaged, the transport-lag check (`control.state_lag_ms`, on domain 0; off in
  every G1 config), in arm_sdk mode a waist kp below `arm_sdk.waist_kp_min`, and
  with `use_estop` a hardware estop that is triggered, unplugged or silent. What
  it writes comes from the config's `estop_response`:
  - H1-2 (all 8 configs): 50 zero-gain commands (mode 0, every value 0), then
    no `rt/lowcmd` until restart. The limit checks run from boot.
  - G1 arm_sdk mode: the last good command, with the weight ramping to 0 over
    `arm_sdk.release_seconds` (the waist and arms go back to Unitree).
  - G1 split (and full-body) mode: the estop command, written every cycle
    (`mode: zero` in every config: every motor off, so the robot drops;
    support it).
  - On the G1 nothing is written for a latch before the engage gate has passed
    (`estop_latched_silent`), and the limit checks start at engagement.
- **Robot identity.** `golem_safety` checks every `rt/lowstate` against its
  robot profile's `mode_machine`, 6 for the H1-2 and 5 for the G1: no input
  engages without it, another value latches the estop once engaged, and every
  command it writes carries the profile's `mode_machine`. Each robot's sims
  report the real value, and `wait_for_sim.py` checks the same values (§3).
- **The controller's exit is one last command.** On SIGINT or SIGTERM, or a
  start-up that fails after its first command, `golem_upper_ik`'s
  `frame_task_server` sends its upper-body slots one command with mode 0 and
  zero kp, kd and tau on the IK joints, then stops publishing. The held joints
  get the same, except on the G1 when `golem_safety` writes `rt/lowcmd` (the
  profile's `shutdown_keeps_held_joints_on_lowcmd`). On the H1-2 `golem_safety`
  relays it, so the torso and arms go limp and stay so (the quiet input keeps
  its last command relayed). On the G1 in split (and full-body) mode it relays
  it too: the waist keeps its hold angle, gains and torque and only the arms go
  limp, while the walking node keeps the legs. On the G1 in arm_sdk mode it
  refuses the command (its waist kp is below `arm_sdk.waist_kp_min`), latches
  its estop, logged as an estop (`estop_command_written`), and hands the waist
  and arms back to Unitree.
- **`golem_safety`'s own exit** (SIGINT, SIGTERM, SIGHUP): in split and full-body
  mode it stops publishing at once (an estop burst in progress is finished
  first); in arm_sdk mode it ramps the weight to 0 over
  `arm_sdk.release_seconds` first. Once `rt/lowcmd` falls silent, the sims turn
  every body motor off after `GOLEM_CMD_TIMEOUT` (0.5 s of sim time); what the
  real robots do then has not been measured.
- **One robot stack per DDS domain** (§3). Never run the H1-2's and the G1's
  stacks, or `h12_skills`' and `g1_navigation`'s frontier explorers, on one
  domain.
- **HOMIE's policy is non-commercial.** `core_ws/src/g1/g1_locomotion/policy/`
  holds OpenHomie's pretrained policy, CC BY-NC 4.0 by its `LICENSE` (its
  README says CC BY-NC-SA 4.0): treat it as CC BY-NC-SA 4.0, unlike the MIT
  `g1_*` packages. Keep its `LICENSE` and `README.md` beside the files, and keep
  `homie_policy.py` inside `policy/`.
- MuJoCo versions are pinned per image (`3.2.3` in `ros` to match the MJPC ABI,
  `3.3.1` in `robocasa`) — never bump one without the other side of its pin.
- `numpy<2`, `setuptools==59.6.0`, `wheel<0.44` are deliberately re-clamped in
  the Dockerfiles; add new pip deps at the END of a Dockerfile (layer-order
  discipline), each through a lock step (`-c /opt/golem-locks/<step>.txt`,
  enforced by `tests/docker/test_source_pins.py`). Details in `docker/BUILD.md`.
- After bumping the `mujoco_mpc` submodule: rebuild the `ros` image at the
  matching `MJPC_REF` and wipe stale `core_ws/build/h12_deploy_mjpc`.

## 9. Running the robot — the standard three-terminal flow

Everyone runs their package through bringup, which starts the whole robot.

### x86, the H1-2

```bash
# Terminal 1 — simulator (start FIRST so /clock is publishing)
docker/scripts/docker_run.sh robocasa            # or: isaac; add --headless if no display

# Terminal 2 — ROS workspace (auto colcon-builds core_ws, drops to a shell)
docker/scripts/docker_run.sh ros
ros2 launch golem_bringup h12_sim.launch.py   # manual step — the container does NOT launch it
# variants: use_skills:=false use_rviz:=false use_nav:=false

# Terminal 3 — drive skills: attach to the running ROS container
docker exec -it golem_ros /bin/bash
source /opt/ros/humble/setup.bash
source /home/code/core_ws/install/setup.bash
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" --feedback
```

### x86, the G1

```bash
# Paired: the sim plus g1_sim.launch.py (arms), gated on the sim's robot
docker/scripts/docker_stack.sh up robocasa --g1 --headless
docker/scripts/docker_stack.sh logs robocasa

# Or three terminals, to choose the bringup's arguments (the wrappers do not
# check the sim's robot here; golem_safety's own check is in §8)
# Terminal 1 — the sim; wait for "[g1_mujoco] ROS bridges up"
docker/scripts/docker_run.sh robocasa --g1 --layout 9 --style 9 --seed 42
# Terminal 2 — builds core_ws with the g1_* packages, drops to a shell
docker/scripts/docker_run.sh ros --g1
ros2 launch golem_bringup g1_sim.launch.py
# variants: use_rviz:=false use_hands:=false use_locomotion:=true use_nav:=true frontier_explore:=true

# Terminal 3
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 't_pose', plan: false, duration: {sec: 0, nanosec: 0}}"
```

Walking and navigation need the spawn backed into open floor
(`GOLEM_SPAWN_BACKOFF=2.0` before terminal 1's command); `docs/g1/walking.md`,
`navigation.md` and `exploration.md` have those flows. The closed-loop checks
are `tests/g1/test_sim_bringup.sh [--walk]`, `tests/g1/test_sim_nav.sh` and
`tests/g1/test_sim_explore.sh` (`tests/g1/README.md`).

### mac

```bash
# Terminals 1+2 in one: start BOTH services together (bringup auto-launches in ros)
docker/mac/scripts/docker_stack_mac.sh up robocasa
# or two terminals: docker/mac/scripts/docker_run_mac.sh robocasa  /  ... ros
# (start ros within seconds of robocasa or the robot collapses)

# Terminal 3
docker exec -it golem_ros bash                    # fallback: colima ssh -- docker exec -it golem_ros bash
source /home/code/h12_sim_scripts/robot_cli.sh   # rob_pose, rob_grip, rob_stand, rob_explore, ...
```

The mac `ros` container auto-runs
`ros2 launch /home/code/core_ws/src/golem_bringup/launch/h12/h12_sim_mac.launch.py`
(launched by path; only skills built on mac are available — no `h12_skills`,
`model_server`, or nav unless `GOLEM_SLAM/GOLEM_NAV2` are set).

### Real robot (for awareness — coordinate before touching)

H1-2: onboard PC, native (no docker): `export ROS_DOMAIN_ID=0`, source
`core_ws/install/setup.bash`, then
`ros2 launch golem_bringup h12_real_robot.launch.py`
(`start_position_verified:=true` only after physically verifying the robot's
pose — it gates leg commands; the policy then holds its pre-pose until
`/lowerbody/confirm_engage`). Companion desktop:
`ros2 launch golem_bringup h12_real_desktop.launch.py`. The arms start only
once `golem_safety` relays `frame_task_server`'s commands, which needs the estop
box connected: a `safety_node` that starts while `estop_node` reports a failed
connection latches until restart, and `frame_task_server` exits after
`controller.startup_timeout` (30 s) when nothing is relayed (§6).

G1: the laptop wired to the robot, in Docker: `ROS_DOMAIN_ID=0`,
`GOLEM_ROBOT=g1` and `G1_DDS_INTERFACE=<the wired port>` in `docker/.env`,
`docker/scripts/docker_run.sh ros --g1` (check that it prints `robot: g1`
before you confirm domain 0), then
`ros2 launch golem_bringup g1_real_robot.launch.py use_hands:=false use_rviz:=false`
— arms only, Unitree's controller keeps the legs. Nothing has run on the real
G1 yet; follow `docs/g1/first-session.md` step by step.

## 10. Getting-started checklist for a new agent

1. `git submodule update --init --recursive && git lfs install` (most of the
   tree is submodules; builds fail cryptically without this).
2. `cp docker/.env.example docker/.env`; set `GEMINI_API_KEY` and
   `ROS_DOMAIN_ID` (any non-zero; 0 is the real robot).
3. Place SAM3 weights at `core_ws/src/model_server/weights/sam3.pt`
   (gated HF download — see `docs/SETUP.md`).
4. `docker/scripts/docker_build.sh robocasa ros` (add `isaac` if needed).
5. Run the three-terminal flow in §9; confirm with `ros2 topic hz /lowstate`
   and `ros2 action list | grep skill` inside `golem_ros`.
6. Trigger `/skill/grasp` or the frontier explorer (§7).
7. Read first: `docs/RUNNING.md` (run flows), `docker/BUILD.md` (build system),
   your target package's source. `README.md` indexes the rest of `docs/`; the
   G1 starts at `docs/g1/index.md`.

### Common pitfalls

- **Start RoboCasa before bringup** — ROS nodes latch onto sim `/clock`; and if
  you restart the `robocasa` container, restart `golem_ros` too (sim time resets
  to 0 → TF extrapolation errors, frozen SLAM map).
- **Restart the sim containers after pulling.** The sims run the checkout's
  `golem_robocasa` and `CL_isaaclab_sim` from bind mounts, so a running sim keeps
  the code it started with. An H1-2 sim started before the H1-2 sims reported
  `mode_machine` 6 is refused: `wait_for_sim.py` stops `docker_stack.sh`, and
  `golem_safety` never engages (`engage_refused`: `/lowstate mode_machine is 0,
  expected 6`), so `frame_task_server` exits after 30 s. For Isaac, run
  `git submodule update` first: the H1-2 Isaac sim reports 6 from the current
  `CL_isaaclab_sim` pin.
- **x86 `ros` container ≠ bringup.** It only builds and gives you a shell.
- **Check the `robot:` line.** `docker_run.sh` and `docker_stack.sh up` print the
  robot they selected; a `GOLEM_ROBOT` left in the shell or `docker/.env`
  switches robots silently otherwise. Stop one robot's containers before starting the other's.
- Rebuild is incremental across container restarts (build/install are
  host-mounted); `colcon build --symlink-install` inside the container forces
  it; wipe `container_cache/msgs_ws/` for a clean message rebuild.
- MJPC C++ iteration: `docker exec -it golem_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh`
  (`--install` to also refresh assets/proto) — not colcon.
- 9 of the 12 `/skill/*` actions are stubs (§7) — a goal that immediately
  aborts with "not implemented" is expected, not a regression.
- Talking to the sim from the host bypasses `docker/.env`; run
  `set -a; source docker/.env; set +a` first or topics will be invisible.
- Laggy point clouds/images on x86 are usually kernel UDP buffer drops — see
  `docs/DDS_TUNING.md` (sysctls + IRQ pinning).
- Isaac's launcher accepts `--task NAME` and `--headless`; Unitree DDS reads
  `ROS_DOMAIN_ID` directly and rejects domain 0. There is no domain relay.
