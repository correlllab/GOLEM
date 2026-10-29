# Running the stack (x86)

Setup first: [`SETUP.md`](SETUP.md). For Apple Silicon see [`MACOS.md`](MACOS.md).

Always start containers through the provided scripts — they wire up
`docker/.env`, the DDS domain check, bind-mount pre-creation, and stable
container names.

## Paired startup

```bash
docker/scripts/docker_stack.sh up robocasa --headless
docker/scripts/docker_stack.sh logs robocasa
docker/scripts/docker_stack.sh restart robocasa --headless
docker/scripts/docker_stack.sh stop robocasa
```

Use `isaac` instead of `robocasa` for Isaac. Add `--g1` to run the G1 instead
of the H1-2 ([The G1](#the-g1)). `--gui` selects desktop mounts;
RViz follows GUI mode unless `GOLEM_USE_RVIZ` overrides it. The stack wrapper
prepares workspaces, starts both containers, and gates bringup on a fresh DDS
sample. Caches persist across stop/restart. Do not restart only the simulator
while ROS is using its clock. See [Docker testing](DOCKER_TESTING.md).

## The three-terminal flow

```bash
# Terminal 1 — simulator. Start FIRST so /clock is publishing before ROS nodes latch on.
docker/scripts/docker_run.sh robocasa            # or: isaac

# Terminal 2 — ROS workspace. Colcon-builds core_ws if stale, then drops to a shell.
docker/scripts/docker_run.sh ros
ros2 launch golem_bringup h12_sim.launch.py  # bringup is a manual step

# Terminal 3 — drive the robot from inside the running ROS container
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash
source /home/code/core_ws/install/setup.bash
```

## `docker_run.sh`

```bash
docker/scripts/docker_run.sh <isaac|robocasa|ros>          # default launcher
docker/scripts/docker_run.sh robocasa --headless           # leading flags go to the launcher
docker/scripts/docker_run.sh robocasa bash                 # shell instead
docker/scripts/docker_run.sh ros <cmd> <args>              # any command
docker/scripts/docker_run.sh ros --g1                      # the G1 (--g1/--h12 anywhere)
```

Container names are `golem_sim_robocasa`, `golem_sim_isaac`, and `golem_ros`,
for either robot.
The default launchers are `docker/scripts/launch_{robocasa,isaac,ros}.sh`.

**The single-service `ros` default is a shell.** Paired startup launches
bringup automatically. To run an initialized custom command, invoke
`docker_run.sh ros /home/code/h12_sim_scripts/launch_ros.sh <cmd> <args>`.
Existing containers require an explicit `--restart` immediately after the
service name; otherwise attach with `docker exec`.

### RoboCasa

Publishes `rt/lowstate` over CycloneDDS, and on the chosen ROS domain: `/clock`;
the real RealSense drivers' topics, frame names and camera TF for
`/realsense/{head,left_hand,right_hand}` (colour, depth and aligned depth, raw and
compressed, with their camera_info, plus `depth/color/points`), as captured from
the robot in `golem_robocasa/realsense_profile.yaml`. The head camera renders at
`GOLEM_CAMERA_SCALE` (default 0.5) of the real 1280x720 with the real
intrinsics; `GOLEM_CAMERA_DEPTH=0` and `GOLEM_CAMERA_POINTCLOUD=0` drop the depth
stream and point cloud. Then the Livox topics `/livox/lidar` (CustomMsg),
`/livox/pointcloud`, `/livox/imu`; and `/{left,right}/gripper/state`. It consumes
`rt/lowcmd` and the `/{left,right}/gripper/*` services.

Scene flags are passed straight through, e.g.
`docker_run.sh robocasa --task OpenFridge --layout 1 --style 1 --seed 42`.
`--task Battery` loads the battery workcell instead of a kitchen
(`golem_robocasa/battery_env.py`; scene and objects in
`CL_Assets/mujoco_assets/battery/`); `--layout`/`--style` don't apply to it.

### Isaac

Uses the same wrappers, on NVIDIA Linux only. The launcher honors
`ROS_DOMAIN_ID` and communicates directly on that simulation domain; zero is
rejected. Select a task with `docker_run.sh isaac --task base --headless`.
Magpie H1-2 is the default asset, with a floating base. Start the ROS lower-body
controller to support it. Select the Inspire fixture with `--hand_type inspire`.
Use `--fix_base` only for a fixed-root bench test. Logs go to container stdout.
The Magpie ROS gripper process starts inside the Isaac container using an
isolated Humble runtime. Its message interfaces are built into the image.

Isaac's sensor/topic surface still differs from RoboCasa (see `CLAUDE.md`);
starting both containers does not imply full navigation/skills parity.

## Bringup

`golem_bringup` is launch-only (the H1-2's files are in its `launch/h12/`), and
sim versus real are separate launch files (no top-level file takes a `sim:=`
argument; each passes its own to `h12_control`):

| Launch file | Scenario |
|---|---|
| `h12_sim.launch.py` | x86 sim — the full stack |
| `h12_sim_mac.launch.py` | mac sim — trimmed, toggled by `GOLEM_*` env vars |
| `h12_real_robot.launch.py` | real robot, onboard PC: drivers, state publishers, estop + safety + IK, nav, lower body |
| `h12_real_desktop.launch.py` | real: companion desktop — model servers, skills, rviz; hand cameras with `use_hand_cameras:=true` |
| `h12_safety_grippers.launch.py` | real: estop + safety layer + grippers only |

The top-level files above are assembled from building blocks you can also
launch on their own: `h12_state` (state publishers), `h12_control` (estop,
safety, IK), `h12_lowerbody`, `h12_models` (model servers + skills),
`h12_real_drivers`, `h12_grippers`, and `h12_navigation` (in `h12_slam`).

For sensors + SLAM/nav on the real robot with no joint commanded (no estop,
safety or IK), use `ros2 launch h12_slam h12_real_slam.launch.py`, which lives in
`h12_slam`; record those runs with `ros2 run h12_slam record_slam_bag --name <name>`.

The x86 sim bringup starts the state publishers, the `frame_task_server` IK
solver, `safety_node`, the Gemini/SAM/GraspGen model servers, the RL lower-body
controller, the skills node, nav, and RViz. Common arguments:

```bash
ros2 launch golem_bringup h12_sim.launch.py \
  use_rviz:=false use_nav:=false use_skills:=false lowerbody:=fame
```

## Worked example — open the fridge

```bash
# Terminal 1
./docker/scripts/docker_run.sh robocasa --task OpenFridge --layout 1 --style 1 --seed 42

# Terminal 2
docker/scripts/docker_run.sh ros
ros2 launch golem_bringup h12_sim.launch.py

# Terminal 3 (inside golem_ros, workspace sourced)
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'home', duration: {sec: 0, nanosec: 0}}"

ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" \
  --feedback
```

`/skill/grasp`, `/skill/pick_place`, and `/skill/frontier_explore` are
implemented; the other nine `/skill/*` action servers accept a goal and then
abort with "not implemented". `core_ws/src/golem_bringup/common_cmds.md` has a
ready-made `send_goal` snippet and field reference for each one.

## Real robot

Coordinate before touching. This is the H1-2; the G1 is under
[The G1](#the-g1). The onboard PC runs natively, without Docker:

```bash
export ROS_DOMAIN_ID=0
source core_ws/install/setup.bash
ros2 launch golem_bringup h12_real_robot.launch.py start_position_verified:=true
```

A native build there leaves out the G1's packages, as `launch_ros.sh` does
for the H1-2: add `--packages-ignore-regex '^g1_'` to its `colcon build`.
`golem_bringup` exec_depends on the `g1_*` packages, so a
`--packages-up-to golem_bringup` without it builds them too, and they need
`third_party/leprechaun_g1`.

Leg commands are gated behind `start_position_verified:=true` (default false) —
set it only after physically verifying the robot's pose. The `lowerbody:=` policy
(default `almi`) then ramps the legs to its pre-pose crouch and holds until

```bash
ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger
```

The companion desktop runs `h12_real_desktop.launch.py` for the model
servers, skills and rviz; add `use_hand_cameras:=true` to also start the hand
cameras.

## The G1

The wrappers run the Unitree G1 instead of the H1-2 with `--g1` anywhere in
their arguments, or with `GOLEM_ROBOT=g1` in the shell or `docker/.env`
(`--h12` overrides that). Each prints the robot it selected (`robot: g1`)
before the DDS domain check. For the G1 they add the
`docker/docker-compose.g1.yml` overlay, `launch_ros.sh` builds the
`core_ws/src/g1/` packages, and `G1_DDS_INTERFACE` names the host port wired to
the robot (empty for the sim). The G1 runs on x86 Linux with RoboCasa only.
Both robots use the same container names and DDS topics: run one at a time.

```bash
docker/scripts/docker_stack.sh up robocasa --g1 --headless   # sim + g1_sim.launch.py

docker/scripts/docker_run.sh robocasa --g1 --layout 9 --style 9 --seed 42
docker/scripts/docker_run.sh ros --g1
ros2 launch golem_bringup g1_sim.launch.py                   # use_nav:=true frontier_explore:=true ...
```

The G1 bringup always starts HOMIE's walking node in the sim. The robot hangs
from the sim's elastic band with every motor off until the arm controller's
first command; from then on the controller drives the waist and arms, and the
leg motors stay off until the walking node moves the legs to its default pose.
The walking node then starts its policy, and releases the band once
`frame_task_server` has finished its start-up moves.

The paired startup refuses a simulator of the other robot on the domain; the
single-service flow does not check. On the real G1 the `ros` container runs on
the laptop wired to the robot, on domain 0, which needs `G1_DDS_INTERFACE`
(`UP`, with an IPv4 address) before the confirmation prompt, and the robot in
Unitree's debug mode, on a gantry. Start at [`g1/index.md`](g1/index.md); the
first real session follows [`g1/first-session.md`](g1/first-session.md).

## Troubleshooting

- **Start RoboCasa before bringup.** If you restart the sim container, restart
  `golem_ros` too: sim time resets to 0, giving TF extrapolation errors and a
  frozen SLAM map.
- **Restart the sim containers after pulling.** The sims run the checkout's
  `golem_robocasa` and `CL_isaaclab_sim` from bind mounts, so a sim container keeps
  running the code it started with. An H1-2 sim started before the H1-2 sims
  reported `mode_machine` 6 (the real robot's value) is refused: the paired
  startup stops at `the simulator is not a h12 (mode_machine 0)`, and in the
  three-terminal flow `golem_safety` never engages (`engage_refused`:
  `/lowstate mode_machine is 0, expected 6`), so `frame_task_server` exits
  after 30 s. Restart the sim (`docker_stack.sh restart`, or `docker_run.sh
  <sim> --restart`); for Isaac run `git submodule update` first, since the
  change is in the `CL_isaaclab_sim` pin.
- **Stale installs of removed packages.** colcon never removes a package that
  is gone from `core_ws/src`. A workspace built before `golem_bringup` replaced
  `h12_bringup` and `g1_bringup` (and `h12_safety_layer`,
  `h12_ros2_controller`, `g1_safety_layer` and `g1_control` were deleted) keeps
  their build and install directories, and `launch_ros.sh` then stops with
  `stale installed package '<pkg>' has no discovered source`. Delete them
  once, from the repo root (`sudo` for files a container left root-owned):

  ```bash
  rm -rf {core_ws/build,core_ws/install,core_ws/build_stack_tests,core_ws/install_stack_tests,container_cache/g1_tests/build,container_cache/g1_tests/install}/{h12_bringup,g1_bringup,h12_safety_layer,h12_ros2_controller,g1_safety_layer,g1_control}
  ```

  A native workspace (the H1-2's onboard PC) has no such check and keeps
  answering to the old names until they are deleted. A `--symlink-install`
  workspace also keeps links to four deleted G1 config files, which fail the
  next `golem_upper_ik` build (`can't copy ... doesn't exist or not a regular
  file`); the change report,
  `docs/superpowers/specs/2026-10-28-golem-bringup-and-g1-debug-mode/change-report.md`,
  has their cleanup and the Mac's.
- **X11 / GUI:** set `XAUTHORITY` to your session's existing authorization file.
  Use `GOLEM_DISPLAY=headless` or `--headless` when no X server is available.
- **Talking to the sim from the host** bypasses the run scripts and so misses
  `docker/.env` — run `set -a; source docker/.env; set +a` first, or topics stay
  invisible.
- **Rebuilds are incremental** across container restarts (build/install are
  host-mounted). Force one with `colcon build --symlink-install` inside the
  container; for a clean message rebuild, wipe `container_cache/msgs_ws/` on the
  host.
- **MJPC C++ iteration** uses `docker exec -it golem_ros
  /home/code/h12_sim_scripts/rebuild_mjpc.sh` (`--install` also refreshes assets
  and proto), not colcon.
- **Laggy point clouds or images** are usually kernel UDP buffer drops — see
  [`DDS_TUNING.md`](DDS_TUNING.md).
