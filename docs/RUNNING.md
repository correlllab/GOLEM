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

Use `isaac` instead of `robocasa` for Isaac. `--gui` selects desktop mounts;
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
ros2 launch h1_bringup h1_sim_bringup.launch.py  # bringup is a manual step

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
```

Container names are `golem_sim_robocasa`, `golem_sim_isaac`, and `golem_ros`.
The default launchers are `docker/scripts/launch_{robocasa,isaac,ros}.sh`.

**The single-service `ros` default is a shell.** Paired startup launches
bringup automatically. To run an initialized custom command, invoke
`docker_run.sh ros /home/code/h12_sim_scripts/launch_ros.sh <cmd> <args>`.
Existing containers require an explicit `--restart` immediately after the
service name; otherwise attach with `docker exec`.

### RoboCasa

Publishes `rt/lowstate` over CycloneDDS, and on the chosen ROS domain: `/clock`;
the RealSense topics `/realsense/{head,left_hand,right_hand}/color/image_raw` and
`.../aligned_depth_to_color/image_raw` (raw and compressed) with
`.../color/camera_info`; the Livox topics `/livox/lidar` (CustomMsg),
`/livox/pointcloud`, `/livox/imu`; and `/{left,right}/gripper/state`. It consumes
`rt/lowcmd` and the `/{left,right}/gripper/*` services.

Scene flags are passed straight through, e.g.
`docker_run.sh robocasa --task OpenFridge --layout 1 --style 1 --seed 42`.

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

`h1_bringup` is launch-only, and sim versus real are separate launch files (there
is no `sim:=true` argument):

| Launch file | Scenario |
|---|---|
| `h1_sim_bringup.launch.py` | x86 sim — the full stack |
| `h1_sim_bringup_mac.launch.py` | mac sim — trimmed, toggled by `GOLEM_*` env vars |
| `h1_real_robot_bringup.launch.py` | real robot, onboard PC (aggregates the three below) |
| `h1_real_drivers.launch.py` | real: Livox, RealSense, grippers |
| `h1_real_controller.launch.py` | real: estop, state publishers, safety + IK servers |
| `h1_real_desktop_bringup.launch.py` | real: companion desktop — model servers, skills, MJPC |
| `h1_navigation.launch.py` | shared nav stack, included by the sim and real bringups |

The x86 sim bringup starts the state publishers, the `frame_task_server` IK
solver, `safety_node`, the Gemini/SAM/GraspGen model servers, the MJPC estimator
and lower-body controller, the skills node, nav, and RViz. Common arguments:

```bash
ros2 launch h1_bringup h1_sim_bringup.launch.py \
  use_rviz:=false use_nav:=false use_skills:=false lowerbody:=fame
```

## Worked example — open the fridge

```bash
# Terminal 1
./docker/scripts/docker_run.sh robocasa --task OpenFridge --layout 1 --style 1 --seed 42

# Terminal 2
docker/scripts/docker_run.sh ros
ros2 launch h1_bringup h1_sim_bringup.launch.py

# Terminal 3 (inside golem_ros, workspace sourced)
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'home', duration: {sec: 0, nanosec: 0}}"

ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" \
  --feedback
```

`/skill/grasp`, `/skill/pick_place`, and `/skill/frontier_explore` are
implemented; the other nine `/skill/*` action servers accept a goal and then
abort with "not implemented". `core_ws/src/h1_bringup/common_cmds.md` has a
ready-made `send_goal` snippet and field reference for each one.

## Real robot

Coordinate before touching. The onboard PC runs natively, without Docker:

```bash
export ROS_DOMAIN_ID=0
source core_ws/install/setup.bash
ros2 launch h1_bringup h1_real_robot_bringup.launch.py
```

On the companion desktop, `h1_real_desktop_bringup.launch.py` gates leg commands
behind `start_position_verified:=true` (default false) — set it only after
physically verifying the robot's pose.

## Troubleshooting

- **Start RoboCasa before bringup.** If you restart the sim container, restart
  `golem_ros` too: sim time resets to 0, giving TF extrapolation errors and a
  frozen SLAM map.
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
