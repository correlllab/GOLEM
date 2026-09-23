# macOS (Apple Silicon) — headless CPU port

A self-contained, CPU-only port that runs **ROS 2 + MuJoCo/RoboCasa** on arm64.
Isaac is not available (it needs an NVIDIA GPU), MuJoCo renders in software
(OSMesa/llvmpipe), and the ROS layer runs on **FastDDS** instead of CycloneDDS
(an arm64 in-process coexistence requirement). It uses its own compose file,
`docker/mac/docker-compose.yml` — the x86 instructions in
[`RUNNING.md`](RUNNING.md) do not apply here.

## Prerequisites

- [Colima](https://github.com/abiosoft/colima) + the Docker CLI
  (`brew install colima docker docker-compose`). No Docker Desktop, no NVIDIA
  toolkit, no XQuartz.
- Git LFS and submodules, as in [`SETUP.md`](SETUP.md).
- `docker/.env` is optional. Both wrapper families load it, with exported shell
  settings taking precedence. Asset/cache mount roots are configurable there.
- Allocate VM resources according to the Mac's memory. For a 16 GB MacBook Air,
  a starting point is `colima start --cpu 4 --memory 6`. Start headless with
  cameras disabled; software rendering is CPU-intensive. Larger builds and
  simulations may need more RAM. On an 8 GB model, expect tighter limits.

## Build and run

```bash
# build both arm64 images (robocasa + ros); the base image builds automatically
docker/mac/scripts/docker_build_mac.sh          # or: docker_build_mac.sh robocasa | ros

# headless (the default); brings BOTH services up together
docker/mac/scripts/docker_stack_mac.sh up robocasa
```

The stack wrapper prepares both workspaces before starting physics, then starts
both services and waits for a fresh DDS state sample before ROS bringup. This
avoids leaving the simulator running during a long first controller build.
Follow logs with `docker/mac/scripts/docker_stack_mac.sh logs robocasa`.
Use `restart robocasa` to recreate both services together and `stop robocasa`
to stop them while preserving caches.

`docker/mac/scripts/docker_run_mac.sh [robocasa|ros]` remains available for
single-service runs and raw command overrides. An existing stable container
name requires explicit `--restart`; no running container is removed implicitly.

The first startup is slow (colcon builds the message + controller workspaces
into named volumes before either simulation or bringup starts); later runs are cached.
Images use `:arm64` tags, separate from Linux NVIDIA images. Both containers use `network_mode: host`
+ `ipc: host` so FastDDS's shared-memory transport works across them.

## Feature toggles

All are environment variables read by `docker/mac/docker-compose.yml`, so set
them on the `up` command line. They are baked in at container-create time —
changing one needs the stack `restart` command to recreate both containers.

| Variable | Default | Effect |
|---|---|---|
| `GOLEM_DISPLAY` | off | `vnc` → MuJoCo viewer on noVNC port 6080 |
| `GOLEM_RVIZ` | off | `vnc` → RViz on noVNC port 6081 |
| `GOLEM_LOWERBODY` | off (tethered) | `fame` stand-only, `walk` raw policy, `switch` auto stand↔walk |
| `GOLEM_SLAM` | off | `pointcloud_to_laserscan` + `slam_toolbox` → `/map` |
| `GOLEM_NAV2` | off | nav2 stack (implies SLAM) |
| `GOLEM_SIM_ODOM` | off | sim ground-truth `/odom` + `odom→pelvis` TF (SLAM needs it) |
| `GOLEM_CAMERAS` | on | `0` drops the 3 RGBD renders — the heaviest per-step cost |
| `GOLEM_SPAWN_BACKOFF` | 0 | metres to back the robot into open floor at spawn |
| `GOLEM_SPAWN_MARGIN` | 0.20 | minimum metres between the robot and any non-floor geom at spawn (also on x86) |
| `GOLEM_CMD_TIMEOUT` | 0.5 | sim-seconds before the low-level interface zeroes the motors |
| `GOLEM_ROS_MCP` | off | `1` starts the [ROS debugging MCP server](ROS_MCP_DEBUG.md) |

## Viewing the GUIs

MuJoCo's and RViz's OpenGL windows cannot be forwarded to XQuartz
(Apple-Silicon XQuartz has broken indirect GLX). Each is rendered in-container
with software GL into a virtual display and streamed to the browser over noVNC:

```bash
GOLEM_DISPLAY=vnc GOLEM_RVIZ=vnc docker/mac/scripts/docker_stack_mac.sh up robocasa

# Colima does not forward container ports — open the SSH tunnel
./docker/mac/scripts/mac_vnc_tunnel.sh     # --open opens a browser; --stop closes the tunnel
```

- **MuJoCo viewer** → <http://localhost:6080/vnc.html>
- **RViz** (RobotModel + TF) → <http://localhost:6081/vnc.html>

RoboCasa renders on X display `:99` and RViz on `:100`; they must differ because
the containers share one network namespace. The launchers handle this.

## Driving the robot

Bringup starts automatically in the `ros` container (`joint_state_publisher`,
`robot_state_publisher`, `frame_task_server`, `safety_node`):

```bash
docker exec -it golem_ros bash          # fallback if the host docker CLI is flaky: colima ssh -- docker exec …
source /home/code/h12_sim_scripts/robot_cli.sh

rob_poses                # list postures
rob_pose t_pose          # move to one
rob_grip right close     # open/close a gripper
rob_home
```

## Lower-body control

By default an elastic-band tether holds the robot upright and only the upper
body is controlled. `GOLEM_LOWERBODY` releases the tether and balances the robot:

```bash
# stand free (and squat via /lowerbody/squat_cmd), but no locomotion
GOLEM_DISPLAY=vnc GOLEM_LOWERBODY=fame docker/mac/scripts/docker_stack_mac.sh up robocasa

# stand AND walk — the switchable controller
GOLEM_DISPLAY=vnc GOLEM_LOWERBODY=switch docker/mac/scripts/docker_stack_mac.sh up robocasa
```

`switch` starts band-held idle and selects stand↔walk from the commanded
velocity: `rob_stand` settles it into FAME, and `rob_go <vx> <vy> <wz>` engages
the TorchScript walk policy — any non-zero `/cmd_vel` does, so there is no manual
handover step. The walk policy is stable only when handed over from a settled
FAME stance, which is what `switch` provides; `GOLEM_LOWERBODY=walk` launches the
raw policy directly and only marches in place.

## Demos

- **[Navigation demo](NAVIGATION_DEMO.md)** — SLAM + nav2 + autonomous frontier
  exploration of the RoboCasa kitchen.
- **[ROS debugging MCP server](ROS_MCP_DEBUG.md)** — `GOLEM_ROS_MCP=1` exposes
  ROS inspection and driving tools to Claude Code over the same SSH tunnel.

## Gotchas

- **The host `docker` CLI socket is intermittent** under Colima — `docker …` can
  fail with "Cannot connect to the Docker daemon" while the VM and containers are
  healthy. Use `colima ssh -- docker …`, or `colima stop && colima start` to
  relink the socket (this restarts the containers).
- **Cartesian reaching is disabled.** The `/frame_task` action crashes the
  controller node here, so `rob_reach` is a no-op stub — drive in joint space
  with `rob_pose`.
- **Restart the two containers coherently.** The RoboCasa container owns
  `/clock`; restarting it alone resets sim time to 0 while the ROS side keeps its
  old clock, giving TF extrapolation errors and a frozen `0×0` SLAM map. Restart
  `golem_ros` after `robocasa`.
- **The motor watchdog runs on sim time.** The low-level interface zeroes the
  motors if no `rt/lowcmd` arrives within `GOLEM_CMD_TIMEOUT` sim-seconds. It is
  sim time on purpose: a wall-clock timeout is far too tight on a sim running
  ~0.2× real-time. Raise it if a heavy scene trips it.
- **For nav, `GOLEM_SPAWN_BACKOFF` of `1.5`–`2.0`** keeps the robot clear of the
  counters.

For the MacBook Air smoke test and pass criteria, see [Docker testing](DOCKER_TESTING.md).
