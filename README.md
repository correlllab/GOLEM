# GOLEM

**G**eneralized **O**pen **L**ayered **E**mbodied **M**odules — the software
stack for the Correll Lab's Unitree H1-2 humanoid.

Two simulators (MuJoCo/RoboCasa and Isaac Lab) and a ROS 2 Humble workspace run
in separate Docker containers sharing one DDS domain. The workspace holds the
drivers, upper-body IK, lower-body MPC and RL locomotion, a safety layer,
perception model servers, and an LLM-callable skills layer. Sim and real robot
speak the same Unitree `rt/lowcmd` / `rt/lowstate` wire format, so code moves
between them unchanged.

## Quick start

x86 Linux with an NVIDIA GPU is the primary platform. Full detail in
[`docs/SETUP.md`](docs/SETUP.md) and [`docs/RUNNING.md`](docs/RUNNING.md).

```bash
git lfs install && git submodule update --init --recursive
cp docker/.env.example docker/.env        # set GEMINI_API_KEY and ROS_DOMAIN_ID (never 0)
docker/scripts/docker_build.sh robocasa ros
```

Then, in two terminals:

```bash
docker/scripts/docker_run.sh robocasa                # 1. simulator (start first)

docker/scripts/docker_run.sh ros                     # 2. workspace shell
ros2 launch h1_bringup h1_sim_bringup.launch.py      #    bringup is a manual step
```

And in a third, ask the robot to do something:

```bash
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" --feedback
```

On Apple Silicon there is a CPU-only port that runs ROS 2 + MuJoCo without
Isaac — see [`docs/MACOS.md`](docs/MACOS.md).

## Layout

| Path | What it is |
|---|---|
| `core_ws/` | the ROS 2 workspace — 19 packages, where most development happens |
| `h1_robocasa/` | RoboCasa/MuJoCo simulator entry point and its ROS/DDS bridges |
| `CL_isaaclab_sim/` | Isaac Lab simulator |
| `CL_Assets/` | URDF meshes, MuJoCo XML, Isaac USD (Git-LFS) |
| `docker/` | x86 images, compose file, and the build/run scripts |
| `docker/mac/` | the self-contained Apple-Silicon port |
| `mujoco_mpc/`, `unitree_sdk2_python/` | vendored upstream dependencies |
| `tools/` | standalone debugging tools |

Most of the tree is git submodules, and large assets are Git-LFS.

## Documentation

| Document | Covers |
|---|---|
| [`docs/SETUP.md`](docs/SETUP.md) | prerequisites, `.env`, model weights, building the images |
| [`docs/RUNNING.md`](docs/RUNNING.md) | run flows, bringup, skills, real robot, troubleshooting |
| [`docs/MACOS.md`](docs/MACOS.md) | the Apple-Silicon CPU port and its feature toggles |
| [`docs/NAVIGATION_DEMO.md`](docs/NAVIGATION_DEMO.md) | autonomous SLAM + nav2 frontier exploration |
| [`docs/ROS_MCP_DEBUG.md`](docs/ROS_MCP_DEBUG.md) | the in-container ROS debugging MCP server |
| [`docs/DDS_TUNING.md`](docs/DDS_TUNING.md) | kernel and CycloneDDS tuning for the sensor streams |
| [`docker/BUILD.md`](docker/BUILD.md) | how the images are layered and what is pinned |
| [`CLAUDE.md`](CLAUDE.md) | orientation for coding agents: packages, conventions, invariants |

Individual packages document themselves — start with the package's own
`README.md` and source.
