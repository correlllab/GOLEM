# GOLEM

**G**eneralized **O**pen **L**ayered **E**mbodied **M**odules — the software
stack for the Correll Lab's Unitree humanoids: the H1-2 and, beside it, the G1
(see [The G1](#the-g1)).

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
ros2 launch h12_bringup h12_sim_bringup.launch.py      #    bringup is a manual step
```

And in a third, ask the robot to do something:

```bash
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" --feedback
```

On Apple Silicon there is a CPU-only port that runs ROS 2 + MuJoCo without
Isaac (H1-2 only) — see [`docs/MACOS.md`](docs/MACOS.md).

## The G1

GOLEM also runs the Unitree G1 (29 DoF, two Dex3-1 hands): its own packages in
`core_ws/src/g1/` (HOMIE walking policy, Dex3-1 hands, navigation, bringup,
description), the safety layer and arm IK it shares with the H1-2
(`core_ws/src/golem_safety`, `core_ws/src/golem_upper_ik`), its robot
description in the `third_party/leprechaun_g1` submodule, and a RoboCasa sim in
`golem_robocasa/g1/`. The H1-2 stays the default;
`--g1` on a wrapper, or `GOLEM_ROBOT=g1`, selects the G1 (x86 Linux, RoboCasa
only). The images are the same.

```bash
tests/g1/test_sim_bringup.sh                          # the G1 acceptance test (arms); see tests/g1/README.md
docker/scripts/docker_stack.sh up robocasa --g1       # the G1 sim and its arms bringup
```

On the real G1 the whole stack runs in Docker on a laptop wired to the robot.
Nothing has run on it yet; the first session follows
[`docs/g1/first-session.md`](docs/g1/first-session.md). Start at
[`docs/g1/index.md`](docs/g1/index.md).

**Licence note.** HOMIE's walking policy in
`core_ws/src/g1/g1_locomotion/policy/` (from
[OpenHomie](https://github.com/InternRobotics/OpenHomie)) is for
non-commercial use only. Its `LICENSE` is CC BY-NC 4.0, but OpenHomie's README
says CC BY-NC-SA 4.0, which also requires changed versions to be shared under
the same licence: treat it as CC BY-NC-SA 4.0. The `g1_*` packages around it
are MIT. See the policy's `README.md`.

## Layout

| Path | What it is |
|---|---|
| `core_ws/` | the ROS 2 workspace — 28 packages, the G1's in `core_ws/src/g1/`; where most development happens |
| `golem_robocasa/` | RoboCasa/MuJoCo simulator entry point for both robots and its ROS/DDS bridges; the G1 sim in `golem_robocasa/g1/` |
| `CL_isaaclab_sim/` | Isaac Lab simulator (H1-2) |
| `CL_Assets/` | URDF meshes, MuJoCo XML, Isaac USD (Git-LFS) |
| `docker/` | x86 images, compose files (with the G1 overlay), and the build/run scripts |
| `docker/mac/` | the self-contained Apple-Silicon port |
| `mujoco_mpc/`, `unitree_sdk2_python/`, `third_party/leprechaun_g1` | vendored upstream dependencies and the G1 robot description |
| `tests/` | simulator regression and acceptance tests |
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
| [`docs/DOCKER_TESTING.md`](docs/DOCKER_TESTING.md) | the Docker smoke tests on Ubuntu NVIDIA and Apple-Silicon Mac |
| [`docs/g1/`](docs/g1/index.md) | the G1: sim flows, safety layer, DDS, walking, navigation, exploration, the first real session |
| [`tests/README.md`](tests/README.md) | the simulator regression and acceptance tests |
| [`docker/BUILD.md`](docker/BUILD.md) | how the images are layered and what is pinned |
| [`CLAUDE.md`](CLAUDE.md) | orientation for coding agents: packages, conventions, invariants |

Individual packages document themselves — start with the package's own
`README.md` and source.
