# GOLEM

**G**eneralized **O**pen **L**ayered **E**mbodied **M**odules — the software
stack for the Correll Lab's Unitree humanoids: the H1-2 and, beside it, the G1
(see [The G1](#the-g1)).

Two simulators (MuJoCo/RoboCasa and Isaac Lab) and a ROS 2 Humble workspace run
in separate Docker containers sharing one DDS domain. The workspace holds the
drivers, upper-body IK, lower-body RL locomotion, a safety layer, SLAM and
navigation, perception model servers, and an LLM-callable skills layer. Sim and
real robot speak the same Unitree `rt/lowcmd` / `rt/lowstate` wire format, so
code moves between them unchanged.

## Quick start

The stack runs on x86 Linux with an NVIDIA GPU. Full detail in
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
ros2 launch golem_bringup h12_sim.launch.py          #    bringup is a manual step
```

The sim runs the real robot's operator gates. The legs hold their pre-pose until
you confirm them, and ENTER in terminal 1 (or in the sim's viewer) presses the
estop; after an estop, restart the whole bringup. In a third terminal, confirm
the legs and ask the robot to do something:

```bash
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" --feedback
```

## Packages

The workspace is `core_ws/src`. Both robots run the same `golem_*` packages,
each configured per robot.

| Package | What it does |
|---|---|
| `golem_bringup` | Launch files that start the whole robot, sim or real ([`CLAUDE.md`](CLAUDE.md) §6) |
| `golem_safety` | The only writer of the motor command topic: relays, clips and checks every command, latches the estop |
| `golem_upper_ik` | Arm IK controller (`/grasp_frame_target`, the arm interface every caller uses; also `/frame_task`, `/named_config`) with planning and self-collision avoidance |
| `golem_lowerbody_rl` | Leg controller running the walking policies (ALMI, FAME and walk on the H1-2; HOMIE on the G1) |
| `golem_slam` | FAST-LIO odometry, slam_toolbox mapping and Nav2 navigation |
| `golem_skills` | The `/skill/*` actions an LLM can call (grasp, pick and place, frontier exploration) |
| `golem_ros2_models` | Both robots' URDFs, meshes and self-collision models for ROS |
| `model_server` | Gemini, SAM3, GraspGenX and YOLO behind ROS services |
| `lmg_deploy` | The loco-manipulation game's whole-body MARL policy |
| `dex_3` | The G1's Dex3-1 hands |

Each package's `README.md` documents it; [`CLAUDE.md`](CLAUDE.md) §4 lists the
vendored and submodule packages beside them. The RoboCasa sim for both robots is
`golem_robocasa/`, the Isaac Lab sim (H1-2) `CL_isaaclab_sim/`.

## The G1

GOLEM also runs the Unitree G1 (29 DoF, two Dex3-1 hands): its own package
(`dex_3`, the hands), the packages it shares with the H1-2 above, its models in
`CL_Assets/g1`, and a RoboCasa sim in `golem_robocasa/g1/`. The H1-2 stays the
default; `--g1` on a wrapper, or `GOLEM_ROBOT=g1`, selects the G1 (RoboCasa
only). The images are the same, and every `ros` container builds and runs both
robots' packages.

```bash
docker/scripts/docker_stack.sh up robocasa --g1       # the G1 sim and its bringup (the arms and HOMIE)
tests/g1/test_sim_bringup.sh                          # the G1 acceptance test (arms); see tests/g1/README.md
```

On the real G1 the whole stack runs in Docker on a laptop wired to the robot,
with the robot in Unitree's debug mode. Nothing has run on it yet; the first
session follows [`docs/g1/first-session.md`](docs/g1/first-session.md). Start
at [`docs/g1/index.md`](docs/g1/index.md).

**Licence note.** HOMIE's walking policy in
`core_ws/src/golem_lowerbody_rl/policies/homie/` (from
[OpenHomie](https://github.com/InternRobotics/OpenHomie)) is for
non-commercial use only. Its `LICENSE` is CC BY-NC 4.0, but OpenHomie's README
says CC BY-NC-SA 4.0, which also requires changed versions to be shared under
the same licence: treat it as CC BY-NC-SA 4.0. The packages around it
(`golem_lowerbody_rl`, which runs it, and `dex_3`) are MIT. See the policy's
`README.md`.

## Tests

The closed-loop test runs a robot's whole stack in the RoboCasa sim: it stands
unsupported, walks to a Nav2 goal while moving its arms, keeps SLAM healthy, and
`golem_safety` stays the only `/lowcmd` writer with no estop.

```bash
tests/sim/test_sim_stack.sh --h12
tests/sim/test_sim_stack.sh --g1
```

[`tests/README.md`](tests/README.md) covers the rest: the sim unit tests, the
G1 acceptance tests and the Docker script tests. Each package's own tests run
with `colcon test` in the `ros` container.

## Layout and documentation

The repo layout is in [`CLAUDE.md`](CLAUDE.md) §2 (most of the tree is git
submodules, and large assets are Git-LFS). The user pages are in
[`docs/`](docs/index.md), and [`docker/BUILD.md`](docker/BUILD.md) covers how
the images are layered and what is pinned.
