# GOLEM

**G**eneralized **O**pen **L**ayered **E**mbodied **M**odules — the software
stack for the Correll Lab's Unitree humanoids: the H1-2 and, beside it, the G1
(see [The G1](#the-g1)).

Two simulators (MuJoCo/RoboCasa and Isaac Lab) and a ROS 2 Humble workspace run
in separate Docker containers sharing one DDS domain. The workspace holds the
drivers, upper-body IK, lower-body RL locomotion, a safety layer,
perception model servers, and an LLM-callable skills layer. Sim and real robot
speak the same Unitree `rt/lowcmd` / `rt/lowstate` wire format, so code moves
between them unchanged.

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

And in a third, ask the robot to do something:

```bash
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" --feedback
```

## The G1

GOLEM also runs the Unitree G1 (29 DoF, two Dex3-1 hands): its own package
(`core_ws/src/dex_3`, the Dex3-1 hands), the safety layer, arm IK, leg
controller (HOMIE's walking policy on the G1), models, bringup,
SLAM/navigation and skills packages it shares with the H1-2
(`core_ws/src/golem_safety`, `core_ws/src/golem_upper_ik`,
`core_ws/src/golem_lowerbody_rl`, `core_ws/src/golem_ros2_models`,
`core_ws/src/golem_bringup`, `core_ws/src/golem_slam`,
`core_ws/src/golem_skills`), its models in
`CL_Assets/g1`, and a RoboCasa sim in
`golem_robocasa/g1/`. The H1-2 stays the default; `--g1` on a wrapper, or
`GOLEM_ROBOT=g1`, selects the G1 (RoboCasa only). The images are
the same, and every `ros` container builds and runs both robots' packages.

```bash
tests/g1/test_sim_bringup.sh                          # the G1 acceptance test (arms); see tests/g1/README.md
tests/sim/test_sim_stack.sh --g1                      # the closed-loop test of either robot; see tests/sim/README.md
docker/scripts/docker_stack.sh up robocasa --g1       # the G1 sim and its bringup (the arms and HOMIE)
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

## Layout and documentation

The repo layout is in [`CLAUDE.md`](CLAUDE.md) §2 (most of the tree is git
submodules, and large assets are Git-LFS). The user pages are in
[`docs/`](docs/index.md); [`tests/README.md`](tests/README.md) covers the
simulator regression and acceptance tests, and
[`docker/BUILD.md`](docker/BUILD.md) how the images are layered and what is
pinned.

Individual packages document themselves — start with the package's own
`README.md` and source.
