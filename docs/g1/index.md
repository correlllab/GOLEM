# The Unitree G1 in GOLEM

GOLEM runs a second robot beside the H1-2: the Correll Lab's Unitree G1
(29 DoF, two Dex3-1 hands). Its ROS 2 packages are in `core_ws/src/g1/`, its
robot description is the `third_party/leprechaun_g1` submodule, and its
RoboCasa kitchen sim is `h1_robocasa/g1/`, started through the same entry point
as the H1-2's (`h12_mujoco.py --g1`). The same bringup runs against the sim or
the real robot, which speaks the same Unitree DDS interface.

On the real G1 every node runs in Docker on a laptop wired to the robot.
Nothing is installed on, or run from, the robot's own computers.

The G1 packages, sim, tests and these pages come from
[correlllab/G1_GOLEM](https://github.com/correlllab/G1_GOLEM).

## What works today

| Feature | In the sim | On the real robot |
|---|---|---|
| Arms (IK controller and safety layer) | Works | Ready for a first session; nothing has run yet |
| Dex3-1 hands | Works | In the real bringup (`use_hands`), off for the first session |
| Walking with HOMIE | Works | Not covered by these docs |
| Navigation (Nav2) | Works | Refused: the real MID-360 lidar is not connected to the bringup |
| Frontier exploration | Works | Needs navigation |

Not done yet: navigation on the real robot, a self-collision model for the
arms, manipulation skills, sensors and skills in the real bringup, and tuning
of gains and limits on our G1.

## Selecting the G1

The H1-2 is the default. The Linux wrappers in `docker/scripts/` select the G1
with `--g1` anywhere in their arguments, or with `GOLEM_ROBOT=g1` in the shell
or `docker/.env`; `--h12` selects the H1-2 again. `docker_run.sh` and
`docker_stack.sh up` print the robot they selected (`robot: g1`) before the
DDS domain check and before any container starts. For the G1 the wrappers add
`docker/docker-compose.g1.yml` to the compose files, and `launch_ros.sh` builds
the `g1_*` packages. `CLAUDE.md` §3 has the details.

- x86 Linux with an NVIDIA GPU only: the Mac wrappers and the Isaac simulator
  refuse the G1.
- Both robots use the same images (`docker/scripts/docker_build.sh robocasa ros`).
- The robots share container names (`golem_ros`, `golem_sim_robocasa`), node
  names and DDS topics. Run one robot at a time on a host, and never two robot
  stacks on one DDS domain.

## Try it in the sim

The acceptance tests need only the images; each builds `core_ws` for the G1 on
its own and runs with no network, so nothing reaches a robot:

```bash
tests/g1/test_sim_bringup.sh            # arms
tests/g1/test_sim_bringup.sh --walk     # walking
tests/g1/test_sim_nav.sh                # Nav2 walks the G1 to a goal
tests/g1/test_sim_explore.sh            # frontier exploration of the kitchen
```

A passing test ends with `PASSED: <n>/<n> checks` and exits 0.
`tests/g1/README.md` says what each one checks and where its output goes.

The sim and the arms bringup together, gated on a fresh `rt/lowstate` from a
G1 sim:

```bash
docker/scripts/docker_stack.sh up robocasa --g1 --headless
docker/scripts/docker_stack.sh logs robocasa
docker/scripts/docker_stack.sh stop robocasa
```

### Arms by hand

To send your own arm and hand goals, start the stack in three terminals:

```bash
# terminal 1: the sim with its viewer (add --headless for none)
docker/scripts/docker_run.sh robocasa --g1 --layout 9 --style 9 --seed 42
# wait for "[g1_mujoco] ROS bridges up"

# terminal 2: builds core_ws for the G1 if needed, then opens a shell
docker/scripts/docker_run.sh ros --g1
ros2 launch g1_bringup g1_sim_bringup.launch.py    # options: use_rviz:=false use_hands:=false

# terminal 3
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 't_pose', plan: false, duration: {sec: 0, nanosec: 0}}"
ros2 service call /left/hand/close std_srvs/srv/Trigger
ros2 service call /left/hand/open std_srvs/srv/Trigger
```

**What to expect:** the arms move to `home` once the stack is up, then to a
T-pose, and the left hand closes and opens. The sim's stand-in for Unitree's
controller (HOMIE's policy) keeps the legs. The named postures are in
`core_ws/src/g1/g1_control/g1_control/utility/named_config.py`.

- Start the sim first. If you restart the sim, restart the `ros` container
  too, because sim time starts again at 0.
- This flow has no check that the sim on the domain is a G1; `docker_stack.sh`
  has one (`docker/scripts/wait_for_sim.py`).
- These containers use host networking on domain 1, so their DDS traffic can
  reach other machines on your network. The test scripts keep it on the host.
- Leave `G1_DDS_INTERFACE` empty here. It pins only the `ros` container, so on
  the sim domain the two containers can end up on different interfaces; the
  wrappers warn when it is set ([DDS and networking](dds.md)).
- `plan: true` aborts: the arms have no self-collision model yet. So do
  `slow_mode: true` and a `FrameTask` with `relative: true`, which the G1
  controller does not implement.

## Pages

| Page | Covers |
|---|---|
| [First real session](first-session.md) | the arms-only checklist for the first session on the real G1 |
| [Safety layer](safety.md) | modes, the engage gate, estops, stopping, logs |
| [Control in sim and on the robot](control-path.md) | the path of a command, and how the sim stands in for the robot |
| [DDS and networking](dds.md) | domains, the guards against domain 0, the wired port |
| [Walking](walking.md) | HOMIE's policy on the legs, in the sim |
| [Navigation](navigation.md) | FAST_LIO, slam_toolbox and Nav2, in the sim |
| [Frontier exploration](exploration.md) | `/skill/frontier_explore`, in the sim |
| [Packages](packages.md) | the G1 packages, their READMEs and their tests |

## Licence

The `g1_*` packages are MIT. HOMIE's walking policy in
`core_ws/src/g1/g1_locomotion/policy/` is for non-commercial use only (CC BY-NC
4.0 by its `LICENSE`; treat it as CC BY-NC-SA 4.0); see its `README.md`.
