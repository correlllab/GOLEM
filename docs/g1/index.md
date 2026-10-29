# The Unitree G1 in GOLEM

GOLEM runs a second robot beside the H1-2: the Correll Lab's Unitree G1
(29 DoF, two Dex3-1 hands). Its ROS 2 packages are in `core_ws/src/g1/`, its
models (Unitree's URDFs, MJCFs and meshes, and a sphere-packed self-collision
model) are in `CL_Assets/g1`, installed for ROS by `golem_ros2_models` with the
H1-2's, and its RoboCasa kitchen sim is `golem_robocasa/g1/`, started through the same entry point
as the H1-2's (`h12_mujoco.py --g1`). The same bringup runs against the sim or
the real robot, which speaks the same Unitree DDS interface.

On the real G1 every node runs in Docker on a laptop wired to the robot.
Nothing is installed on, or run from, the robot's own computers.

The G1 packages, sim, tests and these pages come from
[correlllab/G1_GOLEM](https://github.com/correlllab/G1_GOLEM).

## What works today

| Feature | In the sim | On the real robot |
|---|---|---|
| Arms (`golem_upper_ik` and `golem_safety`, shared with the H1-2) | Works | Written up for a first session, which waits for the debug-mode procedure; nothing has run yet |
| Dex3-1 hands | Works | In the real bringup (`use_hands`), off for the first session |
| Walking with HOMIE | Works | The pre-pose and the confirm are in the first session; walking is not covered by these docs |
| Navigation (Nav2) | Works | Refused: the real MID-360 lidar is not connected to the bringup |
| Frontier exploration | Works | Needs navigation |

Not done yet: the procedure for entering Unitree's debug mode on our G1,
navigation on the real robot, self-collision spheres on the Dex3-1 fingers,
manipulation skills, sensors and skills in the real bringup, and tuning of
gains and limits on our G1.

## Selecting the G1

The H1-2 is the default. The Linux wrappers in `docker/scripts/` select the G1
with `--g1` anywhere in their arguments, or with `GOLEM_ROBOT=g1` in the shell
or `docker/.env`; `--h12` selects the H1-2 again. `docker_run.sh` and
`docker_stack.sh up` print the robot they selected (`robot: g1`) before the
DDS domain check and before any container starts. The selection picks the
sim's robot, the bringup `docker_stack.sh` starts and the sim it waits for, and
on the real G1 the DDS interface pin (`docker/docker-compose.g1.yml`). Every
`ros` container mounts and builds both robots' packages, so `docker_run.sh ros`
needs `--g1` only on the real G1. `CLAUDE.md` §3 has the details.

- x86 Linux with an NVIDIA GPU only: the Mac wrappers and the Isaac simulator
  refuse the G1.
- Both robots use the same images (`docker/scripts/docker_build.sh robocasa ros`).
- The robots share container names (`golem_ros`, `golem_sim_robocasa`), node
  names and DDS topics. Run one robot at a time on a host, and never two robot
  stacks on one DDS domain.

## Try it in the sim

The acceptance tests need only the images; each builds `core_ws` on
its own and runs with no network, so nothing reaches a robot:

```bash
tests/g1/test_sim_bringup.sh            # arms
tests/g1/test_sim_bringup.sh --walk     # walking
tests/g1/test_sim_nav.sh                # Nav2 walks the G1 to a goal
tests/g1/test_sim_explore.sh            # frontier exploration of the kitchen
```

A passing test ends with `PASSED: <n>/<n> checks` and exits 0.
`tests/g1/README.md` says what each one checks and where its output goes.

The sim and the bringup together, gated on a fresh `rt/lowstate` from a G1
sim:

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

# terminal 2: builds core_ws (both robots' packages) if needed, then opens a shell
docker/scripts/docker_run.sh ros
ros2 launch golem_bringup g1_sim.launch.py    # options: use_rviz:=false use_hands:=false

# terminal 3
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 't_pose', plan: false, duration: {sec: 0, nanosec: 0}}"
ros2 service call /left/hand/close std_srvs/srv/Trigger
ros2 service call /left/hand/open std_srvs/srv/Trigger
```

**What to expect:** the robot starts hanging from the sim's elastic band with
every motor off, as the real G1 starts in Unitree's debug mode. The arms move
to `home` once the stack is up. HOMIE's walking node, which the sim bringup
always starts, moves the legs to its default pose, starts its policy and
releases the band (`elastic band released` in the bringup's log); from then on
it keeps the robot standing. The goals then move the arms to a T-pose, and the
left hand closes and opens. The named postures are `NAMED_CONFIGS` in
`core_ws/src/golem_upper_ik/golem_upper_ik/robots/g1.py`.

- Start the sim first. If you restart the sim, restart the `ros` container
  too, because sim time starts again at 0.
- This flow does not check that the sim on the domain is a G1 before the
  bringup starts; `docker_stack.sh` does (`docker/scripts/wait_for_sim.py`).
  `golem_safety` then engages on no `rt/lowstate` but a G1's (`mode_machine`
  5).
- These containers use host networking on domain 1, so their DDS traffic can
  reach other machines on your network. The test scripts keep it on the host.
- Leave `G1_DDS_INTERFACE` empty here. It pins only the `ros` container, so on
  the sim domain the two containers can end up on different interfaces; the
  wrappers warn when it is set ([DDS and networking](dds.md)).
- The arm controller keeps the arms out of self-collision and plans with
  OMPL (`plan: true`), as on the H1-2. Its spheres cover the Dex3-1 palms but
  not the fingers, and it keeps them 20 mm apart, as on the H1-2. `arms_down`
  rests the hands on the hips, inside the hip spheres, so a goal to it stops
  short and a planned one fails. `slow_mode: true` (a quarter of the speed) and
  a `FrameTask` with `relative: true` work as on the H1-2: a relative target is
  an offset in the `pelvis` frame (the frame of absolute targets and
  `/left_ee_pose`), whatever the IMU says: `{position: {x: 0.05}}` moves the
  hand 5 cm forward of the pelvis however the robot is turned or tilted.

## Pages

| Page | Covers |
|---|---|
| [First real session](first-session.md) | the checklist for the first session on the real G1, in Unitree's debug mode; it waits for the debug-mode procedure |
| [Safety layer](safety.md) | modes, the start-up, estops, stopping, logs |
| [Control in sim and on the robot](control-path.md) | the path of a command, and how the sim plays the robot in debug mode |
| [DDS and networking](dds.md) | domains, the guards against domain 0, the wired port, the robot's network and topics |
| [Our G1](hardware.md) | our unit's details, `rt/lowstate`'s layout, open hardware questions |
| [Walking](walking.md) | HOMIE's policy on the legs, in the sim |
| [Navigation](navigation.md) | FAST_LIO, slam_toolbox and Nav2, in the sim |
| [Frontier exploration](exploration.md) | `/skill/frontier_explore`, in the sim |
| [Packages](packages.md) | the G1 packages, their READMEs and their tests |

## Licence

The `g1_*` packages are MIT. HOMIE's walking policy in
`core_ws/src/g1/g1_locomotion/policy/` is for non-commercial use only (CC BY-NC
4.0 by its `LICENSE`; treat it as CC BY-NC-SA 4.0); see its `README.md`.
