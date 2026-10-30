# G1 acceptance tests

End-to-end tests of the G1 in the RoboCasa sim with its whole bringup (`golem_bringup`). Each one
builds `core_ws` for the G1, starts the sim headless in the layout 9 / style 9 / seed 42 kitchen,
runs `g1_sim.launch.py` and the test's steps beside it, and then checks the sim's own
recording of the run.

```bash
tests/g1/test_sim_bringup.sh                 # the arm test
tests/g1/test_sim_bringup.sh --walk          # the walk test
tests/g1/test_sim_nav.sh                     # the nav test, FAST_LIO odometry
tests/g1/test_sim_explore.sh                 # frontier exploration of the kitchen
```

A test that passes ends with `PASSED: <n>/<n> checks` and exits 0. A failing check prints `FAIL`
with its reason, and the last line reads `FAILED: <passed>/<total> checks`. Each script's header
lists its environment knobs, and each `test_sim_*.py` docstring says what it checks.

Every test starts the G1 as it starts in debug mode. The limp robot hangs from the sim's elastic
band while `golem_safety` streams zero-gain `rt/lowcmd`. HOMIE's walking node pre-poses the legs,
starts its policy under the band and releases the band once `frame_task_server` is up.

- **Arm and walk tests, the start-up**:
  - `band holds`: while the band is on, the pelvis tilts at most 15 deg;
  - `no estop during start-up`: `golem_safety` logs no estop from its start to the end of the
    test's steps;
  - `walking node starts`: the walking node logs `walking: policy started` once, and no fault;
  - `band released after the controller is ready`: the band comes off once and stays off,
    because `frame_task_server` is up: every toggle call the walking node logs gives
    `/left_ee_pose` as its reason, so a release by `band.max_wait` fails the check. The band is
    off after the first `/left_ee_pose` the test sees and within the walking node's
    `band.max_wait` of the policy's start;
  - `standing after release`: the robot stands for 5 s after the release.

  The report also prints the pelvis and the feet around the release, for the record.
- **Arm test**: after the release, the arms are at `home` from the controller's start-up, a
  `t_pose` goal succeeds and the arms reach it, and `/left/hand/close` closes the left hand. Then
  `frame_task_server` alone is stopped while the walking node balances the robot. Its shutdown
  command keeps the waist at its hold, so the waist roll and pitch stay within 0.1 rad of it
  for 4 s (a limp waist slumps about 0.5 rad), and the robot keeps standing. The test finds the
  controller by its process, `lib/golem_upper_ik/frame_task_server`.
- **Walk test**: 5 s after the release the robot walks 5 s at 0.3 m/s, and keeps standing. Then
  `frame_task_server` is stopped, with the arm test's waist check.
- **Arm and walk tests, the sensors**: every sensor topic arrives at its rate with the right
  frames, and the sim's sensor node publishes nothing else.
- **Nav test**: once the band is released, Nav2 walks the robot to one goal 2-3 m from the
  spawn. The goal is picked from the sim's own geometry (or `NAV_GOAL`).
  - The goal must succeed, and the robot must end near it by the sim's truth.
  - The robot must stay up and touch nothing, and `golem_safety` must latch no estop.
  - `/cmd_vel`, `/scan`, the map and the localisation must stay within their limits.
- **Explore test**: once the band is released, the frontier explorer walks the robot until it
  reports the kitchen explored.
  - At least 95 % of the reachable floor must be free in the final map, with no phantom
    obstacles.
  - The robot must stay up and touch nothing, and `golem_safety` must latch no estop.

## How they run

They need the `golem_sim_robocasa` and `golem_ros` images (`docker/scripts/docker_build.sh
robocasa ros`); they never build them.

- **Containers.** Both containers are the compose services as `docker_run.sh` starts them
  (`robocasa` with the G1 overlay, `ros` from `docker-compose.yml`), plus a `test_compose.yml`
  that `common.sh` writes into the output directory. The sim container has no network, and the
  ros container joins its network namespace. Both use its loopback with multicast, so no DDS
  leaves the host and nothing talks to the robot.
- **DDS domains.** The arm test uses 195, the walk test 196, the nav test 197 and the explore
  test 198, unless `ROS_DOMAIN_ID` is set in the shell. Domain 0 is refused.
- **`core_ws`.** Before the sim starts, the ros service builds `core_ws`, both robots'
  packages (`launch_ros.sh`), into `container_cache/g1_tests/{build,install,log}`,
  so the checkout's own `core_ws/build` and `core_ws/install` are left alone. The first run
  builds the whole workspace; later runs rebuild what changed. `CORE_WS=<dir>` builds and uses
  that workspace in place instead. The tests share that build, so one runs at a time.
- **Sim.** The sim role of `test_sim_*.py` runs `golem_robocasa/h12_mujoco.py --g1` through
  `runpy`, with a recorder patched onto `g1.unitree_interface.SimInterface`. The recorder reads
  the robot's state inside the sim process, so no ground truth is published.
- **Knobs.** The sim's knobs come from the shell or the scripts, never from `docker/.env`.

With a warm build each test takes one to two minutes; the first run also builds the whole
workspace.

Open issue: the arm test's `t_pose` check sometimes fails with the arms at the pose
(`goal success False` with a small arm error). The arms hold `t_pose` with an error about as
large as `frame_task_server`'s success threshold, `controller.threshold_joint` in
`core_ws/src/golem_upper_ik/config/g1/sim_split.yaml`, so the server sometimes never reports the goal
reached. Count such a run as failed, and check its arm error against that threshold.

The tests depend on internals of the G1 sim and on the stack's log lines. A change to any of these
must update the tests in the same commit.

- They patch `SimInterface.after_step` (and, for nav and explore, `__init__`).
- They read its `_lowcmd`, `commanded`, `band` (the band's `enabled`, every physics step) and
  `resolver`.
- They rely on the sensor node's name `mujoco_sensors`.
- They rely on the sim's `ROS bridges up`, `elastic band released at t=` and `ROBOT FELL` log lines.
- From `launch.log` they rely on:
  - the walking node's `Loaded <config> and the policy` and `walking: policy started` lines,
    its `releasing the elastic band: <reason>` and `elastic band released` lines, its `fault:`
    lines, and its config's `band.max_wait`;
  - `golem_safety`'s JSON events `relay_started` and `estop_command_written`, and its
    `estop diagnostic` lines.

## Output

Each run writes to `tests/results/g1-<test>-<time>-<pid>/`, where `<test>` is `bringup`, `walk`,
`nav-fastlio` or `explore-fastlio`. The directory holds:

- `run.json`: the exit codes, the domain, the commit, the submodules, the working tree and the
  image IDs.
- `report.log` and `report.json`: the checks and their numbers.
- `sim.log`, `launch.log`, `drive.log` and `ros_build.log`.
- the recordings: `sim.npz`, `events.json` and, for nav and explore, `drive.npz`, `map.npz`,
  `scene.npz`, `goal.json` and `contacts.json`.

Each test stops the sim (`stop` file) and waits for its recording before it stops the bringup,
so the recordings end with the robot standing, not with it falling once `rt/lowcmd` is silent.
The report cuts a recording at the first sim-time reversal (MuJoCo resetting the scene): a reset
before the test's `end` mark fails the run, one after it is noted.

When a test fails, keep the whole directory.
