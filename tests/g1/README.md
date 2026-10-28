# G1 acceptance tests

End-to-end tests of the G1 in the RoboCasa sim with its whole bringup (`golem_bringup`). Each one
builds `core_ws` for the G1, starts the sim headless in the layout 9 / style 9 / seed 42 kitchen,
runs `g1_sim.launch.py` and the test's steps beside it, and then checks the sim's own
recording of the run.

```bash
tests/g1/test_sim_bringup.sh                 # the arm test
tests/g1/test_sim_bringup.sh --walk          # the walk test (use_locomotion:=true)
tests/g1/test_sim_nav.sh                     # the nav test, FAST_LIO odometry
GOLEM_SIM_ODOM=1 tests/g1/test_sim_nav.sh    # the nav test, the sim's ground-truth odometry
tests/g1/test_sim_explore.sh                 # frontier exploration of the kitchen
```

A test that passes ends with `PASSED: <n>/<n> checks` and exits 0. A failing check prints `FAIL`
with its reason, and the last line reads `FAILED: <passed>/<total> checks`. Each script's header
lists its environment knobs, and each `test_sim_*.py` docstring says what it checks.

- **Arm test**: `rt/arm_sdk` engages, the arms reach `home` and then a `t_pose` goal,
  `/left/hand/close` closes the left hand, stopping the controller hands the arms back, and the
  robot keeps standing. The bringup runs `golem_safety` (`sim_arm_sdk`) and `golem_upper_ik`
  (`sim`). The controller's last command has zero gains, which `golem_safety` refuses in
  arm_sdk mode (`arm_sdk.waist_kp_min`), so the stop latches its estop and the weight ramps
  1 -> 0; the test finds the controller by its process, `lib/golem_upper_ik/frame_task_server`.
- **Walk test**: the stand-in hands the legs to the walking node, the robot keeps standing, then
  walks 5 s at 0.3 m/s. Then `frame_task_server` alone is stopped while the walking node
  balances the robot: its shutdown command keeps the waist at its hold, so the waist roll and
  pitch stay within 0.1 rad of it for 4 s (a limp waist slumps about 0.5 rad) and the robot
  keeps standing.
- **Arm and walk tests**: every sensor topic arrives at its rate with the right frames, and the
  sim's sensor node publishes nothing else.
- **Nav test**: Nav2 walks the robot to one goal 2-3 m from the spawn, picked from the sim's own
  geometry (or `NAV_GOAL`). The goal must succeed, the robot must end near it by the sim's
  truth, stay up and touch nothing, and `/cmd_vel`, `/scan`, the map and the localisation must
  stay within their limits.
- **Explore test**: the frontier explorer walks the robot until it reports the kitchen explored.
  At least 95 % of the reachable floor must be free in the final map, with no phantom
  obstacles, and the robot must stay up and touch nothing.

## How they run

They need the `golem_sim_robocasa` and `golem_ros` images (`docker/scripts/docker_build.sh
robocasa ros`); they never build them.

- **Containers.** Both containers are the compose services with the G1 overlay, as
  `docker_run.sh --g1` starts them, plus a `test_compose.yml` that `common.sh` writes into the
  output directory. The sim container has no network, and the ros container joins its network
  namespace. Both use its loopback with multicast, so no DDS leaves the host and nothing
  talks to the robot. The ros container loads the run's own `cyclonedds_local.xml` (no
  interface pin) from the output directory, never the shared one in `container_cache/`,
  which a running G1 `ros` container rereads in every process it starts.
- **DDS domains.** The arm test uses 195, the walk test 196, the nav test 197 and the explore
  test 198, unless `ROS_DOMAIN_ID` is set in the shell. Domain 0 is refused.
- **`core_ws`.** Before the sim starts, the ros service builds `core_ws` for the G1
  (`launch_ros.sh` with `GOLEM_ROBOT=g1`) into `container_cache/g1_tests/{build,install,log}`,
  so the checkout's own `core_ws/build` and `core_ws/install` are left alone. The first run
  builds the whole workspace; later runs rebuild what changed. `CORE_WS=<dir>` builds and uses
  that workspace in place instead. The tests share that build, so one runs at a time.
- **Sim.** The sim role of `test_sim_*.py` runs `golem_robocasa/h12_mujoco.py --g1` through
  `runpy`, with a recorder patched onto `g1.unitree_interface.SimInterface`. The recorder reads
  the robot's state inside the sim process, so no ground truth is published. The only
  exception is `/odom` for the nav and explore tests with `GOLEM_SIM_ODOM=1`.
- **Knobs.** The sim's knobs come from the shell or the scripts, never from `docker/.env`.

With a warm build each test takes one to two minutes; the first run also builds the whole
workspace.

Open issue: the arm test's `t_pose` check sometimes fails with the arms at the pose
(`goal success False` with a small arm error). The arms hold `t_pose` with an error about as
large as `frame_task_server`'s success threshold, `controller.threshold_joint` in
`core_ws/src/golem_upper_ik/config/g1/sim.yaml`, so the server sometimes never reports the goal
reached. Count such a run as failed, and check its arm error against that threshold.

The tests depend on internals of the G1 sim. They patch `SimInterface.after_step` (and, for nav
and explore, `__init__`) and read its `_lowcmd`, `_arm_sdk`, `arm_sdk_w`, `released` and
`resolver`. They also rely on the sensor node's name `mujoco_sensors` and on the sim's
`ROS bridges up` and `ROBOT FELL` log lines. A change to any of these must update the tests in
the same commit.

## Output

Each run writes to `tests/results/g1-<test>-<time>-<pid>/`, where `<test>` is `bringup`, `walk`,
`nav-fastlio`, `nav-sim_gt`, `explore-fastlio` or `explore-sim_gt`. The directory holds:

- `run.json`: the exit codes, the domain, the commit, the submodules, the working tree and the
  image IDs.
- `report.log` and `report.json`: the checks and their numbers.
- `sim.log`, `launch.log`, `drive.log` and `ros_build.log`.
- the recordings: `sim.npz`, `events.json` and, for nav and explore, `drive.npz`, `map.npz`,
  `scene.npz`, `goal.json` and `contacts.json`.

When a test fails, keep the whole directory.
