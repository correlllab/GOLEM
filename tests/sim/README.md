# Closed-loop sim tests of both robots

One test, the same script for both robots, runs the robot's whole sim bringup with navigation in
the RoboCasa sim and checks, in one run, that the robot stands, walks to a goal, moves its arms
and keeps SLAM healthy.

```bash
tests/sim/test_sim_stack.sh --h12
tests/sim/test_sim_stack.sh --g1
```

A test that passes ends with `PASSED: <n>/<n> checks` and exits 0. A failing check prints `FAIL`
with its reason, and the last line reads `FAILED: <passed>/<total> checks`. The script's header
lists its environment knobs, and `test_sim_stack.py`'s docstring says what it checks and with which
limits.

## What it runs

The sim starts headless in the layout 9 / style 9 / seed 42 kitchen with its sensors on, the spawn
backed 2.0 m into open floor (`GOLEM_SPAWN_BACKOFF=2.0`), and the robot hanging from the sim's
elastic band. Beside it, the ros container runs the robot's sim bringup with navigation:

- H1-2: `h12_sim.launch.py use_rviz:=false use_skills:=false`;
- G1: `g1_sim.launch.py use_rviz:=false`.

The robot's leg node releases the band once its policy runs and `frame_task_server` is up. Then
the test's steps wait for Nav2, let the robot stand 3 s, send one `NavigateToPose` goal 1.5 m ahead
of the spawn, and, once `/cmd_vel` moves the robot, `/named_config` `t_pose`, then `home` once
`t_pose` has succeeded, each goal with a 15 s duration (`frame_task_server`'s timeout for it; its
default, 5 s, ends the H1-2's move to `t_pose` before its IK converges). After the goal's result
the robot stands 2 s, and the test waits for a `/map` that slam_toolbox published after the walk
(it publishes every 5 s and adds a scan every 0.5 m travelled).

## The checks

- **The goal is in open floor**: at least 0.75 m from anything the sim has between the floor and the
  robot's height, and reachable from the spawn.
- **Standing unsupported**: from the band's release to the end of the recording the band stays
  off, the pelvis stays above the robot's floor (G1 0.70 m, HOMIE stands at 0.744 m; H1-2 0.77 m,
  ALMI stands at 0.977 m) and tilts at most 15 deg, and no robot geom touches anything but the
  floor; the sim logs no fall. Contacts under the band (the arms' start-up moves can brush the
  kitchen) are reported, not failed.
- **Goal reached**: Nav2 reports success within 90 s of sim time, and the true pelvis ends within
  0.4 m of the goal.
- **Arms while walking**: `t_pose` is sent once `/cmd_vel` moves the robot, before the goal's
  result; `home` is sent once `t_pose` has succeeded; both goals carry a 15 s duration and both
  must succeed (`NamedConfig`'s `success`). Only `t_pose` has to start during the walk: the 1.5 m
  walk takes 4-6 s and each move a few seconds, so `t_pose` may end, and `home` usually runs,
  after the goal's result. The check does not require either move to finish inside the walk.
- **SLAM healthy** from Nav2's start to the end: `/Odometry` (FAST-LIO) at 5 Hz or more in every sim
  second; `level_odom`'s `odom -> base_footprint` in TF at every sample, at most 0.5 s behind
  `/clock`; `/map` knows more cells at the end than at Nav2's start; `/converted_scan` never stops
  for more than 0.5 s.
- **Safety**: `safety_node` is the only publisher of `/lowcmd` at every sample (once a second)
  from its start, with no other publisher at any sample, `golem_safety` latches no estop, and
  nothing publishes the sim's ground-truth `/odom`.

## How it runs

It needs the `golem_sim_robocasa` and `golem_ros` images (`docker/scripts/docker_build.sh robocasa
ros`); it never builds them. The containers are `tests/sim/common.sh`'s, which `tests/g1` shares:

- **Containers.** The compose services as `docker_run.sh` starts them (`robocasa`, with the G1
  overlay for the G1; `ros`), plus a `test_compose.yml` that `common.sh` writes into the output
  directory. The sim container has no network, and the ros container joins its network namespace.
  Both use its loopback with multicast, so no DDS leaves the host and nothing talks to the robot.
- **DDS domain.** 199 unless `ROS_DOMAIN_ID` is set in the shell. Domain 0 is refused.
- **Ground truth.** `GOLEM_SIM_ODOM` must be unset; the sim runs with it off, and the stack
  navigates on FAST-LIO. The sim role records the robot's state inside the sim process
  (`sim.npz`, `contacts.json`); nothing of it is published. The test's steps place the goal in the
  map frame from one recorded pelvis pose and the odometry at its stamp (`live_pose.json`).
- **`core_ws`.** Before the sim starts, the ros service builds `core_ws`, both robots' packages
  (`launch_ros.sh`), into `container_cache/sim_tests/{build,install,log}`, so the checkout's own
  `core_ws/build` and `core_ws/install` are left alone. The first run builds the whole workspace;
  later runs rebuild what changed. `CORE_WS=<dir>` builds and uses that workspace in place instead.
  The runs share that build, so one runs at a time.
- **Knobs.** The sim's knobs come from the shell or the script, never from `docker/.env`.

The test reuses `tests/g1`'s Python, which both containers mount: `test_sim_bringup.py`'s log
parsing and report helpers, its recorder base (`StepRecorder`) and its sim hook runner
(`run_recorded_sim`), and `test_sim_nav.py`'s geometry (`spawn_grid`), its `ContactLog` and
`write_live_pose` (the sim role) and its `SimDriver` (the drive's clock, marks, TF, waits and
`goal_in_map`), which `test_sim_nav.py`'s own driver and recorder use too. It patches each robot's
`SimInterface.__init__` and `after_step` and reads the band the sim hands the interface
(`SimInterface.band`), reads the sim's `ROS bridges up`, `elastic band released at t=` and `ROBOT FELL` log lines,
and from `launch.log` `golem_safety`'s JSON events `relay_started` and `estop_command_written` and
its `estop diagnostic` lines. A change to any of these must update the test in the same commit.

## Output

Each run writes to `tests/results/sim-stack-<robot>-<time>-<pid>/`:

- `run.json`: the robot, the exit codes, the domain, the commit, the submodules, the working tree
  and the image IDs;
- `report.log` and `report.json`: the checks and their numbers;
- `sim.log`, `launch.log`, `drive.log`, `ros_build.log` and `load.log`;
- the recordings: `sim.npz`, `contacts.json`, `scene.npz`, `goal.json`, `events.json` and
  `drive.npz`.

The test stops the sim (`stop` file) and waits for its recording before it stops the bringup, so
the recording ends with the robot standing, not with it falling once `rt/lowcmd` is silent. When a
test fails, keep the whole directory.
