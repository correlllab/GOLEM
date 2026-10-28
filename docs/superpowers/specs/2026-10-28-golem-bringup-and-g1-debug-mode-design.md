# golem_bringup, golem_robocasa and the G1's debug-mode start-up: design

Date: 2026-10-28. Base: branch `golem_upper_ik_safety` (HEAD `38540b2`, not yet merged into
`camera_ready`). The work goes on a new branch on top of it.

## Intent

Four changes, designed together because they touch the same bringups, sims and docs:

1. **One bringup package for both robots.** `h12_bringup` becomes `golem_bringup`, and
   `g1_bringup` is merged into it and removed. A robot starts with
   `ros2 launch golem_bringup {h12,g1}_real_robot.launch.py` (and the matching `_sim` files).
2. **The G1 starts the way the H1-2 does.** G1 development moves to Unitree's **debug mode**:
   Unitree's own controller is released, and our ROS bringup commands every motor at motor
   level, reading `rt/lowstate` and writing `rt/lowcmd` through `golem_safety`. `arm_sdk`
   mode (Unitree's controller keeps the legs, our stack shares the waist and arms) is unused
   from now on and is deleted everywhere, `g1_description` included. In the sim, an elastic
   band holds the G1 up until the toggle is called, as on the H1-2; the G1 bringup starts
   HOMIE (`g1_locomotion`), and HOMIE's node releases the band, as `h12_lowerbody_rl`'s node
   does on the H1-2. The sim no longer depends on HOMIE's files at all.
3. **The four unused packages are deleted**: `h12_safety_layer`, `h12_ros2_controller`,
   `g1_safety_layer`, `g1_control` (replaced by `golem_safety` and `golem_upper_ik`).
4. **`h1_robocasa` is renamed `golem_robocasa`**: it is the RoboCasa sim of both robots.

**Why.** The G1 should follow the groundwork the H1-2 laid: one start-up story, one set of
tools (the band, the heartbeat, the start-position interlock, the operator confirm), and a sim
that behaves like the real robot will in debug mode. Today the G1 sim plays Unitree's
controller (a HOMIE "stand-in" that stands the robot from the first physics step), which only
models the arm_sdk workflow the user is leaving behind. The names follow: what both robots
use is called `golem_*`, and what nothing uses is gone.

## Goals

- `golem_bringup` replaces `h12_bringup` and holds both robots' launch files, in
  `launch/h12/` and `launch/g1/`; `config/` and `rviz/` are split the same way.
- `golem_robocasa` replaces `h1_robocasa`, with the same behavior for both robots.
- The G1 sim has no stand-in and reads nothing of HOMIE's: the band holds the robot, every
  motor is off until a valid `rt/lowcmd`, and `/elastic_band/toggle` releases the band.
- On the G1, `golem_safety` streams zero-gain `rt/lowcmd` from boot and engages each input on
  its first valid command (the H1-2's `zero_stream` start-up); its heartbeat means relaying
  can begin, as on the H1-2.
- `g1_locomotion`'s node takes the role of `h12_lowerbody_rl`'s `lowerbody_controller_node`:
  idle until the first heartbeat, a 2 s pre-pose (sim and real alike), on the real robot an
  operator confirm, the policy, then the band release once `frame_task_server` is ready.
- The G1 bringups always start HOMIE; split mode is the default safety mode in the sim and on
  the real robot.
- `arm_sdk` mode is deleted everywhere; the four unused packages are deleted.
- Every test, wrapper and living doc follows.

## Non-goals

- `golem_ros2_models` (a robot-model package that does not exist yet): a separate design once
  it exists. This change keeps today's model sources.
- A G1 collision model or planner, G1 sensors or model servers on the real robot, a G1 desktop
  launch file (the laptop runs the whole G1 stack; nothing needs a desktop yet).
- Any change to the H1-2's behavior. Its launch file names and paths change; nothing it does
  changes.
- Renaming `h12_mujoco.py` or anything inside the sim beyond what the G1 changes need: only
  the directory is renamed.
- Isaac (H1-2 only, untouched).
- Running anything on the real G1. Nothing has run on it yet, and this change is verified in
  the sim only.

## Anti-goals (a failure even if it works)

- The H1-2 behaving differently afterwards in any way but the names you type and the paths.
- Compatibility shims: no `h12_bringup`, `g1_bringup` or `h1_robocasa` left behind, no alias
  launch files or symlinks under the old names, no `arm_sdk` config kept "just in case". A
  stale name fails loudly.
- The G1 sim publishing anything the real robot does not, or ground truth on by default. The
  band toggle service is the one addition, and the H1-2 sim already serves the same service.
- A G1 start-up whose sequence differs from the H1-2's for no reason the robots force.

## What the user decided

- The package is `golem_bringup`; it replaces `h12_bringup`, takes in `g1_bringup`, and
  `g1_bringup` is removed in the same change.
- Launch files are split into `launch/h12/` and `launch/g1/`. The commands are
  `ros2 launch golem_bringup h12_real_robot.launch.py` / `g1_real_robot.launch.py`, with
  desktop launch files where needed.
- The G1 follows the H1-2's groundwork. G1 development happens in debug mode, with our ROS
  bringup sending motor-level commands on `rt/lowstate` / `rt/lowcmd` through the safety
  layer.
- In the sim the G1 is held by the elastic band until the toggle is called, like the H1-2.
  HOMIE and the band are brought up by the G1 bringup.
- `arm_sdk` mode is unused and deleted, `g1_description`'s arm_sdk constants included.
- The sim does not need HOMIE's policy.
- The pre-pose is 2 s, the same in the sim and on the real robot.
- The four unused packages are deleted now.
- `h1_robocasa` is renamed `golem_robocasa`.
- The start-up below (sections 4-7) was played back and approved as described.
- Naming and moving (section 1) was played back and approved: drop `_bringup` from file
  names, move with `git mv`, update every reference.
- `golem_ros2_models` waits until it exists.

## Design

### 1. `golem_bringup`

`git mv core_ws/src/h12_bringup core_ws/src/golem_bringup`, then `git mv` each `g1_bringup`
file into it, so every file keeps its history. `core_ws/src/g1/g1_bringup` is gone at the end.

| Old | New |
|---|---|
| `h12_bringup/launch/h12_sim_bringup.launch.py` | `golem_bringup/launch/h12/h12_sim.launch.py` |
| `h12_bringup/launch/h12_sim_bringup_mac.launch.py` | `launch/h12/h12_sim_mac.launch.py` |
| `h12_bringup/launch/h12_real_robot_bringup.launch.py` | `launch/h12/h12_real_robot.launch.py` |
| `h12_bringup/launch/h12_real_desktop_bringup.launch.py` | `launch/h12/h12_real_desktop.launch.py` |
| `h12_bringup/launch/h12_safety_grippers.launch.py` | `launch/h12/h12_safety_grippers.launch.py` |
| `h12_bringup/launch/lmg_sim_bringup.launch.py` | `launch/h12/lmg_sim.launch.py` |
| `h12_bringup/launch/lmg_real_bringup.launch.py` | `launch/h12/lmg_real.launch.py` |
| `h12_bringup/launch/h12_{control,state,lowerbody,models,real_drivers,grippers,locomanipulationgame}.launch.py` | `launch/h12/`, same names |
| `g1_bringup/launch/g1_sim_bringup.launch.py` | `launch/g1/g1_sim.launch.py` |
| `g1_bringup/launch/g1_bringup.launch.py` | `launch/g1/g1_real_robot.launch.py` |
| `g1_bringup/launch/g1_core.launch.py` | `launch/g1/g1_core.launch.py` |
| `h12_bringup/config/*.yaml` | `config/h12/` |
| `h12_bringup/rviz/main.rviz` | `rviz/h12/main.rviz` |
| `g1_bringup/rviz/g1.rviz` | `rviz/g1/g1.rviz` |
| `h12_bringup/test/*`, `g1_bringup/test/*` | `golem_bringup/test/` (the two copyright/flake8/pep257 trios become one) |
| `h12_bringup/h12_bringup/` (Python module), `scripts/`, `common_cmds.md` | `golem_bringup/golem_bringup/`, `scripts/`, `common_cmds.md` |

- **Finding a file by name.** `ros2 launch <pkg> <file>` walks the package's whole share
  directory for the file name (`ros2launch.api.get_share_file_path_from_package`, checked in
  the `golem_ros` image) and refuses a name found twice. So the subdirectories do not change
  what anyone types, and every launch file name in the package must be unique. Acceptance:
  every top-level file starts with `ros2 launch golem_bringup <name>` from a
  `colcon build --symlink-install` workspace (`os.walk` does not follow directory symlinks,
  so check the installed layout), and a test asserts the names are unique.
- **Includes.** Every `IncludeLaunchDescription`/`os.path.join(share, 'launch', ...)` and
  config/rviz path inside the package points at the new subdirectory.
- **`package.xml`**: the union of both packages' `exec_depend`s, less the deleted packages.
  `launch_ros.sh` still skips `^g1_` packages for the H1-2; `--packages-ignore-regex` drops
  them from discovery, so `golem_bringup` still builds for the H1-2 and its G1 launch files
  resolve `g1_*` packages only when launched. The Mac build (`--packages-select`) needs the
  `g1_*` dependencies handled the way its other unbuilt `exec_depend`s are.
- **Stale installs (my call).** colcon never removes a package that no longer exists, so
  after the change `core_ws/install/{h12_bringup,g1_bringup,h12_safety_layer,
  h12_ros2_controller,g1_safety_layer,g1_control}` stay in every existing workspace: with
  `--symlink-install` their files dangle, and a copied install (the H1-2's onboard PC) would
  keep launching the old files. The docs' pull-and-restart pitfall and the change notes tell
  users to delete those `core_ws/build` and `core_ws/install` directories once; the
  launchers do not do it.
- **References outside the package** (75 files name either package today; the living ones
  change): `docker/scripts/docker_stack.sh`, `docker/docker-compose.stack.yml`,
  `docker/scripts/docker_common.sh`, `docker/RosDockerfile`, `docker/BUILD.md`, the Mac
  launcher `docker/mac/scripts/launch_ros_mac.sh` (it launches the Mac file by path, and its
  `PKGS` list), `docker/mac/RosDockerfile.slim.arm64`, `h12_slam`'s
  `h12_real_slam.launch.py` (it includes H1-2 building blocks), the READMEs of
  `golem_safety`, `golem_upper_ik`, `g1_locomotion`, `g1_navigation` (and its
  `g1_nav2.yaml` comment), `lmg_deploy`, `h12_lowerbody_rl/policies/almi`,
  `h1_robocasa/g1/g1_sensor_msgs.py`, `tests/g1/*`, `tests/stack/*`,
  `tests/docker/test_workflows.py`, and the docs (section 9).

### 2. `golem_robocasa`

`git mv h1_robocasa golem_robocasa`. Only tracked files move: look at what is left in
`h1_robocasa/` afterwards (ignored files such as `__pycache__` or crash dumps, and any local
outputs) before deleting it, and move anything that is not disposable.

- **Containers**: the bind mount's source and target become `golem_robocasa` and
  `/home/code/golem_robocasa` in `docker/docker-compose.yml` and
  `docker/mac/docker-compose.yml`; `docker/RobocasaDockerfile`'s `WORKDIR` and comment
  follow; `launch_robocasa.sh` and `launch_robocasa_mac.sh` already `cd` to the path, so an
  image built before the rename still runs (no forced rebuild).
- **Everything else that names it**: `docker/scripts/*`, `docker/mac/*`, `tests/robocasa/*`
  (including the surface tooling), `tests/g1/*`, `tests/stack/*`, `tests/docker/*`,
  `tests/README.md`, `.github/workflows/simulator-tests.yml` (paths and `PYTHONPATH`),
  `.gitignore`, `core_ws/src/g1/*` mentions, the docs.
- **My call:** `CL_Assets` (a submodule) keeps its two comments naming
  `h1_robocasa/battery_env.py` until it is next changed for another reason; editing it would
  need a push to its own repo for a comment.

### 3. The four unused packages are deleted

- `core_ws/src/h12_safety_layer` and `core_ws/src/h12_ros2_controller` are submodules:
  `git rm` each, which also removes its `.gitmodules` entry. Their GitHub repos stay as
  they are. `core_ws/src/g1/g1_safety_layer` and `core_ws/src/g1/g1_control` are in-tree:
  `git rm -r`.
- **The tools that import them go too**: `golem_safety/tools/crc_fixtures.py`,
  `snapshot_old_limits.py`, `golem_upper_ik/tools/snapshot_old_controller.py`,
  `snapshot_old_fk.py`. The fixtures they wrote stay as recorded test data; the tests'
  README (or the fixture directory's) says what each was recorded from: the package and its
  commit.
- **`golem_safety/tools/generate_robot_data.py`** reads the H1-2's joint-limit table from
  `h12_safety_layer` and checks it against `h12_ros2_controller`'s copy. The H1-2's table
  moves into `golem_safety` (a data module of the tool), and the generator must still write
  byte-identical `robots/h12.yaml` and `robots/g1.yaml` (apart from the G1's arm_sdk entries
  and the header naming its source). The cross-check against the deleted copy goes.
- **Words naming them**: configs and comments in `golem_safety` and `golem_upper_ik` that
  name an old package as the origin of a value say what the value is instead
  (`sim_safety_split`'s `logging.base_dir: /tmp/h12_safety_layer` becomes
  `/tmp/golem_safety`). `tests/stack/test_launch_packages.py` and the bringup package tests
  drop them. CLAUDE.md's "unused, awaiting deletion" rows go.
- **My call:** the stale comments in `h12_lowerbody_rl` that name them
  (`lowerbody_controller_node.py`, `fame_node.py`, `walking_node.py`) are corrected, since a
  comment naming a deleted package is wrong; only those lines change in that package.
  Submodules (`cl_realsense`'s README, the `mujoco_mpc` fork) are left as they are.

### 4. `arm_sdk` mode is deleted

- **`golem_safety`**: `arm_sdk_mode`; the `arm_sdk` and `sim_arm_sdk` configs; the
  `arm_sdk.*` keys (`engage_seconds`, `release_seconds`, `waist_kp_min`); the `/arm_sdk`
  output and the weight in slot 29; the arm_sdk engage ramp, the waist-kp latch, the release
  ramp after a latch and at exit (`arm_sdk_release_ramping`, `arm_sdk_release_held`); their
  tests and fixtures. The G1 profile lists `split_mode` and `full_body_mode` only and has no
  weight slot; `tools/generate_robot_data.py` writes it that way, defining the upper slots
  (waist and arms, 12-28) itself.
- **`golem_upper_ik`**: the G1 configs that write for arm_sdk (`config/g1/g1.yaml`,
  `config/g1/sim.yaml`); the follow gate's `/arm_sdk` weight check; the arm_sdk entries of
  `pairing.py`; their tests. `g1_split` and `sim_split` stay, under those names.
- **`g1_description`**: `ARM_SDK_SLOTS`, `ARM_SDK_WEIGHT_SLOT` and the `rt/arm_sdk` text in
  `layout.py`, and its tests of them.
- **The G1 sim**: the `rt/arm_sdk` subscription, its weight blend and `GOLEM_ARM_SDK_TIMEOUT`
  (and its pass-through in `docker/docker-compose.g1.yml`).
- **The bringups and docs**: every `arm_sdk` default, refusal rule and paragraph.

### 5. `golem_safety` on the G1: the H1-2's start-up

- The G1's `split`, `sim_split` and `full` configs use `startup.rule: zero_stream`: zero-gain
  `rt/lowcmd` (mode 0, every value 0) on every slot from boot; each input engages on its own
  first valid command once `/lowstate` reports `mode_machine` 5; the `/lowstate` limit
  checks run from boot. The arms engage when `frame_task_server` commands, the legs when
  HOMIE's node sends its first command.
- The heartbeat is unchanged code: no beat until `/lowstate` reports 5 (`use_estop` stays
  false on the G1: it has no estop node), then true at 10 Hz, false once latched.
- **My call:** `engage_gate` is deleted with arm_sdk, since no config uses it afterwards
  (with it go the `engage.*` keys if nothing else reads them, `estop_latched_silent`, and the
  G1-only "limit checks start at engagement" rule). One start-up path for both robots.
- **My call:** the G1 keeps `estop_response` `mode: zero`, `after: keep_writing` (every motor
  off, written every cycle) rather than the H1-2's `burst_then_silence`. What the real G1
  does when `rt/lowcmd` stops in debug mode has not been measured, and writing zero every
  cycle guarantees the motors are off whatever it does.
- The sim configs keep their relaxed stale times and estop limits.

### 6. The G1 sim: the band, and no HOMIE

The shared sim files keep the H1-2's behavior exactly; the H1-2 surface golden must not
change.

- **Remove** `golem_robocasa/g1/stand_in.py` and everything that uses it: `_settle_stance`,
  the `released` state, the "stand-in" wording of the sim's logs (the `ROBOT FELL` line keeps
  its prefix, which the tests match).
- **No HOMIE in the sim.** `g1_robosuite.py` stops reading HOMIE's `g1.yaml`: the G1's joint
  dynamics (damping, armature, frictionloss for every joint) and its spawn stance become the
  sim's own constants in `golem_robocasa/g1/`, taken from a Unitree G1 source rather than
  from HOMIE's files; the builder finds that source and cites it in the comment (Unitree's
  public G1 models and default joint angles are the expected one; HOMIE's values appear to
  match them). The policy mount leaves the sim service in `docker/docker-compose.g1.yml`;
  the `ros` service keeps it, and `prepare_mounts` still checks it for the `ros` container.
  The walk test is the check that HOMIE still walks on the sim's dynamics.
- **Add** the H1-2's band: `mujoco_env.ElasticBand` on the G1's `torso_link`, anchored 0.01 m
  above the torso's spawn position, with the H1-2's constants (stiffness 1e4 N/m, damping
  2e3 N*s/m) unless they fail the check below. `MeasurementBridge(env, elastic_band=band)`
  serves `/elastic_band/toggle` (`std_srvs/Trigger`), SPACE toggles it in the viewer, and
  the viewer draws it, as `h12_mujoco.py` does.
- **Motors**: off (zero torque) until the first valid `rt/lowcmd` (valid CRC, `mode_machine`
  5, `mode_pr` 0), then driven by it; `GOLEM_CMD_TIMEOUT` of silence turns every motor off,
  as now.
- **Spawn**: the robot spawns standing in the sim's stance, hanging from the band. Check:
  through the whole start-up (zero-gain legs hanging, the pre-pose, HOMIE under the band) the
  torso stays near upright, and the release does not drop the robot.
- **My call:** the H1-2's tether rotation knobs (`GOLEM_SIM_TETHER_ROT_*`) stay H1-2 only.
- Re-record `tests/robocasa/surface/g1.json` in the same commit: it loses `/arm_sdk` and
  gains `/elastic_band/toggle`.

### 7. `g1_locomotion`: the lower-body controller's role

HOMIE's node follows `lowerbody_controller_node`'s sequence:

1. **Waiting**: today's conditions (steady, on-time, finite, upright `rt/lowstate`; the tick
   rate; the arm controller commanding in the last 0.5 s) plus a first `/safety/heartbeat`
   that is true. It sends nothing while waiting.
2. **Heartbeat watchdog** (the H1-2's): once the first beat has arrived, a false beat, or no
   beat for 0.5 s, stops the node's output for good, with the reason logged.
3. **Pre-pose**: the legs ramp from their measured positions to the policy's default pose
   over 2.0 s, one target per policy step, at the training gains. 2.0 s is the H1-2's
   `prepose_ramp_s`, in both the sim and the real config (today 0.5 s and 3.0 s).
4. **Confirm (real robot only)**: the node holds the default pose until the operator calls
   `/lowerbody/confirm_engage` (`std_srvs/Trigger`). The sim config does not wait. **My
   call:** the H1-2's service name, so the operator types the same command on both robots
   (one stack per domain, so the names never meet).
5. **Policy**: HOMIE starts at zero velocity and 0.74 m, as now, while the band still holds
   the robot in the sim.
6. **Band release (sim only)**: once the policy runs and `/left_ee_pose` has been seen
   (`frame_task_server` publishes it only after its start-up moves), the node calls
   `/elastic_band/toggle` and retries until one call succeeds, never twice (the service
   flips the band). A maximum wait from the policy's start releases it anyway. The real
   config has no band and never calls the service.
7. **Walking** on `/cmd_vel` and the faults: unchanged.

The new settings live in the node's own config files (`config/sim.yaml`, `config/g1.yaml`),
like its other settings.

### 8. The G1 bringups

- `g1_core.launch.py` always starts `locomotion_node`; `use_locomotion` is gone. The safety
  config defaults to `split` on the real robot and `sim_split` in the sim, and the controller
  config to its pair (`g1_split`, `sim_split`).
- **Real robot** (`g1_real_robot.launch.py`): `locomotion_node` starts only with
  `start_position_verified:=true` (default false), as the H1-2's leg node does. Without it the
  legs stay on `golem_safety`'s zero-gain stream and the arms run. The operator puts the G1
  into debug mode, on a gantry, before launching; nothing in the stack does that. `use_nav`
  is still refused on the real robot.
- **Sim** (`g1_sim.launch.py`): HOMIE always runs; `use_nav` and `frontier_explore` no longer
  need `use_locomotion`.
- The refusal rules (`startup_refusal`, `golem_upper_ik.pairing`) shrink to what is left:
  config pairing, sim versus real configs, the start-position interlock, no real navigation.

### 9. Tests

- Unit tests: the `golem_safety` and `golem_upper_ik` arm_sdk and engage-gate tests go; the
  G1 gets the H1-2's zero-stream start-up tests. `g1_locomotion` gets tests for the heartbeat
  gate and watchdog, the 2 s pre-pose, the confirm hold and the band release (a fake toggle
  service: called once, retried until it succeeds, never twice, the maximum wait).
  `generate_robot_data.py`'s output is checked byte for byte against the profiles.
- `golem_bringup/test`: the moved tests of both packages; the launch-name uniqueness test; the
  refusal rules.
- `tests/g1`: the arm test no longer uses arm_sdk. It checks the band holds the robot, HOMIE
  starts, the band is released after `frame_task_server` is ready, the arms reach `home` and
  `t_pose`, the left hand closes, stopping the controller leaves the waist at its hold and the
  robot standing. The walk test walks as now. The nav and explore tests drop
  `use_locomotion`. The recorders' sim internals (`_arm_sdk`, `arm_sdk_w`, `released`)
  change to the band's state.
- H1-2 regression: `tests/stack` (arms and locomotion) and the H1-2 surface golden pass
  unchanged; `tests/docker` passes.
- The G1 surface golden is re-recorded (section 6).

### 10. Docs

- Living docs follow every change: `CLAUDE.md`, `AGENTS.md`, `README.md`,
  `docs/RUNNING.md`, `docs/DOCKER_TESTING.md` and `docs/SETUP.md` if they name the packages
  or `h1_robocasa`, `docs/g1/*`, `tests/*/README.md`, and the package READMEs above.
- `docs/g1/first-session.md` is rewritten for debug mode: the gantry, entering debug mode,
  `start_position_verified`, the confirm, what you should see, how to stop. **Open:** the
  procedure for entering debug mode on our G1, confirming it before launching, and when the
  robot comes off the gantry comes from the user; the page says plainly that it is missing
  until it does, and invents nothing.
- **My call:** the dated specs and plans under `docs/superpowers/` are records of past work
  and are left as they are.

## Constraints (from the project)

- Build and test inside the containers; only `tests/docker` runs on the host. Tests never use
  `ROS_DOMAIN_ID=0`.
- Comments describe the code as it is; history goes in commit messages.
- The H1-2 stays the default everywhere; nothing on the G1 path changes the H1-2's behavior.
- A change to what a sim publishes re-records its surface golden in the same commit.
- HOMIE's policy stays in `core_ws/src/g1/g1_locomotion/policy/` with its `LICENSE` and
  `README.md`; nothing copies its files elsewhere.
- Submodules are not edited here, apart from removing the two deleted ones.
- The Mac port runs the H1-2 only; it follows the renames (its compose file, launchers and
  `PKGS`).
- Commits are made as the plan's execution flow does; nothing is pushed without the user.

## Left to the builder

- The names of the new `g1_locomotion` config keys.
- Whether the arm and walk tests stay two scripts or become one.
- The G1 band's constants, if the H1-2's fail the upright check in section 6.
- How the sim spawns the hanging robot without penetrating the floor or the kitchen.
- Where the H1-2's joint-limit table lives inside `golem_safety`'s tools.

## Risks

- **Nothing here runs on the real G1.** Debug mode, the zero-gain stream on a gantry and the
  confirm flow are verified in the sim only; what the real G1's motors do when `rt/lowcmd`
  stops is unmeasured.
- **HOMIE under the band.** The policy runs while the band holds the torso, then the band
  drops. The H1-2's policies do the same, but HOMIE was trained without a band; the sim must
  show the release does not drop the robot.
- **The sim's own joint dynamics.** If the Unitree source's values differ from HOMIE's, the
  walk test may show HOMIE walking worse than today; that is a finding to report, not a
  reason to read HOMIE's files again.
- **Limit checks from boot.** With `zero_stream` the G1's estop limits apply from boot, while
  the legs hang at zero gain. The sim configs' relaxed limits should cover it; the sim run
  must show no latch during start-up.
- **Stale installs** of the removed packages (section 1) until users delete them.
