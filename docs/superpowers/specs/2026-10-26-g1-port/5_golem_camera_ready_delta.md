# Audit 5: GOLEM `camera_ready` today and what it gained since the G1 fork base

Repo: `/home/max/GOLEM`, branch `camera_ready` @ `42d6559` (2026-10-22). Fork base: `c00feca` (2026-07-16).
G1_GOLEM forked from `origin/spatial-memory-robocasa-benchmark` @ `c0af423`, which is `c00feca` plus 13 spatial-memory
commits by Xuan that are **not** on camera_ready (see §2.3).

Method: read-only `git -C` queries against local refs (no fetch). Remote-tracking refs were last fetched
2026-10-20 (CL_isaaclab_sim) and 2026-10-22 (superproject, CL_Assets, controller, safety layer); anything said about
"remote" means "as of that fetch".
Legend: **[V]** verified from code/git; **[I]** inferred (not built or run).

---

## 0. Headline findings

1. **[V] Safety layer pin regression.** `27a7c0b` (2026-10-20, HumanoidDesktop, "Track latest h12_safety_layer and
   h12_ros2_controller") moved `core_ws/src/h12_safety_layer` from `0eab20b` (its `origin/camera_ready`) to `ad06903`
   (its `origin/main`). The two branches diverged at `6bdfc66`, so the move **dropped 5 commits** that earlier
   camera_ready pins had shipped (`5b08888`, `e0396fc`, `58d0f99`, `5182fbe`, `0eab20b`):
   steady-time heartbeat, newest-only relay loop plus an e-stop on late `low_state`, going silent on `rt/lowcmd` after an e-stop,
   e-stop reason logging, and the 95 Nm ankle-pitch torque e-stop that `4cbdb25` says the lmg Mid360v2 policy needs.
   Consequences:
   - The CLAUDE.md invariant "after estop … stops publishing `rt/lowcmd`" is false for the pinned code. In `ad06903:h12_safety_layer/core/safety_layer.py:256-266`,
     `_publisher_loop` keeps publishing `make_estop_cmd()` forever.
   - **[I]** `tests/stack/test_heartbeat_clock.py` (added on camera_ready) should fail against the pinned code. Its
     heartbeat timer uses the ROS clock, and the STEADY_TIME fix is `5b08888`. The test is skipped in CI because it needs rclpy.
   - `relax_safety_split.yaml` (the default for real and lmg bringups) is back to a 60 Nm ankle-pitch torque e-stop.
     The dropped commit's comment says this trips "in every bag" of walking.
   - The controller pin `b53a6b2` includes `deacb9a` ("Drop the controller-side estop; keep command clipping").
     That change relied on the safety layer's late-state e-stop, which is gone, so **neither** side now e-stops on stale `low_state` **[I]**.
   - Fix (not done, read-only): merge h12_safety_layer `camera_ready` into its `main`, or pin a merge commit, then re-pin.
2. **[V] CL_Assets pin `a6c44e4` exists only on `origin/grasp_sweep_assets`.** It is 4 commits ahead of CL_Assets
   `origin/main` (`ed592de`) and 0 behind, so main can fast-forward. **CL_isaaclab_sim pin `6f89c64` exists only on
   `origin/camera_ready`** (4 ahead of its main `deec4de`, 0 behind). Every other pin is on its default branch, or on
   the `.gitmodules` branch for mujoco_mpc (`extended_hw`).
3. **[V] Local `camera_ready` == `origin/camera_ready` == `42d6559`.** `origin/main` (`8709a1d`) has **one commit that is not in
   camera_ready**: a 2026-09-23 safety-layer pin bump to `6bdfc66`, an ancestor of both safety pins. The
   statement "main is fully contained in camera_ready" is therefore true only for local `main` (`e0a33df`, which is 1 behind origin/main).
   Merging camera_ready into main needs a trivial merge.
4. **[V] H1-2 is hardcoded throughout.** There is no robot abstraction. The only existing knobs are listed in §6.1.
   Several pieces can be reused for the G1 more or less as they are: `model_server`, `custom_ros_messages`, `h12_slam` (URDF-driven),
   `unitree_ros2`/`unitree_hg`, FAST_LIO, and the docker/test infrastructure.
5. **[V] CLAUDE.md has about 11 concrete stale claims** (§5). AGENTS.md is an older copy that lags CLAUDE.md on the lmg and real-bringup rows.

---

## 1. Areas changed since c00feca (summary only)

`git diff --stat c00feca HEAD`: 297 files, +23,500 / −4,777. 140 commits (112 first-parent, 3 merges).
Authors: HumanoidDesktop / "humanoid desktop" (shared lab desktop) 76, max / maxconway 56, mateo 7, Yutong Zhang 1.

| Area | Δ (files, +/−) | What changed | Main authors |
|---|---|---|---|
| `h12_bringup` (renamed from `h1_bringup` in `5386f96`) | 32 files +3944 (h1_bringup −4533) | split into building-block launches (`c03d76c`); lmg_sim/real bringups; one `main.rviz`; ALMI legs onboard; new gripper boards | HumanoidDesktop |
| `h12_slam` (new, `c03d76c`) | 28 files +3397 | nav stack moved out of bringup; FAST-LIO extrinsic and body→pelvis read from the URDF; level_odom, URDF self-filter, octomap costmaps; holonomic MPPI Nav2; record_slam_bag; h12_real_slam launch | HumanoidDesktop |
| `h12_skills` | 17 files +3883/−858 | pluggable grasp synthesis (`grasp_synthesis/`: centroid, antipodal, GGX, heuristic, top_down, ranking); world-anchored approach and servo; hand-camera visual servo; run telemetry; frontier-explore fixes | HumanoidDesktop, max |
| `lmg_deploy` (new, renamed from `locomotion_game_deploy` in `d932cb3`) | 31 files +3595 | MARL loco-manipulation policy node (27 joints), FAST-LIO attitude, MID360 IMU, walk/arm modes, bags in LFS, tests | HumanoidDesktop, max |
| `h12_lowerbody_rl` | 9 files +894/−93 | ALMI policy, one pinned policy, confirm-engage, sim2real gap monitor (almi27 removed) | max, HumanoidDesktop |
| `model_server` | 3 files +118/−36 | yolo bundles, graspgen tweaks, Gemini model `gemini-robotics-er-2-preview` | max |
| `h12_deploy_mjpc` | 1 line | arg-list fix | — |
| `h1_robocasa` | 12 files +1737/−340 | RealSense-faithful cameras, Battery task, force-limited Magpie, experiment surface, elastic-band options (another auditor covers this in depth) | max |
| `docker` | 76 files +3232/−828 | `docker/locks/` (pinned apt/pip), shared `docker_common.sh`, `docker_stack.sh`, mac wrappers onto shared scripts, livox staging in `launch_ros.sh` (another auditor covers this) | max, HumanoidDesktop |
| `tests` (new) | 54 files +3717 | `tests/{docker,isaaclab,robocasa,stack}`, CI workflow | max |
| `docs` | 7 files +691 | DDS_TUNING, DOCKER_TESTING, MACOS, RUNNING, SETUP | both |
| submodule pins | 8 gitlinks | see §2 | — |

---

## 2. Submodule pin changes c00feca → HEAD (content diff of old pin vs new pin)

All new pins are descendants of the old pins (none rewound). Pins that are unchanged: `estop c30fb50`, `h12_ros2_model 0b46cfc`,
`livox_ros_driver2 13eb05e`, `magpie_msgs 75a5267`, `unitree_ros2 668d1ec`, `unitree_sdk2_python c158c8d`.

| Submodule | old → new | commits | `diff --shortstat` |
|---|---|---|---|
| h12_ros2_controller | `87bca34` → `b53a6b2` | 110 | 173 files, +43,787/−809 |
| h12_safety_layer | `8ad5988` → `ad06903` | 14 | 18 files, +336/−33 |
| custom_ros_messages | `2d7d2ff` → `c60205c` | 6 | 7 files, +37/−3 |
| cl_realsense | `fc9a656` → `1f5d921` | 4 | 2 files, +287 |
| magpie_control | `7f86208` → `f215020` | 2 | 2 files, +32/−3 |
| CL_Assets | `6101b25` → `a6c44e4` | 13 | 45 files, +1274/−118 |
| CL_isaaclab_sim | `a58e592` → `6f89c64` | 26 | 100 files, +2752/−8769 |
| mujoco_mpc | `d1b118a` → `824ce4a` | 22 | 49 files, +10,393/−450 |

### 2.1 What G1_GOLEM's stale in-tree copies lack

G1_GOLEM imported these as in-tree copies at the c00feca pins. custom_ros_messages is the exception: it was imported at
`bb54bbd` = `2d7d2ff` + `SkillRetrieveMemory.action`.

**h12_ros2_controller `87bca34..b53a6b2` [V]**

Runtime changes that matter for a port:
- FrameTask `slow_mode` (about 0.25× speed via UpperController `speed_scale`) and `relative` targets (position offset in the pelvis frame, orientation premultiplied) in `ros2/frame_task_server.py` (+193). NamedConfig `slow_mode` (`utility/named_config.py`). Slow-mode time-stepping fix (`f3a9a55`).
- `core/latest_reader.py` (new) and `core/channel_interface.py` (+161/−): `rt/lowstate` is read latest-only, and motion stops on stale state (`caf318e`).
- `core/low_cmd_handler.py`, `utility/controller_config.py:51-57,270-271`: commands go only to `rt/safety/*` inputs, and any other `topics.low_cmd` is refused. Sport mode is removed (`ec3d1e6`). The controller-side e-stop is removed but clipping stays (`deacb9a`).
- `core/controller/upper_controller.py` (+265): moves start from the last command and carry the hold bias, so the arm does not drop at goal boundaries (`772f946`).
- `core/ik_solver.py`: IK is seeded from the measured pose first (`9eec384`), and the reduced model is re-locked at the measured torso on each goal (`0159d45`).
- Hold reports a joint pinned at its position limit instead of "Goal reached" (`3d87152`). Full x/y/z min/max frame limits were added, with a `frame_z_min` floor of 0.0.
- Configs: `balance_safety_split.yaml`, `counter_safety_split.yaml` (renamed from `sport.yaml`), `real_benchmark_upper.yaml`, `legacy/*`. The `safety_full`, `safety_split` and `sim_safety_split` configs were edited. Staging named configs were re-recorded. `elbow_only` was removed.

Research additions, probably irrelevant to the G1: `core/controller/counter_balance/*` (Crocoddyl DDP, residual H2/H3 and related, about 9k lines), `zmp_controller*.py`, `zmp_legacy/*`, `momentum_controller.py`, `robot_dynamics.py` +747, `support_region.py`, about 40 new `test/test_*.py` files, about 50 `docs/*.md`, and `data/motor_record/temp/*.npz`.

**h12_safety_layer `8ad5988..ad06903` (pinned) [V]**
- `/safety/heartbeat` (Bool, 10 Hz) in `ros2/safety_node.py`.
- Configurable upper-stale watchdog (`estop.upper_stale_seconds`). Relaxed wrist e-stop limits.
- `default_safety_split` clip `position_offset` 0.01 → 0.001.
- `relax_safety_split.yaml` expanded (+97). New `locomotion_safety_full.yaml`. Relaxed sim limits.
- Lower-body merge bug fix (`6bdfc66`).
- Logging off by default, plus a `safety_logger` interface (`d3d2725`).
- **Not in the pin** (on `origin/camera_ready` only; see finding 1): `5b08888`, `e0396fc`, `58d0f99`, `5182fbe`, `0eab20b`. `git diff --stat ad06903 0eab20b` gives 13 files, +309/−226, including the new `core/latest_reader.py` and `safety_layer.py` ±302.

**custom_ros_messages `2d7d2ff..c60205c` [V]** (full diff read)
- `FrameTask.action` adds `bool slow_mode` and `bool relative`. `NamedConfig.action` adds `bool slow_mode`.
- `SkillGrasp.action`: `arm` may be `""`/`"none"` (auto), plus `top_down`, `min_downward_pitch_deg`, `visual_servo`. Timeout 0 now means no deadline.
- `SkillPickPlace.action`: auto arm, plus `top_down` and `visual_servo`.
- New `srv/LmgSetMode.srv` and `srv/LmgSetArmGoals.srv`.
- **Not on camera_ready:** `bb54bbd` "Add spatial memory retrieval action" (`SkillRetrieveMemory.action`). It exists only on custom_ros_messages `origin/spatial-memory-retrieval-action`, is not an ancestor of `c60205c`, and is not on main. G1_GOLEM's `g1_msgs` contains it.
- Spot check of G1's copy (`G1_GOLEM/core_ws/src/g1/g1_msgs`): its FrameTask has no `slow_mode`/`relative`, and its SkillGrasp has no `top_down`/`visual_servo`. That confirms it is a pre-fork snapshot.

**cl_realsense `fc9a656..1f5d921` [V]**
`h12_rs_cams` was split into `launch/h12_headcamera.launch.py` (head, 15 fps) and `launch/h12_hand_cameras.launch.py`.
Bringup uses both (`h12_real_drivers.launch.py:67`, `h12_real_desktop_bringup.launch.py:47`). `h12_rs_cams.launch.py`
remains, and only CLAUDE.md, the README and a mujoco_mpc helper script reference it.

**magpie_control `7f86208..f215020` [V]**
`reset_overload` (clears an overload without releasing the gripper) and logging, in `gripper.py` and `gripper_node.py`.

**CL_Assets `6101b25..a6c44e4` [V]**
- MuJoCo battery workcell: `mujoco_assets/battery/*` (cell, bus bar, fuse block, cable harness).
- Magpie gripper joint dynamics and intra-gripper contact excludes (`robosuite_assets/grippers/*`, `ros_assets/h1_2_magpie_collision.srdf` +98, `exclude_pair_magpie.py`).
- H1-2 MID-360 mount pitch of 9.41°, with separate lidar-point and IMU-chip frames in all `ros_assets/h1_2_*.urdf`.
- Isaac `isaac_assets/robots/h1_2_magpie/*`. The old inspire USD and table USD were removed. `scene_h1_2_handless_table.xml` was added.
- No G1 assets anywhere in CL_Assets.

**CL_isaaclab_sim `a58e592..6f89c64` [V]**
- mateo's July cleanup ("cyberjanitor"), whose commits say it **removed the G1 definitions**: the old pin had `dds/g1_robot_dds.py` and `tasks/common_observations/g1_29dof_state.py`. It also removed sim_state, dex3 and gripper DDS, moved `dds/` → `src/python/dds/`, and moved action_provider → control.
- max's September work: the Magpie H1-2 asset with coupled grippers and native ROS services, the restored ROS digital twin, and a reset fix. October: the Magpie IMU on the MID-360 chip frame.

**mujoco_mpc `d1b118a..824ce4a` [V]**
AllenDevaraj and badinkajink, July 16-30: lean, stabilize, stand and grasp tasks; deploy helper scripts; ankle autocalibration; estimator filters v2-v4.
The pin is on `origin/extended_hw`, matching `.gitmodules`.

### 2.2 Historical pins
Every camera_ready historical pin for the submodules above is an ancestor of the current pin, **except** the 5 h12_safety_layer
commits. Two old h12_ros2_controller pins (`ed274ad`, `ac8a90b`, from July/August) are missing from the local clone. That is cosmetic.

### 2.3 Fork-only commits
`c00feca..c0af423` holds 13 Xuan commits: the RoboCasa spatial-memory benchmark, the "H12 spatial memory skill" `d74738e`, and ROS-container wiring.
They live only on `origin/spatial-memory-robocasa-benchmark`, so any spatial-memory piece of the G1 port lands on code camera_ready has never had.

---

## 3. core_ws packages on camera_ready (one line each) [V]

`core_ws/src` holds 18 tracked directories and 21 `package.xml` files. LOC means py/cpp/h lines in tracked files.

| Dir | Package(s) | Tracking | Size | Purpose |
|---|---|---|---|---|
| `FAST_LIO` | `fast_lio` | in-tree (vendored) | 72 files, 12.7k LOC | LiDAR-inertial odometry |
| `cl_realsense` | `cl_realsense` | submodule `1f5d921` (main) | 16 files, 1.1k | RealSense launches (head/hand) and point-cloud accumulator |
| `custom_ros_messages` | same | submodule `c60205c` (main) | 18 actions, 8 msgs, 15 srvs | shared interfaces |
| `estop` | `estop` | submodule `c30fb50` (main) | 15 files, 1.0k | serial e-stop → Bool (real robot) |
| `h12_bringup` | same | in-tree | 32 files, 1.8k | 14 launch files (§5), lowerbody/mjpc configs, `livox_imu_upright`, `main.rviz`, `common_cmds.md` |
| `h12_deploy_mjpc` | same | in-tree | 8 files, 0.5k (plus the C++ core built from the `mujoco_mpc` sources) | MJPC launcher and RW-EKF estimator |
| `h12_lowerbody_rl` | same | in-tree | 36 files, 3.1k | ALMI / FAME / walk policies, one pinned policy, confirm_engage, sim2real gap monitor |
| `h12_ros2_controller` | same | submodule `b53a6b2` (main) | 603 files, 36.7k | Pinocchio arm IK, frame_task/dual_arm/named_config servers, joint_state_publisher (and the counter-balance research) |
| `h12_ros2_model` | same | submodule `0b46cfc` (`unified_sim`, also on main) | 14 files | **not in the CLAUDE.md table**. Installs CL_Assets URDF/SRDF under `share/`. A `<depend>` of h12_ros2_controller (`path_definition_ros.py`). Needs `../../../CL_Assets`. Its README is stale ("Humanoid_Simulation"). |
| `h12_safety_layer` | same | submodule `ad06903` (main) **regressed, see §0** | 50 files, 2.2k | sole `rt/lowcmd` publisher; `MOTOR_COUNT = 27` (`core/joint_limits.py:3`); legs 0:12 / upper 12:27 (`safety_layer.py:242`) |
| `h12_skills` | same | in-tree | 34 files, 5.9k | 12 `/skill/*` servers: grasp, pick_place and frontier_explore implemented; 9 stubs |
| `h12_slam` | same | in-tree (new) | 28 files, 1.2k | FAST-LIO → level_odom → self_filter → octomap → pc2scan → slam_toolbox → Nav2 (MPPI omni) |
| `livox_ros_driver2` | same | submodule `13eb05e` (master) | 103 files | Livox driver. **Locally modified `build.sh`** (§4) |
| `lmg_deploy` | same | in-tree (new) | 31 files, 2.7k, plus LFS policies and bags | **not in the CLAUDE.md table**. MARL loco-manipulation `marl_deploy_node` (27 joints) |
| `magpie_control` | same | submodule `f215020` (main) | 40 files, 7.3k | Dynamixel gripper node, F/T and tactile |
| `magpie_msgs` | same | submodule `75a5267` (unified_sim and main) | 23 files | gripper interfaces |
| `model_server` | same | in-tree | 20 files, 2.1k | gemini / sam / graspgen / yolo service servers. GraspGen weights in-tree. SAM3 weights untracked. |
| `unitree_ros2` | `unitree_api`, `unitree_go`, `unitree_hg`, `unitree_ros2_example` | submodule `668d1ec` | 173 files | Unitree IDL (`unitree_hg` LowCmd/LowState, which the G1 also uses) |

**Ignored leftovers [V]**
- `core_ws/src/h1_bringup/` holds only 11 `__pycache__/*.pyc` (cpython-313) for the pre-rename launch files.
- `core_ws/src/locomotion_game_deploy/` holds only 19 `__pycache__` and `.pytest_cache` files.
- Neither has a `package.xml` or any source, so colcon ignores both. They are leftovers from the renames `5386f96` and `d932cb3` (2026-10-11), not stale source copies, and can be deleted. The pre-rename source exists as a commit on branch `camera_ready-local-2026-10-11` (`67156bc`).
- Similar `__pycache__`-only dirs: `docker/tests/` and `h1_robocasa/tests/` (those tests moved to `tests/`).
- Other ignored items: `work/lock_python.py`, `bags/`, `FAST_LIO/Log/*.txt`, `core_ws/{build,install,log}_stack_tests/`, `tests/results/` (ignored by `tests/.gitignore`).

---

## 4. Branch and push hygiene [V]

**Superproject**
- `camera_ready` = `origin/camera_ready` = `42d6559`. The working tree is clean.
- `main` (`e0a33df`, 2026-08-13) is behind `origin/main` (`8709a1d`) by 1 and is fully contained in camera_ready (92 behind).
- `origin/main` has 1 commit not in camera_ready: `8709a1d`, which bumps the safety layer to `6bdfc66`.

Other local branches:

| Branch | Tip | Relation to camera_ready | What it is |
|---|---|---|---|
| `camera_parity` | `654e344` | ancestor (0 ahead / 31 behind) | Fully merged. Checked out in a **worktree** at `/tmp/claude-1000/-home-max-GOLEM/17aaa9f8-008c-4728-9a96-195f6c763c76/scratchpad/parity`. Upstream is set to `origin/camera_ready`. Safe to delete after removing the worktree. |
| `camera_ready-local-2026-10-11` (and its `origin/` copy) | `67156bc` | 1 ahead / 84 behind (base `189037f`) | "Snapshot: local locomotion_game_deploy/bringup work before merging onto camera_ready": pre-rename `h1_bringup` and `locomotion_game_deploy` with the `marl_ik` policy. Superseded by `lmg_deploy` **[I]**. |
| `backup/pre-rebase-993596c` | `993596c` | 1 ahead / 33 behind | Patch-identical to camera_ready's `8a7c4b7` (`git patch-id` matches), so it is a pure pre-rebase backup. |

- **Stash:** `stash@{0}` "On camera_ready: grasp-sweep uncommitted work (task #2)" (2026-10-22 16:02, based on `993596c`, 13 files).
  7 of its 13 files are byte-identical to HEAD. The other 6 (`grasp.py`, `h12_mujoco.py`, the compose file, two sim launch files, `common_cmds.md`) differ from HEAD.
  It is probably superseded by the 10-22 commits **[I]**. Review before dropping.

**Submodule pins vs remote-tracking branches**
- All pins are on their default branch, except:
  - **CL_Assets `a6c44e4`**: only on `origin/grasp_sweep_assets`. The checkout is on local branch `grasp_sweep_assets`. Fast-forwardable onto main (4 ahead).
  - **CL_isaaclab_sim `6f89c64`**: only on `origin/camera_ready`. Fast-forwardable onto main (4 ahead).
  - **mujoco_mpc `824ce4a`**: on `origin/extended_hw` (the `.gitmodules` branch) and others, not on `origin/main`. Intended.
- **h12_safety_layer:** `origin/camera_ready` is 5 ahead of the pin and contains the dropped work (§0).
- **h12_ros2_controller:** `origin/feature/ompl-collision` (+8, 10-20) and `origin/feature/ompl-improvement` (+4, 10-21) are newer, unmerged work.
- **livox_ros_driver2:** `origin/master` is +2 newer, but must not be bumped per CLAUDE.md.
- No submodule has uncommitted changes except livox.

**livox_ros_driver2 local modification** (file mtime 2026-09-16, owner max):
```diff
--- a/build.sh
+++ b/build.sh
@@ -62,7 +62,7 @@ elif [ $ROS_VERSION = ${VERSION_ROS2} ]; then
     cd ../../
-    colcon build --cmake-args -DROS_EDITION=${VERSION_ROS2} -DDISTRO_ROS=${ROS_DISTRO}
+    colcon build --symlink-install --cmake-args -DROS_EDITION=${VERSION_ROS2} -DDISTRO_ROS=${ROS_DISTRO}
 fi
```
This matters because `docker/RobocasaDockerfile:147` and `mac/RobocasaDockerfile.arm64:111` `COPY` the **working tree** of this
submodule (it is whitelisted in `.dockerignore`) and run `./build.sh humble` (`RobocasaDockerfile:153-155`).
The Dockerfile comment at `:148-152` ("build.sh uses --symlink-install …") describes this unpushed local edit, not upstream `13eb05e`.
So images built on this machine differ from images built from a fresh clone **[V]**. The fresh-clone build would still work, because without symlinks `install/` holds real copies **[I]**.
`launch_ros.sh` does not use `build.sh` (`launch_ros.sh:55-74`; `tests/docker/test_launch_ros_cache.py:68`).

---

## 5. CLAUDE.md accuracy on camera_ready

**Wrong or stale [V]**

1. **Header:** "verified against the code on 2026-07-12", yet the text describes October changes.
2. **§4 "20 ROS packages in 17 directories":** there are 21 `package.xml` files in 18 tracked directories. The table (16 rows) omits **`lmg_deploy`** and **`h12_ros2_model`**.
3. **§4 cl_realsense entry point `launch/h12_rs_cams.launch.py`:** bringup uses `h12_headcamera.launch.py` and `h12_hand_cameras.launch.py`.
4. **§4/§6 h12_slam pipeline "FAST-LIO → pointcloud_to_laserscan → slam_toolbox → nav2":** it now also runs `level_odom`, `self_filter` and `octomap_server` (`h12_navigation.launch.py:96-135`). There are also configs `octomap.yaml`, `fastlio_nav2_config.yaml` and `test_nav2_config.yaml`, and `behavior_trees/navigate_to_pose_humanoid.xml`.
5. **§6 `lmg_sim_bringup` "the lmg_deploy MARL policy (Explicit500)":** `policy:=<name>` is now a required argument (`lmg_sim_bringup.launch.py:50`, `lmg_real_bringup.launch.py:48`). Only `policies/mid360v2` ships. Explicit500 survives only as `test/data/explicit500.yaml`.
   The lmg_deploy README contradicts itself: line 5 says "No policy ships", line 28 says "Mid360v2 since 10-20" (that dated note also breaks the CLAUDE.md comment rule).
6. **§8 "Other invariants":** "Once estop latches, the safety layer writes a short burst of zero-gain commands and then stops publishing `rt/lowcmd` until restart" is **false at the pinned `ad06903`** (§0). The rest of that bullet is true: `controller_config.py:51-57` refuses non-`rt/safety/` topics.
7. **§8 Livox section is largely stale:**
   - "both Dockerfiles install the SDK with an unpinned `git clone --depth 1 …master`" is wrong. All three Dockerfiles pin Livox-SDK2 at `08f523c` (`RosDockerfile:82-83`, `RobocasaDockerfile:119-120`, `mac/RobocasaDockerfile.arm64:89-90`), and `tests/docker/test_source_pins.py:16-26` forbids any unpinned `git clone`.
   - "`launch_ros.sh` doesn't check either" is wrong. `launch_ros.sh` no longer runs `build.sh`.
   - Still true: `RobocasaDockerfile:155` runs `./build.sh humble` with no `CustomMsg` import assertion.
   - **[I]** `08f523c` is the SDK commit that adds `kLivoxLidarDoubleEchoData`, so the documented `4a1def9` compile failure probably would not reproduce now. "Do not bump" is still a reasonable policy.
8. **§8 line references:** `h12_mujoco.py:184` is now `:260` (GOLEM_SIM_ODOM). `mujoco_ros_bridge.py:36` is now `:47` (the `livox_ros_driver2` import).
9. **§3 mac toggles `GOLEM_LOWERBODY=fame|walk|switch`:** the valid values are `fame|walk|almi` (`h12_sim_bringup_mac.launch.py:110-112`; `switch` maps to nothing).
   Undocumented env vars used by compose and code: `GOLEM_SIM_EXPERIMENT`, `GOLEM_HAND_CAMERA_RES`, `GOLEM_SPAWN_OFFSET`, `GOLEM_SIM_TETHER_ROT_STIFFNESS`, `GOLEM_SIM_TETHER_ROT_BODY`, `GOLEM_SIM_TIMEOUT`, `GOLEM_WAIT_FOR_SIM`, `GOLEM_USE_RVIZ`, `GOLEM_LIDAR_SELF_FILTER`, `GOLEM_PREPARE_ONLY`, `GOLEM_PLATFORM`, `GOLEM_MJPC_GIT_DIR`.
10. **§2 "`docker/mac/` self-contained":** the mac scripts are now thin wrappers that `exec ../../scripts/docker_*.sh` with `GOLEM_PLATFORM=mac` (`docker_build_mac.sh:4-5`, and likewise for run and stack). Only the Dockerfiles, compose files and `launch_*_mac.sh`/`robot_cli.sh` are mac-specific.
11. **§6 `h12_sim_bringup` "camera optical-frame static TF":** the node is still there (`h12_sim_bringup.launch.py:84-94`, `camera_link → camera_color_optical_frame`), but its comment is stale. RoboCasa now stamps `head_color_optical_frame` and similar (`h1_robocasa/realsense_profile.yaml:27`), and nothing in `h1_robocasa`, `h12_skills` or `model_server` references `camera_color_optical_frame`, so this TF is probably dead **[I]**.
    Also, `use_sliders` (`:35`) feeds only commented-out code (`:116-128`), which goes against CLAUDE.md's own "no commented-out code" rule.

**Verified correct [V]**
- Skills table: `base.py:218-231`. 8 stub files raise `NotImplementedError`, and `turn.py` covers turn_lever and twist_knob, giving 9 stub servers.
- 18 actions.
- The domain-0 guard (`docker_common.sh:55-59`; x86 `ros` prompts, mac rejects).
- docker_build/run/stack arguments.
- MuJoCo 3.2.3 (ros) / 3.3.1 (robocasa) in `docker/locks`.
- The numpy<2 / setuptools 59.6.0 / wheel<0.44 clamps (`RosDockerfile:193,555`).
- The `/lowerbody/confirm_engage` flow.
- The mac `PKGS` list (`launch_ros_mac.sh:125`).
- Every docs file named in §2 exists, plus `docs/DOCKER_TESTING.md` and `tests/README.md`.
- No remaining references to `h1_bringup` or `locomotion_game_deploy` in tracked files.

**AGENTS.md vs CLAUDE.md [V]** (`diff CLAUDE.md AGENTS.md`). AGENTS.md is the older copy. It:
- lacks the `lmg_sim_bringup` and `lmg_real_bringup` rows and the `h12_locomanipulationgame.launch.py` building block;
- lacks the safety-layer `rt/lowcmd` invariant;
- still puts the lower body on the desktop bringup ("lowerbody (`lowerbody_real.yaml` + MJPC estimator)", with the leg interlock on the desktop), which is stale since `2b2c665` moved ALMI onboard;
- lacks the `/lowerbody/confirm_engage` sentence.

Otherwise the two differ only in the title and "Codex" vs "Claude Code". Both were last touched in `7b62149` (10-17).

---

## 6. Readiness to receive the G1 port

### 6.1 Existing parameterization (all of it) [V]
- `urdf_file` launch argument: `h12_state.launch.py:21-23` and `h12_slam/launch/h12_navigation.launch.py:80-83`. h12_navigation derives the FAST-LIO extrinsic and body→pelvis **from the URDF** (`:16-45`). This is the best-prepared seam.
- `GOLEM_ASSETS_DIR` (`h12_state.launch.py:13`, `lmg_real_bringup.launch.py:27`, compose files).
- `H12_URDF` env override in `h12_slam/test/test_mid360_frames.py:15`.
- Isaac `--robot_type h1_2` (`docker/scripts/launch_isaac.sh:67`) and `--hand_type magpie|inspire`.
- RoboCasa `--task` and per-task `spawn_margin`.

Nothing else selects a robot.

### 6.2 H1-2 hardcoding a port has to deal with [V]
- **Names:**
  - packages `h12_*`;
  - containers `golem_ros`, `golem_sim_robocasa`, `golem_sim_isaac`;
  - in-container paths `/home/code/h1_robocasa` (`docker-compose.yml:117`, `launch_robocasa.sh:47-51` runs `h12_mujoco.py`) and `/home/code/h12_sim_scripts`;
  - CI paths.
  - `/home/code` appears 96× in `docker/` and 45× in `tests/`.
- **URDF filenames:** `h1_2_magpie_ros.urdf` is hardcoded in `h12_sim_bringup.launch.py:50`, `h12_sim_bringup_mac.launch.py:31`, `lmg_sim_bringup.launch.py:60,72`, `lmg_real_bringup.launch.py:80`, plus the defaults in h12_state and h12_navigation.
- **Safety layer:** `MOTOR_COUNT = 27`, the leg/upper split 0:12 / 12:27, and per-joint YAML lists for 27 H1-2 joints. The G1 needs its own layout. G1_GOLEM has a separate `g1_safety_layer` that includes `core/arm_sdk.py`, which clashes with camera_ready's "no `rt/arm_sdk` path" invariant **[I, design question]**.
- **h12_skills constants (`base.py:63-134`):**
  - `/realsense/head/*` topics;
  - `{left,right}_graspgenx_frame`;
  - `/left|right/gripper` namespaces;
  - `GRIPPER_MAX_WIDTH_MM = 106` (Magpie);
  - `WORLD_FRAME = 'camera_init'`.
  - Grasping is Magpie-specific. `frontier_explore.py` only needs `/map` and `/navigate_to_pose`, so it is robot-agnostic. G1_GOLEM has its own copy in `g1_navigation/frontier_explore.py`.
- **h12_slam:**
  - frame names `livox_link`, `livox_imu_link`, `pelvis` (`h12_navigation.launch.py:16-17,37`);
  - `MID360_config.json` (H1-2 mount, roll 180);
  - Nav2 footprint `[0.20, 0.275]` (`nav2_config.yaml:198,223`) and speed limits.
  - Everything else (level_odom, URDF self_filter, octomap, MPPI omni) is generic.
- **h12_ros2_controller:** the H1-2 model via `h12_ros2_model`.
- **lmg_deploy:** H1-2 27-joint policy.
- **h12_lowerbody_rl:** H1-2 leg policies.
- **CL_Assets:** has no G1 assets. CL_isaaclab_sim **deleted** its G1 definitions in July (§2.1).

### 6.3 What the G1 code would plausibly reuse from camera_ready [I]
- **`model_server`:** robot-agnostic; `h12` appears only in comments. G1_GOLEM's `perception/model_server` is a fork-time copy and lacks the yolo bundle, graspgen and Gemini-model changes.
- **`custom_ros_messages`:** shared interfaces. The port must add `SkillRetrieveMemory` (still on an unmerged custom_ros_messages branch) and G1's `SetHandPose.srv` instead of keeping a separate `g1_msgs` copy.
- **`h12_slam`, FAST_LIO, livox_ros_driver2:** the G1 head also carries a MID-360, and the URDF-driven extrinsic should transfer if the G1 URDF names its frames to match or the names are parameterized.
- **`unitree_ros2/unitree_hg`:** the G1 uses the same hg LowCmd/LowState.
- **`cl_realsense`**, and the `h12_skills` frontier explorer.
- **Docker infrastructure:** `docker/locks`, `docker_common.sh` domain guard, `docker_stack.sh`, `launch_ros.sh` cache and livox staging.
- **Test harnesses:** `tests/stack/run.sh` (sim × {arms, locomotion}) could take a robot axis.

### 6.4 Tests a port must keep green [V]
- **CI CPU job** (`.github/workflows/simulator-tests.yml`, on PRs touching `core_ws/**`, `docker/**`, `h1_robocasa/**`, `tests/**` and others):
  - `tests/isaaclab/test_*.py` (`PYTHONPATH=CL_isaaclab_sim:unitree_sdk2_python`);
  - `tests/robocasa/test_loop_timing.py`;
  - **all of `tests/docker`**: `test_source_pins.py` requires base images by digest, every `git clone` pinned to a 40-hex SHA, every `pip install` using `-c /opt/golem-locks/…`, and exact lock lines. `test_workflows.py` covers domain guards, `--restart`, mac rejecting isaac, compose mounts, stack restart and stop. `test_launch_ros_cache.py` covers colcon cache and livox staging.
  - `tests/stack/test_*.py` (scoring; `test_heartbeat_clock.py` is skipped without rclpy).
  - The CPU job initializes only the `CL_isaaclab_sim` and `unitree_sdk2_python` submodules.
- **Manual GPU job:** `tests/isaaclab/run_*.sh` (smoke, livox interop, sensor transport, robot interop magpie/inspire, magpie couplings/transport) and `tests/stack/run_all.sh` (robocasa/isaac × arms/locomotion, results under `tests/results/stack-*`).
- **Container-only:** `tests/robocasa/run_in_container.sh` (`test_sensor_bridge.py`, `test_realsense_profile.py`) and `h1_robocasa/test_spawn_collision.py`.
- **colcon test per package, not in CI:**
  - `h12_slam`: `test_level_odom`, `test_mid360_frames` (needs the H1-2 URDF), `test_robot_geometry`;
  - `h12_bringup/test/test_real_lidar_config.py`;
  - `h12_skills/test/test_grasp_synthesis.py`;
  - `lmg_deploy/test/*`;
  - `h12_lowerbody_rl` (mypy, xmllint);
  - ament flake8, pep257 and copyright tests in most packages.
- Smoke flows in `docs/DOCKER_TESTING.md`: `docker_stack.sh up robocasa --headless`, `ros2 topic hz /clock`, Isaac `--task base --headless`, `wait_for_sim.py`.

### 6.5 Pre-port cleanup worth doing on camera_ready first
1. Re-pin `h12_safety_layer` to a commit that contains both `origin/camera_ready` and `origin/main`.
2. Fast-forward CL_Assets `main` to `a6c44e4` and CL_isaaclab_sim `main` to `6f89c64` (or accept the branch pins knowingly).
3. Merge `origin/main`'s `8709a1d` into camera_ready, or merge camera_ready into main.
4. Revert or decide on the livox `build.sh` local edit and fix the RobocasaDockerfile comment.
5. Update CLAUDE.md (§5) and bring AGENTS.md back into sync.
6. Delete the `__pycache__`-only leftovers, the merged `camera_parity` branch and worktree, and the backup branch and stash once reviewed.
