# G1_GOLEM → GOLEM camera_ready: audit summary (2026-10-26)

Compared HEAD to HEAD: G1_GOLEM `main` @ `df64d43` vs GOLEM `camera_ready` @ `42d6559`
(local = origin). Nothing was built or run, and neither repo was modified.
Detail reports in this directory: `1_g1_robot_packages.md`, `2_g1_bringup_nav_msgs_memory.md`,
`3_sim_tests_demo.md`, `4_docker_infra_shared.md`, `5_golem_camera_ready_delta.md`.

## Lineage
- G1_GOLEM's history was rewritten, so no SHAs are shared. It forked from GOLEM
  `origin/spatial-memory-robocasa-benchmark` @ `c0af423` (2026-07-23). That branch's merge-base
  with camera_ready is `c00feca` (2026-07-16).
- **G1 side:** 13 spatial-memory commits (Xuan) that never reached camera_ready, then about
  200 G1 commits (niraj, 09-29 to 10-06).
- **camera_ready side:** 140 commits since `c00feca`. The main changes:
  - every build input pinned and locked;
  - HAMS renamed to GOLEM;
  - the lmg policies, grasp synthesis and the battery task;
  - an h12_slam rewrite.
- The imported copies match the GOLEM pins at the fork:
  - `g1_control` = h12_ros2_controller `87bca34`
  - `g1_safety_layer` = h12_safety_layer `8ad5988`
  - `g1_msgs` = custom_ros_messages `bb54bbd`, on the unmerged `spatial-memory-retrieval-action` branch

## Inventory and verdict

| G1_GOLEM | What it is | Verdict for GOLEM |
|---|---|---|
| `g1_control` (8.2k LOC) | arm IK on 14 joints, follow gate, config pairing | **Bring as new package.** A G1-only fork; every shared file differs. |
| `g1_safety_layer` (2.3k) | `rt/arm_sdk` (weight slot 29) + split/full modes, engage gate | **Bring as new package.** Don't merge with h12_safety_layer now. |
| `g1_locomotion` | HOMIE ONNX walking policy | **Bring.** `policy/` is CC BY-NC 4.0, so keep its LICENSE and credits. Needs onnxruntime. |
| `g1_dex3` | Dex3-1 hands on `rt/dex3/*` | **Bring.** |
| `g1_description` | URDFs through a symlink to the `leprechaun_g1` submodule | **Bring, plus submodule** at `third_party/leprechaun_g1`. Its setup.py raises when the submodule is missing, so the H1 colcon build must not break. |
| `g1_bringup` | `g1_core` with interlocks and 13 launch tests | **Bring as separate package.** Not merged into h12_bringup. |
| `g1_navigation` | FAST_LIO + identity map→odom + Nav2 RPP + standalone frontier_explore | **Bring as separate package.** Never run it with h12_skills' explorer on one robot. |
| `g1_msgs` | renamed custom_ros_messages snapshot | **Don't bring the package.** Add `SetHandPose` (and `SkillRetrieveMemory`) to custom_ros_messages. g1_control must handle or reject the newer FrameTask `relative`/`slow_mode` fields. |
| `g1_spatial_memory` + `spatial_memory/` + SpatialMemory/VLMaps Dockerfiles | robot-agnostic spatial memory (Xuan's branch, not in camera_ready) | **Separate decision.** Agree it with that branch's owner. |
| `sim/` | G1 RoboCasa sim: HOMIE stand-in, arm_sdk/dex3 DDS, MID-360 + D435 | **Bring as sibling `g1_robocasa/`** with its own compose service. Adopt GOLEM's `measurement_bridge.py`, a strict superset of G1's. |
| `tests/` (4 acceptance tests, about 4.3 min) | `--network none`, a shared network namespace, `PASSED n/n` | **Bring as `tests/g1/`** with GOLEM image names and output paths. |
| `demo/walk`, `demo/navigation` | demo scripts; mp4s in LFS (46 MB) | Bring the scripts. The videos are optional (they need LFS). |
| `model_server`, `cl_realsense`, unitree/livox/sdk pins | shared | **Keep GOLEM's versions.** camera_ready is newer, and the pins are identical. |
| `FAST_LIO` | G1 restored upstream `mapping.launch.py` | Optional and harmless. |
| MkDocs site + Pages workflow, `docs/` | G1-specific docs | Bring the G1 docs as `docs/g1/`. The site is optional. |

## Port blockers in camera_ready (verified)
1. `docker/scripts/docker_common.sh:25` aborts on any `HAMS_*` env var. G1 reads 11 of them.
   - Reuse the GOLEM names `SIM_ODOM`, `SPAWN_BACKOFF`, `CMD_TIMEOUT` and `CAMERAS`.
   - `HAMS_CAMERA_SIZE` vs `GOLEM_CAMERA_SCALE` is the same idea in different units, so it needs its own name.
2. `.gitignore:70` `core` silently ignores 26 tracked G1 files under `*/core/`. Needs `!core/`.
3. `tests/docker/test_source_pins.py` rejects unlocked `pip install`s, so onnxruntime (1.23.2)
   and pynput go in as new numbered lock steps. Check onnxruntime against `protobuf==3.19.6`.
4. `prepare_mounts` (`docker_common.sh:81-109`) requires H1-only assets for every service.
5. `tests/stack/run.sh:69-73` expects `h12_mujoco.py` exactly once in `launch_robocasa.sh`.
   `tests/docker/test_workflows.py` forbids X11 mounts in the base robocasa service. Use a separate launcher and service.
6. The ros image lacks onnxruntime, and the robocasa image lacks it too. GOLEM already has
   everything else the G1 needs: Nav2, slam_toolbox, Pinocchio, Pink, OMPL and the Livox messages.
7. **DDS:**
   - The real G1 is commanded from the laptop's `ros` container on domain 0, unlike the H1, which runs natively onboard. G1's interface pin and UP/IPv4 checks must go into `check_domain`.
   - The two stacks share node, action and DDS topic names with identical `unitree_hg` types, so the rule is one robot per domain.

## Untested risks (inferred)
- G1 was validated on robosuite/robocasa master (about 10-06). GOLEM pins `5ce6643`/`4f8a298` from 09-17.
- GOLEM locks Pinocchio 2.7 + pin-pink 4.4.0. G1 ran unlocked; the API calls exist in 2.7 (checked by grep).
- No G1 code has run on the real robot yet. The arms-only bringup is ready for its first session.

## Pre-existing camera_ready issues (not caused by the port)
1. **Safety-layer pin regression (HIGH, verified).** `27a7c0b` (2026-10-20, HumanoidDesktop)
   moved h12_safety_layer `0eab20b` (submodule `origin/camera_ready`) to `ad06903` (`origin/main`).
   - Dropped:
     - `5b08888` steady-time heartbeat
     - `e0396fc` newest-only relay + late-lowstate estop
     - `58d0f99` silent `rt/lowcmd` after estop
     - `5182fbe` estop reason logging
     - `0eab20b` 95 Nm ankle-pitch estop for the Mid360v2 policy
   - main added only a logger (`d3d2725`).
   - CLAUDE.md's "stops publishing rt/lowcmd after estop" invariant is false at the current pin.
   - Reported, not verified by me: the controller pin bump also dropped the controller-side
     estop (`deacb9a`), so nothing e-stops on stale lowstate.
   - Fix: merge the submodule's camera_ready into main, then re-pin.
2. `origin/main` has `8709a1d` (a safety-layer pin bump) that camera_ready lacks (reported).
3. livox `build.sh` has a local `--symlink-install` edit. It leaks into local robocasa images
   (`RobocasaDockerfile:147-155`). Fix: `git -C core_ws/src/livox_ros_driver2 checkout -- build.sh`.
4. CL_Assets is pinned only on `origin/grasp_sweep_assets`. CL_isaaclab_sim is pinned only on its
   `origin/camera_ready`. Both can fast-forward their main.
5. CLAUDE.md has about 11 stale claims; AGENTS.md is older still. `core_ws/src/h1_bringup/` and
   `locomotion_game_deploy/` are ignored `__pycache__` leftovers and can be deleted.

## Decisions (user, 2026-10-26)
| Item | Decision |
|---|---|
| Interfaces | Extend `custom_ros_messages` (`SetHandPose`, `SkillRetrieveMemory`); no `g1_msgs` |
| Safety layer | Separate `g1_safety_layer` now; unified `golem_safety_layer` is a later follow-up |
| Arm controller | Separate `g1_control` |
| Lower body | Separate `g1_locomotion` |
| Bringup | Separate `g1_bringup` |
| Navigation | Separate `g1_navigation` |
| Frontier explorer | G1's inside g1_navigation now; one shared explorer later |
| Sim | **One runner for both robots**, `--h12` / `--g1`, h12 default |
| Spatial memory | Shared standalone, robot-agnostic package + benchmark |
| Scope extras | Spatial memory and the G1 docs site in scope; demo videos out |
| History | Copy in steps; each commit cites the G1_GOLEM SHA |
| Safety-pin regression | Open |

## Proposed shape of the port
- **Robot selection:** `GOLEM_ROBOT=h12|g1` (default `h12`, so H1 is unchanged).
  - A G1 compose service and launcher.
  - G1 container names distinct from `golem_ros`/`golem_sim_robocasa`.
  - `docker_stack.sh` made robot-aware.
- **Layout:**
  - `core_ws/src/g1/` for the G1 packages
  - `third_party/leprechaun_g1` (the symlink works unchanged)
  - `g1_robocasa/` for the G1 sim
  - `tests/g1/` for the acceptance tests
  - GOLEM keeps its other vendored paths
- **Order:**
  1. Infra prep: `.gitignore`, lock steps, mounts, env-var names, DDS interface check.
  2. Interfaces: custom_ros_messages PR + pin bump.
  3. Description + safety + control + hands.
  4. Bringup.
  5. Sim + acceptance tests: arms, then walk.
  6. Locomotion.
  7. Navigation + explore.
  8. Docs (CLAUDE.md sections, `docs/g1/`).
- Gate each step on GOLEM's CPU CI tests (`tests/docker`, `tests/isaaclab`,
  `tests/robocasa/test_loop_timing.py`, `tests/stack` scoring), the G1 package tests and the G1
  acceptance tests.
