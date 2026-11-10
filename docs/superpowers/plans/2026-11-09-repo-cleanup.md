# Repo cleanup: dead and redundant code

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Delete dead code and fold redundant code across the repo, on the user's request (2026-11-09: "audit the repo, look for dead and redundant code, then clean it up"), without changing robot behaviour except where the user chose to retire a feature.

**Architecture:** Four read-only audits (`.superpowers/sdd/2026-11-09-dead-code/audit-{A,B,C,D}.md`, with `audit-brief.md` for their rules) list every finding with its grep evidence, risk and confidence. Four tasks apply them area by area, each with tests and a review. Branch `repo_cleanup`, from `sim_speed` at 1e86e92.

**Spec:** this plan's "User decisions" section and the audits; CLAUDE.md §8 (dead code is deleted; comment rules).

## User decisions (2026-11-09)

- **Retire MJPC entirely:** `deploy_mjpc`, the `mujoco_mpc` submodule, the MJPC block of the `ros` image and its build plumbing (`MJPC_REF`, `rebuild_mjpc.sh`, `launch_ros.sh`'s MJPC hydration, the compose mounts), `policy:=mjpc`, `use_estimator` and the estimator node, `mjpc_sim.yaml`, and every doc and test of them. The `ros` image's MuJoCo 3.2.3 pin and OSMesa go too if nothing else in the image uses them (prove it).
- **Delete the sim experiment surface and the tether rotational hold:** `GOLEM_SIM_EXPERIMENT`, `/sim/ground_truth`, `/sim/reset_objects` (and what only they use in `measurement_bridge.py`), `GOLEM_SIM_TETHER_ROT_STIFFNESS`, `GOLEM_SIM_TETHER_ROT_BODY` and the band's rotational hold; compose, test clear-lists, CLAUDE.md §8's ground-truth text and the surface goldens follow in the same commit.
- **Keep:** the YOLO battery route in `golem_skills` (`BATTERY_USE_GEMINI` and what it gates), the H1-2's `walk` and `fame` policies, the unused safety configs (H1-2 `*_full`, `safety_full.yaml`, G1 `full.yaml`), every Docker image package (apt and pip) that is not part of MJPC.
- **Submodules:** list only. `CL_isaaclab_sim`, `custom_ros_messages`, `magpie_msgs`, `estop`, `cl_realsense`, `magpie_control` are not edited; their findings go into `docs/superpowers/specs/2026-11-09-submodule-dead-code.md` for their owners.

## Global Constraints

- Delete only what the audits (or your own grep, recorded) prove dead; a finding of risk "behaviour" is done only if a user decision above covers it; otherwise it is left and listed.
- Fold a redundancy only inside one package or where one package already depends on the other (CLAUDE.md §5; do not invent a cross-package library). Cross-package duplicates without such a dependency are listed, not folded.
- No change to robot behaviour, the sims' topic surface (except the experiment topics), safety logic, real-robot paths or Isaac.
- Vendored code, `policies/homie/` and historical records under `docs/superpowers/` are untouched.
- Folding code keeps its tests: a test of a removed duplicate moves to the shared code, not out of the suite.
- CLAUDE.md §8 comment rules; AGENTS.md stays identical to CLAUDE.md apart from its title.

### Task 1: golem_robocasa, tests/, tools/ (audit A)

- [ ] Apply audit A: every dead row; the redundancies #16-#19 (one shared robosuite placement module, the shared lowcmd command path in `unitree_common.py`, loop scaffolding in `sim_common.py`), #22 (one stub-env base), #25-#29 (test scaffolding: `tests/g1/common.sh` forwarders, `run_sim`/`map_scene`/`Recorder`, fake DDS classes, H1-2 `spawn()`, G1 mini runners); move `golem_robocasa/test_spawn_collision.py` into `tests/robocasa/` and its runner; make `test_lowstate_writer_is_faster` assert; the experiment surface and tether hold (user decision).
- [ ] Run: both robocasa suites, both surface probes (re-record only the experiment fields if the golden holds them, named), `tests/docker`, `tests/sim --h12/--g1`, `tests/stack robocasa arms`, `tests/g1 --walk`, `tests/g1 nav`.

### Task 2: golem_safety, golem_upper_ik, dex_3, golem_ros2_models (audit B)

- [ ] Apply audit B's dead and redundant rows except the safety configs the user keeps: `debug.yaml` and `joint_state_publisher`'s topic parameter (larger redundancy A), `frame_task_server.py`'s four goal loops (E), the dead parameters and methods (#3-#9), dex_3 #27-#28. `golem_safety` rows of risk "behaviour" stay unless a user decision covers them.
- [ ] Run each package's colcon tests, `tests/stack robocasa arms`, `tests/sim --h12/--g1`.

### Task 3: golem_lowerbody_rl, golem_slam, golem_skills, golem_bringup (audit C)

- [ ] Apply audit C except what the user keeps (YOLO route, walk/fame) and the MJPC rows (Task 4): `golem_lowerbody_rl/reference/`, the pinned grasp testing switches `CONFIRM_BEFORE_CONTACT`/`APPROACH_IMAGE_ONLY` and what only they use, the rviz displays, guard tests of removed code, duplicated helpers inside a package or along an existing dependency.
- [ ] Run each package's colcon tests, `tests/stack robocasa arms` and `locomotion`, `tests/sim --h12/--g1`, `tests/g1 nav`.

### Task 4: Retire MJPC; docker, docs, lmg_deploy, model_server (audit D)

- [ ] Retire MJPC (user decision) across `core_ws/src/deploy_mjpc`, the `mujoco_mpc` submodule (`git rm`, `.gitmodules`), Dockerfiles, compose, scripts, `golem_bringup` (`policy:=mjpc`, `use_estimator`, the estimator node, `h12_legs.MJPC`, `mjpc_sim.yaml`), tests and docs (CLAUDE.md §§2, 4, 8 MuJoCo pins, 10 pitfalls).
- [ ] Apply audit D's rows that are not image packages: docs (broken anchor, missing smoke test, copied lists, mkdocs extensions), AGENTS.md as a symlink to CLAUDE.md if every tool that reads it follows symlinks (check `tests/docker`), dead script code, `lmg_deploy`/`model_server` dead rows; write the submodule findings file.
- [ ] Rebuild the `ros` image; run `tests/docker`, the touched packages' colcon tests, `tests/stack robocasa arms` and `locomotion`, `tests/sim --h12/--g1`.
