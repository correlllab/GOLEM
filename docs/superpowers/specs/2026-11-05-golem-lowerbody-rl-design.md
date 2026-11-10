# One leg package for both robots (`golem_lowerbody_rl`) — design

Status: approved by the user on 2026-11-05 ("Okay").
Branch: `golem_lowerbody_rl`, on top of `dead_code_cleanup`.
Supersedes the "Legs: separate packages" row of
`docs/superpowers/specs/2026-10-31-shared-robot-stack-design.md`.
Audit it argues from: `docs/superpowers/specs/2026-11-05-golem-lowerbody-rl-audit.md`
(committed with this design; `path:line` citations there).

## Intent

The user: "the h12 lower body rl package is golem lower body rl and merges in
homie and the h12 policies following the same pattern as the safety layer and
upper body controller. Identical code, different config. The config should
determine how the observations and actions are made and interpreted."

So the leg controller joins `golem_safety`, `golem_upper_ik` and `golem_slam`:
one package, one node, one state machine and one policy runner for both robots.
A robot is a config directory; a policy is a config file that says how its
observation vector is built and how its action is turned into leg commands.
Nothing in the code names a robot or a policy.

## User decisions (2026-11-05)

1. `h12_lowerbody_rl` is renamed `golem_lowerbody_rl` (`git mv`, history kept)
   and absorbs `g1_homie`, which is then deleted.
2. **Transport: ROS for both robots**, as `golem_safety` and `golem_upper_ik`.
   The Unitree SDK / raw cyclonedds path of `g1_homie` goes. The `/lowstate`
   subscription gets its own callback group and executor thread so executor
   stalls do not starve it; the G1's latency fault uses tick lag (as
   `golem_upper_ik/core/ros_io.py`), not the DDS source timestamp.
3. **Drift is unified now**, not carried as config switches. Where the two
   state machines differ without a robot reason, the code has one behaviour,
   taking the safer side (table below). Config holds only real robot
   differences, sim-vs-real differences and policy properties. The closed-loop
   sim tests prove both robots.
4. **Runtimes: TorchScript and ONNX side by side**, chosen by the policy spec.
   No policy file changes. Backends import lazily: the Mac never imports
   onnxruntime.
5. **HOMIE's values are transcribed into our config**
   (`config/g1/policies/homie.yaml`), as `almi.yaml` transcribes ALMI's. The
   weights stay under their CC BY-NC-SA `LICENSE` and `README.md`.

## Goals

### Shape

1. **Package** `core_ws/src/golem_lowerbody_rl/`: one ROS node
   (`lowerbody_controller_node`), a ROS-free state machine (from
   `LegSupervisor`, gaining `g1_homie`'s checks), a policy runner (observation
   builder, action mapper, runtime backends), `launch/lowerbody.launch.py`, and
   `config/<robot>/`.
2. **Robot data comes from config, joints by name.** Leg and observed joints
   are names; slots, `mode_machine` and `mode_pr` come from `golem_safety`'s
   `robots/<robot>.yaml`, as `golem_upper_ik` reads them. No index literals
   (`OBS_JOINT_IDX`, `[:12]`, `POLICY_SLOTS`), no copy of the G1 joint table.
3. **Per-robot config** (`config/<robot>/`): `robot.yaml` (leg joints, IMU
   frame and trim, the policies the robot may run, its default policy),
   `sim.yaml` and `real.yaml` (the run differences: confirm, band, gap and
   latency limits, tick-rate window, command limits). Strict like
   `golem_upper_ik`'s: every key required, unknown keys refused.
4. **Per-policy spec** (`config/<robot>/policies/<name>.yaml`): runtime
   (`torchscript` | `onnx`, file, reset method, threads), rate, observed joints
   and their default pose, observation groups (each a term list with a history
   length), command processing, action mapping (scale, raw clip, target clip,
   gains), and when the policy is reset. A small term registry covers all four
   policies: `ang_vel`, `projected_gravity`, `velocity_command`,
   `height_command`, `joint_pos_rel`, `joint_vel`, `last_action`,
   `gait_phase`, `latent`. The audit's §7.2 sketch is the starting schema.
   Policies: ALMI, WALK, FAME on the H1-2 (each bringup runs ALMI; the Mac's
   `GOLEM_LOWERBODY` picks any), HOMIE on the G1. Only the active policy is
   loaded.
5. **Launch**: `lowerbody.launch.py` (`robot`, `config`, `policy`, `sim`,
   `use_sim_time`, `start_delay`) refuses a bad combination before any node
   starts (a run config whose sim/real does not match `sim`, a policy the robot
   does not list), as `upper_ik.launch.py` does. `h12_lowerbody.launch.py`
   (keeping `start_position_verified`, the `mjpc` exclusion and the MJPC
   estimator) and `g1_core.launch.py` (keeping its interlock and delays)
   include it. `lowerbody:=` on the H1-2 bringups becomes `policy:=`.
6. **Guarded**: `golem_lowerbody_rl` joins `tests/docker/test_shared_packages.py`,
   and a test refuses any `unitree_sdk2py`/`cyclonedds` import in it.

### One behaviour where the robots drifted (safer side)

| Behaviour | Unified behaviour (both robots) | From |
|---|---|---|
| Pre-pose ramp | cosine blend over 2 s, then the settle check (max error < 0.15 rad for 10 ticks, or 400 ticks after the ramp, then "NOT SETTLED" and hold anyway) | H1-2 (a real-robot hip-yaw estop) |
| Policy clock | 50 Hz on a 20 ms grid of `rt/lowstate`'s millisecond tick; ignore a sample whose tick did not advance, fault on a tick that goes back | G1 |
| Readiness before the policy starts | 1 s of steady `/lowstate`: the tick advances at the rate of the node's clock (sim time in the sims, wall time on the robots) within the run config's window, no gap over `max_state_gap`, the arm controller's commands fresh within 0.5 s. A tick that is not in ms of that clock never becomes ready, and the node logs why | G1 (CPU-starvation findings; in split mode arms with no command are limp), measured against the node clock so a sim's tick in sim-time ms passes |
| Confirm (real) | accepted only while holding the pre-pose and ready | G1 |
| Band (sim) | the policy is held until the first `/left_ee_pose` or `band.max_wait` after the pre-pose; the toggle is called once, once the policy has started and (`/left_ee_pose` or `max_wait` from policy start); one call, reply final; success resets the policy immediately; the `/left_ee_pose` subscription is dropped after the first message | G1 (`BandRelease`) |
| Faults | gap, non-finite, bad quaternion, fall, policy failure, tick back; tick-lag latency over `max_latency` while running only where the run config sets it (null in `sim.yaml`, as `golem_upper_ik` and `golem_safety` keep tick-lag checks off in the sims, whose tick is sim time; null in `real.yaml` until the pre-flight measurement sets a number) | union |
| Gravity | the quaternion is normalised, then `quat_rotate_inverse(q, [0,0,-1])`, for every policy | one formula (equal to both today's for a unit quaternion) |
| Engage | zero command, the policy's default height, policy reset | same intent |
| Topics | `/lowerbody/active_policy` (latched) and `/lowerbody/confirm_engage` on both; height on `/lowerbody/height` | H1-2 names; new neutral height name |
| Timing log | step spacing and latency every 10 s | G1 |
| `max_state_gap` | one value per run kind for both robots: sim 0.5 s; real 0.25 s until the pre-flight measurement on each robot | H1-2 (the only value measured on rclpy) |

Policy properties stay per policy, in the spec: ALMI's command gain, EMA and
onset reset, and its LSTM `reset_memory`; HOMIE's trained command range,
acceleration and height rate limits and its warm-up step; WALK's and ALMI's
gait phase; FAME's latent group. Command timeout (0.5 s) and non-finite
handling are shared.

## Non-goals

- `lmg_deploy` (only its comments that name the old packages change),
  `deploy_mjpc` (its estimator stays where it is), Isaac.
- **Isaac's H1-2 legs stop starting** (my call, within "ignore Isaac"): Isaac's
  `rt/lowstate` tick counts publishes, not milliseconds, so the readiness gate
  never passes there and the node logs that the tick is not in ms. Before, the
  legs ran in Isaac on a ROS timer. Restoring them is a one-line tick change in
  `CL_isaaclab_sim` (its own submodule), not part of this work.
- Retraining or re-exporting any policy.
- Any change to `golem_safety`'s behaviour or configs.

## Anti-goals

- Robot or policy names as branches in code, or YAML flags that pick between
  two drifted code paths (`ramp: cosine|linear`).
- An observation or action that changes by accident: each policy's
  observation vector and leg targets match today's for the same inputs, proven
  by golden traces recorded from the current code before it is rewritten
  (HOMIE's existing oracle test counts). The only intended numeric change is
  the gravity normalisation for a non-unit quaternion.
- Ground truth anywhere.

## Constraints

- CLAUDE.md applies, including §8 "Comments and docstrings" (a purpose line
  per definition, `SAFETY:` on safety invariants, an annotated `example.yaml`
  per config directory, launch arguments documented; disposable
  containers; never `ROS_DOMAIN_ID` 0; never touch the user's `golem_ros`
  container, `g1.rviz` or the `livox_ros_driver2` change).
- HOMIE's weights and `homie_policy.py` keep their `LICENSE` and `README.md`
  beside them (moved to `golem_lowerbody_rl/policies/homie/`);
  `homie_policy.py` stays only as the golden oracle and for its own MuJoCo
  script. CLAUDE.md §8's licence rule is rewritten for the new place.
- Stale installs: `h12_lowerbody_rl` and `g1_homie` join CLAUDE.md's pitfall
  `rm -rf` list.

## Done when

- Golden traces: every policy's observation and targets match the recorded
  ones; the supervisor tests (both packages' cases ported) pass.
- Package tests, `tests/docker`, `tests/stack` (locomotion),
  `tests/sim/test_sim_stack.sh --h12` and `--g1` (14/14 each),
  `tests/g1/test_sim_bringup.sh --walk` and `tests/g1/test_sim_nav.sh` pass.
- `g1_homie` and `h12_lowerbody_rl` no longer exist; nothing in the repo
  references them except CLAUDE.md's stale-install text and the dated
  history under `docs/superpowers/`.
- The Mac launch builds its graph in the x86 image with each `GOLEM_LOWERBODY`.

## Before the real robots (not part of done)

- G1: measure the leg node's `/lowstate` gap and tick lag on the laptop under
  the ROS path; set `real.yaml`'s `max_state_gap` and `max_latency` from it.
  (The G1 loses `g1_homie`'s 25 ms source-timestamp latency fault: it measured
  write-to-publish, which a ROS subscriber on Humble cannot see.)
- H1-2: the existing gap measurement, now also the tick lag.

## Left to the builder

- Module layout, the term registry's code shape, file names inside
  `config/<robot>/`, the exact spec schema (audit §7.2 as a start).
- Whether `tests/stack`'s legs-alone check starts the arm controller (the
  readiness gate needs its commands) or feeds an upper command itself.
- Whether the compose bind mount of HOMIE's policy is still needed once the
  package installs its weights.
- The package licence: `MIT` (as `g1_homie`), flagged for the user.
- The audit predates the comment pass and the dead-code commits on
  `dead_code_cleanup`; its line numbers are approximate, its facts stand.
