# Test fixtures

Recorded data. The tests compare `golem_upper_ik` against these files; nothing in this
package imports the code they were recorded from, which no longer exists in this repo.
The tools that wrote them were deleted with it.

## Where they were recorded from

| Package | Where it lived | Recorded at |
|---|---|---|
| `h12_ros2_controller` | git submodule `core_ws/src/h12_ros2_controller`, `git@github.com:correlllab/h12_ros2_controller.git` | `b53a6b2b7fca20f8135d98d4825e0e390e549894` |
| `g1_control` | in-tree, `core_ws/src/g1/g1_control` | GOLEM `9dbcbb194d8a76e0aec3529b74d5e0079973b176` (its code is unchanged since `e5a83e0`, which touched READMEs only) |

## The files

| File | Holds | Read by |
|---|---|---|
| `old_controller/<robot>.json` (`h12`, `g1`) | The robot's joint tables (body, upper-body and IK joints, the old joint limits), its named configurations and start-up sequence, the planner's default frames, and for each carried controller config what that package's own `load_controller_config` returned from it: topics, network, controller, frequency, planner, gains, the clip limits and logging. The H1-2's configs are `safety_split`, `sim_safety_split`, `safety_full` and `debug`; the G1's are `g1_split` and `sim_split`. | `test_profile.py`, `test_config.py` |
| `old_controller/g1_pairing.json` | `g1_control`'s `config_pairing` verdict (`accepted`, or `refused` with its message) for every combination of 7 controller config names, 7 safety config names and `sim` (unset, false, true), without a walking policy, and its `default_controller_config` for every safety config name in the table. | `test_pairing.py` |
| `fk_<robot>.json` (`h12`, `g1`) | 50 configurations drawn with seed 20261027 and what the package's `RobotModel` and `RobotDynamics` computed at each, built from the URDF its `frame_task_server` loads: `q` (the body joints, rad, uniform within the URDF limits), `imu_quat` (w, x, y, z), `dq` (rad/s, uniform in [-1, 1]), `frames` (the placement of each end effector, planner frame and the IMU frame in the free-flyer model, `[x, y, z, rotation row by row]`), `reduced_frames` (the same in the reduced IK model, the other joints frozen at 0), `full_q`, `gravity` (N*m per body joint), `twist` (per end effector, `(vx, vy, vz, wx, wy, wz)` with the measured state at `q`) and, for the H1-2 only, `collision_free_reduced` (the G1 has no self-collision model). | `test_robot_model.py` |

## What was changed after recording

- `old_controller/g1_pairing.json`: the refusal messages named `g1_safety_layer` as the
  layer that owns split mode; they name `golem_safety` now.
- `old_controller/g1_pairing.json`: it held 10 controller and 10 safety config names. The
  cases of the names of configs that no longer exist (the controller configs `g1`,
  `g1.yaml` and `sim`, the safety configs of the G1 mode that shared the waist and arms with
  Unitree's controller, and their `default_controller_config` entries) are gone. The
  refusal messages that listed those configs list the remaining ones: `one of split,
  sim_split`, and `use sim_split.yaml with sim_*, g1_split.yaml with the others`. No verdict
  changed.
- `old_controller/g1.json`: `configs` held the controller configs `g1` and `sim` too; they
  no longer exist, and their entries are gone. The other values are unchanged.
- `old_controller/g1_pairing.json`: each combination was recorded twice, with and without
  `locomotion` (a walking policy on the legs, which needed the split configs and raised a
  `KeyError` with `sim` unset). `pairing_refusal` no longer takes it: the G1 bringups start
  the walking policy with the split configs they always run. The cases recorded with it, and
  the `locomotion` field, are gone: 147 of 294 remain, each `accepted` or `refused`. No
  verdict changed.
