# Test fixtures

Recorded data. The tests compare `golem_safety` against these files; nothing in this
package imports the code they were recorded from, which no longer exists in this repo.
The tools that wrote them were deleted with it.

## Where they were recorded from

| Package | Where it lived | Recorded at |
|---|---|---|
| `h12_safety_layer` | git submodule `core_ws/src/h12_safety_layer`, `git@github.com:correlllab/h12_safety_layer.git` | `a178f0bfdf74606fd11fdcf561f5f49fcadda6dc` |
| `g1_safety_layer` | in-tree, `core_ws/src/g1/g1_safety_layer` | GOLEM `9dbcbb194d8a76e0aec3529b74d5e0079973b176` (its code is unchanged since `e5a83e0`, which touched READMEs only) |
| `g1_description` | in-tree, `core_ws/src/g1/g1_description` (its `joints.py` generated from `leprechaun_g1`'s `g1_29dof_with_hand_rev_1_0.urdf`) | GOLEM `7a3c8a3f955fc8f48a74355fa89b39bfd1547fef` |

## The files

| File | Holds | Read by |
|---|---|---|
| `h12_joint_limits.json` | The H1-2's joint table as `h12_safety_layer/core/joint_limits.py` held it: `joint_names`, `position_limits` (`low`/`high`, rad), `velocity_limits` (rad/s), `torque_limits` (N*m), in slot order. It is the H1-2's table, not the URDF's: shoulder roll is +-0.38 rad here and +-0.19 rad in `h1_2_magpie.urdf`. | `test_robot_data.py`: `robots/h12.yaml` must hold the same values |
| `g1_joint_limits.json` | The G1's body joint table as `g1_description.joints.BODY` held it, in `h12_joint_limits.json`'s format and slot order. These are Unitree's URDF values. `tools/g1_joint_limits.py` holds the same values, and `tools/generate_robot_data.py` writes `robots/g1.yaml` from it. | `test_robot_data.py`: `robots/g1.yaml` must hold the same values |
| `limits/<robot>/<config>.json` | One file per safety config of `h12_safety_layer` (`h12/`) and `g1_safety_layer` (`g1/`), from that package's own `load_config`. `limits`: the derived clip and estop arrays by body slot (`q_clip_low`, `q_clip_high`, `dq_clip`, `tau_clip`, `q_estop_low`, `q_estop_high`, `dq_estop`, `tau_estop`) and the single numbers `kp_max` and `kd_max`. `old`: the other values the old loader returned, under their old meaning: `mode`, `topics` (`rt/` names), `publish_hz`, the stale times, `state_lag_ms`, the engage gate, `estop_response` and `logging`. | `test_limit_parity.py`, against `golem_safety_dump_config` |
| `lowcmd_crc.json` | `commands`: four `unitree_hg` `LowCmd` messages with known field values, one list per field over the 35 motor slots, and the CRC `unitree_sdk2py`'s `CRC().Crc` computed for each (field values rounded to float32 first). `old_commands`: the estop command each old layer wrote, with the header `golem_safety` stamps (`mode_machine` 6 for the H1-2, 5 for the G1) and the SDK's CRC for it: `h12_estop` is `make_estop_cmd(LowCmdDefault())` of `h12_safety_layer`, `g1_damp` is `make_estop_cmd(template, kd)` of `g1_safety_layer` with its input in `kd_input`. The `generator` field names `tools/crc_fixtures.py`, the tool that wrote the file and has been deleted. | `test_lowcmd.cpp` |

## What was changed after recording

- `limits/*/*.json`: `old.logging.base_dir` was `/tmp/h12_safety_layer` and
  `/tmp/g1_safety_layer`; it is `/tmp/golem_safety` now, the value every config carries.
- `limits/*/*.json`: `old` held the old loaders' settings of the G1 mode that shared the
  waist and arms with Unitree's controller through a blend weight in slot 29 (null on the
  H1-2). `golem_safety` no longer has that mode: the entry is gone, and so are the fixtures
  of its two G1 configs. The other values are unchanged.
- `lowcmd_crc.json`: `commands` held a fifth message, a G1 command with that blend weight
  in slot 29; it is gone. The other four and `old_commands` are unchanged.
- `h12_joint_limits.json`: `source` held `h12_safety_layer/core/joint_limits.py`; it points
  here. The table's values are unchanged. `tools/h12_joint_limits.py` is a copy of that
  file, and `tools/generate_robot_data.py` writes `robots/h12.yaml` from it.
