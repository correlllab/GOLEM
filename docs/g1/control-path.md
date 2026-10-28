# Control in sim and on the robot

The G1 sim speaks the robot's Unitree DDS interface exactly: the same topics,
the same `unitree_hg` message types, the same joint slots and the same CRC. So
the laptop stack runs unchanged against either one. Only the DDS domain and the
YAML limits and timings differ.

## The path of a command

```text
frame_task_server (arm IK) -- rt/safety/lowcmd_upper_in --+
                                                          +--> safety_node --> rt/arm_sdk  (arms only)
locomotion_node (HOMIE)   -- rt/safety/lowcmd_lower_in --+                --> rt/lowcmd   (walking, all 29 motors)
dex3_hands ------------------ rt/dex3/{left,right}/cmd ---------------------> hands (no safety layer)
everything reads <---------- rt/lowstate, rt/dex3/{left,right}/state
```

- **Arms only (the default).** The arm controller (`golem_upper_ik`) solves IK
  at `frequency.ctrl_hz` and streams commands at `frequency.pub_hz`. The safety
  layer (`golem_safety`) engages, ramps the `rt/arm_sdk` weight from 0 to 1, and
  relays the newest command at its `control.publish_hz`. The legs stay with
  Unitree's controller.
- **Walking (`use_locomotion:=true`).** The walking node (`g1_locomotion`)
  runs HOMIE's policy at 50 Hz on `/cmd_vel` and writes the legs. The safety
  layer merges legs and arms into one `rt/lowcmd` (split mode).
- **Hands.** `dex3_hands` writes the hand topics directly, at its `rate_hz`
  parameter.

The motor commands are DDS topics in Unitree's IDL. `golem_safety` and
`golem_upper_ik` read and write them as ROS 2 topics (ROS `/lowcmd` is DDS
`rt/lowcmd`, the same wire type), while `locomotion_node` and `dex3_hands` use
the Unitree SDK. The user-facing interface is ROS 2 too: `/cmd_vel`, the
`/frame_task` and `/named_config` actions, the hand services, `/joint_states`,
sensors.

## What sits behind the topics

- **On the robot:** Unitree's own software, which is not in this repository.
  It balances the robot and shares the waist and arms through the `rt/arm_sdk`
  weight. Before `rt/lowcmd` can drive the legs, the operator must release
  Unitree's controller by hand. Nothing in this stack does that step.
- **In the sim:** `golem_robocasa/g1/` (`g1_mujoco.py`, `unitree_interface.py`,
  `stand_in.py`). HOMIE's policy stands in for Unitree's controller and holds
  the robot up. `rt/arm_sdk` blends the waist and arms with it by the weight.
  The first valid `rt/lowcmd` releases the stand-in for good, which plays the
  operator's release. `g1/unitree_interface.py`'s docstring is the reference.

## Sim and real compared

The real and sim values are in pairs of config files; the table names them
instead of copying the numbers.

| | Real G1 | Sim |
|---|---|---|
| DDS domain | 0, pinned to the wired port (`G1_DDS_INTERFACE`) | 1 or another test domain; the sim refuses 0 |
| Who holds the robot before the stack | Unitree's controller | the HOMIE stand-in |
| Arm hand-over (`rt/arm_sdk`) | Unitree mixes by the weight; the method is not known | torque blend by the weight |
| Taking the legs (`rt/lowcmd`) | manual operator release first | the first valid frame releases the stand-in |
| Motor torques | the robot's motor drivers | PD law every physics step, in sim time |
| Silence on the command topics | not measured | `GOLEM_ARM_SDK_TIMEOUT` hands the arms back to the stand-in; `GOLEM_CMD_TIMEOUT` turns the motors off |
| Time | wall clock | sim time, paced to at most real time; the safety layer's stale times and ramps still run on the host's clock |
| Safety timings | `golem_safety/config/g1/arm_sdk.yaml`, `split.yaml` | `sim_arm_sdk.yaml`, `sim_split.yaml`: looser, since the sim can run slower than real time |
| Arm speed limits (`controller.v_lim`, `w_lim`, `dq_lim`) | `golem_upper_ik/config/g1/g1.yaml`: deliberately low for the first sessions | `sim.yaml` |
| Walking speed limits (`commands.vx`, `vy`, `yaw`) | `g1_locomotion/config/g1.yaml`: inside the trained ranges | `sim.yaml`: the trained ranges |
| Navigation | refused (real lidar not connected yet) | allowed with `use_locomotion:=true` |
| Status | nothing has run on the robot yet | arm, walk, nav and exploration tests pass |

## Not known yet

These depend on the robot's side, which is not in the repository:

- how Unitree's controller applies the `rt/arm_sdk` weight;
- what the robot does if `rt/arm_sdk` or `rt/lowcmd` simply stops;
- which frame (pelvis or torso) the real `rt/lowstate` IMU reports in;
- the real `rt/lowstate` rate and the real Dex3 hand behaviour.

More: [Safety layer](safety.md), [DDS and networking](dds.md), and the package
READMEs listed in [Packages](packages.md).
