# Control in sim and on the robot

The G1 sim speaks the robot's Unitree DDS interface exactly: the same topics,
the same `unitree_hg` message types, the same joint slots and the same CRC. So
the laptop stack runs unchanged against either one. Only the DDS domain and the
YAML limits and timings differ.

## The path of a command

```text
frame_task_server (arm IK) -- rt/safety/lowcmd_upper_in --+
                                                          +--> safety_node --> rt/lowcmd  (all 29 motors)
locomotion_node (HOMIE)   -- rt/safety/lowcmd_lower_in --+
dex3_hands ------------------ rt/dex3/{left,right}/cmd ---------------------> hands (no safety layer)
everything reads <---------- rt/lowstate, rt/dex3/{left,right}/state
```

- **The safety layer** (`golem_safety`, split mode) writes every body motor on
  `rt/lowcmd`: the waist and arms from the arm controller's input, the legs
  from the walking node's. From boot it writes zero gains on every motor, and
  each input's slots carry its commands from that input's first command on
  ([Safety layer](safety.md), Start-up).
- **Arms.** The arm controller (`golem_upper_ik`) solves IK at
  `frequency.ctrl_hz` and streams commands at `frequency.pub_hz`. The safety
  layer relays the newest one at its `control.publish_hz`.
- **Legs.** The walking node (`g1_homie`) waits for the safety layer's
  heartbeat, moves the legs to HOMIE's default pose over 2 s, then runs HOMIE's
  policy at 50 Hz on `/cmd_vel`; on the real robot it holds the default pose
  until the operator calls `/lowerbody/confirm_engage`. The sim bringup always
  starts it, the real bringup only with `start_position_verified:=true`; without
  it the leg motors stay off.
- **Hands.** `dex3_hands` writes the hand topics directly, at its `rate_hz`
  parameter.

The motor commands are DDS topics in Unitree's IDL. `golem_safety` and
`golem_upper_ik` read and write them as ROS 2 topics (ROS `/lowcmd` is DDS
`rt/lowcmd`, the same wire type), while `locomotion_node` and `dex3_hands` use
the Unitree SDK. The user-facing interface is ROS 2 too: `/cmd_vel`, the
`/frame_task` and `/named_config` actions, the hand services,
`/lowerbody/confirm_engage`, `/joint_states`, sensors.

## What sits behind the topics

- **On the robot:** Unitree's own software, which is not in this repository.
  The stack runs the G1 in Unitree's debug mode: Unitree's own controller is
  released, and nothing but `rt/lowcmd` drives the body motors. The operator
  puts the robot into debug mode before the bringup starts; nothing in this
  stack does that step or checks it
  ([First real session](first-session.md#entering-debug-mode-not-yet-written)).
- **In the sim:** `golem_robocasa/g1/` (`g1_mujoco.py`, `unitree_interface.py`)
  plays the robot in debug mode. Every motor is off (zero torque) until the
  first `rt/lowcmd` with a valid CRC, `mode_machine` 5 and `mode_pr` 0; from
  then on `rt/lowcmd` drives every body motor. Meanwhile an elastic band on the
  torso holds the robot up, as the H1-2 sim's does: the walking node releases it
  through `/elastic_band/toggle` once its policy runs and the arm controller has
  finished its start-up moves, or `band.max_wait` after the policy started
  (SPACE in the viewer toggles it by hand).
  `g1/unitree_interface.py`'s and `g1_mujoco.py`'s docstrings are the
  reference.

## Sim and real compared

The real and sim values are in pairs of config files; the table names them
instead of copying the numbers.

| | Real G1 | Sim |
|---|---|---|
| DDS domain | 0, pinned to the wired port (`G1_DDS_INTERFACE`) | 1 or another test domain; the sim refuses 0 |
| Before the bringup | Unitree's debug mode, entered by the operator (the procedure is not written yet) | every motor off |
| Who holds the robot until the legs stand | a gantry | the elastic band, until the walking node releases it |
| Motor torques | the robot's motor drivers | PD law every physics step, in sim time |
| Silence on `rt/lowcmd` | not measured | `GOLEM_CMD_TIMEOUT` turns the motors off |
| Time | wall clock | sim time, paced to at most real time; the safety layer's stale times still run on the host's clock |
| Safety timings | `golem_safety/config/g1/split.yaml` | `sim_split.yaml`: looser, since the sim can run slower than real time |
| Arm speed limits (`controller.v_lim`, `w_lim`, `dq_lim`) | `golem_upper_ik/config/g1/g1_split.yaml`: deliberately low for the first sessions | `sim_split.yaml` |
| The walking node | `g1_homie/config/g1.yaml`: starts only with `start_position_verified:=true`, holds the default pose until `/lowerbody/confirm_engage` | `sim.yaml`: always starts, no confirm, releases the band |
| Walking speed limits (`commands.vx`, `vy`, `yaw`) | `g1.yaml`: inside the trained ranges | `sim.yaml`: the trained ranges |
| Navigation | refused (real lidar not connected yet) | allowed (`use_nav:=true`) |
| Status | nothing has run on the robot yet | arm, walk, nav and exploration tests pass |

## Not known yet

These depend on the robot's side, which is not in the repository:

- how to enter debug mode on our G1, and how to confirm it
  ([First real session](first-session.md#entering-debug-mode-not-yet-written));
- what the robot does if `rt/lowcmd` simply stops;
- which frame (pelvis or torso) the real `rt/lowstate` IMU reports in;
- the real `rt/lowstate` rate and the real Dex3 hand behaviour.

More: [Safety layer](safety.md), [DDS and networking](dds.md), and the package
READMEs listed in [Packages](packages.md).
