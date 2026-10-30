# Our G1

The Correll Lab's G1, as read from the robot on 2026-09-28, the layout of its
`rt/lowstate`, and what is still unknown about it. The models are Unitree's,
in `CL_Assets/g1` (its `README.md` has their provenance, the Dex3-1 joint
order and the loader quirks). How the laptop reaches the robot is in
[DDS and networking](dds.md).

| | |
| --- | --- |
| Variant | `mode_machine` 5: 29 DoF rev 1.0, 4010 wrist. Model: `CL_Assets/g1/g1_description/g1_29dof_with_hand_rev_1_0.{urdf,xml}` |
| Waist | 3 joints: yaw, roll, pitch |
| Hands | Dex3-1, left and right |
| Motion computer | `192.168.123.161`, hostname `Unitree`. Runs Unitree's control; not for our code |
| Dev computer | `192.168.123.164`, hostname `unitree-g1-nx`. `ssh unitree@192.168.123.164` |
| DDS | CycloneDDS 0.10.2, domain 0 |
| SDK | `unitree_sdk2_python` at `814556d` |
| Not yet recorded | serial number; firmware `version` field from `rt/lowstate` |

## Already running on the dev computer

`g1_deploy_onnx_ref` (pid 1261, so most likely started at boot) publishes
`rt/lowcmd` and `rt/dex3/{left,right}/cmd`, and calls `motion_switcher`, the
service that hands the motors from Unitree's controller to a user program. It
also reads `rt/lowstate`, which carries the remote's buttons, so a button combo
may start it. Find out what it is before running anything that publishes
commands: two publishers on `rt/lowcmd` fight over the motors. The first
session's step 6 lists every writer of the command topics before launching
([First real session](first-session.md)).

## mode_machine

`mode_machine` in `rt/lowstate` names the hardware variant. Ours is 5, and
`golem_safety` engages on no other value ([Safety layer](safety.md)).

| `mode_machine` | Waist joints |
| --- | --- |
| 5 | 3; matches `CL_Assets/g1/g1_description/` |

The full table is in unitree_ros, `robots/g1_description/README.md`.

## rt/lowstate

`tools/read_lowstate.py` prints it once a second (the first session's step 6).
`rt/lowstate` holds a small header, then the IMU, then the motors, then the
remote:

```
LowState_
├─ version        [2]   firmware/version numbers
├─ mode_pr              0 = ankles and waist as pitch/roll, 1 = raw motor pairs A/B
├─ mode_machine         hardware variant ID
├─ tick                 counter, keeps increasing while data is live
├─ imu_state
│   ├─ quaternion     [4]  orientation
│   ├─ gyroscope      [3]  angular velocity (rad/s)
│   ├─ accelerometer  [3]  linear acceleration
│   ├─ rpy            [3]  roll, pitch, yaw (rad)
│   └─ temperature
├─ motor_state   [35]   one per motor slot; the G1 uses 0-28
│   ├─ q               joint angle (rad)
│   ├─ dq              joint velocity (rad/s)
│   ├─ ddq             joint acceleration
│   ├─ tau_est         estimated torque (N·m)
│   ├─ temperature [2] two temperature readings
│   ├─ vol             voltage
│   └─ mode, motorstate, sensor [2]   motor mode and status bits
├─ wireless_remote [40] the remote's buttons and sticks, as raw bytes
└─ reserve, crc          spare fields and a checksum
```

Motor slots: left leg 0-5, right leg 6-11, waist 12-14 (yaw, roll, pitch), left
arm 15-21, right arm 22-28. `dq` is derived from the joint encoder and `tau_est`
from motor current; the joints have no velocity or torque sensors. The hands'
state topics carry the same motor fields, 7 motors per hand. The joint names
per slot are `golem_safety`'s `robots/g1.yaml`.

## Open hardware questions

These are unknowns about our unit, not about the models. They gate anything
torque-enabled:

- **Ankle and waist roll/pitch torque limits.** Unitree's URDF says 35 Nm, its
  MJCF 50 Nm (`CL_Assets/g1/README.md`, Torque limits disagree between URDF and
  MJCF). Neither is a measurement.
- **Whether the Dex3-1 `thumb_0` joint mirrors between hands.** The other six
  joints demonstrably do; `thumb_0` has a symmetric range, so the model cannot
  say (`CL_Assets/g1/README.md`, Mirroring between hands).
- **Quaternion convention** (w-first or w-last) in whatever observation builder
  we deploy. A wrong sense of "down" is painful to debug from a fall video.
