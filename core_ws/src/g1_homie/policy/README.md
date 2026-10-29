# g1_homie/policy: HOMIE's walking policy

The walking policy for GOLEM's Unitree G1. It holds HOMIE's pretrained
locomotion policy and a MuJoCo script that runs it in simulation (sim2sim).
The `g1_homie` package around it runs it on the G1's legs
(`locomotion_node`); the G1 sim reads none of its files. The `ros` container
mounts this folder at `/home/code/g1_homie`.

The policy and the script come from [OpenHomie](https://github.com/InternRobotics/OpenHomie);
see [Credits](#credits). We kept only its MuJoCo deployment. The training code
(HomieRL), the real-robot deployment (HomieDeploy) and the teleoperation
hardware (HomieHardware) were removed. The robot model is the one in the
repo's `CL_Assets/g1`, so this folder has no model files of its own.

## Status

- **Walks in MuJoCo.** Tracking over 10 s, measured after the first 3 s:

    | Command | Result |
    |---|---|
    | stand | stays at 0.744 m |
    | forward 0.5 m/s | 0.46 m/s |
    | forward 1.0 m/s | 0.98 m/s |
    | backward 0.5 m/s | 0.48 m/s |
    | sideways 0.3 m/s | 0.23 m/s, drifting 0.17 rad/s in yaw |
    | turn 0.5 rad/s | 0.46 rad/s |
    | squat to 0.40 m | pelvis at 0.34 m |

- **Not on the real robot yet.** It will run on the laptop and send the legs'
  commands through `golem_safety`'s `split` mode.

## Contents

| Path | What it is |
|---|---|
| `deploy.onnx` | HOMIE's pretrained G1 policy (the example checkpoint from HomieDeploy) |
| `homie_policy.py` | The policy's observation, history and action, shared by everything that runs it |
| `mujoco_deploy_g1.py` | MuJoCo sim2sim loop: PD control at 500 Hz, policy at 50 Hz |
| `g1.yaml` | Model and policy paths, joints, PD gains, observation scales, commands and their trained ranges |
| `test_homie_policy.py` | Tests for `homie_policy.py`; they need no simulator |
| `LICENSE` | OpenHomie's license |

## Run it

It uses the `g1` conda env, the Python 3.10 env with the Unitree SDK. The first
time, install the extra packages:

```bash
conda activate g1
pip install onnxruntime mujoco==3.2.3 pyyaml
```

Then, from this folder:

```bash
python mujoco_deploy_g1.py                                # viewer, standing
python mujoco_deploy_g1.py --cmd 0.5 0 0                  # walk forward at 0.5 m/s
python mujoco_deploy_g1.py --cmd 0 0 0.5 --height 0.7     # turn at 0.5 rad/s, crouched a little
python mujoco_deploy_g1.py --headless --duration 10 --cmd 0.5 0 0
```

- `--cmd VX VY YAW` sets the velocity command in m/s and rad/s. Training covered
  vx from -0.8 to 1.2, vy from -0.5 to 0.5, and yaw from -0.8 to 0.8.
- `--height` sets the pelvis height target, from 0.24 m to 0.74 m. The default,
  0.74 m, is standing. Turning keeps its speed down to 0.70 m and slows below
  that. For a 0.5 rad/s turn it measured 0.46 rad/s at 0.74 m, 0.49 at 0.70 m,
  0.35 at 0.65 m and 0.01 at 0.60 m (over 10 s, after the first 3 s).
- `--headless` prints the pelvis position once a second, and stops if the robot
  falls.

## The policy

- **Output:** 12 leg joint targets, applied as `default_angles + 0.25 * action`.
- **Input:** the last 6 steps of a 76-value observation (456 values). Each step
  holds:
    - the velocity command;
    - the height command;
    - the pelvis angular velocity;
    - gravity in the pelvis frame;
    - the positions and velocities of 27 joints;
    - the previous action.
- **The 27 joints:** the G1's 29 body joints minus waist roll and waist pitch.
  That is 12 legs, waist yaw, and 14 arms. They are listed in `g1.yaml`.
- **IMU:** the policy was trained on the pelvis IMU (`imu_in_pelvis`). That IMU
  has no rotation relative to the pelvis, so the script reads the pelvis state.

## The policy module

`homie_policy.py` holds the policy's observation, history and action code, so
every program that runs the policy shares one copy. It needs only numpy,
onnxruntime and pyyaml, and works with numpy 1.26 and 2.x.

```python
from homie_policy import HomiePolicy

policy = HomiePolicy()                          # loads g1.yaml and deploy.onnx
kp, kd = policy.gains(joint_names)              # PD gains, matched by substring
cmd, height = policy.clip_command(cmd, height)  # to the trained ranges
targets = policy.step(cmd, height, gyro, quat_wxyz, q, dq)  # at 50 Hz
policy.reset()                                  # before starting again
```

- `q` and `dq` are the positions and velocities of `policy.policy_joints`, in
  that order. `gyro` is the pelvis angular velocity in the pelvis frame.
  `quat_wxyz` is the pelvis orientation with w first, the order of MuJoCo's
  `qpos` and Unitree's `imu_state.quaternion`.
- `targets` are position targets for `policy.leg_joints`, to track with the
  gains from `policy.gains()`.
- `reset()` zeros the observation history and the last action.
- `step()` raises ValueError if the observation has the wrong size or a value
  that is not finite, such as a NaN in `dq`. It checks before it changes the
  history or the last action, so later steps are unaffected.
- The onnxruntime session (`policy_session()`) runs each step on the calling
  thread: one intra-op and one inter-op thread, no spinning, nothing pinned
  to a CPU. onnxruntime's default instead pins a pool of threads as wide as
  the machine, the same CPUs in every process, and two of them spin at 100%
  between 50 Hz steps. With two processes running the policy that way on one
  host, the second one's threads on those two CPUs got no CPU time and it
  never finished loading. The actions are identical.
- `clip_command()` clips to `cmd_range` and `height_range` in `g1.yaml`, and
  rejects values that are not finite. `mujoco_deploy_g1.py` does not clip, so
  the sim can still try commands outside the trained ranges.
- Run the tests with `python test_homie_policy.py`, or with pytest.

## Changes from HOMIE's MuJoCo script

All values now come from HOMIE's training config (HomieRL's
`g1_29dof_config.py` and `legged_robot_config.py`).

- **Scales:**
    - Angular velocity: 0.5 instead of 0.25.
    - Yaw command: 0.5 instead of 0.25.

  With HOMIE's values the robot fell when walking backward and did not turn.
- **Height command:** 0.74 m (standing) instead of 0.34 m, which is a deep squat.
- **Model:** Unitree's unmodified G1 in `CL_Assets/g1` (`g1_29dof_with_hand_rev_1_0`,
  the variant our robot reports). HOMIE used its own copy, which differed in two
  ways:
    - It locked waist roll, waist pitch and the hands. Here those joints are
      actuated and held at 0 with the training gains.
    - It gave every joint a small damping, armature and friction loss. `g1.yaml`
      sets the same values in code.
- **Gains:** the waist and arms use the training gains, instead of the single
  kp 100 / kd 0.5 of HOMIE's script.
- **Joints:** looked up by name instead of by position in the model.
- **Policy:** loaded with onnxruntime. TorchScript and `legged_gym` are no longer
  needed.

## Credits

- **HOMIE** by Qingwei Ben, Feiyu Jia, Jia Zeng, Junting Dong, Dahua Lin and
  Jiangmiao Pang (Shanghai AI Laboratory and The Chinese University of Hong
  Kong). The policy, the MuJoCo script and its config come from
  [OpenHomie](https://github.com/InternRobotics/OpenHomie).
  [Paper](https://arxiv.org/abs/2502.13013),
  [project page](https://homietele.github.io/).

  ```bibtex
  @article{ben2025homie,
    title={HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit},
    author={Ben, Qingwei and Jia, Feiyu and Zeng, Jia and Dong, Junting and Lin, Dahua and Pang, Jiangmiao},
    journal={arXiv preprint arXiv:2502.13013},
    year={2025}
  }
  ```

- The MuJoCo script builds on Unitree's
  [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym).
- The G1 model is Unitree's, from
  [unitree_ros](https://github.com/unitreerobotics/unitree_ros) (see
  `CL_Assets/g1/README.md`).

## License

The code is for non-commercial use only. OpenHomie's `LICENSE` file is CC BY-NC
4.0, but its README says CC BY-NC-SA 4.0, which also requires changed versions
to be shared under the same license. Treat it as the stricter CC BY-NC-SA 4.0.
This license is different from the MIT license of GOLEM's G1 packages
(`dex_3`, `g1_homie`, `g1_navigation`), including the `g1_homie` code around
this folder.
