# Packages

The G1's own ROS 2 packages live in `core_ws/src/g1/`. `launch_ros.sh` builds
them in every `ros` container, whichever robot the wrappers selected, so one
container runs either robot's bringup. The G1 shares four packages with the H1-2, built for both robots: the
safety layer `golem_safety` and the arm controller `golem_upper_ik`, each with
a profile and configs per robot, the models `golem_ros2_models`, and the
bringup `golem_bringup`. Each package
below has its own README with the full details, except the launch-only
`golem_bringup`.

## G1 packages

| Package | What it does | Key nodes and launch files | README |
|---|---|---|---|
| `g1_locomotion` | HOMIE's walking policy on the legs, driven by `/cmd_vel`: waits for `golem_safety`'s heartbeat, pre-poses the legs, on the real robot holds until `/lowerbody/confirm_engage`, and in the sim releases the elastic band. | `locomotion_node` | [README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_locomotion/README.md) |
| `g1_dex3` | Dex3-1 hand driver. Bypasses the safety layer. | `dex3_hands` (`/<side>/hand/set_pose`, `open`, `close`, `release`); `hands.launch.py` | [README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_dex3/README.md) |
| `g1_navigation` | Odometry (FAST_LIO or sim ground truth), a 2D map (slam_toolbox) and Nav2; the frontier explorer. Sim only for now. | `g1_navigation.launch.py`, `livox_to_pointcloud`, `level_frame`, `body_to_pelvis`, `frontier_explore` | [README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_navigation/README.md) |

## Shared with the H1-2

| Package or path | What the G1 uses it for |
|---|---|
| `golem_bringup` | starts the whole stack, on the real G1 or in the sim; launch only. The G1's files are in `launch/g1/`: `g1_real_robot.launch.py` (real), `g1_sim.launch.py` (sim), `g1_core.launch.py` (shared). No README; each launch file's docstring, and `CLAUDE.md` §6. |
| `golem_ros2_models` | both robots' models from `CL_Assets`: the G1's URDF (and its ROS copy with `package://` URIs) and meshes, and its sphere-packed self-collision model and SRDF. `golem_upper_ik`, `g1_navigation` and the bringups' `robot_state_publisher` read them; `display.launch.py` shows a model in RViz with joint sliders ([README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/golem_ros2_models/README.md)). |
| `golem_safety` | the safety layer, node `safety_node`: the only writer of the body command topic, `rt/lowcmd`. It streams zero gains from boot, engages each input on its own first command, clips commands, holds quiet inputs and latches estops ([Safety layer](safety.md)). The G1's profile is `robots/g1.yaml`, its configs `config/g1/` ([README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/golem_safety/README.md)). |
| `golem_upper_ik` | the arm IK controller (Pink) for the 14 arm joints, with the waist held at its hold targets: `frame_task_server` (`/frame_task`, `/named_config`) and `joint_state_publisher` (the 29 body joints). A self-collision barrier in the IK and OMPL planning, from the G1's sphere model, as on the H1-2; its spheres cover the Dex3-1 palms but not the fingers. The G1's data is `golem_upper_ik/robots/g1.py`, its configs `config/g1/` ([README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/golem_upper_ik/README.md)). |
| `custom_ros_messages` | the actions and services (`FrameTask`, `NamedConfig`, `SetHandPose`, `SkillFrontierExplore`, ...). |
| `FAST_LIO` (package `fast_lio`) | lidar-inertial odometry for `g1_navigation` |
| `livox_ros_driver2` | the `CustomMsg` type of `/livox/lidar`; the driver itself is not in the G1 bringup yet |
| `unitree_ros2` | Unitree message definitions (`unitree_hg` and others) |

Outside `core_ws`:

- `unitree_sdk2_python/` is the Unitree DDS SDK that `g1_locomotion`,
  `g1_dex3` and the sim use. `golem_safety` and `golem_upper_ik` reach the
  same DDS topics as ROS 2 topics (ROS `/lowcmd` is DDS `rt/lowcmd`).
- `CL_Assets/g1` (in the `CL_Assets` submodule) holds the G1's models, which
  `golem_ros2_models` installs and the sim builds its robot from. The wrappers
  stop when a file a container reads is missing or still a Git LFS pointer
  (`git -C CL_Assets lfs pull`). `CL_Assets/g1/README.md` has their
  provenance, the Dex3-1 joint order and the loader quirks; our unit is
  [Our G1](hardware.md).
- HOMIE's policy is in `core_ws/src/g1/g1_locomotion/policy/`
  ([README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_locomotion/policy/README.md)), for
  non-commercial use only. The `ros` container mounts it at
  `/home/code/g1_locomotion`; the sim reads none of its files.

## Tests

The package tests run in the `ros` container
(`docker/scripts/docker_run.sh ros`), from the package directory,
`/home/code/core_ws/src/g1/<package>`. Each README's Tests section has the
command; start it with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, or pytest fails to
load the `anyio` plugin. The DDS tests use domain 97 and fake robot peers; some
tests are skipped when ROS is missing, and a skip is not a failure. `golem_safety` and `golem_upper_ik` run through `colcon test` in the
same container, on domains 91 and 92; their READMEs have the commands.
`golem_bringup`'s G1 launch logic is in `test_g1_launch.py`, `test_nav_launch.py`
and the G1 case of `test_bringup_packages.py` (whose other cases are the
H1-2's):

```bash
cd /home/code/core_ws/src/golem_bringup
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_g1_launch.py test/test_nav_launch.py test/test_bringup_packages.py
```

HOMIE's policy test needs no simulator:

```bash
cd /home/code/g1_locomotion && python3 test_homie_policy.py
```

The G1 sim's unit tests, the topic-surface probe and the acceptance tests run
from the host; `tests/README.md` lists them.
