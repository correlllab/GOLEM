# Packages

The G1's own ROS 2 packages live in `core_ws/src/g1/`. `launch_ros.sh` builds
them only for the G1 (`GOLEM_ROBOT=g1` or `--g1`), so the H1-2 never needs
them. Its safety layer and arm controller are the H1-2's: `golem_safety` and
`golem_upper_ik`, built for both robots, each with a profile and configs per
robot. Each package except the launch-only `g1_bringup` has its own README with
the full details.

## G1 packages

| Package | What it does | Key nodes and launch files | README |
|---|---|---|---|
| `g1_bringup` | Starts the whole stack, on the real G1 or in the sim. Launch only. | `g1_bringup.launch.py` (real), `g1_sim_bringup.launch.py` (sim), `g1_core.launch.py` (shared) | none; each launch file's docstring, and `CLAUDE.md` §6 |
| `g1_locomotion` | HOMIE's walking policy on the legs, driven by `/cmd_vel`. | `locomotion_node` | [README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_locomotion/README.md) |
| `g1_dex3` | Dex3-1 hand driver. Bypasses the safety layer. | `dex3_hands` (`/<side>/hand/set_pose`, `open`, `close`, `release`); `hands.launch.py` | [README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_dex3/README.md) |
| `g1_navigation` | Odometry (FAST_LIO or sim ground truth), a 2D map (slam_toolbox) and Nav2; the frontier explorer. Sim only for now. | `g1_navigation.launch.py`, `livox_to_pointcloud`, `level_frame`, `body_to_pelvis`, `frontier_explore` | [README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_navigation/README.md) |
| `g1_description` | G1 URDFs and meshes, the joint table (names, DDS slots, limits) and the DDS slot layout. | `display.launch.py` (RViz with joint sliders) | [README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_description/README.md) |

`g1_safety_layer` and `g1_control`, the G1's former safety layer and arm
controller, are still in `core_ws/src/g1/` and still build for the G1, but
nothing launches or depends on them: they are unused and awaiting deletion.

## Shared with the H1-2

| Package or path | What the G1 uses it for |
|---|---|
| `golem_safety` | the safety layer, node `safety_node`: the only writer of the body command topics. It clips commands, gates engagement, holds quiet inputs and latches estops ([Safety layer](safety.md)). The G1's profile is `robots/g1.yaml`, its configs `config/g1/` ([README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/golem_safety/README.md)). |
| `golem_upper_ik` | the arm IK controller (Pink) for the 14 arm joints, with the waist held at its hold targets: `frame_task_server` (`/frame_task`, `/named_config`) and `joint_state_publisher` (the 29 body joints). No self-collision model or planner on the G1 yet. The G1's data is `golem_upper_ik/robots/g1.py`, its configs `config/g1/` ([README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/golem_upper_ik/README.md)). |
| `custom_ros_messages` | the actions and services (`FrameTask`, `NamedConfig`, `SetHandPose`, `SkillFrontierExplore`, ...). |
| `FAST_LIO` (package `fast_lio`) | lidar-inertial odometry for `g1_navigation` |
| `livox_ros_driver2` | the `CustomMsg` type of `/livox/lidar`; the driver itself is not in the G1 bringup yet |
| `unitree_ros2` | Unitree message definitions (`unitree_hg` and others) |

Outside `core_ws`:

- `unitree_sdk2_python/` is the Unitree DDS SDK that `g1_locomotion`,
  `g1_dex3` and the sim use. `golem_safety` and `golem_upper_ik` reach the
  same DDS topics as ROS 2 topics (ROS `/lowcmd` is DDS `rt/lowcmd`).
- `third_party/leprechaun_g1` (a submodule) holds the G1 model that
  `g1_description` installs and the sim builds its robot from. Without it the
  G1 wrappers stop with `git submodule update --init third_party/leprechaun_g1`.
- HOMIE's policy is in `core_ws/src/g1/g1_locomotion/policy/`
  ([README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_locomotion/policy/README.md)), for
  non-commercial use only. The containers mount it at `/home/code/g1_locomotion`.

## Tests

The package tests run in the G1's `ros` container
(`docker/scripts/docker_run.sh ros --g1`), from the package directory,
`/home/code/core_ws/src/g1/<package>`. Each README's Tests section has the
command; start it with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, or pytest fails to
load the `anyio` plugin. The DDS tests use domain 97 and fake robot peers; some
tests are skipped when ROS or `leprechaun_g1` is missing, and a skip is not a
failure. `golem_safety` and `golem_upper_ik` run through `colcon test` in the
same container, on domains 91 and 92; their READMEs have the commands. `g1_bringup`'s launch logic:

```bash
cd /home/code/core_ws/src/g1/g1_bringup
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_locomotion_launch.py test/test_nav_launch.py
```

HOMIE's policy test needs no simulator:

```bash
cd /home/code/g1_locomotion && python3 test_homie_policy.py
```

The G1 sim's unit tests, the topic-surface probe and the acceptance tests run
from the host; `tests/README.md` lists them.
