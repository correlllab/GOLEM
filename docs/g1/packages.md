# Packages

The G1's ROS 2 packages live in `core_ws/src/g1/`. `launch_ros.sh` builds them
only for the G1 (`GOLEM_ROBOT=g1` or `--g1`), so the H1-2 never needs them.
Each package except the launch-only `g1_bringup` has its own README with the
full details.

## G1 packages

| Package | What it does | Key nodes and launch files | README |
|---|---|---|---|
| `g1_bringup` | Starts the whole stack, on the real G1 or in the sim. Launch only. | `g1_bringup.launch.py` (real), `g1_sim_bringup.launch.py` (sim), `g1_core.launch.py` (shared) | none; each launch file's docstring, and `CLAUDE.md` §6 |
| `g1_safety_layer` | The only writer of the body command topics. Clips commands, gates engagement, latches estops. See [Safety layer](safety.md). | `safety_node` | [README](../../core_ws/src/g1/g1_safety_layer/README.md) |
| `g1_control` | Arm IK controller (Pink) for the 14 arm joints. Holds the waist. | `frame_task_server` (`/frame_task`, `/named_config`), `dual_arm_server`, `joint_state_publisher` | [README](../../core_ws/src/g1/g1_control/README.md) |
| `g1_locomotion` | HOMIE's walking policy on the legs, driven by `/cmd_vel`. | `locomotion_node` | [README](../../core_ws/src/g1/g1_locomotion/README.md) |
| `g1_dex3` | Dex3-1 hand driver. Bypasses the safety layer. | `dex3_hands` (`/<side>/hand/set_pose`, `open`, `close`, `release`); `hands.launch.py` | [README](../../core_ws/src/g1/g1_dex3/README.md) |
| `g1_navigation` | Odometry (FAST_LIO or sim ground truth), a 2D map (slam_toolbox) and Nav2; the frontier explorer. Sim only for now. | `g1_navigation.launch.py`, `livox_to_pointcloud`, `level_frame`, `body_to_pelvis`, `frontier_explore` | [README](../../core_ws/src/g1/g1_navigation/README.md) |
| `g1_description` | G1 URDFs and meshes, the joint table (names, DDS slots, limits) and the DDS slot layout. | `display.launch.py` (RViz with joint sliders) | [README](../../core_ws/src/g1/g1_description/README.md) |

## Shared with the H1-2

| Package or path | What the G1 uses it for |
|---|---|
| `custom_ros_messages` | the actions and services (`FrameTask`, `NamedConfig`, `DualArm`, `SetHandPose`, `SkillFrontierExplore`, ...). `FrameTask`'s `relative` and `slow_mode` and `NamedConfig`'s `slow_mode` are not implemented on the G1: a goal that sets one aborts. |
| `FAST_LIO` (package `fast_lio`) | lidar-inertial odometry for `g1_navigation` |
| `livox_ros_driver2` | the `CustomMsg` type of `/livox/lidar`; the driver itself is not in the G1 bringup yet |
| `unitree_ros2` | Unitree message definitions (`unitree_hg` and others) |

Outside `core_ws`:

- `unitree_sdk2_python/` is the Unitree DDS SDK that every robot-facing node
  uses.
- `third_party/leprechaun_g1` (a submodule) holds the G1 model that
  `g1_description` installs and the sim builds its robot from. Without it the
  G1 wrappers stop with `git submodule update --init third_party/leprechaun_g1`.
- HOMIE's policy is in `core_ws/src/g1/g1_locomotion/policy/`
  ([README](../../core_ws/src/g1/g1_locomotion/policy/README.md)), for
  non-commercial use only. The containers mount it at `/home/code/g1_locomotion`.

## Tests

The package tests run in the G1's `ros` container
(`docker/scripts/docker_run.sh ros --g1`), from the package directory,
`/home/code/core_ws/src/g1/<package>`. Each README's Tests section has the
command; start it with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, or pytest fails to
load the `anyio` plugin. The DDS tests use domain 97 and fake robot peers; some
tests are skipped when ROS or `leprechaun_g1` is missing, and a skip is not a
failure. `g1_bringup`'s launch logic:

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
