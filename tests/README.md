# Simulator regression tests

All simulator regression and acceptance tests and repeatable validation
commands live here, outside the simulator submodule. CPU tests run on every
relevant pull request through `.github/workflows/simulator-tests.yml`; select
its `cpu` job as a required branch-protection check to gate merges.

## Directories

| Path | What it tests | README |
|---|---|---|
| `tests/isaaclab/` | the Isaac simulator (H1-2) | this page |
| `tests/robocasa/` | the RoboCasa sim: loop, sensors, robot selection and task handling (H1-2) | `tests/robocasa/README.md` |
| `tests/robocasa/g1/` | the G1 sim's unit tests | [The G1](#the-g1) |
| `tests/robocasa/surface/` | the topic surface of the RoboCasa sim, per robot, against a golden | `tests/robocasa/surface/README.md` |
| `tests/stack/` | H1-2 sim + ROS stack integration (arms, locomotion) | `tests/stack/README.md` |
| `tests/g1/` | the G1 acceptance tests: sim + whole G1 bringup | `tests/g1/README.md` |
| `tests/docker/` | the wrappers, compose files, launchers and pins (CPU) | this page |

No test runs on DDS domain 0. The H1-2 tests use the domains shown on this
page and in `tests/stack/README.md`; the G1 tests and the surface probe use
their own (below).

## CPU tests

The Isaac suite needs Python 3.11, CPU PyTorch, NumPy, Pillow and CycloneDDS 0.10.2
plus the checked-out Unitree SDK. It does not start Kit or DDS participants.
See the workflow for the exact dependency setup.

```bash
PYTHONPATH=CL_isaaclab_sim:unitree_sdk2_python python -m unittest discover -s tests/isaaclab -p 'test_*.py' -v
PYTHONPATH=golem_robocasa python -m unittest discover -s tests/robocasa -p test_loop_timing.py -v
python -m unittest discover -s tests/docker -v
```

Using existing Docker images:

```bash
ROS_DOMAIN_ID=187 tests/isaaclab/run_in_container.sh
ROS_DOMAIN_ID=187 tests/robocasa/run_in_container.sh
```

`tests/robocasa/run_in_container.sh` runs the H1-2 sim's tests whatever
robot the shell or `docker/.env` selects.

## NVIDIA / ROS integration

These require the GOLEM Isaac and ROS images, initialized assets including LFS,
Docker NVIDIA runtime, and an NVIDIA GPU for the scene smoke test. Run them on
an isolated GPU worker, not an untrusted pull request with access to a real
robot network. The wrappers reject DDS domain 0 and use disposable containers.
They do not replace the normal simulator containers.

```bash
ROS_DOMAIN_ID=187 tests/isaaclab/run_smoke.sh
ROS_DOMAIN_ID=188 tests/isaaclab/run_livox_interop.sh
ROS_DOMAIN_ID=189 tests/isaaclab/run_sensor_transport.sh
ROS_DOMAIN_ID=191 tests/isaaclab/run_robot_interop.sh
```

`run_smoke.sh` requires the actual Isaac Lab scene to complete 50 steps and
shut down normally. Its log is `tests/results/isaac-smoke.log`; a timeout fails.
`run_livox_interop.sh` checks that Isaac's CycloneDDS CustomMsg publisher is
decoded by the ROS container's generated Humble Livox message type.

`run_robot_interop.sh` publishes native ROS LowCmd and checks wrist motion,
Magpie gripper services/actions (or Inspire commands when selected), measured
hand state and all 19 sensor topics. It releases body motors on exit.
The workflow has an explicit manual GPU job for a prepared self-hosted
runner labelled `nvidia`; it is not run automatically on untrusted pull requests.

The CPU suite cannot establish physical asset fidelity, real-time performance,
or controller stability. Those need the GPU/ROS integration environment and
robot-specific acceptance tests when the CL_Assets pin changes.

## Robot variants

Isaac defaults to the floating-base Magpie H1-2 in `CL_Assets`. Use
`docker/scripts/docker_run.sh isaac --hand_type inspire --headless` for the
Inspire fixture. `--fix_base` fixes the Magpie root for isolated motor tests;
normal launches leave it free and require a lower-body controller.

Magpie exposes `/{left,right}/gripper/state`, the gripper services and DeliGrasp
through a native ROS process in the Isaac container. Its isolated Humble/Python runtime and messages are built into the Isaac image
from the pinned `magpie_msgs` source. Rebuild Isaac after changing that interface
package. Humble is required for discovery compatibility with the ROS container. Physics and ROS exchange commands/state over a
private local socket; ROS callbacks do not access simulator tensors.

The gripper aperture calibration and torque-to-force estimate are approximate,
as in RoboCasa. These tests establish control/transport, not calibrated grasp
forces or hardware equivalence under contact.

```bash
ROS_DOMAIN_ID=191 tests/isaaclab/run_robot_interop.sh magpie
ROS_DOMAIN_ID=192 tests/isaaclab/run_robot_interop.sh inspire
```

## RoboCasa topic surface

The RoboCasa sim must publish what the real robot publishes. The probe
measures what the sim publishes and consumes over ROS 2 and Unitree DDS and
compares it with a checked-in golden per robot (`h12.json`, `g1.json`). It
always uses DDS domain 194, one probe at a time.

```bash
tests/robocasa/surface/run.sh h12            # compare against h12.json
tests/robocasa/surface/run.sh g1             # compare against g1.json
tests/robocasa/surface/run.sh g1 --record    # rewrite g1.json, only for an intended change
```

## The G1

The G1 sim's unit tests run in a disposable robocasa container with the G1
overlay's mounts, no network and DDS domain 97 (`tests/robocasa/g1/run_in_container.sh`
lists the options):

```bash
tests/robocasa/g1/run_in_container.sh                         # every test_g1_*.py
tests/robocasa/g1/run_in_container.sh test_g1_sensors.py      # one file
```

The acceptance tests start the G1 sim headless with the whole G1 bringup and
check the sim's own recording of the run. They need the `golem_sim_robocasa`
and `golem_ros` images and Docker's NVIDIA runtime, run with no network of their
own, and use domains 195 (arm), 196 (walk), 197 (nav) and 198 (explore) unless
`ROS_DOMAIN_ID` is set:

```bash
tests/g1/test_sim_bringup.sh                 # arms
tests/g1/test_sim_bringup.sh --walk          # walking
tests/g1/test_sim_nav.sh                     # Nav2 to one goal, FAST_LIO odometry
tests/g1/test_sim_explore.sh                 # frontier exploration
```

A passing test ends with `PASSED: <n>/<n> checks` and exits 0. The G1
packages' own tests run in the G1 `ros` container; `docs/g1/packages.md` says
how.

The G1 nav/explore tests use `golem_slam` and require FAST-LIO; `GOLEM_SIM_ODOM=1`
is refused.
