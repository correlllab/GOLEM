# Simulator regression tests

All simulator regression tests and repeatable validation commands live here,
outside the simulator submodule. CPU tests run on every relevant pull request
through `.github/workflows/simulator-tests.yml`; select its `cpu` job as a
required branch-protection check to gate merges.

## CPU tests

The Isaac suite needs Python 3.11, CPU PyTorch, NumPy, Pillow and CycloneDDS 0.10.2
plus the checked-out Unitree SDK. It does not start Kit or DDS participants.
See the workflow for the exact dependency setup.

```bash
PYTHONPATH=CL_isaaclab_sim:unitree_sdk2_python python -m unittest discover -s tests/isaaclab -p 'test_*.py' -v
PYTHONPATH=h1_robocasa python -m unittest discover -s tests/robocasa -p test_loop_timing.py -v
python -m unittest discover -s tests/docker -v
```

Using existing Docker images:

```bash
ROS_DOMAIN_ID=187 tests/isaaclab/run_in_container.sh
ROS_DOMAIN_ID=187 tests/robocasa/run_in_container.sh
```

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

`run_robot_interop.sh` publishes native ROS LowCmd and Inspire commands, checks
wrist and right-index finger motion in both directions, receives LowState,
Inspire state and all 19 sensor topics, and releases body motors on exit.
The workflow has an explicit manual GPU job for a prepared self-hosted
runner labelled `nvidia`; it is not run automatically on untrusted pull requests.

The CPU suite cannot establish physical asset fidelity, real-time performance,
or controller stability. Those need the GPU/ROS integration environment and
robot-specific acceptance tests when the CL_Assets pin changes.
