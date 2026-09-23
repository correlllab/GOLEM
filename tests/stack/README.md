# ROS controller integration tests

Requires the built GOLEM ROS and corresponding simulator images, initialized
submodules and LFS assets, and an Ubuntu/NVIDIA Docker host. Run sequentially
on an isolated simulation domain:

```bash
ROS_DOMAIN_ID=193 tests/stack/run.sh robocasa arms
ROS_DOMAIN_ID=193 tests/stack/run.sh robocasa locomotion
ROS_DOMAIN_ID=193 tests/stack/run.sh isaac arms
ROS_DOMAIN_ID=193 tests/stack/run.sh isaac locomotion
```

Each run builds the minimal actual ROS controller stack, waits for simulator
state before launching controllers, and saves logs, physical telemetry, and
JSON acceptance results under `tests/results/stack-*`. Nonzero exit means
failure, including startup failures and timeouts. Controllers cannot access
the private telemetry mount. The default wall timeout is 900 seconds; override
with `GOLEM_TEST_WALL_TIMEOUT` for slower workers.
Simulator cameras are off by default (`GOLEM_TEST_CAMERAS=0`) because these
controllers use no images and CPU camera rendering slows RoboCasa physics well
below the real-time pace the wall-clock controllers assume. `run.json` records
the camera setting, the git state and the exact image IDs used.

Arms move both wrist frames forward 2 cm and back through `/frame_task`.
Acceptance requires successful actions, measured end-effector progress and
joint motion, and advancing LowState ticks. Isaac uses a fixed base for this
isolated arm test; RoboCasa retains its normal support tether.

Locomotion requires 5 simulation seconds of unsupported ALMI balance,
3 seconds of forward `/cmd_vel`, and 3 seconds stopping. The normal stack
switches ALMI → walking policy → ALMI; this does not establish that ALMI itself
walks. Tests check physical height, tilt, displacement, both knees, support
release, telemetry continuity, and active policies. No fixed base is allowed.
Isaac currently lacks RoboCasa's startup support tether, which may cause a
real startup/balance failure. Thresholds are in `check_lowerbody.py`.

The scoring unit tests run in the GitHub CPU merge workflow:

```bash
python3 -m unittest discover -s tests/stack -p 'test_*.py' -v
```

Live acceptance runs on trusted NVIDIA workers through the manual GitHub
workflow. `tests/stack/run_all.sh` runs all four cases and returns nonzero if
any fails. Physical failures remain release blockers; CPU scoring tests alone
do not establish simulator/controller stability.
