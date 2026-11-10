# ROS controller integration tests

Requires the built GOLEM ROS and corresponding simulator images, initialized
submodules and LFS assets, and an Ubuntu/NVIDIA Docker host. Run sequentially
on an isolated simulation domain:

```bash
ROS_DOMAIN_ID=193 tests/stack/run.sh robocasa arms
ROS_DOMAIN_ID=193 tests/stack/run.sh robocasa locomotion
ROS_DOMAIN_ID=193 tests/stack/run.sh isaac arms
```

Each run builds the minimal actual ROS controller stack, waits for simulator
state before launching controllers, and saves logs, physical telemetry, and
JSON acceptance results under `tests/results/stack-*`. Nonzero exit means
failure, including startup failures and timeouts. Controllers cannot access
the private telemetry mount. The default wall timeout is 900 seconds; override
with `GOLEM_TEST_WALL_TIMEOUT` for slower workers.
Simulator cameras are off by default (`GOLEM_TEST_CAMERAS=0`) because these
controllers read no images, and RoboCasa's camera renders cost the sim physics
time (the physics waits for a frame still rendering). `run.json` records
the camera setting, the git state and the exact image IDs used.

Arms move both wrist frames forward 2 cm and back through `/frame_task`.
Acceptance requires successful actions, measured end-effector progress and
joint motion, and advancing LowState ticks. Isaac uses a fixed base for this
isolated arm test; RoboCasa retains its normal support tether.

Locomotion requires 5 simulation seconds of unsupported ALMI balance,
3 seconds of forward `/cmd_vel`, and 3 seconds stopping, with ALMI the active
policy throughout, as in bringup. The leg node runs on the H1-2's sim run config
(`golem_lowerbody_rl`'s `config/h12/sim.yaml`): its policy starts only once the
arm controller's commands are fresh and `frame_task_server` has published its
first `/left_ee_pose`, so this case runs `frame_task_server` too, holding the
torso and arms as the sim bringup does. Tests check physical height, tilt, displacement, both knees, support
release, telemetry continuity, and active policies. No fixed base is allowed.
`isaac locomotion` cannot pass: `run.sh` refuses it and `run_all.sh` skips it. Isaac's
`rt/lowstate` tick counts publishes, not milliseconds of the node's clock, so
the leg node's readiness check never passes and the policy never starts (the
node logs that the tick is off the node's clock), and Isaac serves no
`/elastic_band/toggle`. Thresholds are in `check_lowerbody.py`.

The scoring unit tests run in the GitHub CPU merge workflow:

```bash
python3 -m unittest discover -s tests/stack -p 'test_*.py' -v
```

Live acceptance runs on trusted NVIDIA workers through the manual GitHub
workflow. `tests/stack/run_all.sh` runs the three cases that can pass and returns
nonzero if any fails. Physical failures remain release blockers; CPU scoring tests alone
do not establish simulator/controller stability.
