# g1_locomotion

ROS 2 node that runs HOMIE's walking policy on the Unitree G1's legs. The legs'
commands go to `golem_safety`'s split mode, which merges them with the arm
controller's waist and arms and alone writes `rt/lowcmd`. It needs
`unitree_sdk2py` and its cyclonedds, onnxruntime, numpy and PyYAML, which the
ros image has.

## License

This package's code (`g1_locomotion/`, `config/`, `test/`) is MIT. The policy
it runs is not: `deploy.onnx`, `g1.yaml` and `homie_policy.py` in `policy/`
are HOMIE's, for non-commercial use only (treat them as CC BY-NC-SA 4.0; see
`policy/README.md` and `policy/LICENSE` for the license and the credits).
The node copies nothing from `policy/`: it loads it in place from the
`policy_dir` parameter, by default `/home/code/g1_locomotion`, where the
containers mount `policy/`.

## ROS interface

- `/cmd_vel` (`geometry_msgs/Twist`): `linear.x`, `linear.y` (m/s) and
    `angular.z` (rad/s) in the base frame. A command older than
    `commands.timeout` (0.5 s) means zero velocity.
- `/locomotion/height` (`std_msgs/Float32`): the pelvis height target, m,
    from 0.24 to 0.74. It starts at 0.74, standing.
- `/safety/heartbeat` (`std_msgs/Bool`): `golem_safety`'s. The node starts
    only after the first true beat, and a false beat, or none for
    `heartbeat.timeout` after the first, stops it for good (Behavior).
- `/lowerbody/confirm_engage` (`std_srvs/Trigger`): the operator's go-ahead
    on the real robot, the H1-2's leg node's service. Accepted only while the
    legs hold the default pose; refused at any other time, naming the phase.
- With `band.release` (`sim.yaml`): `/left_ee_pose`
    (`geometry_msgs/PoseStamped`), which `frame_task_server` publishes once its
    start-up moves are done, and a client of the sim's `/elastic_band/toggle`
    (`std_srvs/Trigger`). `g1.yaml` has neither.
- Parameter `policy_dir`: the policy's directory (this package's `policy/`).
- `--config`: `g1.yaml` (real robot), `sim.yaml` (sim), or a path.
- `use_sim_time` is always false, whatever the launch passes. The node times
    everything by `rt/lowstate`'s tick and the monotonic clock; sim time would
    only add a `/clock` subscription (500 Hz in the sim) to its Python.

```bash
ros2 run g1_locomotion locomotion_node --config sim.yaml
ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger  # g1.yaml, while holding
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"
ros2 topic pub --once /locomotion/height std_msgs/msg/Float32 "{data: 0.7}"
```

The G1 bringups (`golem_bringup`'s `launch/g1/`) start it after the arm controller:
`g1_sim.launch.py` always, with `sim.yaml`, and `g1_real_robot.launch.py` only with
`start_position_verified:=true`, with `g1.yaml`; see their launch files.

## DDS interface

- Reads `rt/lowstate`, only its newest sample. The reader is a plain DDS
    reader that keeps one sample (keep-last 1, best effort) and has no
    listener: a backlog is dropped inside DDS, and only the samples the node
    uses are decoded. The tick paces the policy; the DDS source timestamp
    starts the latency. The IMU (`imu_state.quaternion`, w first, and
    `imu_state.gyroscope`) is taken as the pelvis IMU, as in HOMIE's own
    deployment. The positions and velocities of the policy's 27 joints (legs,
    waist yaw, arms) are looked up by name in `g1_description`: slots 0-12 and
    15-28.
- Reads `rt/safety/lowcmd_upper_in` the same way, and only while waiting to
    start, to know the arm controller is running. It never writes there.
- Writes `rt/safety/lowcmd_lower_in`: slots 0-11 with mode 1, the policy's
    target q, dq = tau = 0 and the training gains (hips kp 100 kd 2, knees
    150/4, ankles 40/2). Every other slot is zero; split mode ignores them.
- Like the other G1 nodes, `ROS_DOMAIN_ID` overrides `network.domain_id` and
    `G1_DDS_INTERFACE` (e.g. `enp130s0`) overrides `network.interface`.

## Behavior

The node follows the H1-2's leg node (`h12_lowerbody_rl`'s
`lowerbody_controller_node`): idle until `golem_safety` beats, a pre-pose,
on the real robot the operator's confirm, the policy, and in the sim the
band's release.

1. **Waiting.** Silent until `golem_safety` has sent a first true
    `/safety/heartbeat` (it beats once it can relay: `/lowstate` reports
    `mode_machine` 5) and, for 1 s of `rt/lowstate`'s tick, every sample
    is steady (no gap over `faults.max_state_gap`), on time (taken within
    half of `faults.max_latency` of its write, and no two that far apart, so
    that a start does not run straight into the latency fault), finite,
    with a unit IMU quaternion, and upright (gravity z in the pelvis frame
    below -0.9, a tilt under about 25 deg); the tick kept to the wall clock
    over that second within `startup.tick_rate`; and the arm controller has
    commanded in the last 0.5 s. The beat and the arm controller do not
    restart that second: once the state has held, the first sample with both
    starts the pre-pose. Until the node's first command `golem_safety`
    streams zero gains to the legs (they hang limp, in the sim from the
    band). The reason for waiting is logged when it changes, each kind at
    most every 5 s. The 1 s on time covers ten of the kernel's 100 ms
    CPU-quota periods, so a laptop that cannot keep up usually keeps the
    node waiting instead of faulting on its first late command. The tick
    check catches a `tick_seconds` that does not match the robot's tick,
    which would run the policy at the wrong rate.
2. **Heartbeat watchdog.** From the first beat on, in every phase, a false
    beat (`golem_safety`'s estop has latched; it beats false from then on)
    or no beat for `heartbeat.timeout` (0.5 s; it beats at 10 Hz) stops all
    output for good and logs why. The worker checks the beat's age on every
    wake, before it takes a sample.
3. **Pre-pose.** The legs move in a straight line from their measured
    positions to the policy's default pose over `startup.ramp_seconds`
    (2.0 s in both configs, the H1-2's `prepose_ramp_s`), one target per
    policy step, at the training gains.
4. **Confirm (real robot).** With `startup.wait_for_confirm` (`g1.yaml`)
    the node then holds the default pose, one target per policy step with
    every fault active, and logs the command that ends the hold:
    `ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger`. The
    policy starts on the next policy step. A call at any other time is
    refused, naming the phase. `sim.yaml` does not wait: the policy follows
    the pre-pose.
5. **Walking.** The policy starts with an empty history, at zero velocity
    and 0.74 m: a `/cmd_vel` or `/locomotion/height` that came before it
    started is dropped. In the sim the band still holds the robot. The
    policy steps on a grid of `rt/lowstate`'s tick 20 ms apart, from the
    pre-pose's first step. A step is due at the first sample at or past its
    grid tick; the next one is due at the first grid tick at least half a
    period after that sample. So the policy follows the robot's clock (the
    sim's, in the sim) at 50 Hz on average, however many samples the node
    sees, and never steps twice within 10 ms. It always works from the
    newest sample: when the node falls behind, the samples it missed are
    dropped, not worked through. Each target is checked for latency and
    published at once, and the safety layer holds it at 500 Hz. Commands are
    clipped to the trained ranges, then to the `commands` limits, and move
    toward the request by at most `commands.accel` (velocities) and
    `commands.height_rate` (height).
6. **Band release (sim).** With `band.release` (`sim.yaml`), a 10 Hz timer
    calls the sim's `/elastic_band/toggle` once the policy runs and a
    `/left_ee_pose` has come (`frame_task_server` publishes it only after
    its start-up moves), or `band.max_wait` (30 s) after the policy started
    without one. The service flips the band, so `band.py`'s `BandRelease`
    keeps at most one call in flight, with no timeout (a late response can
    never meet a second call), sends a failed call again, and sends nothing
    after one has succeeded. A call that never answers leaves the band on.
    Nothing is called once the node has faulted. The real robot has no band
    (`g1.yaml`: `band.release: false`), and the node has no client then.
7. **Faults.** Once the legs are commanded (pre-pose, hold or walking), any
    of these stops all output for good and logs why:
    - a gap over `faults.max_state_gap` between the `rt/lowstate` samples
      the node takes, or none for that long (a sample whose tick did not
      advance does not count);
    - a tick that goes back;
    - a value that is not finite, or an IMU quaternion whose norm is not 1;
    - a fall: gravity z above -0.5 (tilted over 60 deg);
    - a policy step that fails or gives targets that are not finite;
    - a command later than `faults.max_latency`, measured just before it is
      published, from the DDS write of the `rt/lowstate` sample it answers.
      The robot's clock is not the laptop's, so the node takes the fastest
      delivery of the last ~2 s as the offset between them. The latency then
      counts any slower delivery, the time the sample waited for the node,
      and the policy step.

    And in every phase, the heartbeat watchdog (2).

    The node never sends a zero-gain or damping command itself. On every
    fault above but a non-finite joint value, `golem_safety` holds a quiet
    input: it keeps relaying the node's last leg command, which does not
    balance the robot, and after `stale.lower_seconds` (0.2 s in
    `split.yaml`) reports the quiet input (`input_quiet`). It does not estop
    on it: the command stands until `golem_safety` estops for another reason
    (a fall that drives a joint of `rt/lowstate` past `split.yaml`'s
    `limits.estop`, say) or the bringup stops. A non-finite joint value is
    the exception: `golem_safety` checks q, dq and tau_est of every body
    joint in each `rt/lowstate` it takes, latches its estop on a non-finite
    one, and `split.yaml`'s `estop_response` (`mode: zero`, `after:
    keep_writing`) turns every motor off and keeps writing that command. It
    does not check the IMU, so a non-finite gyro or quaternion is held like
    the other faults. A heartbeat fault means `golem_safety` has latched its
    estop (and writes its estop command) or stopped; restart the bringup.
    Only a restart clears a fault. Shutdown (Ctrl-C) also just stops
    publishing.
8. **Timing.** The pre-pose's log line gives the tick's rate against the
    wall clock while waiting (1.00 on the robot). Every 10 s, after a
    publish, the node logs the latency of its commands (median and max) and
    how far apart its policy steps were in ticks.

## Real time

HOMIE stood in the sim with a constant 30 ms of extra delay and fell at
60 ms (the sim's stage-3a probes). The node used to fall in the sim
walk test when the ros container was capped at 1.5 CPUs. An instrumented
copy of that node showed where its delay came from:

- `rt/lowstate` reached its handler up to 558 ms late (90th percentile
    244 ms, against at most 2.4 ms at full CPU). Once a container has used
    its CPU quota for a 100 ms period, every thread in it waits for the next
    one, the DDS receive threads too, and onnxruntime's default thread pool
    spent much of that quota spinning.
- It then worked through the queued samples in order (up to 81 at once),
    stepping on every 10th, so its targets went out in bursts (median 9 ms
    apart, up to 184 ms). The safety layer's 500 Hz relay forwards only the
    newest of a burst.
- Its own work was small: 0.55 ms of CPU per policy step and write.

So the node keeps its own work small, acts only on the newest sample, and
measures what it cannot control:

- onnxruntime runs with one intra-op and one inter-op thread, no spinning and
    nothing pinned: HomiePolicy's own session (`homie_policy.policy_session`).
    onnxruntime's default session has a thread pool as wide as the machine,
    pinned one thread per CPU, and its threads
    can spin between runs: 1.3-2.2 cores at 50 Hz in four benchmarks in the
    ros image on the 24-core laptop, and two threads at 100% in the node
    above. One thread takes a median 0.27-0.29 ms a step, 1.4-1.6% of a core
    (0.89 ms and 4.5% once, on a loaded host), with identical actions
    (`test_locomotion.py` checks it).
- `rt/lowstate` and the arm controller's input (500 Hz too) are plain DDS
    readers that keep the newest sample, decoded only when the node takes
    it. The SDK's subscribers decode every sample in the DDS receive thread,
    under the GIL: about 0.5 ms each, 26% of a core for `rt/lowstate`.
    Taking and decoding the newest sample costs 0.25-0.27 ms, 13% of a core
    while the node keeps up with every sample. The arm controller's input is
    read only while waiting.
- No `/clock` subscription (see `use_sim_time` above).
- The publish path is the policy step, the CRC and one write; the only
    logging on it is the timing line every 10 s.

In the walk test at full CPU (`tests/g1/test_sim_bringup.sh --walk`,
five runs at a host load of 5-10) the node used 0.20-0.24 cores, and its
commands left a median 0.7-0.9 ms (at most 1.9-7.0 ms) after their
`rt/lowstate` sample. The old node used 2.39 cores in a run the same hour
(two onnxruntime threads at 100%), 0.56 in another, and 2.43 in the probe
above. A traced copy of the node, timed against the sim's own record, put
its latency estimate within 0.6 ms of the true delay.

What the node cannot fix is the CPU the rest of its container needs, 2.2-2.6
cores in those runs (the safety layer 0.8-1.0, the arm controller 0.7-0.9,
`joint_state_publisher` 0.4-0.5, the hands 0.2-0.3). Capped below that with
`docker run --cpus`, the container is throttled in 98-100% of its 100 ms CFS
periods, and every thread in it, the safety layer's and the DDS receive
threads too, then waits for the next period:

- `rt/lowstate` reached the node 41-54 ms late at 1.5 CPUs and 64-73 ms
    late at 1.0 CPUs, so the node did not start. In four runs (two at each
    cap) it was still waiting, logging why, when the test gave up 209-227 s
    of sim time in. In earlier runs of this code it never started at 1.0
    CPUs (6 of 6), and at 1.5 CPUs it started in 2 of 5 (not counting one
    with the sim at a real-time factor of 0.06), then faulted within 2 s on a
    command 27.1 or 59.1 ms late.
- Capped at 1.5 CPUs while walking (three runs), it faulted on a command
    25.5-35.9 ms late, 0.57-0.87 s after the cap, with the robot still up.
    The safety layer of those runs then held its last target until its 1 s
    leg watchdog (`sim_split.yaml`) estopped, and the robot dropped when the
    estop turned every motor off. `golem_safety` holds the last target and
    reports the quiet input instead (Behavior, Faults).
- The old node fell in every capped run (five here, three in stage 3a).

How the container gets its CPU matters as much as how much (this is the
docker setup's choice, not the node's). The same 1.5 CPUs as a 10 ms CFS
period (`--cpu-period 10000 --cpu-quota 15000`) walked in 3 of 6 runs; two
more stood but tilted to 27 deg or sank to 0.699 m, and one fell. Two pinned
cores with no quota (`--cpuset-cpus`) walked in 2 of 2. In the traced 10 ms
runs the node's side stayed fast (a median 1.3-1.4 ms from the state to its
write, at most 18.5-22.0 ms), but the safety layer's relay, starved in the
same container, took a median 25-32 ms (at most 137-171 ms) to get a command
into the sim and overwrote 17-18% of them. Of the sim's 157 holds over 40 ms,
155 came after the node had already written a newer command. The node's
latency fault cannot see that part.

## Configs

| Key | `g1.yaml` (real) | `sim.yaml` (sim) |
|---|---|---|
| `network.domain_id` | 0 | 1 |
| `imu_frame` | `pelvis`, the only one accepted | `pelvis` |
| `tick_seconds` | 0.001, assumed (see below) | 0.001: the sim counts ms of sim time |
| `startup.ramp_seconds` | 2.0, the H1-2's pre-pose | 2.0 |
| `startup.wait_for_confirm` | true: hold until `/lowerbody/confirm_engage` | false |
| `startup.tick_rate` | [0.9, 1.1]: the robot keeps real time | [0.25, 1.25]: the sim's real-time factor |
| `heartbeat.timeout` | 0.5 s, above the 0.1 s beat period | same |
| `band.release` | false: no band | true: call `/elastic_band/toggle` |
| `band.max_wait` | 30.0 s (unused) | 30.0 s from the policy's start |
| `commands.vx`, `vy`, `yaw` | [-0.3, 0.5], [-0.3, 0.3], [-0.6, 0.6] | the trained ranges: [-0.8, 1.2], [-0.5, 0.5], [-0.8, 0.8] |
| `commands.accel` | 0.5 m/s², 0.5 m/s², 1.0 rad/s² | same |
| `commands.timeout` | 0.5 s | same |
| `commands.height_rate` | 0.1 m/s | same |
| `faults.max_state_gap` | 0.05 s | 0.5 s, the sim can run slower than real time |
| `faults.max_latency` | 0.025 s | same |

- Every key is required and an unknown key is an error, so a typo cannot fall
    back to a default.
- `topics.low_cmd` must be a `golem_safety` input (`rt/safety/...`), and
    differ from `topics.upper_cmd`.
- The command limits must include 0 and lie inside the trained ranges
    (`cmd_range` in `policy/g1.yaml`).
- `tick_seconds` must divide the 20 ms policy period, and
    `startup.tick_rate` must include 1.
- `heartbeat.timeout` must be longer than `golem_safety`'s 0.1 s beat period.
- `faults.max_latency` is well under the constant delay HOMIE stood with in
    the sim (30 ms; it fell at 60 ms, the sim's stage-3a probes).
- The limits and rates are starting values, not yet tuned on our G1.

## Before using it on the robot

- Only one program may command `rt/lowcmd`. The robot runs in Unitree's
    debug mode, with Unitree's controller released, from before the bringup
    starts: `golem_safety` writes every motor from its first moment. Entering
    debug mode is a manual operator step that nothing in this stack does
    (`docs/g1/first-session.md`).
- Support the robot (a gantry): `split.yaml`'s estop zeroes every motor.
- After the pre-pose the node holds the legs at the default pose until you
    call `ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger`
    (Behavior, Confirm); check the stance before you do.
- A fault silences the node (Behavior, Faults), and `golem_safety` then
    holds the legs on the node's last command instead of turning every motor
    off: the robot stands on a frozen target that does not balance it. A
    non-finite joint value in `rt/lowstate`, a fall that drives a joint past
    `split.yaml`'s `limits.estop`, or a `rt/lowstate` from another robot
    (`mode_machine` not 5) makes `golem_safety` estop (every motor off).
- The frame of `rt/lowstate`'s IMU on our unit is not verified yet; the node
    takes it as the pelvis IMU, and accepts only `imu_frame: pelvis`.
- The unit of `rt/lowstate`'s tick on our unit is not verified yet; the node
    assumes ms (`tick_seconds: 0.001`), so the tick must grow by 2 per 500 Hz
    sample. A wrong unit would change the policy's rate, so the node waits
    (logging the rate) unless the tick runs at 0.9-1.1 s per s of wall clock.

## Files

- `config/`: `g1.yaml` and `sim.yaml`.
- `policy/`: HOMIE's pretrained policy and its MuJoCo script, with their own
    README and non-commercial LICENSE.
- `g1_locomotion/`
    - `config.py` loads and checks a config, and starts Unitree DDS from it.
    - `locomotion.py` holds the logic with no ROS or DDS: loading the policy,
        the joint slots, the observed state, the command filter, and the
        startup and fault handling (`LocomotionController`).
    - `band.py` decides when the sim's band is released (`BandRelease`), with
        no ROS.
    - `driver.py` runs the controller on Unitree DDS (`LocomotionDriver`).
    - `locomotion_node.py` is the ROS 2 node.
- `test/`: unit tests of the logic, DDS tests, and tests of the node.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:../g1_description:$PYTHONPATH python3 -m pytest test
```

- They need the workspace built and sourced (`golem_safety`). They load
    the policy from this package's `policy/`.
- The DDS tests run on domain 97: a fake G1, a fake arm controller and the
    node in the test's process (its controller and driver, and its
    `/safety/heartbeat` subscription), and `golem_safety`'s installed
    `safety_node` (robot `g1`, a copy of its `split.yaml`, its heartbeat
    remapped to a topic of the test's own) as a subprocess. Its heartbeat
    gates the start; the heartbeat tests stop it or make it latch. They never
    touch the robot. The test's process is busy enough that `rt/lowstate` is
    sometimes tens of ms late, so they relax `faults.max_state_gap` and
    `faults.max_latency` to 0.25 s and `startup.tick_rate` to [0.5, 2.0],
    and `golem_safety`'s quiet-`rt/lowstate` report (`stale.state_seconds`)
    to 0.25 s; the unit tests check the real values. Their 50 Hz checks still
    need the host to keep up: at a load average of 11-19 on the 24-core laptop
    the fake G1 itself fell behind once (565 samples in 1 s) and one failed;
    the repeats passed.
- The fault tests (a stale or fallen state, a non-finite gyro, a late command) check
    that the node faults and sends nothing more, and that `golem_safety` keeps
    relaying its last leg command, reports the quiet leg input once and does
    not estop. The stale-`rt/lowstate` test keeps `split.yaml`'s own
    `stale.state_seconds`, and `golem_safety` reports the quiet `rt/lowstate`
    and its return too.
