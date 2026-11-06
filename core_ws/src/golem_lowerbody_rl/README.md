# golem_lowerbody_rl

The leg node, `lowerbody_controller_node`: one policy drives a robot's 12 leg
joints through `golem_safety`'s lower-body input, `/safety/lowcmd_lower_in`.
The code names no robot and no policy; config says everything:

- `config/<robot>/robot.yaml`: the leg joints, the IMU frame, the policies the
  robot may run and its default one (keys: `config/example.robot.yaml`). The
  joint table, `mode_machine` and `mode_pr` are `golem_safety`'s
  `robots/<robot>.yaml`.
- `config/<robot>/{sim,real}.yaml`: the run config, what differs between a
  simulator and the real robot (keys: `config/example.run.yaml`).
- `config/<robot>/policies/<name>.yaml`: a policy's spec, how its observation
  is built, which network runs it and how its action becomes leg targets
  (keys: `config/example.policy.yaml`). The weights are in `policies/<name>/`
  (ALMI's interface: `policies/almi/README.md`; HOMIE's licence:
  `policies/homie/`).

## Launch

```bash
ros2 launch golem_lowerbody_rl lowerbody.launch.py robot:=h12 config:=sim sim:=true \
    use_sim_time:=true [policy:=almi] [start_delay:=0.0]
```

`launch/lowerbody.launch.py` refuses before any node starts, with
`lowerbody: not starting: <reason>`: an unknown robot, a run config whose
`sim` differs from `sim:=`, a real-robot run config without
`startup.wait_for_confirm` or with `band.release`, and a policy the robot's
`robot.yaml` does not list. The node applies the same check when it starts.
The H1-2's bringups include it from `golem_bringup`'s `h12_lowerbody.launch.py`
(`policy:=`, default `almi`; `policy:=mjpc` starts no leg node).

| Node parameter | Meaning |
|---|---|
| `robot` | required: the robot |
| `config` | required: the run config, `sim`, `real` or a path |
| `policy` | the policy by name or path; empty: the robot's `default_policy` |
| `imu_trim_{roll,pitch,yaw}_deg` | live, deg within ±15: what the IMU reads when the body is level; seeded from the run config's `imu_trim_deg`, applied to every `/lowstate` quaternion |

## Start-up and faults

`golem_lowerbody_rl/supervisor.py` (`LegSupervisor`, no ROS) holds the phases
and every check, with the run config's values; its module docstring is the
reference. The node logs `loaded run config <path> and policy <path>`, then:

1. **Waiting.** Silent until `golem_safety`'s first true `/safety/heartbeat`
    and a `/lowstate` sample that is finite, has a unit IMU quaternion (within
    0.05) and is upright (gravity z in the IMU frame below -0.9). The reason
    is logged at most every 5 s.
2. **Ramping.** The legs ramp (cosine blend over `startup.ramp_seconds`, 2 s)
    from their measured pose to the policy's pre-pose, then the node waits
    until they settle (within 0.15 rad for 10 steps) or 400 steps pass
    (`legs pre-posed (NOT SETTLED: ...)`, held all the same).
3. **Holding.** `legs pre-posed (...) — holding`. The policy may start once
    `/lowstate`'s tick has advanced for 1 s at the rate of the node's clock
    (sim time in the sims; a tick that is not in ms of that clock, as Isaac's,
    never qualifies), the arm controller's `/safety/lowcmd_upper_in` is fresh
    within 0.5 s and, with `band.release` (the sims), `frame_task_server`'s
    first `/left_ee_pose` has come or `band.max_wait` has passed. The node logs
    `the policy waits: <reason>` every 5 s. On the real robot it then waits
    for `ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger`,
    accepted only now and refused with the reason otherwise.
4. **Running.** `policy started at zero velocity and the default height ...`;
    the policy resets as its spec's `reset_on` says. `/cmd_vel` goes to the
    policy raw (its spec clips and shapes it); one older than
    `commands.timeout` (0.5 s), or non-finite, is zero velocity.
    `/lowerbody/height` (`std_msgs/Float32`, in the policy's own units) is its
    height command; a non-finite one is ignored. Every 10 s the node logs
    `timing over the last 10 s: ...` (step spacing and tick lag).
5. **Faulted.** Once the legs are commanded, any of these stops the node for
    good: a `/lowstate` gap over `faults.max_state_gap` (0.25 s real, 0.5 s
    sim), or none for that long; a tick that goes back; a non-finite value or
    a non-unit IMU quaternion; a fall (gravity z above -0.5); while running, a
    tick lag over `faults.max_latency` when it is set (null in every shipped
    config); a policy step that fails or gives anything but 12 finite targets.
    In every phase, a false heartbeat (`golem_safety`'s estop has latched) or
    none for `heartbeat.timeout` (0.5 s) after the first does too. The node
    logs `fault: <reason>` once, publishes nothing more and stays up.
    `golem_safety` keeps relaying its last leg command, which does not balance
    the robot. Restart the node (after a heartbeat fault, the bringup).

`/lowerbody/active_policy` (latched) is the policy's name while it runs and
`idle` otherwise.

## Threads

`/lowstate` has a callback group of its own and the node spins a two-thread
executor, so a state sample never queues behind the other callbacks (heartbeat,
commands, the 100 Hz steady-clock watchdog timer, the service, the band). One
lock serialises every call into the supervisor and the policy: the band reply's
policy reset never runs inside a policy step, and a callback holding the lock
delays a state sample by as long as it holds it.

## Elastic band (sim)

With `band.release` (the sims' run configs) the node calls
`/elastic_band/toggle` once: when the policy runs and `/left_ee_pose` has come,
or `band.max_wait` after the policy started, and only when the service is there
(it warns once while it is not). It logs `releasing the elastic band:
<reason>`. The service flips the band, so the reply is final. A success logs
`elastic band released (...)` and resets the policy at once, so its first free
steps are not computed from band-held ones. No response, an exception or
`success: false` is logged as `elastic band release failed: <why>`, and the
band stays on. A success whose message is `enabled` means the band was already
off (SPACE in the viewer) and the call hung the robot from it again: the node
warns, counts no release and sends no further call.

Before relying on the real robot's `max_state_gap`, measure the worst
`/lowstate` gap and tick lag on the onboard PC in a hoisted ALMI run (the
timing log): the gap must stay well under 0.25 s, or the legs latch silent.

## Tests

`colcon test --packages-select golem_lowerbody_rl`: the linters;
`test_profile.py`, `test_run_config.py`, `test_policy_spec.py`,
`test_example_configs.py` (the loaders and examples); `test_imu.py`,
`test_timing.py`, `test_commands.py`, `test_observation.py` and
`test_runner_goldens.py` (the policy runner against the golden traces in
`test/golden/`); `test_supervisor.py` and `test_band.py` (the phases and faults
on a fake clock); `test_launch.py` (the launch file's refusals); and
`test_node.py` (the node in one process against fake publishers, on
`ROS_DOMAIN_ID`, or 97 when it is unset, 0 or 1). The mypy test is skipped where
`ament_mypy` is not installed (the ros image). `policies/homie/test_homie_policy.py`
checks HOMIE's own policy file. `reference/` is not linted: it keeps the
original standalone deploy scripts as they were.
