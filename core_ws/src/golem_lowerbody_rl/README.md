# golem_lowerbody_rl

The H1-2's leg node, `lowerbody_controller_node`: one policy (`active_policy`:
`fame`, `walk` or `almi`; every bringup runs `almi`) drives the 12 leg joints
through `golem_safety`'s lower-body input, `/safety/lowcmd_lower_in`. The
policies are in `golem_lowerbody_rl/policy.py` and `policies/` (ALMI's
interface: `policies/almi/README.md`); the bringups set the parameters from
`golem_bringup/config/h12/lowerbody_{sim,real}.yaml`.

## Start-up and faults

`golem_lowerbody_rl/supervisor.py` (`LegSupervisor`, no ROS) holds the phases
and every check, with `g1_homie`'s thresholds; the node's module docstring has
the rest.

1. **Waiting.** Silent until `golem_safety`'s first true `/safety/heartbeat`
    and a `/lowstate` sample that is finite, has a unit IMU quaternion (within
    0.05) and is upright (gravity z in the base frame below -0.9, a tilt under
    about 25 deg). The reason is logged when it changes, each kind at most
    every 5 s.
2. **Ramping.** The legs ramp (cosine blend over `prepose_ramp_s`, 2 s) from
    their measured pose to the policy's nominal crouch, and the node waits
    until they settle there (within 0.15 rad for 0.2 s) or 8 s after the ramp.
3. **Holding** (`engage_wait_for_confirm`, the real robot). The legs hold the
    crouch until `ros2 service call /lowerbody/confirm_engage
    std_srvs/srv/Trigger`, which is refused, naming the phase, at any other
    time. The sim engages without it.
4. **Running.** The policy engages from zero velocity and the default squat
    height (commands sent before are dropped). `/cmd_vel` is clipped to ±1; a
    command older than `cmd_vel_timeout` (0.5 s) is zero velocity, and a
    non-finite one is zero velocity with a warning. A non-finite squat command
    is ignored.
5. **Faulted.** Once the legs are commanded (ramping, holding, running), any
    of these stops the node for good: a gap over `max_state_gap` between
    `/lowstate` callbacks, or none for that long (0.25 s on the real robot,
    0.5 s in the sim, which can run slower than real time; the callbacks
    share one executor with the policy tick, so the gap includes executor
    stalls); a non-finite
    value or a non-unit IMU quaternion; a fall (gravity z above -0.5, over
    60 deg); a policy step that raises or gives anything but 12 finite
    targets; an unexpected error in a control tick. In every phase, a false
    heartbeat (`golem_safety`'s estop has latched) or none for
    `heartbeat_timeout` (0.5 s) after the first does too. The node logs the
    fault once, publishes nothing more and stays up. `golem_safety` keeps
    relaying its last leg command, which does not balance the robot.
    Restart the node (after a heartbeat fault, the bringup) to clear it.

## Elastic band (sim)

With `disable_elastic_band: true` (the sim's default; `lowerbody_real.yaml`
sets false, and the node then never calls it) the node releases the sim's
band with one call to `/elastic_band/toggle`: once the policy is engaged and
`/left_ee_pose` has come (`frame_task_server` publishes it after its start-up
moves; an earlier release drops the robot), or `band_max_wait` after the
node started, and only when the service is there (it warns once while it is
not). The service flips the band, so the reply is final. A success takes the
band off, and the policy's history is reset on the next tick, so its first
free steps are not computed from the band-held ones. No response, an
exception or `success: false` is logged as `elastic band release failed:
<why>`, and the band stays on. A success whose message
is `enabled` means the band was already off (SPACE in the viewer) and the call
hung the robot from it again: the node warns, counts no release and sends no
further call.

Before relying on the real robot's `max_state_gap`, measure the worst
`/lowstate` callback gap on the onboard PC in a hoisted ALMI run: it must stay
well under 0.25 s, or the legs latch silent.

## Parameters

| Parameter | Default | `lowerbody_sim.yaml` | `lowerbody_real.yaml` |
|---|---|---|---|
| `active_policy` | `fame` | from `lowerbody:=` | from `lowerbody:=` |
| `engage_wait_for_confirm` | true | false | (true) |
| `prepose_ramp_s` | 2.0 s | | |
| `heartbeat_timeout` | 0.5 s | | |
| `max_state_gap` | 0.25 s | 0.5 s | 0.25 s |
| `cmd_vel_timeout` | 0.5 s | 0.5 s | 0.5 s |
| `disable_elastic_band` | true | (true) | false |
| `band_max_wait` | 30.0 s | 34.0 s | |
| `imu_offset_{roll,pitch,yaw}_deg` | 0 | 0 | 0.50, -2.5, 0 (live trims) |

## Tests

`colcon test --packages-select golem_lowerbody_rl`: the linters,
`test_supervisor.py` (the phases and faults on a fake clock) and
`test_band_node.py` (the band release against a fake toggle service, in one
process on domain 97). The mypy test is skipped where `ament_mypy` is not
installed (the ros image). `reference/` is not linted: it keeps the original
standalone deploy scripts as they were.
