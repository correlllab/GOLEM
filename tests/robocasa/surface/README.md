# RoboCasa topic-surface probe

The RoboCasa sim must publish what the real robot publishes. This probe starts the sim
headless, measures everything that other processes can see of it over ROS 2 and DDS, and
compares that with a checked-in golden, one per robot (`h12.json`, `g1.json`). A change to the
simulator that alters a topic, type, frame, encoding, point layout, rate or node fails the
comparison and names what differs.

```bash
tests/robocasa/surface/run.sh h12            # compare against h12.json
tests/robocasa/surface/run.sh g1             # compare against g1.json
tests/robocasa/surface/run.sh h12 --record   # rewrite h12.json
GOLDEN=/path/to/other.json tests/robocasa/surface/run.sh h12
```

It needs the `golem_sim_robocasa` image (it never rebuilds) and runs in a disposable container
that leaves a running simulator alone. Each run takes about 40 s with a warm image and build cache.
The `g1` run selects the G1 overlay the way `docker_run.sh --g1` does.

It always uses DDS domain 194, one probe at a time: `run.sh` holds a lock on
`/tmp/golem-dds-domain-194.lock` and a second run fails at once. The probe also refuses a domain
with anything on it before the sim starts, and fails if it then finds a node name or `/clock`
publisher twice, which means a second sim is on the domain.

## What is measured

The launcher (`launch_robocasa.sh`) and `probe.py` run in the same container. The probe starts
the launcher with `--headless` and a fixed task or layout and seed (see `run.sh`), waits for its
"ROS bridges up" line, measures over 10 s of simulated time, then stops the sim with SIGINT, the
only way out of the sim loop, and checks that it exits.

`probe.py` writes:

- `nodes`: per sim node, its publishers, subscriptions, services and actions with types. The
  parameter services, `/rosout`, `/parameter_events` and an action's internal topics and services
  are left out, since rclpy adds them to every node.
- `bare_dds`: what the Unitree SDK publishes and consumes on `rt/lowstate` and `rt/lowcmd`; it
  has no ROS node.
- `topics`: per published topic, its type, QoS, `frame_id`, `encoding` (for a compressed image,
  its `format`), `width`/`height`, `point_fields`, `rate_hz` and whether any message arrived.
  A point cloud's `width` is kept only when the cloud is organized (height above 1).
- `tf_static`, `tf`: the `parent -> child` edges.
- `livox_imu_mean_acc`: the mean of |a| on `/livox/imu` over the window.
- `lowstate_ticks_advance`: whether the tick in `rt/lowstate`, read through `unitree_sdk2py`, rose
  during the window.

`rate_hz` is per second of simulated time, so it does not depend on how fast the machine runs the
sim. The exception is a topic that the sim publishes from a wall-clock timer (the gripper state,
10 Hz like the real driver); its `rate_clock` is `wall` and `WALL_TIMER_TOPICS` in `probe.py` lists
it. The window length and the other timings are `probe.py` arguments.

## Comparison

Names, types, QoS, frames, encodings, sizes, point fields, TF edges and node surfaces must match
exactly; list order does not matter. `rate_hz` and `livox_imu_mean_acc` must be within 5 %. A
mismatch prints the path of each differing value with the golden and observed values, and exits 1.
Exit 2 means the probe could not measure (the sim did not start, the domain was shared, a required
measurement was missing).

The sim is stopped with SIGINT once the surface is measured. If it then crashes or hangs, the run
prints a warning with the exit status and the kept `sim.log`, and `run.json` records both
(`sim_exit`, `sim_log`); the comparison still decides the exit code.

## Recording

Re-record only for an intended change to what the sim publishes, in the same commit as that
change, and read the diff of the golden. The golden describes the default sim: `run.sh` clears
the sim knobs it lists as `pinned` (cameras, lidar, odometry, spawn, `G1_DDS_INTERFACE` and so on)
that the shell or `docker/.env` sets, so a stray setting does not end up in it. It also fails,
naming the variable, if the robocasa service in the compose files of the run interpolates a
`GOLEM_*` or `G1_*` variable that is neither pinned nor `allowed`; a new sim knob goes into one of
the two lists. `--record` refuses a run that is not fit to be a golden (no `/clock`, no IMU samples,
`rt/lowstate` not ticking); a sim that crashed on SIGINT afterwards only warns.

Each run leaves `observed.json`, `sim.log`, `sim_exit`, `probe.log` and `run.json` in
`tests/results/surface-<robot>-<time>-<pid>/`.
