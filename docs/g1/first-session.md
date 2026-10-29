# First real G1 session

The first session on the real G1 runs the G1 bringup (`g1_real_robot.launch.py`)
in Unitree's debug mode, with the robot hanging in a gantry. In debug mode
Unitree's own controller is released, and the laptop's stack drives every motor
on `rt/lowcmd` through the [safety layer](safety.md) (`golem_safety`), as on
the H1-2. The session has two launches: the arms alone (the arm controller,
`golem_upper_ik`), then the arms with HOMIE's walking node
(`start_position_verified:=true`), which moves the legs to its default pose and
holds them there until the operator's confirm. No hand driver runs.

> **Do not run this session yet.** The procedure for entering debug mode on our
> G1 has not been written ([Entering debug mode (not yet
> written)](#entering-debug-mode-not-yet-written)).

> **Status.** Nothing has run on the real robot yet. Gains, speed limits and
> estop limits are conservative starting values, not tuned on our G1.

Every command runs on the laptop wired to the robot. Nothing runs on, or
changes, the robot's own computers. Practise the same bringup in the sim first
([Arms by hand](index.md#arms-by-hand), sending only `home` and `arms_down`):
the sim starts the robot as debug mode does, every motor off, with an elastic
band in place of the gantry.

## Entering debug mode (not yet written)

The bringup assumes that the G1 is in Unitree's debug mode before it starts:
from its first moment the safety layer writes every motor on `rt/lowcmd`, and
no Unitree controller may be running beside it. Nothing in this stack enters
debug mode or checks that the robot is in it.

The procedure for our G1 has not been provided:

- how to enter debug mode on our G1;
- how to confirm, before launching, that the robot is in it;
- when the robot comes off the gantry.

**Nobody runs the first session until it is written here.** Steps 4, 9 and 10
below depend on it.

## The steps in short

| Step | What happens |
|---|---|
| 1. Freeze the code | Clean `git status`, note `git rev-parse HEAD`. |
| 2. Network | `docker/.env`: `ROS_DOMAIN_ID=0`, `GOLEM_ROBOT=g1`, `G1_DDS_INTERFACE=` the wired port (ours: `enp130s0`). See [DDS and networking](dds.md). |
| 3. Power | Laptop on AC, lid open, container run under `systemd-inhibit`. |
| 4. Robot and people | Robot in debug mode, hanging in the gantry, which carries its whole weight. Nobody within reach of the arms or the legs. |
| 5. Start the container | `docker/scripts/docker_run.sh ros --g1`; it prints `robot: g1` and asks you to confirm domain 0. |
| 6. Read-only checks | Gate: the container is the G1's and pinned to the robot's port, or exit and fix `docker/.env` and the shell's exports. Then nothing else writes the command topics, `rt/lowstate` arrives, and every joint is inside the safety layer's estop limits. |
| 7. First launch: the arms | `ros2 launch golem_bringup g1_real_robot.launch.py use_hands:=false use_rviz:=false`: the zero-gain stream, the heartbeat, the arms' start-up. The legs stay limp. |
| 8. Named configs | Send only `home` and `arms_down` on `/named_config`, one at a time, `arms_down` last. |
| 9. Stop the bringup | Stop the controller alone (the arms go limp, the waist holds), then Ctrl+C the launch, with the gantry carrying the robot. |
| 10. Second launch: the legs | Step 6's last checks again (no other writer, the joints inside the estop limits), then the same launch with `start_position_verified:=true`: the walking node's pre-pose and hold, then the confirm. Stop with step 9. |
| 11. End the session | Copy the safety layer's logs out, exit the container, set `docker/.env` back to the sim. |

## What you should see

- **The zero-gain stream.** The safety layer prints `relay_started` and from
  then on writes `rt/lowcmd` at 500 Hz with zero gains on every motor (mode 0,
  every value 0): every motor is off.
- **The heartbeat.** Once `rt/lowstate` reports `mode_machine` 5,
  `/safety/heartbeat` is true at 10 Hz.
- **The arms' start-up.** The arm controller starts 2 s after the safety layer.
  Its first command holds the arms and the waist where they are; the safety
  layer relays it (`input_engaged`, `split_mode_upper`), and the controller then
  moves the arms to `home` (forearms forward at waist height) and the waist to
  its hold angles, at up to 0.3 rad/s per joint. The leg motors stay off.
- **The pre-pose** (second launch only). The walking node waits for the
  heartbeat, a steady and upright `rt/lowstate` and the arm controller's
  commands, then moves the legs from where they hang to HOMIE's default pose
  over 2 s (`input_engaged`, `split_mode_lower`) and holds them there.
- **The confirm** (second launch only). `/lowerbody/confirm_engage` starts
  HOMIE's policy at zero velocity. When the robot bears its own weight on its
  legs is part of the procedure that is not written yet ([Entering debug mode
  (not yet written)](#entering-debug-mode-not-yet-written)).

Any `estop_command_written` from the safety layer, a `fault:` line from the
walking node, or a controller that exits with an error: go to step 9. After any
estop, restart the whole bringup.

## Not in this session

- **Hands.** The Dex3 driver bypasses the safety layer, so it stays off.
- **Walking on `/cmd_vel`.** The session ends at the confirm at most. Know this
  before any walking session: a fault of the walking node (a stale
  `rt/lowstate`, a non-finite IMU value, a fall, a late command) silences it,
  and the safety layer then **holds its last leg command instead of turning
  every motor off**. That frozen target does not balance the robot, and it
  stands until the safety layer estops for another reason or the bringup stops
  ([Safety layer](safety.md), Quiet inputs). Keep the robot on a gantry.
- **Navigation.** Sim only. The real bringup refuses `use_nav:=true` until the
  real MID-360 lidar is connected.

## The checklist

A laptop shell is a plain terminal on the laptop, outside any container.
Terminal 1 is the `golem_ros` container started in step 5. Terminal 2 is a
second shell in it, opened from a laptop shell:

```bash
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
```

### 1. Freeze the code (laptop shell)

The workspace is built with `--symlink-install`, so a Python edit under
`core_ws/` takes effect the next time a node starts.

- Stop every editor and agent that works in the repo.
- From the repo root, `git status --short` prints nothing, and
  `git submodule status` shows no line starting with `+` or `-`.
- Write `git rev-parse HEAD` into the session notes. Leave the tree alone until
  the session ends.
- Steps 7 to 10 have not been rehearsed against the real robot. The sim's arm
  test (`tests/g1/test_sim_bringup.sh`) runs the same core
  (`g1_core.launch.py`, with the sim configs: the walking node always runs, and
  without the confirm) and stops the controller the same way (SIGINT). If
  `core_ws/src/golem_safety/README.md`, `core_ws/src/golem_upper_ik/README.md`
  or `core_ws/src/g1/g1_locomotion/README.md` at the frozen commit disagrees
  with them, fix this checklist before the session.

### 2. Network and docker/.env (laptop shell)

- `docker/.env` (copy `docker/.env.example` if it is missing) has:

  ```
  ROS_DOMAIN_ID=0
  GOLEM_ROBOT=g1
  G1_DDS_INTERFACE=enp130s0
  ```

- `ip -br addr show dev enp130s0` shows `UP` and `192.168.123.222/24`.
- With the G1 selected, `docker_run.sh` refuses domain 0 if
  `G1_DDS_INTERFACE` is empty, names no interface on the laptop, is not `UP`,
  or has no IPv4 address. It also refuses a `ROS_DOMAIN_ID` that is not an
  integer from 0 to 232, as in a `docker/.env` with Windows line endings.
  With the H1-2 selected there is no interface check at all, which is why
  step 5 selects the G1 twice (`GOLEM_ROBOT=g1` and `--g1`) and checks the
  `robot:` line.

### 3. Power and sleep (laptop shell)

- The laptop is on AC: `cat /sys/class/power_supply/ADP0/online` prints `1`.
- The lid stays open.
- Sleep is blocked for as long as the container runs, because step 5 starts it
  under `systemd-inhibit`.

### 4. Robot and people

- The robot is in Unitree's debug mode ([Entering debug mode (not yet
  written)](#entering-debug-mode-not-yet-written)).
- It hangs in the gantry, which carries its whole weight. From the moment the
  safety layer starts, every motor it does not command is off: the legs hang
  limp until the walking node commands them, and an estop turns every motor
  off.
- It hangs still: the safety layer checks every joint's position, speed and
  torque from its first `rt/lowstate` (step 6).
- Nobody is within reach of the arms or the legs. The arms start moving to
  `home` a few seconds after the launch in step 7, and the legs a few seconds
  after the controller in step 10.
- A second person watches the laptop.

### 5. Start the container (terminal 1)

From the repo root in a laptop shell:

```bash
systemd-inhibit --what=sleep:idle:handle-lid-switch --who=G1 --why="real-robot session" \
    docker/scripts/docker_run.sh ros --g1
```

The wrapper prints `robot: g1`, then asks:

```
Domain 0 is the REAL ROBOT bus.
  robot: g1
  interface: enp130s0 (IPv4 192.168.123.222/24)
Proceed? [y/N]
```

Answer `y` only if it names `robot: g1` and the robot's wired port with its
address. By then the wrapper has refused a missing, `DOWN` or unaddressed
`G1_DDS_INTERFACE`, but it does not know which port is the robot's: a wrong pin
such as `wlan0` with its WiFi address passes. Step 6 starts with the gate that
catches it.

If the wrapper stops with `golem_ros already exists`, a container from an
earlier run is still there. Find out what it runs (`docker ps -a`,
`docker logs golem_ros`). If it may be running a bringup, end that bringup
with step 9 first: `--restart` removes the old container with `docker rm -f`,
which kills every node at once, without the controller's last command
([Safety layer](safety.md), Stopping). Then start this one with
`docker/scripts/docker_run.sh ros --restart --g1` under the same
`systemd-inhibit`.

The container builds `core_ws` with the `g1_*` packages before it opens the
shell (`launch_ros.sh`, an incremental colcon build that checks every
package's inputs). A failed build stops the container instead of opening the
shell, so the shell means the build matches the frozen tree. To rebuild by
hand anyway, run the container's own build again:

```bash
/home/code/h12_sim_scripts/launch_ros.sh true && source /home/code/core_ws/install/setup.bash
```

### 6. Read-only checks (terminal 1, before launching anything)

**Gate: the container is the G1's, on domain 0, pinned to the robot's port.**

```bash
echo "$GOLEM_ROBOT $ROS_DOMAIN_ID $G1_DDS_INTERFACE"
cat /home/code/cyclonedds_local.xml
```

The two lines must read exactly:

```
g1 0 enp130s0
<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="enp130s0"/></Interfaces></General></Domain></CycloneDDS>
```

If either line differs, `exit` the container and fix `docker/.env` (step 2).
A variable exported in the laptop shell wins over `docker/.env`, so in the
shell you start the container from, also check
`env | grep -E '^(ROS_DOMAIN_ID|GOLEM_ROBOT|G1_DDS_INTERFACE)='` and `unset`
any of them that disagrees with step 2. Then start again at step 5. Do not
continue.

Nothing else may be writing the robot's command topics. An empty listing
means "no other writer" only if the gate above passed and `read_lowstate.py`
below then shows the robot: on the wrong interface, or with no link to the
robot, discovery finds nothing and the listing is just as empty.

```bash
cyclonedds ls -i 0 -r 5s -t 'rt/(lowcmd|dex3/.*/cmd)' --suppress-progress-bar --color none \
    | grep -E '__Hostname|__ProcessName|rt/|Publications|Subscriptions'
```

The listing shows each DDS participant that reads or writes those topics.
Under a topic, `Subscriptions` is a reader and `Publications` is a writer.
Nothing may show `Publications` under `rt/lowcmd` or `rt/dex3/.../cmd`: two
writers fight over the motors. If a writer is a program on this laptop (its
`__Hostname` and `__ProcessName` say which; `docker ps` in a laptop shell
lists leftover containers), stop it and list again. If it is anywhere else, do
not launch: resolving that is not a laptop-side step.

Then the robot's state:

```bash
python3 /home/code/third_party/leprechaun_g1/tools/read_lowstate.py
```

It only subscribes (domain 0 on `enp130s0`, both fixed in the script), prints
once a second, and stops with Ctrl+C. Check that:

- it shows `mode_machine: 5`;
- `tick` grows from one print to the next;
- the joints (slots 0-28) match how the robot hangs.

If it only prints `no data yet`, check the cable, then in a laptop shell
`ip -br addr show dev enp130s0` and `sudo ufw status` (it should be inactive).

Last, every joint must be inside the safety layer's estop limits. It checks
every `rt/lowstate` against them from its first sample, before anything
engages, and `split.yaml`'s position limits are the URDF's ranges exactly
(`limits.estop.position_offset: 0.0`). Print them by slot:

```bash
ros2 run golem_safety golem_safety_dump_config g1 split | python3 -c '
import json, sys
limits = json.load(sys.stdin)["limits"]
for slot, (low, high) in enumerate(zip(limits["q_estop_low"], limits["q_estop_high"])):
    print(f"{slot:2d}  {low:7.3f} .. {high:7.3f} rad")'
```

Each slot's `q` from `read_lowstate.py` (same slot numbers) must lie inside its
range, with room to spare. A joint outside it latches the safety layer's estop
the moment it starts, and nothing engages: do not launch.

### 7. First launch: the arms (terminal 1)

```bash
PYTHONUNBUFFERED=1 ros2 launch golem_bringup g1_real_robot.launch.py use_hands:=false use_rviz:=false
```

`PYTHONUNBUFFERED=1` is harmless: it makes every Python node's prints appear as
they happen. The safety layer logs its events through its ROS logger, as
`[INFO] [...] [safety_node]: {"event": ...}` (or `[WARN]`, `[ERROR]`). With
`use_hands:=false` no hand driver runs, so nothing commands the hands.
`use_rviz:=false` leaves out RViz, which this session does not need. Without
`start_position_verified:=true` the walking node does not start.

What you should see:

1. The safety layer prints `relay_started` (`"robot": "g1"`, its `split.yaml`,
   `"mode": "split_mode"`, its log directory). From then on it writes zero
   gains on every motor. In terminal 2, `ros2 topic hz /lowcmd` shows about
   500 Hz (Ctrl+C it after a few seconds), and
   `ros2 topic echo --once /safety/heartbeat` prints `data: true` once
   `rt/lowstate` reports `mode_machine` 5.
2. Two seconds after the launch the controller starts. On its first command
   the safety layer prints `{"event": "input_engaged", "source":
   "split_mode_upper", ...}`. The controller then moves the arms to `home` and
   the waist to its hold angles, and prints `Frame Task Server initialized
   (robot g1, config g1_split)` when it takes goals.
3. No `split_mode_lower` engages: the leg motors stay off, and the legs hang.

Until the arms engage, the safety layer may print `engage_refused` with a
reason (`no /lowstate received yet`, or another `mode_machine`), at most once
a second. The controller stops with an error if `rt/lowstate` does not arrive
within 5 s, if the safety layer has not relayed its commands within 30 s, or if
the moves to `home` and the waist hold take more than 30 s after that.

If the safety layer prints `estop_command_written` at any point, or the
controller exits with an error, go to step 9.

### 8. Named configs only (terminal 2)

This session uses only `home` and `arms_down`. Send one goal at a time, and
wait for each to finish before sending the next. Send `arms_down` last, so the
arms already hang when step 9 turns them off:

```bash
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'home', plan: false, duration: {sec: 0, nanosec: 0}}"
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'arms_down', plan: false, duration: {sec: 0, nanosec: 0}}"
```

Send no other goals, and start no second controller: a second live writer on
the safety layer's input latches its estop.

### 9. Stop the bringup

1. Wait until no goal is running. (In a later session with the hands, empty
   them first: on exit the hand driver releases them and the fingers go limp.)
2. The gantry carries the robot: from here on the motors may turn off.
3. In terminal 2, stop the controller alone:

   ```bash
   pgrep -af frame_task_server    # one line: the controller node
   pkill -INT -f frame_task_server
   ```

   The controller sends one last command and stops: mode 0 and zero gains on
   the arms, the waist at its hold. The safety layer relays it and keeps
   relaying it, reporting the quiet input (`input_quiet`) 2 s later: the arms
   go limp, the waist holds, and no estop latches. That is the expected stop.
4. Press Ctrl+C on the launch in terminal 1. The safety layer stops writing
   `rt/lowcmd` at once, and the walking node (step 10) stops too. What the real
   G1's motors do when `rt/lowcmd` stops has not been measured (the sim turns
   every motor off 0.5 s later), which is why the gantry carries the robot.

After an estop, stop the same way; the safety layer keeps writing its estop
command (every motor off) until the Ctrl+C in item 4 above.

### 10. Second launch: the legs (terminal 1)

Only after the first launch went as step 7 describes, and step 9 stopped it.
Before launching again, repeat step 6's last two checks in terminal 1, as they
are now: nothing else writes the command topics (the `cyclonedds ls` listing
shows no `Publications` under `rt/lowcmd` or `rt/dex3/.../cmd`; the first
launch's nodes are gone), and every joint's `q` from `read_lowstate.py` lies
inside its estop range with room to spare. The robot now hangs as step 9 left
it, not as it hung before the first launch, and a joint outside its range
latches the safety layer's estop the moment it starts.

Then check the start pose: the robot hangs upright and still in the gantry
(the walking node waits for a tilt under about 25 deg), the legs are free to
move to HOMIE's default pose, and nobody is near them.
`start_position_verified:=true` is your word that you did; nothing measures it.

```bash
PYTHONUNBUFFERED=1 ros2 launch golem_bringup g1_real_robot.launch.py use_hands:=false use_rviz:=false \
    start_position_verified:=true
```

What you should see, beyond step 7:

1. `locomotion_node` prints `Loaded .../g1.yaml and the policy in
   /home/code/g1_locomotion. Silent until golem_safety beats on
   /safety/heartbeat, ...`. It starts 2 s after the controller and sends nothing
   while it waits; it may print `waiting to start: <reason>`. It waits for the
   heartbeat, for 1 s of steady, on-time and upright `rt/lowstate`, for the
   tick to run at 0.9-1.1 s per s of wall clock, and for the arm controller's
   commands (`g1_locomotion`'s README, Behavior).
2. `moving the legs from where they are to the default pose over 2.0 s
   (rt/lowstate's tick ran at 1.00 s per s of wall clock)`, and the safety
   layer's `input_engaged` with `"source": "split_mode_lower"`. The legs move in
   a straight line from where they hang to HOMIE's default pose over 2 s.
3. `legs at the default pose: holding it until the operator confirms (ros2
   service call /lowerbody/confirm_engage std_srvs/srv/Trigger)`. The legs hold
   the pose, with every fault check active.
4. The confirm, in terminal 2:

   ```bash
   ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger
   ```

   It answers `success=True` with `confirmed: the policy starts on the next
   policy step`, and the walking node prints `walking: policy started at zero
   velocity and 0.74 m height`: HOMIE's policy commands the legs from then on,
   at zero velocity. When the robot may bear its own weight, and so when to
   confirm and what the policy does with the gantry still carrying the robot,
   is part of the missing procedure ([Entering debug mode (not yet
   written)](#entering-debug-mode-not-yet-written)). A call at any other time
   is refused, naming the phase.

A `fault:` line from the walking node means it has stopped commanding the legs
for good, and the safety layer keeps relaying their last command, which does
not balance the robot: go to step 9. Stop this launch with step 9 too; the
walking node keeps commanding the legs until the Ctrl+C.

### 11. End the session

1. The container is deleted when its shell exits, so copy the safety layer's
   logs out first: each run is a `<YYYYmmdd-HHMMSS>-<pid>` directory under
   `/tmp/golem_safety` (the configs' `logging.base_dir`). From the repo root
   in a laptop shell:

   ```bash
   docker cp golem_ros:/tmp/golem_safety container_cache/g1_safety_logs_$(date +%Y%m%d_%H%M)
   ```

2. Run `exit` in terminal 1. The container stops and `systemd-inhibit` lets go.
3. In a laptop shell, set `docker/.env` back to `ROS_DOMAIN_ID=1`,
   `GOLEM_ROBOT=h12` and an empty `G1_DDS_INTERFACE=`, so the next container
   starts on the sim domain.

A controller that dies without its last command (a crash, `docker rm -f`)
leaves the waist and arms on its last command: the safety layer holds it and
only reports the quiet input, until the bringup stops. A walking node that
stops leaves the legs the same way.
