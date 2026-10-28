# First real G1 session

The first session on the real G1 is **arms only**. Unitree's controller keeps
balancing the legs. The laptop runs the bringup (`g1_bringup.launch.py`) with
the [safety layer](safety.md) (`golem_safety`) in arm_sdk mode and the arm
controller (`golem_upper_ik`), and no hand driver.

> **Status.** Nothing has run on the real robot yet. Gains, speed limits and
> estop limits are conservative starting values, not tuned on our G1.

Every command runs on the laptop wired to the robot. Nothing runs on, or
changes, the robot's own computers. Practise the same named configs in the sim
first ([Arms by hand](index.md#arms-by-hand), sending only `home` and
`arms_down`).

## The steps in short

| Step | What happens |
|---|---|
| 1. Freeze the code | Clean `git status`, note `git rev-parse HEAD`. |
| 2. Network | `docker/.env`: `ROS_DOMAIN_ID=0`, `GOLEM_ROBOT=g1`, `G1_DDS_INTERFACE=` the wired port (ours: `enp130s0`). See [DDS and networking](dds.md). |
| 3. Power | Laptop on AC, lid open, container run under `systemd-inhibit`. |
| 4. Robot and people | Robot standing under Unitree's controller, on a gantry or in a harness. Nobody within reach of the arms. |
| 5. Start the container | `docker/scripts/docker_run.sh ros --g1`; it prints `robot: g1` and asks you to confirm domain 0. |
| 6. Read-only checks | Gate: the container is the G1's and pinned to the robot's port, or exit and fix `docker/.env` and the shell's exports. Then nothing else writes the command topics, and `rt/lowstate` arrives. |
| 7. Launch | `ros2 launch g1_bringup g1_bringup.launch.py use_hands:=false use_rviz:=false` |
| 8. Named configs | Send only `home` and `arms_down` on `/named_config`, one at a time. |
| 9. Stop | Stop the controller first (the safety layer latches and hands the arms back), wait for the hand-back, then Ctrl+C the launch. |

## What you should see

- `{"event": "arm_sdk_engage"}`: the safety layer takes the waist and arms
  from Unitree over `engage_seconds`. The controller then moves the arms to
  `home`.
- After stopping the controller (`pkill -INT -f frame_task_server`), at once
  `estop_command_written` with the reason `Command validation failed on
  /safety/lowcmd_upper_in: motor 12 (waist_yaw_joint) kp 0.0 is below
  arm_sdk.waist_kp_min 40.0`. That is the expected stop: the controller's last
  command has zero gains, which the safety layer refuses in arm_sdk mode, so it
  latches and hands the waist and arms back to Unitree over `release_seconds`.
  Then Ctrl+C the launch.
- Any other estop: go straight to step 9.3. After any estop, restart the whole
  bringup.

## Not in this session

- **Hands.** The Dex3 driver bypasses the safety layer, so it stays off.
- **Walking.** It needs Unitree's controller released first. That is a manual
  operator step that nothing in this stack performs, and these docs do not
  describe it. Know this before any walking session: in split mode a fault of
  the walking node (a stale `rt/lowstate`, a non-finite IMU value, a fall, a late
  command) silences it, and the safety layer then **holds its last leg command
  instead of turning every motor off**. That frozen target does not balance the
  robot, and it stands until the safety layer estops for another reason or the
  bringup stops ([Safety layer](safety.md), Quiet inputs). Keep the robot on a
  gantry.
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
- Steps 7 and 9 have not been rehearsed against the real robot. The sim's arm
  test (`tests/g1/test_sim_bringup.sh`) runs the same core (`g1_core.launch.py`,
  with the sim configs) and stops the controller the same way (SIGINT). If `core_ws/src/golem_safety/README.md` or
  `core_ws/src/golem_upper_ik/README.md` at the frozen commit disagrees with
  them, fix this checklist before the session.

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

- The robot stands under Unitree's own controller, on a gantry or in a harness.
- The operator holds Unitree's remote for the whole session.
- Nobody is within reach of the arms. They start moving to `home` about 10 s
  after the launch in step 7.
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
with step 9 first (the controller first, then the hand-back, then Ctrl+C):
`--restart` removes the old container with `docker rm -f`, which kills the
safety node without its hand-back ([Safety layer](safety.md), Stopping). Then
start this one with `docker/scripts/docker_run.sh ros --restart --g1` under the
same `systemd-inhibit`.

The container builds `core_ws` with the `g1_*` packages before it opens the
shell (`launch_ros.sh`, an incremental colcon build that checks every
package's inputs). A failed build stops the container instead of opening the
shell, so the shell means the build matches the frozen tree. To rebuild by
hand anyway, run the container's own build again:

```bash
/home/code/h12_sim_scripts/launch_ros.sh true && source /home/code/core_ws/install/setup.bash
```

Note the safety layer's two ramp times, used in steps 7 and 9:

```bash
grep -E '^ *(engage|release)_seconds:' /home/code/core_ws/src/golem_safety/config/g1/arm_sdk.yaml
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
cyclonedds ls -i 0 -r 5s -t 'rt/(arm_sdk|lowcmd|dex3/.*/cmd)' --suppress-progress-bar --color none \
    | grep -E '__Hostname|__ProcessName|rt/|Publications|Subscriptions'
```

The listing shows each DDS participant that reads or writes those topics.
Under a topic, `Subscriptions` is a reader and `Publications` is a writer.
Nothing may show `Publications` under `rt/arm_sdk`, `rt/lowcmd` or
`rt/dex3/.../cmd`: two writers fight over the motors. If a writer is a program
on this laptop (its `__Hostname` and `__ProcessName` say which; `docker ps` in a
laptop shell lists leftover containers), stop it and list again. If it is
anywhere else, do not launch: resolving that is not a laptop-side step.

Then the robot's state:

```bash
python3 /home/code/third_party/leprechaun_g1/tools/read_lowstate.py
```

It only subscribes (domain 0 on `enp130s0`, both fixed in the script), prints
once a second, and stops with Ctrl+C. Check that:

- it shows `mode_machine: 5`;
- `tick` grows from one print to the next;
- the arm joints (slots 15-28) match how the arms look.

If it only prints `no data yet`, check the cable, then in a laptop shell
`ip -br addr show dev enp130s0` and `sudo ufw status` (it should be inactive).

### 7. First launch (terminal 1)

```bash
PYTHONUNBUFFERED=1 ros2 launch g1_bringup g1_bringup.launch.py use_hands:=false use_rviz:=false
```

`PYTHONUNBUFFERED=1` is harmless: it makes every Python node's prints appear as
they happen. The safety layer logs its events through its ROS logger, as
`[INFO] [...] [safety_node]: {"event": ...}` (or `[WARN]`, `[ERROR]`). With
`use_hands:=false` no hand driver runs, so nothing commands the hands.
`use_rviz:=false` leaves out RViz, which this session does not need.

The safety layer starts first, prints `relay_started`, and writes nothing yet.
Two seconds later the controller starts. When the safety layer takes the waist
and arms from Unitree, it prints `{"event": "arm_sdk_engage"}` and ramps them
over in `engage_seconds` (step 5), and the controller moves the arms to `home`.
Until then it may print `engage_refused` with a reason, at most once a second;
the controller stops with an error if the safety layer has not taken over
within 30 s.

If the safety layer prints `estop_command_written` or `estop_latched_silent`
at any point, or the controller exits with an error, continue at step 9.3.

### 8. Named configs only (terminal 2)

This session uses only `home` and `arms_down`. Send one goal at a time, and
wait for each to finish before sending the next:

```bash
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'arms_down', plan: false, duration: {sec: 0, nanosec: 0}}"
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'home', plan: false, duration: {sec: 0, nanosec: 0}}"
```

Send no other goals, and start no second controller: a second live writer on
the safety layer's input latches its estop. `g1_control`, unused and awaiting
deletion, still builds, and its nodes and the scripts in its `example/`
directory publish to the same input as `frame_task_server`; run none of
them.

### 9. Stop

1. Wait until no goal is running. (In a later session with the hands, empty
   them first: on exit the hand driver releases them and the fingers go limp.)
2. In terminal 2, stop the controller alone:

   ```bash
   pgrep -af frame_task_server    # one line: the controller node
   pkill -INT -f frame_task_server
   ```

   The controller sends one last command with zero gains and stops. The
   safety layer refuses it at once: it prints `"event": "estop_command_written"`
   with the reason `Command validation failed on /safety/lowcmd_upper_in:
   motor 12 (waist_yaw_joint) kp 0.0 is below arm_sdk.waist_kp_min 40.0`, and
   it ramps the waist and arms back to Unitree over `release_seconds` (step 5),
   holding the last good command meanwhile. That estop is the expected stop. It
   prints `arm_sdk_release_ramping` when that ramp runs, and
   `arm_sdk_release_held` if `rt/lowstate` goes stale, which pauses the
   ramp.
3. The safety layer prints nothing when that ramp ends. Wait
   `release_seconds` of ramping after the `estop_command_written` line, then
   press Ctrl+C on the launch in terminal 1.
4. The container is deleted when its shell exits, so copy the safety layer's
   logs out first: each run is a `<YYYYmmdd-HHMMSS>-<pid>` directory under
   `/tmp/g1_safety_layer` (the configs' `logging.base_dir`). From the repo root
   in a laptop shell:

   ```bash
   docker cp golem_ros:/tmp/g1_safety_layer container_cache/g1_safety_logs_$(date +%Y%m%d_%H%M)
   ```

5. Run `exit` in terminal 1. The container stops and `systemd-inhibit` lets go.
6. In a laptop shell, set `docker/.env` back to `ROS_DOMAIN_ID=1`,
   `GOLEM_ROBOT=h12` and an empty `G1_DDS_INTERFACE=`, so the next container
   starts on the sim domain.

Stopping the controller first makes the safety layer ramp the `rt/arm_sdk`
weight down to 0 while it still runs. That is how Unitree's own arm_sdk example
(`unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py`) hands
the arms back: it ramps the weight from 1 to 0 over 3 s. Ctrl+C on the launch
also hands back (`golem_safety`'s README: SIGINT, SIGTERM and SIGHUP ramp the
weight to 0 before the node exits), but this checklist does not rely on it. A
controller that dies without its last command is different: the safety layer
holds its last command at weight 1 and only reports the quiet input, until the
bringup stops. What the G1 does if `rt/arm_sdk` just stops while the weight is
still 1 is unknown.

If anything looks wrong at any point, the operator's remote comes first. On the
laptop, stop with the `pkill` in step 9.2.
