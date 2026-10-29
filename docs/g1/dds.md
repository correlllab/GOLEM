# DDS and networking

The stack talks to the robot (or the sim) over **Unitree DDS** (CycloneDDS).
`golem_safety` and `golem_upper_ik` read and write those topics as ROS 2
topics, which run on CycloneDDS too (ROS `/lowcmd` is DDS `rt/lowcmd`, the
same wire type); `g1_locomotion` and `g1_dex3` use the Unitree SDK in
`unitree_sdk2_python/`. ROS 2 also carries the user-facing topics.

## Domains

| Domain | Used for |
|---|---|
| `1` | the sim (the default) |
| `0` | the real robots. This is their **command bus**. |
| `97` | the G1 package DDS tests and `tests/robocasa/g1/`, which use fake robot peers or a disposable sim |
| `194` | the topic-surface probe, `tests/robocasa/surface/run.sh` |
| `195`-`198` | the G1 acceptance tests in `tests/g1/` (arm, walk, nav, explore) |

`tests/README.md` lists the H1-2 tests' domains.

Set the domain in `docker/.env` (copy it from `docker/.env.example`):

```bash
ROS_DOMAIN_ID=1        # 0 only for a real-robot session
GOLEM_ROBOT=h12        # g1 selects the G1 (or pass --g1 to a wrapper)
G1_DDS_INTERFACE=      # empty for the sim; the wired port for the robot (ours: enp130s0)
```

Keep `ROS_DOMAIN_ID=1` unless you are running a real-robot session. Set it
back to `1` and empty `G1_DDS_INTERFACE` afterwards. A variable exported in
the shell wins over `docker/.env`, so also check the shell
(`env | grep -E '^(ROS_DOMAIN_ID|GOLEM_ROBOT|G1_DDS_INTERFACE)='`, then `unset`
what should come from the file).

`G1_DDS_INTERFACE` is for the real robot only. The interface pin below applies
to the `ros` container alone; the sim's ROS nodes are never pinned, so with it
set on a sim domain the sim and the `ros` container can end up on different
interfaces and miss each other's topics. The wrappers print a warning when it
is set and `ROS_DOMAIN_ID` is not 0.

**One robot per domain.** The H1-2 and G1 stacks share node, action and DDS
topic names, with identical `unitree_hg` types, so two of them on one domain
command each other's robots. `docker/scripts/wait_for_sim.py` refuses to start
a bringup against a simulator of the other robot (it reads `mode_machine` from
`rt/lowstate`: 5 for the G1, 6 for the H1-2, each sim reporting its real
robot's value), but only in `docker_stack.sh` and the mac stack
(`GOLEM_WAIT_FOR_SIM=1`) and in `tests/stack`. `docker_run.sh ros` followed by
a manual `ros2 launch` is not checked by the wrappers. `golem_safety` checks
the value as well: it engages no input on another robot's `rt/lowstate` and
latches its estop on one once engaged. That does not replace one robot per
domain: the two stacks share the `rt/safety/*` input topics and the node and
action names, `golem_safety` does not check which robot an input command was
written for, and `golem_upper_ik` does not check `rt/lowstate`'s
`mode_machine`.

## Guards against the wrong domain

`docker/scripts/docker_common.sh` holds the wrappers' checks:

- It rejects a `ROS_DOMAIN_ID` that is not an integer from 0 to 232 (a
  `docker/.env` with Windows line endings fails here), and defaults an empty
  one to `1`.
- It refuses domain 0 for every simulator. For the `ros` container on Linux it
  accepts 0 only after you answer `y` to a prompt, which names the robot. With
  the G1 selected it first requires `G1_DDS_INTERFACE` to name an interface
  that exists, is `UP` and has an IPv4 address, and the prompt also names the
  interface and its IPv4 addresses; with the H1-2 selected there is no
  interface check.
- `docker_stack.sh` refuses domain 0 altogether.

Behind the wrappers:

- `launch_robocasa.sh` and the G1 sim (`golem_robocasa/g1/unitree_interface.py`)
  refuse domain 0 even when started some other way.
- `golem_safety`'s `safety_node` refuses domain 0 when started with
  `sim:=true`, as the sim bringups start it, an unset `ROS_DOMAIN_ID`
  included.
- The G1 tests run their containers with no network of their own, so no DDS
  leaves the laptop. `tests/g1/test_sim_*.sh` refuse domain 0, and
  `tests/robocasa/g1/run_in_container.sh` always uses 97.
- The by-hand sim flow (`docker_run.sh robocasa --g1` and `docker_run.sh ros`)
  uses host networking on domain 1, so its traffic can reach other machines on
  your network.

## The wired port

On a laptop with WiFi and a wired port to the robot, DDS must use the wired
port or it never sees the robot. For the G1, `docker_run.sh ros` and
`docker_stack.sh up|restart` write `$GOLEM_CACHE_DIR/cyclonedds_local.xml` (by
default `container_cache/cyclonedds_local.xml`) from `G1_DDS_INTERFACE` just
before they start a `ros` container (empty: `<CycloneDDS/>`, and CycloneDDS
chooses). The `ros` container mounts it at
`/home/code/cyclonedds_local.xml` and loads it after `core_ws/cyclonedds.xml`
(`CYCLONEDDS_URI` in `docker/docker-compose.g1.yml`), which pins ROS 2 to that
port. The Unitree SDK ignores `CYCLONEDDS_URI`, so the nodes that use it read
`G1_DDS_INTERFACE` directly.

The mount is of that single file, so a running container keeps the file it
started with, and CycloneDDS reads it again in every process started there
(each `ros2` command, each node). That is why the wrappers replace the file by
a rename, never by writing it in place: a running domain-0 container keeps its
pin whatever runs next on the host. Nothing else writes it: `docker_run.sh
robocasa`, `tests/robocasa/g1/run_in_container.sh` and
`tests/robocasa/surface/run.sh` start no `ros` container, and the acceptance
tests in `tests/g1/` mount a file of their own. The wrappers refuse a
`G1_DDS_INTERFACE` that is not an interface name (letters, digits and
`_ . @ : -`, at most 15 characters). Check the port with:

```bash
ip -br addr show dev enp130s0     # UP, 192.168.123.222/24 on our laptop
```

and, inside the container, `cat /home/code/cyclonedds_local.xml`.

## The robot's network

What worked on 2026-09-28 from an Ubuntu 24.04 laptop (our unit's details are
in [Our G1](hardware.md)):

```
laptop ─┬─ enp130s0   192.168.123.222 ── cable ── robot switch ─┬─ 192.168.123.161  motion computer (Unitree's)
        │                                                        └─ 192.168.123.164  dev computer (ours)
        └─ WiFi ── internet
```

- The robot's network is a private wired subnet, `192.168.123.0/24`, with fixed
  addresses and no DHCP. Give the laptop's wired port a manual address,
  `192.168.123.222/24`, with no gateway, so internet traffic stays on WiFi.
- Use the cable. The SDK finds the robot by DDS multicast on the wired subnet,
  which does not reach across from WiFi.
- If pings come and go, check `ip -br link show enp130s0`; it should say UP.
  Ours flapped at 100 Mb/s until the cable connection was fixed, then linked at
  1 Gb/s.

The robot talks DDS (CycloneDDS 0.10.2) on domain 0. `unitree_sdk2_python`
speaks DDS directly. ROS 2, through Unitree's `unitree_ros2` (in GOLEM,
`core_ws/src/unitree_ros2`), sees the same topics with `/` in place of `rt/`.

## The robot's topics

The ones that matter first:

| Kind | Topics |
| --- | --- |
| State | `rt/lowstate` (joints, IMU, remote; `rt/lf/lowstate` is a lower-rate copy), `rt/dex3/{left,right}/state`, `rt/secondary_imu`, `rt/lf/bmsstate` (battery), `rt/sportmodestate`, `rt/odommodestate`, `rt/state_estimator/*` |
| Commands | `rt/lowcmd` (every motor), `rt/arm_sdk` (arms, while Unitree's controller balances), `rt/dex3/{left,right}/cmd` |
| Services | `rt/api/<name>/request` and `/response`, e.g. `sport`, `motion_switcher`, `arm`, `voice` |

`rt/lowstate`'s layout and motor slots are in [Our G1](hardware.md#rtlowstate).

## Reading the robot without the stack

The first session reads the robot from the G1's `ros` container
([First real session](first-session.md), step 6), which has the SDK and the
`cyclonedds` tool. Outside the containers, on the laptop itself:

cyclonedds 0.10.2 ships wheels only up to Python 3.10, so make an environment
for it:

```bash
conda create -n g1 python=3.10
conda activate g1
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e .
```

List every topic on the robot (the export points `cyclonedds`, which comes
with the cyclonedds package, at the robot's port):

```bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="enp130s0"/></Interfaces></General></Domain></CycloneDDS>'
cyclonedds ls -r 5s --suppress-progress-bar --color none | grep -o 'rt/[A-Za-z0-9_/]*' | sort -u
```

Print `rt/lowstate`, from the repo root:

```bash
python tools/read_lowstate.py
```

Read-only: it subscribes and never publishes. Once a second it prints
`mode_machine`, `mode_pr` and `tick`, the IMU roll, pitch and yaw, and each of
the 29 joints by name with its angle. Stop it with Ctrl+C.

## Dropped samples (UDP buffers)

Camera images and point clouds arrive in bursts that overflow the kernel's
default UDP receive buffer, and DDS drops samples silently.
`core_ws/cyclonedds.xml` asks for a larger buffer, but the host must allow it:
see [`DDS_TUNING.md`](../DDS_TUNING.md) for the sysctls. To check, watch
`RcvbufErrors`; it should stop climbing:

```bash
docker exec golem_ros grep '^Udp:' /proc/net/snmp
```

## Talking to the sim from the host

Host shells do not read `docker/.env`. Load it first, or the topics are
invisible:

```bash
set -a; source docker/.env; set +a
```
