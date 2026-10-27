# DDS and networking

The stack talks to the robot (or the sim) over **Unitree DDS** (CycloneDDS),
through the Unitree SDK in `unitree_sdk2_python/`. ROS 2 also runs on
CycloneDDS, as a separate layer for the user-facing topics.

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
back to `1` and empty `G1_DDS_INTERFACE` afterwards.

**One robot per domain.** The H1-2 and G1 stacks share node, action and DDS
topic names, with identical `unitree_hg` types, so two of them on one domain
command each other's robots. `docker/scripts/wait_for_sim.py` refuses to start
a bringup against a simulator of the other robot (it reads `mode_machine` from
`rt/lowstate`: 5 for the G1, 0 for the H1-2 sims), but only in
`docker_stack.sh` and the mac stack (`GOLEM_WAIT_FOR_SIM=1`) and in
`tests/stack`. `docker_run.sh ros` followed by a manual `ros2 launch` is not
checked.

## Guards against the wrong domain

`docker/scripts/docker_common.sh` holds the wrappers' checks:

- It rejects a `ROS_DOMAIN_ID` that is not an integer from 0 to 232 (a
  `docker/.env` with Windows line endings fails here), and defaults an empty
  one to `1`.
- It refuses domain 0 for every simulator. For the `ros` container on Linux it
  accepts 0 only after you answer `y` to a prompt. With the G1 selected it
  first requires `G1_DDS_INTERFACE` to name an interface that exists, is `UP`
  and has an IPv4 address; with the H1-2 selected there is no interface check.
- `docker_stack.sh` refuses domain 0 altogether.

Behind the wrappers:

- `launch_robocasa.sh` and the G1 sim (`h1_robocasa/g1/unitree_interface.py`)
  refuse domain 0 even when started some other way.
- The safety layer's sim configs (`sim_arm_sdk.yaml`, `sim_split.yaml`) refuse
  domain 0.
- The G1 tests run their containers with no network of their own, so no DDS
  leaves the laptop. `tests/g1/test_sim_*.sh` refuse domain 0, and
  `tests/robocasa/g1/run_in_container.sh` always uses 97.
- The by-hand sim flow (`docker_run.sh robocasa --g1` and `docker_run.sh ros --g1`)
  uses host networking on domain 1, so its traffic can reach other machines on
  your network.

## The wired port

On a laptop with WiFi and a wired port to the robot, DDS must use the wired
port or it never sees the robot. For the G1 the wrappers write
`$GOLEM_CACHE_DIR/cyclonedds_local.xml` (by default
`container_cache/cyclonedds_local.xml`) from `G1_DDS_INTERFACE`. The `ros` container mounts it at
`/home/code/cyclonedds_local.xml` and loads it after `core_ws/cyclonedds.xml`
(`CYCLONEDDS_URI` in `docker/docker-compose.g1.yml`), which pins ROS 2 to that
port. The Unitree SDK ignores `CYCLONEDDS_URI`, so the nodes that use it read
`G1_DDS_INTERFACE` directly. Check the port with:

```bash
ip -br addr show dev enp130s0     # UP, 192.168.123.222/24 on our laptop
```

and, inside the container, `cat /home/code/cyclonedds_local.xml`.

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
