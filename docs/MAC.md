# The Mac (Apple Silicon)

The stack's platform is x86 Linux with an NVIDIA GPU. An Apple Silicon Mac runs
a reduced RoboCasa sim of either robot at real time: the command path
(`golem_safety`, `golem_upper_ik`), the legs (`golem_lowerbody_rl`), and
navigation (FAST-LIO, SLAM, Nav2) on a MID-360 cast on the CPU. It has no model
servers or skills, and no Isaac sim. Use it to develop and check controllers,
bringup and navigation away from the lab machines; the closed-loop tests
(`tests/sim`, `tests/g1`) stay x86 gates.

## What differs from x86

| | x86 + NVIDIA | Mac |
|---|---|---|
| Images | `golem_*:latest`, `docker/*Dockerfile` | `golem_*:arm64`, `docker/mac/*Dockerfile`, native arm64 |
| Python locks | `docker/locks/*-amd64*.txt` | `docker/locks/*-arm64*.txt`, derived from the amd64 locks by `docker/mac/derive_arm64_locks.sh` (CPU torch, no CUDA wheels, open3d 0.18.0) |
| Compose | `docker/docker-compose.yml` | the same, plus `docker/docker-compose.mac.yml` (no NVIDIA runtime) |
| Sim rendering | EGL on the GPU (`GOLEM_RENDER=gpu`) | OSMesa on the CPU (`GOLEM_RENDER=cpu`); cameras off by default (`GOLEM_CAMERAS=0`) |
| MID-360 lidar | cast on the GPU (`mid360.GpuCaster`), 20,000 points a frame | cast exactly on the CPU (`mid360.CpuCaster`); `GOLEM_LIDAR_POINTS` 6,000 (H1-2) or 10,000 (G1) a frame by default |
| Bringup | `h12_sim`, `g1_sim` | `h12_mac_sim`, `g1_mac_sim`: the sim bringups without model servers or skills; the G1's without the hand driver (`use_hands:=true` adds it) |
| Not available | — | Isaac, `golem_skills`, the model servers (GraspGenX has no CPU path) |

The wrappers pick the Mac by themselves on Darwin arm64 (`GOLEM_PLATFORM=mac`;
`GOLEM_PLATFORM=linux` forces the x86 path). Nothing on Linux changes: there the
wrappers make the same docker calls as before.

## Setup

- [OrbStack](https://orbstack.dev) or Docker Desktop (4.34 or later, host
  networking turned on), Compose 2.24 or later. Both containers share the Linux
  VM's network, so DDS works between them; the macOS host is not on it, so run
  `ros2` inside `golem_ros`.
- Submodules and LFS as in [Setup](SETUP.md) §1; `docker/.env` as in §2
  (`GEMINI_API_KEY` is not needed).
- About 70 GB of disk for the images (the RoboCasa image holds about 10 GB of
  kitchen assets), and every core and 8 GB of memory for the VM.

```bash
docker/scripts/docker_build.sh            # golem_base, golem_sim_robocasa, golem_ros, all :arm64 (about 35 min)
```

## Running

```bash
# The G1: the sim plus g1_mac_sim; the leg node releases the band by itself
GOLEM_SPAWN_BACKOFF=2.0 docker/scripts/docker_stack.sh up robocasa --g1
docker/scripts/docker_stack.sh logs robocasa      # wait for "elastic band released"

# The H1-2: the sim plus h12_mac_sim; the legs hold until the operator confirms
GOLEM_SPAWN_BACKOFF=2.0 docker/scripts/docker_stack.sh up robocasa --h12
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger
```

Then send `/navigate_to_pose` goals in `map` once Nav2's lifecycle nodes are up,
as on x86 ([Navigation demo](NAVIGATION_DEMO.md)). The three-terminal flow of
[Running the stack](RUNNING.md) works too (`docker_run.sh robocasa --g1`, then
`ros2 launch golem_bringup g1_mac_sim.launch.py` in `golem_ros`).

## Measured speed

Measured 2026-11-19 on a MacBook Air M3 (4 performance + 4 efficiency cores,
16 GB), OrbStack with 8 CPUs and 8 GB, headless, one run each: the whole
navigation stack (`*_mac_sim`), kitchen layout 9, style 9, seed 42, the spawn
backed off 2.0 m, a Nav2 goal 1.5 m ahead and one back to the start. The
real-time factor (RTF) is `/clock` against wall time in `golem_ros`.

| | Points a frame | RTF | `/lowstate` largest gap | Lidar casts the sim waited for | Goal ahead / back to start |
|---|---|---|---|---|---|
| G1, hand driver on | 20,000 | 0.80-0.89 | 40 ms | 80% | succeeded / aborted 0.85 m short |
| G1 (default) | 10,000 | 1.00 (sim loop idle 26%) | 24 ms | 2% | succeeded (0.27 m off) / not run |
| H1-2 | 20,000 | 0.55-0.57 | 46 ms | 99% | succeeded (0.18 m) / succeeded (0.04 m) |
| H1-2 | 10,000 | 0.88-0.95 | 25 ms | 87% | succeeded (0.23 m) / succeeded (0.04 m) |
| H1-2 (default) | 6,000 | 1.00 | 17 ms | 22% | succeeded (0.27 m) / succeeded (0.11 m) |

No fault or estop latch in these runs. The VM is CPU-bound: `golem_ros` takes
about 4 cores and the sim about 2. The H1-2 needs fewer points than the G1
because its rays hit every geom group, the kitchen's collision geoms included
(`h12_sensors.LIDAR_GEOM_GROUPS`), where the G1's hit the visual geoms only.
The sparser scans have not been checked for SLAM quality beyond these goals. A
Mac with more cores holds real time with more points (`GOLEM_LIDAR_POINTS`).

The H1-2's `estop_node` declares the sim's estop stream stale after 0.1 s, so a
VM stall longer than that latches `golem_safety`'s estop and the robot falls; it
happened once, in an earlier run with the VM overloaded.

## Image notes

- The Mac base starts from a dated Ubuntu image older than the apt snapshot
  (`ubuntu:jammy-20260901.2`): a newer one carries packages newer than the
  snapshot, which apt cannot install over.
- The sims run the Unitree SDK's CycloneDDS beside ROS's in one process, each
  creating the same DDS domain, which needs two separate `libddsc` copies. On
  arm64 the SDK's binding would bind to ROS's copy by their shared soname and
  segfault in its first topic, so the Mac base renames the SDK's copy
  (`libddsc_unitree.so.0`) and checks the pair at build time.
- The Mac's GPU is out of reach: a container gets no OpenGL/EGL on macOS, which
  MuJoCo's renderer and `mid360.GpuCaster` need.
- `core_ws/cyclonedds.xml` requires 16 MB UDP receive buffers, which the VM's
  default `net.core.rmem_max` refuses (every node then fails to create its
  domain); the Mac overlay's `CYCLONEDDS_URI` drops that minimum.
