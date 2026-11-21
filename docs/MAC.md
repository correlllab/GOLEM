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
| Bringup | `h12_sim`, `g1_sim` | `h12_mac_sim`, `g1_mac_sim`: the sim bringups without model servers or skills; the G1's without the hand driver (`use_hands:=true` adds it); RViz on a virtual display, in a browser at `localhost:6080` |
| Not available | — | Isaac, `golem_skills`, the model servers (GraspGenX has no CPU path) |

The wrappers pick the Mac by themselves on Darwin arm64 (`GOLEM_PLATFORM=mac`;
`GOLEM_PLATFORM=linux` forces the x86 path). Nothing on Linux changes: there the
wrappers make the same docker calls as before.

## Setup

- [OrbStack](https://orbstack.dev), Docker Desktop (4.34 or later, host
  networking turned on) or [Colima](https://github.com/abiosoft/colima) (the
  `vz` VM), Compose 2.24 or later. Both containers share the Linux VM's
  network, so DDS works between them; the macOS host is not on it, so run
  `ros2` inside `golem_ros`. Colima shares only your home folder with its VM
  by default, so keep the checkout under `~`, and give its VM the disk and
  cores below (`colima start --cpu 12 --memory 16 --disk 150`; the disk only
  grows).
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

**RViz.** The Mac bringups start RViz by default (`use_rviz:=false` leaves it
out). A Mac has no X server a container can draw OpenGL through, so the
`golem_ros` container draws on a virtual display (Xvfb, Mesa's software OpenGL)
that `docker/mac/vnc_display.sh` starts with the container. Open it in a
browser at <http://localhost:6080/vnc.html>, or in Screen Sharing at
`vnc://localhost:5901`. Software OpenGL draws on one thread (`LP_NUM_THREADS=1`):
on every core RViz takes about 1.5 of them and the H1-2's stack falls to
0.78-0.83x; on one it holds 0.98-1.00x. `GOLEM_VNC=0` starts no display, and
`GOLEM_VNC_SIZE` sets its size (default `1280x800`).

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

### By sensor, on an M4 Pro

Measured 2026-11-20 on a MacBook Pro M4 Pro (10 performance + 4 efficiency
cores, 48 GB), Colima with 12 CPUs and 16 GB, headless: the G1, the spawn backed
off 2.0 m, the robot standing after the band's release, no Nav2 goals. RTF is
the sim's own log (`golem_sim`, sim against wall time over 30 s of sim time),
the first 30 s dropped and the next four averaged. Lidar is `GOLEM_LIDAR` at
10,000 points unless noted; the camera is `GOLEM_CAMERAS=1`, the head D435 at
640x360 and 6 Hz. The stack is `g1_mac_sim`, RViz included.

Paced at real time (`GOLEM_SIM_RATE` unset), one run each:

| | Lidar | Camera | RTF | Sim loop idle | Camera frames dropped | CPUs: sim / `golem_ros` |
|---|---|---|---|---|---|---|
| Sim alone | off | off | 1.00 | 68% | | 0.3 / - |
| Sim alone | off | on | 1.00 | 73% | 0% | 1.0 / - |
| Sim alone | on | off | 1.00 | 71% | | 0.7 / - |
| Sim alone | on | on | 1.00 | 70% | 0% | 1.8 / - |
| Sim and stack | off | off | 1.00 | 58% | | 0.6 / 4.4 |
| Sim and stack | off | on | 1.00 | 52% | 1% | 2.8 / 4.3 |
| Sim and stack | on | off | 1.00 | 47% | | 1.5 / 5.3 |
| Sim and stack | on | on | 1.00 | 50% | 0% | 2.6 / 5.3 |
| Sim and stack | 20,000 points | off | 1.00 | 50% | | 1.6 / 5.6 |

No fall or estop in any run, and the sim never waited for a lidar cast.

The sim alone unpaced (`GOLEM_SIM_RATE=max`), three runs each:

| Lidar | Camera | RTF | Lidar casts the sim waited for | Camera frames dropped |
|---|---|---|---|---|
| off | off | 4.0-5.9 | | |
| off | on | 4.3-5.3 | | 50% |
| on | off | 4.0-4.6 | 0-71% | |
| on | on | 3.5-4.4 | 0-70% | 50% |
| 20,000 points | off | 2.4-3.2 | 3-99% | |

The physics alone runs on one core at 4-6 times real time. The lidar's 10,000
points cost it 10-20%, the real sensor's 20,000 about 40%, and then the cast is
what the sim waits for. The camera takes one to two cores but renders on its own
thread, so it hardly slows the physics: it renders about 16 frames a wall
second, dropping half the frames due when unpaced and at most 1% at real time.
Each run held steady within itself, but the same setup differed by up to 40%
between runs, presumably the VM's physics thread landing on performance or
efficiency cores (not checked).

## Image notes

- The Mac base starts from a dated Ubuntu image older than the apt snapshot
  (`ubuntu:jammy-20260901.2`): a newer one carries packages newer than the
  snapshot, which apt cannot install over.
- The sims run the Unitree SDK's CycloneDDS beside ROS's in one process, each
  creating the same DDS domain, which needs two separate `libddsc` copies. On
  arm64 the SDK's binding would bind to ROS's copy by their shared soname and
  segfault in its first topic, so the Mac base renames the SDK's copy
  (`libddsc_unitree.so.0`) and checks the pair at build time.
- The colcon build trees (`core_ws`'s `build`, `install` and `log`, and the
  sim's `msgs_ws`) are Docker volumes on the VM's disk, not folders in the
  checkout (`docker-compose.mac.yml`). Every file a container writes through
  the VM's file sharing reaches the macOS host, and on a university-managed
  Mac its endpoint security scans each one: on an M4 Pro under Colima a write
  took about 180 ms, and a cold `core_ws` build ran over 40 minutes at 15% CPU
  without finishing; with the volumes a cold build and stack start takes 2.5
  minutes. The sources stay bind-mounted, so edits show up as before; for a clean build,
  `docker volume rm golem_core_build golem_core_install golem_core_log
  golem_msgs_ws` with the containers stopped.
- The Mac's GPU is out of reach: a container gets no OpenGL/EGL on macOS, which
  MuJoCo's renderer and `mid360.GpuCaster` need.
- `core_ws/cyclonedds.xml` requires 16 MB UDP receive buffers, which the VM's
  default `net.core.rmem_max` refuses (every node then fails to create its
  domain); the Mac overlay's `CYCLONEDDS_URI` drops that minimum.
