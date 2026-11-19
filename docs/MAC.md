# The Mac (Apple Silicon)

The stack's platform is x86 Linux with an NVIDIA GPU. An Apple Silicon Mac runs
a reduced RoboCasa sim of either robot: the command path (`golem_safety`,
`golem_upper_ik`), the legs (`golem_lowerbody_rl`) and walking on `/cmd_vel`.
It has no lidar, no navigation, no model servers or skills, and no Isaac sim.
Use it to develop and check controllers and bringup away from the lab
machines; the closed-loop tests (`tests/sim`, `tests/g1`) stay x86 gates.

## What differs from x86

| | x86 + NVIDIA | Mac |
|---|---|---|
| Images | `golem_*:latest`, `docker/*Dockerfile` | `golem_*:arm64`, `docker/mac/*Dockerfile`, native arm64 |
| Python locks | `docker/locks/*-amd64*.txt` | `docker/locks/*-arm64*.txt`, derived from the amd64 locks by `docker/mac/derive_arm64_locks.sh` (CPU torch, no CUDA wheels, open3d 0.18.0) |
| Compose | `docker/docker-compose.yml` | the same, plus `docker/docker-compose.mac.yml` (no NVIDIA runtime) |
| Sim rendering | EGL on the GPU (`GOLEM_RENDER=gpu`) | OSMesa on the CPU (`GOLEM_RENDER=cpu`); cameras off by default (`GOLEM_CAMERAS=0`) |
| MID-360 lidar | on | none (`GOLEM_LIDAR=0`): its caster renders on the GPU |
| Bringup | `h12_sim`, `g1_sim` | `h12_mac_sim`, `g1_mac_sim`: the sim bringups without navigation, model servers or skills |
| Not available | — | Isaac, navigation and SLAM, `golem_skills`, the model servers (GraspGenX has no CPU path) |

The wrappers pick the Mac by themselves on Darwin arm64 (`GOLEM_PLATFORM=mac`;
`GOLEM_PLATFORM=linux` forces the x86 path) and print
`platform: mac (arm64, CPU rendering, no lidar)` beside `robot:`. Nothing on
Linux changes: there the wrappers make the same docker calls as before.

## Setup

- [OrbStack](https://orbstack.dev) or Docker Desktop (4.34 or later, host
  networking turned on), Compose 2.24 or later. Both containers share the Linux
  VM's network, so DDS works between them; the macOS host is not on it, so run
  `ros2` inside `golem_ros`.
- Submodules and LFS as in [Setup](SETUP.md) §1; `docker/.env` as in §2
  (`GEMINI_API_KEY` is not needed).
- About 70 GB of disk for the images (the RoboCasa image holds about 10 GB of
  kitchen assets) and 8 GB of memory for the VM.

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
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: t_pose, plan: false, duration: {sec: 15, nanosec: 0}}"
```

`docker_stack.sh` runs a random kitchen task; with the arms out (`t_pose`) the
robot can touch a fixture, and the goal then fails its steady-state hold. The
three-terminal flow of [Running the stack](RUNNING.md) works too
(`docker_run.sh robocasa --g1 --task Blank`, then `ros2 launch golem_bringup
g1_mac_sim.launch.py` in `golem_ros`).

## Measured speed

Measured 2026-11-18 on a MacBook Air M3 (4 performance + 4 efficiency cores,
16 GB), OrbStack with 8 CPUs and 8 GB, headless, one run each. The real-time
factor (RTF) is sim seconds per wall second: the G1's from its
`[g1_mujoco] ... real-time factor` lines, the H1-2's from `--timing`, the
stacks' from `/clock` against wall time in `golem_ros`.

**The sim alone** (no stack: motors off, unpaced, `GOLEM_SIM_RATE=max`):

| Scene | G1 | H1-2 |
|---|---|---|
| Blank, no sensors | 9.1-10.0x | — |
| Kitchen (layout 9, style 9, seed 42), no sensors | 5.6x | 7.4x |
| Kitchen, cameras on (software rendering) | 5.3x (head camera drops frames due while one renders: about 10 frames per wall s) | 0.25x (three RGBD cameras, about 72 ms per colour render; the H1-2 sim waits for them) |

**The whole stack at real time** (`docker_stack.sh`, `GOLEM_SIM_RATE` unset):

| | G1 (`g1_mac_sim`) | H1-2 (`h12_mac_sim`) |
|---|---|---|
| RTF | 1.00 (sim loop idle 60-74%) | 0.95-0.99 |
| `/lowstate` | 500 Hz, largest gap 10-19 ms | 470-490 Hz, gaps up to 180 ms |
| Start-up | safety, legs pre-posed, arms, policy, band released: all as on x86 | safety, estop, legs pre-posed, arms, confirm, band released |
| Walking | 0.3 m/s forward, turning, arms moving while walking: no fall, no fault | stood 53 s after release, then fell: the estop latched (below) |
| Arms (`/named_config`, 15 s) | Blank scene: 8 of 8 goals succeed, standing and walking | not measured |
| CPU | sim about 0.4 core, ros about 1.8 cores | sim about 0.25 core, ros about 1.85 cores |

For comparison, the x86 lab machine (i7-14700F, RTX 5070 Ti, sensors and
navigation on) runs the full stack at 0.83x (H1-2) and 1.00x (G1)
(`docs/superpowers/specs/2026-11-05-sim-speed-results.md`).

**The H1-2 stack is not yet usable on the Mac.** Its stack stalls for up to
about 180 ms of wall time now and then. `estop_node` declares the sim's estop
stream stale after 0.1 s (`line_stale_timeout_sec`, the real robot's value), so
a stall latches `golem_safety`'s estop, the legs stop and the robot falls; in
the measured run that came 53 s after the band's release. The G1 has no estop
node and its stack's worst gap was 19 ms. Where the stalls come from (the VM's
scheduling or the heavier H1-2 ROS side) is not measured.

## Image notes

- The Mac base starts from a dated Ubuntu image older than the apt snapshot
  (`ubuntu:jammy-20260901.2`): a newer one carries packages newer than the
  snapshot, which apt cannot install over.
- The sims run the Unitree SDK's CycloneDDS beside ROS's in one process, each
  creating the same DDS domain, which needs two separate `libddsc` copies. On
  arm64 the SDK's binding would bind to ROS's copy by their shared soname and
  segfault in its first topic, so the Mac base renames the SDK's copy
  (`libddsc_unitree.so.0`) and checks the pair at build time
  (`docker/mac/check_unitree_dds.py`).
- `core_ws/cyclonedds.xml` requires 16 MB UDP receive buffers, which the VM's
  default `net.core.rmem_max` refuses (every node then fails to create its
  domain); the Mac overlay's `CYCLONEDDS_URI` drops that minimum.
