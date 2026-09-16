# Docker smoke tests: Ubuntu NVIDIA and Apple-Silicon Mac

The host wrappers are the supported entry points. They load `docker/.env`,
validate domains and source mounts, and create cache directories. Commands
below run from the GOLEM checkout. Do not run two simulators on the same DDS
domain at once. Keep domain 0 reserved for the real robot.

## Before copying to another machine

The Isaac DDS fix includes an edit in
`CL_isaaclab_sim/src/python/dds/common/dds_master.py`. A superproject commit alone
does not include that file. Commit/push the simulator change in its own repo,
then update GOLEM's submodule pin before testing a fresh clone. No dependency
submodule pins, including Livox, need to change for these Docker fixes.

Initialize submodules and Git LFS assets as described in [SETUP.md](SETUP.md).
Copy `docker/.env.example` to `docker/.env` if it does not exist. Linux's full
model-server bringup also needs the documented weights and API key. Optional
mount roots (quote paths with spaces):

```bash
GOLEM_ASSETS_DIR=/absolute/path/to/CL_Assets
GOLEM_CACHE_DIR=/absolute/path/to/container_cache
ROS_DOMAIN_ID=7
```

Set those in `.env` or export them. Shell exports win. On Mac, the host asset
path must be shared into the Colima VM. Build/install caches remain in Docker
named volumes on the VM; `GOLEM_CACHE_DIR` does not relocate those volumes.

## Ubuntu on Intel/AMD with NVIDIA

Install Docker Compose v2+ and NVIDIA Container Toolkit first.

```bash
# Pull public upstream bases and build the two local images.
docker/scripts/docker_build.sh --pull robocasa ros

# Starts detached; builds ROS/messages before physics, then starts bringup.
docker/scripts/docker_stack.sh up robocasa --headless
docker/scripts/docker_stack.sh logs robocasa
```

Look for `[bringup] simulator is publishing`, then normal ROS node startup.
A readiness timeout fails ROS startup; simulator logs remain available. Check
that required controller/model nodes have not exited. In another terminal:

```bash
docker exec golem_ros bash -lc 'source /home/code/core_ws/install/setup.bash; ros2 node list'
docker exec golem_ros bash -lc 'source /home/code/core_ws/install/setup.bash; ros2 topic hz /clock'
# Ctrl-C ends the topic check.
docker/scripts/docker_stack.sh stop robocasa
```

For GUI testing, set `DISPLAY` and `XAUTHORITY` to the existing desktop session,
then use `up robocasa --gui`. No X11 files are required for headless runs.

Test Isaac separately:

```bash
docker/scripts/docker_build.sh isaac
docker/scripts/docker_run.sh isaac bash       # confirms command override; exit the shell
ROS_DOMAIN_ID=7 docker/scripts/docker_run.sh isaac --task base --headless
```

Isaac must log successful DDS initialization and run without `/dev/tty` errors.
From a second terminal, check a fresh DDS sample using the ROS image:

```bash
ROS_DOMAIN_ID=7 docker/scripts/docker_run.sh ros python3 /home/code/h12_sim_scripts/wait_for_sim.py
```

Stop this single-service simulator before trying `docker_stack.sh up isaac
--headless`. Isaac's existing sensor coverage differs from RoboCasa; this
Docker change does not add missing clock/lidar/IMU functionality or establish
full navigation/skills parity.

## MacBook Air (Apple Silicon): RoboCasa + ROS only

Use native arm64 Colima with Docker Compose; see [MACOS.md](MACOS.md) for setup.
Start with cameras off and no GUI to reduce CPU/memory pressure:

```bash
docker/mac/scripts/docker_build_mac.sh --pull
GOLEM_CAMERAS=0 docker/mac/scripts/docker_stack_mac.sh up robocasa

docker/mac/scripts/docker_stack_mac.sh logs robocasa
docker exec golem_ros bash -lc 'source /opt/core_ws/install/setup.bash; ros2 node list'
docker exec golem_ros bash -lc 'source /opt/core_ws/install/setup.bash; ros2 topic hz /clock'
```

Expect the same readiness message, an advancing `/clock`, and the minimal Mac
controller nodes. The VM uses software rendering and FastDDS; there should be
no NVIDIA runtime or host X11 mount requirement. Isaac is intentionally rejected.

For the optional browser viewer:

```bash
GOLEM_CAMERAS=0 GOLEM_DISPLAY=vnc docker/mac/scripts/docker_stack_mac.sh restart robocasa
docker/mac/scripts/mac_vnc_tunnel.sh --open
```

Then test `GOLEM_RVIZ=vnc` and cameras enabled if resources permit. Finish with:

```bash
docker/mac/scripts/docker_stack_mac.sh stop robocasa
```

## Mount and restart checks (both platforms)

- Set custom asset/cache paths, including one with spaces. Launch and check
  `docker inspect golem_sim_robocasa --format '{{json .Mounts}}'`.
- Temporarily select a nonexistent asset path: the wrapper must fail before
  creating a container, with the missing path in the error.
- Set `ROS_DOMAIN_ID=0`: simulation startup must be refused, including direct
  Compose launcher use. Restore a positive domain afterward.
- Invoke a single-service wrapper twice: the second invocation must refuse to
  replace the first. Explicit `--restart` replaces that selected service.
- Use stack `restart` after a single-service run: it must replace the selected
  simulator and ROS together. The next startup should reuse build caches.

## Automated host checks

```bash
python3 -m unittest discover -s tests/docker -v
```

These exercise wrapper behavior with a recording Docker substitute and render
real Compose configuration without starting containers. They cover mount roots,
missing sources, domain checks, build flags, environment precedence, restart
behavior, and the Mac configuration. They do not establish GPU/GUI runtime,
image build success, controller stability, or performance on a MacBook Air.
