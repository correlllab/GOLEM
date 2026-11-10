# Docker smoke tests

The host wrappers are the supported entry points. They load `docker/.env`,
validate domains and source mounts, and create cache directories. Commands
below run from the GOLEM checkout. Do not run two simulators on the same DDS
domain at once. Keep domain 0 reserved for the real robot.

## Before you start

Initialize submodules and Git LFS assets as described in [SETUP.md](SETUP.md).
Copy `docker/.env.example` to `docker/.env` if it does not exist. The full
model-server bringup also needs the documented weights and API key. Optional
mount roots (quote paths with spaces):

```bash
GOLEM_ASSETS_DIR=/absolute/path/to/CL_Assets
GOLEM_CACHE_DIR=/absolute/path/to/container_cache
ROS_DOMAIN_ID=7
```

Set those in `.env` or export them. Shell exports win.

## RoboCasa and ROS

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

## Mount and restart checks

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
missing sources, domain checks, build flags, environment precedence and restart
behavior. They do not establish GPU/GUI runtime, image build success,
controller stability, or performance.
