# GOLEM

**G**eneralized **O**pen **L**ayered **E**mbodied **M**odules is the software
stack for the Correll Lab's Unitree humanoids. It runs two robots, the H1-2 and
the G1, in simulation and on the real robot. The same Unitree DDS interface
(`rt/lowcmd` and `rt/lowstate`) is used in both, so code moves between them
unchanged.

The simulators (MuJoCo/RoboCasa and Isaac Lab) and the ROS 2 Humble workspace
run in separate Docker containers that share one DDS domain. The source,
including the `README.md` and `CLAUDE.md` developer guide, is at
[correlllab/GOLEM](https://github.com/correlllab/GOLEM).

## Where to start

| Robot | Start here | Platforms |
|---|---|---|
| Unitree H1-2 (the default) | [Setup](SETUP.md), then [Running the stack](RUNNING.md) | x86 Linux with an NVIDIA GPU; an arm64 CPU port for [macOS](MACOS.md) |
| Unitree G1 | [The G1](g1/index.md), then [First real session](g1/first-session.md) for the real robot | x86 Linux, RoboCasa simulator only |

Both robots use the same images and the same setup. The wrappers in
`docker/scripts/` run the H1-2 unless `--g1` or `GOLEM_ROBOT=g1` selects the G1.

## Pages

- **Install and run:** [Setup](SETUP.md), [Running the stack](RUNNING.md),
  [macOS](MACOS.md), [Navigation demo](NAVIGATION_DEMO.md).
- **Debugging and testing:** [DDS tuning](DDS_TUNING.md),
  [ROS MCP debug](ROS_MCP_DEBUG.md), [Docker testing](DOCKER_TESTING.md).
- **The G1:** [overview](g1/index.md), [packages](g1/packages.md),
  [safety layer](g1/safety.md), [walking](g1/walking.md),
  [navigation](g1/navigation.md) and the other pages in the G1 tab.
