# golem_safety

The safety layer of the Unitree H1-2 and G1. Its node, `safety_node`, is the only writer of
the robots' body command topics: `/lowcmd`, and on the G1 in `arm_sdk_mode` `/arm_sdk`. It
relays the controllers' commands from their `/safety/*` inputs, clips them to the config's
limits, checks the measured `/lowstate` against the estop limits, and latches an estop until
restart. The design is in
`docs/superpowers/specs/2026-10-27-golem-upper-ik-and-safety-design.md` (section
golem_safety); what differs from the old `h12_safety_layer` and `g1_safety_layer` is in the
change report beside it.

## Running it

The bringups start the node. On its own:

```bash
ros2 launch golem_safety safety.launch.py robot:=h12 config:=sim_safety_split sim:=true
ros2 run golem_safety safety_node --ros-args -p robot:=h12 -p config:=sim_safety_split -p sim:=true
```

| Parameter | Default | Meaning |
|---|---|---|
| `robot` | required | `h12` or `g1`: the profile in `robots/<robot>.yaml` |
| `config` | required | a config name in `config/<robot>/`, or a path (not one in another robot's `config/` directory) |
| `sim` | `false` | `true` for a simulator; refused on ROS domain 0, the real robots' command bus |
| `use_estop` | `false` | `true` to read the hardware estop on the config's `estop` topics |

The node refuses to start, with one `FATAL` line and a non-zero exit code, when a parameter
is missing or the profile or config does not load. It logs its events (many of them one
JSON object per line) through its ROS logger, and publishes `/safety/heartbeat`
(`std_msgs/Bool`, 10 Hz) true until the estop latches. On SIGINT, SIGTERM or SIGHUP it stops
publishing in `full_body_mode` and `split_mode`, and ramps the `arm_sdk` weight to 0 first in
`arm_sdk_mode`.

## Tests

In the G1-mode container (the H1-2 and G1 tests both run there):

```bash
docker/scripts/docker_run.sh ros --restart --g1 /home/code/h12_sim_scripts/launch_ros.sh bash -lc \
  'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 && cd /home/code/core_ws && colcon build --packages-select golem_safety && source install/setup.bash && colcon test --packages-select golem_safety --event-handlers console_direct+; colcon test-result --verbose'
```

The gtests cover `SafetyCore`, the config loader, the LowCmd builders and the chunk logger.
`test/test_node_ros.py` runs the installed node against a fake robot over ROS on
`ROS_DOMAIN_ID` 91 (`test/conftest.py`).
