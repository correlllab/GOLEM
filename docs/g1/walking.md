# Walking

Walking uses HOMIE's pretrained policy (`g1_locomotion`, with the policy in its
`policy/`). These docs cover it in the sim only.

## Check it

```bash
tests/g1/test_sim_bringup.sh --walk
```

The walking node takes the legs, the robot keeps standing, then walks 5 s at
0.3 m/s. A passing run ends with `PASSED: <n>/<n> checks` and exits 0
(`tests/g1/README.md`).

## Drive it yourself

```bash
# terminal 1: the sim, the spawn backed 2 m into open floor
GOLEM_SPAWN_BACKOFF=2.0 docker/scripts/docker_run.sh robocasa --g1 --layout 9 --style 9 --seed 42
# wait for "[g1_mujoco] ROS bridges up"

# terminal 2
docker/scripts/docker_run.sh ros --g1
ros2 launch g1_bringup g1_sim_bringup.launch.py use_locomotion:=true

# terminal 3: once the walking node has taken the legs
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"
```

**What to expect:** the robot walks forward while the command is published.
It stops once the last command is older than `commands.timeout`
(`core_ws/src/g1/g1_locomotion/config/sim.yaml`), so keep publishing at a
steady rate. `/locomotion/height` sets the pelvis height (the package README's
ROS interface section).

`GOLEM_SPAWN_BACKOFF` backs the spawn off the counter into open floor, as the
walk test does (its `WALK_BACKOFF`). `use_locomotion:=true` switches the safety
layer to `sim_split` and the arm controller to its `sim_split` config. The
test is the tested path; this by-hand flow puts together the same parts.

## On the real robot

Not covered by these docs. Split mode needs Unitree's controller released by
the operator first, which nothing in this stack does. The interlocks the
bringup enforces (`safety_config:=split`, `start_position_verified:=true`, a
gantry because the split estop turns every motor off) are in
`g1_bringup.launch.py`'s docstring and in [Safety layer](safety.md). A
walking-node fault does not trip that estop: the safety layer holds the legs on
the node's last command ([Safety layer](safety.md), Quiet inputs).

More: the [g1_locomotion README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_locomotion/README.md)
(behaviour, faults, real-time limits, configs) and the
[policy README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_locomotion/policy/README.md)
(HOMIE, its credits and its non-commercial licence).
