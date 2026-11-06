# Walking

Walking uses HOMIE's pretrained policy, run by `golem_lowerbody_rl`'s leg node
(`lowerbody_controller_node`, the H1-2's leg node too). These docs cover it in
the sim only.

Every G1 sim bringup walks: `g1_sim.launch.py` always starts the leg node with
policy `homie` and `golem_lowerbody_rl`'s `config/g1/sim.yaml` run config, with
the safety layer on `sim_split` and the arm controller on its `sim_split`
config. No launch argument turns it on or off. The robot starts hanging from
the sim's elastic band, its leg motors off. Once `golem_safety` beats on
`/safety/heartbeat` and `rt/lowstate` is upright, the leg node moves the legs
to the policy's default pose over 2 s and holds it. Once `rt/lowstate` has been
steady for 1 s, the arm controller commands and it has finished its start-up
moves (its first `/left_ee_pose`), the node starts the policy at zero velocity
and then releases the band; without a `/left_ee_pose`, each waits
`band.max_wait`. From then on HOMIE keeps the robot standing and walks it on
`/cmd_vel`.

## Check it

```bash
tests/g1/test_sim_bringup.sh --walk
```

The test checks the start-up under the band and the release, the robot keeps
standing, then it walks 5 s at 0.3 m/s. A passing run ends with
`PASSED: <n>/<n> checks` and exits 0 (`tests/g1/README.md`).

## Drive it yourself

```bash
# terminal 1: the sim, the spawn backed 2 m into open floor
GOLEM_SPAWN_BACKOFF=2.0 docker/scripts/docker_run.sh robocasa --g1 --layout 9 --style 9 --seed 42
# wait for "[g1_mujoco] ROS bridges up"

# terminal 2: wait for the leg node's "elastic band released" line
docker/scripts/docker_run.sh ros
ros2 launch golem_bringup g1_sim.launch.py

# terminal 3: once the band is released
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"
```

**What to expect:** in terminal 2 the leg node logs `policy started at zero
velocity and the default height 0.74`, then `elastic band released
(/elastic_band/toggle: ...): the policy restarts from an empty history`;
terminal 1 logs `[fall-logger] elastic band released at t=...`. The leg node
drops a `/cmd_vel` that arrives before its policy starts. Once the band is
off, the robot walks forward while the command is published. It stops once the
last command is older than `commands.timeout`
(`core_ws/src/golem_lowerbody_rl/config/g1/sim.yaml`), so keep publishing at a
steady rate. `/lowerbody/height` (`std_msgs/Float32`, m for HOMIE) sets the pelvis
height (the package README's interfaces).

`GOLEM_SPAWN_BACKOFF` backs the spawn off the counter into open floor, as the
walk test does (its `WALK_BACKOFF`). The test is the tested path; this by-hand
flow puts together the same parts.

## On the real robot

Not covered by these docs beyond the first session. The real bringup starts
the leg node only with `start_position_verified:=true`, set after checking
the robot's start pose; without it the leg motors stay off on the safety
layer's zero-gain stream while the arms run. With it the node moves the legs to
the default pose and holds them there until
`ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger` starts the
policy, accepted once `rt/lowstate` has been steady for 1 s and the arm
controller commands ([First real session](first-session.md)). The robot must be in Unitree's
debug mode and on a gantry, since the split estop turns every motor off;
`g1_real_robot.launch.py`'s docstring and [Safety layer](safety.md) have the
interlocks. A leg-node fault does not trip that estop: the safety layer
holds the legs on the node's last command ([Safety layer](safety.md), Quiet
inputs).

More: the [golem_lowerbody_rl README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/golem_lowerbody_rl/README.md)
(phases, faults, timing, run configs) and the
[HOMIE README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/golem_lowerbody_rl/policies/homie/README.md)
(HOMIE, its credits and its non-commercial licence).
