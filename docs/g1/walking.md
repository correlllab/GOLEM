# Walking

Walking uses HOMIE's pretrained policy (`g1_locomotion`, with the policy in its
`policy/`). These docs cover it in the sim only.

Every G1 sim bringup walks: `g1_sim.launch.py` always starts HOMIE's walking
node (`locomotion_node`, with `g1_locomotion`'s `sim.yaml`), with the safety
layer on `sim_split` and the arm controller on its `sim_split` config. No
launch argument turns it on or off. The robot starts hanging from the sim's
elastic band, its leg motors off. Once `golem_safety` beats on
`/safety/heartbeat` and the arm controller commands, the walking node moves the
legs to the policy's default pose over 2 s and starts the policy at zero
velocity, still under the band. It releases the band once the arm controller
has finished its start-up moves (its first `/left_ee_pose`), or `band.max_wait`
after the policy started without one. From then on HOMIE keeps the robot
standing and walks it on `/cmd_vel`.

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

# terminal 2: wait for the walking node's "elastic band released" line
docker/scripts/docker_run.sh ros
ros2 launch golem_bringup g1_sim.launch.py

# terminal 3: once the band is released
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"
```

**What to expect:** in terminal 2 the walking node logs `walking: policy
started at zero velocity and 0.74 m height`, then `elastic band released
(/elastic_band/toggle: ...)`; terminal 1 logs `[fall-logger] elastic band
released at t=...`. The walking node ignores a `/cmd_vel` that arrives before
its policy starts. Once the band is off, the robot walks forward while the
command is published. It stops once the last command is older than
`commands.timeout` (`core_ws/src/g1/g1_locomotion/config/sim.yaml`), so keep
publishing at a steady rate. `/locomotion/height` sets the pelvis height (the
package README's ROS interface section).

`GOLEM_SPAWN_BACKOFF` backs the spawn off the counter into open floor, as the
walk test does (its `WALK_BACKOFF`). The test is the tested path; this by-hand
flow puts together the same parts.

## On the real robot

Not covered by these docs beyond the first session. The real bringup starts
the walking node only with `start_position_verified:=true`, set after checking
the robot's start pose; without it the leg motors stay off on the safety
layer's zero-gain stream while the arms run. With it the node moves the legs to
the default pose and holds them there until
`ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger` starts the
policy ([First real session](first-session.md)). The robot must be in Unitree's
debug mode and on a gantry, since the split estop turns every motor off;
`g1_real_robot.launch.py`'s docstring and [Safety layer](safety.md) have the
interlocks. A walking-node fault does not trip that estop: the safety layer
holds the legs on the node's last command ([Safety layer](safety.md), Quiet
inputs).

More: the [g1_locomotion README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_locomotion/README.md)
(behaviour, faults, real-time limits, configs) and the
[policy README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_locomotion/policy/README.md)
(HOMIE, its credits and its non-commercial licence).
