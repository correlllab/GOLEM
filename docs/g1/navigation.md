# Navigation

Navigation is **sim only**. The real MID-360 lidar is not connected to the
bringup yet, and the real bringup refuses `use_nav:=true`.

The stack (`g1_navigation`): FAST_LIO gives odometry from the lidar, and
slam_toolbox draws a 2D map; Nav2 plans and sends `/cmd_vel` to the walking
node. FAST_LIO localises the robot; slam_toolbox only draws the map at its
poses (no scan matching). Both slam_toolbox and Nav2 work on `base_footprint`,
a level frame on the floor under the pelvis, so `map -> odom` is the identity.
With `odom_source:=sim_gt` the sim's ground-truth `odom -> pelvis` replaces
FAST_LIO; the sim publishes it only with `GOLEM_SIM_ODOM=1`, and the real robot
has no such thing.

## Check it

```bash
tests/g1/test_sim_nav.sh                         # FAST_LIO odometry
GOLEM_SIM_ODOM=1 tests/g1/test_sim_nav.sh        # the sim's ground-truth odometry
```

Nav2 walks the G1 to a goal 2-3 m away. The test checks that it arrives, stays
up, touches nothing, and that the map and the localisation match the sim. A
passing run ends with `PASSED: <n>/<n> checks`. `NAV_GOAL="x,y[,yaw_deg]"` (in
the sim's world frame) picks your own goal; the script's header lists the
other knobs.

## Send your own goals

```bash
# terminal 1: the sim, the spawn backed 2 m into open floor
GOLEM_SPAWN_BACKOFF=2.0 docker/scripts/docker_run.sh robocasa --g1 --layout 9 --style 9 --seed 42
# wait for "[g1_mujoco] ROS bridges up"

# terminal 2
docker/scripts/docker_run.sh ros --g1
ros2 launch golem_bringup g1_sim.launch.py use_locomotion:=true use_nav:=true

# terminal 3: once Nav2 is up
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
    "{pose: {header: {frame_id: map}, pose: {position: {x: 1.33, y: 2.08}, orientation: {w: 1.0}}}}"
```

Goals are in the `map` frame. With FAST_LIO it starts under the robot at the
spawn, x along its heading. For `odom_source:=sim_gt`, start the sim with
`GOLEM_SIM_ODOM=1` too (without it `odom -> pelvis` has no publisher; with it
and `fastlio` it has two, and TF breaks); `map` is then the sim's world frame.
The test is the tested path; this by-hand flow puts together the same parts.

More: the [g1_navigation README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1/g1_navigation/README.md)
(frames, the scan, slam_toolbox, Nav2, the sim runs, and what the real lidar
still needs).
