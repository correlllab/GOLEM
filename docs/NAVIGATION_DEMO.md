# Navigation demo — autonomous SLAM + nav2 exploration

This walks the H1-2 through a RoboCasa kitchen **autonomously**: it builds a map
with SLAM, plans collision-free paths around the furniture with nav2, and drives
itself to unexplored frontiers.

Set up the host and build the images first ([`SETUP.md`](SETUP.md)).

## What it demonstrates

- **The shared navigation graph** — `h12_sim.launch.py` runs `golem_slam`'s
  `navigation.launch.py` for the H1-2 (`use_nav`, on by default; `golem_slam`'s
  README has the details). FAST-LIO turns the Livox lidar and IMU into odometry
  (`/Odometry`) and a deskewed cloud; `self_filter` drops the robot's own body
  from it; `level_odom` publishes the gravity-aligned `odom → base_footprint`;
  OctoMap builds the 3D map and its 2D projection `/projected_map`; the cloud is
  flattened to `/converted_scan` for `slam_toolbox`, which publishes
  `map → odom`. No ground truth is used: the sim's `GOLEM_SIM_ODOM` stays off.
- **nav2** — the MPPI controller and the Smac planner plan a collision-free path
  on `/projected_map` and the live scan, then drive `/cmd_vel`.
- **Locomotion** — the lower-body controller runs ALMI (the bringup's default
  `policy:=almi`) and follows `/cmd_vel` directly (stands at zero, walks
  otherwise).
- **Frontier exploration** — the `/skill/frontier_explore` action (the
  `FrontierExploreSkill` in `golem_skills`, started with `use_skills`, on by
  default) repeatedly sends the nearest unexplored frontier as a nav2 goal until
  the reachable space is mapped.

## Run it

Start the simulator and the bringup together; the stack wrapper builds the
workspace first and starts `h12_sim.launch.py` once the sim publishes:

```bash
# from the repo root; --headless when there is no display
GOLEM_CAMERAS=0 GOLEM_SPAWN_BACKOFF=1.5 docker/scripts/docker_stack.sh up robocasa --gui
docker/scripts/docker_stack.sh logs robocasa     # wait for the leg node's "elastic band released"
```

Then send the explorer:

```bash
docker exec -it golem_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
ros2 action send_goal /skill/frontier_explore custom_ros_messages/action/SkillFrontierExplore \
  "{min_frontier_cells: 6, blacklist_radius: 0.6, min_goal_distance: 0.7, goal_timeout: 120.0, timeout: {sec: 1800, nanosec: 0}}" \
  --feedback
```

With a display, the MuJoCo viewer shows the robot in the kitchen and RViz the
map, the costmap and the planned path. You'll see the map fill in, the costmap
mark the counters and appliances, a plan appear to each frontier, and the robot
walk to it. Unreachable frontiers are blacklisted after a timeout and the
explorer moves on; the action succeeds with
`map explored: X of N frontier goal(s) reached (A by Nav2, B seen on the way), F failed`
when no frontiers remain.

## The env knobs

| Variable | For the nav demo | Notes |
|---|---|---|
| `GOLEM_SIM_ODOM` | unset or `0` | Leave it off: FAST-LIO's odometry already parents `pelvis`, and the sim's ground truth would give it a second parent. |
| `GOLEM_SPAWN_BACKOFF` | `~1.5` | Metres to back the robot into open floor at spawn so nav2 has room. Read when the container is created (see gotchas). |
| `GOLEM_CAMERAS` | `0` | Drops the 3 RGBD renders, the heaviest per-step CPU cost; the nav demo doesn't need them. |

## A single goal (without exploration)

Set a goal in RViz with the **2D Nav Goal** tool, or send one to nav2's
`/navigate_to_pose` action, and let nav2 drive.

## How the pipeline fits together

```
MuJoCo (robocasa)                  ROS (golem_ros)
  Livox ─/livox/lidar, /livox/imu─► FAST-LIO ─/Odometry, deskewed cloud─► self_filter ─/cloud_body_filtered─┐
                                      level_odom (odom → base_footprint) ◄─────────────────────────────────┤
                                      OctoMap ─/projected_map─► nav2 costmaps ◄─────────────────────────────┤
                                      level_cloud + pointcloud_to_laserscan ─/converted_scan─► slam_toolbox ┘
                                      nav2 ─ plan ─► controller ─/cmd_vel─► lowerbody_controller_node
  rt/lowcmd ◄─────────────────────────────────────────── safety_node ◄──────────────┘  (ALMI policy)
```

## Gotchas specific to the nav demo

- **Change a sim knob with a restart of the pair.** `GOLEM_SPAWN_BACKOFF` and the
  other knobs are read when the container is created; `docker restart` keeps the
  old values. Restart both containers through the wrapper:
  ```bash
  GOLEM_CAMERAS=0 GOLEM_SPAWN_BACKOFF=1.5 docker/scripts/docker_stack.sh restart robocasa --gui
  ```
  The layout is randomized each launch, so a restart also re-rolls the kitchen —
  handy if the robot spawns cramped against a counter. `1.5`–`2.0` reliably
  clears the fixtures.
- **Restart the two containers coherently.** The sim owns `/clock`; if you restart
  it, its sim time resets to 0 and the still-running ROS side sees the clock jump
  backward (TF extrapolation errors, a frozen `0x0` SLAM map). The stack
  wrapper's `restart` replaces both.
- **Let the stance settle before commanding motion.** Send the explorer once the
  leg node has released the elastic band.
- **Some exploration goals time out.** Walking to a far frontier can exceed the
  goal timeout; the explorer blacklists it and picks another. That's expected,
  not a failure.
