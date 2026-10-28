# g1_navigation

Navigation for the Unitree G1: odometry from FAST_LIO on the MID-360 (or the
sim's ground truth), which also localises the robot; a 2D map that slam_toolbox
draws at the odometry's poses; and Nav2, which drives the walking node through
`/cmd_vel`. It is a launch and config package with three small helper nodes.
The sim bringup starts it with `use_nav:=true`; the real bringup refuses that
until the real lidar is connected (see [The real lidar](#the-real-lidar)). It
has not run on the robot.
`tests/g1/test_sim_nav.sh` walks the G1 to a goal in the RoboCasa kitchen
with it, closed loop ([In the G1 sim](#in-the-g1-sim)). With
`frontier_explore:=true` it also serves `/skill/frontier_explore`, which walks
the robot to the frontiers of the map until none is left
([Frontier exploration](#frontier-exploration));
`tests/g1/test_sim_explore.sh` explores the kitchen with it.

## Run

```bash
# with the sim's bringup (golem_bringup): the core, the walking node and this
ros2 launch golem_bringup g1_sim.launch.py use_locomotion:=true use_nav:=true
ros2 launch golem_bringup g1_sim.launch.py use_locomotion:=true use_nav:=true odom_source:=sim_gt
ros2 launch golem_bringup g1_sim.launch.py use_locomotion:=true use_nav:=true frontier_explore:=true

# alone, beside a running core
ros2 launch g1_navigation g1_navigation.launch.py                          # real robot, FAST_LIO
ros2 launch g1_navigation g1_navigation.launch.py sim:=true use_sim_time:=true
ros2 launch g1_navigation g1_navigation.launch.py odom_source:=sim_gt use_sim_time:=true
ros2 launch g1_navigation g1_navigation.launch.py --show-args
```

It needs `robot_state_publisher` on the G1 URDF, which the core bringup
starts, and the MID-360 on `/livox/lidar` (`livox_ros_driver2/CustomMsg`) and
`/livox/imu` (`sensor_msgs/Imu`), both with frame `livox_frame`, the sensor's
own frame. Send goals to `/navigate_to_pose` (RViz's "2D Goal Pose" does).

| Argument | Default | |
|---|---|---|
| `odom_source` | `fastlio` | `fastlio`, or `sim_gt`: the sim's ground-truth `odom -> pelvis`, sim only |
| `sim` | `false` | picks `g1_fastlio_sim.yaml` instead of `g1_fastlio.yaml` |
| `use_sim_time` | `false` | `true` with the sim |
| `fastlio_config` | (by `sim`) | a FAST_LIO config in `config/` |
| `pelvis_height` | `0.744` | pelvis above the floor when FAST_LIO starts (HOMIE standing) |
| `nav2_params`, `slam_params` | `config/g1_nav2.yaml`, `config/g1_slam_toolbox.yaml` | |
| `frontier_explore` | `false` | `true` adds the frontier explorer ([below](#frontier-exploration)); `g1_sim.launch.py` passes it through |

- `fastlio`: start it with the robot standing still. FAST_LIO initialises on its
  first scans, and odom assumes the torso level and the pelvis at
  `pelvis_height` then. The sim must not publish its ground-truth
  `odom -> pelvis` (leave `GOLEM_SIM_ODOM` unset); two parents for `pelvis`
  break TF.
- `sim_gt`: the sim publishes `/odom` and `odom -> pelvis` only with
  `GOLEM_SIM_ODOM=1`. Ground truth does not exist on the robot; use it to test
  navigation apart from odometry.

## What runs

| Node | What it does |
|---|---|
| `fast_lio` (fastlio only) | lidar-inertial odometry: `camera_init -> body` TF, `/Odometry`, `/cloud_registered(_body)` |
| `odom_camera_init_tf` (fastlio only) | static `odom -> camera_init`: hangs FAST_LIO's frames off a level odom (below) |
| `body_to_pelvis` (fastlio only) | `body -> pelvis` through the measured waist (below) |
| `livox_frame_tf` | `mid360_link -> livox_frame`, identity |
| `livox_to_pointcloud` | `/livox/lidar` to `/lidar_points` (`PointCloud2`, same stamp), deskewed for the sensor's rotation with `/livox/imu`'s gyro, in `pelvis_level` through TF |
| `level_frame` | `pelvis -> pelvis_level`: the pelvis with its roll and pitch removed; `pelvis_level -> base_footprint`: the same on the floor, the base frame of slam_toolbox and Nav2 |
| `pointcloud_to_laserscan` | `/lidar_points` to `/scan`, in the cloud's frame (`pelvis_level`); no TF (below) |
| `slam_toolbox` | `/map` from `/scan`, each scan at the odometry's pose (no scan matching; [below](#slam_toolbox)), and `map -> odom` |
| Nav2: `planner_server`, `controller_server`, `behavior_server`, `bt_navigator`, `velocity_smoother`, `lifecycle_manager_navigation` | plans on the map, follows the path, publishes `/cmd_vel` |
| `frontier_explore` (`frontier_explore:=true` only) | serves `/skill/frontier_explore`: Nav2 goals to the frontiers of `/map` ([below](#frontier-exploration)) |

## Frames

```
map -(slam_toolbox)-> odom -(static)-> camera_init -(FAST_LIO)-> body -(body_to_pelvis)-> pelvis
    -(robot_state_publisher)-> ... torso_link -> mid360_link -(static)-> livox_frame
                                               pelvis -(level_frame)-> pelvis_level -(level_frame)-> base_footprint
```

`map -> odom` is the identity ([slam_toolbox](#slam_toolbox)).

In `sim_gt` mode the sim publishes `odom -> pelvis` instead of the three
FAST_LIO links.

The MID-360 sits upside down on the torso (URDF `mid360_joint`: rpy pi, 0.0511,
0 on `torso_link`, 0.472 m above the pelvis with the waist at 0). FAST_LIO
tracks the sensor's IMU (`body`) relative to where the IMU was when it
started (`camera_init`), so both frames are upside down. The H1 used
`camera_init` as its odom frame; on the G1 that would put the floor above the
robot and mirror the map. Instead, two transforms hang FAST_LIO off the robot:

- `odom -> camera_init`, static, computed by `g1_navigation/frames.py` from
  the URDF at launch: the IMU's start pose with the torso level (as with the
  pelvis level and the waist at 0), the pelvis at `pelvis_height` and odom's
  x axis along the pelvis's. The floor is z = 0 in odom, in both odom sources.
- `body -> pelvis`, published by `body_to_pelvis` from TF: FAST_LIO's
  lidar-to-IMU extrinsic composed with `livox_frame -> pelvis`, which
  `robot_state_publisher` gives at the measured waist angles. The extrinsic
  is read from the FAST_LIO config the launch file uses, so the two cannot
  disagree. The IMU FAST_LIO tracks and the lidar the scan is cut from then
  stay one rigid body (and so do the D435 and anything else on the torso),
  whatever the waist does. A fixed `body -> pelvis` with the waist at 0
  would count the waist twice: once in FAST_LIO's pose of the IMU, once in
  `robot_state_publisher`'s torso. With HOMIE's 1.5 deg waist that tilted
  the scan band by 1.1 to 3 deg, depending on the start pose and the heading,
  and put the floor into the scan from 3 to 8 m out. Now, apart from
  FAST_LIO's own error, the only tilt left is the start pose's (0.3 deg in
  HOMIE's stance).

Assumptions, and what breaks them:

- The robot stands still with a level torso when FAST_LIO starts. In HOMIE's
  MuJoCo stance the pelvis leans back 1.8 deg and the waist forward 1.5 deg, so
  the IMU is within 0.3 deg of the assumed pose; odom is tilted by the
  difference. Unitree's own controller stands differently; not measured.
- The IMU is where FAST_LIO's extrinsic says: 0.011, 0.023, -0.044 m from the
  lidar origin in the sensor's axes. The sim has to put it there too.
- `robot_state_publisher` gets the waist joints on `/joint_states` (the core
  bringup's body `joint_state_publisher` at 100 Hz, or the sim), on the same
  clock as the lidar. Without them there is no `body -> pelvis`, and no
  `torso_link` either.

## The scan

`livox_to_pointcloud` deskews each frame and places it in `pelvis_level` at the
frame's stamp (TF lookups at the stamp; a frame waits for its transform, in
order, up to 0.5 s of frames). `pointcloud_to_laserscan` then cuts `/scan` from
it, in the cloud's own frame, with these values in the launch file:

- height band -0.59 to 0.45 m relative to the pelvis: from 0.15 m above the
  floor with the pelvis at 0.744 m to just under the lidar. It holds with the
  deskew (below); in the G1 sim's kitchen no `/scan` return within 4 m lay
  over 0.25 m from the kitchen's geometry in the last 8 runs;
- `range_min` 0.50 m, horizontal from the pelvis: drops the robot's own arms.
  In the arm controller's `home` pose (all arm joints at 0) the Dex3 finger
  collision meshes reach 0.445 m from the pelvis horizontally, 0.417 m in
  front of it (FK of the URDF). Navigate with the arms in `home`; other poses
  may put the hands in the scan. Neither `fake_g1.py` nor the G1 sim's lidar
  returns anything from the robot's own body, so no test checks this; the real
  robot will;
- `range_max` 8 m, 0.5 deg bins, no return reported as inf.

The deskew (`sim:=true` only for now, TODO(lidar): on the robot, MID360_config.json's
roll-180 extrinsic turns the points but not the IMU, so the gyro would have the wrong
sign for them). A MID-360 frame is measured over 0.1 s, and the scan is cut from
all of it with the sensor's pose at the frame's stamp. The walking G1's torso turns fast: in the G1 sim's kitchen test,
while navigating, the MID-360's tilt changed by a median 4.6 deg and up to 8.9
deg within one frame (its heading up to 4.7 deg; standing under the walking
node, at most 2.4 deg). Cut with the stamp's pose, the band's bottom, 0.15 m
above the floor, then reached the floor from 0.9 m out:

- re-cut offline with the sim's true sensor poses, 82 frames of navigating
  kept 3878 floor points (345 within 1.5 m, 2381 at 1.5-2.5 m) in 38 frames,
  and cut 1.5% of the returns of obstacles 0.3-1.1 m high;
- in the closed loop, 1.2-1.7% of the `/scan` returns within 4 m lay more
  than 0.25 m from anything in the band, in 28-30 of 159-174 scans, and the
  floor showed up as single occupied cells in the SLAM map;
- raising the band does not fix it: at -0.45 (0.29 m above the floor) the
  floor was still in 9 of the 82 frames, and 2.9% of the obstacle returns were
  cut.

So `livox_to_pointcloud` turns each point back to the sensor's orientation at
the stamp, by the rotation the sensor's own gyro (`/livox/imu`, the same axes)
measured between the stamp and the point's time. That rotation alone was
enough offline (no floor point kept, every obstacle return kept); the
translation within a frame (at most 0.05 m at the 0.5 m/s cap) is not
corrected. In the closed loop, no `/scan` return within 4 m lay more than 0.25
m from the kitchen's geometry in any of 32 runs with the deskew (99th
percentile 0.05 m). A frame without gyro samples around it is passed through
as it is, with a warning (points all stamped at the frame's stamp, as
`fake_g1.py`'s, need none).

Why the converter, not `pointcloud_to_laserscan`, does the TF: with its own
TF message filter (`target_frame: pelvis_level`, the cloud in `livox_frame`),
`pointcloud_to_laserscan` stopped for good in 7 of 27 runs of the G1 sim's nav
test, about 1.5 s after the walking node started walking: 6 of 13 on sim time,
and 1 of 14 once it ran on the wall clock. `/lidar_points` kept arriving and its
subscription stayed in the graph, but its executor thread made no context
switch for a second (on the wall clock its TF thread neither), so `/scan`
stopped and slam_toolbox and both costmaps starved. The mechanism is not found
(there is no debugger in the image). Now it gets the cloud in `pelvis_level`,
with `target_frame` empty, so it has no TF and no message filter, and it runs on
the wall clock because it needs no clock. The converter's TF lookups are plain
`lookup_transform` calls at the stamp, with no waiting inside tf2. Since then,
nothing in the scan path stopped in 24 runs. Still users of tf2_ros's
`MessageFilter`: slam_toolbox and both costmaps (on `/scan`). In the earlier
runs, slam_toolbox once stopped processing scans and the local costmap twice
never took one, so watch for it; the sim's nav test detects a stopped `/scan`
or `map -> odom` and writes the scan path's threads to `stall.txt`.

Why the raw points (`livox_to_pointcloud`) and not FAST_LIO's deskewed
`/cloud_registered_body`:

- the scan path is the same in both odom sources: the converter's deskew,
  `livox_frame -> mid360_link -> ... -> pelvis` through
  `robot_state_publisher`, `pelvis_level` and `pointcloud_to_laserscan`. The
  `sim_gt` test covers the obstacle path; `test_fastlio.py` covers what
  fastlio mode adds (the odometry, `body_to_pelvis` and the waist);
- `/cloud_registered_body` is FAST_LIO's input after its odometry filters
  (the `blind` radius, `point_filter_num` decimation, line and tag filters);
  tuning those for odometry would change what the costmaps see.

Why `pelvis_level` and not `pelvis`: the band is cut in the cloud's frame, and
the pelvis tilts. Under HOMIE in MuJoCo it leans back 1.8 deg standing, and
about 5 deg back with up to 6 deg of roll walking and turning. With a band in
the pelvis frame, the floor behind the walking robot enters the band from about
1.5 m out.
In `test_nav2_smoke.py`'s room with that lean, the pelvis frame put a phantom
wall 1.5 m behind the robot into the local costmap, and 761 of the SLAM map's
1998 occupied cells were more than 0.1 m from anything in the room. With
`pelvis_level` none were (450 cells, all within 6 cm).
`pelvis_level` needs `odom` to be level (see the assumptions above).

`livox_to_pointcloud` costs about 45 ms for a 20k-point scan, the real
MID-360's 10 Hz frame, so about 45% of a core (measured on the dev laptop
under other load, before the deskew). rclpy's deserialisation of the CustomMsg
is 41 ms of it; parsing the raw message with numpy would remove most of that.
With the deskew, the 200 Hz gyro and its own TF listener it used 0.47-0.51
cores on the sim's 5000-point frames (0.34-0.37 before the TF listener). If
the scan stops or lags, the costmaps stop being current (below) and Nav2 waits.

## slam_toolbox

The odometry localises the robot, and slam_toolbox only draws the map
(`config/g1_slam_toolbox.yaml`: `use_scan_matching: false` and
`do_loop_closing: false`, in both odom sources). Each time the pelvis has moved
about 0.1 m, at most every 0.2 s, it adds a `/scan` to `/map` where the
odometry (FAST_LIO, or the sim's ground truth in `sim_gt`) has the pelvis at
the scan's stamp. Nav2 plans on that map (the global costmap's static layer),
and the frontier explorer ([Frontier exploration](#frontier-exploration))
looks for its unknown cells. Spatial memory keeps its poses in odom and does
not need it.

With scan matching on, as before, slam_toolbox localised the robot itself by
matching each scan against the ones before it. In the G1 sim's kitchen it
drifted in the corridor behind the counter peninsula. The nav test with
`NAV_GOAL=0.62,-5.08,180` (3.9 m from the spawn straight through the counter,
7.6-7.7 m walked around it), fastlio, three runs each:

| | scan matching (before) | odometry only (now) |
|---|---|---|
| test | 0 of 3 passed (map phantoms; the localisation check, added since, fails them too) | 3 of 3 passed |
| `map -> pelvis` against the sim: max (at the end) | 15.4-15.7 cm (13.7-15.0 cm), 2.3-2.6 deg | 0.5-0.7 cm (0.2-0.4 cm), 0.25-0.47 deg |
| FAST_LIO's `odom -> pelvis` against the sim: max | 0.5-0.6 cm, 0.19-0.35 deg | 0.5-0.6 cm, 0.12-0.22 deg |
| final map: occupied cells over 0.15 m from the kitchen | 1-8, up to 0.21 m, along the counter's edge | none (99th percentile 0.07-0.10 m) |
| goal succeeded after, true pelvis from the goal | 21.3-22.0 s, 0.15-0.21 m | 21.2-21.5 s, 0.18-0.19 m |
| host load, real-time factor | 4.0-24.6, 0.93-0.97 | 4.9-16.8, 0.95-1.00 |

In all three runs with scan matching, and in the demo run where it showed
first, its error grew across the corridor (in y; the corridor runs along x),
mostly while the robot walked the corridor's length, with up to 2.6 deg of yaw
error. Why the matcher drifted there was not looked into. On the default goal
(2.5 m in open floor), `map -> pelvis` was within 4.7 cm with scan matching
(one run here; 3.3-5.2 cm in the eight runs that introduced the test) and
within 0.4-0.5 cm without (two runs). The nav test passed those runs with scan
matching, so it now checks `map -> pelvis` against the truth while navigating,
within 2.5 cm ([In the G1 sim](#in-the-g1-sim)).

- Loop closing is off as well: Karto closes loops on the scan matcher's pose
  graph only, so it would do nothing, and a closure would move the map frame
  against odom, in which the robot's poses are kept. FAST_LIO's own drift is
  not corrected: over these 7.6-7.7 m walks at most 0.6 cm in x, y and 4.1 cm
  in height; over long walks, and on the robot, not measured.
- Given up: relocalising in a map saved earlier (slam_toolbox's localization
  mode matches scans). FAST_LIO starts odom at the robot's start pose on every
  launch, so odom and the map hold for one session.
- `base_frame` is `base_footprint`: `pelvis_level` dropped to odom's floor,
  which `level_frame` publishes at each `odom -> pelvis` stamp. slam_toolbox
  puts its base frame at z = 0 and level in map at each scan it adds, and
  `base_footprint` already is that in odom, so `map -> odom` is the identity:
  in the 11 runs measured below it stayed within 0.1 mm in z, 0.002 deg in
  tilt and 0.0 mm in x and y, and in all 12 runs at the current settings (the
  8 of [In the G1 sim](#in-the-g1-sim) and the 4 of
  [Exploring in the G1 sim](#exploring-in-the-g1-sim)) `map -> pelvis`
  equalled `odom -> pelvis` at the same stamps. A pose in odom (spatial
  memory's among them) is the same pose in map.
- Before, `base_frame` was `pelvis`. Then `map -> odom` carried the pelvis's
  height and tilt at the last scan added (in the 5 runs measured below: z
  -1.24 to -0.03 m, tilt 1.3-8.7 deg, x, y 1.7-10.8 cm). Only points at the
  pelvis's height mapped as in odom (the floor under the pelvis 10.1-10.9 cm
  off at most, median 3.9-7.5 cm; now 0.0 cm), and the scan lay tilted about
  z = 0 in map, where the global costmap's obstacle layer
  (`global_frame: map`, `min_obstacle_height: 0.0`) dropped every return
  below z = 0. Measured in the costmap: a second costmap node running beside
  the nav test, with the global costmap's obstacle layer alone (the same
  parameters, from `g1_nav2.yaml`), against each `/scan` while navigating,
  its returns within the layer's 2.5 m placed in map through TF at the scan's
  stamp, looked up in the layer's first update at least 0.5 s later
  (`base_frame: pelvis` in 5 nav runs: default goal fastlio and `sim_gt`,
  corridor 2 fastlio and `sim_gt`; `base_footprint` in 8 nav and 3
  exploration runs at the current settings):

  | | `pelvis` (before) | `base_footprint` (now) |
  |---|---|---|
  | `/scan` returns within 2.5 m below z = 0 in map | 42.9-59.2% (per scan 25-100%) | none |
  | their cells LETHAL in the obstacle layer | 44.6-64.0% | 74.2-80.6% |
  | the same, or a neighbouring cell | 63.1-78.6% | 95.7-98.2% |
  | the returns the layer kept, and dropped, LETHAL | 71.0-81.4%, 24.0-47.1% (marked from other scans) | all kept |
  | their cells LETHAL in `/global_costmap/costmap` (with the map and the inflation) | 84.4-89.4% | 89.8-94.5% |

  A return's own cell is not always marked: the layer takes only the newest
  scan at each 5 Hz update, and the rays of later scans clear cells they pass
  through, so the mark can sit in the next cell. The local costmap is in odom,
  where the scan lies at the pelvis's height, so it kept all of it before
  too, and the map, drawn in 2D, was not affected. A negative
  `min_obstacle_height` on the global obstacle layer would have kept the
  returns too, with map still tilted.
- The nav test converts its goal, a floor-level pose, through `map -> odom`,
  now the identity: the goal as sent lands 0.6-1.1 cm from the picked one in
  fastlio, where the test puts odom at the spawn's pelvis pose and FAST_LIO's
  start pose differs from it by that, and 0.00 cm in `sim_gt` (2.2-3.7 cm
  before, through the tilted `map -> odom`).

## Nav2

`config/g1_nav2.yaml` (Humble 1.1). The costmap layers' parameters sit directly
under each layer's name. A `ros__parameters` level there hides them and Nav2
falls back to its defaults without a warning.

- Nav2's `robot_base_frame` is `base_footprint` too (the navigator, the
  behaviors and both costmaps): level, on the floor, at the pelvis's x, y and
  heading, so the robot's tilt does not enter Nav2. RPP transforms the path
  into the robot's base frame and decides there, among other things, when the
  goal is close enough to turn in place to its heading; the goal checker
  takes x, y in odom. In the tilted pelvis frame a floor point moves by about
  its depth below the pelvis times sin(tilt) (2.3 cm at the standing lean of
  1.8 deg), and once map was on the floor (base_frame above) the two
  disagreed at the goal: with `robot_base_frame: pelvis`, `test_nav2_smoke.py`
  stopped the robot 0.24-0.25 m from its goal (0.41 m once) and gave no
  result within 150 s in 3 of 8 runs, and one more only after 159 s; with
  `base_footprint`, 5 of 5 succeeded after 21.7-23.8 s, 0.17-0.18 m from it.
- Smac Planner 2D (A* on the costmap) plans on the SLAM map plus the live
  scan. It replaced NavFn, which in the G1 sim's kitchen test failed to extract
  a path from its potential ("Failed to create a plan from potential when a
  legal potential was found") in 4 of 27 runs, 14 times in a row in one, which
  aborted the goal; Smac 2D has no such step. Regulated Pure Pursuit follows
  the path at 0.4 m/s, turns in place towards the path, and never walks
  backwards on a path.
- Speed caps, accepted by the user: vx -0.3 to 0.5 m/s, vy 0, |wz| <= 0.6
  rad/s, as in `g1_locomotion`'s real-robot config. The velocity
  smoother enforces them (accel 0.4 m/s^2 and 0.8 rad/s^2, decel 0.5 and 1.0,
  inside the walking node's own rate limits). The launch file routes the
  controller's and the recovery behaviors' commands through it (Humble's
  `navigation_launch.py` lets the behaviors publish `/cmd_vel` directly), so
  it is the only publisher of `/cmd_vel`. After 0.3 s without input it ramps
  to zero at its deceleration limits and stops publishing. The walking node
  takes 0.5 s without a command as zero velocity, so a dead controller or
  behavior never leaves the robot on a held command for longer than that
  (`test_velocity_smoother.py`).
- Footprint: x -0.20 to 0.44 m, y -0.24 to 0.24 m, the walking robot with the
  arms in `home`. Replayed through the kitchen model's kinematics, the bounding
  boxes of the robot's collision geoms in the G1 sim's nav test reached, from
  the pelvis along its heading, x -0.190 m (the swing feet; the torso -0.155)
  to +0.435 m (the Dex3 fingers) and y +-0.238 m (the elbows) while walking,
  and x -0.124 to +0.426 m, y +-0.221 m standing. The earlier estimate, x
  -0.15 to 0.43 m, y +-0.23 m, left the walking robot's feet, fingers and
  elbows out.
- The footprint is centred on the pelvis but reaches further forward than
  back, so its inscribed radius is 0.20 m. The planner plans for that radius and
  can plan closer to an obstacle than the footprint allows; RPP's collision
  check can then stop the robot, and a costmap-clearing recovery replans. The
  global costmap updates at 5 Hz so that the replan does not wait long. A
  footprint-aware planner (Smac Hybrid or Lattice) would avoid it; not tried.
- RPP turns in place at most `max_angular_accel` x 0.05 s (one control step)
  faster than the yaw rate on `/odom`. FAST_LIO publishes no `/odom`, and the
  H1 sim's had no twist, so that rate reads 0: with the default 3.2, turns
  would stop at 0.16 rad/s. It is 12, and the velocity smoother limits the
  acceleration instead.
- The costmaps ignore scan bins with no return (`inf_is_valid: false`): an
  obstacle inside the 0.50 m `range_min` shows as no return and stays marked.
- Each obstacle layer expects a scan every 0.5 s (`expected_update_rate`).
  Without one the costmap is not current, and the controller and planner wait
  rather than drive on stale obstacles.
- slam_toolbox stamps `map -> odom` with the last scan's stamp plus
  `transform_timeout`, 0.3 s. At its 0.2 default, the sim's ground-truth
  `odom -> pelvis` (`sim_gt`) was newer than `map -> odom` by up to 68 ms, and
  RPP, which transforms at the robot pose's stamp, waited or failed: 3-42 missed
  20 Hz control cycles in an 8-10 s goal, and in one run 4 aborted FollowPaths
  before the goal succeeded. With 0.3, none in 8 runs. FAST_LIO's own stamps lag
  26-120 ms, so fastlio mode was fine either way.

## In the G1 sim

`tests/g1/test_sim_nav.sh` (`GOLEM_SIM_ODOM=1` for `sim_gt`) runs the G1
RoboCasa sim with its sensors and `g1_sim.launch.py use_locomotion:=true
use_nav:=true`, and sends one goal: by default 2.48 m from the spawn in open
floor, picked from the sim's own geometry (a 57 deg turn in place, then a walk),
or `NAV_GOAL`'s (the corridor above: a turn, then 7.6-7.7 m around the
counter peninsula). After the goal it stops the sim, then the bringup, so the
robot stands to the end of the recording (the other way round, the safety
layer's exit estop turns every motor off while the sim runs, and the robot
falls).
Eight runs at the current settings (default goal: two fastlio, one `sim_gt`;
corridor: four fastlio, one `sim_gt`; host load 1.0-16.7, real-time factor
0.88-1.00 per run and 0.89-1.00 from the goal to its result; for a second it
dipped as low as 0.52-0.91), all passing:

- the goal succeeded after 7.6-7.7 s (default) and 20.9-21.1 s (corridor) of
  sim time, with no recovery, no missed control cycle and no aborted path;
- the true pelvis was 0.23-0.26 m (default) and 0.20-0.25 m (corridor) from the
  goal at the result, and its heading 9-12 deg off: the goal checker's 0.25 m
  and 0.25 rad tolerances end the goal as the robot enters them (Nav2's own
  estimate 0.22-0.27 m). The 0.3-3.1 cm between the two is not the
  localisation's error (`map -> pelvis` was within 0.7 cm of the truth
  throughout) but the test's bookkeeping. The goal as sent lands 0.6-1.1 cm
  from the picked one in fastlio and 0.00 cm in `sim_gt`
  ([slam_toolbox](#slam_toolbox)), and the true error is taken at the sim's
  first sample after the result, 18-120 ms after the stamp of Nav2's view,
  with the robot still walking;
- FAST_LIO's `odom -> pelvis` against the sim's truth: at most 0.4-0.7 cm off
  in x, y and 0.07-0.33 deg in yaw, over 2.6-2.7 m (default) and 7.4-7.6 m
  (corridor) walked; in height -1.1 to +0.6 cm (default) and -4.1 to +0.6 cm
  (corridor); in tilt at most 0.8-1.7 deg, mostly the TF's timing against the
  walking pelvis's sway (0.6 deg in `sim_gt`, where `odom -> pelvis` is the
  truth);
- `map -> pelvis` against the truth while navigating, which the test checks
  against 2.5 cm: at most 0.38-0.71 cm (fastlio) and 0.02-0.06 cm (`sim_gt`),
  and 0.08-0.34 deg in yaw over the whole drive: `map -> odom` is the
  identity, so this is the odometry's error. Before, with slam_toolbox's base
  frame on the tilted pelvis, it reached 1.5-1.6 cm in `sim_gt` on the
  corridor goal, 6 m from odom's origin (the world's), where TF interpolated
  the tilted `map -> odom` between slam_toolbox's scans. With scan matching
  it was 3.3-5.5 cm on the default goal in the 36 runs since the scan was
  deskewed (2.5-7.3 cm in 11 earlier runs with the floor still in the scan)
  and 13.1-18.1 cm on the corridor (6 runs). The limit sits halfway between
  1.6 and 3.3 cm ([slam_toolbox](#slam_toolbox));
- over the whole recording the pelvis stayed at 0.712-0.780 m and tilted at
  most 9.4 deg, no robot geom touched the kitchen and the sim logged no fall;
  `/cmd_vel` stayed at vx 0..0.4 m/s, vy 0, wz -0.44..0.44 rad/s;
- `/scan` never paused more than 0.1 s; every occupied cell of the final map
  was within 0.10 m of the kitchen's geometry in the scan's band, and no
  `/scan` return within 4 m was over 0.25 m from it;
- `livox_to_pointcloud` used 0.46-0.54 cores (5000-point frames, the 200 Hz
  gyro and its TF listener).

## Frontier exploration

`frontier_explore` (`g1_navigation/frontier_explore.py`, started with
`frontier_explore:=true`) serves `/skill/frontier_explore` with
`custom_ros_messages/action/SkillFrontierExplore`, the action the H1-2's
skills node (`h12_skills`) serves, so callers do not change. It needs
`/map`, the global costmap (`/global_costmap/costmap`), `map -> pelvis` in TF
and Nav2's `/navigate_to_pose`, all of which the rest of this stack provides.
It has not run on the robot. `tests/g1/test_sim_explore.sh` is the tested
path ([below](#exploring-in-the-g1-sim)); the commands here were not run by hand.

```bash
ros2 launch golem_bringup g1_sim.launch.py use_locomotion:=true use_nav:=true frontier_explore:=true
ros2 action send_goal /skill/frontier_explore custom_ros_messages/action/SkillFrontierExplore \
    "{timeout: {sec: 120}}" --feedback
```

| Goal field | 0 means | |
|---|---|---|
| `min_frontier_cells` | 6 | smaller clusters are left out |
| `blacklist_radius` | 0.6 m | no goal this close to one that failed |
| `min_goal_distance` | 0.7 m | no goal this close to the robot |
| `goal_timeout` | 120 s | a Nav2 goal not reached by then is canceled and blacklisted |
| `timeout` | 300 s (`default_timeout`) | the whole exploration, on the node's clock (sim time in the sim) |

The result is `success` when no frontier is left ("map explored: R of S
frontier goal(s) reached (N by Nav2, O seen on the way), F failed"), aborted
with "skill timeout" at the timeout, or canceled. `goals_reached` (R) counts
the goals Nav2 reached and the goals seen on the way: either way the frontier
is resolved. The 360-degree MID-360 sees most frontiers before the robot gets
there, so in the G1 sim Nav2 reached none ([below](#exploring-in-the-g1-sim)),
and counting only Nav2's successes, as the H1-2 skill did, gave 0 for every
finished exploration. Failed and blocked goals, which are blacklisted, do not
count. The feedback gives the phase (`exploring`, `navigating`, `done`) and
the candidates left. It runs one exploration at a time and refuses goals with
negative fields. Parameters: `footprint` (the launch file passes
`nav2_params`' global costmap footprint), `default_timeout`, `map_frame`,
`base_frame`, `map_topic`, `costmap_topic`.

The policy is the H1-2 skill's: frontier cells are the free cells of `/map`
next to unknown ones (4-adjacent, off the map counting as unknown) and not
next to an occupied one; 8-connected clusters of at least
`min_frontier_cells`; the nearest (straight line) that is farther than
`min_goal_distance` and not blacklisted is sent to Nav2; a goal that fails or
times out is blacklisted; done when no candidate is left. No information-gain
scoring. On the G1, four things are added, each for something the G1 sim's
kitchen showed:

- Only frontiers the robot can walk to count: walkable cells are free in
  `/map` and below Nav2's inscribed cost in the global costmap (the test Smac
  2D plans with), and a frontier counts if it is walkable and 8-connected
  through walkable cells to the robot (the nearest walkable cell within 0.5 m
  if the robot's own is not). The scan, cut 0.15 to 1.2 m above the floor,
  sees over and between what stands on the kitchen's counters. In each of
  the 11 runs below, the last `/map` before the explorer was done held 6 to
  17 clusters of 6 or more frontier cells (the largest 15 to 53 cells), and
  by the explorer's own count the robot could walk to none of them (in one
  run to one, where the footprint fit at none of its cells). The H1-2 policy
  would send Nav2 to each of them in turn.
- The goal is the cluster's cell nearest its centroid where the footprint,
  facing from the robot to the cell, covers no lethal costmap cell (a
  cluster without such a cell is skipped). The H1-2 skill sent the centroid,
  which for a curved frontier can lie in a wall or in unknown space.
- A goal on its way is checked against each new `/map` and costmap: with
  fewer than `min_frontier_cells` frontier cells left within 0.5 m of it, the
  robot has seen that space on the way, and the goal is canceled (not
  blacklisted) and the next one chosen; if its footprint covers a lethal cell
  now, it is canceled and blacklisted. Without this, in the first run, Nav2
  took 53 s to give up on a goal at the edge of the corridor's unknown that
  turned out to be 0.2 m from a cabinet (RPP: "collision ahead", recoveries),
  long after all of the corridor had been seen: 72 s for the exploration.
- Each round uses a `/map` from after the last goal and a costmap made at
  least 0.3 s after that map (the global costmap updates at 5 Hz and
  publishes at 1 Hz, slam_toolbox's map at 1 Hz), waiting up to 3 s. The
  frontier cells are the map's newest, and a costmap older than the map has
  them unknown, so none is walkable. In one run before this the explorer
  reported the map explored after 12.5 s, with the corridor behind the
  counter peninsula unseen (88.7% of the reachable floor); the costmap's lag
  is the likely cause, not proven (the run recorded neither grid at that
  moment).

Nav2 not taking a goal (not active, or gone) aborts the exploration instead
of blacklisting the frontier, which would strike out every frontier in turn
and end in a false "explored". Everything is in `map`: in this stack the
odometry localises and slam_toolbox draws the map at its poses, with
`map -> odom` the identity, so `map -> pelvis` is the odometry's
`odom -> pelvis` ([slam_toolbox](#slam_toolbox)), and the goals are
floor-level poses in `map`.

The first round sees the map from the start, which slam_toolbox draws from
one scan or a few (it adds a scan once the pelvis has moved 0.1 m or turned
0.1 rad). Its rays leave unknown cells between them beyond about 5.7 m (0.5
deg bins, 5 cm cells), so in 7 of the 11 earlier runs the first cluster had
784-878 cells and the first goal was 1.5-1.6 m out (in 2 of the 4 below, 769
and 804 cells, 1.5-1.6 m); it was seen on the way after 2-3 s. On the robot, with its own scan pattern, this may differ.

### Exploring in the G1 sim

`tests/g1/test_sim_explore.sh` (`GOLEM_SIM_ODOM=1` for `sim_gt`) runs the
G1 sim in the nav test's kitchen and spawn, with
`g1_sim.launch.py use_locomotion:=true use_nav:=true
frontier_explore:=true`, and sends one `SkillFrontierExplore` goal (timeout
120 s, the defaults otherwise). It ends when the explorer reports, then stops
the sim and the bringup as the nav test does. The floor the robot can reach
comes from the sim's own geometry (`scene.npz`, the nav test's grid): 6166
cells of 5 cm (15.4 m^2) connected to the spawn through cells at least 0.35 m
from anything the robot can touch. It passes if the robot never fell (pelvis
height, tilt and the sim's fall log over the whole recording), no robot geom
touched anything but the floor, every `/cmd_vel` was inside the caps with the
smoother its only publisher, the final `/map` has at least 95% of the
reachable floor free, and every occupied cell of it is within 0.15 m of the
kitchen's geometry in the scan's band. When the explorer reports the map
explored, its `goals_reached` has to be above 0 and equal the goals its log
has reached by Nav2 plus those seen on the way. Otherwise the explorer's
goals and outcome, and the nav test's scan-path, scan-phantom and
localisation checks, are reported, not counted (their limits are for one
goal).

- The limits: 120 s is four times the longest exploration (29.8 s below,
  28.7 s in the 11 earlier runs). Coverage was 100% in every run below and in those 11, and
  88.7% in the run that stopped before the corridor (694 reachable cells
  unseen), so 95% catches a skipped corridor with room for a few unseen cells.

4 runs at the current settings (3 fastlio, 1 `sim_gt`; host load 4.5-7.7,
real-time factor 0.88-0.99 per run and 0.93-0.99 while exploring), all
passing:

- the explorer reported the map explored after 23.6-29.8 s of sim time
  (fastlio 23.6-26.4 s, `sim_gt` 29.8 s), with 4-6 goals sent and 3-6
  reached. Nav2 reached none of them: each was seen on the way after 1.0-6.2
  s and canceled, except 2 in one run whose footprint came to cover an
  obstacle (blacklisted after 2.0 and 3.1 s). The robot walked 5.8-6.8 m. In
  the 11 earlier runs (slam_toolbox's base frame on the pelvis, the global
  costmap with about half of the scan) it was done after 20.7-28.7 s, with
  4-6 goals, at most one blocked per run, and `goals_reached` 0 in all;
- 100.0% of the reachable floor was free in the final `/map`; its 1451-1743
  occupied cells were all within 0.10 m of the kitchen's geometry in the
  band (99th percentile 0.05 m);
- `/cmd_vel` stayed at vx 0..0.4 m/s, vy 0, wz -0.44..0.6 rad/s; the pelvis at
  0.712-0.781 m and tilted at most 9.3-9.6 deg; no contact, no fall logged;
- for the record: `/scan` never paused more than 0.1 s, no `/scan` return
  within 4 m lay over 0.25 m from the kitchen, and `map -> pelvis` was
  within 0.02-0.53 cm of the truth while exploring.

## FAST_LIO configs

- `g1_fastlio_sim.yaml`: the G1 sim's MID-360. The raw sensor frame (extrinsic
  R identity), every point kept, 0.1 m voxels, `blind` 0.7 m (the arm and
  hand meshes in `home` are 0.38 to 0.57 m from the lidar), extrinsic not
  estimated online.
- `g1_fastlio.yaml`: the real robot, not used yet. Every value marked
  `TODO(lidar)` depends on how the driver will run.

## The real lidar

Connecting the Livox driver to the MID-360 is deferred, and the G1 has no
`MID360_config.json` of its own (`h12_slam/config/MID360_config.json` is the
H1-2's). Before the first run:

- The driver applies `MID360_config.json`'s extrinsic (roll 180 today) to the
  points but not to the IMU. With it, `g1_fastlio.yaml`'s `extrinsic_R` becomes
  diag(1, -1, -1) and `LIDAR_FRAME` in the launch file becomes a roll of pi.
  With a zero extrinsic, both stay identity (as in the sim).
- The stamps have to be on the laptop's clock: unsynced, the driver stamps
  packets with the host clock on arrival; PTP or GPS sync passes the lidar's
  time through.
- `mid360_link -> livox_frame` is published here for now. When the lidar comes
  into the bringup, one of the two should publish it.
- `livox_to_pointcloud`'s deskew takes `/livox/imu`'s angular velocity in the
  points' axes (rad/s). With the roll-180 extrinsic above, the driver turns the
  points but not the IMU, so the gyro would need the same turn first; and the
  driver's IMU and point stamps have to be on one clock.

## Tests

In the ROS image, with the workspace built (they need `g1_description`,
`livox_ros_driver2` and `fast_lio`):

```bash
cd core_ws/src/g1/g1_navigation
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:../g1_description:$PYTHONPATH python3 -m pytest test
```

They run on DDS domain 97 on loopback only (`test/stack.py` points every
process at a unicast-localhost CycloneDDS config), so they need neither the
robot, the sim, multicast nor `NET_ADMIN`. About 120 s in all.

- `test_frames.py`: `frames.py` against Pinocchio's forward kinematics of the
  URDF. With a standing pose whose soles touch the floor, lidar points on the
  floor and on a wall land at z = 0 and at the wall, unmirrored, through
  FAST_LIO's frames; the pelvis starts level at `pelvis_height`. At any waist
  angle the IMU stays fixed on the torso, and a fixed `body -> pelvis` would
  tilt the lidar by the waist angle.
- `test_launch.py`: `--show-args`, and the launch description for both odom
  sources: the nodes, the static transforms, `body_to_pelvis`'s FAST_LIO
  config, the `/cmd_vel` and scan wiring (the converter in `pelvis_level`,
  `pointcloud_to_laserscan` with no TF and on the wall clock, the rest on sim
  time), and slam_toolbox's config: no scan matching, no loop closing and
  `base_footprint` as its base frame.
  The frontier explorer only with `frontier_explore:=true`, on sim time with
  `g1_nav2.yaml`'s footprint.
- `test_nodes.py`: the three helper nodes, also over DDS. The deskew undoes a
  9 deg turn within a frame and a rate that sways at HOMIE's 1.75 Hz (to 2 mm at
  10 m), holds the gyro's last rate for a little, and passes a frame through
  without gyro samples around it. The converter places the cloud in its
  `target_frame` through TF, and drops a frame it cannot place once it is
  `max_tf_wait` behind the newest. `level_frame` publishes `pelvis_level`
  level in odom and `base_footprint` under it on odom's floor, both at the
  stamp of `odom -> pelvis`.
- `test_frontier_explore.py`: the explorer on synthetic OccupancyGrids:
  frontier cells (next to unknown, the map's edge counting as unknown, not
  next to a wall), cluster sizes against `min_frontier_cells`, frontiers
  behind a wall or in a gap the robot does not fit through left out, the
  goal cell nearest the centroid and moved where the footprint fits, the
  robot snapped to walkable space, a costmap of another extent, the nearest
  candidate past `min_goal_distance` and the blacklist, done when nothing is
  left, goals outdated once seen or blocked, and the default footprint equal
  to `g1_nav2.yaml`'s. Then the node over DDS against a fake slam_toolbox
  and Nav2 (two rooms, the costmap lagging the map): it explores both rooms
  and reports done, blacklists failing goals and goals blocked on the way and
  sends none near them, cancels goals seen on the way, counts the goals Nav2
  reached and those seen on the way in `goals_reached` and the failed and
  blocked ones not, cancels Nav2's goal on a cancel, refuses a second goal
  and negative fields, and aborts at its timeout.
- `test_velocity_smoother.py`: the smoother with `g1_nav2.yaml` and the
  launch file's remappings clamps out-of-range commands to the caps, and
  after its input stops holds the last command for less than 0.5 s, then
  ramps to zero and goes quiet.
- `test_fastlio.py`: FAST_LIO from the launch file (fastlio, `sim:=true`)
  subscribes to the MID-360's topics, `odom -> camera_init` is `frames.py`'s,
  `body_to_pelvis` follows the fake's waist, and in TF `body -> livox_frame`
  is FAST_LIO's extrinsic. The fake stands the way HOMIE does (waist pitched
  forward 0.06 rad, the pelvis back by as much; more than HOMIE's 0.0256, so
  that a waist counted twice stands clear of FAST_LIO's own error). On its
  synthetic MID-360 and IMU, FAST_LIO's `odom -> pelvis` and
  `odom -> mid360_link` follow the fake out 0.85 m, through a 180 deg turn
  and back, within 5 cm, 1.5 deg of tilt and 1 deg of yaw (typically 2 cm,
  0.2 deg and 0.1 deg; the tilt error reaches about 1 deg if it turns before
  it has walked). The scan and the SLAM map, which hang off FAST_LIO in this
  mode, come out too, with nothing that is not in the room.
- `test_nav2_smoke.py`: the whole stack in `sim_gt` mode with `fake_g1.py` (a
  kinematic G1 that leans like HOMIE, pelvis and waist, and ray-casts the
  upside-down MID-360 in a 7 x 4 m room) and `robot_state_publisher`. A goal 4
  m away behind a 0.6 m box succeeds (the test takes about 22 s), the
  footprint never overlaps the box, every `/cmd_vel` is inside the caps, the
  smoother is its only publisher, and the SLAM map holds nothing that is not
  in the room. The
  goal is in the box's shadow at first, so the first plans can hug the box's
  unseen back; RPP's collision check may then stop the robot, and a
  costmap-clearing recovery replans. With NavFn (it priced unknown cells like
  the cells next to an obstacle) that happened once per run with the earlier
  0.40 m footprint and 1 Hz global costmap, and NavFn also failed to extract a
  path once or twice per run. With Smac 2D, the current footprint, the map
  drawn at the odometry's poses and Nav2 on `base_footprint`, the goal
  succeeded in 5 of 5 runs, the robot ending 0.17-0.18 m from it after
  3.85-3.86 m (with Nav2 on the tilted pelvis, 4 of 8 did not finish within
  150 s or took 159 s; [Nav2](#nav2)).

## License

MIT.
