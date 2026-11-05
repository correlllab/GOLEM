# Navigation

G1 bringup uses `golem_slam`, the one navigation graph both robots run, on
the G1's config directory, `golem_slam/config/g1/`.
Both G1 bringups start it by default (`use_nav`). On the real G1 the bringup
also starts the MID-360 driver (`golem_bringup`'s `launch/g1/g1_real_drivers.launch.py`,
on `golem_slam/config/g1/MID360_config.json`: the laptop at 192.168.123.222,
the lidar at 192.168.123.120, points and IMU in the raw sensor frame) and
FAST-LIO on `config/g1/fastlio.yaml`; the sim bringup runs FAST-LIO on
`config/g1/fastlio_sim.yaml`. The real path has not run on our G1 yet.

## Pipeline

FAST-LIO consumes `/livox/lidar` and `/livox/imu`. Its deskewed body cloud is
filtered against the G1 URDF collision geometry, then used for gravity/floor
alignment and OctoMap. A stamped `level_cloud` adapter supplies the scan
converter in `base_footprint`, avoiding its TF-filter stall on the moving G1.
slam_toolbox uses `/converted_scan`, with scan matching
and loop closure enabled. Nav2 uses the OctoMap `/projected_map` as its static
obstacle map, plus the scan for local obstacle updates. MPPI drives HOMIE
through the velocity smoother: forward/backward and yaw, with lateral velocity
held at zero. The G1's configs set its own footprint, motion limits and height
bands (`nav2.yaml`, `octomap.yaml`, and `pipeline.yaml`'s scan band).

The dynamic body-to-pelvis transform follows the measured waist. `level_odom`
measures gravity and the floor rather than assuming a standing pelvis height.
Goals are in `map`; do not assume map equals odom or the simulator's world.

## Run

Start the simulator through the provided wrappers on a nonzero DDS domain,
with `GOLEM_SIM_ODOM` unset or 0, then in the ROS container:

```bash
ros2 launch golem_bringup g1_sim.launch.py
# Optional skills (including frontier exploration):
ros2 launch golem_bringup g1_sim.launch.py use_skills:=true
# The real G1 (the laptop's golem_ros container, domain 0):
ros2 launch golem_bringup g1_real_robot.launch.py
```

Run only one of those bringups. The navigation-only entry point, requiring an
already-running G1 state publisher, sensor stream and locomotion stack, is:

```bash
ros2 launch golem_slam navigation.launch.py robot:=g1 use_sim_time:=true \
  fast_lio_config:=$(ros2 pkg prefix --share golem_slam)/config/g1/fastlio_sim.yaml
```

Send `/navigate_to_pose` goals in `map` once Nav2 is active. Its other file
arguments are `nav2_params_file`, `octomap_params_file` and `slam_params_file`;
each defaults to the robot's (`golem_slam`'s README).

## Check

```bash
tests/g1/test_sim_nav.sh
tests/g1/test_sim_explore.sh
```

These run the shared pipeline in isolated containers. `GOLEM_SIM_ODOM=1` is
refused: the shared pipeline requires FAST-LIO's cloud. The test reports goal
completion, stability, contacts, scan quality and localization; inspect its
report rather than treating launch success as a navigation pass.

### Validation status

In the sim, the shared pipeline passes the point-to-point navigation acceptance
test (13/13 checks). Extended G1 frontier exploration does **not** pass yet: a
120-second run reached 55.3% coverage versus the 95% requirement and exceeded
the stability limits (22.3° maximum tilt, 0.693 m minimum pelvis height). Scans remained continuous,
localization error stayed below 1.4 cm, and there were no obstacle contacts.
The explorer remains opt-in (`use_skills:=true`); its integration is present, but these results do
not establish reliable autonomous exploration or real-G1 navigation.

For old install directories after upgrading, see the stale-install item in
[Troubleshooting](../RUNNING.md#troubleshooting).
