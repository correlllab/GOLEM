# golem_slam

Shared SLAM/navigation for the H1-2 and the G1, with robot-specific
configuration and, for the G1, the dynamic waist transform (`g1_body_to_pelvis`).
Both robots' launch files build the same pipeline (`golem_slam/navigation_launch.py`).

- `h12_navigation.launch.py`: the H1-2's pipeline, on `mid360.yaml`.
- `h12_real_slam.launch.py`: H1-2 sensors and navigation without motor control.
- `g1_navigation.launch.py`: the G1's pipeline, on the sim's FAST-LIO config
  (`g1_fastlio_sim.yaml`) by default or the real robot's with
  `fast_lio_config_file:=g1_fastlio.yaml`; it needs the robot's state
  (robot_state_publisher) and the MID-360's `/livox/lidar` and `/livox/imu`.
  FAST-LIO is the only odometry. The G1 bringups include it with `use_nav:=true`.

FAST-LIO's deskewed cloud is self-filtered against the URDF, then used for
floor alignment and OctoMap. slam_toolbox receives `/converted_scan`; Nav2
uses `/projected_map` and local scan obstacles. The G1 uses a dynamic waist
transform and its own footprint, height bands and motion limits. A bounded
`level_cloud` adapter transforms its scan input into `base_footprint` before
pointcloud_to_laserscan, avoiding the Humble TF-filter stall observed while
walking. A G1-only remap routes recovery commands through the velocity smoother.

See [G1 navigation](../../../docs/g1/navigation.md) and
[stale installs after a package rename](../../../docs/RUNNING.md#troubleshooting).

From this package directory in a built and sourced ROS container:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test -q
```
