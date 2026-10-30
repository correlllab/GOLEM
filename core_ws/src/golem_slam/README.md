# golem_slam

Shared SLAM/navigation for H1-2 and G1. This is the renamed `h12_slam`, with
robot-specific configuration and the dynamic waist transform (`g1_body_to_pelvis`)
for the G1.

- `h12_navigation.launch.py`: unchanged H1-2 defaults.
- `h12_real_slam.launch.py`: H1-2 sensors and navigation without motor control.
- `g1_navigation.launch.py sim:=true use_sim_time:=true`: G1 shared pipeline,
  requiring state publishing and the sim sensor stream. Real mode and sim_gt
  are refused. Bringup selects this launch with `use_nav:=true`.

FAST-LIO's deskewed cloud is self-filtered against the URDF, then used for
floor alignment and OctoMap. slam_toolbox receives `/converted_scan`; Nav2
uses `/projected_map` and local scan obstacles. The G1 uses a dynamic waist
transform and its own footprint, height bands and motion limits. A bounded
`level_cloud` adapter transforms its scan input into `base_footprint` before
pointcloud_to_laserscan, avoiding the Humble TF-filter stall observed while
walking. A G1-only remap routes recovery commands through the velocity smoother.

See [G1 navigation](../../../docs/g1/navigation.md) and
[cache migration](../../../docs/RUNNING.md#package-name-migration).

From this package directory in a built and sourced ROS container:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test -q
```
