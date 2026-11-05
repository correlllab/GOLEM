# golem_slam

SLAM and navigation for the H1-2 and the G1: one node graph, built by
`golem_slam/navigation_launch.py`, with each robot's values in its config
directory.

- `navigation.launch.py robot:=<robot>`: the graph on `config/<robot>/`
  (`robot` is required; `h12` or `g1`). It needs the robot's state
  (robot_state_publisher with the joint states) and the MID-360's
  `/livox/lidar` and `/livox/imu`, stamped `livox_frame`. FAST-LIO is the only
  odometry. The bringups include it: the H1-2's with `use_nav` (`use_slam` on
  the `lmg_*` ones), the G1's `g1_core.launch.py` with `use_nav`.
- `h12_real_slam.launch.py`: H1-2 sensors and navigation without motor control.

Arguments: `use_sim_time` (default false), and the files, each defaulting to
the robot's: `fast_lio_config` (`config/<robot>/fastlio.yaml`; the G1 sim
passes `config/g1/fastlio_sim.yaml`), `nav2_params_file` (`nav2.yaml`),
`octomap_params_file` (`octomap.yaml`) and `slam_params_file` (the shared
`config/slam_toolbox.yaml`).

## The graph

The same nodes, executables and remaps on every robot:

- a static identity `<lidar link> → livox_frame` (`livox_frame_tf`): the
  driver and the sims stamp the points and the IMU in the sensor's raw frame,
  which is the URDF's lidar link;
- FAST-LIO (`fast_lio`) on `fast_lio_config`;
- `body_to_pelvis`: FAST-LIO's `body` (its IMU) → `pelvis`, from the config's
  lidar-to-IMU extrinsic and the live TF from `pelvis` to `livox_frame`, so the
  pelvis follows the torso joint (H1-2) or the waist (G1);
- `self_filter`: drops the robot's own body from FAST-LIO's deskewed cloud
  (`/cloud_body_filtered`);
- `level_odom`: gravity- and floor-aligned `odom`, and `odom → base_footprint`
  per FAST-LIO odometry, with `body → pelvis` at each odometry stamp;
- OctoMap (`/projected_map`, the static layer of both costmaps);
- `level_cloud`: the filtered cloud in `base_footprint` (`/cloud_base_filtered`),
  with stamp-exact lookups and a bounded queue instead of a TF message filter,
  which can stall while the robot walks;
- `cloud_registered_to_laserscan` (pointcloud_to_laserscan) → `/converted_scan`
  → slam_toolbox (`map → odom`);
- Nav2's navigation half, with `behavior_server`'s `cmd_vel` remapped so every
  Nav2 command goes through the velocity smoother.

## Configs

`config/<robot>/` holds the robot's FAST-LIO config(s), its MID-360 driver
config (`MID360_config.json`, read by the real robot's driver launch), Nav2
(`nav2.yaml`), OctoMap (`octomap.yaml`) and `pipeline.yaml`, whose keys are all
required: `lidar_link` (the URDF link of the raw sensor frame), `scan`
(`min_height`, `max_height`: the scan band in m above the floor) and
`level_odom` (`min_floor_depth`). The H1-2's FAST-LIO extrinsic is the URDF's
`livox_link` in `livox_imu_link`; `test/test_navigation_launch.py` checks it
against the URDF.

See [G1 navigation](../../../docs/g1/navigation.md) and
[stale installs after a package rename](../../../docs/RUNNING.md#troubleshooting).

From this package directory in a built and sourced ROS container:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test -q
```
