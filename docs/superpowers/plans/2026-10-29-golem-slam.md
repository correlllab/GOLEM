# Shared SLAM and Package Renames Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Rename the three requested packages and expose the H1-2 navigation pipeline on the G1.

**Architecture:** `golem_slam` owns the shared FAST-LIO, self-filter, level-odom, OctoMap, slam_toolbox and Nav2 pipeline. Robot entry points select configuration and frame adapters. The existing `g1_navigation` package supplies the moving-waist adapter and optional explorer, and retains its standalone legacy pipeline.

**Tech Stack:** ROS 2 Humble, Python launch, ament Python/CMake, YAML, pytest, Docker, RoboCasa.

**Spec:** `docs/superpowers/specs/2026-10-29-golem-slam-design.md`

## Global Constraints

- Preserve the H1-2 defaults and behavior.
- Do not merge `g1_navigation` into `golem_slam`.
- The new entry point supports FAST-LIO only.
- Retain the real G1 navigation refusal until the real lidar integration is implemented.
- No motor safety changes or edits to vendored models or FAST-LIO.
- Preserve Mac H1-2 support; defer G1 imports until G1 launch selection.
- Run simulator checks on a nonzero DDS domain through repository wrappers.

## Review Focus

- Moving waist: successive odometry callbacks must use the matching pelvis transform.
- Short/upside-down robot: gravity and floor alignment must work with G1 sensor geometry.
- Missing TF: dynamic mode must skip publication instead of using a stale cached pose.
- Launch composition: real mode and sim_gt must fail before any new navigation node starts.
- Installed resources/Mac: renamed packages must resolve without importing G1-only modules on the H1-2 path.

## File Structure

- Rename `core_ws/src/h12_slam` and its Python module/resource to `golem_slam`.
- Rename `core_ws/src/h12_deploy_mjpc` and Python module to `deploy_mjpc`.
- Rename `core_ws/src/h12_skills` and Python module/resource to `golem_skills`.
- Create `core_ws/src/golem_slam/golem_slam/navigation_launch.py`: shared launch construction.
- Keep `launch/h12_navigation.launch.py` as the H1-2 entry point; add `launch/g1_navigation.launch.py`.
- Add `config/g1_fastlio_sim.yaml`, `g1_nav2.yaml`, `g1_octomap.yaml`, and `g1_pipeline.yaml`.
- Extend `golem_slam/level_odom.py` for optional dynamic base transforms.
- Add `test/test_g1_pipeline.py` and `test/test_navigation_launch.py`; extend `test/test_level_odom.py`.
- Update bringup launch files/tests, Docker package references, maintained docs and simulation harnesses.

## Task 1: Rename packages and their consumers

**Interfaces:** All existing executables and ROS topics remain; package/import names change.

- [x] Inventory tracked references with `git grep -n -E 'h12_slam|h12_deploy_mjpc|h12_skills'` and distinguish current code/docs from historical design documents and vendored submodules.
- [x] Move package/module/resource paths using the exact mapping:

```python
renames = {
    'h12_slam': 'golem_slam',
    'h12_deploy_mjpc': 'deploy_mjpc',
    'h12_skills': 'golem_skills',
}
```

- [x] Apply those replacements to active tracked sources, manifests, setup files, CMake project declarations, launch files, Docker mount/build scripts, current documentation and tests. Keep executable names unchanged. Read `docker/BUILD.md` before modifying `docker/RosDockerfile`.
- [x] Check discovery with `colcon list --base-paths core_ws/src`; require the new names and absence of the three removed names. Check Python syntax with `python3 -m compileall -q core_ws/src/golem_slam core_ws/src/deploy_mjpc core_ws/src/golem_skills`.
- [x] Run the renamed SLAM geometry/frame/floor test suite in the ROS environment. Retain the test results as the behavior baseline before adding G1 support.

## Task 2: Dynamic base transform and G1 floor regression

**Interfaces:** Add `dynamic_base_transform: bool = False` to `LevelOdom`; G1 sets it true. Preserve the H1-2 cache default.

- [x] Add a callback test with a fake TF buffer returning two different body-to-pelvis translations. Send two identical body odometry poses; assert the second footprint changes only with dynamic mode enabled. Add a third callback whose TF lookup raises `TransformException`; assert no footprint is published in dynamic mode.
- [x] Add a floor regression using the existing `plane` helper:

```python
def test_g1_floor_with_upside_down_sensor():
    points = plane(5000, -0.9)
    rotation = rpy_to_matrix(np.pi, 0.0, 0.0)
    up = rotation.T @ np.array([0.0, 0.0, 1.0])
    floor = find_floor_height(points @ rotation, up, 0.0, 0.5, 300)
    assert abs(floor + 0.9) < 0.01
```

- [x] Run `python3 -m pytest core_ws/src/golem_slam/test/test_level_odom.py -q`; confirm the dynamic callback regression fails before changing implementation.
- [x] Declare the new parameter and change the lookup branch to:

```python
if self.dynamic_base_transform or self.body_from_base is None:
    stamp = Time.from_msg(msg.header.stamp) if self.dynamic_base_transform else Time()
    try:
        body_from_base = transform_to_rt(self.tf_buffer.lookup_transform(
            self.body_frame, self.base_frame, stamp).transform)
    except TransformException:
        return
    self.body_from_base = body_from_base
```

- [x] Run the floor and callback tests again. Verify the unchanged default still caches the fixed H1-2 transform.

## Task 3: Shared launch and G1 profiles

**Interfaces:** `generate_navigation_launch_description(robot: str)` returns the shared LaunchDescription. Thin robot entry points call it with `h12` or `g1`. G1 declares `sim`, `use_sim_time`, `odom_source`, `frontier_explore` and config overrides.

- [x] Add launch tests evaluating both profiles. H1-2 must retain the existing nodes/parameters, G1 must include self_filter, level_odom, OctoMap and Nav2 and exclude legacy level_frame/raw cloud conversion/static odom alignment. Assert G1 real mode and sim_gt raise before returning executable actions.
- [x] Move existing H1-2 launch construction into the installed module, keeping its argument defaults intact. Replace the original entry point with:

```python
from golem_slam.navigation_launch import generate_navigation_launch_description


def generate_launch_description():
    return generate_navigation_launch_description('h12')
```

- [x] Add the corresponding G1 entry point using `'g1'`. Resolve G1 imports only in the G1 branch. Use the existing G1 FAST-LIO config and dynamic `body_to_pelvis` executable, plus its identity mid360_link-to-livox_frame static transform. Do not start the legacy static odom transform.
- [x] Use the following G1 pipeline profile, loaded into the level_odom, scan and OctoMap construction:

```yaml
level_odom:
  dynamic_base_transform: true
  min_floor_depth: 0.5
scan:
  min_height: 0.15
  max_height: 1.3
octomap:
  occupancy_min_z: 0.15
  occupancy_max_z: 1.4
```

- [x] Base G1 Nav2 on the H1-2 configuration, set both footprints to `[[0.44, 0.24], [0.44, -0.24], [-0.20, -0.24], [-0.20, 0.24]]`, use MPPI `DiffDrive` with `vy_max: 0.0`, `vy_std: 0.0`, and preserve conservative longitudinal/yaw limits `vx_max: 0.30`, `vx_min: -0.20`, `wz_max: 0.50`. Set smoother max/min velocities to `[0.30, 0.0, 0.50]` and `[-0.20, 0.0, -0.50]`. Update copied comments and lateral deadband settings to match.
- [x] Reuse H1-2 scan-matching/loop-closure settings and humanoid behavior tree. Add the explorer only on request, with the G1 footprint. Keep explorer goals in map coordinates; verify no code assumes identity map-to-odom.
- [x] Declare runtime/test dependencies and install every added config/launch file. Keep G1 package dependency compatibility with the Mac wrapper's existing stubs.
- [x] Run launch/profile tests and check controller/smoother limits agree. Load the actual G1 URDF into `link_boxes` to verify self-filter resource resolution, including mesh formats.

## Task 4: Bringup, harnesses and migration documentation

**Interfaces:** `g1_core` includes `golem_slam/g1_navigation.launch.py` with its existing explicit arguments; navigation remains off by default.

- [x] Update `golem_bringup/test/test_nav_launch.py` to expect the new include, FAST-LIO-only behavior, shared nodes, nested Nav2 include, and early sim_gt refusal. Preserve real-mode refusal tests.
- [x] Switch the include and validate odom_source in `g1_core` before starting any nodes. Update sim launch descriptions and docs to name the new pipeline. Do not change motor interlocks.
- [x] Update G1 integration harness expectations from `/scan` to `/converted_scan` and add `/cloud_body_filtered` and `/projected_map` readiness where appropriate. Refuse `GOLEM_SIM_ODOM=1` early in harnesses that use the new bringup, explaining the legacy standalone launch remains available. Keep legacy package tests targeting their old pipeline.
- [x] Update `docs/g1/navigation.md`, `docs/g1/packages.md`, `docs/RUNNING.md`, `tests/README.md`, package README and `AGENTS.md` to distinguish the new shared pipeline from retained legacy navigation. Document package-scoped removal of stale caches for `h12_slam`, `h12_deploy_mjpc`, and `h12_skills`; do not remove unrelated build outputs.
- [x] Run the bringup navigation/lidar tests plus package tests. Run `git diff --check` and repeat the active-reference inventory, allowing old names only in historical docs and explicit migration instructions.

## Task 5: Runtime validation and review

- [x] Inspect Docker availability, existing containers and DDS domains without replacing an active stack. Use the provided build/test scripts and the documented isolated test paths.
- [x] Build the ROS workspace with the new package names. Run `colcon test --packages-select golem_slam golem_bringup` in the sourced ROS environment, then `colcon test-result --verbose`.
- [x] Run `tests/g1/test_sim_nav.sh` on its isolated nonzero domain. Check initial floor alignment, continuous dynamic TF, self-filtered scans, OctoMap projection, costmaps and goal result. Run `tests/g1/test_sim_explore.sh` for explorer integration.
- [x] Run the existing H1-2 smoke path documented in `docs/DOCKER_TESTING.md` through the repository wrappers; do not run on domain 0.
- [x] Review the complete diff for changed defaults, missed resource paths, duplicate TF publishers, Mac import/dependency problems and accidental vendored changes. Report which runtime checks passed and which were unavailable, with the specific blocker.

## Execution

Recommended: native execution in this task, because renames, shared launch extraction,
and bringup changes touch overlapping files and benefit from one consistent view.
Implementation and validation runs are complete. The point-to-point G1 test
passed; extended exploration was run but failed coverage and stability checks.
The renamed skills package retains pre-existing lint failures. See
[validation report](../reports/2026-10-29-golem-slam.md) for evidence and limits.
Checked items above record performed work, not a claim that every runtime
acceptance test passed.
