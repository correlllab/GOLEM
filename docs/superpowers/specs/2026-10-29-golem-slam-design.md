# Shared SLAM package and package renames

## Intent

Rename `h12_slam` to `golem_slam` and make its existing navigation pipeline
available on the G1 through additional configuration and launch files. Do not
merge `g1_navigation` into it. Preserve the H1-2 defaults and behavior.
The original request also renames `h12_deploy_mjpc` to `deploy_mjpc` and
`h12_skills` to `golem_skills`; these remain part of this migration.

## Shared pipeline

Both robots use FAST-LIO's deskewed body cloud, URDF-based self filtering,
gravity/floor alignment, OctoMap occupancy and projection, slam_toolbox with
scan matching and loop closure, and Nav2 with MPPI and the humanoid recovery
tree. Keep the H1-2 launch filenames as entry points under the new package.
Factor shared launch construction into an installed Python module so the G1
entry point does not duplicate the full pipeline.

Keep robot-specific values in explicit configurations: FAST-LIO extrinsics
and preprocessing, floor-detection thresholds, scan and occupancy heights,
footprint, and controller/smoother limits. Preserve the H1-2 config values.
The G1 uses its existing footprint and conservative locomotion limits; lateral
commands remain disabled initially. MPPI configuration must agree with that
motion constraint. This change does not establish new HOMIE motion limits.

## G1 adaptation

Add `g1_navigation.launch.py` under `golem_slam`. It runs the shared pipeline
with G1 configuration. Use the existing G1 FAST-LIO simulation extrinsics and
sensor frames, including the upside-down sensor mounting. Reuse
`g1_navigation`'s dynamic `body_to_pelvis` node and sensor-frame helpers rather
than the H1-2 fixed body-to-pelvis transform. Do not run the old `level_frame`,
raw cloud converter, or static odom-to-camera_init publisher alongside the
shared pipeline: `level_odom` owns odom alignment and base_footprint.

Set a G1-specific floor-depth threshold and scan/occupancy height bands based
on its model. Verify that the self filter loads its collision geometry and
that level_odom follows the current waist-dependent pelvis transform rather
than caching a fixed transform. Adapt the latter if inspection/tests show it
assumes a rigid sensor-to-pelvis relationship.

The new entry point supports FAST-LIO only: its cloud is required by the
shared processing pipeline. Reject `odom_source:=sim_gt` with an actionable
message before launching nodes. The old `g1_navigation` package and its
standalone launch retain ground-truth navigation for comparison. Do not
silently fall back to the old pipeline in the new launch.

Keep optional frontier exploration by launching the existing explorer with
the G1 footprint. Validate its map/costmap handling with scan matching enabled;
do not assume map and odom remain identical.

## Bringup and compatibility

Route G1 bringup's optional navigation through `golem_slam`; retain the real
G1 navigation refusal until the real lidar integration is implemented.
The new standalone G1 launch must also refuse real mode for now, rather than
presenting the unverified real sensor configuration as supported.
Update H1-2 bringup references to the renamed package. Keep Mac H1-2 support;
G1-only imports and runtime dependencies must not break its launch/build path.

Rename package directories, Python modules, resource markers, manifest names,
entry points, and references for all three renames. Update maintained docs,
tests, Docker build/launch scripts, and package tables. Historical design
documents need not be rewritten. Document targeted stale build/install cleanup
for the old names, including native and container test workspaces.

## Verification

Run existing geometry, floor alignment, lidar frame, and bringup tests against
the renamed package. Add focused tests for G1 launch selection/refusals,
sensor extrinsics, moving-waist transforms, G1 floor detection, installed
resources, and agreement between controller and smoother limits. Check that
no active references use the removed package names.

Build through the provided container scripts on a nonzero DDS domain. Run
G1 simulation navigation with the shared pipeline and inspect floor alignment,
self-filtered cloud, map/projected_map, costmaps, goal completion, and optional
exploration. Run the existing H1-2 smoke checks to detect regression. Report
unavailable runtime checks explicitly; static checks alone do not establish
that the G1 navigation pipeline works in simulation or on hardware.

## Scope exclusions

No deletion or wholesale refactoring of `g1_navigation`, no real G1 sensor
bringup, no motor safety changes, no new robot capabilities implied by the
MJPC/skills package renames, and no edits to vendored models or FAST-LIO.
