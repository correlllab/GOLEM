# Shared SLAM package migration — 2026-10-29

Implemented on `codex/golem-slam` in the existing checkout. No commit or push
was made. All three package directories, Python modules, resource markers,
manifests and callers were renamed:

- `h12_slam` → `golem_slam`
- `h12_deploy_mjpc` → `deploy_mjpc`
- `h12_skills` → `golem_skills`

G1 bringup now selects the shared H1-2 SLAM pipeline with G1 sensor extrinsics,
footprint, height bands and conservative motion limits. H1-2 launch defaults
are preserved. `g1_navigation` remains separate for its legacy pipeline and
reused waist transform/frontier explorer. Real G1 navigation and shared-pipeline
`sim_gt` remain refused; motor interlocks are unchanged.

## Validation

- Humble build of all three renamed packages and `golem_bringup`: passed,
  including both MJPC C++ cores. Installed executable discovery and imports
  resolve under the new names.
- `golem_slam`: 29 passed, 1 skipped; `golem_bringup`: 53 passed, 1 skipped.
- `golem_skills` functional suite: 24 passed. Its full suite still fails
  flake8/pep257; the pre-change package reproduces both failures. Existing
  ignored logs also add lint noise. Rename-induced indentation was fixed.
- G1 point-to-point acceptance: 13/13 passed, with all velocity commands
  routed through the smoother. The final rerun after the delayed-TF fix also
  passed 13/13: goal reached in 15.1 simulated seconds, maximum localization
  error 1.03 cm, maximum tilt 7.8°. Artifacts:
  `tests/results/golem-slam-g1-nav-final`.
- H1-2 RoboCasa arm smoke: passed forward and return movements. Artifacts:
  `tests/results/stack-robocasa-arms-20261029T131756-2935319`.
- G1 extended exploration: **7/9 checks passed**. It timed out after 120 sim
  seconds with 55.3% coverage (95% required) and exceeded stability limits:
  tilt 22.3° (15° allowed), pelvis height 0.693 m (0.700 m required).
  Localization stayed within 1.4 cm; scans remained continuous and no obstacle
  contacts occurred. Artifacts: `tests/results/golem-slam-g1-explore-2`.
  Reliable autonomous exploration remains unverified; no locomotion policy
  changes or weakened acceptance thresholds were made to hide this result.

## Implementation decisions discovered during validation

- Added a G1-only bounded, stamp-exact cloud transform before scan conversion
  because Humble's scan TF filter stalled despite continuous input clouds.
- Retry the latest G1 odometry sample when waist TF arrives a few milliseconds
  later; never substitute a stale transform. H1-2 retains its fixed transform.
- Scope a G1-only behavior-server remap so recovery commands also pass through
  Nav2's velocity smoother.
- Preserve stale renamed-package build/install caches under the ignored
  `.superpowers/sdd/2026-10-29-golem-slam/stale-cache` directory. Current ament
  discovery no longer finds their old resource markers. User migration steps
  are documented in `docs/RUNNING.md`.
- Used the existing checkout on a dedicated branch to retain Docker bind mounts.
  No vendored submodule changes were made.

The requested independent review could not run because the reviewer hit an
account usage limit. A self-review and the automated/runtime checks above were
performed instead. Mac and physical-robot execution were not tested.
