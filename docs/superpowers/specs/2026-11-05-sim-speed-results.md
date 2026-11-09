# Faster RoboCasa sims: measured results

This page records the real-time factors of the RoboCasa sims after each tier of
[the sim speed design](2026-11-05-sim-speed-design.md), against that spec's baseline
("What the measurements say"). One table per tier, added when the tier is finished.

How the numbers are taken:

- **Full stack** is the `real-time factor` line of the summary of `tests/sim/test_sim_stack.sh
  --h12|--g1` (sim seconds per wall second over the run, headless, sensors on, the robot's whole
  bringup and navigation beside the sim). The wrappers print `cpus: sim <list>, ros <list>` first;
  the pinning is part of what is measured.
- **With the viewer** is the sim alone (no ROS stack beside it) with the MuJoCo window open on a
  display, in a disposable container on its own DDS domain, read from the sim's own log: the G1's
  `[g1_mujoco] ... real-time factor ... over the last 30 s` lines, the H1-2's `[sim-timing] RTF=`
  lines (it runs with `--timing`). These are not full-stack runs and not from the suite run of
  the table's commit; each row says which commit it was taken at.
- **Sim alone** is the sim without the ROS stack; no suite of the closed-loop tests reports it.
- A factor of 1.00x is a ceiling: the sim paces itself to real time, so it cannot read higher.

## Tier 1

Tier 1 changes no physics and no topic: the G1's viewer runs on its own thread, `rt/lowstate`'s
bytes are built with numpy, the sim is pinned to the performance cores and the ROS stack to the
efficiency cores of a hybrid CPU, and the H1-2's camera snapshot copies only what the renderer
reads.

Measured 2026-11-08 on the i7-14700F with an RTX 5070 Ti, at commit `2d8efae` (branch
`sim_speed`), `cpus: sim 0-15, ros 16-27`.

| | Baseline (spec, 2026-11-05) | Tier 1 H1-2 | Tier 1 G1 |
|---|---|---|---|
| Full stack (`tests/sim`) | H1-2 0.75x, G1 0.92x | 0.83x | 1.00x (capped by real-time pacing) |
| With the viewer (sim alone, not the commit above) | H1-2 0.89x, G1 0.21x | 0.93-0.97x: the H1-2 sim alone with the viewer and `--timing`, at commit `d200e9f`, unpinned, 150 s wall on domain 231, read from the `[sim-timing] RTF=` lines | 1.00x: the G1 sim alone with its sensors and elastic band, at commit `d200e9f` (before the pinning, `rt/lowstate` and camera changes), unpinned, about 90 s wall on domain 231, read from the `[g1_mujoco]` lines at sim t 31 s and 61 s (1.00x each) |
| Sim alone | H1-2 1.11x, G1 1.45x | not measured | not measured |

The other closed-loop suites measured on the same run, all passing:

| Suite | Result | Real-time factor |
|---|---|---|
| `tests/sim/test_sim_stack.sh --h12` | 14/14 | 0.83x |
| `tests/sim/test_sim_stack.sh --g1` | 14/14 | 1.00x |
| `tests/g1/test_sim_bringup.sh --walk` | 22/22 | 1.00x (min 1.00) |
| `tests/g1/test_sim_nav.sh` | 13/13 | 1.00x (min 0.99) |
| `tests/stack/run.sh robocasa arms` | passed | not reported |
| `tests/stack/run.sh robocasa locomotion` | passed | not reported |

Pinning alone, measured separately on 2026-11-05 (one run each, before the other Tier 1 changes):
the H1-2 sim ran at 0.86x instead of 0.79x unpinned, and the G1 at 0.98x instead of 0.92x. That
is not part of the table above, which is the whole of Tier 1.

Every figure here is a single run. The H1-2's full-stack 0.83x against the pinning-alone 0.86x is
within run-to-run spread, not a regression shown.

## Tier 2

Tier 2 changes only the lidar's points: both robots' sims publish the shared MID-360 model
(`golem_robocasa/mid360.py`), the real scan pattern at 200,000 points per second in 20,000-ray
frames at 10 Hz, 0.1 m blind zone and 2 cm range noise, with each ray without a return a point
at (0, 0, 0) as the real driver publishes it. The rays are cast on a scan thread beside the
physics, from depth images on the GPU where MuJoCo renders through EGL (the `auto` backend), else
exactly on the CPU (`GOLEM_LIDAR_BACKEND=cpu`, the Mac's path). If the casts fall more than a frame
behind, the physics waits for them instead of dropping rays. The G1 sim's FAST-LIO values
(`golem_slam/config/g1/fastlio_sim.yaml`) are retuned for the denser, noisier scan; the H1-2's
`fastlio.yaml`, shared with the real robot, is unchanged.

Measured 2026-11-08 on the i7-14700F with an RTX 5070 Ti, on the tree committed as 128186f (a unit test tightened after the runs; branch
`sim_speed`, on top of `1693ad4`), `cpus: sim 0-15, ros 16-27`. The closed-loop suites give their
sim container the GPU (the nvidia runtime and `MUJOCO_GL=egl`), so `auto` cast on the GPU; each
sim logged `lidar rays cast on the GPU every 20 ms of sim time`.

| | Tier 1 H1-2 | Tier 1 G1 | Tier 2 H1-2 | Tier 2 G1 |
|---|---|---|---|---|
| Full stack (`tests/sim`), GPU caster | 0.83x | 1.00x (capped) | 1.00x (capped; no cast waited) | 1.00x (capped; no cast waited) |
| Full stack (`tests/sim`), CPU caster | | | 0.74x: the physics waited for the lidar at 4827 of 5500 casts | not measured |

The CPU row is one extra run with `GOLEM_LIDAR_BACKEND=cpu` in the shell, which the closed-loop
harnesses now pass to the sim: it shows what a host without a GPU caster gets with the stack beside
it, not the Mac itself (the Mac image cannot be built on this host).

The closed-loop suites on the final values, all on the GPU caster:

| Suite | Result | Real-time factor |
|---|---|---|
| `tests/sim/test_sim_stack.sh --h12` | 14/14 | 1.00x |
| `tests/sim/test_sim_stack.sh --g1` | 14/14 | 1.00x |
| `tests/g1/test_sim_nav.sh` | 13/13 in each of 3 runs | 1.00x |
| `tests/g1/test_sim_bringup.sh` | 22/22 | 1.00x |
| `tests/g1/test_sim_bringup.sh --walk` | 22/22 | 1.00x |
| `tests/g1/test_sim_explore.sh` | 7/9 in each of 2 runs: coverage and tilt, the open issue in `tests/g1/README.md` (6/9 and 7/9 before Tier 2) | 1.00x |

How the G1's FAST-LIO values were chosen, one nav run or more per setting (the test's goal is the
same in every run):

| `point_filter_num` | `filter_size_surf`, `filter_size_map` | Nav runs |
|---|---|---|
| 1 | 0.1 m | 2 failed (10/13): FAST-LIO fell behind the scans, Nav2's controller missed its rate 58-69 times and lost the robot's pose in the plan's frame, and the robot tilted past 15 deg |
| 3 | 0.1 m | 1 passed, 1 failed (12/13: one phantom map cell); a CPU-caster run failed the same way |
| 4 | 0.1 m | 1 passed, 1 failed (12/13: two phantom map cells) |
| 3 | 0.2 m | 3 passed (the final values) |

The phantom cells are floor returns that the navigation pipeline lifts into the scan's height
band: the raw `/livox/lidar` returns of a run, placed with the sim's true sensor pose, hold none
in free space (0 of 2.1 million in the band within 4 m).

Every figure here is a single run unless the table says otherwise. The localisation check of the
nav runs on the final values peaked at 1.48-2.47 cm against its 2.5 cm limit (Tier 1's runs:
about 1.0 cm).
