# Faster RoboCasa sims: measured results

This page records the real-time factors of the RoboCasa sims after each tier of
[the sim speed design](2026-11-05-sim-speed-design.md), against that spec's baseline
("What the measurements say"). One table per tier, added when the tier is finished.

How the numbers are taken:

- **Full stack** is the `real-time factor` line of the summary of `tests/sim/test_sim_stack.sh
  --h12|--g1` (sim seconds per wall second over the run, headless, sensors on, the robot's whole
  bringup and navigation beside the sim). The wrappers print `cpus: sim <list>, ros <list>` first;
  the pinning is part of what is measured.
- **With the viewer** is the sim's own `real-time factor ... over the last 30 s` log line with the
  MuJoCo window open on a display, in a disposable container with the stack idle.
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
| With the viewer | H1-2 0.89x, G1 0.21x | 0.93-0.97x (controller's run) | 1.00x (Task 1) |
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
