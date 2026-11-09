# Faster RoboCasa sims with the real MID-360 — design

Status: approved by the user (2026-11-08: "begin implementing").
Evidence: the profiling and research of 2026-11-05 (measurements and scripts in the
session scratchpad `simspeed/`; summarised below).

## Intent

The user: speed up the simulator "without turning off sensors or reducing resolution";
losing Mac compatibility is acceptable for a better real-time factor. Also adopt what is
useful from the MID-360 simulation in
`bethalageetachandraraju/livox-mid360-lidar-unitree-g1-mjlab`.

## What the measurements say (2026-11-05, i7-14700F, RTX 5070 Ti)

| | H1-2 | G1 |
|---|---|---|
| Sim alone, unthrottled | 1.11x (lidar thread: 90-115 ms per 20k-ray scan) | 1.45x (main thread) |
| Full stack (`tests/sim`) | 0.65x before the leg merge, 0.75x after it (one torch thread) | 0.92x |
| With the viewer | 0.89x | 0.21x (`viewer.sync()` every step) |

- Physics is ~0.9 ms per step; 80% of it is collision, mostly kitchen fixture internals
  (dishwasher racks and buttons, drawer boxes) against their own housings.
- `rt/lowstate` costs 0.32 ms per step on the main thread (cyclonedds' Python serializer
  0.21 ms, CRC 0.04 ms).
- The sim's main thread lands on E-cores ~half the time (0.92 vs 1.42 ms per step).
- Cameras already render on the GPU (EGL, NVIDIA) and are off the critical path.
- MJX and MuJoCo Warp give no speed-up for one world (MuJoCo's own docs) and change the
  dynamics the walking policies were validated on: physics stays on CPU MuJoCo.
- The G1's lidar casts 5,000 points per frame (a quarter of the real 20,000); the H1-2's
  casts a uniform 360x56 grid, not the MID-360's non-repetitive pattern;
  `mj_multiRay`'s BVH culling lets 0.75-5.5% of rays through geometry.

## User decisions

- 2026-11-05: all four tiers, built after the `golem_lowerbody_rl` merge.
- 2026-11-08: begin implementing. Branch `sim_speed`, on top of `golem_lowerbody_rl`
  (not yet merged into `dead_code_cleanup`).

## MuJoCo versions measured (2026-11-08)

`mj_step` on the exported kitchen scenes, P-cores, in the `robocasa` image (3.3.1) and a
Python 3.12 venv (3.17.0, 4.0.0), 1,500 steps after 300 of warm-up, two runs each:

| | 3.3.1 (today) | 3.17.0 | 4.0.0 |
|---|---|---|---|
| H1-2 scene | 0.97 ms | 0.59 ms | 0.51 ms |
| G1 scene | 0.97 ms | 0.62 ms | 0.62 ms |
| 20k CPU lidar rays (BVH) | 76-99 ms | 80-109 ms | 89-117 ms |

Contact counts differ between 3.3.1 and the newer versions on the same state (H1-2: 27 vs
108), so the dynamics change. 4.0.0 (released 2026-11-06) drops Python 3.10 outright and
"can slightly change contacts" for convex and mesh collisions.

## Goals

### Tier 1 — no change to physics or the topic surface

1. **G1 viewer off the physics thread:** `viewer.sync()` runs on its own ~60 Hz thread
   from a copied `MjData`, as the H1-2 sim does.
2. **One torch thread in the leg node:** done by the leg merge (every policy spec
   `threads: 1`).
3. **Sim on the fast cores:** on a hybrid CPU the wrappers pin the `robocasa` container
   to the performance cores and the `ros` container to the rest (detected from
   `/sys/devices/cpu_core/cpus` and `/sys/devices/cpu_atom/cpus`); nothing is pinned on
   other CPUs or the Mac. An env var in `docker/.env` overrides or disables it.
4. **`rt/lowstate` fast path:** build the CDR bytes and CRC with numpy into a preallocated
   buffer and write them through cyclonedds without the Python IDL serializer. The bytes
   on the wire are identical to today's (a test compares them sample for sample), so the
   surface goldens do not change.
5. **Small main-thread savings:** the H1-2 snapshot copies only the geom and camera poses
   the sensors read (as the G1 does); the G1 encodes PNG/JPEG only when the topic has a
   subscriber.

### Tier 2 — one MID-360 model for both robots, real pattern, GPU rays

6. **One lidar module** in `golem_robocasa` (shared, like `unitree_common.py`) replaces
   `h12_sensors.py`'s grid and `g1/g1_lidar.py`'s approximation. Both robots publish what
   their real driver publishes today (topics, frames, message types unchanged).
7. **The real scan pattern:** MuJoCo-LiDAR's `mid360.npy` (MIT; vendored with its licence
   and attribution) fired in order at 200,000 points per second of sim time: 20,000
   points per 10 Hz frame on both robots, per-point `offset_time`, `line` = index mod 4,
   each batch cast from the sensor pose at its own sim time.
8. **GPU depth rendering on x86** (my call): the pattern's rays are read from depth
   images the GPU renders around the sensor (EGL, as the cameras already do), prototyped
   at 4.7-10.9 ms per 20k-ray scan against 90-250 ms on the CPU, median error 0.7-1.5 mm,
   no ray through geometry. It works on today's MuJoCo and Python, unlike
   `mujoco_warp` (which needs Python >= 3.11 and MuJoCo >= 3.13). The returns it loses
   (~2.5%) are expected inside the 0.1 m blind zone (goal 10); the builder checks that.
   Physics stays on CPU MuJoCo.
9. **CPU fallback (the Mac, or no CUDA):** the same pattern cast with exact
   `mj_multiRay` (BVH culling off) split across worker threads. Same points, slower.
10. **Datasheet effects** (the repo's model, my call): 0.1 m blind zone and 2-3 cm
    (1 sigma) range noise. Not adopted: reflectivity (geom colours are not real
    reflectivities), angular noise, false alarms, quantisation, IMU noise. Rays with no
    return are published as the real driver publishes them: the builder checks a
    recorded real-robot bag (`record_slam_bag` recordings) for zero points and matches it.
    `golem_slam`'s sim FAST-LIO settings (`config/g1/fastlio_sim.yaml`, the comments in
    `config/h12/fastlio.yaml`) may be retuned for the real point count and noise, with
    the SLAM checks of `tests/sim` and `tests/g1` nav as the gate; the real-robot values
    do not change.

### Tier 3

11. **(3a) Fixture-internal collisions off:** at scene build, collisions between parts of
    the same fixture, and between fixture parts and the static scene, are excluded.
    Robot-anything and object-anything contacts are unchanged, so objects inside
    cabinets and racks still rest on them. Measured -28% (dishwasher + drawers) to -52%
    (all jointed fixtures) of step time.
12. **(3b) MuJoCo 3.3.1 -> 3.17 or 4.0 in the robocasa image:** -36 to -47% step time
    measured (table above). The spike tries 4.0 first (my call: the faster H1-2 step, and
    it is the line upstream maintains) and falls back to 3.17 if 4.0 does not build or
    walk. Blocker to resolve first: MuJoCo >= 3.4 ships wheels for Python >= 3.11 only
    (4.0 removes 3.10 support from its sources too), and the
    sim runs Humble's rclpy and cyclonedds-python on Python 3.10. The first step is a
    spike: build the chosen version's Python bindings from source for Python 3.10 in a scratch
    image. If that fails, 3b stops and comes back to the user (the alternatives — a
    second physics process on Python 3.11, or a ROS distribution change — are larger
    projects). With the new version: robosuite's `mj_fullM` call patched, RoboCasa's pin overridden,
    the API changes absorbed (`mj_multiRay`'s `normal` argument, summed contact
    margin/gap, renamed camera fields, and for 4.0 the `mujoco-core`/`mujoco-studio`
    package split). The `ros` image keeps MuJoCo 3.2.3 (MJPC's ABI).
    The Mac image stays on 3.3.1 (no arm64 build can be verified on this host); the
    shared sim code absorbs the API differences in one small compatibility module keyed
    on `mujoco.__version__`, so the Mac keeps working with its older physics.
    **User checkpoint:** 3b lands only after the walking tests pass AND the user has
    watched both robots walk in the viewer on the new version and agreed. If walking gets worse,
    3b is dropped, not tuned: the G1's models stay byte-for-byte upstream.

## Order

Tier 1 -> Tier 2 -> 3a -> 3b (spike, then the user checkpoint). Each lands with its
closed-loop tests green before the next starts; 3b failing leaves the other three in.

## Non-goals

- Moving physics to MJX, MuJoCo Warp or Isaac.
- Changing sensor rates or resolutions (other than raising the G1's lidar to the real
  20,000 points).
- Ground truth anywhere (CLAUDE.md §8).

## Anti-goals

- A topic, frame, message type or rate that differs from the real robot's.
- A physics change hidden inside tiers 1 or 2.
- Mac breakage: the Mac may be slower, never broken.

## Done when

- The real-time factor of each robot in the full stack (`tests/sim`) and of the G1
  with the viewer is measured and reported after every tier. 0.95x is the target, not a
  gate: a tier that lands its goals is done even if a robot is still below it.
- `tests/robocasa` (goldens re-recorded only for the lidar change of tier 2),
  `tests/robocasa/g1`, `tests/sim --h12/--g1`, `tests/g1` walk and nav, `tests/stack`
  pass after every tier; after 3b also a manual look at the walking in the viewer.
- The Mac launch still builds its graph (CPU lidar path).

## Left to the builder

- How the GPU model copy is refreshed (every lidar batch vs every physics step).
- The exclusion mechanism for 3a (`<exclude>` pairs vs contype/conaffinity bits).
- Whether the camera-encode saving needs a subscriber count or a lazy publisher.
