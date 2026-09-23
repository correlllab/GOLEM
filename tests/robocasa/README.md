# Loop regression tests and timings

Run the tests using the existing Linux RoboCasa image, with no rebuild:

```bash
ROS_DOMAIN_ID=187 tests/robocasa/run_in_container.sh
```

The runner uses GOLEM's shared Docker preflight and a disposable container. It
leaves existing simulator containers alone. Tests use real MuJoCo EGL rendering
and ROS message types on a tiny fixed scene, without starting ROS nodes or DDS
participants. They cover pixel/payload equivalence, sensor scheduling, lock
boundaries, and pacing. The image must include the existing Livox messages.

Pure pacing/timing tests also run without ROS or MuJoCo:

```bash
PYTHONPATH=h1_robocasa python3 -m unittest discover -s tests/robocasa -p test_loop_timing.py -v
```

To measure the actual kitchen loop, add `--timing` to a normal launch:

```bash
docker/scripts/docker_run.sh robocasa --headless --task OpenFridge --layout 1 --style 1 --seed 42 --timing
```

Every five wall-clock seconds, `[sim-timing]` reports simulation/wall-clock time
(RTF: 1 means real time) and mean/max milliseconds per stage call, with call
counts. `camera_render` includes scene capture and both render passes per
camera; `camera_publish` includes image conversion, compression, and publishing.
`lidar_cast` includes scene capture and raycasting; `lidar_publish` includes
point filtering, message construction, and publishing. Physics includes lock
acquisition and PD control. Timings are silent unless requested.

Compare the same task/layout/style/seed, display mode, subscribers, and hardware.
Discard the first interval for renderer warm-up. PD torque is recomputed every
physics step. RoboCasa fixture updates and success checks run once per control
step (`env.control_timestep`, 20 Hz), matching RoboCasa's own `env.step`.

Lidar scans and camera frames are produced from a ~1 ms `MjData` snapshot on
one worker thread per sensor, overlapping physics, with one job in flight per
sensor. A due job waits for the previous one rather than being dropped, so
schedules, stamps, message formats and topics are unchanged. `*_capture` is the
locked snapshot, `*_wait` is time the loop blocked on the previous job.
The launcher selects EGL for offscreen camera rendering in both display modes;
the camera context is created before the passive viewer opens, because a new
EGL context cannot be made current once the GLFW window exists. The viewer
draws its own `MjData`, refreshed and synced at 60 Hz on a separate thread
(`viewer` stage), so `handle.sync()` never blocks physics. Mouse perturbation
forces from each sync are applied to the simulation on every step.

The pacer repays at most 100 ms of wall-clock lag without dropping simulation
steps. It does not make an overloaded scene run in real time.
