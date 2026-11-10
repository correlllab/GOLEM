# Loop regression tests and timings

Run the tests using the existing Linux RoboCasa image, with no rebuild:

```bash
ROS_DOMAIN_ID=187 tests/robocasa/run_in_container.sh
```

The runner uses GOLEM's shared Docker preflight and a disposable container. It
leaves existing simulator containers alone. Tests use real MuJoCo EGL rendering
and ROS message types on a tiny fixed scene, without starting ROS nodes or DDS
participants. They cover pixel/payload equivalence, sensor scheduling, lock
boundaries, and pacing, plus how `--task` names resolve and where the viewer
frames the robot. The image must include the existing Livox messages.

The sims render through EGL on the NVIDIA GPU and stop at start without one;
`test_gpu_required.py` checks that refusal for both robots. Tests that render the
lidar's rays (`mid360.GpuCaster`) are skipped, naming why, without a GPU. The
scanner's, the scan ring's and the G1 mount's tests cast through `exact_rays.py`
(MuJoCo's exact `mj_ray`, the reference the GPU caster is checked against)
and fakes, so they need none.

The RealSense profile tests (camera placement, projection against
camera_info, TF tree, point cloud) need only MuJoCo, NumPy and PyYAML; the
projection check renders, so give it an offscreen GL context:

```bash
MUJOCO_GL=egl PYTHONPATH=golem_robocasa python3 -m unittest discover -s tests/robocasa -p test_realsense_profile.py -v
```

The spawn placement both robots build on (`robosuite_common.py`) and the H1-2's search
for its margin to the fixtures (`h1_2_robosuite.place_robot_collision_free`) are checked
on a small MuJoCo scene by `test_robosuite_common.py` and `test_h12_spawn_collision.py`
(`spawn_scene.py` holds the scene and a stand-in env); the G1's search is
`g1/test_g1_spawn_collision.py`'s.

Pure pacing/timing tests also run without ROS or MuJoCo:

```bash
PYTHONPATH=golem_robocasa python3 -m unittest discover -s tests/robocasa -p test_loop_timing.py -v
PYTHONPATH=golem_robocasa python3 -m unittest discover -s tests/robocasa -p test_sim_rate.py -v
```

`test_sim_rate.py` covers `GOLEM_SIM_RATE` (the parse and the pacer at a target real-time
factor); its gripper tests, which check that the grippers' rates count sim time, need rclpy and
are skipped outside the container.

To measure the actual kitchen loop, add `--timing` to a normal launch:

```bash
docker/scripts/docker_run.sh robocasa --headless --task OpenFridge --layout 1 --style 1 --seed 42 --timing
```

Every five wall-clock seconds, `[sim-timing]` reports simulation/wall-clock time
(RTF: 1 means real time) and mean/max milliseconds per stage call, with call
counts. `camera_render` includes scene capture and both render passes per
camera (with the wait for `mid360.GL_LOCK`, which the cameras and the lidar's
GPU caster take turns on); `camera_depth_render` the depth stream's render;
`camera_publish` image conversion, compression, and publishing; `camera_cloud`
the point cloud. `lidar_capture` is the physics thread's copy of the poses one
lidar cast reads, under the sim lock; `lidar_wait` the physics thread waiting
for a free slot when the lidar's thread is a frame behind; `lidar_cast` one
cast of the rays due since the last (every caster's `batch_s` of sim time, on
the lidar's thread); `lidar_publish` packing and publishing a frame on both
lidar topics. The line ends with the lidar's counts: frames, the casts that
waited, and how far behind the sim a frame was at most. Physics includes lock
acquisition and PD control; `lowstate` is the `rt/lowstate` sample built,
signed and published after every physics step. Timings are silent unless requested.
Both sims build that sample with numpy (`LowStateWriter` in `unitree_common.py`);
`test_lowstate_writer.py` checks its bytes against `unitree_sdk2py`'s serializer and
CRC, and that they take at most a third of the SDK path's time, printing both paths'
microseconds per sample.

Compare the same task/layout/style/seed, display mode, subscribers, and hardware.
Discard the first interval for renderer warm-up. PD torque is recomputed every
physics step. RoboCasa fixture updates and success checks run once per control
step (`env.control_timestep`, 20 Hz), matching RoboCasa's own `env.step`.

Camera frames are rendered from a partial `MjData` snapshot on a worker
thread, overlapping physics, with one job in flight. A due job waits for the
previous one rather than being dropped, so every scheduled frame is published.
`camera_capture` is the locked snapshot, `camera_wait` is time the loop blocked
on the previous job. The MID-360's casts run on their own thread from a ring of
pose snapshots (`mid360.ScanRing`): the ring holds one frame of casts, and when
the thread falls that far behind the loop waits for it, so the real-time factor
drops instead of the lidar falling more than a frame behind.
The launcher selects EGL for offscreen camera rendering in both display modes;
the camera context is created before the passive viewer opens, because a new
EGL context cannot be made current once the GLFW window exists. The viewer
draws its own `MjData`, refreshed and synced at 60 Hz on a separate thread
(`viewer_sync.py`, shared by both sims; the H1-2's `viewer` stage), so
`handle.sync()` never blocks physics. Mouse perturbation forces from each sync
are applied to the simulation on every step. `test_viewer_sync.py` checks the
thread against a fake handle.

The pacer repays at most 100 ms of wall-clock lag without dropping simulation
steps. It does not make an overloaded scene run in real time.
