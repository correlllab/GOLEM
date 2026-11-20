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

The spawn placement and margin search both robots build on (`robosuite_common.py`:
backing off, the sideways shifts, the give-up) are checked on a small MuJoCo scene by
`test_robosuite_common.py` and `test_h12_spawn_collision.py` (`spawn_scene.py` holds the
scene and a stand-in env), and through the G1's wrapper by `g1/test_g1_spawn_collision.py`.
That the H1-2's spawn clears `golem_upper_ik`'s start-up arm moves is kept by
`tests/docker/test_sim_startup_poses.py` (the sim's copy of them equals the controller's).

The experiment surface `GOLEM_SIM_EXPERIMENT=1` serves (`/sim/ground_truth`,
`/sim/reset_objects`; `measurement_bridge.py`), for grasp-experiment harnesses only,
is checked by `test_measurement_experiment.py`: off by default, and on a small scene
the ground truth it publishes and the object reset.

Pure pacing/timing tests also run without ROS or MuJoCo:

```bash
PYTHONPATH=golem_robocasa python3 -m unittest discover -s tests/robocasa -p test_loop_timing.py -v
PYTHONPATH=golem_robocasa python3 -m unittest discover -s tests/robocasa -p test_sim_rate.py -v
```

`test_sim_rate.py` covers `GOLEM_SIM_RATE` (the parse and the pacer at a target real-time
factor); its gripper tests, which check that the grippers' rates count sim time, need rclpy and
are skipped outside the container.

To measure the actual kitchen loop, read the sim's own log of a normal launch:

```bash
docker/scripts/docker_run.sh robocasa --headless --task OpenFridge --layout 1 --style 1 --seed 42
```

Every 30 s of sim time it logs `real-time factor ... over the last 30 s`, the share of
wall time the loop slept, and the sensors' counts: lidar frames, the casts that made the
physics wait and how far behind the sim a frame was at most, and on the G1 the head
camera's frames published and dropped. `GOLEM_SIM_RATE=max` gives the unpaced ceiling.
Both sims build that sample with numpy (`LowStateWriter` in `unitree_common.py`);
`test_lowstate_writer.py` checks its bytes against `unitree_sdk2py`'s serializer and
CRC, and that they take at most a third of the SDK path's time, printing both paths'
microseconds per sample.

Compare the same task/layout/style/seed, display mode, subscribers, and hardware.
Discard the first line for renderer warm-up. PD torque is recomputed every
physics step. RoboCasa fixture updates and success checks run once per control
step (`env.control_timestep`, 20 Hz), matching RoboCasa's own `env.step`.

Camera frames are rendered from a partial `MjData` snapshot on a worker
thread, overlapping physics, with one job in flight. A due job waits for the
previous one rather than being dropped, so every scheduled frame is published
(the H1-2's cameras; the G1's head camera drops a frame due while it renders). The MID-360's casts run on their own thread from a ring of
pose snapshots (`mid360.ScanRing`): the ring holds one frame of casts, and when
the thread falls that far behind the loop waits for it, so the real-time factor
drops instead of the lidar falling more than a frame behind.
The launcher selects EGL for offscreen camera rendering in both display modes;
the camera context is created before the passive viewer opens, because a new
EGL context cannot be made current once the GLFW window exists. The viewer
draws its own `MjData`, refreshed and synced at 60 Hz on a separate thread
(`viewer_sync.py`, shared by both robots), so
`handle.sync()` never blocks physics. Mouse perturbation forces from each sync
are applied to the simulation on every step. The thread also shows the measured
real-time factor (sim time over the last second of wall time), the
`GOLEM_SIM_RATE` target and the sim time in the viewer's top-left corner.
`test_viewer_sync.py` checks the thread against a fake handle.

The pacer repays at most 100 ms of wall-clock lag without dropping simulation
steps. It does not make an overloaded scene run in real time.
