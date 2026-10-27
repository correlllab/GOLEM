import sys

import sim_common

# The G1 runs from this entry point too. It is dispatched before the H1-2 imports below,
# which need magpie_msgs and CL_Assets; the G1 needs neither, and its test containers
# provide neither.
if __name__ == "__main__" and sim_common.select_robot(sys.argv[1:]) == "g1":
    from g1 import g1_mujoco
    sys.exit(g1_mujoco.main(sys.argv[1:]))

import argparse
import os
import threading
import time

import mujoco
import numpy as np
import rclpy.executors

from mujoco_ros_bridge import init_ros, shutdown_ros
import mujoco.viewer

import h1_2_robosuite  # registers H1_2 + Magpie grippers, patches RoboCasa
import realsense_profile
import battery_env  # registers the Battery workcell env
from h12_sensors import H12Sensors
from magpie_hand_bridge import MagpieHandBridge
from measurement_bridge import MeasurementBridge
from mujoco_env import ElasticBand
from mujoco_ros_bridge import RosSensorBridge
from sim_names import NameResolver
from sim_timing import RealtimePacer, LoopTimings
from unitree_interface import SimInterface

NUM_LEG_JOINTS = 12
NUM_MOTORS = 27
INIT_LEG_POS = np.array([
    0.0, -0.16, 0.0, 0.36, -0.2, 0.0,
    0.0, -0.16, 0.0, 0.36, -0.2, 0.0,
], dtype=float)
INIT_ARM_POS = np.array([
    0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=float)


def _initial_motor_qpos():
    """Build the 27-motor spawn pose without depending on controller configs."""
    qpos = np.concatenate([INIT_LEG_POS, INIT_ARM_POS])
    if qpos.shape != (NUM_MOTORS,):
        raise ValueError(f"initial motor pose must contain {NUM_MOTORS} entries")
    return qpos


# Motor indices (ROS order): 12 torso, 13-19 left arm, 20-26 right arm, each arm
# shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw.
_TORSO, _ARMS = 12, (13, 20)


def _limp_upper_body_poses(model, motor_qpos, spawn):
    """Poses the unpowered upper body settles into before the arm controller
    first commands it, as (qpos indices, values) pairs for the spawn clearance
    check. Measured in RoboCasa: the torso twists up to ~0.7 rad (either way),
    shoulders pitch back ~0.25 rad, elbows bend ~0.45 rad and the wrists drop
    onto their pitch stops, swinging the grippers forward."""
    poses = []
    for twist in (0.75, -0.75):
        q = np.array(spawn, dtype=float)
        q[_TORSO] = twist
        for base in _ARMS:
            q[base + 0] = 0.25    # shoulder pitch (rad, + is backward)
            q[base + 2] = -0.2    # shoulder yaw
            q[base + 3] = 0.45    # elbow
            wrist_pitch = int(np.flatnonzero(model.jnt_qposadr == motor_qpos[base + 5])[0])
            q[base + 5] = model.jnt_range[wrist_pitch, 0]
        poses.append((motor_qpos, q))
    return poses


# Viewer display rate. A viewer thread syncs its own MjData at this rate so the
# ~11 ms handle.sync() (it waits for the render thread) never blocks physics.
VIEWER_HZ = 60.0


def _copy_display_state(dst, src):
    """Copy the state the viewer draws from src (caller holds sim_lock)."""
    dst.time = src.time
    dst.qpos[:] = src.qpos
    dst.qvel[:] = src.qvel
    dst.act[:] = src.act
    dst.mocap_pos[:] = src.mocap_pos
    dst.mocap_quat[:] = src.mocap_quat


def _draw_band(handle, anchor, body_pos):
    """Draw the elastic band (anchor sphere + capsule to the torso) in the passive
    viewer's user scene. Mirrors the legacy MujocoEnv.draw_elastic_band."""
    scn = handle.user_scn
    scn.ngeom = 0
    color = np.array([0.5, 0.6, 1.0, 0.8])
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.02, 0.0, 0.0]), anchor, np.eye(3).flatten(), color)
        scn.ngeom += 1
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.eye(3).flatten(), color)
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                             0.01, anchor, body_pos)
        scn.ngeom += 1


def sim_loop(task, viewer=True, layout=None, style=None, seed=None, timing=False, duration=0.0):
    """Launch a RoboCasa task env around the H1-2 and run the shared-MjData loop.

    Builds the env with robots='H1_2' and steps the env's *own* MjData with our
    loop (never env.step()), driving the full sim<->ROS bridge layer:
      - SimInterface     : DDS rt/lowstate + rt/lowcmd (27 body motors, name-resolved)
      - RosSensorBridge  : /clock, and H12Sensors: RGBD cameras, livox lidar + IMU
      - MagpieHandBridge : /{left,right}/gripper/* (x2)
      - MeasurementBridge: /robocasa/{task_goal,success,reward} + /elastic_band/toggle
    An elastic-band tether holds the robot upright until the ROS walking policy
    drives it; RoboCasa's _check_success/reward/lang are read off the shared env.
    duration is part of the signature both runners share; this loop runs until the
    viewer closes or Ctrl-C, and the command line rejects a non-zero --duration.
    """

    create_kwargs = {}
    if layout is not None:
        create_kwargs["layout_ids"] = layout
    if style is not None:
        create_kwargs["style_ids"] = style
    if seed is not None:
        create_kwargs["seed"] = seed

    env = h1_2_robosuite.make_kitchen_env(task, **create_kwargs)

    # robosuite.make() constructs but does NOT reset; reset() builds env.sim, runs
    # _load_model + _reset_internal (placement patch fires here, sets ep_meta /
    # init_robot_base_pos), and places the task objects.
    env.reset()

    model = env.sim.model._model
    data = env.sim.data._data
    resolver = NameResolver(model)  # ROS<->sim name map for the DDS / sensor bridges

    # Clean, contact-free initial state. RoboCasa's zero-action settling loop
    # perturbs the pose during reset, so restore a baked-in bent-knee stance,
    # zero all velocities, and re-place the pelvis. Upper-body motors remain at
    # zero. place_robot_collision_free auto-fits floor clearance AND backs the
    # base away (the robot's -x) until every robot geom is GOLEM_SPAWN_MARGIN
    # (unset: the env's spawn_margin, else 0.20 m) from any non-floor geom, in
    # the spawn pose and in the sagged pose the unpowered upper body falls into,
    # keeping the widest-gap spot if it can't reach that.
    try:
        init_qpos = _initial_motor_qpos()
        data.qpos[resolver.motor_qpos] = init_qpos
        data.qvel[:] = 0.0
        # GOLEM_SPAWN_BACKOFF backs the robot further off the counter into open
        # floor (default 0 = at the counter for manipulation). Set ~1.0 for nav2,
        # which needs the robot's start cell to be free to plan.
        _spawn_backoff = float(os.environ.get("GOLEM_SPAWN_BACKOFF", "0") or 0)
        _spawn_margin = float(os.environ.get("GOLEM_SPAWN_MARGIN", "").strip()
                              or getattr(env, "spawn_margin", 0.20))
        # GOLEM_SPAWN_OFFSET="dx,dy" (m, world) shifts the task's spawn before
        # placement, e.g. to stand the robot in front of a different part.
        # Unset = the task's own spawn.
        _spawn_pos = list(env.init_robot_base_pos)
        _offset = os.environ.get("GOLEM_SPAWN_OFFSET", "").strip()
        if _offset:
            _dx, _dy = (float(v) for v in _offset.split(","))
            _spawn_pos[0] += _dx
            _spawn_pos[1] += _dy
            print(f"[h12_mujoco] spawn offset ({_dx:+.3f}, {_dy:+.3f}) m")
        h1_2_robosuite.place_robot_collision_free(
            env, _spawn_pos,
            h1_2_robosuite._euler_to_wxyz(getattr(env, "init_robot_base_ori", None)),
            extra_backoff=_spawn_backoff,
            margin=_spawn_margin,
            # Motors stay unpowered until the first command, so the margin must
            # also hold once the torso and arms sag.
            poses=_limp_upper_body_poses(model, resolver.motor_qpos, init_qpos),
        )
        print("[h12_mujoco] initial stance from baked-in sim defaults")
    except Exception as e:
        print(f"[h12_mujoco] clean reset skipped: {e}")

    # Elastic-band balance tether on the torso: holds the free-floating biped
    # upright until the ROS walking policy takes over. Anchor a fixed point above
    # the torso's spawn; apply the spring force each step (SPACE / the
    # /elastic_band/toggle service disables it).
    band = None
    band_body_id = -1
    rot_body_id = -1
    band_vel = np.zeros(6)
    rot_vel = np.zeros(6)
    try:
        band_body_id = resolver.body_id("torso_link")            # robot0_torso_link
        anchor = data.xpos[band_body_id].copy()
        anchor[2] += 0.01
        # Stiff enough to hold the LIMP robot near-upright through the lowerbody
        # controller's band-held idle phase (~20s of frame_task arm-home startup
        # before FAME engages); at stiffness=1e3 it pitched forward to ~60deg and
        # FAME inherited a toppling robot. Static tilt ~ gravity_torque/stiffness.
        # damping=2e3 N*s/m damps a lateral shove out in well under a second.
        # GOLEM_SIM_TETHER_ROT_STIFFNESS (N*m/rad, default 0 = off) adds a
        # rotational hold at the spawn orientation while the band is on: a rigid
        # hoist fixture for tethered manipulation experiments.
        # GOLEM_SIM_TETHER_ROT_BODY picks the held body: torso (default) or
        # pelvis — the pelvis hold also stops the legs spinning the pelvis about
        # the waist while a policy steps with the feet off the ground. The torque
        # is applied explicitly, so damping must keep d*dt/I well under 2 at the
        # 2 ms step: torso yaw I ~0.13 kg*m^2 -> 0.04 s x stiffness; pelvis
        # ~0.009 kg*m^2 -> 0.006 s x stiffness.
        rot_k = float(os.environ.get("GOLEM_SIM_TETHER_ROT_STIFFNESS", "0") or 0)
        rot_body = os.environ.get("GOLEM_SIM_TETHER_ROT_BODY", "torso").strip() or "torso"
        rot_body_id = (resolver.body_id("pelvis") if rot_body == "pelvis" else band_body_id)
        rot_d = (0.006 if rot_body == "pelvis" else 0.04) * rot_k
        band = ElasticBand(anchor, length=0, stiffness=1e4, damping=2e3,
                           orientation=data.xquat[rot_body_id].copy(),
                           rot_stiffness=rot_k, rot_damping=rot_d)
        print(f"[h12_mujoco] elastic band on torso anchored at {anchor.round(3)}"
              + (f", rotational hold {rot_k:g} N*m/rad on the {rot_body}" if rot_k > 0 else ""))
    except Exception as e:
        print(f"[h12_mujoco] elastic band setup skipped: {e}")

    print(f"[h12_mujoco] RoboCasa task {task!r} built around H1_2: "
          f"nq={model.nq} nu={model.nu}; resolver mapped {len(resolver.motor_ctrl)} motors")

    # Shared lock: the DDS + ROS bridge threads touch MjData (read sensors, write
    # ctrl, render) while the main loop runs mj_step. All MjData access is
    # serialized through sim_lock.
    sim_lock = threading.Lock()
    pfx = env.robots[0].robot_model.naming_prefix   # "robot0_"

    # Body carrying the pelvis free joint — its world pose is ground-truth base
    # odometry, which the bridge publishes as odom -> pelvis TF + /odom (feeds
    # SLAM without needing FAST-LIO). OFF by default so the shared sim bridge does
    # not inject a second odom->pelvis anchor that competes with the real stack's
    # FAST-LIO (camera_init) TF; the Mac nav/SLAM demo opts in with GOLEM_SIM_ODOM=1.
    odom_base_body_id = -1
    _sim_odom_on = os.environ.get('GOLEM_SIM_ODOM', '0').strip().lower() not in ('0', 'off', 'false', 'no', '')
    if _sim_odom_on:
        try:
            _fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{pfx}{h1_2_robosuite.FREE_JOINT_NAME}")
            odom_base_body_id = int(model.jnt_bodyid[_fj])
        except Exception as e:
            print(f"[h12_mujoco] base odom body resolve skipped: {e}")
    print(f"[h12_mujoco] ground-truth sim odom {'ON (GOLEM_SIM_ODOM=1)' if _sim_odom_on else 'OFF'}")

    # DDS control bridge: publishes rt/lowstate, subscribes rt/lowcmd, drives the
    # 27 body motors by name via the resolver (grippers handled by the hand
    # bridges below; gripper ctrl indices are disjoint from the body motors).
    sim_interface = SimInterface(model, data, lock=sim_lock, resolver=resolver)  # noqa: F841

    init_ros()

    # Sensor bridge: /clock, RGBD cameras (head + both hands), livox lidar + IMU.
    # MuJoCo lookups use robosuite-prefixed names (the head camera rides the robot
    # prefix, the eye-in-hand cameras the gripper prefixes, as the hand bridges
    # below); ROS-facing names are the real robot's. Camera rendering and encoding
    # is the heaviest per-step cost on CPU; GOLEM_CAMERAS=0 drops the cameras to
    # speed the sim up for locomotion/SLAM (lidar + odom are unaffected).
    # The cameras publish what the real robot's RealSense drivers do
    # (realsense_profile.yaml): frame names, TF, intrinsics and topics.
    # GOLEM_CAMERA_SCALE renders the head camera at that fraction of its real
    # 1280x720 (same projection, camera_info scaled to match). The hand cameras
    # keep their gripper-MJCF fovy at GOLEM_HAND_CAMERA_RES=<W>x<H> (default
    # 256x256; 640x480 is the D405 colour stream). GOLEM_CAMERA_DEPTH=0 drops
    # depth/image_rect_raw and GOLEM_CAMERA_POINTCLOUD=0 drops depth/color/points
    # (with both off, the extra depth render is skipped).
    def _env_on(name):
        return os.environ.get(name, '1').strip().lower() not in ('0', 'off', 'false', 'no')
    _cameras_on = _env_on('GOLEM_CAMERAS')
    _camera_scale = float(os.environ.get('GOLEM_CAMERA_SCALE', '').strip() or 0.5)
    _profile = realsense_profile.load_profile()
    _camera_names = [
        ("head",       f"{pfx}{h1_2_robosuite.HEAD_CAMERA}"),
        ("left_hand",  "gripper0_left_hand_cam"),
        ("right_hand", "gripper0_right_hand_cam"),
    ] if _cameras_on else []
    _hand_res = tuple(int(v) for v in
                      (os.environ.get('GOLEM_HAND_CAMERA_RES', '').strip() or '256x256').lower().split('x'))
    _cameras = realsense_profile.camera_specs(model, _camera_names, _profile, _camera_scale, _hand_res)
    _depth_on, _cloud_on = _env_on('GOLEM_CAMERA_DEPTH'), _env_on('GOLEM_CAMERA_POINTCLOUD')
    print(f"[h12_mujoco] RGBD cameras {'ON' if _cameras_on else 'OFF (GOLEM_CAMERAS=0)'}"
          + (f": head {_cameras[0].color.width}x{_cameras[0].color.height} "
             f"(GOLEM_CAMERA_SCALE={_camera_scale:g}), hands {_hand_res[0]}x{_hand_res[1]}, "
             f"depth stream {'on' if _depth_on else 'off'}, "
             f"point cloud {'on' if _cloud_on else 'off'}" if _cameras else ""))
    timings = LoopTimings(enabled=timing)
    ros_bridge = RosSensorBridge(
        model, data,
        lambda node: H12Sensors(
            node, model, data,
            cameras=_cameras,
            camera_tf=realsense_profile.tf_edges(_profile, [ns for ns, _ in _camera_names]),
            camera_tf_rate_hz=float(_profile["tf_rate_hz"]),
            depth_stream=_depth_on,
            pointcloud=_cloud_on,
            # MID-360 fidelity: 360x56 @ 10Hz ~= 201k pts/s (real ~200k), 0.1m near /
            # 40m far range, per-point offset_time for FAST-LIO deskew. el_rays/rate
            # are the knobs to dial back if the ray cast (main thread) hurts RTF.
            lidar_az_rays=360, lidar_el_rays=56,
            lidar_rate_hz=10.0, lidar_max_range=40.0, lidar_min_range=0.1,
            # 200 Hz IMU matches the real MID-360 (ICM-40609); the bridge default is
            # 100 Hz. FAST-LIO integrates the IMU to propagate between the 10 Hz lidar
            # scans, and the walking gait's high-frequency accelerations alias badly
            # at 100 Hz -> pose drift/divergence. 200 Hz gives cleaner propagation
            # through the gait wobble. (IMU sensor reads are cheap vs the ray cast.)
            imu_rate_hz=200.0,
            lidar_body=f"{pfx}livox_link",
            lidar_exclude_body=f"{pfx}torso_link",
            imu_quat_sensor=f"{pfx}livox_imu_quat",
            imu_gyro_sensor=f"{pfx}livox_imu_gyro",
            imu_acc_sensor=f"{pfx}livox_imu_acc",
            # Drop the robot's own legs/arms from the lidar (self-returns otherwise map
            # a phantom obstacle under the robot and nav2 rejects the start). Default on
            # for this locomotion/SLAM/nav sim; GOLEM_LIDAR_SELF_FILTER=0 restores them
            # as occluders.
            lidar_self_prefixes=(
                (pfx, "gripper0_")
                if os.environ.get("GOLEM_LIDAR_SELF_FILTER", "1").strip().lower()
                not in ("0", "off", "false", "no")
                else ()
            ),
            sim_lock=sim_lock,
            timings=timings,
        ),
        base_body_id=odom_base_body_id,   # -> odom -> pelvis TF + /odom
        sim_lock=sim_lock,
    )

    # Gripper bridges (gripper0_<side>_ prefixed actuators/sensors).
    hand_right = MagpieHandBridge(model, data, side="right", sim_lock=sim_lock,
                                  name_prefix="gripper0_right_")
    hand_left = MagpieHandBridge(model, data, side="left", sim_lock=sim_lock,
                                 name_prefix="gripper0_left_")

    measurement = MeasurementBridge(env, elastic_band=band, task_name=task)
    print(f"[h12_mujoco] task goal: {measurement.publish_goal()!r}")

    # Background executor serves the gripper services/timers/action + the band
    # toggle service. ros_bridge.tick() is driven from the main loop instead (its
    # MuJoCo renderer context is thread-affine), so it is NOT added here.
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in (measurement, hand_right, hand_left):
        executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True, name="ros_bridge_exec")
    executor_thread.start()
    print("[h12_mujoco] ROS bridges up: rt/lowstate+rt/lowcmd, /clock, "
          "RGBD /realsense/{head,left_hand,right_hand}, livox lidar+imu, "
          "/{left,right}/gripper/*, /elastic_band/toggle")

    # The viewer draws its own MjData, refreshed from `data` by the viewer
    # thread below; mouse perturbation forces flow back through viewer_perturb.
    viewer_data = mujoco.MjData(model) if viewer else None
    if viewer_data is not None:
        with sim_lock:
            _copy_display_state(viewer_data, data)
        mujoco.mj_forward(model, viewer_data)
    # SPACE in the viewer toggles the band (ElasticBand.key_callback).
    handle = (
        mujoco.viewer.launch_passive(model, viewer_data, key_callback=band.key_callback)
        if viewer and band is not None
        else (mujoco.viewer.launch_passive(model, viewer_data) if viewer else None)
    )
    if handle is not None:
        handle.opt.geomgroup[0] = 0   # hide collision geoms by default
        # Frame the free camera on the robot's spawn pose (function of robot
        # position + yaw). band_body_id is the torso; if the band setup failed,
        # resolve the torso directly so framing still works.
        cam_body_id = band_body_id
        if cam_body_id < 0:
            try:
                cam_body_id = resolver.body_id("torso_link")
            except Exception:
                cam_body_id = -1
        if cam_body_id >= 0:
            try:
                with handle.lock():
                    sim_common.frame_viewer_on_robot(handle, viewer_data, cam_body_id)
            except Exception as e:
                print(f"[h12_mujoco] viewer camera framing skipped: {e}")
    # Fall logger: watch the torso's uprightness + height and log once when the
    # robot falls, including how long it stood after the elastic band was released.
    _fall_logged = False
    _band_release_t = None
    _band_prev_on = bool(band.enabled) if band is not None else False

    # RoboCasa advances fixture state and checks success once per control step
    # (env._post_action at control_freq, 20 Hz by default), not per physics step.
    # Several fixtures count calls (toaster heat, scrub contact), and the success
    # debounce is in control steps, so matching that cadence keeps task semantics.
    task_every = max(1, round(env.control_timestep / model.opt.timestep))
    step_count = 0
    viewer_perturb = np.zeros_like(data.xfrc_applied)
    viewer_stop = threading.Event()

    def _viewer_loop():
        period = 1.0 / VIEWER_HZ
        while not viewer_stop.is_set() and handle.is_running():
            frame_start = time.perf_counter()
            viewer_start = timings.start()
            with sim_lock:
                _copy_display_state(viewer_data, data)
            mujoco.mj_forward(model, viewer_data)
            # sync() applies the current mouse perturbation to viewer_data.
            viewer_data.xfrc_applied[:] = 0.0
            if band is not None and band.enabled:
                _draw_band(handle, band.point, viewer_data.xpos[band_body_id])
            else:
                handle.user_scn.ngeom = 0   # clear overlay when band off
            handle.sync()
            with sim_lock:
                viewer_perturb[:] = viewer_data.xfrc_applied
            timings.stop("viewer", viewer_start)
            viewer_stop.wait(max(0.0, period - (time.perf_counter() - frame_start)))

    viewer_thread = None
    if handle is not None:
        viewer_thread = threading.Thread(target=_viewer_loop, daemon=True, name="viewer_sync")
        viewer_thread.start()

    timings.reset(float(data.time))
    pacer = RealtimePacer(model.opt.timestep)
    try:
        while True:
            if viewer and not handle.is_running():
                break
            physics_start = timings.start()
            with sim_lock:
                if viewer_data is not None:
                    # Mouse perturbation from the viewer's latest sync (zero when idle).
                    data.xfrc_applied[:] = viewer_perturb
                if band is not None:
                    # Re-write every step; MuJoCo persists xfrc_applied, so we must
                    # zero it when the band is disabled or the last force lingers.
                    if band.enabled:
                        # [ang(3), lin(3)] of the torso itself, world frame
                        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                                 band_body_id, band_vel, 0)
                        data.xfrc_applied[band_body_id, :3] = band.evaluate_force(
                            data.xpos[band_body_id], band_vel[3:])
                        if band.rot_stiffness > 0.0:
                            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                                     rot_body_id, rot_vel, 0)
                            data.xfrc_applied[rot_body_id, 3:] = band.evaluate_torque(
                                data.xquat[rot_body_id], rot_vel[:3])
                    else:
                        data.xfrc_applied[band_body_id, :] = 0.0
                        if rot_body_id >= 0:
                            data.xfrc_applied[rot_body_id, :] = 0.0
                if sim_interface is not None:
                    # Recompute the DDS command's PD torque every sim step (emulates
                    # the real motor's ~1kHz servo) instead of only on each 50Hz
                    # /lowcmd, which held a stale-damping torque and toppled FAME.
                    sim_interface.write_ctrl()
                mujoco.mj_step(model, data)
                timings.stop("physics", physics_start)
                # --- fall logger ---
                if band is not None and _band_prev_on and not band.enabled:
                    _band_release_t = float(data.time)
                    _band_prev_on = False
                    print(f"[fall-logger] elastic band released at t={data.time:.2f}s", flush=True)
                if band_body_id >= 0 and not _fall_logged:
                    _upz = float(data.xmat[band_body_id][8])   # torso z-axis . world up (1=upright)
                    _torso_h = float(data.xpos[band_body_id][2])
                    if _upz < 0.5 or _torso_h < 0.5:
                        _fall_logged = True
                        _since = (f"{data.time - _band_release_t:.2f}s after band release"
                                  if _band_release_t is not None
                                  else f"{data.time:.2f}s (band still on)")
                        print(f"[fall-logger] *** ROBOT FELL at sim t={data.time:.2f}s — stood "
                              f"{_since} (uprightness={_upz:.2f}, torso_h={_torso_h:.2f}) ***",
                              flush=True)
                step_count += 1
                task_step = step_count % task_every == 0
                if task_step:
                    task_start = timings.start()
                    env.update_state()              # REQUIRED before _check_success
                    timings.stop("task_update", task_start)
            # tick owns short sensor capture locks; encoding/publication cannot
            # block the DDS state reader. No physics step occurs inside tick.
            ros_bridge.tick()                       # /clock + camera + lidar + imu
            if task_step:
                task_start = timings.start()
                with sim_lock:
                    done = measurement.tick()
                timings.stop("task_check", task_start)
                if done:
                    print("[h12_mujoco] task success (debounced).")
            pacer.wait()
            timings.maybe_report(float(data.time))
    finally:
        viewer_stop.set()
        if viewer_thread is not None:
            viewer_thread.join()
        if handle is not None:
            handle.close()
        executor.shutdown()
        ros_bridge.shutdown()
        for node in (measurement, hand_right, hand_left, ros_bridge):
            node.destroy_node()
        shutdown_ros()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the H1-2 RoboCasa MuJoCo sim")
    sim_common.add_common_args(
        parser,
        task_help="RoboCasa env name (e.g. TurnOnToasterOven), or Battery for the "
                  "battery workcell (CL_Assets/mujoco_assets/battery). "
                  "Omit to launch a random registered kitchen task.",
    )
    args = sim_common.parse_args(parser, sys.argv[1:], "h12")

    task = args.task
    if task is None:
        task = sim_common.random_task(seed=args.seed)
        print(f"[h12_mujoco] no --task given; randomly selected {task!r}")
    else:
        task = sim_common.resolve_task(task, seed=args.seed, tag="h12_mujoco",
                                       robot_task=battery_env.canonical_task)

    sim_loop(task, viewer=not args.headless,
             layout=args.layout, style=args.style, seed=args.seed, timing=args.timing,
             duration=args.duration)
