"""Run the RoboCasa MuJoCo sim of the H1-2, or of the G1 with --g1 or GOLEM_ROBOT=g1.

Usage: h12_mujoco.py [--h12 | --g1] [--headless] [--task NAME] [--layout N]
[--style N] [--seed N] [--timing | --duration S]. The robot is chosen
(sim_common.select_robot) before the H1-2's imports, and --g1 hands the whole
command line to g1.g1_mujoco.main. The H1-2's loop is sim_loop below.
"""
import sys

import sim_common

# The G1 is dispatched before the H1-2 imports below, which need magpie_msgs and
# the H1-2's CL_Assets files: the G1 reads only CL_Assets/g1, and its test
# containers do not provide magpie_msgs.
if __name__ == "__main__" and sim_common.select_robot(sys.argv[1:]) == "g1":
    from g1 import g1_mujoco
    sys.exit(g1_mujoco.main(sys.argv[1:]))

import argparse
import os
import threading

import mujoco
import numpy as np
import rclpy.executors

from mujoco_ros_bridge import init_ros, shutdown_ros
import mujoco.viewer

import h1_2_robosuite  # registers H1_2 and the MAGPIE grippers
import realsense_profile
import battery_env  # registers the Battery env
import blank_env  # registers the Blank env
from h12_sensors import H12Sensors
from magpie_hand_bridge import MagpieHandBridge
from measurement_bridge import MeasurementBridge
from mujoco_env import ElasticBand
from mujoco_ros_bridge import RosSensorBridge
from sim_names import NameResolver
from sim_timing import RealtimePacer, LoopTimings
from unitree_interface import SimInterface
from viewer_sync import ViewerSync, band_overlay, copy_display_state

NUM_MOTORS = 27
# The spawn pose (rad) in sim_names.ROS_MOTOR_ORDER: bent-knee legs, then the
# torso and both arms at 0.
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
    """Return the 27-motor spawn pose (rad), without reading any controller config."""
    qpos = np.concatenate([INIT_LEG_POS, INIT_ARM_POS])
    if qpos.shape != (NUM_MOTORS,):
        raise ValueError(f"initial motor pose must contain {NUM_MOTORS} entries")
    return qpos


# Slots in ROS_MOTOR_ORDER: 12 torso, 13-19 left arm, 20-26 right arm, each arm
# shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw.
_TORSO, _ARMS = 12, (13, 20)


def _limp_upper_body_poses(model, motor_qpos, spawn):
    """Return the poses the unpowered upper body sags into, as (qpos indices, values) pairs.

    The spawn clearance must hold in them too, since every motor is off until
    the first command. Measured in RoboCasa: the torso twists up to ~0.7 rad
    either way, the shoulders pitch back ~0.25 rad, the elbows bend ~0.45 rad
    and the wrists drop onto their pitch stops, swinging the grippers forward.
    """
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


def sim_loop(task, viewer=True, layout=None, style=None, seed=None, timing=False, duration=0.0):
    """Run the H1-2 in task (a RoboCasa kitchen, Battery or Blank) until the viewer closes or Ctrl-C.

    The loop steps the env's own MjData (never env.step()) and drives:
      - SimInterface     : Unitree DDS rt/lowstate + rt/lowcmd (27 body motors)
      - RosSensorBridge  : /clock, opt-in odom, and H12Sensors: RGBD cameras,
                           MID-360 lidar + IMU
      - MagpieHandBridge : /{left,right}/gripper/*
      - MeasurementBridge: /robocasa/*, /elastic_band/toggle
    An elastic band holds the robot until it is released. timing enables the
    --timing stage report. duration is in the signature both runners share and
    is unused here: the command line refuses --duration for the H1-2.
    """

    create_kwargs = {}
    if layout is not None:
        create_kwargs["layout_ids"] = layout
    if style is not None:
        create_kwargs["style_ids"] = style
    if seed is not None:
        create_kwargs["seed"] = seed

    env = h1_2_robosuite.make_kitchen_env(task, **create_kwargs)

    # robosuite.make() does not reset; reset() builds env.sim, runs the
    # placement patch (h1_2_robosuite) and places the task objects.
    env.reset()

    model = env.sim.model._model
    data = env.sim.data._data
    resolver = NameResolver(model)

    # RoboCasa's zero-action settling during reset folds the limp robot, so put
    # it back in the spawn pose at rest and re-place it: floor clearance fitted,
    # and backed off (the robot's -x) until every robot geom is
    # GOLEM_SPAWN_MARGIN (unset: the env's spawn_margin, else 0.20 m) from any
    # non-floor geom, in the spawn pose and the sagged poses.
    try:
        init_qpos = _initial_motor_qpos()
        data.qpos[resolver.motor_qpos] = init_qpos
        data.qvel[:] = 0.0
        # GOLEM_SPAWN_BACKOFF (m, default 0: at the counter) backs the robot
        # further into open floor; Nav2 needs a free start cell to plan (~1.0).
        _spawn_backoff = float(os.environ.get("GOLEM_SPAWN_BACKOFF", "0") or 0)
        _spawn_margin = float(os.environ.get("GOLEM_SPAWN_MARGIN", "").strip()
                              or getattr(env, "spawn_margin", 0.20))
        # GOLEM_SPAWN_OFFSET="dx,dy" (m, world) shifts the task's spawn before
        # placement; unset keeps the task's own.
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
            poses=_limp_upper_body_poses(model, resolver.motor_qpos, init_qpos),
        )
        print("[h12_mujoco] initial stance from baked-in sim defaults")
    except Exception as e:
        print(f"[h12_mujoco] clean reset skipped: {e}")

    # The elastic band on the torso holds the robot until SPACE or
    # /elastic_band/toggle releases it.
    band = None
    band_body_id = -1
    rot_body_id = -1
    band_vel = np.zeros(6)
    rot_vel = np.zeros(6)
    try:
        band_body_id = resolver.body_id("torso_link")
        anchor = data.xpos[band_body_id].copy()
        # 4 cm below the torso's spawn (m): the feet carry part of the weight.
        anchor[2] -= 0.04
        # Stiffness 1e4 N/m holds the limp robot near upright through the ~20 s
        # the leg policy waits under the band for the arms' start-up (static
        # tilt ~ gravity torque / stiffness; at 1e3 it pitched forward ~60 deg).
        # Damping 2e3 N*s/m settles a sideways shove well within a second.
        # GOLEM_SIM_TETHER_ROT_STIFFNESS (N*m/rad, default 0: off) adds a
        # rotational hold at the spawn orientation while the band is on, a rigid
        # hoist for tethered manipulation experiments. GOLEM_SIM_TETHER_ROT_BODY
        # holds the torso (default) or the pelvis, which also stops the legs
        # spinning the pelvis about the waist with the feet off the ground. The
        # torque is applied explicitly, so the damping keeps d*dt/I well under 2
        # at the 2 ms step: torso yaw I ~0.13 kg*m^2 -> 0.04 s x stiffness,
        # pelvis ~0.009 kg*m^2 -> 0.006 s x stiffness.
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

    # Every MjData access off the main loop's thread (sensors, grippers,
    # viewer) holds sim_lock, which the main loop holds around mj_step.
    sim_lock = threading.Lock()
    pfx = env.robots[0].robot_model.naming_prefix

    # Ground-truth odom (odom -> pelvis on /tf, /odom) from the pelvis free
    # joint's body, only with GOLEM_SIM_ODOM: the real robot has none, and it
    # would compete with FAST-LIO's odometry.
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

    # The gripper actuators are the hand bridges' and disjoint from the 27 body motors.
    sim_interface = SimInterface(model, data, resolver=resolver)

    init_ros()

    # The cameras publish what the real RealSense drivers do
    # (realsense_profile.yaml). Rendering is the heaviest per-step CPU cost:
    # GOLEM_CAMERAS=0 drops every camera. GOLEM_CAMERA_SCALE renders the head
    # camera at that fraction of its real 1280x720 (same projection, scaled
    # camera_info). The hand cameras keep their gripper-MJCF fovy at
    # GOLEM_HAND_CAMERA_RES=<W>x<H> (default 256x256; the D405's colour stream
    # is 640x480). GOLEM_CAMERA_DEPTH=0 drops depth/image_rect_raw and
    # GOLEM_CAMERA_POINTCLOUD=0 drops depth/color/points; with both off the
    # depth render is skipped.
    def _env_on(name):
        """Return whether the env flag name is on (default on; 0/off/false/no turn it off)."""
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
            # 360 x 56 rays at 10 Hz is ~201k points/s, the real MID-360's
            # ~200k; 0.1-40 m range.
            lidar_az_rays=360, lidar_el_rays=56,
            lidar_rate_hz=10.0, lidar_max_range=40.0, lidar_min_range=0.1,
            # The real MID-360's IMU rate. FAST-LIO propagates the pose with the
            # IMU between scans, and at 100 Hz the gait's accelerations alias
            # into drift.
            imu_rate_hz=200.0,
            lidar_body=f"{pfx}livox_link",
            lidar_exclude_body=f"{pfx}torso_link",
            imu_quat_sensor=f"{pfx}livox_imu_quat",
            imu_gyro_sensor=f"{pfx}livox_imu_gyro",
            imu_acc_sensor=f"{pfx}livox_imu_acc",
            # Drop the robot's own limbs from the scan: their returns map an
            # obstacle under the robot and Nav2 refuses the start.
            # GOLEM_LIDAR_SELF_FILTER=0 keeps them as returns.
            lidar_self_prefixes=(
                (pfx, "gripper0_")
                if os.environ.get("GOLEM_LIDAR_SELF_FILTER", "1").strip().lower()
                not in ("0", "off", "false", "no")
                else ()
            ),
            sim_lock=sim_lock,
            timings=timings,
        ),
        base_body_id=odom_base_body_id,
        sim_lock=sim_lock,
    )

    hand_right = MagpieHandBridge(model, data, side="right", sim_lock=sim_lock,
                                  name_prefix="gripper0_right_")
    hand_left = MagpieHandBridge(model, data, side="left", sim_lock=sim_lock,
                                 name_prefix="gripper0_left_")

    measurement = MeasurementBridge(env, elastic_band=band, task_name=task)
    print(f"[h12_mujoco] task goal: {measurement.publish_goal()!r}")

    # The executor thread serves the grippers and the band toggle. ros_bridge is
    # not on it: the main loop calls its tick(), whose renderer GL context is
    # thread-affine.
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in (measurement, hand_right, hand_left):
        executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True, name="ros_bridge_exec")
    executor_thread.start()
    print("[h12_mujoco] ROS bridges up: rt/lowstate+rt/lowcmd, /clock, "
          "RGBD /realsense/{head,left_hand,right_hand}, livox lidar+imu, "
          "/{left,right}/gripper/*, /elastic_band/toggle")

    # The viewer draws its own MjData, refreshed from data by the viewer thread;
    # its mouse perturbation forces come back through view_sync.
    viewer_data = mujoco.MjData(model) if viewer else None
    if viewer_data is not None:
        with sim_lock:
            copy_display_state(viewer_data, data)
        mujoco.mj_forward(model, viewer_data)
    # SPACE in the viewer toggles the band (ElasticBand.key_callback).
    handle = (
        mujoco.viewer.launch_passive(model, viewer_data, key_callback=band.key_callback)
        if viewer and band is not None
        else (mujoco.viewer.launch_passive(model, viewer_data) if viewer else None)
    )
    if handle is not None:
        handle.opt.geomgroup[0] = 0   # hide collision geoms
        # Frame the torso, resolved again if the band setup failed.
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
    # The fall logger logs once when the torso tips or drops, with how long the
    # robot stood after the band's release.
    _fall_logged = False
    _band_release_t = None
    _band_prev_on = bool(band.enabled) if band is not None else False

    # RoboCasa updates fixtures and checks success once per control step
    # (20 Hz), not per physics step: several fixtures count calls (toaster
    # heat, scrub contact), and the success debounce counts control steps.
    task_every = max(1, round(env.control_timestep / model.opt.timestep))
    step_count = 0
    # The viewer thread draws the band while it is enabled; mouse forces come back
    # through view_sync.apply_perturbation().
    view_sync = None
    if handle is not None:
        view_sync = ViewerSync(
            model, data, handle, sim_lock, viewer_data,
            draw=band_overlay(band, band_body_id) if band is not None else None,
            timings=timings)
        view_sync.start()

    timings.reset(float(data.time))
    pacer = RealtimePacer(model.opt.timestep)
    try:
        while True:
            if viewer and not handle.is_running():
                break
            physics_start = timings.start()
            with sim_lock:
                if view_sync is not None:
                    view_sync.apply_perturbation()
                if band is not None:
                    # MuJoCo keeps xfrc_applied between steps, so a released
                    # band's force is zeroed every step.
                    if band.enabled:
                        # [angular(3), linear(3)] of the torso, world frame
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
                sim_interface.write_ctrl()
                mujoco.mj_step(model, data)
                timings.stop("physics", physics_start)
                lowstate_start = timings.start()
                sim_interface.after_step()
                timings.stop("lowstate", lowstate_start)
                if band is not None and _band_prev_on and not band.enabled:
                    _band_release_t = float(data.time)
                    _band_prev_on = False
                    print(f"[fall-logger] elastic band released at t={data.time:.2f}s", flush=True)
                if band_body_id >= 0 and not _fall_logged:
                    _upz = float(data.xmat[band_body_id][8])   # torso z-axis . world up (1: upright)
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
                    env.update_state()              # before _check_success
                    timings.stop("task_update", task_start)
            # tick() takes sim_lock only to capture sensor state, so encoding and
            # publishing do not hold up physics. No physics step runs inside it.
            ros_bridge.tick()
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
        if view_sync is not None:
            view_sync.stop()
        if handle is not None:
            handle.close()
        executor.shutdown()
        ros_bridge.shutdown()
        for node in (measurement, hand_right, hand_left, ros_bridge):
            node.destroy_node()
        sim_interface.shutdown()
        shutdown_ros()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the H1-2 RoboCasa MuJoCo sim")
    sim_common.add_common_args(
        parser,
        task_help="RoboCasa env name (e.g. TurnOnToasterOven), Battery for the "
                  "battery workcell (CL_Assets/mujoco_assets/battery), or Blank for "
                  "the robot alone on a floor (no --layout or --style). "
                  "Omit to launch a random registered kitchen task.",
    )
    args = sim_common.parse_args(parser, sys.argv[1:], "h12")

    task = args.task
    if task is None:
        task = sim_common.random_task(seed=args.seed)
        print(f"[h12_mujoco] no --task given; randomly selected {task!r}")
    else:
        task = sim_common.resolve_task(
            task, seed=args.seed, tag="h12_mujoco",
            robot_task=lambda name: battery_env.canonical_task(name) or blank_env.canonical_task(name))
    sim_common.refuse_kitchen_args(parser, args, task, blank_env.TASKS)

    sim_loop(task, viewer=not args.headless,
             layout=args.layout, style=args.style, seed=args.seed, timing=args.timing,
             duration=args.duration)
