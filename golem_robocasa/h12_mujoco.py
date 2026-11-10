"""Run the RoboCasa MuJoCo sim of the H1-2, or of the G1 with --g1 or GOLEM_ROBOT=g1.

Usage: h12_mujoco.py [--h12 | --g1] [--headless] [--task NAME] [--layout N]
[--style N] [--seed N] [--timing | --duration S]. The robot is chosen
(sim_common.select_robot) before the H1-2's imports, and --g1 hands the whole
command line to g1.g1_mujoco.main. Each runner parses its command line, then
checks the GPU (sim_common.require_gpu: either sim stops there without one), so
--help and a usage error need no GPU; the H1-2 does both before its imports. The
H1-2's loop is sim_loop below.
"""
import argparse
import sys

import sim_common


def _parser() -> argparse.ArgumentParser:
    """Return the H1-2 runner's argument parser."""
    parser = argparse.ArgumentParser(description="Run the H1-2 RoboCasa MuJoCo sim")
    sim_common.add_common_args(
        parser,
        task_help="RoboCasa env name (e.g. TurnOnToasterOven), Battery for the "
                  "battery workcell (CL_Assets/mujoco_assets/battery), or Blank for "
                  "the robot alone on a floor (no --layout or --style). "
                  "Omit to launch a random registered kitchen task.",
    )
    return parser


# The G1 is dispatched before the H1-2 imports below, which need magpie_msgs and
# the H1-2's CL_Assets files: the G1 reads only CL_Assets/g1, and its test
# containers do not provide magpie_msgs. The H1-2 parses its command line and
# checks the GPU (its cameras and the lidar's rays render through EGL on it) first.
if __name__ == "__main__":
    _robot = sim_common.select_robot(sys.argv[1:])
    if _robot == "g1":
        from g1 import g1_mujoco
        sys.exit(g1_mujoco.main(sys.argv[1:]))
    _PARSER = _parser()
    _ARGS = sim_common.parse_args(_PARSER, sys.argv[1:], "h12")
    sim_common.require_gpu()

import os
import threading

import mujoco
import numpy as np
import rclpy.executors

from mujoco_ros_bridge import init_ros, shutdown_ros

import h1_2_robosuite  # registers H1_2 and the MAGPIE grippers
import realsense_profile
import robosuite_common
import battery_env  # registers the Battery env
import blank_env  # registers the Blank env
from h12_sensors import H12Sensors
from magpie_hand_bridge import MagpieHandBridge
from measurement_bridge import MeasurementBridge
from mujoco_env import ElasticBand, FallLogger, apply_band
from mujoco_ros_bridge import RosSensorBridge
from sim_names import NameResolver
from sim_timing import RealtimePacer, LoopTimings
from unitree_interface import SimInterface
from viewer_sync import close_viewer, open_viewer

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


def spawn(env, resolver, backoff=0.0, margin=None, offset=(0.0, 0.0)):
    """Put the robot at rest in its spawn pose, clear of the fixtures by margin, in the sagged poses too.

    Call after env.reset(), whose zero-action settling folds the limp robot.
    The robot is placed at RoboCasa's spawn shifted by offset ((dx, dy) m,
    world), its lowest geom 2 cm above the floor, and backed off (its -x) until
    every robot geom is margin (m; None: the env's spawn_margin, else 0.20)
    from any non-floor geom, in the spawn pose and the poses the unpowered
    upper body sags into; backoff (m) then backs it up to that much further
    into open floor. Zeroes every velocity in the scene.
    """
    data = env.sim.data._data
    init_qpos = _initial_motor_qpos()
    data.qpos[resolver.motor_qpos] = init_qpos
    data.qvel[:] = 0.0
    pos = list(env.init_robot_base_pos)
    pos[0] += offset[0]
    pos[1] += offset[1]
    h1_2_robosuite.place_robot_collision_free(
        env, pos, robosuite_common.euler_to_wxyz(getattr(env, "init_robot_base_ori", None)),
        extra_backoff=backoff,
        margin=getattr(env, "spawn_margin", 0.20) if margin is None else margin,
        poses=_limp_upper_body_poses(env.sim.model._model, resolver.motor_qpos, init_qpos),
    )


def sim_loop(task, viewer=True, layout=None, style=None, seed=None, timing=False):
    """Run the H1-2 in task (a RoboCasa kitchen, Battery or Blank) until the viewer closes or Ctrl-C.

    The loop steps the env's own MjData (never env.step()) and drives:
      - SimInterface     : Unitree DDS rt/lowstate + rt/lowcmd (27 body motors)
      - RosSensorBridge  : /clock, opt-in odom, and H12Sensors: RGBD cameras,
                           MID-360 lidar + IMU
      - MagpieHandBridge : /{left,right}/gripper/*
      - MeasurementBridge: /robocasa/*, /elastic_band/toggle
    An elastic band holds the robot until it is released. The loop paces to
    GOLEM_SIM_RATE (sim_common.sim_rate_from_env; default real time). timing
    enables the --timing stage report.
    """

    # A bad GOLEM_SIM_RATE stops the sim before the kitchen is built.
    sim_rate = sim_common.sim_rate_from_env()

    env = h1_2_robosuite.make_kitchen_env(task, layout_ids=layout, style_ids=style, seed=seed)

    # robosuite.make() does not reset; reset() builds env.sim, runs the
    # placement patch (h1_2_robosuite) and places the task objects.
    env.reset()
    # The fixtures' own parts stop colliding with each other and with the static scene; the
    # robot and the objects collide with everything as before.
    sim_common.exclude_fixture_collisions(env)

    model = env.sim.model._model
    data = env.sim.data._data
    resolver = NameResolver(model)

    try:
        # GOLEM_SPAWN_BACKOFF (m, default 0: at the counter) backs the robot
        # further into open floor; Nav2 needs a free start cell to plan (~1.0).
        # GOLEM_SPAWN_MARGIN (m, unset: the env's spawn_margin, else 0.20) is
        # the clearance to the fixtures. GOLEM_SPAWN_OFFSET="dx,dy" (m, world)
        # shifts the task's spawn before placement; unset keeps the task's own.
        _spawn_margin = os.environ.get("GOLEM_SPAWN_MARGIN", "").strip()
        _offset = os.environ.get("GOLEM_SPAWN_OFFSET", "").strip()
        _dx, _dy = (float(v) for v in _offset.split(",")) if _offset else (0.0, 0.0)
        if _offset:
            print(f"[h12_mujoco] spawn offset ({_dx:+.3f}, {_dy:+.3f}) m")
        spawn(env, resolver, backoff=float(os.environ.get("GOLEM_SPAWN_BACKOFF", "0") or 0),
              margin=float(_spawn_margin) if _spawn_margin else None, offset=(_dx, _dy))
        print("[h12_mujoco] initial stance from baked-in sim defaults")
    except Exception as e:
        print(f"[h12_mujoco] clean reset skipped: {e}")

    # The elastic band on the torso holds the robot until SPACE or
    # /elastic_band/toggle releases it.
    band = None
    band_body_id = -1
    band_vel = np.zeros(6)   # apply_band's scratch
    try:
        band_body_id = resolver.body_id("torso_link")
        anchor = data.xpos[band_body_id].copy()
        # 4 cm below the torso's spawn (m): the feet carry part of the weight.
        anchor[2] -= 0.04
        # Stiffness 1e4 N/m holds the limp robot near upright through the ~20 s
        # the leg policy waits under the band for the arms' start-up (static
        # tilt ~ gravity torque / stiffness; at 1e3 it pitched forward ~60 deg).
        # Damping 2e3 N*s/m settles a sideways shove well within a second.
        band = ElasticBand(anchor, length=0, stiffness=1e4, damping=2e3)
        print(f"[h12_mujoco] elastic band on torso anchored at {anchor.round(3)}")
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
    _sim_odom_on = sim_common.env_flag('GOLEM_SIM_ODOM', False)
    if _sim_odom_on:
        try:
            _fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{pfx}{robosuite_common.FREE_JOINT_NAME}")
            odom_base_body_id = int(model.jnt_bodyid[_fj])
        except Exception as e:
            print(f"[h12_mujoco] base odom body resolve skipped: {e}")
    print(f"[h12_mujoco] ground-truth sim odom {'ON (GOLEM_SIM_ODOM=1)' if _sim_odom_on else 'OFF'}")

    # The gripper actuators are the hand bridges' and disjoint from the 27 body motors.
    sim_interface = SimInterface(model, data, resolver=resolver)
    # For the closed-loop tests that record the sim from inside its process.
    sim_interface.band = band

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
    _cameras_on = sim_common.env_flag('GOLEM_CAMERAS', True)
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
    _depth_on = sim_common.env_flag('GOLEM_CAMERA_DEPTH', True)
    _cloud_on = sim_common.env_flag('GOLEM_CAMERA_POINTCLOUD', True)
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
                (pfx, "gripper0_") if sim_common.env_flag("GOLEM_LIDAR_SELF_FILTER", True) else ()
            ),
            sim_lock=sim_lock,
            timings=timings,
        ),
        base_body_id=odom_base_body_id,
        sim_lock=sim_lock,
    )

    print(f"[h12_mujoco] {sim_common.describe_sim_rate(sim_rate)}", flush=True)

    hand_right = MagpieHandBridge(model, data, side="right", sim_lock=sim_lock,
                                  name_prefix="gripper0_right_")
    hand_left = MagpieHandBridge(model, data, side="left", sim_lock=sim_lock,
                                 name_prefix="gripper0_left_")

    measurement = MeasurementBridge(env, elastic_band=band, task_name=task)
    print(f"[h12_mujoco] task goal: {measurement.publish_goal()!r}")

    # The executor thread serves the grippers and the band toggle. ros_bridge is
    # not on it: the main loop calls its tick(), whose renderer GL context is
    # thread-affine. The thread starts just before the loop's try, whose finally
    # shuts it down.
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in (measurement, hand_right, hand_left):
        executor.add_node(node)

    # The viewer draws its own MjData, refreshed from data by the viewer thread,
    # with the band while it is enabled; its mouse perturbation forces come back
    # through view_sync.apply_perturbation().
    handle = view_sync = None
    if viewer:
        handle, view_sync = open_viewer(model, data, sim_lock, band, band_body_id, "h12_mujoco",
                                        timings=timings)
    # Logs the band's release and the first time the torso tips or drops.
    fall_logger = FallLogger(band_body_id, "torso", upright_min=0.5, height_min=0.5)

    # RoboCasa updates fixtures and checks success once per control step
    # (20 Hz), not per physics step: several fixtures count calls (toaster
    # heat, scrub contact), and the success debounce counts control steps.
    task_every = max(1, round(env.control_timestep / model.opt.timestep))
    step_count = 0

    timings.reset(float(data.time))
    pacer = RealtimePacer(model.opt.timestep, sim_rate)
    executor_thread = threading.Thread(target=executor.spin, daemon=True, name="ros_bridge_exec")
    executor_thread.start()
    print("[h12_mujoco] ROS bridges up: rt/lowstate+rt/lowcmd, /clock, "
          "RGBD /realsense/{head,left_hand,right_hand}, livox lidar+imu, "
          "/{left,right}/gripper/*, /elastic_band/toggle")
    try:
        while True:
            if viewer and not handle.is_running():
                break
            physics_start = timings.start()
            with sim_lock:
                if view_sync is not None:
                    view_sync.apply_perturbation()
                if band is not None:
                    apply_band(model, data, band, band_body_id, band_vel)
                sim_interface.write_ctrl()
                mujoco.mj_step(model, data)
                timings.stop("physics", physics_start)
                lowstate_start = timings.start()
                sim_interface.after_step()
                timings.stop("lowstate", lowstate_start)
                fall_logger.step(data, band)
                step_count += 1
                task_step = step_count % task_every == 0
                if task_step:
                    task_start = timings.start()
                    env.update_state()              # before _check_success
                    timings.stop("task_update", task_start)
            # tick() takes sim_lock only to capture sensor state, so encoding and
            # publishing do not hold up physics. No physics step runs inside it.
            ros_bridge.tick()
            # The grippers' state publishes at its rate of sim time.
            sim_t = float(data.time)
            hand_right.tick(sim_t)
            hand_left.tick(sim_t)
            if task_step:
                task_start = timings.start()
                with sim_lock:
                    done = measurement.tick()
                timings.stop("task_check", task_start)
                if done:
                    print("[h12_mujoco] task success (debounced).")
            pacer.wait()
            timings.maybe_report(float(data.time), ros_bridge.stats)
    finally:
        # A DeliGrasp settle waits on the sim clock, which no longer advances: end it first,
        # whatever the viewer's shutdown does, so its goal aborts and the executor's worker
        # thread can exit.
        hand_right.stop()
        hand_left.stop()
        close_viewer(handle, view_sync)
        executor.shutdown()
        ros_bridge.shutdown()
        for node in (measurement, hand_right, hand_left, ros_bridge):
            node.destroy_node()
        sim_interface.shutdown()
        shutdown_ros()


if __name__ == "__main__":
    _TASK = sim_common.select_task(_PARSER, _ARGS, tag="h12_mujoco",
                                   own_tasks=battery_env.TASKS + blank_env.TASKS, no_kitchen=blank_env.TASKS)
    sim_loop(_TASK, viewer=not _ARGS.headless,
             layout=_ARGS.layout, style=_ARGS.style, seed=_ARGS.seed, timing=_ARGS.timing)
