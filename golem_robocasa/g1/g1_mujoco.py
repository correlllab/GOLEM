import argparse
import os
import sys
import threading
import time

import mujoco
import numpy as np
import rclpy.executors

from mujoco_ros_bridge import RosSensorBridge, init_ros, shutdown_ros
import mujoco.viewer

from . import g1_robosuite  # registers G1Dex3 + its grippers, patches RoboCasa
import sim_common
from measurement_bridge import MeasurementBridge
from mujoco_env import ElasticBand, draw_band
from .g1_sensors import G1Sensors, SensorConfig
from .sim_names import BODY_JOINTS, NameResolver
from .unitree_interface import SimInterface

# The fall logger's test, as mujoco_deploy_g1.py's: the pelvis tilted more than
# 60 deg from upright, or down on the floor.
FALL_UPRIGHTNESS = 0.5   # pelvis z-axis . world up
FALL_PELVIS_HEIGHT = 0.2  # m

# The elastic band, the H1-2 sim's (h12_mujoco.py): a spring-damper pulling the
# torso toward a point BAND_ANCHOR_DZ above its spawn position, which holds the
# robot while no motor drives it, until /elastic_band/toggle (or SPACE in the
# viewer) releases it. It is twice as stiff as the H1-2's: at the H1-2's 1e4 N/m
# the limp G1 (34.4 kg) rocks fore and aft on its limp legs, its pelvis tilting
# 16-19 deg for 20 s and more, and the physics can go unstable; at 2e4 N/m it
# settles within 4 s, upright to 0.3 deg with its torso ~1.4 cm below the anchor
# (measured in three RoboCasa kitchens; tests/robocasa/g1/test_g1_band.py checks
# one).
BAND_BODY = "torso_link"
BAND_ANCHOR_DZ = 0.01    # m
BAND_STIFFNESS = 2e4     # N/m
BAND_DAMPING = 2e3       # N*s/m
# The spawn puts the lowest collision geom this far above the floor, so the
# robot hangs from the band with its feet on the floor.
SPAWN_CLEARANCE = 0.001  # m
# Real-time pacing resyncs instead of catching up once the sim lags this far behind.
MAX_LAG = 0.1  # s
# The loop logs its real-time factor every this many sim seconds.
RTF_LOG_S = 30.0
# Python's GIL switch interval while the sim runs (its default is 5 ms). The physics
# loop sleeps between steps to keep real time and needs the GIL back to wake; the
# sensor threads' Python work (the head camera's messages) held it up to the 5 ms
# default, stalling rt/lowstate 5-9 ms (the real G1 sends it every 2 ms). At
# 0.5 ms the sim alone with both sensors had no gap over 5 ms in 51 s of sim time.
GIL_SWITCH_S = 0.0005
# MuJoCo resets the whole scene (robot and kitchen) when it finds the physics unstable
UNSTABLE = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC)


def spawn(env, resolver, backoff=0.0):
    """Put the robot at rest in the stance (g1_robosuite.STANCE), hands open, at
    RoboCasa's spawn, after env.reset(): RoboCasa's zero-action settling loop
    folds the limp robot during reset (and its initialization noise perturbs
    every joint). place_robot_collision_free fits the lowest robot geom
    SPAWN_CLEARANCE above the floor AND backs the base away (the robot's -x)
    until no robot geom penetrates a fixture, keeping the least-penetrating spot
    if it can't fully clear; backoff (m) then backs it up to that much further
    into open floor. Zeroes every velocity in the scene."""
    data = env.sim.data._data
    data.qpos[resolver.body.qpos] = [g1_robosuite.STANCE.get(n, 0.0) for n in BODY_JOINTS]
    for hand in resolver.hands.values():
        data.qpos[hand.qpos] = 0.0
    data.qvel[:] = 0.0
    g1_robosuite.place_robot_collision_free(
        env, env.init_robot_base_pos,
        g1_robosuite._euler_to_wxyz(getattr(env, "init_robot_base_ori", None)),
        clearance=SPAWN_CLEARANCE, extra_backoff=backoff,
    )


def make_band(data, resolver):
    """The elastic band on the torso, anchored BAND_ANCHOR_DZ above where the
    torso is now (call it at the spawn); returns (band, the torso's body id)"""
    body = resolver.body_id(BAND_BODY)
    if body < 0:
        raise KeyError(f"no body {resolver.robot_prefix}{BAND_BODY} for the elastic band")
    anchor = data.xpos[body].copy()
    anchor[2] += BAND_ANCHOR_DZ
    return ElasticBand(anchor, length=0, stiffness=BAND_STIFFNESS, damping=BAND_DAMPING), body


def apply_band(model, data, band, body, vel):
    """Write the band's force on body into xfrc_applied for the next physics
    step: its pull while enabled, else zero. MuJoCo keeps xfrc_applied between
    steps, so call it before every mj_step; vel is a 6-vector scratch buffer."""
    if band.enabled:
        # [ang(3), lin(3)] of the torso itself, world frame
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
        data.xfrc_applied[body, :3] = band.evaluate_force(data.xpos[body], vel[3:])
    else:
        data.xfrc_applied[body, :3] = 0.0


def sim_loop(task, viewer=True, layout=None, style=None, seed=None, timing=False, duration=0.0):
    """Launch a RoboCasa task env around the G1 and run the shared-MjData loop.

    Builds the env with robots='G1Dex3' and steps the env's *own* MjData with our
    loop (never env.step()), driving the sim<->ROS bridge layer:
      - SimInterface     : Unitree DDS, as the G1's own computers serve it in
                           debug mode: rt/lowstate, rt/lowcmd, rt/dex3/{left,right}/*;
                           every motor off until the first valid rt/lowcmd
      - RosSensorBridge  : /clock, and G1Sensors: the MID-360 (/livox/lidar,
                           /livox/imu) and the head D435 (/realsense/head/*), on
                           unless GOLEM_LIDAR=0 / GOLEM_CAMERAS=0 (SensorConfig),
                           + ground-truth /odom with GOLEM_SIM_ODOM=1
      - MeasurementBridge: /robocasa/{task_name,task_goal,success,reward} +
                           /elastic_band/toggle
    An elastic band on the torso holds the robot until /elastic_band/toggle or
    SPACE in the viewer releases it, as in the H1-2 sim. RoboCasa's
    _check_success/reward/lang are read off the shared env.
    duration: sim seconds to run from the first step, counted in physics steps
    (0 = until the viewer closes or Ctrl-C).
    timing is part of the signature both runners share; the command line rejects
    --timing for the G1, which logs its real-time factor itself.
    """

    create_kwargs = {}
    if layout is not None:
        create_kwargs["layout_ids"] = layout
    if style is not None:
        create_kwargs["style_ids"] = style
    if seed is not None:
        create_kwargs["seed"] = seed

    sensors = SensorConfig.from_env()   # before the slow env build, to fail fast on a bad knob
    env = g1_robosuite.make_kitchen_env(task, **create_kwargs)

    # robosuite.make() constructs but does NOT reset; reset() builds env.sim, runs
    # _load_model + _reset_internal (placement patch fires here, sets ep_meta /
    # init_robot_base_pos), and places the task objects.
    env.reset()

    model = env.sim.model._model
    data = env.sim.data._data
    pfx = env.robots[0].robot_model.naming_prefix   # "robot0_"
    resolver = NameResolver(model, robot_prefix=pfx)  # DDS<->sim name map

    # Clean, contact-free initial state: the robot at rest in the stance (spawn).
    try:
        # GOLEM_SPAWN_BACKOFF backs the robot further off the counter into open
        # floor (default 0 = at the counter for manipulation). Set ~1.0 for nav2,
        # which needs the robot's start cell to be free to plan.
        spawn(env, resolver, backoff=float(os.environ.get("GOLEM_SPAWN_BACKOFF", "0") or 0))
        print("[g1_mujoco] initial stance: legs at Unitree's default angles, everything else at 0")
    except Exception as e:
        print(f"[g1_mujoco] clean reset skipped: {e}")

    # The elastic band holds the robot from the first physics step, while every
    # motor is off until the first valid rt/lowcmd.
    band, band_body = make_band(data, resolver)
    band_vel = np.zeros(6)
    print(f"[g1_mujoco] elastic band on {BAND_BODY} anchored at {band.point.round(3)}; "
          "/elastic_band/toggle or SPACE in the viewer releases it")

    print(f"[g1_mujoco] RoboCasa task {task!r} built around the G1: "
          f"nq={model.nq} nu={model.nu}; resolver mapped {len(resolver.body.names)} body "
          f"+ {sum(len(h.names) for h in resolver.hands.values())} hand motors")

    # Shared lock: every MjData access from outside the main loop (bridge
    # threads, services) goes through sim_lock while the main loop runs mj_step.
    sim_lock = threading.Lock()

    # Body carrying the pelvis free joint — its world pose is ground-truth base
    # odometry, which the bridge publishes as odom -> pelvis TF + /odom (feeds
    # SLAM without needing FAST-LIO). OFF by default so the shared sim bridge does
    # not inject a second odom->pelvis anchor that competes with the real stack's
    # FAST-LIO (camera_init) TF; g1_navigation's odom_source:=sim_gt opts in with GOLEM_SIM_ODOM=1.
    _sim_odom_on = os.environ.get('GOLEM_SIM_ODOM', '0').strip().lower() not in ('0', 'off', 'false', 'no', '')
    odom_base_body_id = resolver.base_body if _sim_odom_on else -1
    print(f"[g1_mujoco] ground-truth sim odom {'ON (GOLEM_SIM_ODOM=1)' if _sim_odom_on else 'OFF'}")

    sim_interface = SimInterface(model, data, resolver)
    sim_interface.band = band

    sys.setswitchinterval(GIL_SWITCH_S)
    init_ros()

    # Sensor bridge: /clock, the MID-360 and the head D435 (G1Sensors), and the
    # optional ground-truth odom. The loop calls tick() holding sim_lock, which
    # is not reentrant, so the bridge takes none of its own.
    ros_bridge = RosSensorBridge(
        model, data,
        lambda node: G1Sensors(node, model, data, prefix=pfx, config=sensors),
        base_body_id=odom_base_body_id,   # -> odom -> pelvis TF + /odom
        sim_lock=None,
    )
    print(f"[g1_mujoco] sensors: {sensors.describe()}")

    measurement = MeasurementBridge(env, elastic_band=band, task_name=task)
    print(f"[g1_mujoco] task goal: {measurement.publish_goal()!r}")

    # Background executor serves the nodes' ROS callbacks (the band toggle,
    # parameter services). ros_bridge.tick() is driven from the main loop instead
    # (it reads MjData between steps; the camera renders on its own thread), so it
    # is NOT added here.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(measurement)
    executor_thread = threading.Thread(target=executor.spin, daemon=True, name="ros_bridge_exec")
    executor_thread.start()
    print("[g1_mujoco] ROS bridges up: rt/lowstate, rt/lowcmd, "
          "rt/dex3/{left,right}/{cmd,state}, /clock, "
          + ("/livox/lidar, /livox/imu, " if sensors.lidar else "")
          + ("/realsense/head/*, " if sensors.camera else "") + "/robocasa/*, /elastic_band/toggle")

    # SPACE in the viewer toggles the band (ElasticBand.key_callback).
    handle = mujoco.viewer.launch_passive(model, data, key_callback=band.key_callback) if viewer else None
    if handle is not None:
        handle.opt.geomgroup[0] = 0   # hide collision geoms by default
        # Frame the free camera on the robot's spawn pose (function of robot
        # position + yaw).
        try:
            with handle.lock():
                sim_common.frame_viewer_on_robot(handle, data, band_body)
        except Exception as e:
            print(f"[g1_mujoco] viewer camera framing skipped: {e}")
    # Fall logger: watch the pelvis's uprightness + height and log once when the
    # robot falls, including how long it stood after the elastic band was released.
    _fall_logged = False
    _release_t = None
    pelvis = resolver.base_body
    # RoboCasa updates its fixtures once per control step (its timers count
    # update_state() calls), so update them and read the task's success at
    # env.control_freq (20 Hz) of sim time, not every physics step.
    task_every = max(1, round(1.0 / (env.control_freq * model.opt.timestep)))
    steps = 0
    max_steps = round(duration / model.opt.timestep) if duration > 0 else None
    done = _success = False
    _resets = 0
    wall_start, sim_start = time.time(), float(data.time)
    # real-time factor log: since the last line, the sim and wall time and the time slept
    log_sim, log_wall, slept = float(data.time), time.time(), 0.0

    try:
        while max_steps is None or steps < max_steps:
            if viewer and not handle.is_running():
                break
            with sim_lock:
                apply_band(model, data, band, band_body, band_vel)
                # Recompute every motor's PD torque every sim step (emulates the
                # real motors' servo loop) from the latched DDS commands.
                sim_interface.write_ctrl()
                _before = float(data.time)
                mujoco.mj_step(model, data)
                # MuJoCo's auto-reset clears the warning counters and then counts
                # the one warning, so clear them here to see every reset; the
                # reset also puts the sim time back to 0
                if any(data.warning[w].number for w in UNSTABLE) or data.time < _before:
                    for w in UNSTABLE:
                        data.warning[w].number = 0
                    _resets += 1
                    sim_interface.physics_reset()
                    print(f"[g1_mujoco] *** WARNING: the physics went unstable and MuJoCo reset the "
                          f"whole scene, robot and kitchen, to its initial state and the sim time to 0 "
                          f"(reset {_resets} at step {steps}); restart the sim ***", flush=True)
                # publishes rt/lowstate and rt/dex3/*/state
                sim_interface.after_step()
                # --- fall logger ---
                if _release_t is None and not band.enabled:
                    _release_t = float(data.time)
                    print(f"[fall-logger] elastic band released at t={data.time:.2f}s", flush=True)
                if not _fall_logged:
                    _upz = float(data.xmat[pelvis][8])   # pelvis z-axis . world up (1=upright)
                    _pelvis_h = float(data.xpos[pelvis][2])
                    if _upz < FALL_UPRIGHTNESS or _pelvis_h < FALL_PELVIS_HEIGHT:
                        _fall_logged = True
                        _since = (f"{data.time - _release_t:.2f}s after band release"
                                  if _release_t is not None
                                  else f"{data.time:.2f}s (band still on)")
                        print(f"[fall-logger] *** ROBOT FELL at sim t={data.time:.2f}s — stood "
                              f"{_since} (uprightness={_upz:.2f}, pelvis_h={_pelvis_h:.2f}) ***",
                              flush=True)
                ros_bridge.tick()                   # /clock (+ odom)
                steps += 1
                if steps % task_every == 0:
                    env.update_state()              # REQUIRED before _check_success
                    done = measurement.tick()
            if done and not _success:
                print("[g1_mujoco] task success (debounced).")
            _success = done
            if viewer:
                if band.enabled:
                    draw_band(handle, band.point, data.xpos[band_body])
                else:
                    handle.user_scn.ngeom = 0   # clear the overlay once the band is off
                handle.sync()
            # real-time pacing against sim time
            ahead = (data.time - sim_start) - (time.time() - wall_start)
            if ahead > 0:
                _t = time.time()
                time.sleep(ahead)
                slept += time.time() - _t
            elif ahead < -MAX_LAG:
                wall_start, sim_start = time.time(), float(data.time)
            if data.time < log_sim:   # MuJoCo reset the time
                log_sim, log_wall, slept = float(data.time), time.time(), 0.0
            elif data.time - log_sim >= RTF_LOG_S:
                _wall = time.time() - log_wall
                _stats = ros_bridge.stats()
                print(f"[g1_mujoco] sim t {data.time:.0f} s: real-time factor {(data.time - log_sim) / _wall:.2f} "
                      f"over the last {data.time - log_sim:.0f} s, the loop idle {slept / _wall:.0%} of the time"
                      + (f"; {_stats}" if _stats else ""), flush=True)
                log_sim, log_wall, slept = float(data.time), time.time(), 0.0
    finally:
        if handle is not None:
            handle.close()
        executor.shutdown()
        ros_bridge.shutdown()
        for node in (measurement, ros_bridge):
            node.destroy_node()
        sim_interface.shutdown()
        shutdown_ros()


def main(argv):
    """Run the G1 sim with the command-line arguments argv (without the program
    name). h12_mujoco.py calls it for --g1 or GOLEM_ROBOT=g1."""
    parser = argparse.ArgumentParser(description="Run the G1 RoboCasa MuJoCo sim")
    sim_common.add_common_args(parser)
    args = sim_common.parse_args(parser, argv, "g1")

    task = args.task
    if task is None:
        task = sim_common.random_task(seed=args.seed)
        print(f"[g1_mujoco] no --task given; randomly selected {task!r}")
    else:
        task = sim_common.resolve_task(task, seed=args.seed, tag="g1_mujoco")

    sim_loop(task, viewer=not args.headless,
             layout=args.layout, style=args.style, seed=args.seed, timing=args.timing,
             duration=args.duration)
