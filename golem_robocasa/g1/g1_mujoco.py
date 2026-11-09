"""Run the G1 RoboCasa MuJoCo sim: h12_mujoco.py hands --g1 and GOLEM_ROBOT=g1 to main()."""
import argparse
import contextlib
import os
import sys
import threading
import time

import mujoco
import numpy as np
import rclpy.executors

from mujoco_ros_bridge import RosSensorBridge, init_ros, shutdown_ros
import mujoco.viewer

from . import g1_robosuite  # registers G1Dex3 and its gripper
import battery_env  # registers the Battery env
import blank_env  # registers the Blank env
import sim_common
from measurement_bridge import MeasurementBridge
from mujoco_env import ElasticBand
from sim_timing import RealtimePacer
from .g1_sensors import G1Sensors, SensorConfig
from .sim_names import BODY_JOINTS, NameResolver
from .unitree_interface import SimInterface
from viewer_sync import ViewerSync, band_overlay, copy_display_state

# The fall logger's test: the pelvis tilted more than 60 deg from upright (as
# golem_lowerbody_rl's policies/homie/mujoco_deploy_g1.py tests it), or down on
# the floor.
FALL_UPRIGHTNESS = 0.5   # pelvis z-axis . world up
FALL_PELVIS_HEIGHT = 0.2  # m

# The elastic band pulls the torso toward a point BAND_ANCHOR_DZ above its
# spawn and holds the robot while no motor drives it, until /elastic_band/toggle
# or SPACE in the viewer releases it. It is twice as stiff as the H1-2's: at
# 1e4 N/m the limp G1 (34.4 kg) rocks fore and aft on its legs, the pelvis
# tilting 16-19 deg for 20 s and more, and the physics can go unstable; at
# 2e4 N/m it settles within 4 s, upright to 0.3 deg, the torso ~1.4 cm below
# the anchor (measured in three RoboCasa kitchens; tests/robocasa/g1/
# test_g1_band.py checks one).
BAND_BODY = "torso_link"
BAND_ANCHOR_DZ = 0.01    # m
BAND_STIFFNESS = 2e4     # N/m
BAND_DAMPING = 2e3       # N*s/m
# m: the spawn puts the lowest collision geom this far above the floor, so the
# robot hangs from the band with its feet on the floor.
SPAWN_CLEARANCE = 0.001
# s of wall lag the pacer repays at most; past it, it resyncs instead of catching up.
MAX_LAG = 0.1
# s of sim time between the loop's real-time factor logs.
RTF_LOG_S = 30.0
# s, Python's GIL switch interval while the sim runs (default 5 ms). The physics
# loop sleeps between steps to keep its pace and needs the GIL back to wake;
# at 5 ms the sensor threads' Python work (the head camera's messages) stalled
# rt/lowstate 5-9 ms, where the real G1 sends it every 2 ms. At 0.5 ms the sim
# with both sensors had no gap over 5 ms in 51 s of sim time.
GIL_SWITCH_S = 0.0005
# The warnings on which MuJoCo resets the whole scene (robot and kitchen).
UNSTABLE = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC)


def spawn(env, resolver, backoff=0.0):
    """Put the robot at rest in g1_robosuite.STANCE, hands open, clear of the fixtures.

    Call after env.reset(), whose zero-action settling folds the limp robot
    (and whose initialization noise perturbs every joint). The robot is placed
    at RoboCasa's spawn, its lowest collision geom SPAWN_CLEARANCE above the
    floor, and backed off (its -x) until it penetrates no fixture; backoff (m)
    then backs it up to that much further into open floor. Zeroes every
    velocity in the scene.
    """
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
    """Return (band, torso body id): the band on the torso, anchored BAND_ANCHOR_DZ above it now.

    Call it at the spawn.
    """
    body = resolver.body_id(BAND_BODY)
    if body < 0:
        raise KeyError(f"no body {resolver.robot_prefix}{BAND_BODY} for the elastic band")
    anchor = data.xpos[body].copy()
    anchor[2] += BAND_ANCHOR_DZ
    return ElasticBand(anchor, length=0, stiffness=BAND_STIFFNESS, damping=BAND_DAMPING), body


def apply_band(model, data, band, body, vel):
    """Write the band's force on body into xfrc_applied for the next physics step.

    Its pull while enabled, else zero: MuJoCo keeps xfrc_applied between steps,
    so call it before every mj_step. vel is a 6-vector scratch buffer.
    """
    if band.enabled:
        # [angular(3), linear(3)] of the torso, world frame
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
        data.xfrc_applied[body, :3] = band.evaluate_force(data.xpos[body], vel[3:])
    else:
        data.xfrc_applied[body, :3] = 0.0


def sim_loop(task, viewer=True, layout=None, style=None, seed=None, timing=False, duration=0.0):
    """Run the G1 in task (a RoboCasa kitchen, Battery or Blank) for duration or until stopped.

    The loop steps the env's own MjData (never env.step()) and drives:
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
    SPACE in the viewer releases it. The loop paces to GOLEM_SIM_RATE
    (sim_common.sim_rate_from_env; default real time). duration is sim seconds
    from the first step, counted in physics steps (0: until the viewer closes
    or Ctrl-C).
    timing is in the signature both runners share and is unused here: the
    command line refuses --timing for the G1, which logs its real-time factor
    itself.
    """

    create_kwargs = {}
    if layout is not None:
        create_kwargs["layout_ids"] = layout
    if style is not None:
        create_kwargs["style_ids"] = style
    if seed is not None:
        create_kwargs["seed"] = seed

    # Before the slow env build: a bad knob fails fast.
    sensors = SensorConfig.from_env()
    sim_rate = sim_common.sim_rate_from_env()
    env = g1_robosuite.make_kitchen_env(task, **create_kwargs)

    # robosuite.make() does not reset; reset() builds env.sim, runs the
    # placement patch (g1_robosuite) and places the task objects.
    env.reset()
    # The fixtures' own parts stop colliding with each other and with the static scene; the
    # robot and the objects collide with everything as before.
    sim_common.exclude_fixture_collisions(env)

    model = env.sim.model._model
    data = env.sim.data._data
    pfx = env.robots[0].robot_model.naming_prefix
    resolver = NameResolver(model, robot_prefix=pfx)

    try:
        # GOLEM_SPAWN_BACKOFF (m, default 0: at the counter) backs the robot
        # further into open floor; Nav2 needs a free start cell to plan (~1.0).
        spawn(env, resolver, backoff=float(os.environ.get("GOLEM_SPAWN_BACKOFF", "0") or 0))
        print("[g1_mujoco] initial stance: legs at Unitree's default angles, everything else at 0")
    except Exception as e:
        print(f"[g1_mujoco] clean reset skipped: {e}")

    # The band holds the robot from the first physics step; every motor is off
    # until the first valid rt/lowcmd.
    band, band_body = make_band(data, resolver)
    band_vel = np.zeros(6)
    print(f"[g1_mujoco] elastic band on {BAND_BODY} anchored at {band.point.round(3)}; "
          "/elastic_band/toggle or SPACE in the viewer releases it")

    print(f"[g1_mujoco] RoboCasa task {task!r} built around the G1: "
          f"nq={model.nq} nu={model.nu}; resolver mapped {len(resolver.body.names)} body "
          f"+ {sum(len(h.names) for h in resolver.hands.values())} hand motors")

    # Ground-truth odom (odom -> pelvis on /tf, /odom) from the pelvis, only
    # with GOLEM_SIM_ODOM: the real robot has none, and it would compete with
    # FAST-LIO's odometry.
    _sim_odom_on = os.environ.get('GOLEM_SIM_ODOM', '0').strip().lower() not in ('0', 'off', 'false', 'no', '')
    odom_base_body_id = resolver.base_body if _sim_odom_on else -1
    print(f"[g1_mujoco] ground-truth sim odom {'ON (GOLEM_SIM_ODOM=1)' if _sim_odom_on else 'OFF'}")

    sim_interface = SimInterface(model, data, resolver)
    # For the closed-loop tests that record the sim from inside its process.
    sim_interface.band = band

    sys.setswitchinterval(GIL_SWITCH_S)
    init_ros()

    # The physics loop is the only writer of MjData: the sensor threads work on
    # copies of the state tick() takes, and the ROS and DDS callbacks only latch
    # values the loop reads, so the bridge takes no lock and the loop calls
    # tick() between steps. The one other reader is the viewer's sync thread,
    # which copies the state under physics_lock (held around the step; a
    # null context headless, where nothing else reads).
    physics_lock = threading.Lock() if viewer else contextlib.nullcontext()
    ros_bridge = RosSensorBridge(
        model, data,
        lambda node: G1Sensors(node, model, data, prefix=pfx, config=sensors),
        base_body_id=odom_base_body_id,
        sim_lock=None,
    )
    print(f"[g1_mujoco] sensors: {sensors.describe()}")
    print(f"[g1_mujoco] {sim_common.describe_sim_rate(sim_rate)}", flush=True)

    measurement = MeasurementBridge(env, elastic_band=band, task_name=task)
    print(f"[g1_mujoco] task goal: {measurement.publish_goal()!r}")

    # The executor thread serves the band toggle and the parameter services.
    # ros_bridge is not on it: the main loop calls its tick() between steps.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(measurement)
    executor_thread = threading.Thread(target=executor.spin, daemon=True, name="ros_bridge_exec")
    executor_thread.start()
    print("[g1_mujoco] ROS bridges up: rt/lowstate, rt/lowcmd, "
          "rt/dex3/{left,right}/{cmd,state}, /clock, "
          + ("/livox/lidar, /livox/imu, " if sensors.lidar else "")
          + ("/realsense/head/*, " if sensors.camera else "") + "/robocasa/*, /elastic_band/toggle")

    # The viewer draws its own MjData, refreshed from data by the viewer thread;
    # its mouse perturbation forces come back through view_sync.
    viewer_data = mujoco.MjData(model) if viewer else None
    handle = view_sync = None
    if viewer:
        copy_display_state(viewer_data, data)
        mujoco.mj_forward(model, viewer_data)
        # SPACE in the viewer toggles the band (ElasticBand.key_callback).
        handle = mujoco.viewer.launch_passive(model, viewer_data, key_callback=band.key_callback)
        handle.opt.geomgroup[0] = 0   # hide collision geoms
        try:
            with handle.lock():
                sim_common.frame_viewer_on_robot(handle, viewer_data, band_body)
        except Exception as e:
            print(f"[g1_mujoco] viewer camera framing skipped: {e}")
        view_sync = ViewerSync(model, data, handle, physics_lock, viewer_data,
                               draw=band_overlay(band, band_body))
        view_sync.start()
    # The fall logger logs once when the pelvis tips or drops, with how long the
    # robot stood after the band's release.
    _fall_logged = False
    _release_t = None
    pelvis = resolver.base_body
    # RoboCasa updates its fixtures once per control step (its timers count
    # update_state() calls), so the loop updates them and reads the task's
    # success at env.control_freq (20 Hz) of sim time, not every physics step.
    task_every = max(1, round(1.0 / (env.control_freq * model.opt.timestep)))
    steps = 0
    max_steps = round(duration / model.opt.timestep) if duration > 0 else None
    done = _success = False
    _resets = 0
    pacer = RealtimePacer(model.opt.timestep, sim_rate, max_lag=MAX_LAG)
    # The real-time factor log's sim time, wall time and time slept since its last line.
    log_sim, log_wall, slept = float(data.time), time.time(), 0.0

    try:
        while max_steps is None or steps < max_steps:
            if viewer and not handle.is_running():
                break
            with physics_lock:
                if view_sync is not None:
                    view_sync.apply_perturbation()
                apply_band(model, data, band, band_body, band_vel)
                sim_interface.write_ctrl()
                _before = float(data.time)
                mujoco.mj_step(model, data)
                # MuJoCo's auto-reset clears the warning counters and then counts
                # the one warning, so they are cleared here to see every reset;
                # the reset also puts the sim time back to 0.
                if any(data.warning[w].number for w in UNSTABLE) or data.time < _before:
                    for w in UNSTABLE:
                        data.warning[w].number = 0
                    _resets += 1
                    sim_interface.physics_reset()
                    print(f"[g1_mujoco] *** WARNING: the physics went unstable and MuJoCo reset the "
                          f"whole scene, robot and kitchen, to its initial state and the sim time to 0 "
                          f"(reset {_resets} at step {steps}); restart the sim ***", flush=True)
                sim_interface.after_step()
            if _release_t is None and not band.enabled:
                _release_t = float(data.time)
                print(f"[fall-logger] elastic band released at t={data.time:.2f}s", flush=True)
            if not _fall_logged:
                _upz = float(data.xmat[pelvis][8])   # pelvis z-axis . world up (1: upright)
                _pelvis_h = float(data.xpos[pelvis][2])
                if _upz < FALL_UPRIGHTNESS or _pelvis_h < FALL_PELVIS_HEIGHT:
                    _fall_logged = True
                    _since = (f"{data.time - _release_t:.2f}s after band release"
                              if _release_t is not None
                              else f"{data.time:.2f}s (band still on)")
                    print(f"[fall-logger] *** ROBOT FELL at sim t={data.time:.2f}s — stood "
                          f"{_since} (uprightness={_upz:.2f}, pelvis_h={_pelvis_h:.2f}) ***",
                          flush=True)
            ros_bridge.tick()
            steps += 1
            if steps % task_every == 0:
                with physics_lock:
                    env.update_state()          # before _check_success
                done = measurement.tick()
            if done and not _success:
                print("[g1_mujoco] task success (debounced).")
            _success = done
            slept += pacer.wait()
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
        if view_sync is not None:
            view_sync.stop()
        if handle is not None:
            handle.close()
        executor.shutdown()
        ros_bridge.shutdown()
        for node in (measurement, ros_bridge):
            node.destroy_node()
        sim_interface.shutdown()
        shutdown_ros()


def main(argv):
    """Run the G1 sim with the command-line arguments argv (without the program name)."""
    parser = argparse.ArgumentParser(description="Run the G1 RoboCasa MuJoCo sim")
    sim_common.add_common_args(
        parser,
        task_help="RoboCasa env name (e.g. TurnOnToasterOven), Battery for the ARPA "
                  "battery workcell (CL_Assets/mujoco_assets/battery), or Blank for the robot "
                  "alone on a floor (no --layout or --style). "
                  "Omit to launch a random registered kitchen task.",
    )
    args = sim_common.parse_args(parser, argv, "g1")

    task = args.task
    if task is None:
        task = sim_common.random_task(seed=args.seed)
        print(f"[g1_mujoco] no --task given; randomly selected {task!r}")
    else:
        task = sim_common.resolve_task(task, seed=args.seed, tag="g1_mujoco",
                                       robot_task=lambda name: battery_env.canonical_task(name)
                                       or blank_env.canonical_task(name))
    sim_common.refuse_kitchen_args(parser, args, task, blank_env.TASKS)

    sim_loop(task, viewer=not args.headless,
             layout=args.layout, style=args.style, seed=args.seed, timing=args.timing,
             duration=args.duration)
