import argparse
import copy
import math
import os
import random
import sys
import threading
import time

import mujoco
import numpy as np
import rclpy.executors

from .mujoco_ros_bridge import init_ros, shutdown_ros
import mujoco.viewer

from . import g1_robosuite  # registers G1Dex3 + its grippers, patches RoboCasa
from measurement_bridge import MeasurementBridge
from .mujoco_ros_bridge import RosSensorBridge, SensorConfig
from .sim_names import BODY_JOINTS, NameResolver
from .stand_in import StandIn
from .unitree_interface import SimInterface

# Viewer free-camera spawn pose, anchored to the robot so the passive viewer
# opens framed on the robot wherever it spawns (RoboCasa places the base by
# layout/seed and place_robot_collision_free may back it off), instead of
# MuJoCo's default whole-scene framing. Top-down (straight down) view: lookat
# follows the robot base body, elevation looks straight down, and azimuth (the
# in-plane rotation of the top-down image) is offset from the base yaw so the
# robot's facing points a consistent way in frame at any spawn orientation.
# Spawn-time only — the user can orbit/zoom freely after.
VIEW_CAM_DISTANCE  = 3.0     # m, camera height above lookat (straight down)
VIEW_CAM_AZIMUTH   = 90.0    # deg, ADDED to robot yaw — rotates the top-down image
VIEW_CAM_ELEVATION = -90.0   # deg, -90 = look straight down
VIEW_CAM_LOOKAT_DZ = 0.0     # m, focal point offset along z (irrelevant for top-down)

# The fall logger's test, as mujoco_deploy_g1.py's: the pelvis tilted more than
# 60 deg from upright, or down on the floor.
FALL_UPRIGHTNESS = 0.5   # pelvis z-axis . world up
FALL_PELVIS_HEIGHT = 0.2  # m

# HOMIE stands more crouched than the stance the robot spawns in (at its 0.74 m
# height target, knees ~0.76 rad instead of 0.3 and the pelvis at 0.744 m instead
# of ~0.78 m), so from the stance it settles for ~1.5 s, moving the pelvis ~7 cm
# back over its feet. The sim settles that on a copy of the state before it
# starts, then puts the settled robot back where the spawn placed it.
SETTLE_SECONDS = 3.0
# The spawn puts the lowest collision geom this far above the floor: a 2 cm drop
# onto the feet makes HOMIE slide them ~3 cm while it settles.
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


def _frame_viewer_on_robot(handle, data, body_id):
    """Set the passive viewer free camera's spawn pose as a function of the robot
    base body's world pose. lookat tracks the body position; the orbit azimuth is
    offset from the body's yaw, so framing is invariant to spawn orientation."""
    pos = data.xpos[body_id]
    w, x, y, z = (float(v) for v in data.xquat[body_id])
    yaw_deg = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    cam = handle.cam
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (float(pos[0]), float(pos[1]), float(pos[2]) + VIEW_CAM_LOOKAT_DZ)
    cam.distance = VIEW_CAM_DISTANCE
    cam.azimuth = yaw_deg + VIEW_CAM_AZIMUTH
    cam.elevation = VIEW_CAM_ELEVATION


def _settle_stance(env, resolver, stand_in, seconds=SETTLE_SECONDS):
    """Let the stand-in settle the robot on a copy of the state, so the sim
    starts with the robot standing still in HOMIE's own posture; the kitchen's
    objects and the sim time are untouched. The settled joints, velocities and
    pelvis orientation are copied back, and the pelvis is put back where the
    spawn placed it: place_robot_collision_free fits it to the floor again and
    checks the settled posture for collisions (backing off if it must). The
    policy sees neither the pelvis position nor its yaw, and the stand-in keeps
    its policy state, so the sim's first step continues it.
    Returns False, changing nothing, if the robot did not stay upright."""
    model = env.sim.model._model
    data = env.sim.data._data
    scratch = copy.copy(data)
    body = resolver.body
    base_q = np.arange(resolver.base_qpos, resolver.base_qpos + 7)
    base_v = np.arange(resolver.base_qvel, resolver.base_qvel + 6)
    placed = data.qpos[base_q[:3]].copy()
    for hand in resolver.hands.values():
        scratch.ctrl[hand.ctrl] = 0.0   # limp, as before any rt/dex3 command
    for _ in range(round(seconds / model.opt.timestep)):
        tau = stand_in.torques(scratch.qpos[body.qpos], scratch.qvel[body.qvel])
        scratch.ctrl[body.ctrl] = np.clip(tau, body.tau_min, body.tau_max)
        mujoco.mj_step(model, scratch)
        stand_in.after_step(scratch.qpos[body.qpos], scratch.qvel[body.qvel],
                            scratch.qpos[base_q[3:]], scratch.qvel[base_v[3:]])
    pelvis = resolver.base_body
    if scratch.xmat[pelvis][8] < FALL_UPRIGHTNESS or scratch.xpos[pelvis][2] < FALL_PELVIS_HEIGHT:
        return False
    hands = list(resolver.hands.values())
    qpos = np.concatenate([body.qpos] + [h.qpos for h in hands])
    qvel = np.concatenate([base_v, body.qvel] + [h.qvel for h in hands])
    data.qpos[qpos] = scratch.qpos[qpos]
    data.qvel[qvel] = scratch.qvel[qvel]
    g1_robosuite.place_robot_collision_free(env, placed, scratch.qpos[base_q[3:]],
                                            clearance=SPAWN_CLEARANCE)
    return True


def sim_loop(task, viewer=True, layout=None, style=None, seed=None, duration=0.0):
    """Launch a RoboCasa task env around the G1 and run the shared-MjData loop.

    Builds the env with robots='G1Dex3' and steps the env's *own* MjData with our
    loop (never env.step()), driving the sim<->ROS bridge layer:
      - SimInterface     : Unitree DDS, as the G1's own computers serve it:
                           rt/lowstate, rt/lowcmd, rt/arm_sdk, rt/dex3/{left,right}/*
      - StandIn          : Unitree's controller stand-in, HOMIE standing the robot
                           until the first valid rt/lowcmd releases it
      - RosSensorBridge  : /clock, the MID-360 (/livox/lidar, /livox/imu) and the
                           head D435 (/realsense/head/*), on unless GOLEM_LIDAR=0 /
                           GOLEM_CAMERAS=0 (SensorConfig), + ground-truth /odom
                           with GOLEM_SIM_ODOM=1
      - MeasurementBridge: /robocasa/{task_name,task_goal,success,reward}
    RoboCasa's _check_success/reward/lang are read off the shared env.
    duration: sim seconds to run from the first step, counted in physics steps
    (0 = until the viewer closes or Ctrl-C).
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

    # Clean, contact-free initial state. RoboCasa's zero-action settling loop
    # folds the limp robot during reset (and its initialization noise perturbs
    # every joint), so restore the standing stance (the legs at HOMIE's default
    # angles, every other joint at 0), zero all velocities, and re-place the
    # pelvis. place_robot_collision_free auto-fits floor clearance AND backs the
    # base away (the robot's -x) until no robot geom penetrates a fixture, keeping
    # the least-penetrating spot if it can't fully clear. The stand-in then
    # settles the stance (_settle_stance).
    try:
        data.qpos[resolver.body.qpos] = [g1_robosuite.STANCE.get(n, 0.0) for n in BODY_JOINTS]
        for hand in resolver.hands.values():
            data.qpos[hand.qpos] = 0.0
        data.qvel[:] = 0.0
        # GOLEM_SPAWN_BACKOFF backs the robot further off the counter into open
        # floor (default 0 = at the counter for manipulation). Set ~1.0 for nav2,
        # which needs the robot's start cell to be free to plan.
        _spawn_backoff = float(os.environ.get("GOLEM_SPAWN_BACKOFF", "0") or 0)
        g1_robosuite.place_robot_collision_free(
            env, env.init_robot_base_pos,
            g1_robosuite._euler_to_wxyz(getattr(env, "init_robot_base_ori", None)),
            clearance=SPAWN_CLEARANCE, extra_backoff=_spawn_backoff,
        )
        print("[g1_mujoco] initial stance: legs at HOMIE's default angles, everything else at 0")
    except Exception as e:
        print(f"[g1_mujoco] clean reset skipped: {e}")

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

    # The stand-in for Unitree's controller stands the robot from the first
    # physics step; the DDS bridge hands the waist and arms to rt/arm_sdk by its
    # weight, and every motor to the first valid rt/lowcmd.
    stand_in = StandIn(resolver.body.names, model.opt.timestep)
    spawn = data.xpos[resolver.base_body].copy()
    if _settle_stance(env, resolver, stand_in):
        pos = data.xpos[resolver.base_body]
        print(f"[g1_mujoco] stance settled under the stand-in ({SETTLE_SECONDS:.0f} s on a copy) and put "
              f"back at the spawn: pelvis {np.linalg.norm(pos[:2] - spawn[:2]) * 100:.1f} cm from it, "
              f"at z {pos[2]:.3f} m")
    else:
        print("[g1_mujoco] WARNING: the robot fell while settling the stance; starting from the stance")
        stand_in = StandIn(resolver.body.names, model.opt.timestep)
    sim_interface = SimInterface(model, data, resolver, stand_in)

    sys.setswitchinterval(GIL_SWITCH_S)
    init_ros()

    # Sensor bridge: /clock, the MID-360 and the head D435, and the optional
    # ground-truth odom
    ros_bridge = RosSensorBridge(
        model, data,
        prefix=pfx,
        sensors=sensors,
        base_body_id=odom_base_body_id,   # -> odom -> pelvis TF + /odom
        sim_lock=sim_lock,
    )
    print(f"[g1_mujoco] sensors: {sensors.describe()}")

    measurement = MeasurementBridge(env, task_name=task)
    print(f"[g1_mujoco] task goal: {measurement.publish_goal()!r}")

    # Background executor serves the nodes' ROS callbacks (parameter services).
    # ros_bridge.tick() is driven from the main loop instead (it reads MjData
    # between steps; the camera renders on the bridge's own thread), so it is
    # NOT added here.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(measurement)
    executor_thread = threading.Thread(target=executor.spin, daemon=True, name="ros_bridge_exec")
    executor_thread.start()
    print("[g1_mujoco] ROS bridges up: rt/lowstate, rt/lowcmd, rt/arm_sdk, "
          "rt/dex3/{left,right}/{cmd,state}, /clock, "
          + ("/livox/lidar, /livox/imu, " if sensors.lidar else "")
          + ("/realsense/head/*, " if sensors.camera else "") + "/robocasa/*")

    handle = mujoco.viewer.launch_passive(model, data) if viewer else None
    if handle is not None:
        handle.opt.geomgroup[0] = 0   # hide collision geoms by default
        # Frame the free camera on the robot's spawn pose (function of robot
        # position + yaw).
        cam_body_id = resolver.body_id("torso_link")
        if cam_body_id >= 0:
            try:
                with handle.lock():
                    _frame_viewer_on_robot(handle, data, cam_body_id)
            except Exception as e:
                print(f"[g1_mujoco] viewer camera framing skipped: {e}")
    # Fall logger: watch the pelvis's uprightness + height and log once when the
    # robot falls, including how long it stood after the stand-in was released.
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
                # the stand-in's policy at 50 Hz, rt/lowstate, rt/dex3/*/state
                sim_interface.after_step()
                # --- fall logger ---
                if _release_t is None and sim_interface.released:
                    _release_t = float(data.time)
                if not _fall_logged:
                    _upz = float(data.xmat[pelvis][8])   # pelvis z-axis . world up (1=upright)
                    _pelvis_h = float(data.xpos[pelvis][2])
                    if _upz < FALL_UPRIGHTNESS or _pelvis_h < FALL_PELVIS_HEIGHT:
                        _fall_logged = True
                        _since = (f"{data.time - _release_t:.2f}s after rt/lowcmd released the stand-in"
                                  if _release_t is not None
                                  else f"{data.time:.2f}s (stand-in still in control)")
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


def _runnable_tasks():
    """Return (env_classes, runnable).

    `env_classes` maps every robosuite-registered env name -> its class. This is
    the same REGISTERED_ENVS that robosuite.make() resolves against, and it DOES
    include RoboCasa's abstract bases (OpenDoor/CloseDoor/ManipulateDoor/
    PickPlace) — RoboCasa's metaclass deliberately omits those five from its own
    REGISTERED_KITCHEN_ENVS, but the robosuite parent metaclass still registers
    them. We need them here so an abstract base name can be looked up and expanded
    to its concrete subclasses.

    `runnable` is the set of concrete benchmark task names from RoboCasa's curated
    ATOMIC_/COMPOSITE_TASK_DATASETS (intersected with what's registered, to
    tolerate version skew). Those bases can't be instantiated directly — they
    require a `fixture_id` only a concrete subclass supplies — so they are never
    in `runnable`. Falls back to REGISTERED_KITCHEN_ENVS (which already excludes
    the five bases) if the curated lists are unavailable.
    """
    import robocasa  # noqa: F401  -> registers all kitchen envs on import
    from robosuite.environments.base import REGISTERED_ENVS
    env_classes = dict(REGISTERED_ENVS)
    try:
        from robocasa.utils.dataset_registry import (
            ATOMIC_TASK_DATASETS,
            COMPOSITE_TASK_DATASETS,
        )
        runnable = (set(ATOMIC_TASK_DATASETS) | set(COMPOSITE_TASK_DATASETS)) & set(env_classes)
    except Exception:
        from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS
        runnable = set(REGISTERED_KITCHEN_ENVS) & set(env_classes)
    return env_classes, runnable


def _random_task(seed=None):
    """Pick a random *runnable* (concrete) RoboCasa kitchen env name."""
    _, runnable = _runnable_tasks()
    if seed is not None:
        random.seed(seed)
    return random.choice(sorted(runnable))


def _resolve_task(name, seed=None):
    """Resolve a user-supplied --task name to a concrete runnable task.

    If `name` is already a concrete benchmark task, return it unchanged. If it's
    an abstract base (e.g. OpenDoor, ManipulateDoor) — registered but not directly
    instantiable — randomly pick one of its concrete runnable subclasses instead
    of crashing.
    """
    env_classes, runnable = _runnable_tasks()
    if name in runnable:
        return name
    if name not in env_classes:
        raise SystemExit(f"[g1_mujoco] unknown RoboCasa task {name!r}")
    base = env_classes[name]
    concrete = sorted(
        n for n in runnable
        if isinstance(env_classes.get(n), type)
        and issubclass(env_classes[n], base) and env_classes[n] is not base
    )
    if not concrete:
        raise SystemExit(f"[g1_mujoco] {name!r} is abstract with no runnable concrete tasks")
    if seed is not None:
        random.seed(seed)
    choice = random.choice(concrete)
    print(f"[g1_mujoco] {name!r} is an abstract task base; "
          f"randomly selected concrete subclass {choice!r} "
          f"(from {len(concrete)} options)")
    return choice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the G1 RoboCasa MuJoCo sim")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the MuJoCo passive viewer (default: viewer on)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="RoboCasa env name (e.g. TurnOnToasterOven). "
             "Omit to launch a random registered kitchen task.",
    )
    parser.add_argument("--layout", type=int, default=None, help="RoboCasa kitchen layout id")
    parser.add_argument("--style", type=int, default=None, help="RoboCasa kitchen style id")
    parser.add_argument("--seed", type=int, default=None, help="episode seed")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="sim seconds to run from the first step, then exit "
                             "(default 0: run until closed)")
    args = parser.parse_args()

    task = args.task
    if task is None:
        task = _random_task(seed=args.seed)
        print(f"[g1_mujoco] no --task given; randomly selected {task!r}")
    else:
        task = _resolve_task(task, seed=args.seed)

    sim_loop(task, viewer=not args.headless,
             layout=args.layout, style=args.style, seed=args.seed, duration=args.duration)
