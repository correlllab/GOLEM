"""Run the RoboCasa MuJoCo sim of the H1-2, or of the G1 with --g1 or GOLEM_ROBOT=g1.

Usage: golem_sim.py [--h12 | --g1] [--headless] [--task NAME] [--layout N]
[--style N] [--seed N] [--no_band_lowering]. One loop (sim_loop) runs both robots; what
differs between them is a robot object with one interface (SimRobot below): h12_robot.H12Robot or
g1.g1_robot.G1Robot. The robot is chosen (sim_common.select_robot) and the command line
parsed before either robot's module is imported: the H1-2's needs magpie_msgs and the
H1-2's CL_Assets files, which a G1 container may not have. The renderer is checked next
(sim_common.require_renderer: the sim stops there without a GPU, unless GOLEM_RENDER=cpu),
so --help and a usage error need neither.
"""
import argparse
import sys
import threading
import time
from typing import Protocol

import sim_common

# The warnings on which MuJoCo resets the whole scene (robot and kitchen), by mjtWarning name.
UNSTABLE = ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC")
# s of sim time between the loop's real-time factor logs.
RTF_LOG_S = 30.0
TAG = "golem_sim"


class SimRobot(Protocol):
    """What sim_loop needs of a robot; h12_robot.H12Robot and g1.g1_robot.G1Robot provide it."""

    name: str           # 'h12' or 'g1'
    label: str          # the robot's name in log lines
    gil_switch_s: float | None   # s, Python's GIL switch interval while the sim runs; None keeps Python's

    def make_env(self, task, layout, style, seed):
        """Return the robot's robosuite env for task, not yet reset."""

    def make_resolver(self, env):
        """Return the robot's name resolver for the reset env's model."""

    def spawn(self, env, resolver):
        """Put the robot at rest at its spawn, clear of the scene; reads the robot's spawn knobs."""

    def make_band(self, data, resolver, lowering=True):
        """Return (ElasticBand, body id) of the band that holds the robot at its spawn.

        lowering True hangs the robot with its feet clear of the floor, for band_lowering;
        False is the fixed anchor of --no_band_lowering.
        """

    def band_lowering(self, band):
        """Return the mujoco_env.BandLowering that lowers make_band's hanging band onto the feet."""

    def odom_body(self, env, resolver):
        """Return the id of the body the opt-in ground-truth odom reports."""

    def make_interface(self, model, data, resolver):
        """Return the robot's Unitree DDS interface (write_ctrl, after_step, physics_reset, shutdown)."""

    def make_lock(self, viewer):
        """Return (physics lock, sensor lock): the context the loop steps under, and the lock the robot's sensors and extras take (None: none)."""

    def make_sensors(self, node, env, model, data, lock):
        """Return the robot's sensor suite (mujoco_ros_bridge.SensorSuite) on node."""

    def start_extras(self, env, model, data, lock):
        """Start what only this robot serves; return (ROS nodes for the executor, viewer key callbacks)."""

    def tick_extras(self, sim_t):
        """Publish the extras' state due at sim time sim_t (s)."""

    def stop_extras(self):
        """Stop the extras; their nodes are destroyed by the loop afterwards."""

    def make_fall_logger(self, resolver, band_body):
        """Return the robot's mujoco_env.FallLogger."""

    def describe(self, model, resolver):
        """Return the log text naming what the resolver mapped."""

    def surface(self):
        """Return the topics the sim serves, for the 'ROS bridges up' line."""


def _parser() -> argparse.ArgumentParser:
    """Return the runner's argument parser."""
    parser = argparse.ArgumentParser(description="Run the RoboCasa MuJoCo sim of the H1-2 or the G1")
    sim_common.add_common_args(
        parser,
        task_help="RoboCasa env name (e.g. TurnOnToasterOven), Battery for the "
                  "battery workcell (CL_Assets/mujoco_assets/battery), or Blank for "
                  "the robot alone on a floor (no --layout or --style). "
                  "Omit to launch a random registered kitchen task.",
    )
    parser.add_argument(
        "--no_band_lowering",
        action="store_true",
        help="Keep the elastic band's anchor fixed where it starts, the feet on the floor, instead of "
             "hanging the robot and lowering it onto its feet once the legs are held (default: lowering)",
    )
    return parser


def make_robot(name):
    """Import and return the robot object for name ('h12' or 'g1').

    Only the chosen robot's modules are imported.
    """
    if name == "g1":
        from g1.g1_robot import G1Robot
        return G1Robot()
    from h12_robot import H12Robot
    return H12Robot()


def sim_loop(robot, task, viewer=True, layout=None, style=None, seed=None, duration=0.0,
             band_lowering=True):
    """Run robot (a SimRobot) in task (a RoboCasa kitchen, Battery or Blank) until stopped.

    The loop steps the env's own MjData (never env.step()) and drives:
      - the robot's SimInterface : Unitree DDS rt/lowstate and rt/lowcmd (and what else the
                                   robot speaks); every motor off until the first valid rt/lowcmd
      - RosSensorBridge          : /clock, the robot's sensor suite, and ground-truth /odom
                                   only with GOLEM_SIM_ODOM=1
      - MeasurementBridge        : /robocasa/*, /elastic_band/toggle
      - the robot's extras       : what only it serves (the H1-2's grippers and estop box)
    An elastic band on the torso holds the robot until /elastic_band/toggle or SPACE in the
    viewer releases it. With band_lowering (the default) it hangs the robot with its feet clear
    of the floor and lowers it onto them once the legs are held (mujoco_env.BandLowering, the
    training env's hoist); without it, its anchor stays fixed with the feet on the floor. The
    loop paces to GOLEM_SIM_RATE (sim_common.sim_rate_from_env; default real time) and logs
    its real-time factor every RTF_LOG_S of sim time. duration is
    sim seconds from the first step, counted in physics steps (0: until the viewer closes or
    Ctrl-C); the tests pass it, the command line does not.
    """
    import mujoco
    import numpy as np
    import rclpy.executors

    from measurement_bridge import MeasurementBridge
    from mujoco_env import apply_band
    from mujoco_ros_bridge import RosSensorBridge, init_ros, shutdown_ros
    from sim_timing import RealtimePacer
    from viewer_sync import close_viewer, open_viewer

    # A bad GOLEM_SIM_RATE stops the sim before the kitchen is built.
    sim_rate = sim_common.sim_rate_from_env()
    env = robot.make_env(task, layout, style, seed)

    # robosuite.make() does not reset; reset() builds env.sim, runs the robot's placement
    # patch and places the task objects.
    env.reset()
    # The fixtures' own parts stop colliding with each other and with the static scene; the
    # robot and the objects collide with everything as before.
    sim_common.exclude_fixture_collisions(env)

    model = env.sim.model._model
    data = env.sim.data._data
    resolver = robot.make_resolver(env)

    try:
        robot.spawn(env, resolver)
    except Exception as e:
        print(f"[{TAG}] clean reset skipped: {e}")

    # The band holds the robot from the first physics step; every motor is off until the
    # first valid rt/lowcmd.
    band, band_body = robot.make_band(data, resolver, band_lowering)
    band_vel = np.zeros(6)   # apply_band's scratch
    lowering = robot.band_lowering(band) if band_lowering else None
    print(f"[{TAG}] elastic band on the torso anchored at {band.point.round(3)}"
          + (f", a rope that lifts the robot {lowering.lift:.2f} m and lowers it {lowering.drop:.2f} m "
             "once the legs are held" if lowering else ", fixed (--no_band_lowering)")
          + "; /elastic_band/toggle or SPACE in the viewer releases it")

    print(f"[{TAG}] RoboCasa task {task!r} built around {robot.label}: "
          f"nq={model.nq} nu={model.nu}; {robot.describe(model, resolver)}")

    # Ground-truth odom (odom -> pelvis on /tf, /odom), only with GOLEM_SIM_ODOM: the real
    # robot has none, and it would compete with FAST-LIO's odometry.
    sim_odom = sim_common.env_flag('GOLEM_SIM_ODOM', False)
    odom_body = robot.odom_body(env, resolver) if sim_odom else -1
    print(f"[{TAG}] ground-truth sim odom {'ON (GOLEM_SIM_ODOM=1)' if sim_odom else 'OFF'}")

    sim_interface = robot.make_interface(model, data, resolver)
    # For the closed-loop tests that record the sim from inside its process.
    sim_interface.band = band

    if robot.gil_switch_s is not None:
        sys.setswitchinterval(robot.gil_switch_s)
    init_ros()

    # physics_lock is held around every write of MjData by this loop; sensor_lock is what
    # the robot's sensors and extras take to read or write it from other threads (None when
    # they only work on copies the loop hands them). Each robot says which it needs.
    physics_lock, sensor_lock = robot.make_lock(viewer)
    ros_bridge = RosSensorBridge(
        model, data,
        lambda node: robot.make_sensors(node, env, model, data, sensor_lock),
        base_body_id=odom_body,
        sim_lock=sensor_lock,
    )
    print(f"[{TAG}] {sim_common.describe_sim_rate(sim_rate)}", flush=True)

    extra_nodes, key_callbacks = robot.start_extras(env, model, data, sensor_lock)

    measurement = MeasurementBridge(env, elastic_band=band, task_name=task)
    print(f"[{TAG}] task goal: {measurement.publish_goal()!r}")

    # The executor thread serves the band toggle and the robot's extras. ros_bridge is not
    # on it: the loop calls its tick() between steps (a renderer's GL context is
    # thread-affine). The thread starts just before the loop's try, whose finally shuts it down.
    executor = rclpy.executors.MultiThreadedExecutor()
    nodes = [measurement, *extra_nodes]
    for node in nodes:
        executor.add_node(node)

    # The viewer draws its own MjData, refreshed from data by the viewer thread, with the
    # band while it is enabled; its mouse perturbation forces come back through
    # view_sync.apply_perturbation().
    handle = view_sync = None
    if viewer:
        handle, view_sync = open_viewer(model, data, physics_lock, band, band_body, TAG,
                                        sim_rate=sim_rate, key_callbacks=key_callbacks)
    # Logs the band's release and the robot's first fall.
    fall_logger = robot.make_fall_logger(resolver, band_body)
    # RoboCasa updates its fixtures and checks success once per control step (20 Hz) of sim
    # time, not per physics step: several fixtures count update_state() calls (toaster heat,
    # scrub contact), and the success debounce counts control steps.
    task_every = max(1, round(env.control_timestep / model.opt.timestep))
    steps = 0
    max_steps = round(duration / model.opt.timestep) if duration > 0 else None
    done = success = False
    resets = 0
    unstable = [getattr(mujoco.mjtWarning, name) for name in UNSTABLE]
    pacer = RealtimePacer(model.opt.timestep, sim_rate)
    # The real-time factor log's sim time, wall time and time slept since its last line.
    log_sim, log_wall, slept = float(data.time), time.time(), 0.0

    executor_thread = threading.Thread(target=executor.spin, daemon=True, name="ros_bridge_exec")
    executor_thread.start()
    print(f"[{TAG}] ROS bridges up: {robot.surface()}, /robocasa/*, /elastic_band/toggle")
    try:
        while max_steps is None or steps < max_steps:
            if viewer and not handle.is_running():
                break
            with physics_lock:
                if view_sync is not None:
                    view_sync.apply_perturbation()
                if lowering is not None:
                    lowering.step(float(data.time), sim_interface)
                apply_band(model, data, band, band_body, band_vel)
                sim_interface.write_ctrl()
                before = float(data.time)
                mujoco.mj_step(model, data)
                # MuJoCo's auto-reset clears the warning counters and then counts the one
                # warning, so they are cleared here to see every reset; the reset also puts
                # the sim time back to 0.
                if any(data.warning[w].number for w in unstable) or data.time < before:
                    for w in unstable:
                        data.warning[w].number = 0
                    resets += 1
                    sim_interface.physics_reset()
                    print(f"[{TAG}] *** WARNING: the physics went unstable and MuJoCo reset the "
                          f"whole scene, robot and kitchen, to its initial state and the sim time to 0 "
                          f"(reset {resets} at step {steps}); restart the sim ***", flush=True)
                sim_interface.after_step()
            fall_logger.step(data, band)
            # tick() takes the sensors' lock only to capture their state, so encoding and
            # publishing do not hold up physics. No physics step runs inside it.
            ros_bridge.tick()
            robot.tick_extras(float(data.time))
            steps += 1
            if steps % task_every == 0:
                # The fixtures' update writes data, which the other readers copy under the lock.
                with physics_lock:
                    env.update_state()          # before _check_success
                    done = measurement.tick()
            if done and not success:
                print(f"[{TAG}] task success (debounced).")
            success = done
            slept += pacer.wait()
            if data.time < log_sim:   # MuJoCo reset the time
                log_sim, log_wall, slept = float(data.time), time.time(), 0.0
            elif data.time - log_sim >= RTF_LOG_S:
                wall = time.time() - log_wall
                stats = ros_bridge.stats()
                print(f"[{TAG}] sim t {data.time:.0f} s: real-time factor {(data.time - log_sim) / wall:.2f} "
                      f"over the last {data.time - log_sim:.0f} s, the loop idle {slept / wall:.0%} of the time"
                      + (f"; {stats}" if stats else ""), flush=True)
                log_sim, log_wall, slept = float(data.time), time.time(), 0.0
    finally:
        # The extras first: a DeliGrasp settle waits on the sim clock, which no longer
        # advances, and must end before the executor's worker thread can exit.
        robot.stop_extras()
        close_viewer(handle, view_sync)
        executor.shutdown()
        ros_bridge.shutdown()
        for node in (*nodes, ros_bridge):
            node.destroy_node()
        sim_interface.shutdown()
        shutdown_ros()


def main(argv):
    """Run the sim with the command-line arguments argv (without the program name)."""
    name = sim_common.select_robot(argv)
    parser = _parser()
    args = sim_common.parse_args(parser, argv, name)
    # The cameras and the lidar's rays render through EGL on the GPU: stop here without
    # one, unless GOLEM_RENDER=cpu renders on the CPU.
    sim_common.require_renderer()
    robot = make_robot(name)
    task = sim_common.select_task(parser, args, tag=TAG, own_tasks=robot.own_tasks,
                                  no_kitchen=robot.no_kitchen)
    sim_loop(robot, task, viewer=not args.headless, layout=args.layout, style=args.style, seed=args.seed,
             band_lowering=not args.no_band_lowering)


if __name__ == "__main__":
    main(sys.argv[1:])
