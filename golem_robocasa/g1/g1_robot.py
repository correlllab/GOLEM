"""Define the G1 for golem_sim.sim_loop: its spawn, band, sensors and Dex3-1 hands' DDS interface."""
import contextlib
import os
import threading

import numpy as np

from . import g1_robosuite  # registers G1Dex3 and its gripper
import battery_env  # registers the Battery env
import blank_env  # registers the Blank env
import robosuite_common
from mujoco_env import ElasticBand, FallLogger
from .g1_sensors import G1Sensors, SensorConfig
from .sim_names import BODY_JOINTS, NameResolver
from .unitree_interface import SimInterface

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
# m of free space the spawn keeps from every robot collision geom to any non-floor geom, in
# the spawn pose, the limp poses and along the start-up arm moves (GOLEM_SPAWN_MARGIN
# overrides it).
SPAWN_MARGIN = 0.05
# Slots in sim_names.BODY_JOINTS: 12-14 the waist (yaw, roll, pitch), 15-21 the left arm,
# 22-28 the right, each arm shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw.
_WAIST_PITCH, _ARMS = 14, (15, 22)
# The poses (rad) the unpowered arms pass through under the band, as (shoulder pitch,
# shoulder yaw outward, elbow, wrist pitch): where they come to rest, and the far end of
# their first swing. Measured over 30 s limp in the Blank task and in a kitchen (the same
# to the milliradian): the forearms drop from pointing forward, the elbows swinging to
# 1.95 rad and resting at 1.31, the shoulders pitching to -0.37, the wrists to -0.59; the
# waist stays within 0.08 rad.
LIMP_ARM_POSES = ((-0.06, 0.07, 1.31, -0.10), (-0.37, 0.07, 1.95, -0.59))
LIMP_WAIST_PITCH = (0.08, -0.07)   # rad, with each arm pose above
# The arm postures (rad, left arm then right) golem_upper_ik's frame_task_server moves through
# at start, while the band still holds the robot at its spawn: golem_upper_ik/config/g1/
# robot.yaml's startup_sequence and named_configs, copied because the sim reads no controller
# config (tests/docker/test_sim_startup_poses.py keeps them equal).
STARTUP_ARM_STAGES = (
    ("home", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
)


# s, Python's GIL switch interval while the sim runs (default 5 ms). The physics
# loop sleeps between steps to keep its pace and needs the GIL back to wake;
# at 5 ms the sensor threads' Python work (the head camera's messages) stalled
# rt/lowstate 5-9 ms, where the real G1 sends it every 2 ms. At 0.5 ms the sim
# with both sensors had no gap over 5 ms in 51 s of sim time.
GIL_SWITCH_S = 0.0005


def _limp_upper_body_poses(motor_qpos, spawn):
    """Return the poses the unpowered upper body sags into, as (qpos indices, values) pairs.

    The spawn clearance must hold in them too, since every motor is off until the first
    command (LIMP_ARM_POSES).
    """
    poses = []
    for (pitch, yaw, elbow, wrist), waist in zip(LIMP_ARM_POSES, LIMP_WAIST_PITCH):
        q = np.array(spawn, dtype=float)
        q[_WAIST_PITCH] = waist
        for base, side in zip(_ARMS, (1.0, -1.0)):   # the right arm's yaw mirrors the left's
            q[base + 0] = pitch
            q[base + 2] = side * yaw
            q[base + 3] = elbow
            q[base + 5] = wrist
        poses.append((motor_qpos, q))
    return poses


def spawn(env, resolver, backoff=0.0, margin=SPAWN_MARGIN):
    """Put the robot at rest in g1_robosuite.STANCE, hands open, clear of the fixtures by margin, in the sagged poses too.

    Call after env.reset(), whose zero-action settling folds the limp robot
    (and whose initialization noise perturbs every joint). The robot is placed
    at RoboCasa's spawn, its lowest collision geom SPAWN_CLEARANCE above the
    floor, and backed off (its -x), shifting sideways where that is not enough
    (robosuite_common.place_robot_collision_free), until every robot collision
    geom is margin (m) from any non-floor geom, in the spawn pose, the poses the
    unpowered upper body sags into and along the controller's start-up arm moves
    (STARTUP_ARM_STAGES); backoff (m) then backs it up to that much further into
    open floor. Zeroes every velocity in the scene.
    """
    data = env.sim.data._data
    stance = np.array([g1_robosuite.STANCE.get(n, 0.0) for n in BODY_JOINTS], dtype=float)
    data.qpos[resolver.body.qpos] = stance
    for hand in resolver.hands.values():
        data.qpos[hand.qpos] = 0.0
    data.qvel[:] = 0.0
    g1_robosuite.place_robot_collision_free(
        env, env.init_robot_base_pos,
        robosuite_common.euler_to_wxyz(getattr(env, "init_robot_base_ori", None)),
        clearance=SPAWN_CLEARANCE, extra_backoff=backoff, margin=margin,
        poses=(_limp_upper_body_poses(resolver.body.qpos, stance)
               + robosuite_common.startup_arm_poses(resolver.body.qpos, stance, STARTUP_ARM_STAGES,
                                                    _ARMS[0])),
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


class G1Robot:
    """The G1 with its Dex3-1 hands, in Unitree's debug mode, as golem_sim.SimRobot."""

    name = "g1"
    label = "the G1"
    own_tasks = battery_env.TASKS + blank_env.TASKS
    no_kitchen = blank_env.TASKS
    gil_switch_s = GIL_SWITCH_S

    def __init__(self):
        """Read the sensor knobs, so a bad one fails before the slow env build."""
        self.sensors = SensorConfig.from_env()

    def make_env(self, task, layout, style, seed):
        """Return the G1's robosuite env for task, not yet reset."""
        return g1_robosuite.make_kitchen_env(task, layout_ids=layout, style_ids=style, seed=seed)

    def make_resolver(self, env):
        """Return the G1's sim_names.NameResolver."""
        return NameResolver(env.sim.model._model, robot_prefix=env.robots[0].robot_model.naming_prefix)

    def spawn(self, env, resolver):
        """Spawn the G1 (spawn above) with the spawn knobs.

        GOLEM_SPAWN_BACKOFF (m, default 0: at the counter) backs the robot further into open
        floor; Nav2 needs a free start cell to plan (~1.0). GOLEM_SPAWN_MARGIN (m, unset:
        SPAWN_MARGIN) is the clearance to the fixtures.
        """
        margin = os.environ.get("GOLEM_SPAWN_MARGIN", "").strip()
        spawn(env, resolver, backoff=float(os.environ.get("GOLEM_SPAWN_BACKOFF", "0") or 0),
              margin=float(margin) if margin else SPAWN_MARGIN)
        print("[g1_robot] initial stance: legs at Unitree's default angles, everything else at 0")

    def make_band(self, data, resolver):
        """Return (band, torso body id): the band on the torso, anchored BAND_ANCHOR_DZ above it now."""
        return make_band(data, resolver)

    def odom_body(self, env, resolver):
        """Return the pelvis."""
        return resolver.base_body

    def make_interface(self, model, data, resolver):
        """Return the G1's Unitree DDS interface: rt/lowstate, rt/lowcmd and rt/dex3/{left,right}/*."""
        return SimInterface(model, data, resolver)

    def make_lock(self, viewer):
        """Return (the viewer's lock or a null context, None).

        The physics loop is the only writer of MjData: the sensor threads work on copies of
        the state tick() takes, and the ROS and DDS callbacks only latch values the loop
        reads, so the sensors take no lock. The one other reader is the viewer's sync thread,
        which copies the state under the physics lock (a null context headless, where nothing
        else reads).
        """
        return (threading.Lock() if viewer else contextlib.nullcontext()), None

    def make_sensors(self, node, env, model, data, lock):
        """Return the G1's sensor suite: the MID-360 and the head D435, as SensorConfig switches them."""
        print(f"[g1_robot] sensors: {self.sensors.describe()}")
        return G1Sensors(node, model, data, prefix=env.robots[0].robot_model.naming_prefix,
                         config=self.sensors)

    def start_extras(self, env, model, data, lock):
        """Start nothing: the G1's hands are on its DDS interface, and it has no estop box."""
        return [], ()

    def tick_extras(self, sim_t):
        """Publish nothing."""

    def stop_extras(self):
        """Stop nothing."""

    def make_fall_logger(self, resolver, band_body):
        """Return the logger of the band's release and the first time the pelvis tips or drops."""
        return FallLogger(resolver.base_body, "pelvis", upright_min=FALL_UPRIGHTNESS,
                          height_min=FALL_PELVIS_HEIGHT)

    def describe(self, model, resolver):
        """Return how many body and hand motors the resolver mapped."""
        return (f"resolver mapped {len(resolver.body.names)} body "
                f"+ {sum(len(h.names) for h in resolver.hands.values())} hand motors")

    def surface(self):
        """Return the G1 sim's topics, with the sensors that are on."""
        return ("rt/lowstate, rt/lowcmd, rt/dex3/{left,right}/{cmd,state}, /clock"
                + (", /livox/lidar, /livox/imu" if self.sensors.lidar else "")
                + (", /realsense/head/*" if self.sensors.camera else ""))
