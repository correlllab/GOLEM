"""The Blank task (golem_robocasa/blank_env.py) for both robots: the env builds,
holds only the floor and the robot, spawns the robot at the origin facing +x,
and the elastic band holds the limp robot up.

Each robot is spawned and banded as its runner does it: the G1 with g1_mujoco's
own spawn() and make_band(); the H1-2 with h12_mujoco's spawn pose
and placement call and its band's values, which live inline in its sim_loop.
No motor is driven (ctrl 0), as in both sims before the first rt/lowcmd.

Needs RoboCasa and MuJoCo, so it runs in the robocasa container:
ROS_DOMAIN_ID=187 tests/robocasa/run_in_container.sh
"""
import functools
import sys
import types
import unittest

import mujoco
import numpy as np

import blank_env

STEPS = 500            # physics steps with the band on
# m the pelvis may sink below its spawn height while hanging: the feet drop onto
# the floor from the spawn clearance and the band stretches under the robot's
# weight (the H1-2 sinks ~4 cm, the G1 under 1 cm).
MAX_PELVIS_DROP = 0.10
# m: the lowest robot geom's height above the floor at spawn is at most this
# (the runners' clearances: 0.02 for the H1-2, 0.001 for the G1).
SPAWN_CLEARANCE_MAX = 0.03
# The H1-2's band, as h12_mujoco.sim_loop builds it.
H12_BAND = dict(dz=0.01, stiffness=1e4, damping=2e3)
# Bodies robosuite's Arena adds to every scene: mocap end-effector targets for
# its GUI, with non-colliding geoms.
ARENA_MOCAP_BODIES = {"left_eef_target", "right_eef_target"}
ROBOT_PREFIXES = ("robot0_", "gripper0_")


def _h12_mujoco():
    """h12_mujoco as a module. Its gripper bridge needs magpie_msgs, which this
    container's shell does not source; the bridge is not used here."""
    if "magpie_hand_bridge" not in sys.modules:
        stub = types.ModuleType("magpie_hand_bridge")
        stub.MagpieHandBridge = None
        sys.modules["magpie_hand_bridge"] = stub
    import h12_mujoco
    return h12_mujoco


@functools.lru_cache(maxsize=None)
def _h12():
    """(env, model, data, pelvis body, band, band body, lowest robot geom z)
    for the H1-2, spawned"""
    import h1_2_robosuite
    from mujoco_env import ElasticBand
    from sim_names import NameResolver
    h12_mujoco = _h12_mujoco()
    env = h1_2_robosuite.make_kitchen_env("Blank")
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    resolver = NameResolver(model)
    init_qpos = h12_mujoco._initial_motor_qpos()
    data.qpos[resolver.motor_qpos] = init_qpos
    data.qvel[:] = 0.0
    h1_2_robosuite.place_robot_collision_free(
        env, list(env.init_robot_base_pos),
        h1_2_robosuite._euler_to_wxyz(getattr(env, "init_robot_base_ori", None)),
        extra_backoff=0.0, margin=getattr(env, "spawn_margin", 0.20),
        poses=h12_mujoco._limp_upper_body_poses(model, resolver.motor_qpos, init_qpos))
    torso = resolver.body_id("torso_link")
    anchor = data.xpos[torso].copy()
    anchor[2] += H12_BAND["dz"]
    band = ElasticBand(anchor, length=0, stiffness=H12_BAND["stiffness"],
                       damping=H12_BAND["damping"])
    return env, model, data, resolver.body_id("pelvis"), band, torso, \
        h1_2_robosuite.lowest_robot_point_z(env)


@functools.lru_cache(maxsize=None)
def _g1():
    """(env, model, data, pelvis body, band, band body, lowest robot geom z)
    for the G1, spawned"""
    from g1 import g1_mujoco, g1_robosuite
    from g1.sim_names import NameResolver
    env = g1_robosuite.make_kitchen_env("Blank")
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    resolver = NameResolver(model, robot_prefix=env.robots[0].robot_model.naming_prefix)
    g1_mujoco.spawn(env, resolver)
    band, body = g1_mujoco.make_band(data, resolver)
    return env, model, data, resolver.base_body, band, body, \
        g1_robosuite.lowest_robot_point_z(env)


def _step_banded(model, data, band, body, steps):
    """Step with every motor off and the band pulling the torso, as both sims do"""
    vel = np.zeros(6)
    for _ in range(steps):
        data.ctrl[:] = 0.0
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
        data.xfrc_applied[body, :3] = band.evaluate_force(data.xpos[body], vel[3:])
        mujoco.mj_step(model, data)
        yield


class _BlankChecks:
    """The checks, for the robot `build` returns"""
    build = None

    def setUp(self):
        (self.env, self.model, self.data, self.pelvis, self.band, self.body,
         self.lowest) = type(self).build()

    def test_scene_is_the_floor_and_the_robot(self):
        model = self.model
        others = {}
        for geom in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])
            if not body.startswith(ROBOT_PREFIXES):
                others.setdefault(body, []).append(geom)
        self.assertEqual(set(others), {"world"} | ARENA_MOCAP_BODIES)
        self.assertEqual([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                          for g in others["world"]], ["floor"])
        for name in ARENA_MOCAP_BODIES:
            for geom in others[name]:
                self.assertEqual((model.geom_contype[geom], model.geom_conaffinity[geom]), (0, 0))

    def test_signals_are_stubbed(self):
        self.env.update_state()
        self.assertIs(self.env._check_success(), False)
        self.assertEqual(self.env.reward(), 0.0)
        self.assertTrue(self.env.get_ep_meta()["lang"])

    def test_spawn_stands_at_the_origin_facing_x(self):
        # Runs before the band test steps the shared sim (unittest sorts by name).
        pos, quat = self.data.xpos[self.pelvis], self.data.xquat[self.pelvis]
        np.testing.assert_allclose(pos[:2], (0.0, 0.0), atol=1e-6)
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        self.assertAlmostEqual(float(yaw), 0.0, places=6)
        self.assertGreater(self.lowest, 0.0)
        self.assertLess(self.lowest, SPAWN_CLEARANCE_MAX)

    def test_the_band_holds_the_pelvis_up(self):
        model, data = self.model, self.data
        start = float(data.xpos[self.pelvis][2])
        lowest = start
        for _ in _step_banded(model, data, self.band, self.body, STEPS):
            lowest = min(lowest, float(data.xpos[self.pelvis][2]))
        print(f"\n  {type(self).__name__}: pelvis z {start:.3f} m at spawn, lowest {lowest:.3f} m "
              f"over {STEPS} steps ({STEPS * model.opt.timestep:.1f} s)")
        self.assertTrue(self.band.enabled)
        self.assertGreater(lowest, start - MAX_PELVIS_DROP)


class H12BlankTests(_BlankChecks, unittest.TestCase):
    build = staticmethod(_h12)


class G1BlankTests(_BlankChecks, unittest.TestCase):
    build = staticmethod(_g1)


if __name__ == "__main__":
    unittest.main()
