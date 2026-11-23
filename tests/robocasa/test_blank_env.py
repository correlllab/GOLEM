"""Check the Blank task (golem_robocasa/blank_env.py) for both robots.

The env builds, holds only the floor and the robot, spawns the robot at the origin facing +x,
and the elastic band hangs the limp robot with its feet clear of the floor. Each robot is spawned
and banded as golem_sim does it, with its robot module's spawn(), make_band() and
band_lowering(). No motor is driven (ctrl 0), as in both sims before the first rt/lowcmd, so the
band never lowers.

Needs RoboCasa and MuJoCo, so it runs in the robocasa container:
ROS_DOMAIN_ID=187 tests/robocasa/run_in_container.sh
"""
import functools
import unittest

import mujoco
import numpy as np

from spawn_scene import import_h12_robot

STEPS = 1000  # physics steps with the band on: 2 s, the anchor's 1 s lift and a settle
# m the pelvis may sink below its spawn height: the H1-2 drops onto its feet from its 2 cm
# spawn clearance before the rising anchor takes its weight.
MAX_PELVIS_DROP = 0.10
# m: the lowest robot geom above the floor once the robot hangs (6-9 cm measured).
MIN_HANG_CLEARANCE = 0.04
# m: the most the lowest robot geom may stand above the floor at spawn (the runners'
# clearances are 0.02 for the H1-2 and 0.001 for the G1).
SPAWN_CLEARANCE_MAX = 0.03
# Bodies robosuite's Arena adds to every scene: mocap end-effector targets for its GUI, with
# non-colliding geoms.
ARENA_MOCAP_BODIES = {"left_eef_target", "right_eef_target"}
ROBOT_PREFIXES = ("robot0_", "gripper0_")


@functools.lru_cache(maxsize=None)
def _h12():
    """Spawn the H1-2 in the Blank task as its runner does, once per process.

    Returns (env, model, data, pelvis body, band, band body, lowering, lowest robot geom z, and
    (the --no_band_lowering band, the torso's spawn position, its anchor offset)).
    """
    import h1_2_robosuite
    import robosuite_common
    from sim_names import NameResolver
    h12_robot = import_h12_robot()
    env = h1_2_robosuite.make_kitchen_env("Blank")
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    resolver = NameResolver(model)
    h12_robot.spawn(env, resolver)
    band, torso = h12_robot.make_band(data, resolver)
    fixed = (h12_robot.make_band(data, resolver, lowering=False)[0], data.xpos[torso].copy(),
             h12_robot.BAND_FIXED_DZ)
    return env, model, data, resolver.body_id("pelvis"), band, torso, h12_robot.band_lowering(band), \
        robosuite_common.lowest_robot_point_z(env), fixed


@functools.lru_cache(maxsize=None)
def _g1():
    """Spawn the G1 in the Blank task as its runner does, once per process.

    Returns (env, model, data, pelvis body, band, band body, lowering, lowest robot geom z, and
    (the --no_band_lowering band, the torso's spawn position, its anchor offset)).
    """
    from g1 import g1_robot, g1_robosuite
    from g1.sim_names import NameResolver
    env = g1_robosuite.make_kitchen_env("Blank")
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    resolver = NameResolver(model, robot_prefix=env.robots[0].robot_model.naming_prefix)
    g1_robot.spawn(env, resolver)
    band, body = g1_robot.make_band(data, resolver)
    fixed = (g1_robot.make_band(data, resolver, lowering=False)[0], data.xpos[body].copy(),
             g1_robot.BAND_FIXED_DZ)
    return env, model, data, resolver.base_body, band, body, g1_robot.band_lowering(band), \
        g1_robosuite.lowest_robot_point_z(env), fixed


class _NoCommand:
    """Stand in for the robot's SimInterface before its first rt/lowcmd: no motor is held."""

    @staticmethod
    def holds(slots):
        """Return False: nothing commands the robot."""
        return False


def _step_banded(model, data, band, body, lowering, steps):
    """Step `steps` times with every motor off and the band moved and applied as both sims do; yield after each."""
    from mujoco_env import apply_band
    vel = np.zeros(6)
    for _ in range(steps):
        lowering.step(float(data.time), _NoCommand)
        data.ctrl[:] = 0.0
        apply_band(model, data, band, body, vel)
        mujoco.mj_step(model, data)
        yield


class _BlankChecks:
    """Check the Blank task for the robot that `build` spawns."""
    build = None

    def setUp(self):
        """Take the spawned robot from `build` (shared by every test of the class)."""
        (self.env, self.model, self.data, self.pelvis, self.band, self.body, self.lowering,
         self.lowest, self.fixed) = type(self).build()

    def test_scene_is_the_floor_and_the_robot(self):
        """The scene holds only the floor, the robot and the Arena's non-colliding mocap targets."""
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
        """The task reports no success, zero reward and a non-empty language goal."""
        self.env.update_state()
        self.assertIs(self.env._check_success(), False)
        self.assertEqual(self.env.reward(), 0.0)
        self.assertTrue(self.env.get_ep_meta()["lang"])

    def test_spawn_stands_at_the_origin_facing_x(self):
        """The robot spawns at the origin facing +x, just above the floor."""
        # This runs before the band test steps the shared sim: unittest sorts tests by name.
        pos, quat = self.data.xpos[self.pelvis], self.data.xquat[self.pelvis]
        np.testing.assert_allclose(pos[:2], (0.0, 0.0), atol=1e-6)
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        self.assertAlmostEqual(float(yaw), 0.0, places=6)
        self.assertGreater(self.lowest, 0.0)
        self.assertLess(self.lowest, SPAWN_CLEARANCE_MAX)

    def test_no_band_lowering_keeps_the_fixed_anchor(self):
        """golem_sim.py --no_band_lowering's band is anchored BAND_FIXED_DZ from the torso's origin, no rope, no hook."""
        band, torso, dz = self.fixed
        np.testing.assert_allclose(band.point - torso, (0.0, 0.0, dz))
        self.assertEqual((band.rope, band.hook), (False, None))
        self.assertEqual((band.stiffness, band.damping, band.length),
                         (self.band.stiffness, self.band.damping, 0))

    def test_the_band_hangs_the_robot(self):
        """With every motor off, the band keeps the pelvis within MAX_PELVIS_DROP of its spawn height and lifts the feet clear."""
        import robosuite_common
        model, data = self.model, self.data
        start = float(data.xpos[self.pelvis][2])
        lowest = start
        for _ in _step_banded(model, data, self.band, self.body, self.lowering, STEPS):
            lowest = min(lowest, float(data.xpos[self.pelvis][2]))
        clearance = robosuite_common.lowest_robot_point_z(self.env, colliding_only=True)
        print(f"\n  {type(self).__name__}: pelvis z {start:.3f} m at spawn, lowest {lowest:.3f} m, "
              f"{float(data.xpos[self.pelvis][2]):.3f} m at the end, feet {clearance:.3f} m clear "
              f"over {STEPS} steps ({STEPS * model.opt.timestep:.1f} s)")
        self.assertTrue(self.band.enabled)
        self.assertGreater(lowest, start - MAX_PELVIS_DROP)
        self.assertGreater(clearance, MIN_HANG_CLEARANCE)


class H12BlankTests(_BlankChecks, unittest.TestCase):
    """Check the Blank task with the H1-2."""

    build = staticmethod(_h12)


class G1BlankTests(_BlankChecks, unittest.TestCase):
    """Check the Blank task with the G1."""

    build = staticmethod(_g1)


if __name__ == "__main__":
    unittest.main()
