"""Check the MID-360's GPU caster (golem_robocasa/mid360.py) against the exact CPU caster.

The scenes are the surface probe's RoboCasa kitchen (layout 9, style 9, seed 42) with each
robot spawned as its runner spawns it, built once per process (about 20 s for both); the
casters see them as the sims do. The GPU tests need MuJoCo on EGL with a GPU renderer and are
skipped, naming why, without one; the backend tests need neither. Run them in the robocasa
image: ROS_DOMAIN_ID=231 tests/robocasa/run_in_container.sh -p test_mid360_gpu.py
"""
import functools
import os
import sys
import time
import types
import unittest
from typing import NamedTuple
from unittest import mock

import mujoco
import numpy as np

import mid360
import sim_common

LAYOUT, STYLE, SEED = 9, 9, 42        # the surface probe's kitchen
FRAME_POINTS = 20_000                 # one 10 Hz frame of the pattern
NEAR = 0.15                           # m: rays shorter than this on either caster are not compared
MAX_MEDIAN_ERROR = 0.002              # m
MIN_AGREEMENT = 0.99                  # fraction of rays on which both casters return or both do not
# Fraction of the rays that hit the robot first on the CPU that give no return on the GPU too:
# lower than MIN_AGREEMENT, since a thin limb's silhouette pixels are a larger share of its rays.
MIN_ROBOT_AGREEMENT = 0.95
MAX_CAST_S = 0.025                    # s per 20,000-ray cast; catches a broken path only
H12_GROUPS = (1, 1, 1, 1, 1, 1)       # the H1-2's rays hit every geom group (h12_sensors)
# Two walls ahead of the origin, their faces at x = 1 m (a plain geom) and 3 m (a geom with a
# material).
WALLS = """
<mujoco>
  <asset><material name="paint" rgba="0.8 0.8 0.8 1"/></asset>
  <worldbody>
    <geom name="near" type="box" group="1" pos="1.05 0 0" size="0.05 5 5"/>
    <geom name="far" type="box" group="1" material="paint" pos="3.05 0 0" size="0.05 5 5"/>
  </worldbody>
</mujoco>
"""
TINY = ('<mujoco><worldbody><geom type="box" group="1" pos="2 0 0" size="0.1 1 1"/>'
        '</worldbody></mujoco>')               # one wall in front of the origin


class Kitchen(NamedTuple):
    """Hold a spawned robot's kitchen and how its lidar casts in it."""
    model: mujoco.MjModel
    data: mujoco.MjData
    exclude_body: int          # the body the sensor sits in
    geomgroup: tuple
    origin: np.ndarray         # (3,) m, the sensor in the world
    rotation: np.ndarray       # (3, 3), sensor -> world


def _rotation(axis, degrees):
    """Return the rotation matrix of degrees about the world axis 'x', 'y' or 'z'."""
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, np.eye(3)["xyz".index(axis)], np.radians(degrees))
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


@functools.lru_cache(maxsize=None)
def _g1():
    """Return the G1's Kitchen, spawned by g1_mujoco.spawn() at RoboCasa's spawn."""
    from g1 import g1_lidar, g1_mujoco, g1_robosuite
    from g1.sim_names import NameResolver
    env = g1_robosuite.make_kitchen_env(sim_common.random_task(seed=SEED), layout_ids=LAYOUT,
                                        style_ids=STYLE, seed=SEED)
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    prefix = env.robots[0].robot_model.naming_prefix
    g1_mujoco.spawn(env, NameResolver(model, robot_prefix=prefix))
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + g1_lidar.SITE)
    return Kitchen(model, data, int(model.site_bodyid[site]), g1_lidar.GEOM_GROUPS,
                   data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy())


@functools.lru_cache(maxsize=None)
def _h12():
    """Return the H1-2's Kitchen, spawned with h12_mujoco's pose and placement call."""
    if "magpie_hand_bridge" not in sys.modules:   # needs magpie_msgs, which the shell lacks
        stub = types.ModuleType("magpie_hand_bridge")
        stub.MagpieHandBridge = None
        sys.modules["magpie_hand_bridge"] = stub
    import h12_mujoco
    import h1_2_robosuite
    from sim_names import NameResolver
    env = h1_2_robosuite.make_kitchen_env(sim_common.random_task(seed=SEED), layout_ids=LAYOUT,
                                          style_ids=STYLE, seed=SEED)
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
    mujoco.mj_forward(model, data)
    lidar = resolver.body_id("livox_link")
    return Kitchen(model, data, resolver.body_id("torso_link"), H12_GROUPS,
                   data.xpos[lidar].copy(), data.xmat[lidar].reshape(3, 3).copy())


@functools.lru_cache(maxsize=None)
def _no_gpu_reason():
    """Return why no GPU caster can be built here, or None when one can."""
    model = mujoco.MjModel.from_xml_string(TINY)
    try:
        caster = mid360.GpuCaster(model, -1, (0, 1, 0, 0, 0, 0))
    except mid360.GpuUnavailable as exc:   # any other failure is a fault the tests must show
        return f"no GPU caster: {exc}"
    caster.close()
    return None


def _frame():
    """Return one frame's directions, (20,000, 3) in the sensor's frame."""
    return mid360.directions(mid360.load_pattern(), 0, FRAME_POINTS)


class GpuCasterTests(unittest.TestCase):
    """Compare GpuCaster with CpuCaster in both robots' kitchens."""

    @classmethod
    def setUpClass(cls):
        """Skip without a GPU renderer; build both kitchens and a GPU caster in each."""
        reason = _no_gpu_reason()
        if reason:
            raise unittest.SkipTest(reason)
        cls.kitchens = {"g1": _g1(), "h12": _h12()}
        cls.gpu = {}
        for name, kitchen in cls.kitchens.items():
            cls.gpu[name] = mid360.GpuCaster(kitchen.model, kitchen.exclude_body,
                                             kitchen.geomgroup)

    @classmethod
    def tearDownClass(cls):
        """Free the GPU casters' GL contexts."""
        for caster in cls.gpu.values():
            caster.close()

    def _compare(self, name, rotation):
        """Cast a frame from the sensor turned to rotation on both casters; return (cpu, gpu).

        Asserts the median range error and the agreement on return or no return, rays within
        NEAR of the sensor on either caster left out.
        """
        kitchen = self.kitchens[name]
        local = _frame()
        cpu = mid360.CpuCaster(kitchen.model, kitchen.exclude_body, kitchen.geomgroup)
        try:
            expected = cpu.cast(kitchen.origin, rotation, local, kitchen.data)
        finally:
            cpu.close()
        got = self.gpu[name].cast(kitchen.origin, rotation, local, kitchen.data)
        with np.errstate(invalid="ignore"):   # NaN, no return, compares False
            compared = ~(expected < NEAR) & ~(got < NEAR)
        both = compared & np.isfinite(expected) & np.isfinite(got)
        agreement = (np.isnan(expected) == np.isnan(got))[compared].mean()
        error = np.median(np.abs(expected[both] - got[both]))
        print(f"\n{name}: {compared.sum()} rays compared, {both.sum()} returns on both, "
              f"agreement {agreement:.4f}, median error {error * 1e3:.3f} mm")
        self.assertGreater(both.sum(), FRAME_POINTS // 2)
        self.assertGreaterEqual(agreement, MIN_AGREEMENT)
        self.assertLessEqual(error, MAX_MEDIAN_ERROR)
        return expected, got

    def test_gpu_matches_cpu(self):
        """20,000 pattern rays: median error within 2 mm, return or none agreed on 99% of rays.

        Both robots, sensor as mounted. Rays whose first hit is the robot give no return on
        the GPU too.
        """
        for name, kitchen in self.kitchens.items():
            with self.subTest(robot=name):
                expected, got = self._compare(name, kitchen.rotation)
                cpu = mid360.CpuCaster(kitchen.model, kitchen.exclude_body, kitchen.geomgroup,
                                       robot_geoms=np.zeros(kitchen.model.ngeom, dtype=bool))
                try:
                    robot_too = cpu.cast(kitchen.origin, kitchen.rotation, _frame(), kitchen.data)
                finally:
                    cpu.close()
                on_robot = np.isfinite(robot_too) & np.isnan(expected) & ~(robot_too < NEAR)
                self.assertGreater(on_robot.sum(), 0, "no ray hits the robot")
                print(f"{name}: {on_robot.sum()} rays hit the robot first, "
                      f"{np.isnan(got[on_robot]).mean():.4f} of them without a return on the GPU")
                self.assertGreaterEqual(np.isnan(got[on_robot]).mean(), MIN_ROBOT_AGREEMENT)

    def test_gpu_matches_cpu_at_any_orientation(self):
        """The same holds with the sensor turned 90 deg about x or 180 deg about y (a fall)."""
        for name, kitchen in self.kitchens.items():
            for axis, degrees in (("x", 90.0), ("y", 180.0)):
                with self.subTest(robot=name, axis=axis):
                    self._compare(name, _rotation(axis, degrees) @ kitchen.rotation)

    def test_live_alpha_hides_and_shows_geoms(self):
        """A geom made transparent between casts stops returning on both casters, which agree.

        Hiding the near wall's geom alpha shows the far wall; hiding the far wall's material
        alpha leaves no return; alpha None keeps the last cast's; the model's alpha restores
        both walls.
        """
        model = mujoco.MjModel.from_xml_string(WALLS)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        near = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "near")
        rng = np.random.default_rng(3)
        local = np.column_stack([np.ones(300), rng.uniform(-0.5, 0.5, (300, 2))])
        local /= np.linalg.norm(local, axis=1, keepdims=True)
        visible_geoms, visible_mats = model.geom_rgba[:, 3].copy(), model.mat_rgba[:, 3].copy()
        hidden_near = visible_geoms.copy()
        hidden_near[near] = 0.0
        steps = [("as built", None, None, 1.0), ("near hidden", hidden_near, None, 3.0),
                 ("paint hidden", None, np.zeros_like(visible_mats), np.nan),
                 ("alpha kept", None, None, np.nan),
                 ("restored", visible_geoms, visible_mats, 1.0)]
        cpu = mid360.CpuCaster(model, -1, (0, 1, 0, 0, 0, 0), threads=1)
        gpu = mid360.GpuCaster(model, -1, (0, 1, 0, 0, 0, 0))
        try:
            for label, geom_alpha, mat_alpha, wall_x in steps:
                with self.subTest(step=label):
                    ranges = [caster.cast(np.zeros(3), np.eye(3), local, data,
                                          geom_alpha=geom_alpha, mat_alpha=mat_alpha)
                              for caster in (cpu, gpu)]
                    expected = wall_x / local[:, 0]   # the wall's face at x = wall_x
                    for got in ranges:
                        np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.01)
        finally:
            cpu.close()
            gpu.close()

    def test_gpu_cast_time(self):
        """A 20,000-ray cast takes under 25 ms on average over 20 calls; the mean is printed."""
        local = _frame()
        for name, kitchen in self.kitchens.items():
            caster = self.gpu[name]
            caster.cast(kitchen.origin, kitchen.rotation, local, kitchen.data)   # warm up
            start = time.perf_counter()
            for _ in range(20):
                caster.cast(kitchen.origin, kitchen.rotation, local, kitchen.data)
            mean = (time.perf_counter() - start) / 20
            print(f"\n{name}: GPU cast of {FRAME_POINTS} rays {mean * 1e3:.2f} ms (mean of 20)")
            self.assertLess(mean, MAX_CAST_S)


class _Refused:
    """Stand in for a GpuCaster whose construction fails, counting the attempts."""

    attempts = 0

    def __init__(self, *args, **kwargs):
        """Raise as a GpuCaster without a GPU renderer does."""
        type(self).attempts += 1
        raise RuntimeError("the GL renderer is llvmpipe, a software renderer")


class BackendTests(unittest.TestCase):
    """Check make_caster's choice of caster for each backend; needs no GPU."""

    def setUp(self):
        """Compile a one-box scene and count GPU caster attempts from zero."""
        self.model = mujoco.MjModel.from_xml_string(TINY)
        _Refused.attempts = 0

    def test_auto_backend_falls_back_to_cpu(self):
        """auto falls back to a CpuCaster, logging why once, when the GPU's fails; gpu raises."""
        logged = []
        with mock.patch.object(mid360, "GpuCaster", _Refused):
            caster = mid360.make_caster(self.model, -1, (0, 1, 0, 0, 0, 0), "auto",
                                        log=logged.append)
            try:
                self.assertIsInstance(caster, mid360.CpuCaster)
            finally:
                caster.close()
            self.assertEqual(len(logged), 1)
            self.assertIn("llvmpipe", logged[0])
            with self.assertRaisesRegex(RuntimeError, "llvmpipe"):
                mid360.make_caster(self.model, -1, (0, 1, 0, 0, 0, 0), "gpu", log=logged.append)
        self.assertEqual(len(logged), 1)

    def test_cpu_backend_never_tries_the_gpu(self):
        """cpu returns a CpuCaster on its own copy of the model, without trying the GPU; an unknown backend raises."""
        with mock.patch.object(mid360, "GpuCaster", _Refused):
            caster = mid360.make_caster(self.model, -1, (0, 1, 0, 0, 0, 0), "cpu", log=self.fail)
            caster.close()
            # The cast clears body_bvhadr, which mj_step on the caller's model reads.
            for field in ("body_bvhadr", "geom_rgba", "mat_rgba"):
                self.assertFalse(np.shares_memory(getattr(caster.model, field),
                                                  getattr(self.model, field)), field)
            with self.assertRaises(ValueError):
                mid360.make_caster(self.model, -1, (0, 1, 0, 0, 0, 0), "cuda", log=self.fail)
        self.assertIsInstance(caster, mid360.CpuCaster)
        self.assertEqual(_Refused.attempts, 0)

    def test_gpu_caster_needs_egl(self):
        """Without MUJOCO_GL=egl (the Mac's osmesa or glfw) the GPU caster refuses to build."""
        for value in ("osmesa", "glfw", ""):
            with self.subTest(MUJOCO_GL=value), mock.patch.dict(os.environ, {"MUJOCO_GL": value}):
                with self.assertRaisesRegex(mid360.GpuUnavailable, "MUJOCO_GL"):
                    mid360.GpuCaster(self.model, -1, (0, 1, 0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
