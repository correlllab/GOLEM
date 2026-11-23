"""Check the G1 sim's elastic band and spawn, in the kitchen the surface probe uses.

The kitchen is tests/robocasa/surface/run.sh's: layout 9, style 9, seed 42. The env is built
headless with g1_robosuite.make_kitchen_env, the robot spawned and the band built by
g1_robot's own spawn(), make_band() and band_lowering(), and the physics stepped as sim_loop
steps it: the anchor's move (mujoco_env.BandLowering, which never lowers here: no leg is
held), the interface's motor torques (SimInterface with no rt/lowcmd, so every motor off), the
band's force (mujoco_env.apply_band), mj_step. Run it in the robocasa image with golem_robocasa on PYTHONPATH;
tests/robocasa/g1/run_in_container.sh runs it with no network on ROS_DOMAIN_ID 97, because
SimInterface opens Unitree DDS channels.
"""
import functools
import math
import os
import subprocess
import sys
import textwrap
import unittest

import mujoco
import numpy as np

import function_tests
import mujoco_env
import robosuite_common
import sim_common
from g1 import g1_robot, g1_robosuite
from g1.sim_names import NameResolver
from g1.unitree_interface import SimInterface

LAYOUT, STYLE, SEED = 9, 9, 42  # the surface probe's G1 kitchen
HOLD_S = 3.0  # s of sim time hanging with nothing commanding the robot
LIFTED_S = 1.5  # s of sim time after which the anchor is up and the robot hangs
MAX_HOOK_DZ = 0.05  # m: the band's hook z from its anchor while it hangs
MAX_TILT_DEG = 15.0  # deg: the pelvis z axis from world up while it hangs
MIN_CLEARANCE = 0.04  # m: the lowest robot geom above the floor while it hangs (6-9 cm measured)
DROP_S = 1.5  # s of sim time after the release
MIN_DROP = 0.15  # m the pelvis falls in DROP_S once the band is released


@functools.lru_cache(maxsize=None)
def _sim():
    """Build the kitchen with the robot spawned, once per process.

    Returns (env, model, data, resolver, interface, the spawn state (qpos, qvel, time)).
    """
    task = sim_common.random_task(seed=SEED)
    env = g1_robosuite.make_kitchen_env(task, layout_ids=LAYOUT, style_ids=STYLE, seed=SEED)
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    resolver = NameResolver(model, robot_prefix=env.robots[0].robot_model.naming_prefix)
    g1_robot.spawn(env, resolver)
    iface = SimInterface(model, data, resolver)
    spawn = (data.qpos.copy(), data.qvel.copy(), float(data.time))
    return env, model, data, resolver, iface, spawn


def _from_spawn(lowering=True):
    """Restore the spawn state and make a new band there, as golem_sim does; return the sim, the band and its lowering."""
    env, model, data, resolver, iface, (qpos, qvel, t) = _sim()
    data.qpos[:], data.qvel[:], data.time = qpos, qvel, t
    data.ctrl[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    iface.sim_time = t
    band, body = g1_robot.make_band(data, resolver, lowering)
    return model, data, resolver, iface, band, body, (g1_robot.band_lowering(band) if lowering else None)


def _run(model, data, iface, band, body, lowering, seconds, each_step=None):
    """Step the physics for `seconds` of sim time as sim_loop does, with no rt/lowcmd; call `each_step` after each."""
    vel = np.zeros(6)
    for _ in range(round(seconds / model.opt.timestep)):
        if lowering is not None:
            lowering.step(float(data.time), iface)
        iface.write_ctrl()
        mujoco_env.apply_band(model, data, band, body, vel)
        mujoco.mj_step(model, data)
        iface.after_step()
        if each_step is not None:
            each_step()


def _kitchen_contacts(model, data):
    """Return the robot's penetrating contacts with anything but the floor (robosuite_common.robot_env_contacts)."""
    def floor(geom):
        """Return whether geom is the floor."""
        return (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith("floor")
    return [c for c in robosuite_common.robot_env_contacts(model, data) if not (floor(c.geom1) or floor(c.geom2))]


def _tilt_deg(data, body):
    """Return the angle between a body's z axis and world up, degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[body][8])))))


def test_spawn_clear_of_floor_and_kitchen():
    """The robot spawns clear of the floor and the kitchen, on a rope hooked above its shoulders and anchored at the hook."""
    env, model, data, resolver, iface, spawn = _sim()
    model, data, resolver, iface, band, body, lowering = _from_spawn()
    assert g1_robosuite.lowest_robot_point_z(env) > 0.0
    contacts = robosuite_common.robot_env_contacts(model, data)
    assert not contacts, f"{len(contacts)} robot<->kitchen contacts at spawn"
    np.testing.assert_allclose(band.hook, g1_robot.BAND_HOOK)
    np.testing.assert_allclose(band.point, band.hook_point(data, body))
    # Twice the H1-2 band's stiffness: at its 1e4 N/m the limp G1 keeps rocking (g1_robot.py).
    assert (band.stiffness, band.damping, band.length, band.rope) == (2e4, 2e3, 0, True)
    assert band.enabled
    assert lowering.lift == g1_robot.BAND_LIFT
    # Legs first, then legs and waist (g1_robot.py: a limp waist folds under a loaded lower body).
    assert [list(stage[0]) for stage in lowering.stages] == [list(range(12)), list(range(15))]


def test_no_band_lowering_keeps_the_fixed_anchor():
    """golem_sim.py --no_band_lowering's band is the fixed one: 0.01 m above the torso's origin, no rope, no hook."""
    model, data, resolver, iface, band, body, lowering = _from_spawn(lowering=False)
    np.testing.assert_allclose(band.point - data.xpos[body], [0.0, 0.0, 0.01])
    assert (band.stiffness, band.damping, band.length, band.rope, band.hook) == (2e4, 2e3, 0, False, None)


def test_band_holds_limp_robot():
    """With nothing commanding it the robot hangs from the band: motors off, feet clear of the floor, pelvis upright, kitchen untouched."""
    env, *_ = _sim()
    model, data, resolver, iface, band, body, lowering = _from_spawn()
    pelvis = resolver.base_body
    t0 = float(data.time)
    worst = {"dz": 0.0, "tilt": 0.0, "ctrl": 0.0, "low": float("inf"), "contacts": 0}
    steps = [0]

    def check():
        """Track the worst hook offset, pelvis tilt, clearance, kitchen contact and motor torque once the robot hangs."""
        worst["ctrl"] = max(worst["ctrl"], float(np.abs(data.ctrl).max()))
        worst["contacts"] = max(worst["contacts"], len(_kitchen_contacts(model, data)))
        if data.time - t0 < LIFTED_S:
            return
        worst["dz"] = max(worst["dz"], abs(float(band.hook_point(data, body)[2] - band.point[2])))
        worst["tilt"] = max(worst["tilt"], _tilt_deg(data, pelvis))
        steps[0] += 1
        if steps[0] % 10 == 0:   # every 20 ms: the AABB scan is slow
            worst["low"] = min(worst["low"], g1_robosuite.lowest_robot_point_z(env))

    _run(model, data, iface, band, body, lowering, HOLD_S, check)
    print(f"  held {HOLD_S:.1f} s: hook z within {worst['dz']:.3f} m of the anchor, pelvis tilt "
          f"<= {worst['tilt']:.1f} deg, lowest geom >= {worst['low']:.3f} m; anchor at z {band.point[2]:.3f}, "
          f"pelvis at z {data.xpos[pelvis][2]:.3f}")
    assert not iface.commanded and worst["ctrl"] == 0.0, "a motor was on with no rt/lowcmd"
    assert not lowering.starts, "the band lowered with no leg held"
    assert worst["contacts"] == 0, worst
    assert worst["dz"] <= MAX_HOOK_DZ, worst
    assert worst["tilt"] <= MAX_TILT_DEG, worst
    assert worst["low"] >= MIN_CLEARANCE, worst


def test_release_drops_limp_robot():
    """Once released, the band pulls no more and the limp robot falls."""
    model, data, resolver, iface, band, body, lowering = _from_spawn()
    pelvis = resolver.base_body
    _run(model, data, iface, band, body, lowering, HOLD_S)
    before = float(data.xpos[pelvis][2])
    assert band.toggle() is False
    _run(model, data, iface, band, body, lowering, DROP_S)
    after = float(data.xpos[pelvis][2])
    print(f"  released: pelvis z {before:.3f} -> {after:.3f} m in {DROP_S:.1f} s")
    assert np.all(data.xfrc_applied[body] == 0.0)
    assert before - after >= MIN_DROP, (before, after)


def test_sim_reads_no_policy():
    """Importing g1_robosuite, which builds the sim's robot, opens nothing of HOMIE's policy.

    It touches no file or directory with "homie" in its path (golem_lowerbody_rl's
    policies/homie/), puts none on sys.path and imports no such module.
    """
    code = textwrap.dedent("""
        import os, sys
        touched = []

        def hook(event, args):
            if event in ("open", "os.listdir", "os.scandir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
                touched.append(os.fsdecode(args[0]))

        sys.addaudithook(hook)
        _stat = os.stat

        def stat(path, *args, **kwargs):  # os.path.isdir and exists probe through it
            if isinstance(path, (str, bytes, os.PathLike)):
                touched.append(os.fsdecode(path))
            return _stat(path, *args, **kwargs)

        os.stat = stat
        from g1 import g1_robosuite
        assert g1_robosuite.STANCE, "no stance"
        policy = sorted({p for p in touched + sys.path if "homie" in p.lower()})
        modules = sorted(m for m in sys.modules if "homie" in m.lower())
        if policy or modules:
            print("policy paths:", policy, "modules:", modules)
            sys.exit(1)
        print(f"{len(touched)} paths touched, none of the policy's")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ), timeout=300)
    print(textwrap.indent((r.stdout + r.stderr).strip()[-2000:], "  "))
    assert r.returncode == 0, r.returncode


def tearDownModule():
    """Close the shared interface's DDS channels, if a test built it."""
    if _sim.cache_info().currsize:
        _sim()[4].shutdown()


G1BandTests = function_tests.test_case(globals(), "G1BandTests")

if __name__ == "__main__":
    unittest.main()
