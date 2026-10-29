"""Tests for the G1 sim's elastic band and its spawn, on the RoboCasa kitchen the
surface probe runs (tests/robocasa/surface/run.sh: layout 9, style 9, seed 42).

The env is built headless with g1_robosuite.make_kitchen_env, the robot spawned
and the band built by g1_mujoco's own spawn() and make_band(), and the physics
stepped as sim_loop steps it: the interface's motor torques (SimInterface, with
no rt/lowcmd: every motor off), the band's force (apply_band), mj_step. Run it in
the robocasa image with golem_robocasa on PYTHONPATH:
tests/robocasa/g1/run_in_container.sh runs it with --network none and
ROS_DOMAIN_ID=97 (SimInterface opens Unitree DDS channels).
"""
import functools
import math
import os
import subprocess
import sys
import textwrap

import mujoco
import numpy as np

import sim_common
from g1 import g1_mujoco, g1_robosuite
from g1.sim_names import NameResolver
from g1.unitree_interface import SimInterface

LAYOUT, STYLE, SEED = 9, 9, 42   # the surface probe's G1 kitchen
HOLD_S = 3.0          # sim s of hanging with nothing commanding the robot
MAX_TORSO_DZ = 0.05   # m: torso_link z from the band's anchor while it hangs
MAX_TILT_DEG = 15.0   # pelvis z-axis from world up while it hangs
DROP_S = 1.5          # sim s after the release
MIN_DROP = 0.15       # m the pelvis falls in DROP_S once the band is released


@functools.lru_cache(maxsize=None)
def _sim():
    """The kitchen env with the robot spawned, built once; returns
    (env, model, data, resolver, interface, the spawn state)"""
    task = sim_common.random_task(seed=SEED)
    env = g1_robosuite.make_kitchen_env(task, layout_ids=LAYOUT, style_ids=STYLE, seed=SEED)
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    resolver = NameResolver(model, robot_prefix=env.robots[0].robot_model.naming_prefix)
    g1_mujoco.spawn(env, resolver)
    iface = SimInterface(model, data, resolver)
    spawn = (data.qpos.copy(), data.qvel.copy(), float(data.time))
    return env, model, data, resolver, iface, spawn


def _from_spawn():
    """Back to the spawn state, with a new band anchored there"""
    env, model, data, resolver, iface, (qpos, qvel, t) = _sim()
    data.qpos[:], data.qvel[:], data.time = qpos, qvel, t
    data.ctrl[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    iface.sim_time = t
    band, body = g1_mujoco.make_band(data, resolver)
    return model, data, resolver, iface, band, body


def _run(model, data, iface, band, body, seconds, each_step=None):
    """Step the physics as sim_loop does, with no rt/lowcmd"""
    vel = np.zeros(6)
    for _ in range(round(seconds / model.opt.timestep)):
        iface.write_ctrl()
        g1_mujoco.apply_band(model, data, band, body, vel)
        mujoco.mj_step(model, data)
        iface.after_step()
        if each_step is not None:
            each_step()


def _tilt_deg(data, body):
    """Angle (deg) between a body's z-axis and world up"""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[body][8])))))


def test_spawn_clear_of_floor_and_kitchen():
    """The robot spawns in the stance without touching the floor or penetrating
    the kitchen, and the band's anchor is 0.01 m above the torso"""
    env, model, data, resolver, iface, spawn = _sim()
    model, data, resolver, iface, band, body = _from_spawn()
    assert g1_robosuite.lowest_robot_point_z(env) > 0.0
    contacts = g1_robosuite._robot_env_contacts(model, data)
    assert not contacts, f"{len(contacts)} robot<->kitchen contacts at spawn"
    assert np.allclose(band.point - data.xpos[body], [0.0, 0.0, 0.01])
    # twice the H1-2 band's stiffness: at its 1e4 N/m the limp G1 keeps rocking (g1_mujoco)
    assert (band.stiffness, band.damping, band.length) == (2e4, 2e3, 0)
    assert band.enabled


def test_band_holds_limp_robot():
    """With nothing commanding it the robot hangs from the band: motors off,
    torso near the anchor, pelvis near upright"""
    model, data, resolver, iface, band, body = _from_spawn()
    pelvis = resolver.base_body
    worst = {"dz": 0.0, "tilt": 0.0, "ctrl": 0.0}

    def check():
        worst["dz"] = max(worst["dz"], abs(float(data.xpos[body][2] - band.point[2])))
        worst["tilt"] = max(worst["tilt"], _tilt_deg(data, pelvis))
        worst["ctrl"] = max(worst["ctrl"], float(np.abs(data.ctrl).max()))

    _run(model, data, iface, band, body, HOLD_S, check)
    print(f"  held {HOLD_S:.1f} s: torso z within {worst['dz']:.3f} m of the anchor, pelvis tilt "
          f"<= {worst['tilt']:.1f} deg; torso at z {data.xpos[body][2]:.3f} (anchor {band.point[2]:.3f}), "
          f"pelvis at z {data.xpos[pelvis][2]:.3f}")
    assert not iface.commanded and worst["ctrl"] == 0.0, "a motor was on with no rt/lowcmd"
    assert worst["dz"] <= MAX_TORSO_DZ, worst
    assert worst["tilt"] <= MAX_TILT_DEG, worst


def test_release_drops_limp_robot():
    """Once released, the band pulls no more and the limp robot falls"""
    model, data, resolver, iface, band, body = _from_spawn()
    pelvis = resolver.base_body
    _run(model, data, iface, band, body, HOLD_S)
    before = float(data.xpos[pelvis][2])
    assert band.toggle() is False
    _run(model, data, iface, band, body, DROP_S)
    after = float(data.xpos[pelvis][2])
    print(f"  released: pelvis z {before:.3f} -> {after:.3f} m in {DROP_S:.1f} s")
    assert np.all(data.xfrc_applied[body, :3] == 0.0)
    assert before - after >= MIN_DROP, (before, after)


def test_sim_reads_no_policy():
    """Importing g1_robosuite, which builds the sim's robot, touches no file or
    directory of HOMIE's policy (g1_homie's policy/, mounted at
    /home/code/g1_homie in the ros container) and puts none on sys.path"""
    code = textwrap.dedent("""
        import os, sys
        touched = []

        def hook(event, args):
            if event in ("open", "os.listdir", "os.scandir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
                touched.append(os.fsdecode(args[0]))

        sys.addaudithook(hook)
        _stat = os.stat

        def stat(path, *args, **kwargs):   # os.path.isdir / exists probe through it
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


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    if _sim.cache_info().currsize:
        _sim()[4].shutdown()
    sys.exit(1 if failed else 0)
