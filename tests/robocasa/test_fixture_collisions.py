"""Check that the sims stop only the collisions inside a RoboCasa fixture and between its moving parts and the static scene.

The pair check runs in the surface probe's kitchen (layout 9, style 9, seed 42) with each robot
spawned as its sim spawns it. The resting check runs in the same layout and style with the G1
on its band, in two of RoboCasa's tasks that put an object in a fixture, a drawer (a moving
part) and a cabinet shelf (a static part), and in one with a loose fixture on a free joint (a
blender lid). Run it with tests/robocasa/run_in_container.sh.
"""
import sys
import types
import unittest

import mujoco
import numpy as np

import sim_common

LAYOUT, STYLE, SEED = 9, 9, 42        # the surface probe's kitchen
PAIR_STEPS = 100                      # physics steps over which the broadphase pairs are recorded
REST_S = 1.0                          # s of sim time an object must keep resting in its fixture
MAX_OBJECT_DROP = 0.02                # m an object or loose fixture may sink in REST_S
# RoboCasa tasks that put an object in a fixture (the drawer's inner box is a moving part,
# the cabinet's shelf a static one) and one with a loose fixture (the blender's lid, on a free
# joint, sitting on the blender).
RESTING_TASKS = ("CloseDrawer", "PickPlaceCabinetToCounter", "OpenBlenderLid")


def _h12_kitchen(task):
    """Return the H1-2's env for task, spawned with h12_mujoco's pose and placement call."""
    if "magpie_hand_bridge" not in sys.modules:   # needs magpie_msgs, which the image lacks
        stub = types.ModuleType("magpie_hand_bridge")
        stub.MagpieHandBridge = None
        sys.modules["magpie_hand_bridge"] = stub
    import h12_mujoco
    import h1_2_robosuite
    from sim_names import NameResolver
    env = h1_2_robosuite.make_kitchen_env(task, layout_ids=LAYOUT, style_ids=STYLE, seed=SEED)
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
    return env


def _g1_kitchen(task):
    """Return the G1's env for task, spawned by g1_mujoco.spawn()."""
    from g1 import g1_mujoco, g1_robosuite
    from g1.sim_names import NameResolver
    env = g1_robosuite.make_kitchen_env(task, layout_ids=LAYOUT, style_ids=STYLE, seed=SEED)
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    g1_mujoco.spawn(env, NameResolver(model, robot_prefix=env.robots[0].robot_model.naming_prefix))
    mujoco.mj_forward(model, data)
    return env


KITCHENS = {"h12": _h12_kitchen, "g1": _g1_kitchen}


class _Geoms:
    """Classify an env's geoms as the exclusion rule reads them, from RoboCasa's own fixtures and objects."""

    def __init__(self, env):
        """Map every geom to its fixture or object (the nearest root body above it) and its robot."""
        model = env.sim.model._model
        body = mujoco.mjtObj.mjOBJ_BODY
        roots = {mujoco.mj_name2id(model, body, f.root_body): ("fixture", name)
                 for name, f in env.fixtures.items()}
        roots.update({mujoco.mj_name2id(model, body, o.root_body): ("object", name)
                      for name, o in env.objects.items()})
        self.owner = []          # per body: (kind, name) or None
        for b in range(model.nbody):
            up = b
            while up > 0 and up not in roots:
                up = int(model.body_parentid[up])
            self.owner.append(roots.get(up))
        robot_roots = {int(model.body_rootid[mujoco.mj_name2id(model, body, r.robot_model.root_body)])
                       for r in env.robots}
        free = set(model.jnt_bodyid[model.jnt_type == mujoco.mjtJoint.mjJNT_FREE].tolist())
        self.free_bodies = free
        bodies = model.geom_bodyid
        self.body = bodies
        self.weld = model.body_weldid[bodies]
        self.floating = np.array([int(model.body_rootid[b]) in free for b in bodies])
        self.robot = np.isin(model.body_rootid[bodies], list(robot_roots))
        self.object = np.array([(self.owner[b] or ("",))[0] == "object" for b in bodies])
        self.fixture = np.array([self.owner[b][1] if (self.owner[b] or ("",))[0] == "fixture" else ""
                                 for b in bodies])
        self.name = [mujoco.mj_id2name(model, body, b) or str(b) for b in bodies]

    def named(self, g1, g2):
        """Return whether the rule excludes geom pair (g1, g2).

        Two geoms of one fixture with at least one on a jointed body, or a fixture's
        jointed-body geom and a static (world-welded) geom that is neither the robot's nor an
        object's. A fixture on a free joint is loose, as an object is: none of its geoms is a
        jointed-body geom.
        """
        def moving_part(g):
            """Return whether geom g is on a jointed body of a fixture, not a loose one."""
            return bool(self.fixture[g]) and self.weld[g] != 0 and not self.floating[g]

        def static(g):
            """Return whether geom g is world-welded and neither the robot's nor an object's."""
            return self.weld[g] == 0 and not (self.robot[g] or self.object[g])

        if self.fixture[g1] and self.fixture[g1] == self.fixture[g2] and (moving_part(g1) or moving_part(g2)):
            return True
        return (moving_part(g1) and static(g2)) or (moving_part(g2) and static(g1))

    def label(self, pair):
        """Return a readable name for a geom pair: its bodies."""
        return f"{self.name[pair[0]]} / {self.name[pair[1]]}"


def _record_pairs(model, data, steps):
    """Return the geom pairs (lower id first) MuJoCo's collision passes to its geom filter over steps physics steps.

    A contact filter set with mujoco.set_mjcb_contactfilter replaces MuJoCo's own
    contype/conaffinity test (mj_collideGeoms in MuJoCo 3.17.0's engine_collision_driver.c)
    after the body-pair filters and the broadphase have run, so this filter applies that test
    itself: it records a pair and lets it through when the test passes, and drops it otherwise,
    so the physics stepped is the model's own. The filter is process-global; the previous one
    is restored.
    """
    pairs = set()
    contype, conaffinity = model.geom_contype, model.geom_conaffinity

    def record(m, d, g1, g2):
        """Record (g1, g2) and return 0 (collide) when the bitmask test passes, else 1 (drop)."""
        if (contype[g1] & conaffinity[g2]) or (contype[g2] & conaffinity[g1]):
            pairs.add((min(g1, g2), max(g1, g2)))
            return 0
        return 1

    previous = mujoco.get_mjcb_contactfilter()
    mujoco.set_mjcb_contactfilter(record)
    try:
        for _ in range(steps):
            mujoco.mj_step(model, data)
    finally:
        mujoco.set_mjcb_contactfilter(previous)
    return pairs


def _bitmask_pairs(model, weld):
    """Return the geom pairs (lower id first) on bodies not welded together that pass the contype/conaffinity test."""
    contype = model.geom_contype.astype(np.int64)
    conaffinity = model.geom_conaffinity.astype(np.int64)
    test = ((contype[:, None] & conaffinity[None, :]) | (contype[None, :] & conaffinity[:, None])) != 0
    test &= weld[:, None] != weld[None, :]
    g1, g2 = np.nonzero(np.triu(test, 1))
    return set(zip(g1.tolist(), g2.tolist()))


class FixtureCollisionTests(unittest.TestCase):
    """Check what exclude_fixture_collisions removes, and that objects still rest in fixtures."""

    def test_only_fixture_internal_pairs_are_excluded(self):
        """Over 100 steps from the spawn, the exclusion removes exactly the rule's pairs, none with a robot or object geom.

        The model's contype/conaffinity table loses exactly those pairs too, and a second call
        changes nothing. Each robot in the surface probe's kitchen; then the G1 in the resting
        check's kitchens, which hold objects and a loose fixture whose pairs must all be kept.
        """
        cases = [(robot, sim_common.random_task(seed=SEED)) for robot in KITCHENS]
        cases += [("g1", task) for task in RESTING_TASKS]
        for robot, task in cases:
            with self.subTest(robot=robot, task=task):
                env = KITCHENS[robot](task)
                model, data = env.sim.model._model, env.sim.data._data
                geoms = _Geoms(env)
                spec = mujoco.mjtState.mjSTATE_INTEGRATION
                spawn = np.empty(mujoco.mj_stateSize(model, spec))
                mujoco.mj_getState(model, data, spawn, spec)
                table_before = _bitmask_pairs(model, geoms.weld)
                expected = {p for p in table_before if geoms.named(*p)}

                before = _record_pairs(model, data, PAIR_STEPS)
                mujoco.mj_setState(model, data, spawn, spec)
                mujoco.mj_forward(model, data)
                count = sim_common.exclude_fixture_collisions(env)
                table_after = _bitmask_pairs(model, geoms.weld)
                after = _record_pairs(model, data, PAIR_STEPS)

                named = {p for p in before if geoms.named(*p)}
                self.assertTrue(named, "no pair the rule names was tested: the check proves nothing")
                kept = sorted(geoms.label(p) for p in named & after)
                lost = sorted(geoms.label(p) for p in before - named - after)
                new = sorted(geoms.label(p) for p in after - before)
                self.assertEqual(kept, [], "pairs the rule names are still tested")
                self.assertEqual(lost, [], "pairs the rule does not name were dropped")
                self.assertEqual(new, [], "pairs appeared that were not tested before")
                removed = before - after
                self.assertFalse(any(geoms.robot[g] or geoms.object[g] for p in removed for g in p))
                # The static contype/conaffinity table loses exactly the rule's pairs and gains none.
                self.assertEqual(sorted(geoms.label(p) for p in table_before - table_after),
                                 sorted(geoms.label(p) for p in expected))
                self.assertEqual(sorted(geoms.label(p) for p in table_after - table_before), [])
                self.assertEqual(count, len(expected))
                # A second call on the same model changes nothing.
                bits = (model.geom_contype.copy(), model.geom_conaffinity.copy())
                self.assertEqual(sim_common.exclude_fixture_collisions(env), 0)
                np.testing.assert_array_equal(model.geom_contype, bits[0])
                np.testing.assert_array_equal(model.geom_conaffinity, bits[1])
                print(f"\n[{robot} {task}] {count} geom pairs excluded; broadphase pairs over "
                      f"{PAIR_STEPS} steps {len(before)} -> {len(after)}", flush=True)
                del env

    def test_objects_still_rest_in_fixtures(self):
        """An object in a drawer or on a cabinet shelf, and a loose fixture, keep their contacts after 1 s.

        An object must still touch the fixture it was placed in; a loose fixture (on a free
        joint) must still touch a geom of another fixture. Neither may sink.
        """
        from g1 import g1_mujoco
        from g1.sim_names import NameResolver
        for task in RESTING_TASKS:
            with self.subTest(task=task):
                env = _g1_kitchen(task)
                model, data = env.sim.model._model, env.sim.data._data
                geoms = _Geoms(env)
                band, band_body = g1_mujoco.make_band(
                    data, NameResolver(model, robot_prefix=env.robots[0].robot_model.naming_prefix))
                vel = np.zeros(6)
                self.assertGreater(sim_common.exclude_fixture_collisions(env), 0)
                body = mujoco.mjtObj.mjOBJ_BODY
                # root body -> (the fixture it must touch, None for any other fixture; its
                # height at the start)
                resting = {}
                for cfg in env.object_cfgs:
                    fixture = cfg["placement"].get("fixture")
                    if getattr(fixture, "name", None) in env.fixtures:
                        root = mujoco.mj_name2id(model, body, env.objects[cfg["name"]].root_body)
                        resting[root] = (fixture.name, float(data.xpos[root][2]))
                for root in geoms.free_bodies:
                    if (geoms.owner[root] or ("",))[0] == "fixture":
                        resting[root] = (None, float(data.xpos[root][2]))
                self.assertTrue(resting, f"{task} put nothing in or on a fixture")
                for _ in range(round(REST_S / model.opt.timestep)):
                    g1_mujoco.apply_band(model, data, band, band_body, vel)
                    mujoco.mj_step(model, data)
                touching = set()
                for c in data.contact[:data.ncon]:
                    for mine, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                        root = int(model.body_rootid[model.geom_bodyid[mine]])
                        if root not in resting:
                            continue
                        want = resting[root][0]
                        if (geoms.fixture[other] == want if want is not None else
                                geoms.fixture[other] not in ("", geoms.owner[root][1])):
                            touching.add(root)
                for root, (fixture, z0) in resting.items():
                    name = mujoco.mj_id2name(model, body, root)
                    self.assertIn(root, touching,
                                  f"{name} lost its contact with {fixture or 'every other fixture'}")
                    self.assertLess(z0 - float(data.xpos[root][2]), MAX_OBJECT_DROP,
                                    f"{name} sank into {fixture or 'the scene'}")
                del env


if __name__ == "__main__":
    unittest.main()
