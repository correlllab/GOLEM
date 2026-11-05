"""Check g1_robosuite's collision-free spawn, and the joint dynamics and stance it gives the G1 MJCF.

Runs on the host as well as in the container: robosuite is stubbed, and a small MjModel with
a robot0_-prefixed free body and a fixed counter exercises the real geometry, contact
filtering and back-off loop. The spawn in a real kitchen, hanging from the band, is
test_g1_band.py's.

Run from the repo root: PYTHONPATH=golem_robocasa python3 tests/robocasa/g1/test_g1_spawn_collision.py
(g1_robosuite reads CL_Assets/g1's MJCF and URDF at import).
"""
import math
import sys
import types

import mujoco
import numpy as np


def _install_robosuite_stubs():
    """Register stub robosuite modules so that g1_robosuite imports without robosuite.

    g1_robosuite imports robosuite at module level, for class decorators and base classes; the
    functions under test use only NumPy and MuJoCo, so only that unused plumbing is stubbed.
    """
    def mod(name):
        """Create, register and return an empty module."""
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    mod("robosuite")
    mod("robosuite.models")
    grippers = mod("robosuite.models.grippers")
    grippers.register_gripper = lambda cls: cls
    ng = mod("robosuite.models.grippers.null_gripper")
    ng.NullGripper = type("NullGripper", (), {"__init__": lambda self, *a, **k: None})
    mod("robosuite.models.robots")
    mod("robosuite.models.robots.manipulators")
    lm = mod("robosuite.models.robots.manipulators.legged_manipulator_model")
    lm.LeggedManipulatorModel = type("LeggedManipulatorModel", (), {"__init__": lambda self, *a, **k: None})
    robots = mod("robosuite.robots")
    robots.register_robot_class = lambda *a, **k: (lambda cls: cls)


_install_robosuite_stubs()
from g1 import g1_robosuite as gr  # noqa: E402  (must follow stub install)

# A robot0_-prefixed free body whose "hand" box overlaps a fixed counter when the free joint
# sits at x=0, and clears it once the base is pushed back along -x.
SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="counter_body" pos="0.3 0 0.5">
      <geom name="counter" type="box" size="0.2 0.2 0.5"/>
    </body>
    <body name="robot0_base" pos="0 0 1">
      <freejoint name="robot0_floating_base_joint"/>
      <geom name="robot0_hand" type="box" pos="0.15 0 0" size="0.15 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

_FREE_JOINT = "robot0_floating_base_joint"


def _load():
    """Return SCENE's model and data, forwarded."""
    m = mujoco.MjModel.from_xml_string(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _free_qadr(m):
    """Return the free joint's qpos address."""
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, _FREE_JOINT)
    return int(m.jnt_qposadr[jid])


def _total_penetration(m, d):
    """Return the summed penetration depth of the robot's contacts with the scene, m."""
    return sum(-c.dist for c in gr._robot_env_contacts(m, d))


class _FakeSimModel:
    """Stand in for robosuite's sim.model."""

    def __init__(self, m):
        """Wrap a MuJoCo model."""
        self._model = m

    def get_joint_qpos_addr(self, name):
        """Return a joint's qpos address."""
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self._model.jnt_qposadr[jid])  # place_robot_clear handles int or tuple


class _FakeSimData:
    """Stand in for robosuite's sim.data."""

    def __init__(self, d):
        """Wrap MuJoCo data."""
        self._data = d

    @property
    def qpos(self):
        """Return the wrapped qpos."""
        return self._data.qpos


class _FakeSim:
    """Stand in for robosuite's sim."""

    def __init__(self, m, d):
        """Wrap a MuJoCo model and data."""
        self._m, self._d = m, d
        self.model = _FakeSimModel(m)
        self.data = _FakeSimData(d)

    def forward(self):
        """Run mj_forward."""
        mujoco.mj_forward(self._m, self._d)


class _FakeRobotModel:
    """Stand in for the robot model, with robosuite's robot0_ prefix."""

    naming_prefix = "robot0_"


class _FakeRobot:
    """Stand in for robosuite's robot."""

    robot_model = _FakeRobotModel()


class _FakeEnv:
    """Stand in for the robosuite env that place_robot_clear expects."""

    def __init__(self, m, d):
        """Wrap SCENE with the robot's initial base position."""
        self.sim = _FakeSim(m, d)
        self.robots = [_FakeRobot()]
        self.init_robot_base_pos = np.array([0.0, 0.0, 1.0])


def test_backward_dir_identity():
    """At yaw 0 the robot faces +x, so backward is -x."""
    b = gr._backward_dir(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(b, [-1.0, 0.0, 0.0], atol=1e-6), b


def test_backward_dir_yaw_90():
    """At yaw +90 degrees the robot faces +y, so backward is -y, a horizontal unit vector."""
    q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])  # wxyz
    b = gr._backward_dir(q)
    assert np.allclose(b, [0.0, -1.0, 0.0], atol=1e-6), b
    assert abs(np.linalg.norm(b) - 1.0) < 1e-6 and abs(b[2]) < 1e-9


def test_robot_env_contacts_detected():
    """The robot's overlap with the counter is found."""
    m, d = _load()
    contacts = gr._robot_env_contacts(m, d)
    assert len(contacts) >= 1, f"expected robot<->counter contact, got {len(contacts)}"


def test_robot_env_contacts_clear_when_moved_back():
    """Backed off 0.5 m, the robot has no contact."""
    m, d = _load()
    d.qpos[_free_qadr(m)] = -0.5  # slide the free base back along -x
    mujoco.mj_forward(m, d)
    assert gr._robot_env_contacts(m, d) == [], "expected no contact after backing off"


def test_place_robot_collision_free_backs_off():
    """A spawn in collision is backed off along -x until it is clear."""
    m, d = _load()
    env = _FakeEnv(m, d)
    base_pos = np.array([0.0, 0.0, 1.0])
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity -> backward = -x
    gr.place_robot_collision_free(env, base_pos, base_quat,
                                  step=0.02, max_iters=50, clearance=0.02)
    mujoco.mj_forward(m, d)
    assert gr._robot_env_contacts(m, d) == [], "spawn should be collision-free after back-off"
    assert d.qpos[_free_qadr(m)] < 0.0, "base should have moved back along -x"


def test_no_move_when_already_clear():
    """A spawn already clear is not moved."""
    m, d = _load()
    env = _FakeEnv(m, d)
    base_pos = np.array([-1.0, 0.0, 1.0])  # already clear of the counter
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    gr.place_robot_collision_free(env, base_pos, base_quat,
                                  step=0.02, max_iters=50, clearance=0.02)
    assert abs(d.qpos[_free_qadr(m)] - (-1.0)) < 1e-9, "must not move a collision-free spawn"


def test_keeps_best_on_giveup():
    """With too few iterations to clear, the least-penetrating placement tried is kept."""
    m, d = _load()
    env = _FakeEnv(m, d)
    base_pos = np.array([0.0, 0.0, 1.0])
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    # The raw placement penetrates; clearing the counter takes about 11 steps.
    gr.place_robot_clear(env, base_pos, base_quat, 0.02)
    start_pen = _total_penetration(m, d)
    assert start_pen > 0, "scene must start in collision"
    # Too few iterations to clear it, so the give-up path runs.
    gr.place_robot_collision_free(env, base_pos, base_quat,
                                  step=0.02, max_iters=3, clearance=0.02)
    mujoco.mj_forward(m, d)
    final_pen = _total_penetration(m, d)
    assert final_pen <= start_pen + 1e-9, (start_pen, final_pen)  # best is never worse than start
    assert d.qpos[_free_qadr(m)] < 0.0, "give-up should keep the most-backed-off (least-penetrating) try"


# From Unitree's unitree_rl_gym: g1_12dof.xml's default joint and g1_config.py's
# default_joint_angles (rad).
UNITREE_JOINT = {"damping": 0.001, "armature": 0.01, "frictionloss": 0.1}
UNITREE_LEGS = {"hip_pitch": -0.1, "hip_roll": 0.0, "hip_yaw": 0.0,
                "knee": 0.3, "ankle_pitch": -0.2, "ankle_roll": 0.0}


def test_joint_dynamics_are_unitrees():
    """The 29 body joints have Unitree's damping, armature and friction; the fingers a tenth of the friction and no damping."""
    assert (gr.JOINT_DAMPING, gr.JOINT_ARMATURE, gr.JOINT_FRICTIONLOSS) == (0.001, 0.01, 0.1)
    joints = list(gr.build_robot_xml().iter("joint"))
    body = [j for j in joints if "_hand_" not in j.get("name")]
    assert len(body) == 29, len(body)
    for joint in body:
        assert {k: float(joint.get(k)) for k in UNITREE_JOINT} == UNITREE_JOINT, joint.attrib
    for joint in joints:  # the fingers: the armature, a tenth of the friction, no damping
        if "_hand_" in joint.get("name"):
            assert (float(joint.get("damping")), float(joint.get("armature")),
                    float(joint.get("frictionloss"))) == (0.0, 0.01, 0.1 * 0.1), joint.attrib


def test_stance_is_unitrees_default():
    """The spawn stance is Unitree's default leg pose."""
    expected = {f"{side}_{name}_joint": q for side in ("left", "right") for name, q in UNITREE_LEGS.items()}
    assert {n: q for n, q in gr.STANCE.items() if q} == {n: q for n, q in expected.items() if q}
    assert set(gr.STANCE) <= set(expected), set(gr.STANCE) - set(expected)


def main():
    """Run every test_* function here and exit 1 if any fails."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
