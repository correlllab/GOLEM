"""Host-runnable tests for the collision-free spawn logic in h1_2_robosuite.

mujoco imports on the host (3.x); robosuite/robocasa do NOT (container only), so
these tests build a tiny synthetic MjModel with a robot0_-prefixed free body and
a fixed fixture, and exercise the pure geometry + contact-filtering + back-off
loop without robosuite. Mirrors sim_names.self_test style.

Run: python3 test_spawn_collision.py
"""
import io
import math
import re
import sys
import types
from contextlib import redirect_stdout

import mujoco
import numpy as np


def _install_robosuite_stubs():
    """h1_2_robosuite imports robosuite at module top (container-only). The funcs
    under test (_backward_dir/_robot_env_contacts/place_robot_collision_free) use
    only numpy+mujoco, so register minimal stubs for the symbols the module
    references at import time (class decorators + base classes) to let it import
    on the host. The logic under test is real; only the unused robosuite plumbing
    is stubbed."""
    def mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    mod("robosuite")
    mod("robosuite.models")
    grippers = mod("robosuite.models.grippers")
    grippers.register_gripper = lambda cls: cls
    gm = mod("robosuite.models.grippers.gripper_model")
    gm.GripperModel = type("GripperModel", (), {"__init__": lambda self, *a, **k: None})
    mod("robosuite.models.robots")
    mod("robosuite.models.robots.manipulators")
    lm = mod("robosuite.models.robots.manipulators.legged_manipulator_model")
    lm.LeggedManipulatorModel = type("LeggedManipulatorModel", (), {"__init__": lambda self, *a, **k: None})
    robots = mod("robosuite.robots")
    robots.register_robot_class = lambda *a, **k: (lambda cls: cls)


_install_robosuite_stubs()
import h1_2_robosuite as hr  # noqa: E402  (must follow stub install)

# A robot0_-prefixed free body whose "hand" box overlaps a fixed counter when the
# free joint sits at x=0, and clears it once the base is pushed back along -x.
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
    m = mujoco.MjModel.from_xml_string(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _free_qadr(m):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, _FREE_JOINT)
    return int(m.jnt_qposadr[jid])


def _total_penetration(m, d):
    return sum(-c.dist for c in hr._robot_env_contacts(m, d))


# --- a minimal stand-in for the robosuite env that place_robot_clear expects --- #
class _FakeSimModel:
    def __init__(self, m):
        self._model = m

    def get_joint_qpos_addr(self, name):
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self._model.jnt_qposadr[jid])  # place_robot_clear handles int or tuple


class _FakeSimData:
    def __init__(self, d):
        self._data = d

    @property
    def qpos(self):
        return self._data.qpos


class _FakeSim:
    def __init__(self, m, d):
        self._m, self._d = m, d
        self.model = _FakeSimModel(m)
        self.data = _FakeSimData(d)

    def forward(self):
        mujoco.mj_forward(self._m, self._d)


class _FakeRobotModel:
    naming_prefix = "robot0_"


class _FakeRobot:
    robot_model = _FakeRobotModel()


class _FakeEnv:
    def __init__(self, m, d):
        self.sim = _FakeSim(m, d)
        self.robots = [_FakeRobot()]
        self.init_robot_base_pos = np.array([0.0, 0.0, 1.0])


# --------------------------------- tests ---------------------------------- #
def test_backward_dir_identity():
    # yaw=0: robot faces world +x, so "backward" (away from the counter) = -x.
    b = hr._backward_dir(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(b, [-1.0, 0.0, 0.0], atol=1e-6), b


def test_backward_dir_yaw_90():
    # yaw=+90deg about z: robot faces world +y, backward = -y. Result is unit + planar.
    q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])  # wxyz
    b = hr._backward_dir(q)
    assert np.allclose(b, [0.0, -1.0, 0.0], atol=1e-6), b
    assert abs(np.linalg.norm(b) - 1.0) < 1e-6 and abs(b[2]) < 1e-9


def test_robot_env_contacts_detected():
    m, d = _load()
    contacts = hr._robot_env_contacts(m, d)
    assert len(contacts) >= 1, f"expected robot<->counter contact, got {len(contacts)}"


def test_robot_env_contacts_clear_when_moved_back():
    m, d = _load()
    d.qpos[_free_qadr(m)] = -0.5  # slide the free base back along -x
    mujoco.mj_forward(m, d)
    assert hr._robot_env_contacts(m, d) == [], "expected no contact after backing off"


def test_place_robot_collision_free_backs_off():
    m, d = _load()
    env = _FakeEnv(m, d)
    base_pos = np.array([0.0, 0.0, 1.0])
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity -> backward = -x
    hr.place_robot_collision_free(env, base_pos, base_quat,
                                  step=0.02, max_iters=50, clearance=0.02)
    mujoco.mj_forward(m, d)
    assert hr._robot_env_contacts(m, d) == [], "spawn should be collision-free after back-off"
    assert d.qpos[_free_qadr(m)] < 0.0, "base should have moved back along -x"


def test_no_move_when_already_clear():
    m, d = _load()
    env = _FakeEnv(m, d)
    base_pos = np.array([-1.0, 0.0, 1.0])  # already clear of the counter
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    hr.place_robot_collision_free(env, base_pos, base_quat,
                                  step=0.02, max_iters=50, clearance=0.02)
    assert abs(d.qpos[_free_qadr(m)] - (-1.0)) < 1e-9, "must not move a collision-free spawn"


def test_keeps_best_on_giveup():
    m, d = _load()
    env = _FakeEnv(m, d)
    base_pos = np.array([0.0, 0.0, 1.0])
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    # penetration at the raw placement (clearing the counter needs ~11 steps)
    hr.place_robot_clear(env, base_pos, base_quat, 0.02)
    start_pen = _total_penetration(m, d)
    assert start_pen > 0, "scene must start in collision"
    # too few iterations to fully clear -> exercise the give-up path
    hr.place_robot_collision_free(env, base_pos, base_quat,
                                  step=0.02, max_iters=3, clearance=0.02)
    mujoco.mj_forward(m, d)
    final_pen = _total_penetration(m, d)
    assert final_pen <= start_pen + 1e-9, (start_pen, final_pen)  # best is never worse than start
    assert d.qpos[_free_qadr(m)] < 0.0, "give-up should keep the most-backed-off (least-penetrating) try"


def _gap(m, d, margin):
    mujoco.mj_forward(m, d)
    return hr._robot_env_gap(m, d, margin)


def test_gap_ignores_floor_and_caps_at_margin():
    m, d = _load()
    d.qpos[_free_qadr(m)] = -2.0          # far from the counter, hand near nothing
    d.qpos[_free_qadr(m) + 2] = 0.12      # hand box 2 cm above the floor plane
    assert abs(_gap(m, d, 0.2) - 0.2) < 1e-9, "floor must not count toward spawn clearance"


def test_gap_sees_near_pairs_inside_multi_geom_bodies():
    # MuJoCo's midphase would cull the hand<->counter pair once the robot body has
    # a second geom; the clearance check must still see it.
    two_geoms = SCENE.replace('<geom name="robot0_hand"',
                              '<geom name="robot0_back" type="box" pos="-0.1 0 0" size="0.1 0.1 0.1"/><geom name="robot0_hand"')
    m = mujoco.MjModel.from_xml_string(two_geoms)
    d = mujoco.MjData(m)
    d.qpos[_free_qadr(m)] = -0.22          # hand front 2 cm from the counter face
    assert abs(_gap(m, d, 0.2) - 0.02) < 1e-6
    assert not m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MIDPHASE, "model flags must be restored"

def test_gap_is_negative_when_overlapping():
    m, d = _load()
    assert _gap(m, d, 0.2) < 0.0


def test_margin_backs_off_until_requested_gap():
    m, d = _load()
    env = _FakeEnv(m, d)
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    hr.place_robot_collision_free(env, np.array([0.0, 0.0, 1.0]), base_quat, margin=0.0)
    no_overlap_x = d.qpos[_free_qadr(m)]
    hr.place_robot_collision_free(env, np.array([0.0, 0.0, 1.0]), base_quat, margin=0.2)
    assert _gap(m, d, 0.2) >= 0.2 - 1e-9, "margin spawn must keep 20 cm to the counter"
    # Hand front sits 0.3 m ahead of the base; the counter face is at x=0.1.
    assert d.qpos[_free_qadr(m)] <= -0.4 + 1e-9 < no_overlap_x
    assert d.qpos[_free_qadr(m)] > -0.4 - 0.02 - 1e-9, "must stop at the first spot that meets the margin"


def test_margin_zero_keeps_no_overlap_behavior():
    m, d = _load()
    env = _FakeEnv(m, d)
    hr.place_robot_collision_free(env, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]), margin=0.0)
    assert hr._robot_env_contacts(m, d) == []
    assert -0.22 - 1e-9 <= d.qpos[_free_qadr(m)] < -0.18, "margin 0 only clears the overlap"


def test_margin_blocked_behind_keeps_widest_gap():
    blocked = SCENE.replace(
        '<body name="robot0_base"',
        '<body name="wall" pos="-0.55 0 0.5"><geom name="wall" type="box" size="0.05 1 0.5"/></body>\n    <body name="robot0_base"'
    ).replace(
        '<geom name="robot0_hand"',
        '<geom name="robot0_back" type="box" pos="-0.1 0 0" size="0.1 0.1 0.1"/><geom name="robot0_hand"')
    m = mujoco.MjModel.from_xml_string(blocked)
    d = mujoco.MjData(m)
    env = _FakeEnv(m, d)
    out = io.StringIO()
    with redirect_stdout(out):
        hr.place_robot_collision_free(env, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]), margin=0.2)
    searched = float(re.search(r"searched ([0-9.]+) m back", out.getvalue()).group(1))
    assert searched < 0.5, f"search must stop at the wall behind, went {searched} m"
    gap = _gap(m, d, 0.2)
    assert 0.0 < gap < 0.2, gap   # the wall behind prevents 20 cm, but it never overlaps
    assert d.qpos[_free_qadr(m)] > -0.5, "must stay in front of the wall, not search through it"

ARM_SCENE = SCENE.replace(
    '<geom name="robot0_hand" type="box" pos="0.15 0 0" size="0.15 0.1 0.1"/>',
    '<geom name="robot0_hand" type="box" pos="0.15 0 0" size="0.15 0.1 0.1"/>'
    '<body name="robot0_arm" pos="0 0 0.3"><joint name="robot0_arm_joint" axis="0 1 0"/>'
    '<geom name="robot0_forearm" type="capsule" fromto="0 0 0 0 0 0.4" size="0.03"/></body>')


def test_margin_holds_for_extra_poses_and_keeps_spawn_joints():
    m = mujoco.MjModel.from_xml_string(ARM_SCENE)
    d = mujoco.MjData(m)
    env = _FakeEnv(m, d)
    arm = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot0_arm_joint")])])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    hr.place_robot_collision_free(env, np.array([0.0, 0.0, 1.0]), quat, margin=0.2)
    upright_x = d.qpos[_free_qadr(m)]
    # The forearm swung 90 deg forward reaches 0.4 m ahead of its shoulder.
    hr.place_robot_collision_free(env, np.array([0.0, 0.0, 1.0]), quat, margin=0.2,
                                  poses=[(arm, np.array([math.pi / 2]))])
    assert d.qpos[arm[0]] == 0.0, "the robot must be left in its spawn joint state"
    assert d.qpos[_free_qadr(m)] < upright_x - 0.05, "the swung pose must need more room"
    d.qpos[arm[0]] = math.pi / 2
    assert _gap(m, d, 0.2) >= 0.2 - 1e-9, "margin must hold in the extra pose"

def main():
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
