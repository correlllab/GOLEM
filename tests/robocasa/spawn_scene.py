"""Give the spawn-placement tests a small scene, a stand-in robosuite env around it and the robot modules.

SCENE is a robot0_-prefixed free body whose "hand" box overlaps a fixed counter when the free joint
sits at x=0, and clears it once the base is pushed back along -x. FakeEnv wraps a model and its
data the way the placement functions read a robosuite env. import_robot_module imports a robot's
robosuite module, with stand-ins for the robosuite modules it imports where robosuite is not
installed (the host), so the placement functions run on numpy and mujoco alone. Used by
test_robosuite_common.py, test_h12_spawn_collision.py and g1/test_g1_spawn_collision.py.
"""
import importlib
import sys
import types

import mujoco
import numpy as np

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

FREE_JOINT = "robot0_floating_base_joint"


def load(xml=SCENE):
    """Return (model, data) of the scene xml after mj_forward."""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def free_qadr(m):
    """Return the qpos address of the robot's free joint."""
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, FREE_JOINT)
    return int(m.jnt_qposadr[jid])


def total_penetration(m, d):
    """Return the summed depth (m) of the robot's penetrating environment contacts."""
    import robosuite_common
    return sum(-c.dist for c in robosuite_common.robot_env_contacts(m, d))


class _FakeSimModel:
    """Stand in for robosuite's MjSim.model."""

    def __init__(self, m):
        """Wrap MjModel m."""
        self._model = m

    def get_joint_qpos_addr(self, name):
        """Return the joint's qpos address as an int (robosuite may give a tuple)."""
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self._model.jnt_qposadr[jid])


class _FakeSimData:
    """Stand in for robosuite's MjSim.data."""

    def __init__(self, d):
        """Wrap MjData d."""
        self._data = d

    @property
    def qpos(self):
        """Return the wrapped data's qpos."""
        return self._data.qpos


class _FakeSim:
    """Stand in for robosuite's MjSim."""

    def __init__(self, m, d):
        """Wrap model m and data d."""
        self._m, self._d = m, d
        self.model = _FakeSimModel(m)
        self.data = _FakeSimData(d)

    def forward(self):
        """Run mj_forward."""
        mujoco.mj_forward(self._m, self._d)


class _FakeRobotModel:
    """Stand in for the robot model, with robosuite's naming prefix."""

    naming_prefix = "robot0_"


class _FakeRobot:
    """Stand in for robosuite's robot."""

    robot_model = _FakeRobotModel()


class FakeEnv:
    """Stand in for the env the placement functions take."""

    def __init__(self, m, d):
        """Wrap model m and data d, spawning the base at the origin, 1 m up."""
        self.sim = _FakeSim(m, d)
        self.robots = [_FakeRobot()]
        self.init_robot_base_pos = np.array([0.0, 0.0, 1.0])


def _install_robosuite_stubs():
    """Register stand-ins for the robosuite modules and symbols both robot modules import.

    The placement functions use only numpy and mujoco; only the decorators and base classes the
    robot model classes are declared with are stood in for.
    """
    def mod(name):
        """Return a new empty module registered as name."""
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    base = {"__init__": lambda self, *a, **k: None}
    mod("robosuite")
    mod("robosuite.models")
    grippers = mod("robosuite.models.grippers")
    grippers.register_gripper = lambda cls: cls
    mod("robosuite.models.grippers.gripper_model").GripperModel = type("GripperModel", (), base)
    mod("robosuite.models.grippers.null_gripper").NullGripper = type("NullGripper", (), base)
    mod("robosuite.models.robots")
    mod("robosuite.models.robots.manipulators")
    mod("robosuite.models.robots.manipulators.legged_manipulator_model").LeggedManipulatorModel = \
        type("LeggedManipulatorModel", (), base)
    mod("robosuite.robots").register_robot_class = lambda *a, **k: (lambda cls: cls)


def import_robot_module(name):
    """Import golem_robocasa's module name (h1_2_robosuite, g1.g1_robosuite), standing in for robosuite if it is missing.

    With robosuite installed (the robocasa image) the real modules load, so the stand-ins never
    reach a test that builds a real env in the same process.
    """
    try:
        import robosuite  # noqa: F401
    except ImportError:
        _install_robosuite_stubs()
    return importlib.import_module(name)
