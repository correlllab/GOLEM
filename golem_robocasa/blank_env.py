"""Blank task for both sim runners (``h12_mujoco.py --task Blank``, with or
without ``--g1``): the robot alone on a floor, under a light.

A robosuite env (not a RoboCasa kitchen) around the robot the runner registers
(``H1_2`` or ``G1Dex3``), so every robot0_/gripper0_ name the bridges resolve
is the kitchen tasks'. The arena is robosuite's ``EmptyArena`` without its
visual walls: its floor plane, skybox, light and fixed cameras (cameras are not
geoms, and no bridge reads them).

Importing this module registers the env with robosuite (EnvMeta), so
``robosuite.make("Blank", robots=...)`` resolves it. The task has no scripted
success: the RoboCasa task-signal surface the runners read (update_state /
_check_success / reward / get_ep_meta) is stubbed, as battery_env's is.
"""
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask

# Task names this module provides, as given to h12_mujoco.py --task.
TASKS = ("Blank",)

# Pelvis spawn (m, world) and orientation (euler rad): at the origin, facing +x.
# The z is a start value only: each runner's placement lifts or lowers the
# robot until its lowest collision geom clears the floor.
ROBOT_BASE_POS = (0.0, 0.0, 1.0)
ROBOT_BASE_ORI = (0.0, 0.0, 0.0)

# EmptyArena's geoms other than its floor: walls drawn around the workspace
# (contype 0, visual only), which would show in the cameras.
_WALL_PREFIX = "wall_"


def canonical_task(name):
    """Return the TASKS entry matching `name` case-insensitively, else None."""
    for task in TASKS:
        if name.lower() == task.lower():
            return task
    return None


class BlankArena(EmptyArena):
    """robosuite's EmptyArena with its walls removed: a floor and a light."""

    def __init__(self):
        super().__init__()
        for geom in list(self.worldbody.findall("geom")):
            if (geom.get("name") or "").startswith(_WALL_PREFIX):
                self.worldbody.remove(geom)


class Blank(ManipulationEnv):
    """The robot standing at the origin of an empty world."""

    def __init__(self, robots, use_object_obs=True, **kwargs):
        # use_object_obs is part of make_kitchen_env's kwarg set; this env
        # defines no object observables, so it is stored and otherwise unused.
        self.use_object_obs = use_object_obs
        self.init_robot_base_pos = list(ROBOT_BASE_POS)
        self.init_robot_base_ori = list(ROBOT_BASE_ORI)
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(ROBOT_BASE_POS)
        self.model = ManipulationTask(
            mujoco_arena=BlankArena(),
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    # --- RoboCasa task-signal surface read by the runners / MeasurementBridge
    def update_state(self):
        pass

    def _check_success(self):
        return False

    def reward(self, action=None):
        return 0.0

    def get_ep_meta(self):
        return {"lang": "blank world: the robot alone on an empty floor"}
