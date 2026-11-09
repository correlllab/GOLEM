"""Register the Blank task: the robot alone on a floor, under a light.

Both sim runners run it (``h12_mujoco.py --task Blank``, with or without
``--g1``). It is a robosuite env, not a RoboCasa kitchen, around the robot the
runner registers (``H1_2`` or ``G1Dex3``), so every robot0_/gripper0_ name the
bridges resolve is the kitchen tasks'. The arena is robosuite's ``EmptyArena``
without its visual walls: its floor plane, skybox, light and fixed cameras
(cameras are not geoms, and no bridge reads them).

Importing this module registers the env with robosuite (EnvMeta), so
``robosuite.make("Blank", robots=...)`` resolves it. The task has no scripted
success: the RoboCasa task-signal surface the runners read (update_state /
_check_success / reward / get_ep_meta) is stubbed.
"""
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask

# The --task names this module provides.
TASKS = ("Blank",)

# Pelvis spawn (m, world) and orientation (euler rad): at the origin, facing +x.
# The z is a start value only: each runner's placement lifts or lowers the
# robot until its lowest collision geom clears the floor.
ROBOT_BASE_POS = (0.0, 0.0, 1.0)
ROBOT_BASE_ORI = (0.0, 0.0, 0.0)

# EmptyArena's walls (contype 0, visual only), which would show in the cameras.
_WALL_PREFIX = "wall_"


def canonical_task(name):
    """Return the TASKS entry matching `name` case-insensitively, else None."""
    for task in TASKS:
        if name.lower() == task.lower():
            return task
    return None


class BlankArena(EmptyArena):
    """Build robosuite's EmptyArena without its walls: a floor and a light."""

    def __init__(self):
        """Load EmptyArena and remove its wall geoms."""
        super().__init__()
        for geom in list(self.worldbody.findall("geom")):
            if (geom.get("name") or "").startswith(_WALL_PREFIX):
                self.worldbody.remove(geom)


class Blank(ManipulationEnv):
    """Stand the robot at the origin of an empty world."""

    def __init__(self, robots, **kwargs):
        """Build the env around robots (robosuite's robots argument)."""
        self.init_robot_base_pos = list(ROBOT_BASE_POS)
        self.init_robot_base_ori = list(ROBOT_BASE_ORI)
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        """Put the robot at ROBOT_BASE_POS in a BlankArena."""
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(ROBOT_BASE_POS)
        self.model = ManipulationTask(
            mujoco_arena=BlankArena(),
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def update_state(self):
        """Do nothing: the task has no fixtures to update."""
        pass

    def _check_success(self):
        """Return False: the task has no success condition."""
        return False

    def reward(self, action=None):
        """Return 0: the task has no reward."""
        return 0.0

    def get_ep_meta(self):
        """Return the episode's language goal, which MeasurementBridge publishes."""
        return {"lang": "blank world: the robot alone on an empty floor"}
