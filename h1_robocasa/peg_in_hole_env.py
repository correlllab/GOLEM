"""Native robosuite pegboard workcell using GOLEM's registered H1_2 robot.

The board's functional collision partition and the peg's authored inertia are
retained. Sites are evaluation-only; no object poses are published or consumed
by the robot stack. This scene task does not implement insertion success.
"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask


ASSETS_ROOT = Path(__file__).resolve().parent.parent / "CL_Assets"
SCENE_DIR = ASSETS_ROOT / "mujoco_assets" / "manipulation_net"
TASKS = ("PegInHole",)
TARGET_HOLE = "cylinder_1"
ROBOT_BASE_POS = (0.0, 0.0, 1.03)


def canonical_task(name):
    return next((task for task in TASKS if task.lower() == name.lower()), None)


def _fmt(values):
    return " ".join(f"{v:.12g}" for v in values)


@dataclass(frozen=True)
class PegBoardConfig:
    """World-fixed, level workcell dimensions in metres; robot faces +X.

    Left/right use the reference fixtures' 0.45 m lateral translation. The
    cylinder starts seated in its blind hole; loose mode uses the table's
    20 mm support strip beside the board. Neither mode scores task success.
    """

    side: str = "left"
    peg_mode: str = "seated"
    table_x: float = 0.503832018731
    table_y: float = 0.185369815094
    table_height: float = 0.955172239917
    marker: bool = True

    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if self.peg_mode not in ("loose", "seated"):
            raise ValueError("peg_mode must be 'loose' or 'seated'")
        if self.table_height <= 0.04:
            raise ValueError("table_height must exceed the 0.04 m tabletop thickness")

    @property
    def board_pos(self):
        return (self.table_x, self.table_y if self.side == "left" else self.table_y - 0.45,
                self.table_height)


def _merge_part(asset, worldbody, filename, prefix, position):
    """Merge the selected CL_Assets part, namespacing names AND references.

    These assets use radians, explicit inertia, no defaults and CL_Assets-root
    relative files. Reject a changed schema rather than silently losing physics.
    """
    root = ET.parse(SCENE_DIR / filename).getroot()
    if {child.tag for child in root} - {"compiler", "asset", "worldbody"}:
        raise ValueError(f"unsupported sections in {filename}")
    compiler = root.find("compiler")
    if compiler is None or compiler.attrib != {"angle": "radian"}:
        raise ValueError(f"unexpected compiler settings in {filename}")
    for node in root.iter():
        for attr in ("name", "mesh", "material", "texture", "joint", "body", "site"):
            if node.get(attr):
                node.set(attr, prefix + node.get(attr))
        if node.get("file"):
            path = (ASSETS_ROOT / node.get("file")).resolve()
            if not path.is_relative_to(ASSETS_ROOT.resolve()) or not path.is_file():
                raise FileNotFoundError(path)
            node.set("file", str(path))
    bodies = root.findall("worldbody/body")
    if len(bodies) != 1:
        raise ValueError(f"expected one top-level body in {filename}")
    body = bodies[0]
    body.set("pos", _fmt(position))
    asset.extend(root.find("asset"))
    worldbody.append(body)
    return body


class PegBoardArena(EmptyArena):
    def __init__(self, config):
        super().__init__()
        x, y, z = config.board_pos
        # Floor names are consumed by GOLEM's collision-free spawn exclusion.
        self.floor.set("name", "floor")
        self.floor.set("group", "1")
        table = ET.SubElement(self.worldbody, "body", name="pegboard_table")
        ET.SubElement(table, "geom", name="pegboard_table_top", type="box",
                      pos=_fmt((x, y, z - 0.02)), size="0.18 0.145 0.02",
                      rgba="0.55 0.40 0.28 1", group="1", friction="0.8 0.005 0.0001")
        for i, (dx, dy) in enumerate(((-0.125, -0.09), (-0.125, 0.09),
                                      (0.125, -0.09), (0.125, 0.09))):
            ET.SubElement(table, "geom", name=f"pegboard_table_leg_{i}", type="box",
                          pos=_fmt((x + dx, y + dy, (z - 0.04) / 2)),
                          size=_fmt((0.025, 0.025, (z - 0.04) / 2)),
                          rgba="0.55 0.40 0.28 1", group="1", friction="0.8 0.005 0.0001")
        board = _merge_part(self.asset, self.worldbody,
                            "manipulation_board_functional.xml", "pegboard_", (x, y, z))
        if config.marker:
            from board_marker import mount_board_marker

            self.marker_metadata = mount_board_marker(board, self.asset)
        target = board.find(f"site[@name='pegboard_hole_{TARGET_HOLE}_target']")
        target_pos = [float(v) for v in target.get("pos").split()]
        # The authored cylinder tip is body-local z=0; its 50 mm body therefore
        # spans board z=.003..053 when seated, 22 mm below the .025 entry plane.
        peg_pos = ((x - 0.17, y, z) if config.peg_mode == "loose" else
                   (x + target_pos[0], y + target_pos[1], z + target_pos[2]))
        _merge_part(self.asset, self.worldbody, "manipulation_peg_cylinder.xml",
                    "peg_cylinder_", peg_pos)
        # Diagnostic view, independent of the real-camera bridge topic surface.
        ET.SubElement(self.worldbody, "camera", name="pegboard_overview",
                      pos=_fmt((x, y, z + 0.8)), quat="1 0 0 0", fovy="45")


class PegInHole(ManipulationEnv):
    def __init__(self, robots, use_object_obs=True, scene_config=None, **kwargs):
        self.scene_config = scene_config or PegBoardConfig()
        self.use_object_obs = use_object_obs
        self.init_robot_base_pos = list(ROBOT_BASE_POS)
        self.init_robot_base_ori = [0.0, 0.0, 0.0]
        self.spawn_margin = 0.015
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(ROBOT_BASE_POS)
        self.model = ManipulationTask(
            mujoco_arena=PegBoardArena(self.scene_config),
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )
        option = self.model.root.find("option")
        if option is None:
            option = ET.SubElement(self.model.root, "option")
        option.set("timestep", "0.001")

    def update_state(self):
        pass

    def _check_success(self):
        return False

    def reward(self, action=None):
        return 0.0

    def get_ep_meta(self):
        goal = ("place the loose cylinder peg into cylinder_1"
                if self.scene_config.peg_mode == "loose" else
                "extract the seated cylinder peg from cylinder_1")
        return {"lang": goal,
                "scene_task": "PegInHole", "success_implemented": False,
                "peg_mode": self.scene_config.peg_mode, "target_hole": TARGET_HOLE}
