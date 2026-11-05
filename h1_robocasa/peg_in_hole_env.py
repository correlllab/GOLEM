"""Native robosuite pegboard workcell using GOLEM's registered H1_2 robot.

The board's functional collision partition and the peg's authored inertia are
retained. Sites are evaluation-only; no object poses are published or consumed
by the robot stack. This scene task does not implement insertion success.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
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


@lru_cache(maxsize=1)
def _catalog():
    with (SCENE_DIR / "manipulation_catalog.json").open() as stream:
        return json.load(stream)


def _polygon_clearance(inner, outer):
    """Minimum boundary distance, or zero if polygons cross or are not nested."""
    inner, outer = np.asarray(inner, dtype=float), np.asarray(outer, dtype=float)
    a, b = outer, np.roll(outer, -1, axis=0)
    p = inner[:, None, :]
    # Ray casting works for the non-convex L profile as well as convex profiles.
    crosses_y = (a[None, :, 1] > p[:, :, 1]) != (b[None, :, 1] > p[:, :, 1])
    dy = b[:, 1] - a[:, 1]
    x_cross = a[None, :, 0] + ((p[:, :, 1] - a[None, :, 1]) *
                 (b[:, 0] - a[:, 0])[None, :] / np.where(dy == 0, 1, dy)[None, :])
    if not np.all(np.count_nonzero(crosses_y & (p[:, :, 0] < x_cross), axis=1) % 2):
        return 0.0

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    c, d = inner[:, None, :], np.roll(inner, -1, axis=0)[:, None, :]
    if np.any((cross(b - a, c - a) * cross(b - a, d - a) < 0) &
              (cross(d - c, a - c) * cross(d - c, b - c) < 0)):
        return 0.0

    def vertex_distance(vertices, polygon):
        start, end = polygon, np.roll(polygon, -1, axis=0)
        edge = end - start
        delta = vertices[:, None, :] - start[None, :, :]
        t = np.clip(np.sum(delta * edge, axis=2) / np.sum(edge * edge, axis=1), 0, 1)
        return np.linalg.norm(delta - t[..., None] * edge, axis=2).min()

    return float(min(vertex_distance(inner, outer), vertex_distance(outer, inner)))


@dataclass(frozen=True)
class PegBoardConfig:
    """World-fixed, level workcell dimensions in metres; robot faces +X.

    Left/right use the reference fixtures' 0.45 m lateral translation. The
    Peg IDs select source geometry and a clearance row, not a scale factor.
    Loose mode requires the entire peg footprint to fit the table's 20 mm
    support strip beside the board. Neither mode scores task success.
    """

    side: str = "left"
    peg_mode: str = "seated"
    peg_id: str = "cylinder_1"
    source_hole: str = "cylinder_1"
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
        catalog = _catalog()
        ids = {f"{shape}_{row}" for shape, rows in catalog["variants"].items() for row in rows}
        if not isinstance(self.peg_id, str) or self.peg_id not in ids:
            raise ValueError(f"unknown peg_id {self.peg_id!r}; expected one of {sorted(ids)}")
        if not isinstance(self.source_hole, str) or self.source_hole not in catalog["holes"]:
            raise ValueError(f"unknown source_hole {self.source_hole!r}")
        shape, row = self.peg_id.rsplit("_", 1)
        peg = catalog["variants"][shape][row]
        hole = catalog["holes"][self.source_hole]
        if shape != hole["shape"]:
            raise ValueError("peg and source_hole must have the same shape")
        w, qx, qy, qz = hole["quat"]
        if qx != 0 or qy != 0:
            raise ValueError("source_hole must have a level authored profile")
        rotation = np.array([[1 - 2 * qz * qz, -2 * w * qz],
                             [2 * w * qz, 1 - 2 * qz * qz]])
        profile = np.asarray(peg["profile_xy"]) @ rotation.T + np.asarray(hole["target"][:2])
        if _polygon_clearance(profile, hole["profile_xy"]) <= 1e-9:
            raise ValueError("peg profile does not fit source_hole with positive clearance")
        if self.peg_mode == "loose":
            bounds = np.asarray(peg["bounds"])
            if (bounds[0, 0] < -.01 or bounds[1, 0] >= .01 or
                    bounds[0, 1] < -.145 or bounds[1, 1] > .145):
                raise ValueError("peg footprint does not fit the loose-mode table support strip")

    @property
    def peg_variant(self):
        shape, row = self.peg_id.rsplit("_", 1)
        return _catalog()["variants"][shape][row]

    @property
    def peg_prefix(self):
        return "peg_cylinder_" if self.peg_id == "cylinder_1" else f"peg_{self.peg_id}_"

    @property
    def source_hole_spec(self):
        return _catalog()["holes"][self.source_hole]

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
            from board_marker import board_marker_layout, mount_board_marker

            self.marker_metadata = mount_board_marker(board, self.asset)
            self.marker_layout_metadata = board_marker_layout()
            self.markers_metadata = [self.marker_metadata, mount_board_marker(
                board, self.asset, name="pegboard_aruco_8", marker_id=8)]
        target = board.find(f"site[@name='pegboard_hole_{config.source_hole}_target']")
        target_pos = [float(v) for v in target.get("pos").split()]
        # Authored peg tips are body-local z=0; the blind floor is board z=.003.
        peg_pos = ((x - 0.17, y, z) if config.peg_mode == "loose" else
                   (x + target_pos[0], y + target_pos[1], z + target_pos[2]))
        variant = config.peg_variant
        filename = Path(variant["xml"]).name
        authored = ET.parse(SCENE_DIR / filename).find("worldbody/body")
        tip = authored.find(f"site[@name='{variant['tip_site']}']")
        if (authored.get("name") != variant["body"] or
                authored.get("pos", "0 0 0") != "0 0 0" or
                authored.get("quat", "1 0 0 0") != "1 0 0 0" or
                tip is None or tip.get("pos") != "0 0 0"):
            raise ValueError("peg XML must have an unshifted body and body-local zero tip")
        peg = _merge_part(self.asset, self.worldbody, filename, config.peg_prefix, peg_pos)
        if config.peg_mode == "seated":
            peg.set("quat", _fmt(config.source_hole_spec["quat"]))
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
        peg, hole = self.scene_config.peg_id, self.scene_config.source_hole
        goal = (f"place the loose {peg} peg into {hole}"
                if self.scene_config.peg_mode == "loose" else
                f"extract the seated {peg} peg from {hole}")
        return {"lang": goal,
                "scene_task": "PegInHole", "success_implemented": False,
                "peg_mode": self.scene_config.peg_mode, "peg_id": peg,
                "source_hole": hole, "target_hole": hole}
