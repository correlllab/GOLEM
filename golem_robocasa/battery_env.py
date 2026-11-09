"""Register the Battery task: the robot at the ARPA battery workcell.

Both sim runners run it (``h12_mujoco.py --task Battery``, ``--g1`` for the
G1). It is a robosuite env, not a RoboCasa kitchen, around the robot the
kitchen tasks use (``H1_2`` or ``G1Dex3``), so every robot0_/gripper0_ name the
bridges resolve is the kitchen tasks'. The arena is ``CL_Assets/mujoco_assets/battery/
battery_cell.xml`` (ARPA cell, battery pack, screw row), plus every MJCF in
``battery/objects/``.

Object files are standalone MuJoCo models that must compile on their own
(e.g. in ``CL_Assets/view_mjcf.py``). From each file the merge takes:

  * the top-level ``<worldbody>`` bodies, at their authored (world) poses.
    Other top-level worldbody elements (a floor, lights, cameras) and the
    ``<option>`` / ``<visual>`` / ``<statistic>`` / ``<size>`` sections only
    serve standalone viewing and are skipped;
  * the ``<asset>`` entries those bodies reference, with files resolved
    against the file's own ``meshdir`` / ``texturedir`` / ``assetdir``.

``<default>`` classes are inlined, degree angles (MuJoCo's default) are
converted to the scene's radians, and each body gets an explicit
``<inertial>`` taken from the standalone compile. That keeps every part's mass
and inertia as authored even though robosuite compiles the merged scene with
``inertiagrouprange="0 0"`` (mass from group-0 geoms only). Names must be
unique across the whole scene.

Importing this module registers the env with robosuite (EnvMeta), so
``robosuite.make("Battery", robots=...)`` resolves it. The task has no
scripted success: the RoboCasa task-signal surface the runners read
(update_state / _check_success / reward / get_ep_meta) is stubbed.
"""
import glob
import math
import os
import xml.etree.ElementTree as ET

import mujoco
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from robosuite.models.tasks import ManipulationTask

BATTERY_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "CL_Assets", "mujoco_assets", "battery"))
ARENA_XML = os.path.join(BATTERY_DIR, "battery_cell.xml")
OBJECTS_DIR = os.path.join(BATTERY_DIR, "objects")

# The --task names this module provides.
TASKS = ("Battery",)

# Pelvis spawn (m, world) and orientation (euler rad): facing +x, 0.24 m short
# of the battery's near wall (x = 0.444), within arm reach of the near modules
# and the screw row. Each runner's spawn placement fits the floor clearance and
# backs the robot off along -x until it clears the cell.
ROBOT_BASE_POS = (0.20, 0.0, 1.03)
ROBOT_BASE_ORI = (0.0, 0.0, 0.0)

# m, the H1-2's spawn clearance for this task when GOLEM_SPAWN_MARGIN is unset
# (the G1 sim does not read it). The kitchen default of 0.20 would hold the
# robot ~0.45 m further back: the binding pose is the sagged, unpowered arms,
# whose grippers hang ~2 cm above the module tops from this spawn.
SPAWN_MARGIN = 0.015

# Top-level sections of an object file: merged into the scene, or only for
# standalone viewing and skipped. Any other section is refused.
_MERGED_SECTIONS = {"compiler", "default", "asset", "worldbody"}
_STANDALONE_SECTIONS = {"option", "visual", "statistic", "size"}
# Asset tags whose file= resolves against meshdir (others: texturedir).
_MESHDIR_TAGS = {"mesh", "hfield", "skin"}
# Attributes that name an asset, by the asset tag they refer to.
_ASSET_REFS = {"material": "material", "mesh": "mesh", "hfield": "hfield"}


def canonical_task(name):
    """Return the TASKS entry matching `name` case-insensitively, else None."""
    for task in TASKS:
        if name.lower() == task.lower():
            return task
    return None


def _fmt(values):
    """Return values as an MJCF attribute string."""
    return " ".join(f"{v:.9g}" for v in values)


def _asset_name(node):
    """Return an asset's name, or its file's stem as MuJoCo names an unnamed mesh/hfield/skin."""
    if node.get("name"):
        return node.get("name")
    if node.get("file"):
        return os.path.splitext(os.path.basename(node.get("file")))[0]
    return None


def _default_classes(root):
    """Return {class name: {tag: attrs}} of the file's <default> tree.

    Nested classes inherit their parent's attributes; the root class is 'main'.
    """
    classes = {}

    def walk(el, inherited):
        """Record el's class with inherited attrs overlaid by its own, then its children."""
        merged = {tag: dict(attrs) for tag, attrs in inherited.items()}
        for child in el:
            if child.tag != "default":
                merged[child.tag] = {**merged.get(child.tag, {}), **child.attrib}
        classes[el.get("class", "main")] = merged
        for child in el.findall("default"):
            walk(child, merged)

    for top in root.findall("default"):
        walk(top, {})
    return classes


def _inline_defaults(path, el, classes, active="main"):
    """Write default-class attributes onto el and its subtree and drop the class references.

    Explicit attributes win over the class's.
    """
    cls = el.attrib.pop("class", None) or active
    if cls not in classes and cls != "main":
        raise ValueError(f"[battery_env] {path}: unknown default class {cls!r}")
    for key, value in classes.get(cls, {}).get(el.tag, {}).items():
        el.attrib.setdefault(key, value)
    child_active = el.attrib.pop("childclass", None) or active
    for child in el:
        _inline_defaults(path, child, classes, child_active)


def _degrees_to_radians(el):
    """Convert el's subtree from compiler angle="degree" to radians."""
    deg = math.pi / 180.0
    for node in el.iter():
        if node.get("euler") is not None:
            node.set("euler", _fmt(float(v) * deg for v in node.get("euler").split()))
        if node.get("axisangle") is not None:
            *axis, angle = (float(v) for v in node.get("axisangle").split())
            node.set("axisangle", _fmt([*axis, angle * deg]))
        if node.tag == "joint" and node.get("type", "hinge") in ("hinge", "ball"):
            for key in ("range", "ref", "springref"):
                if node.get(key) is not None:
                    node.set(key, _fmt(float(v) * deg for v in node.get(key).split()))


def _set_inertials(model, bodies):
    """Give each body in bodies the <inertial> the standalone compile computed for it.

    bodies is in document order, which is MuJoCo's body id order from 1.
    """
    for body_id, body in enumerate(bodies, start=1):
        if body.find("inertial") is not None or model.body_mass[body_id] <= 0.0:
            continue
        inertial = ET.Element("inertial", {
            "pos": _fmt(model.body_ipos[body_id]),
            "quat": _fmt(model.body_iquat[body_id]),
            "mass": f"{model.body_mass[body_id]:.9g}",
            "diaginertia": _fmt(model.body_inertia[body_id]),
        })
        body.insert(0, inertial)


class BatteryArena(Arena):
    """Build battery_cell.xml with every objects/*.xml merged in, sorted by name.

    Objects are merged after Arena.__init__, so robosuite's collision-geom
    recolouring leaves their materials alone.
    """

    def __init__(self, arena_xml=ARENA_XML, objects_dir=OBJECTS_DIR):
        """Load arena_xml and merge each object file of objects_dir."""
        super().__init__(arena_xml)
        self.object_files = sorted(glob.glob(os.path.join(objects_dir, "*.xml")))
        for path in self.object_files:
            self._merge_object(path)

    def _merge_object(self, path):
        """Merge one object file's bodies and the assets they use into the arena.

        Raises for a section, compiler setting, file or asset name it cannot merge.
        """
        root = ET.parse(path).getroot()
        unsupported = {c.tag for c in root} - _MERGED_SECTIONS - _STANDALONE_SECTIONS
        if unsupported:
            raise ValueError(
                f"[battery_env] {path}: unsupported top-level section(s) "
                f"{sorted(unsupported)}; object files may only contain "
                f"{sorted(_MERGED_SECTIONS | _STANDALONE_SECTIONS)}")
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
        if compiler.get("eulerseq", "xyz") != "xyz":
            raise ValueError(f"[battery_env] {path}: only eulerseq=\"xyz\" is supported")

        folder = os.path.dirname(os.path.abspath(path))
        assets = [node for section in root.findall("asset") for node in section]
        for node in assets:
            if node.get("file") is not None:
                subdir = compiler.get("meshdir" if node.tag in _MESHDIR_TAGS
                                      else "texturedir", compiler.get("assetdir", ""))
                resolved = os.path.normpath(os.path.join(folder, subdir, node.get("file")))
                if not os.path.isfile(resolved):
                    raise FileNotFoundError(
                        f"[battery_env] {path}: <{node.tag} name={node.get('name')!r}> "
                        f"file not found: {resolved}")
                node.set("file", resolved)

        model = mujoco.MjModel.from_xml_path(path)
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"[battery_env] {path}: no <worldbody>")
        all_bodies = list(worldbody.iter("body"))
        if len(all_bodies) != model.nbody - 1:
            raise RuntimeError(f"[battery_env] {path}: body count mismatch vs compile")
        _set_inertials(model, all_bodies)

        classes = _default_classes(root)
        for node in assets:
            _inline_defaults(path, node, classes)
        parts = [el for el in worldbody if el.tag == "body"]
        for part in parts:
            _inline_defaults(path, part, classes)
            if compiler.get("angle", "degree") != "radian":
                _degrees_to_radians(part)

        # Only the assets the parts use: a standalone file's floor texture and
        # material stay behind with its floor.
        used = {(tag, el.get(attr)) for part in parts for el in part.iter()
                for attr, tag in _ASSET_REFS.items() if el.get(attr)}
        materials = {_asset_name(n): n for n in assets if n.tag == "material"}
        used |= {("texture", materials[name].get("texture")) for tag, name in used
                 if tag == "material" and name in materials and materials[name].get("texture")}
        for node in assets:
            if (node.tag, _asset_name(node)) not in used:
                continue
            existing = self.asset.find(f"./{node.tag}[@name='{_asset_name(node)}']")
            if existing is not None:
                if existing.attrib != {**node.attrib, "name": _asset_name(node)}:
                    raise ValueError(
                        f"[battery_env] {path}: <{node.tag} name={_asset_name(node)!r}> "
                        f"conflicts with an asset of the same name already in the scene")
                continue
            node.set("name", _asset_name(node))
            self.asset.append(node)
        for part in parts:
            self.worldbody.append(part)

        skipped = sorted({c.tag for c in root} & _STANDALONE_SECTIONS) + [
            f"{el.tag} {el.get('name')!r}" if el.get("name") else el.tag
            for el in worldbody if el.tag != "body"]
        print(f"[battery_env] {os.path.basename(path)}: parts "
              f"{[p.get('name') for p in parts]}"
              + (f"; standalone-only, skipped: {skipped}" if skipped else ""))


class Battery(ManipulationEnv):
    """Stand the robot at the ARPA battery workcell with the objects/ parts on the battery."""

    def __init__(self, robots, use_object_obs=True, **kwargs):
        """Build the env; use_object_obs is accepted for make_kitchen_env and unused."""
        self.use_object_obs = use_object_obs
        self.init_robot_base_pos = list(ROBOT_BASE_POS)
        self.init_robot_base_ori = list(ROBOT_BASE_ORI)
        self.spawn_margin = SPAWN_MARGIN
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        """Put the robot at ROBOT_BASE_POS in a BatteryArena, with multi-point contacts."""
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(ROBOT_BASE_POS)
        self.model = ManipulationTask(
            mujoco_arena=BatteryArena(),
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )
        # The merged scene takes robosuite's <option>, not the arena's, so the
        # arena's multiccd flag is set again: with single-point contacts the
        # screws' small neck cylinders rock at ~6 rad/s on their walls
        # (measured on MuJoCo 3.3.1; unmeasured on the image's 3.17.0).
        option = self.model.root.find("option")
        if option is None:
            option = ET.SubElement(self.model.root, "option")
        flag = option.find("flag")
        if flag is None:
            flag = ET.SubElement(option, "flag")
        flag.set("multiccd", "enable")

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
        return {"lang": "battery workcell: manipulate the parts on the battery pack"}
