"""Battery-workcell task for the H1-2 sim runner (``h12_mujoco.py --task Battery``).

A robosuite env (not a RoboCasa kitchen) around the same registered ``H1_2``
robot the kitchen tasks use, so every robot0_/gripper0_ name the bridges
resolve is identical. The arena is ``CL_Assets/mujoco_assets/battery/
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
``robosuite.make("Battery", robots="H1_2", ...)`` resolves it. The task has no
scripted success: the RoboCasa task-signal surface the runner reads
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

# Task names this module provides, as given to h12_mujoco.py --task.
TASKS = ("Battery",)

# Pelvis spawn (m, world) and orientation (euler rad): facing +x, 0.24 m short
# of the battery's near wall (x = 0.444), within arm reach of the near modules
# and the screw row. place_robot_collision_free fits floor clearance and backs
# off along -x only if the margin below isn't met.
ROBOT_BASE_POS = (0.20, 0.0, 1.03)
ROBOT_BASE_ORI = (0.0, 0.0, 0.0)

# Default spawn clearance (m) for this task, used when GOLEM_SPAWN_MARGIN is
# unset. The kitchen default of 0.20 would hold the robot ~0.45 m further
# back: the binding pose is the sagged, unpowered arms, whose grippers hang
# ~2 cm above the module tops from this spawn.
SPAWN_MARGIN = 0.015

_MERGED_SECTIONS = {"compiler", "default", "asset", "worldbody"}
_STANDALONE_SECTIONS = {"option", "visual", "statistic", "size"}
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
    return " ".join(f"{v:.9g}" for v in values)


def _asset_name(node):
    """An asset's name; MuJoCo names an unnamed mesh/hfield/skin after its file."""
    if node.get("name"):
        return node.get("name")
    if node.get("file"):
        return os.path.splitext(os.path.basename(node.get("file")))[0]
    return None


def _default_classes(root):
    """{class name: {tag: attrs}} for the file's <default> tree, with nested
    classes inheriting their parent's attributes. The root class is 'main'."""
    classes = {}

    def walk(el, inherited):
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
    """Write default-class attributes onto `el` and its subtree (explicit
    attributes win) and drop the class/childclass references."""
    cls = el.attrib.pop("class", None) or active
    if cls not in classes and cls != "main":
        raise ValueError(f"[battery_env] {path}: unknown default class {cls!r}")
    for key, value in classes.get(cls, {}).get(el.tag, {}).items():
        el.attrib.setdefault(key, value)
    child_active = el.attrib.pop("childclass", None) or active
    for child in el:
        _inline_defaults(path, child, classes, child_active)


def _degrees_to_radians(el):
    """Convert `el`'s subtree from compiler angle="degree" to radians."""
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
    """Give each body in `bodies` (document order == MuJoCo body id order, from
    1) the inertial the standalone compile computed for it."""
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
    """battery_cell.xml with every objects/*.xml merged in (sorted by name).

    Objects are merged after Arena.__init__, so robosuite's collision-geom
    recolouring leaves their materials alone."""

    def __init__(self, arena_xml=ARENA_XML, objects_dir=OBJECTS_DIR):
        super().__init__(arena_xml)
        self.object_files = sorted(glob.glob(os.path.join(objects_dir, "*.xml")))
        for path in self.object_files:
            self._merge_object(path)

    def _merge_object(self, path):
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
    """H1-2 at the ARPA battery workcell with the objects/ parts on the battery."""

    def __init__(self, robots, use_object_obs=True, **kwargs):
        # use_object_obs is part of make_kitchen_env's kwarg set; this env
        # defines no object observables, so it is stored and otherwise unused.
        self.use_object_obs = use_object_obs
        self.init_robot_base_pos = list(ROBOT_BASE_POS)
        self.init_robot_base_ori = list(ROBOT_BASE_ORI)
        self.spawn_margin = SPAWN_MARGIN
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(ROBOT_BASE_POS)
        self.model = ManipulationTask(
            mujoco_arena=BatteryArena(),
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )
        # The merged scene takes robosuite's <option>, not the arena's. The
        # arena's multiccd flag must be carried over: with single-point
        # contacts the screws' small neck cylinders rock at ~6 rad/s on their
        # walls (MuJoCo 3.3.1).
        option = self.model.root.find("option")
        if option is None:
            option = ET.SubElement(self.model.root, "option")
        flag = option.find("flag")
        if flag is None:
            flag = ET.SubElement(option, "flag")
        flag.set("multiccd", "enable")

    # --- RoboCasa task-signal surface read by h12_mujoco / MeasurementBridge
    def update_state(self):
        pass

    def _check_success(self):
        return False

    def reward(self, action=None):
        return 0.0

    def get_ep_meta(self):
        return {"lang": "battery workcell: manipulate the parts on the battery pack"}
