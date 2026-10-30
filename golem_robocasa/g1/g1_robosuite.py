"""Register the Unitree G1 (29 DoF + Dex3-1 hands) as a first-class robosuite robot.

Importing this module once (before ``robocasa.utils.env_utils.create_env`` /
``robosuite.make``) makes ``robots="G1Dex3"`` resolvable and installs the RoboCasa
placement patch so the legged humanoid is positioned by its pelvis free joint
(RoboCasa's stock path writes mobile-base joints that this robot lacks).

The robot MJCF is generated at import time from ``CL_Assets/g1``'s
``g1_29dof_with_hand_rev_1_0.xml`` (plus the sensor frames of its URDF) into a
temp dir, since robosuite loads robots from a file. The robot keeps its
floating base and all 43 motors (29 body + 14 Dex3-1 hand joints); the hands
are part of the robot model, so each arm gets a massless NullGripper. Locomotion is driven externally over
DDS: robosuite controllers are never stepped (we own ``mujoco.mj_step``).

NOTE: robosuite/robocasa import only inside the MuJoCo container (mujoco==3.3.1).
The class is named G1Dex3 because robosuite_models (installed in the image)
already registers a "G1", which RoboCasa gives its own placement offsets.
"""
import atexit
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.null_gripper import NullGripper
from robosuite.models.robots.manipulators.legged_manipulator_model import (
    LeggedManipulatorModel,
)
from robosuite.robots import register_robot_class

# --------------------------------------------------------------------------- #
# Source files (resolve in both the container, /home/code/..., and on the host)
# --------------------------------------------------------------------------- #
G1_ASSETS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "CL_Assets", "g1", "g1_description"))
SOURCE_XML = os.path.join(G1_ASSETS, "g1_29dof_with_hand_rev_1_0.xml")
SOURCE_URDF = os.path.join(G1_ASSETS, "g1_29dof_with_hand_rev_1_0.urdf")

# Free joint name on the pelvis (used by the placement patch).
FREE_JOINT_NAME = "floating_base_joint"

# The G1's joint dynamics and stance are Unitree's, from unitree_rl_gym
# (github.com/unitreerobotics/unitree_rl_gym, BSD-3-Clause): the default joint of
# resources/robots/g1_description/g1_12dof.xml (damping="0.001" armature="0.01"
# frictionloss="0.1"), and the default_joint_angles of
# legged_gym/envs/g1/g1_config.py. Unitree's G1 MJCF in CL_Assets gives its joints none.
JOINT_DAMPING = 0.001      # N*m*s/rad
JOINT_ARMATURE = 0.01      # kg*m^2
JOINT_FRICTIONLOSS = 0.1   # N*m
# Every body joint gets them. The Dex3-1 fingers get the armature and a tenth of
# the frictionloss: with the hand driver's soft gains (kp 1.5), 0.1 Nm would stop
# each finger ~0.07 rad short of its target, 0.01 Nm ~0.007 rad. Without any, a
# hand nobody commands droops (the thumb swings to its 1.05 rad limit), and the
# driver's close, which starts from the current pose, then jams the thumb on the
# middle finger. robosuite fills a missing damping/frictionloss with 0.1, so the
# hands carry explicit values.
BODY_JOINT_DYNAMICS = {"damping": JOINT_DAMPING, "armature": JOINT_ARMATURE,
                       "frictionloss": JOINT_FRICTIONLOSS}
HAND_JOINT_DYNAMICS = {"damping": 0.0, "armature": JOINT_ARMATURE,
                       "frictionloss": 0.1 * JOINT_FRICTIONLOSS}

# The spawn stance, keyed by joint name (rad): the legs at Unitree's
# default_joint_angles, every other joint (waist, arms, hands) at 0.
# G1Dex3.init_qpos orders it by the model's joints.
STANCE = {f"{side}_{joint}_joint": q
          for side in ("left", "right")
          for joint, q in (("hip_pitch", -0.1), ("hip_roll", 0.0), ("hip_yaw", 0.0),
                           ("knee", 0.3), ("ankle_pitch", -0.2), ("ankle_roll", 0.0))}

# URDF fixed joints whose child links become frames in the MJCF: the sensors
# as sites on torso_link, the palms as the (empty) eef bodies robosuite merges
# a gripper onto.
SENSOR_JOINTS = ("mid360_joint", "d435_joint")
PALM_JOINTS = {"right": "right_hand_palm_joint", "left": "left_hand_palm_joint"}

# The head camera looks along the D435 link's +x (its optical axis) with image
# up along its +z: xyaxes "0 -1 0 0 0 1" in that frame. fovy 42.5: the D435's
# color stream, which /realsense/head/color and aligned_depth_to_color follow
# (Intel's datasheet: 69 x 42 deg, +-1; square pixels at 16:9 give 69.3 x 42.5,
# fx 925.7 at 1280x720).
_CAM_IN_D435 = np.array([0.5, 0.5, -0.5, -0.5])
HEAD_CAM_FOVY = 42.5


def _euler_to_wxyz(euler):
    """RoboCasa base orientation -> wxyz quat. Accepts euler [roll,pitch,yaw],
    a yaw scalar, or an existing wxyz quat. RoboCasa stores init_robot_base_ori
    as a 3-vector euler (typically [0,0,yaw]). The same fixed-axis roll-pitch-yaw
    convention as URDF rpy."""
    if euler is None:
        return np.array([1.0, 0.0, 0.0, 0.0])
    e = np.asarray(euler, dtype=float).reshape(-1)
    if e.size == 4:
        return e
    if e.size == 1:
        e = np.array([0.0, 0.0, float(e[0])])
    r, p, y = e[0], e[1], e[2]
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _urdf_frame(urdf_root, joint_name):
    """(parent link, child link, pos, wxyz quat) of a URDF fixed joint."""
    joint = urdf_root.find(f"joint[@name='{joint_name}']")
    if joint is None:
        raise KeyError(f"URDF joint {joint_name} not found")
    origin = joint.find("origin")
    pos = np.array(origin.get("xyz", "0 0 0").split(), dtype=float)
    quat = _euler_to_wxyz(np.array(origin.get("rpy", "0 0 0").split(), dtype=float))
    return joint.find("parent").get("link"), joint.find("child").get("link"), pos, quat


def _fmt(values):
    return " ".join(f"{v:.9g}" for v in values)


def build_robot_xml(source_xml=SOURCE_XML, source_urdf=SOURCE_URDF):
    """The robosuite robot MJCF (an ElementTree root) for the CL_Assets G1.

    Transforms applied to the source MJCF:
      * <compiler meshdir> dropped; mesh file= made absolute (robosuite joins
        file= with the XML's own dir and ignores meshdir).
      * the file's scene (<statistic>, <visual>, light, floor, skybox and ground
        texture/material) stripped: it belongs to the arena.
      * free joint  <joint type="free">  ->  <freejoint/>  (kept; full-legged base).
      * joint dynamics as above; no joint is renamed.
      * sites mid360_link / d435_link and camera head_cam on torso_link from the
        URDF, a framequat next to the pelvis IMU's gyro/accelerometer, the empty
        eef bodies left/right_hand_palm_link, and the per-arm {arm}_center sites.
    """
    root = ET.parse(source_xml).getroot()
    urdf = ET.parse(source_urdf).getroot()

    compiler = root.find("compiler")
    mesh_dir = os.path.join(os.path.dirname(os.path.abspath(source_xml)),
                            compiler.attrib.pop("meshdir", ""))
    for mesh in root.iter("mesh"):
        mesh.set("file", os.path.join(mesh_dir, mesh.get("file")))

    # scene: the robot's only non-mesh assets and non-body world elements
    for tag in ("statistic", "visual"):
        for elem in root.findall(tag):
            root.remove(elem)
    for section, keep in (("asset", "mesh"), ("worldbody", "body")):
        for elem in root.findall(section):
            for child in list(elem):
                if child.tag != keep:
                    elem.remove(child)
            if not len(elem):
                root.remove(elem)

    pelvis = root.find("worldbody/body")
    free = pelvis.find("joint[@type='free']")
    if free is None or free.get("name") != FREE_JOINT_NAME:
        raise ValueError(f"expected the free joint {FREE_JOINT_NAME} on the root body")
    index = list(pelvis).index(free)
    pelvis.remove(free)
    pelvis.insert(index, ET.Element("freejoint", {"name": FREE_JOINT_NAME}))

    for joint in pelvis.iter("joint"):
        dynamics = HAND_JOINT_DYNAMICS if "_hand_" in joint.get("name") else BODY_JOINT_DYNAMICS
        for attr, value in dynamics.items():
            joint.set(attr, str(value))

    bodies = {body.get("name"): body for body in pelvis.iter("body")}
    for joint_name in SENSOR_JOINTS:
        parent, child, pos, quat = _urdf_frame(urdf, joint_name)
        ET.SubElement(bodies[parent], "site", {"name": child, "pos": _fmt(pos),
                                               "quat": _fmt(quat), "size": "0.01"})
        if child == "d435_link":
            cam_quat = np.zeros(4)
            mujoco.mju_mulQuat(cam_quat, quat, _CAM_IN_D435)
            ET.SubElement(bodies[parent], "camera", {"name": "head_cam", "pos": _fmt(pos),
                                                     "quat": _fmt(cam_quat),
                                                     "fovy": str(HEAD_CAM_FOVY)})
    for joint_name in PALM_JOINTS.values():
        parent, child, pos, quat = _urdf_frame(urdf, joint_name)
        ET.SubElement(bodies[parent], "body", {"name": child, "pos": _fmt(pos), "quat": _fmt(quat)})

    # robosuite's composite controller resolves a per-arm base site
    # "{prefix}{arm}_center" in update_state(); put them where the arms mount
    # on the torso (like GR1's right_center/left_center in its arm mounts).
    torso = bodies["torso_link"]
    for arm in ("right", "left"):
        mount = bodies[f"{arm}_shoulder_pitch_link"]
        ET.SubElement(torso, "site", {"name": f"{arm}_center", "pos": mount.get("pos"),
                                      "quat": mount.get("quat", "1 0 0 0"), "size": "0.01",
                                      "group": "1", "rgba": "1 0.3 0.3 1"})

    ET.SubElement(root.find("sensor"), "framequat",
                  {"name": "imu-pelvis-orientation", "objtype": "site", "objname": "imu_in_pelvis"})
    return root


def _write_robot_xml():
    """Generate the robot MJCF once per process into a temp dir removed at exit."""
    tmp_dir = tempfile.mkdtemp(prefix="g1_robosuite_")
    atexit.register(shutil.rmtree, tmp_dir, True)
    path = os.path.join(tmp_dir, "robot.xml")
    root = build_robot_xml()
    ET.indent(root, space="  ")
    with open(path, "w") as f:
        f.write(f"<!-- generated by golem_robocasa/g1/g1_robosuite.py from {SOURCE_XML} -->\n")
        f.write(ET.tostring(root, encoding="unicode"))
    return path


ROBOT_XML = _write_robot_xml()


# --------------------------------------------------------------------------- #
# Gripper
# --------------------------------------------------------------------------- #
@register_gripper
class MasslessNullGripper(NullGripper):
    """robosuite's NullGripper without its 0.3 kg placeholder inertia.

    The Dex3-1 hands are part of the G1 model, so the gripper only supplies the
    eef body, grip_site and force_ee/torque_ee robosuite requires. We never call
    env.step(), so format_action is the identity and there is no controller.
    """

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        body = self.worldbody.find("body")
        body.remove(body.find("inertial"))


# --------------------------------------------------------------------------- #
# Robot model
# --------------------------------------------------------------------------- #
@register_robot_class("LeggedRobot")
class G1Dex3(LeggedManipulatorModel):
    """Full-legged Unitree G1 (29 DoF) with its Dex3-1 hands in the robot model.

    Keeps the real pelvis free joint + all 43 actuators so the ROS walking policy,
    arm IK and hand driver drive every joint over DDS, and the legs can locomote
    via floor contact. RoboCasa placement is redirected to the pelvis free joint
    by ``install_robocasa_patches()`` below.

    Base = ``NullBase`` (a ``NullBaseModel``), NOT ``NoActuationBase``: the latter
    is a ``MobileBaseModel`` and routes through ``add_mobile_base()``, which
    deletes the pelvis ``<freejoint>`` ("freejoints can only be used at top
    level") and rigidly welds the robot to a jointless ``mobilebase0`` — i.e.
    fixes the robot to the world. ``NullBase`` routes through ``add_null_base()``,
    which leaves the robot tree (and the freejoint) untouched, so the floating
    base survives and the robot can fall/step/balance under physics.

    No joint is renamed. robosuite classifies joints by substring ("leg",
    "torso", "head", "mobile"), so every G1 joint counts as an arm joint; that
    only shapes the construction-only arm controllers, which we never step.
    """

    arms = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(ROBOT_XML, idn=idn)

    @property
    def default_base(self):
        return "NullBase"

    @property
    def default_gripper(self):
        return {"right": "MasslessNullGripper", "left": "MasslessNullGripper"}

    @property
    def default_controller_config(self):
        # Abstract -> must exist, but effectively unused here: robosuite loads
        # controllers via load_composite_controller_config(robot="G1Dex3"), which
        # maps any name containing "G1" to "g1", finds no default_g1.json, and
        # silently falls back to the BASIC composite controller (no crash). We
        # never call env.step(), so the controller is construction-only and this
        # value inert.
        return {"right": "default_g1", "left": "default_g1"}

    @property
    def init_qpos(self):
        return np.array([STANCE.get(name, 0.0) for name in self._joints])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0.0),
            "empty": (-0.29, 0, 0.0),
            "table": lambda table_length: (-0.26 - table_length / 2, 0, 0.0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        return {"right": "right_hand_palm_link", "left": "left_hand_palm_link"}


# --------------------------------------------------------------------------- #
# RoboCasa placement patch: position the full-legged humanoid by its pelvis free
# joint instead of the absent mobilebase0_joint_mobile_* joints.
# --------------------------------------------------------------------------- #
def place_robot_freejoint(env, pos, quat_wxyz):
    """Write the pelvis free joint qpos (xyz + wxyz quat) for a G1Dex3 env.

    Robust primitive usable directly from the eval harness regardless of whether
    the monkeypatch below matched the installed RoboCasa version.
    """
    sim = env.sim
    jname = f"{env.robots[0].robot_model.naming_prefix}{FREE_JOINT_NAME}"
    addr = sim.model.get_joint_qpos_addr(jname)
    start = addr[0] if isinstance(addr, tuple) else addr
    sim.data.qpos[start : start + 3] = np.asarray(pos, dtype=float)
    sim.data.qpos[start + 3 : start + 7] = np.asarray(quat_wxyz, dtype=float)
    sim.forward()


def lowest_robot_point_z(env, prefixes=("robot0_", "gripper0_")):
    """Min world-z over the robot's own colliding geoms (not fixtures/floor), using
    each geom's tight AABB transformed to world. Bounding-sphere rbound is far too
    conservative for flat feet, so we use geom_aabb. Visual geoms are skipped:
    the G1's foot meshes reach 1.7 cm below its collision spheres."""
    m = env.sim.model._model
    d = env.sim.data._data
    corners = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
    )
    best = float("inf")
    for i in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
        if not bn.startswith(prefixes) or not (m.geom_contype[i] or m.geom_conaffinity[i]):
            continue
        aabb = m.geom_aabb[i]
        world = (corners * aabb[3:] + aabb[:3]) @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i]
        best = min(best, float(world[:, 2].min()))
    return best


def place_robot_clear(env, pos, quat_wxyz, clearance=0.02):
    """Place the pelvis free joint, then lift so the lowest robot point clears the
    floor (z=0) by `clearance`. Robust to stance/init_qpos."""
    place_robot_freejoint(env, pos, quat_wxyz)  # writes pose + sim.forward()
    low = lowest_robot_point_z(env)
    if low != float("inf"):
        sim = env.sim
        jname = f"{env.robots[0].robot_model.naming_prefix}{FREE_JOINT_NAME}"
        addr = sim.model.get_joint_qpos_addr(jname)
        start = addr[0] if isinstance(addr, tuple) else addr
        sim.data.qpos[start + 2] += clearance - low
        sim.forward()


def _backward_dir(base_quat):
    """World-frame unit vector pointing 'backward' for a robot at orientation
    base_quat (wxyz). The robot faces +x in its own frame (toward the kitchen
    counter), so backward = -(local +x) rotated into the world, flattened to the
    floor plane and normalized. Falls back to -x if the rotated axis is vertical."""
    fwd = np.zeros(3)
    mujoco.mju_rotVecQuat(fwd, np.array([1.0, 0.0, 0.0]), np.asarray(base_quat, dtype=float))
    back = -fwd
    back[2] = 0.0
    n = np.linalg.norm(back)
    return back / n if n > 1e-9 else np.array([-1.0, 0.0, 0.0])


def _robot_env_contacts(model, data, prefixes=("robot0_", "gripper0_")):
    """Penetrating contacts between a robot geom and a non-robot (environment)
    geom. 'Penetrating' = contact.dist < 0 (actual overlap, not merely within
    margin). Robot-vs-robot (self) and env-vs-env contacts are skipped — moving
    the base can't fix a self-collision. The floor is excluded implicitly:
    place_robot_clear keeps the robot above it, so it never penetrates."""
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist >= 0:
            continue
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
        if b1.startswith(prefixes) == b2.startswith(prefixes):
            continue  # both robot or both env -> not a robot<->env collision
        out.append(c)
    return out


def place_robot_collision_free(env, base_pos, base_quat, step=0.02, max_iters=25,
                               clearance=0.02, extra_backoff=0.0):
    """Place the robot collision-free at spawn by backing it away from whatever it
    overlaps.

    Starting from the RoboCasa placement (base_pos, base_quat wxyz), place the
    robot (place_robot_clear auto-fits floor clearance), then while any robot geom
    penetrates an environment geom, step the base backward (the robot's -x, away
    from the counter it faces) by `step` and retry, up to `max_iters` steps
    (default 25 x 2 cm = 0.5 m). Orientation never changes — pure translation.

    Tracks the least-penetrating position seen (fewest contacts, then least total
    depth). If still colliding after max_iters, restores that best position and
    warns (best-effort), so the sim still launches. Self-collisions at the zero
    pose are not addressed here — translation can't fix them.

    extra_backoff: after clearing the spawn collision, keep backing off up to this
    many more metres (stopping before a new collision behind) so the robot stands
    in OPEN FLOOR with room in front — needed for nav2, which rejects a goal when
    the robot's start cell is occupied. Default 0 keeps the robot at the counter
    for manipulation reach."""
    back = _backward_dir(base_quat)
    model = env.sim.model._model
    data = env.sim.data._data
    pos = np.array(base_pos, dtype=float)
    best_pos, best_score = None, None
    for i in range(max_iters + 1):
        place_robot_clear(env, pos, base_quat, clearance)  # writes pose + z-fit + sim.forward()
        contacts = _robot_env_contacts(model, data)
        score = (len(contacts), sum(-c.dist for c in contacts))
        if best_score is None or score < best_score:
            best_score, best_pos = score, pos.copy()
        if not contacts:
            if i:
                print(f"[g1_robosuite] moved robot back {i * step:.2f} m to clear spawn collision")
            if extra_backoff > 0.0:
                last_clear, moved = pos.copy(), 0.0
                for _ in range(int(extra_backoff / step)):
                    pos[:2] += step * back[:2]
                    place_robot_clear(env, pos, base_quat, clearance)
                    if _robot_env_contacts(model, data):
                        break
                    last_clear, moved = pos.copy(), moved + step
                place_robot_clear(env, last_clear, base_quat, clearance)
                if moved:
                    print(f"[g1_robosuite] backed robot {moved:.2f} m further into open floor (nav spawn)")
            return
        pos[:2] += step * back[:2]
    place_robot_clear(env, best_pos, base_quat, clearance)  # restore least-penetrating try
    print(f"[g1_robosuite] WARNING: spawn still in collision after backing off "
          f"{max_iters * step:.2f} m; keeping best position "
          f"({best_score[0]} contact(s), {best_score[1]:.3f} m penetration)")


_PATCHED = False


def install_robocasa_patches():
    """Redirect RoboCasa base placement to the pelvis free joint for G1Dex3.

    RoboCasa's Kitchen._reset_internal positions the robot two ways and BOTH
    raise for our NullBase + freejoint robot (they write the absent
    mobilebase0_joint_mobile_* joints):
      * fresh/random branch: EnvUtils.set_robot_base(env=..., anchor_pos=...,
        anchor_ori=..., ...) -> returns robot_pos, which kitchen stores.
      * cached (ep_meta) branch: EnvUtils.set_robot_to_position(env, global_pos).
    We fully short-circuit both for G1Dex3 (never call the originals, which would
    KeyError) and write the free joint instead. Other robots are untouched.
    Note set_robot_base internally calls the module-global set_robot_to_position,
    so short-circuiting set_robot_base (not falling through) is required.
    """
    global _PATCHED
    if _PATCHED:
        return
    import warnings

    try:
        from robocasa.utils import env_utils as EU
    except Exception as e:  # pragma: no cover - container only
        warnings.warn(f"[g1_robosuite] robocasa not importable: {e}")
        return

    # Placement offset (x, y along the fixture, then absolute pelvis z) from
    # RoboCasa's anchor, which stands a mobile base 0.20 m off the counter front.
    # At the stance the G1's forearms point forward and its fingers reach 0.42 m
    # ahead of the pelvis at counter-top height: 8 of 12 kitchens tried needed the
    # pelvis 0.27 m further back than the anchor to clear the counter, so it
    # stands 0.30 m back. Pelvis z 0.80 keeps the soles ~1.5 cm clear at the
    # stance (the harness additionally auto-fits clearance via place_robot_clear()).
    offsets = getattr(EU, "_ROBOT_POS_OFFSETS", None)
    if isinstance(offsets, dict):
        offsets.setdefault("G1Dex3", [0.0, -0.30, 0.80])

    def _is_g1(env):
        try:
            return isinstance(env.robots[0].robot_model, G1Dex3)
        except Exception:
            return False

    orig_set_base = getattr(EU, "set_robot_base", None)
    orig_set_pos = getattr(EU, "set_robot_to_position", None)

    if orig_set_base is not None:
        def set_robot_base(*args, **kwargs):
            env = kwargs.get("env", args[0] if args else None)
            if env is not None and _is_g1(env):
                anchor_pos = kwargs.get("anchor_pos", args[1] if len(args) > 1 else None)
                anchor_ori = kwargs.get("anchor_ori", args[2] if len(args) > 2 else None)
                pos = np.array(anchor_pos, dtype=float) if anchor_pos is not None else np.zeros(3)
                env.init_robot_base_pos = pos
                env.init_robot_base_ori = (
                    np.array(anchor_ori, dtype=float) if anchor_ori is not None else np.zeros(3)
                )
                place_robot_freejoint(env, pos, _euler_to_wxyz(anchor_ori))
                return pos  # kitchen does: self.init_robot_base_pos = <return value>
            return orig_set_base(*args, **kwargs)

        EU.set_robot_base = set_robot_base

    if orig_set_pos is not None:
        def set_robot_to_position(*args, **kwargs):
            env = kwargs.get("env", args[0] if args else None)
            global_pos = kwargs.get("global_pos", args[1] if len(args) > 1 else None)
            if env is not None and _is_g1(env) and global_pos is not None:
                ori = getattr(env, "init_robot_base_ori", None)
                place_robot_freejoint(env, np.array(global_pos, dtype=float), _euler_to_wxyz(ori))
                return
            return orig_set_pos(*args, **kwargs)

        EU.set_robot_to_position = set_robot_to_position

    if orig_set_base is None and orig_set_pos is None:
        warnings.warn(
            "[g1_robosuite] no RoboCasa placement entry point found to patch; "
            "call place_robot_freejoint(env, pos, quat) from the harness after reset."
        )
    _PATCHED = True


# --------------------------------------------------------------------------- #
# Convenience env constructor
# --------------------------------------------------------------------------- #
def make_kitchen_env(env_name, layout_ids=None, style_ids=None, seed=None, **kwargs):
    """Build a RoboCasa kitchen task env with robots='G1Dex3', headless and without
    camera observations.

    We bypass robocasa.utils.env_utils.create_env on purpose: it hard-codes
    has_offscreen_renderer=True, use_camera_obs=True and agentview camera_names
    (merged inline via **kwargs, so they can't be overridden), and robosuite's
    reset() unconditionally calls initialize_renderer() which then fails on the
    missing agentview cameras. We own visualization (mujoco.viewer) and read env
    state directly, so we disable robosuite's renderer + camera obs entirely.

    The Kitchen task classes are registered in robosuite.REGISTERED_ENVS by
    robocasa's metaclass (on `import robocasa`, triggered by
    install_robocasa_patches), so robosuite.make resolves the task by name. With
    controller_configs omitted, the robot falls back to the BASIC composite
    controller (construction-only; never stepped).

    Returns the env; the caller drives env.sim's raw MjModel/MjData with the
    custom mj_step loop and reads env._check_success()/reward()/get_ep_meta().
    """
    install_robocasa_patches()  # also triggers `import robocasa` -> env registration
    import robocasa  # noqa: F401  (ensure Kitchen envs are registered)
    import robosuite

    env_kwargs = dict(
        env_name=env_name,
        robots="G1Dex3",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        render_camera=None,   # base.py skips camera_name2id(...) when None
        ignore_done=True,
    )
    for k, v in (("layout_ids", layout_ids), ("style_ids", style_ids), ("seed", seed)):
        if v is not None:
            env_kwargs[k] = v
    env_kwargs.update(kwargs)
    return robosuite.make(**env_kwargs)
