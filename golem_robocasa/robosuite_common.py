"""Place a legged robot by its pelvis free joint in a robosuite env and build its RoboCasa task env, for both robots.

h1_2_robosuite.py and g1/g1_robosuite.py register their robot models and keep
their own collision-free spawn searches; both build on these: writing the
pelvis free joint, fitting the floor clearance, the direction behind the robot,
its penetrating contacts with the environment, RoboCasa's placement patch and
the task env. Only numpy and mujoco are imported at module level (robocasa and
robosuite inside the functions that need them), so the placement helpers run
without robosuite.
"""
import numpy as np

FREE_JOINT_NAME = "floating_base_joint"
# The name prefixes of the robot's bodies and of its grippers' in a robosuite env.
ROBOT_PREFIXES = ("robot0_", "gripper0_")

# The robot model classes whose RoboCasa placement patch is installed.
_PATCHED = set()


def euler_to_wxyz(euler):
    """Return the wxyz quaternion of a RoboCasa base orientation or a URDF rpy.

    Accepts euler [roll, pitch, yaw] (rad, fixed axes as URDF rpy; RoboCasa's
    init_robot_base_ori), a yaw scalar, a wxyz quaternion, or None (identity).
    """
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


def _free_joint_qpos(env):
    """Return the qpos address of the robot's pelvis free joint in env."""
    jname = f"{env.robots[0].robot_model.naming_prefix}{FREE_JOINT_NAME}"
    addr = env.sim.model.get_joint_qpos_addr(jname)
    return addr[0] if isinstance(addr, tuple) else addr


def place_robot_freejoint(env, pos, quat_wxyz):
    """Write the pelvis free joint's pose (pos m, quat_wxyz, world) and run sim.forward()."""
    start = _free_joint_qpos(env)
    env.sim.data.qpos[start : start + 3] = np.asarray(pos, dtype=float)
    env.sim.data.qpos[start + 3 : start + 7] = np.asarray(quat_wxyz, dtype=float)
    env.sim.forward()


def lowest_robot_point_z(env, colliding_only=False, prefixes=ROBOT_PREFIXES):
    """Return the lowest world z (m) of the robot's geoms, from each geom's AABB.

    The bounding sphere (geom_rbound) is far too conservative for flat feet.
    colliding_only skips the visual geoms (contype and conaffinity 0).
    """
    import mujoco

    m = env.sim.model._model
    d = env.sim.data._data
    corners = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
    )
    best = float("inf")
    for i in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
        if not bn.startswith(prefixes):
            continue
        if colliding_only and not (m.geom_contype[i] or m.geom_conaffinity[i]):
            continue
        aabb = m.geom_aabb[i]
        world = (corners * aabb[3:] + aabb[:3]) @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i]
        best = min(best, float(world[:, 2].min()))
    return best


def place_robot_clear(env, pos, quat_wxyz, clearance=0.02, colliding_only=False):
    """Place the pelvis at pos, quat_wxyz, then shift it in z so the robot clears the floor.

    The robot's lowest point (lowest_robot_point_z, with colliding_only) ends
    clearance (m) above z = 0.
    """
    place_robot_freejoint(env, pos, quat_wxyz)
    low = lowest_robot_point_z(env, colliding_only)
    if low != float("inf"):
        env.sim.data.qpos[_free_joint_qpos(env) + 2] += clearance - low
        env.sim.forward()


def backward_dir(base_quat):
    """Return the level world unit vector behind a robot at base_quat (wxyz), which faces its +x.

    Falls back to world -x when the robot's x axis is vertical.
    """
    import mujoco

    fwd = np.zeros(3)
    mujoco.mju_rotVecQuat(fwd, np.array([1.0, 0.0, 0.0]), np.asarray(base_quat, dtype=float))
    back = -fwd
    back[2] = 0.0
    n = np.linalg.norm(back)
    return back / n if n > 1e-9 else np.array([-1.0, 0.0, 0.0])


def robot_env_contacts(model, data, prefixes=ROBOT_PREFIXES):
    """Return the current contacts in which a robot geom penetrates (dist < 0) an environment geom.

    Self-contacts are skipped: moving the base cannot fix them. The floor never
    shows, as place_robot_clear keeps the robot above it.
    """
    import mujoco

    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist >= 0:
            continue
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
        if b1.startswith(prefixes) == b2.startswith(prefixes):
            continue
        out.append(c)
    return out


def install_robocasa_patches(model_class, pos_offset):
    """Patch RoboCasa's robot placement to write model_class's pelvis free joint, once per class and process.

    Kitchen._reset_internal places the robot through env_utils.set_robot_base
    (a new episode; it returns the position the kitchen stores) or
    set_robot_to_position (an episode from ep_meta). Both write the
    mobilebase0_joint_mobile_* joints a legged robot lacks and would raise, so
    for an env of model_class neither original runs; other robots go on to the
    placement in place before (RoboCasa's, or another robot's patch).
    set_robot_base calls set_robot_to_position internally, so it is replaced
    whole rather than wrapped. pos_offset is RoboCasa's placement offset for
    the robot (m): x and y along the fixture from its anchor, then the pelvis z.
    """
    name = model_class.__name__
    if name in _PATCHED:
        return
    import warnings

    try:
        from robocasa.utils import env_utils as EU
    except Exception as e:  # pragma: no cover - container only
        warnings.warn(f"[robosuite_common] robocasa not importable: {e}")
        return

    offsets = getattr(EU, "_ROBOT_POS_OFFSETS", None)
    if isinstance(offsets, dict):
        offsets.setdefault(name, list(pos_offset))

    def is_robot(env):
        """Return whether env's robot is a model_class."""
        try:
            return isinstance(env.robots[0].robot_model, model_class)
        except Exception:
            return False

    orig_set_base = getattr(EU, "set_robot_base", None)
    orig_set_pos = getattr(EU, "set_robot_to_position", None)

    if orig_set_base is not None:
        def set_robot_base(*args, **kwargs):
            """Place the robot at anchor_pos/anchor_ori and return the position; else call the placement before."""
            env = kwargs.get("env", args[0] if args else None)
            if env is not None and is_robot(env):
                anchor_pos = kwargs.get("anchor_pos", args[1] if len(args) > 1 else None)
                anchor_ori = kwargs.get("anchor_ori", args[2] if len(args) > 2 else None)
                pos = np.array(anchor_pos, dtype=float) if anchor_pos is not None else np.zeros(3)
                env.init_robot_base_pos = pos
                env.init_robot_base_ori = (
                    np.array(anchor_ori, dtype=float) if anchor_ori is not None else np.zeros(3)
                )
                place_robot_freejoint(env, pos, euler_to_wxyz(anchor_ori))
                return pos
            return orig_set_base(*args, **kwargs)

        EU.set_robot_base = set_robot_base

    if orig_set_pos is not None:
        def set_robot_to_position(*args, **kwargs):
            """Place the robot at global_pos; else call the placement before."""
            env = kwargs.get("env", args[0] if args else None)
            global_pos = kwargs.get("global_pos", args[1] if len(args) > 1 else None)
            if env is not None and is_robot(env) and global_pos is not None:
                ori = getattr(env, "init_robot_base_ori", None)
                place_robot_freejoint(env, np.array(global_pos, dtype=float), euler_to_wxyz(ori))
                return
            return orig_set_pos(*args, **kwargs)

        EU.set_robot_to_position = set_robot_to_position

    if orig_set_base is None and orig_set_pos is None:
        warnings.warn(
            "[robosuite_common] no RoboCasa placement entry point found to patch; "
            "call place_robot_freejoint(env, pos, quat) from the harness after reset."
        )
    _PATCHED.add(name)


def make_kitchen_env(model_class, pos_offset, env_name, layout_ids=None, style_ids=None, seed=None,
                     **kwargs):
    """Return the task env env_name around model_class, without robosuite's renderer or camera observations.

    Installs the placement patch for model_class with pos_offset
    (install_robocasa_patches) first. robocasa's env_utils.create_env is not
    used: it forces has_offscreen_renderer, use_camera_obs and agentview
    cameras, and robosuite's reset() then fails on the missing cameras. The sim
    draws with mujoco.viewer and reads env state directly. Importing robocasa
    registers the kitchen tasks, so robosuite.make resolves them by name.
    """
    install_robocasa_patches(model_class, pos_offset)
    import robocasa  # noqa: F401  # registers the kitchen tasks
    import robosuite

    env_kwargs = dict(
        env_name=env_name,
        robots=model_class.__name__,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=None,   # robosuite looks up no camera for None
        ignore_done=True,
    )
    for k, v in (("layout_ids", layout_ids), ("style_ids", style_ids), ("seed", seed)):
        if v is not None:
            env_kwargs[k] = v
    env_kwargs.update(kwargs)
    return robosuite.make(**env_kwargs)
