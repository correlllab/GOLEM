"""Offline policy pose loading and fixed-orientation sole placement (no ROS/DDS)."""

from pathlib import Path
import math

import yaml


POLICY_CHOICES = ("fame_finetune", "almi_manip_stand")


def load_leg_pose(policy, policies_dir=None):
    """Read only the pre-engagement leg pose, in Unitree motor order."""
    if policy not in POLICY_CHOICES:
        raise ValueError(f"unknown initialization policy {policy!r}; choose {POLICY_CHOICES}")
    root = (Path(policies_dir) if policies_dir is not None else
            Path(__file__).resolve().parents[1] / "core_ws/src/h12_lowerbody_rl/policies")
    path = root / policy / f"{policy}.yaml"
    key, count = (("lower_start_pos", 12) if policy == "fame_finetune"
                  else ("default_angles", 21))
    try:
        with path.open() as stream:
            config = yaml.safe_load(stream)
        values = config[key]
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"{key} must contain exactly {count} angles")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) for v in values):
            raise ValueError(f"{key} must contain finite numeric angles")
        return values[:12]
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid initialization policy {policy!r} at {path}: {exc}") from exc


def foot_floor_geometry(model, prefix="robot0_"):
    """Resolve collision-only ankle descendants and named native floor planes."""
    import mujoco

    feet = {}
    for side in ("left", "right"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                f"{prefix}{side}_ankle_roll_link")
        if body < 0:
            raise ValueError(f"missing {side} ankle-roll body")
        descendants = {body}
        for b in range(body + 1, model.nbody):
            if int(model.body_parentid[b]) in descendants:
                descendants.add(b)
        feet[side] = [g for g in range(model.ngeom)
                      if int(model.geom_bodyid[g]) in descendants and
                      (model.geom_contype[g] or model.geom_conaffinity[g])]
        if not feet[side]:
            raise ValueError(f"no {side} foot collision geometry")
    floors = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        if ((name.startswith("floor") or body.startswith("floor")) and
                model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE and
                (model.geom_contype[g] or model.geom_conaffinity[g])):
            floors.append(g)
    if not floors:
        raise ValueError("policy initialization requires a named floor* plane")
    for side, geoms in feet.items():
        if not _floor_pairs(model, geoms, floors):
            raise ValueError(f"no collision-compatible floor plane for {side} foot")
    return feet, floors


def _floor_pairs(model, geoms, floors):
    return [(g, f) for g in geoms for f in floors
            if ((model.geom_contype[g] & model.geom_conaffinity[f]) or
                (model.geom_contype[f] & model.geom_conaffinity[g]))]


def _plane_gap(model, data, geom, floor):
    """Exact support distance, not a rotated AABB or bounding sphere."""
    import mujoco
    import numpy as np

    normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
    if normal[2] <= 1e-6:
        raise ValueError("floor plane must have an upward world-Z normal component")
    rotation = data.geom_xmat[geom].reshape(3, 3)
    local_normal = rotation.T @ normal
    size = model.geom_size[geom]
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        support = float(np.min(model.mesh_vert[start:start + count] @ local_normal))
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        support = -float(np.abs(local_normal) @ size)
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        support = -float(size[0])
    elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius = (np.linalg.norm(local_normal[:2]) if kind == mujoco.mjtGeom.mjGEOM_CYLINDER else 1.0)
        support = -float(size[0] * radius + size[1] * abs(local_normal[2]))
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        support = -float(np.linalg.norm(size * local_normal))
    else:
        raise ValueError(f"unsupported foot collision geom type {kind}")
    return float(normal @ (data.geom_xpos[geom] - data.geom_xpos[floor]) + support), float(normal[2])


def sole_gaps(model, data, feet, floors):
    """Per-foot minimum signed plane gap in metres, with current kinematics."""
    return {side: min(_plane_gap(model, data, g, f)[0] for g, f in _floor_pairs(model, geoms, floors))
            for side, geoms in feet.items()}


def make_foot_placer(env, place_freejoint):
    """Placement callback for the existing horizontal obstacle search.

    Only pelvis Z changes after the task pose is applied. A 10 nm separation
    prevents roundoff contacts from looking like obstacles behind the robot.
    Unequal sole heights remain unequal; no joint or orientation fitting occurs.
    """
    import mujoco

    model, data = env.sim.model._model, env.sim.data._data
    prefix = env.robots[0].robot_model.naming_prefix
    feet, floors = foot_floor_geometry(model, prefix)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}floating_base_joint")
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("missing robot pelvis free joint")
    address = int(model.jnt_qposadr[joint])

    def place(target_env, pos, quat, clearance=0.0):
        if clearance != 0.0:
            raise ValueError("policy sole placement requires zero floor clearance")
        place_freejoint(target_env, pos, quat)
        shifts = []
        for geoms in feet.values():
            for g, f in _floor_pairs(model, geoms, floors):
                gap, nz = _plane_gap(model, data, g, f)
                shifts.append((1e-8 - gap) / nz)
        data.qpos[address + 2] += max(shifts)
        mujoco.mj_forward(model, data)

    return place, feet, floors
