"""Cast MID-360 rays exactly with MuJoCo's mj_ray: the tests' reference for mid360.GpuCaster.

ExactCaster follows GpuCaster's rules (mid360.py): the rays hit the geoms in geomgroup with alpha
not 0 and pass through exclude_body's; a ray whose first hit is a robot geom, or nothing within
mid360.MAX_RANGE, gives NaN. It needs no GPU, so the tests of the scanner and of the robots'
mounts run without one.

It casts one mj_ray per ray, on a full MjData (after mj_forward or mj_kinematics). mj_multiRay
is not the reference: it culls bodies through their BVH and misses geoms, on an MjData that
holds only geom poses and, for some of the G1's shoulder meshes, on a full one too.
"""
import copy

import mujoco
import numpy as np

import mid360


class ExactCaster:
    """Cast rays exactly with mj_ray, one call per ray, on a copy of the model (Mid360Scanner's caster)."""

    batch_s = 0.01             # s of sim time between the scanner's update() calls

    def __init__(self, model, exclude_body, geomgroup, robot_geoms=None):
        """Cast on a copy of model, through the geoms of body id exclude_body (-1: none).

        geomgroup is the 6 flags of the geom groups the rays hit. robot_geoms is a bool mask
        (ngeom,) of geoms whose first hit gives no return; by default every geom of
        exclude_body's root subtree, none with exclude_body -1.
        """
        self.model = copy.copy(model)   # cast() writes the live alpha into it
        self.exclude_body = int(exclude_body)
        self._geomgroup = np.array(geomgroup, dtype=np.uint8).reshape(6, 1)
        if robot_geoms is not None:
            self.robot_geoms = np.asarray(robot_geoms, dtype=bool)
        elif self.exclude_body >= 0:
            root = model.body_rootid[self.exclude_body]
            self.robot_geoms = model.body_rootid[model.geom_bodyid] == root
        else:
            self.robot_geoms = np.zeros(model.ngeom, dtype=bool)

    def close(self):
        """Do nothing: the caster holds no thread or context."""

    def cast(self, origin, rotation, directions, data, geom_alpha=None, mat_alpha=None):
        """Return the distance (m) along each unit direction (n, 3) to its first hit, NaN for none.

        As GpuCaster.cast(): origin (3,) m and rotation (3, 3) sensor -> world are the sensor's
        pose; directions are in its own frame; data is a full MjData; geom_alpha (ngeom,) and
        mat_alpha (nmat,) are the live alpha (None: the last cast's).
        """
        if geom_alpha is not None:
            self.model.geom_rgba[:, 3] = geom_alpha
        if mat_alpha is not None:
            self.model.mat_rgba[:, 3] = mat_alpha
        world = np.asarray(directions, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T
        pnt = np.array(origin, dtype=np.float64).reshape(3, 1)
        geomid = np.empty((1, 1), dtype=np.int32)
        dist = np.full(len(world), np.nan)
        for i, vec in enumerate(world):
            d = mujoco.mj_ray(self.model, data, pnt, vec.reshape(3, 1).copy(), self._geomgroup, 1,
                              self.exclude_body, geomid)
            g = geomid[0, 0]
            if g >= 0 and d <= mid360.MAX_RANGE and not self.robot_geoms[g]:
                dist[i] = d
        return dist
