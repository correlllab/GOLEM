"""Give golem_robocasa one call for each MuJoCo API that differs between the versions it runs on.

The x86 robocasa image runs MuJoCo 3.17.0 and the Mac's image 3.3.1 (CLAUDE.md §8), and both
run this code. Each function here behaves the same on both, keyed on mujoco.__version__.
tests/robocasa/test_mj_compat.py checks each one on whichever version is installed.
"""
import mujoco

VERSION = tuple(int(part) for part in mujoco.__version__.split(".")[:3])

# 3.5.0 added a `normal` output to mj_multiRay, between dist and nray, positional with no
# default in the Python binding (MuJoCo's changelog, 3.5.0, breaking change 8).
_MULTI_RAY_NORMAL = VERSION >= (3, 5, 0)

# Up to 3.4.0, mj_multiRay culls each geom by the azimuth and elevation of its bounding box's
# corners, so some rays pass through a geom they hit (mid360.CpuCaster's cast); from 3.5.0 it
# matches mj_ray on every ray of tests/robocasa/test_mid360.py's kitchen (measured on 3.3.1,
# 3.4.0, 3.5.0, 3.6.0, 3.8.0, 3.12.0 and 3.17.0).
MULTI_RAY_CULLS = VERSION < (3, 5, 0)

# From 3.6.0, +principalpixel y moves the rendered image up; 3.5.0 and earlier move it down.
# Unlisted in MuJoCo's changelog; measured by rendering a marker on 3.3.1, 3.3.7, 3.4.0, 3.5.0,
# 3.6.0, 3.7.0, 3.8.0, 3.12.0 and 3.17.0.
_PRINCIPAL_Y_UP = VERSION >= (3, 6, 0)


def multi_ray(model, data, pnt, vec, geomgroup, flg_static, bodyexclude, geomid, dist, nray, cutoff):
    """Cast nray rays from pnt with mj_multiRay, writing each ray's geom id and distance.

    The arguments are mj_multiRay's of MuJoCo 3.3.1: vec (3 * nray, 1) unit directions in the
    world frame; geomid (nray, 1) int32 and dist (nray, 1) m are written, -1 for a ray that hits
    nothing within cutoff (m). The surface normals that newer versions can return are not
    computed. The binding releases the GIL while it casts.
    """
    if _MULTI_RAY_NORMAL:
        mujoco.mj_multiRay(model, data, pnt, vec, geomgroup, flg_static, bodyexclude,
                           geomid, dist, None, nray, cutoff)
    else:
        mujoco.mj_multiRay(model, data, pnt, vec, geomgroup, flg_static, bodyexclude,
                           geomid, dist, nray, cutoff)


def principalpixel(width, height, cx, cy):
    """Return the MJCF camera principalpixel (x, y) that renders a pinhole's principal point at (cx, cy).

    width and height are the image size and (cx, cy) the principal point in camera_info's pixel
    coordinates (integer coordinates are pixel centres, x right, y down). MuJoCo's values are the
    principal point's offset from the image centre in continuous coordinates (so cx + 0.5): +x
    moves the image left on every version, +y moves it down up to 3.5.0 and up from 3.6.0.
    """
    x = width / 2.0 - (cx + 0.5)
    y = (cy + 0.5) - height / 2.0
    return x, (-y if _PRINCIPAL_Y_UP else y)


def joint_type(model, joint_id):
    """Return joint joint_id's type as a Python int, which equals its mjtJoint and finds it as a dict key.

    Under the pybind11 3 bindings (3.17.0; 3.3.1's are pybind11 2), model.jnt_type's numpy
    integers do not find mjtJoint keys of a dict, and `mjtJoint == numpy integer` is False with
    the enum on the left.
    """
    return int(model.jnt_type[joint_id])
