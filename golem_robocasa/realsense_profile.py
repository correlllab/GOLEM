"""The real robot's RealSense cameras, as the sim should reproduce them.

realsense_profile.yaml holds the cameras as the real robot publishes them
(frame names, TF tree, intrinsics). This module has no ROS imports, so
h1_2_robosuite can use it while building the robot MJCF:

  place_profiled_cameras()  puts a profiled camera's MJCF colour camera at the
                            real colour optical pose and adds a depth camera at
                            the real depth optical pose, both with the real
                            pinhole intrinsics;
  camera_specs()            per camera: render sizes, the camera_info the bridge
                            publishes, frame names and the depth -> colour
                            transform, at a render scale;
  tf_edges()                the static TF tree (robot-side mounts + driver);
  colored_cloud()           the driver's depth/color/points from a depth and a
                            colour image.

Frames: an optical frame is +x right, +y down, +z forward (REP-103). A MuJoCo
camera frame is +x right, +y up, looking along -z, so R_mujoco = R_optical F
with F = diag(1, -1, -1). Pixel coordinates follow camera_info: integer
coordinates are pixel centres.
"""
import math
import os
from collections import deque
from dataclasses import dataclass

import numpy as np
import yaml

PROFILE_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realsense_profile.yaml")
DEPTH_CAMERA_SUFFIX = "_depth"

_F = np.diag([1.0, -1.0, -1.0])


def load_profile(path=PROFILE_YAML):
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------- transforms
def _quat_xyzw_to_R(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _R_to_quat_wxyz(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _edges(cam):
    """[(parent, child, xyz, quat_xyzw, driver)] of a profiled camera."""
    out = []
    for e in cam["tf_static"]:
        out.append((e[0], e[1], [float(v) for v in e[2]], [float(v) for v in e[3]],
                    len(e) > 4 and e[4] == "driver"))
    return out


def frame_transform(cam, a, b):
    """4x4 pose of frame `b` in frame `a` (p_a = T p_b) over the camera's TF tree."""
    graph = {}
    for parent, child, xyz, q, _ in _edges(cam):
        T = _T(_quat_xyzw_to_R(q), xyz)
        graph.setdefault(parent, []).append((child, T))
        graph.setdefault(child, []).append((parent, np.linalg.inv(T)))
    seen, queue = {a: np.eye(4)}, deque([a])
    while queue:
        f = queue.popleft()
        for g, T in graph.get(f, ()):
            if g not in seen:
                seen[g] = seen[f] @ T
                queue.append(g)
    if b not in seen:
        raise KeyError(f"no TF path {a} -> {b} in the camera profile")
    return seen[b]


def tf_edges(profile, names):
    """All TF edges of the named cameras: [(parent, child, xyz, quat_xyzw, driver)]."""
    return [e for n in names for e in _edges(profile["cameras"][n])]


# ----------------------------------------------------------------- intrinsics
@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def scaled(self, width, height):
        """The same projection sampled at width x height (pixel-centre convention)."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy,
                          (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5)


def _profile_intrinsics(stream):
    fx, fy, cx, cy = (float(v) for v in stream["k"])
    return Intrinsics(int(stream["width"]), int(stream["height"]), fx, fy, cx, cy)


def fovy_intrinsics(width, height, fovy_deg):
    """Square-pixel pinhole of a MuJoCo fovy camera rendered at width x height."""
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return Intrinsics(width, height, f, f, width / 2.0 - 0.5, height / 2.0 - 0.5)


# ------------------------------------------------------------- MJCF placement
def _camera_pose(el):
    """Local (R_mujoco, pos) of an MJCF <camera> given by pos + xyaxes or quat."""
    pos = np.array([float(v) for v in el.get("pos", "0 0 0").split()])
    if el.get("xyaxes") is not None:
        a = np.array([float(v) for v in el.get("xyaxes").split()])
        x = a[:3] / np.linalg.norm(a[:3])
        y = a[3:] - x * np.dot(x, a[3:])
        y /= np.linalg.norm(y)
        R = np.column_stack([x, y, np.cross(x, y)])
    elif el.get("quat") is not None:
        w, x, y, z = (float(v) for v in el.get("quat").split())
        R = _quat_xyzw_to_R([x, y, z, w])
    else:
        raise ValueError(f"camera {el.get('name')!r}: expected pos + xyaxes or quat")
    return R, pos


def _set_camera(el, T_body_optical, intr):
    R = T_body_optical[:3, :3] @ _F
    for key in ("xyaxes", "quat", "euler", "axisangle", "zaxis", "fovy"):
        el.attrib.pop(key, None)
    el.set("pos", " ".join(f"{v:.9g}" for v in T_body_optical[:3, 3]))
    el.set("quat", " ".join(f"{v:.9g}" for v in _R_to_quat_wxyz(R)))
    # MuJoCo pinhole: principalpixel is the principal point's offset from the
    # image centre, +x moving it left and +y moving it down (continuous pixel
    # coordinates, so cx + 0.5).
    el.set("resolution", f"{intr.width} {intr.height}")
    el.set("sensorsize", "1 1")
    el.set("focalpixel", f"{intr.fx:.9g} {intr.fy:.9g}")
    el.set("principalpixel", f"{intr.width / 2.0 - (intr.cx + 0.5):.9g} "
                             f"{(intr.cy + 0.5) - intr.height / 2.0:.9g}")


def place_profiled_cameras(root, cam_name, cam):
    """Move MJCF camera `cam_name` (whose optical frame is the profile's
    anchor_link) to the real colour optical pose with the real colour
    intrinsics, and add `cam_name + DEPTH_CAMERA_SUFFIX` beside it at the depth
    optical pose with the depth intrinsics. `root` is any ElementTree element
    containing the camera; returns the depth camera's name."""
    parent = next((b for b in root.iter() for c in b if c.tag == "camera" and c.get("name") == cam_name), None)
    if parent is None:
        raise KeyError(f"camera {cam_name!r} not in the MJCF")
    color_el = next(c for c in parent if c.tag == "camera" and c.get("name") == cam_name)
    R_mj, pos = _camera_pose(color_el)
    T_body_anchor = _T(R_mj @ _F, pos)
    anchor = cam["anchor_link"]

    depth_el = type(color_el)(color_el.tag, dict(color_el.attrib))
    depth_name = cam_name + DEPTH_CAMERA_SUFFIX
    depth_el.set("name", depth_name)
    _set_camera(color_el, T_body_anchor @ frame_transform(cam, anchor, cam["color_frame"]),
                _profile_intrinsics(cam["color"]))
    _set_camera(depth_el, T_body_anchor @ frame_transform(cam, anchor, cam["depth_frame"]),
                _profile_intrinsics(cam["depth"]))
    parent.insert(list(parent).index(color_el) + 1, depth_el)
    return depth_name


# ------------------------------------------------------------- camera specs
@dataclass(frozen=True)
class CameraSpec:
    ns: str                    # /realsense/<ns>/...
    color_camera: str          # MuJoCo camera rendering colour + aligned depth
    depth_camera: str          # MuJoCo camera rendering depth/image_rect_raw
    color: Intrinsics          # colour / aligned depth: rendered and published
    depth_info: Intrinsics     # depth camera_info as the driver reports it
    depth_image: Intrinsics    # depth/image_rect_raw + point cloud, decimated
    color_frame: str
    depth_frame: str
    T_color_depth: np.ndarray  # p_color_optical = T p_depth_optical


def camera_specs(model, cameras, profile, scale, fallback_size):
    """CameraSpec per (ns, mujoco_color_camera) in `cameras`.

    A camera with profiled intrinsics renders at `scale` x its real resolution
    and uses its "<name>_depth" MJCF camera for depth. One without keeps its
    MJCF fovy at `fallback_size` (w, h), and its depth comes from the same
    camera, as on a D405."""
    import mujoco

    specs = []
    for ns, color_cam in cameras:
        cam = profile["cameras"][ns]
        dec = int(cam.get("decimation", 1))
        if "color" in cam:
            real_c, real_d = _profile_intrinsics(cam["color"]), _profile_intrinsics(cam["depth"])
            color = real_c.scaled(round(real_c.width * scale), round(real_c.height * scale))
            depth_info = real_d.scaled(round(real_d.width * scale), round(real_d.height * scale))
            depth_cam = color_cam + DEPTH_CAMERA_SUFFIX
        else:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, color_cam)
            if cam_id < 0:
                raise KeyError(f"camera {color_cam!r} not in the model")
            color = fovy_intrinsics(fallback_size[0], fallback_size[1], float(model.cam_fovy[cam_id]))
            depth_info, depth_cam = color, color_cam
        depth_image = depth_info.scaled(max(1, depth_info.width // dec), max(1, depth_info.height // dec))
        specs.append(CameraSpec(
            ns=ns, color_camera=color_cam, depth_camera=depth_cam, color=color,
            depth_info=depth_info, depth_image=depth_image,
            color_frame=cam["color_frame"], depth_frame=cam["depth_frame"],
            T_color_depth=frame_transform(cam, cam["color_frame"], cam["depth_frame"]),
        ))
    return specs


# --------------------------------------------------------------- point cloud
CLOUD_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("_pad", "<f4"), ("rgb", "<f4")])


def colored_cloud(depth_m, depth_intr, rgb, color_intr, T_color_depth):
    """The driver's depth/color/points: every valid depth pixel back-projected
    into the depth optical frame, coloured from the colour image (nearest
    pixel) and dropped when it falls outside it. Returns a CLOUD_DTYPE array
    (x y z, 4 pad bytes, rgb packed as float32 0x00RRGGBB at offset 16)."""
    h, w = depth_m.shape
    v, u = np.nonzero(depth_m > 0.0)
    z = depth_m[v, u].astype(np.float64)
    x = (u - depth_intr.cx) / depth_intr.fx * z
    y = (v - depth_intr.cy) / depth_intr.fy * z
    p = np.stack([x, y, z])
    pc = T_color_depth[:3, :3] @ p + T_color_depth[:3, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        uc = np.rint(color_intr.fx * pc[0] / pc[2] + color_intr.cx)
        vc = np.rint(color_intr.fy * pc[1] / pc[2] + color_intr.cy)
    keep = (pc[2] > 0) & (uc >= 0) & (uc < color_intr.width) & (vc >= 0) & (vc < color_intr.height)
    cols = rgb[vc[keep].astype(np.intp), uc[keep].astype(np.intp)].astype(np.uint32)
    out = np.zeros(int(keep.sum()), dtype=CLOUD_DTYPE)
    out["x"], out["y"], out["z"] = x[keep], y[keep], z[keep]
    out["rgb"] = ((cols[:, 0] << 16) | (cols[:, 1] << 8) | cols[:, 2]).view(np.float32)
    return out
