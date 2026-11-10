"""
Provide the geometry and decoding helpers the skills share.

Decode a compressedDepth image, back-project a pixel mask to camera-frame
points, convert between poses, transforms, matrices and quaternions, and pull
JSON out of a Gemini reply. Depends on numpy, OpenCV, geometry_msgs and
model_server's rotations (mat_to_quat, which graspgen_server shares).
"""

import json
import re
import math

import numpy as np
import cv2

from geometry_msgs.msg import Pose
from model_server.rotations import mat_to_quat


def decode_compressed_depth_image(msg) -> np.ndarray:
    """
    Decode a ROS 2 compressedDepth image ('16UC1; compressedDepth') to uint16 mm.

    Raises ValueError on another format or an undecodable payload, TypeError
    when the decoded image is not uint16.
    """
    if not msg.format.lower().endswith("compresseddepth"):
        raise ValueError(f"Unsupported depth format: {msg.format}")
    # bytes of the compressedDepth config header that precedes the PNG payload
    header_size = 12
    if len(msg.data) <= header_size:
        raise ValueError("compressedDepth data too short to contain a header")
    compressed_data = bytes(msg.data[header_size:])
    np_arr = np.frombuffer(compressed_data, dtype=np.uint8)
    depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError("cv2.imdecode failed on compressed depth image")
    if depth_image.dtype != np.uint16:
        raise TypeError(f"Expected uint16 depth, got {depth_image.dtype}")
    return depth_image


def deproject_mask(mask, depth_m, fx, fy, cx, cy, min_d=0.1, max_d=3.0):
    """
    Back-project the True pixels of `mask` to an (N, 3) float32 array, camera optical frame.

    `mask` is bool HxW and `depth_m` is HxW depth [m] on the same pixel grid;
    pixels with depth outside (`min_d`, `max_d`) are dropped. Pinhole model
    with intrinsics fx, fy, cx, cy [px].
    """
    valid = mask & (depth_m > min_d) & (depth_m < max_d)
    # vs = rows (image y), us = columns (image x)
    vs, us = np.nonzero(valid)
    z = depth_m[valid].astype(np.float32)
    x = ((us - cx) * z / fx).astype(np.float32)
    y = ((vs - cy) * z / fy).astype(np.float32)
    return np.stack([x, y, z], axis=-1)


def transform_points(pts, T):
    """Apply the 4x4 homogeneous transform `T` to an (N, 3) point array."""
    return (pts @ T[:3, :3].T) + T[:3, 3]


def transform_to_matrix(tf_msg):
    """
    Convert a geometry_msgs/Transform into a 4x4 numpy array.

    For lookup_transform(target, source) the result maps source-frame
    coordinates into the target frame.
    """
    tx, ty, tz = tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z
    qx, qy, qz, qw = (tf_msg.rotation.x, tf_msg.rotation.y,
                      tf_msg.rotation.z, tf_msg.rotation.w)
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T


def quat_to_rot(qx, qy, qz, qw):
    """Convert a unit quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ])


def quat_geodesic(q1, q2):
    """Return the angle [rad, 0..pi] between two (x, y, z, w) quaternions, sign-invariant."""
    dot = min(1.0, abs(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]))
    return 2.0 * math.acos(dot)


def pose_to_matrix(pose):
    """Convert a geometry_msgs/Pose to a 4x4 homogeneous matrix."""
    p, q = pose.position, pose.orientation
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def matrix_to_pose(T):
    """Convert a 4x4 homogeneous matrix to a geometry_msgs/Pose (inverse of pose_to_matrix)."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    qx, qy, qz, qw = mat_to_quat(T[:3, :3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
        float(qx), float(qy), float(qz), float(qw))
    return pose


# a "box_2d": [y1, x1, y2, x2] entry, found in replies that are not valid JSON
_BOX_2D_RE = re.compile(
    r'"box_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
    r'\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')


def extract_json(text):
    """
    Return the JSON value in a Gemini text reply, best-effort, or None.

    Strips a ```json fence, else tries the outermost [...] then {...} span.
    When nothing parses but the text carries "box_2d": [...] arrays (Gemini
    sometimes drops a key, e.g. echoes the label as a bare string), returns
    them as [{"box_2d": [...]}, ...].
    """
    if not text:
        return None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            text = "\n".join(lines[i + 1:]).split("```")[0]
            break
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        a, b = text.find(opener), text.rfind(closer)
        if 0 <= a < b:
            try:
                return json.loads(text[a:b + 1])
            except (json.JSONDecodeError, ValueError):
                continue
    boxes = [{'box_2d': [float(v) for v in m]} for m in _BOX_2D_RE.findall(text)]
    return boxes or None
