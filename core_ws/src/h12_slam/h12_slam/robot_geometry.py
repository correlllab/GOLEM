"""
URDF collision geometry as per-link bounding boxes, plus small TF/math helpers.

Every collision element (mesh, box, cylinder, sphere) is reduced to its
axis-aligned bounding box in its link's frame. The boxes are conservative:
a point inside the real geometry is always inside its box.
"""

import os
import struct
import xml.etree.ElementTree as ET

import numpy as np


def rpy_to_matrix(roll, pitch, yaw):
    """URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def quat_to_matrix(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0.0 else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def matrix_to_quat(r):
    """Return (x, y, z, w) for a proper rotation matrix."""
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    return q[0], q[1], q[2], (r[k, j] - r[j, k]) / s


def transform_to_rt(transform):
    """geometry_msgs/Transform -> (3x3 rotation, translation)."""
    q = transform.rotation
    t = transform.translation
    return quat_to_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])


def rotation_aligning(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-9:
        # Antiparallel: rotate pi about any axis perpendicular to a.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def load_stl_vertices(path):
    """Vertices of a binary or ASCII STL as an (N, 3) array."""
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) >= 84:
        (count,) = struct.unpack_from('<I', data, 80)
        if 84 + 50 * count == len(data):
            tri = np.frombuffer(data, dtype=np.dtype([
                ('normal', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')]),
                count=count, offset=84)
            return tri['v'].reshape(-1, 3).astype(np.float64)
    verts = [line.split()[1:4] for line in data.decode('ascii', 'ignore').splitlines()
             if line.strip().startswith('vertex')]
    return np.array(verts, dtype=np.float64).reshape(-1, 3)


def resolve_mesh_path(filename):
    if filename.startswith('package://'):
        from ament_index_python.packages import get_package_share_directory
        pkg, _, rel = filename[len('package://'):].partition('/')
        return os.path.join(get_package_share_directory(pkg), rel)
    if filename.startswith('file://'):
        return filename[len('file://'):]
    return filename


def _parse_floats(text, default):
    return np.array([float(v) for v in text.split()]) if text else np.array(default)


def _geometry_points(geom):
    """Points whose bounding box bounds the geometry, in the geometry's own frame."""
    if geom.tag == 'mesh':
        scale = _parse_floats(geom.get('scale'), [1.0, 1.0, 1.0])
        return load_stl_vertices(resolve_mesh_path(geom.get('filename'))) * scale
    if geom.tag == 'box':
        half = 0.5 * _parse_floats(geom.get('size'), [0.0, 0.0, 0.0])
        return np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                         for sz in (-1, 1)]) * half
    if geom.tag == 'cylinder':
        r, h = float(geom.get('radius')), 0.5 * float(geom.get('length'))
        return np.array([[-r, -r, -h], [r, r, h]])
    if geom.tag == 'sphere':
        r = float(geom.get('radius'))
        return np.array([[-r, -r, -r], [r, r, r]])
    raise ValueError(f'unsupported collision geometry <{geom.tag}>')


def link_collision_points(urdf_xml):
    """{link name: (N, 3) points in the link frame bounding its collision geometry}."""
    root = ET.fromstring(urdf_xml)
    links = {}
    for link in root.findall('link'):
        chunks = []
        for col in link.findall('collision'):
            geom_el = col.find('geometry')
            if geom_el is None or len(geom_el) == 0:
                continue
            pts = _geometry_points(geom_el[0])
            origin = col.find('origin')
            xyz = _parse_floats(origin.get('xyz') if origin is not None else None,
                                [0.0, 0.0, 0.0])
            rpy = _parse_floats(origin.get('rpy') if origin is not None else None,
                                [0.0, 0.0, 0.0])
            chunks.append(pts @ rpy_to_matrix(*rpy).T + xyz)
        if chunks:
            links[link.get('name')] = np.vstack(chunks)
    return links


def link_boxes(urdf_xml, padding):
    """{link name: (lo, hi)} padded axis-aligned boxes in each link's frame."""
    return {name: (pts.min(axis=0) - padding, pts.max(axis=0) + padding)
            for name, pts in link_collision_points(urdf_xml).items()}


def points_in_box(points, rot, trans, lo, hi):
    """Mask of points (in frame F) inside box [lo, hi] of a link posed at (rot, trans) in F."""
    local = (points - trans) @ rot
    return np.all((local >= lo) & (local <= hi), axis=1)
