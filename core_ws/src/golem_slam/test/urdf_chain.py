"""Pose one URDF link in another through the joint tree, for the tests of the models' frames."""
import xml.etree.ElementTree as ET

from golem_slam.robot_geometry import rpy_to_matrix
import numpy as np


def _floats(element, key, default):
    """Return an element's space-separated attribute as an array, or default."""
    text = element.get(key) if element is not None else None
    return np.array([float(v) for v in text.split()]) if text else np.array(default, float)


def _axis_rotation(axis, angle):
    """Return the rotation by angle (rad) about a unit axis (Rodrigues)."""
    ux, uy, uz = axis
    cross = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def link_transform(urdf_xml, parent, child, positions=None):
    """
    Return T_parent_child (4x4) of two links of a URDF.

    positions maps joint names to positions (rad, or m for a prismatic joint); a joint
    missing from it is at 0. The path between the links may run through any joint.
    Raises ValueError when the links do not hang off one root.
    """
    positions = positions or {}
    up = {}
    for joint in ET.fromstring(urdf_xml).findall('joint'):
        origin = joint.find('origin')
        offset = np.eye(4)
        offset[:3, :3] = rpy_to_matrix(*_floats(origin, 'rpy', (0.0, 0.0, 0.0)))
        offset[:3, 3] = _floats(origin, 'xyz', (0.0, 0.0, 0.0))
        axis = _floats(joint.find('axis'), 'xyz', (1.0, 0.0, 0.0))
        up[joint.find('child').get('link')] = (joint, offset, axis / np.linalg.norm(axis))

    def in_root(link):
        """Return (root link, T_root_link)."""
        pose = np.eye(4)
        while link in up:
            joint, offset, axis = up[link]
            q = positions.get(joint.get('name'), 0.0)
            motion = np.eye(4)
            if joint.get('type') in ('revolute', 'continuous'):
                motion[:3, :3] = _axis_rotation(axis, q)
            elif joint.get('type') == 'prismatic':
                motion[:3, 3] = axis * q
            pose = offset @ motion @ pose
            link = joint.find('parent').get('link')
        return link, pose

    root_parent, root_from_parent = in_root(parent)
    root_child, root_from_child = in_root(child)
    if root_parent != root_child:
        raise ValueError(f'{parent!r} and {child!r} are not in one URDF tree')
    return np.linalg.inv(root_from_parent) @ root_from_child
