'''Transforms that hang FAST_LIO's frames and the lidar off the G1 URDF.

The MID-360 is mounted upside down on the G1's torso (URDF mid360_joint on
torso_link, rpy pi 0.0511 0). FAST_LIO estimates the pose of the sensor's IMU,
'body', in 'camera_init', which is where the IMU was when FAST_LIO started, so
both frames are upside down. Used directly as the odom frame (as on the H1),
camera_init would put the floor above the robot and mirror the map. The
navigation launch instead hangs FAST_LIO off a gravity-aligned odom frame:

    map -(slam_toolbox)-> odom -(static)-> camera_init -(FAST_LIO)-> body
        -(body_to_pelvis)-> pelvis -(robot_state_publisher)-> ... torso_link
        -> mid360_link -(static)-> livox_frame

- mid360_link -> livox_frame is the frame the lidar data is expressed in:
  identity for the sensor's raw frame.
- body -> pelvis is the inverse of pelvis -> the IMU: pelvis -> livox_frame
  through the URDF at the measured waist angles, then FAST_LIO's lidar-to-IMU
  extrinsic, read from the FAST_LIO config so the two cannot disagree. The
  body_to_pelvis node publishes it from TF as the waist moves; a fixed one
  would count any waist angle twice in the lidar's pose.
- odom -> camera_init is the IMU's pose when FAST_LIO starts, with the robot
  standing, the torso level (as with the pelvis level and the waist at 0), the
  pelvis at pelvis_height above the floor and odom's x axis along the
  pelvis's. The floor is then z = 0 in odom.
'''

import xml.etree.ElementTree as ET
from typing import NamedTuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

BASE_LINK = 'pelvis'
LIDAR_LINK = 'mid360_link'
LIVOX_FRAME = 'livox_frame'


def transform(xyz=(0.0, 0.0, 0.0), rotation=None):
    '''4x4 homogeneous transform'''
    t = np.eye(4)
    t[:3, 3] = xyz
    if rotation is not None:
        t[:3, :3] = rotation
    return t


def invert(t):
    inv = np.eye(4)
    inv[:3, :3] = t[:3, :3].T
    inv[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return inv


def _floats(element, key, default):
    text = element.get(key) if element is not None else None
    return [float(v) for v in text.split()] if text else list(default)


class Joint(NamedTuple):
    name: str
    parent: str
    type: str
    origin: np.ndarray  # 4x4, parent -> joint frame
    axis: np.ndarray  # unit vector in the joint frame


def load_joints(urdf_file):
    '''{child link: Joint} from a URDF'''
    joints = {}
    for joint in ET.parse(urdf_file).getroot().findall('joint'):
        origin = joint.find('origin')
        # URDF rpy is roll, pitch, yaw about the fixed axes: R = Rz Ry Rx
        rotation = Rotation.from_euler('xyz', _floats(origin, 'rpy', (0, 0, 0))).as_matrix()
        axis = np.array(_floats(joint.find('axis'), 'xyz', (1, 0, 0)))
        joints[joint.find('child').get('link')] = Joint(
            joint.get('name'),
            joint.find('parent').get('link'),
            joint.get('type'),
            transform(_floats(origin, 'xyz', (0, 0, 0)), rotation),
            axis / np.linalg.norm(axis),
        )
    return joints


def link_pose(urdf_file, root, tip, positions=None):
    '''T_root_tip through the URDF; joints missing from positions (by name) are at 0'''
    joints = load_joints(urdf_file)
    positions = positions or {}
    t = np.eye(4)
    link = tip
    while link != root:
        if link not in joints:
            raise ValueError(f'{tip} is not below {root} in {urdf_file}')
        joint = joints[link]
        q = positions.get(joint.name, 0.0)
        if joint.type in ('revolute', 'continuous'):
            motion = transform(rotation=Rotation.from_rotvec(joint.axis * q).as_matrix())
        elif joint.type == 'prismatic':
            motion = transform(joint.axis * q)
        else:
            motion = np.eye(4)
        t = joint.origin @ motion @ t
        link = joint.parent
    return t


def fastlio_extrinsic(fastlio_config):
    '''T_body_lidar from a FAST_LIO config: p_imu = extrinsic_R p_lidar + extrinsic_T'''
    with open(fastlio_config) as f:
        params = yaml.safe_load(f)['/**']['ros__parameters']['mapping']
    rotation = np.array(params['extrinsic_R'], dtype=float).reshape(3, 3)
    return transform(np.array(params['extrinsic_T'], dtype=float), rotation)


def pelvis_to_body(urdf_file, body_from_lidar, lidar_frame=np.eye(4), positions=None):
    '''T_pelvis_body, the IMU FAST_LIO tracks; joints missing from positions are at 0.

    lidar_frame is T_mid360_link_livox_frame, the frame the points arrive in.
    '''
    pelvis_lidar = link_pose(urdf_file, BASE_LINK, LIDAR_LINK, positions) @ lidar_frame
    return pelvis_lidar @ invert(body_from_lidar)


def body_to_pelvis(pelvis_lidar, body_from_lidar):
    '''T_body_pelvis from T_pelvis_livox_frame and FAST_LIO's T_body_lidar'''
    return body_from_lidar @ invert(pelvis_lidar)


def fastlio_transforms(urdf_file, fastlio_config, pelvis_height, lidar_frame=np.eye(4)):
    '''(T_odom_camera_init, T_body_pelvis at the start pose, the waist at 0).

    The launch file publishes the first; body_to_pelvis publishes the second
    from the measured waist.
    '''
    pelvis_body = pelvis_to_body(urdf_file, fastlio_extrinsic(fastlio_config), lidar_frame)
    odom_pelvis = transform((0.0, 0.0, pelvis_height))
    return odom_pelvis @ pelvis_body, invert(pelvis_body)


def static_transform_arguments(t, parent, child):
    '''tf2_ros static_transform_publisher arguments that publish T_parent_child'''
    values = list(t[:3, 3]) + list(Rotation.from_matrix(t[:3, :3]).as_quat())  # x y z w
    args = []
    for name, value in zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), values):
        args += [f'--{name}', f'{float(value):.12g}']
    return args + ['--frame-id', parent, '--child-frame-id', child]
