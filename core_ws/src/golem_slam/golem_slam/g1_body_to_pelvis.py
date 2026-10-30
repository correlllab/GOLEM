"""
Publish body -> pelvis for the G1 FAST-LIO pipeline and compute G1 URDF sensor transforms.

    ros2 run golem_slam g1_body_to_pelvis --ros-args -p fastlio_config:=<FAST_LIO yaml>

FAST-LIO tracks its IMU, 'body', which sits in the MID-360 on the torso, so the
pelvis hangs off body through the waist. Each new pelvis -> livox_frame in TF
(robot_state_publisher at the measured waist angles, then the static
mid360_link -> livox_frame) gives one body -> pelvis with the same stamp:
FAST-LIO's lidar-to-IMU extrinsic (from the FAST-LIO config) composed with
livox_frame -> pelvis. The lidar the scan is cut from and the IMU FAST-LIO
tracks then stay one rigid body at any waist angle. A fixed body -> pelvis with
the waist at 0 would count the waist twice in the lidar's pose: once in
FAST-LIO's pose of the IMU, once in robot_state_publisher's torso.
"""

from typing import NamedTuple
import xml.etree.ElementTree as ET

from geometry_msgs.msg import TransformStamped
from golem_slam.robot_geometry import matrix_to_quat, quat_to_matrix, rpy_to_matrix
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
import yaml

BASE_LINK = 'pelvis'
LIDAR_LINK = 'mid360_link'
LIVOX_FRAME = 'livox_frame'


def transform(xyz=(0.0, 0.0, 0.0), rotation=None):
    """Return a 4x4 homogeneous transform."""
    t = np.eye(4)
    t[:3, 3] = xyz
    if rotation is not None:
        t[:3, :3] = rotation
    return t


def invert(t):
    """Invert a 4x4 rigid transform."""
    inv = np.eye(4)
    inv[:3, :3] = t[:3, :3].T
    inv[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return inv


def _rotvec_to_matrix(rotvec):
    """Rodrigues rotation matrix for a rotation vector (axis * angle)."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    ux, uy, uz = axis
    cross = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _floats(element, key, default):
    text = element.get(key) if element is not None else None
    return [float(v) for v in text.split()] if text else list(default)


class Joint(NamedTuple):
    """Parsed URDF joint kinematic parameters."""

    name: str
    parent: str
    type: str
    origin: np.ndarray  # 4x4, parent -> joint frame
    axis: np.ndarray  # unit vector in the joint frame


def load_joints(urdf_file):
    """Return {child link: Joint} from a URDF."""
    joints = {}
    for joint in ET.parse(urdf_file).getroot().findall('joint'):
        origin = joint.find('origin')
        # URDF rpy is roll, pitch, yaw about the fixed axes: R = Rz Ry Rx
        rotation = rpy_to_matrix(*_floats(origin, 'rpy', (0, 0, 0)))
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
    """Return T_root_tip through the URDF; joints missing from positions are at 0."""
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
            motion = transform(rotation=_rotvec_to_matrix(joint.axis * q))
        elif joint.type == 'prismatic':
            motion = transform(joint.axis * q)
        else:
            motion = np.eye(4)
        t = joint.origin @ motion @ t
        link = joint.parent
    return t


def fastlio_extrinsic(fastlio_config):
    """Return T_body_lidar from a FAST-LIO config: p_imu = extrinsic_R p_lidar + extrinsic_T."""
    with open(fastlio_config) as f:
        params = yaml.safe_load(f)['/**']['ros__parameters']['mapping']
    rotation = np.array(params['extrinsic_R'], dtype=float).reshape(3, 3)
    return transform(np.array(params['extrinsic_T'], dtype=float), rotation)


def pelvis_to_body(urdf_file, body_from_lidar, lidar_frame=np.eye(4), positions=None):
    """
    Return T_pelvis_body, the IMU FAST-LIO tracks; missing joints default to 0.

    lidar_frame is T_mid360_link_livox_frame, the frame the points arrive in.
    """
    pelvis_lidar = link_pose(urdf_file, BASE_LINK, LIDAR_LINK, positions) @ lidar_frame
    return pelvis_lidar @ invert(body_from_lidar)


def body_to_pelvis(pelvis_lidar, body_from_lidar):
    """Return T_body_pelvis from T_pelvis_livox_frame and FAST-LIO's T_body_lidar."""
    return body_from_lidar @ invert(pelvis_lidar)


def fastlio_transforms(urdf_file, fastlio_config, pelvis_height, lidar_frame=np.eye(4)):
    """Return (T_odom_camera_init, T_body_pelvis at the start pose, the waist at 0)."""
    pelvis_body = pelvis_to_body(urdf_file, fastlio_extrinsic(fastlio_config), lidar_frame)
    odom_pelvis = transform((0.0, 0.0, pelvis_height))
    return odom_pelvis @ pelvis_body, invert(pelvis_body)


def static_transform_arguments(t, parent, child):
    """Return tf2_ros static_transform_publisher arguments that publish T_parent_child."""
    values = list(t[:3, 3]) + list(matrix_to_quat(t[:3, :3]))  # x y z qx qy qz qw
    args = []
    for name, value in zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), values):
        args += [f'--{name}', f'{float(value):.12g}']
    return args + ['--frame-id', parent, '--child-frame-id', child]


def to_matrix(tf_msg):
    """Return the 4x4 matrix of a geometry_msgs/Transform."""
    t, q = tf_msg.translation, tf_msg.rotation
    return transform((t.x, t.y, t.z), quat_to_matrix(q.x, q.y, q.z, q.w))


class BodyToPelvis(Node):
    """Publish dynamic body -> pelvis TF from measured waist joint transforms."""

    def __init__(self, **kwargs):
        super().__init__('body_to_pelvis', **kwargs)
        fastlio_config = self.declare_parameter('fastlio_config', '').value
        if not fastlio_config:
            raise ValueError('body_to_pelvis needs the fastlio_config parameter')
        self.body_from_lidar = fastlio_extrinsic(fastlio_config)
        self.body_frame = self.declare_parameter('body_frame', 'body').value
        self.base_frame = self.declare_parameter('base_frame', BASE_LINK).value
        self.lidar_frame = self.declare_parameter('lidar_frame', LIVOX_FRAME).value
        rate = self.declare_parameter('rate', 200.0).value
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = TransformBroadcaster(self)
        self.last_stamp = None
        self.create_timer(1.0 / rate, self._tick)

    def _tick(self):
        try:
            lidar = self.buffer.lookup_transform(self.base_frame, self.lidar_frame, Time())
        except TransformException:
            return  # no joint states yet
        if lidar.header.stamp == self.last_stamp:
            return
        self.last_stamp = lidar.header.stamp
        t = body_to_pelvis(to_matrix(lidar.transform), self.body_from_lidar)
        msg = TransformStamped()
        msg.header.stamp = lidar.header.stamp
        msg.header.frame_id = self.body_frame
        msg.child_frame_id = self.base_frame
        tr, q = msg.transform.translation, msg.transform.rotation
        tr.x, tr.y, tr.z = (float(v) for v in t[:3, 3])
        q.x, q.y, q.z, q.w = (float(v) for v in matrix_to_quat(t[:3, :3]))
        self.broadcaster.sendTransform(msg)


def main(args=None):
    """Run the g1_body_to_pelvis node."""
    rclpy.init(args=args)
    node = BodyToPelvis()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
