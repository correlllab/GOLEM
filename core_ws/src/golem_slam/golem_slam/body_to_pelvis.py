"""
Publish FAST-LIO's dynamic body -> pelvis, with the 4x4 transform helpers around it.

    ros2 run golem_slam body_to_pelvis --ros-args -p fastlio_config:=<FAST_LIO yaml>

FAST-LIO tracks its IMU, 'body', which sits in the MID-360 on the torso, so the
pelvis hangs off body through the torso joint (H1-2) or the waist (G1). Each
new pelvis -> livox_frame in TF (robot_state_publisher at the measured joint
angles, then navigation.launch.py's static <lidar link> -> livox_frame) gives
one body -> pelvis with the same stamp: FAST-LIO's lidar-to-IMU extrinsic (from
the FAST-LIO config) composed with livox_frame -> pelvis. The lidar the scan is
cut from and the IMU FAST-LIO tracks then stay one rigid body at any joint
angle. A fixed body -> pelvis with the joints at 0 would count them twice in
the lidar's pose: once in FAST-LIO's pose of the IMU, once in
robot_state_publisher's torso.
"""

from geometry_msgs.msg import TransformStamped
from golem_slam.robot_geometry import matrix_to_quat, quat_to_matrix
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
import yaml

BASE_LINK = 'pelvis'
LIVOX_FRAME = 'livox_frame'


def transform(xyz=(0.0, 0.0, 0.0), rotation=None):
    """Return the 4x4 homogeneous transform of a translation (m) and a 3x3 rotation."""
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


def fastlio_extrinsic(fastlio_config):
    """Return T_body_lidar from a FAST-LIO config: p_imu = extrinsic_R p_lidar + extrinsic_T."""
    with open(fastlio_config) as f:
        params = yaml.safe_load(f)['/**']['ros__parameters']['mapping']
    rotation = np.array(params['extrinsic_R'], dtype=float).reshape(3, 3)
    return transform(np.array(params['extrinsic_T'], dtype=float), rotation)


def body_to_pelvis(pelvis_lidar, body_from_lidar):
    """Return T_body_pelvis from T_pelvis_livox_frame and FAST-LIO's T_body_lidar."""
    return body_from_lidar @ invert(pelvis_lidar)


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
    """
    Publish dynamic body -> pelvis TF from the measured torso or waist joints.

    One transform per new pelvis -> lidar_frame stamp in TF, at that stamp, checked at
    `rate` Hz. Raises ValueError without a `fastlio_config` parameter.

    ROS parameters: fastlio_config (path of the FAST-LIO yaml whose extrinsic it reads),
    body_frame (FAST-LIO's IMU frame), base_frame, lidar_frame (the frame the extrinsic's
    lidar is), rate (Hz).
    """

    def __init__(self, **kwargs):
        """Read the FAST-LIO extrinsic and start polling TF."""
        super().__init__('body_to_pelvis', **kwargs)
        fastlio_config = self.declare_parameter('fastlio_config', '').value
        if not fastlio_config:
            raise ValueError('body_to_pelvis needs the fastlio_config parameter')
        self.body_from_lidar = fastlio_extrinsic(fastlio_config)
        self.body_frame = self.declare_parameter('body_frame', 'body').value
        self.base_frame = self.declare_parameter('base_frame', BASE_LINK).value
        self.lidar_frame = self.declare_parameter('lidar_frame', LIVOX_FRAME).value
        rate = self.declare_parameter('rate', 200.0).value  # Hz
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = TransformBroadcaster(self)
        self.last_stamp = None
        self.create_timer(1.0 / rate, self._tick)

    def _tick(self):
        """Publish body -> pelvis for the latest pelvis -> lidar_frame, once per stamp."""
        try:
            lidar = self.buffer.lookup_transform(self.base_frame, self.lidar_frame, Time())
        except TransformException:
            return  # robot_state_publisher has no joint states yet
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
    """Run the body_to_pelvis node."""
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
