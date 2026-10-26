'''Publishes body -> pelvis in fastlio mode: FAST_LIO's IMU to the pelvis, through the waist.

    ros2 run g1_navigation body_to_pelvis --ros-args -p fastlio_config:=<FAST_LIO yaml>

FAST_LIO tracks its IMU, 'body', which sits in the MID-360 on the torso, so the
pelvis hangs off body through the waist. Each new pelvis -> livox_frame in TF
(robot_state_publisher at the measured waist angles, then the static
mid360_link -> livox_frame) gives one body -> pelvis with the same stamp:
FAST_LIO's lidar-to-IMU extrinsic (from the FAST_LIO config) composed with
livox_frame -> pelvis. The lidar the scan is cut from and the IMU FAST_LIO
tracks then stay one rigid body at any waist angle. A fixed body -> pelvis with
the waist at 0 would count the waist twice in the lidar's pose: once in
FAST_LIO's pose of the IMU, once in robot_state_publisher's torso.
'''

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from g1_navigation import frames


def to_matrix(transform):
    '''4x4 matrix of a geometry_msgs/Transform'''
    t, q = transform.translation, transform.rotation
    return frames.transform((t.x, t.y, t.z), Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix())


class BodyToPelvis(Node):

    def __init__(self, **kwargs):
        super().__init__('body_to_pelvis', **kwargs)
        fastlio_config = self.declare_parameter('fastlio_config', '').value
        if not fastlio_config:
            raise ValueError('body_to_pelvis needs the fastlio_config parameter')
        self.body_from_lidar = frames.fastlio_extrinsic(fastlio_config)
        self.body_frame = self.declare_parameter('body_frame', 'body').value
        self.base_frame = self.declare_parameter('base_frame', frames.BASE_LINK).value
        self.lidar_frame = self.declare_parameter('lidar_frame', frames.LIVOX_FRAME).value
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
        t = frames.body_to_pelvis(to_matrix(lidar.transform), self.body_from_lidar)
        msg = TransformStamped()
        msg.header.stamp = lidar.header.stamp
        msg.header.frame_id = self.body_frame
        msg.child_frame_id = self.base_frame
        tr, q = msg.transform.translation, msg.transform.rotation
        tr.x, tr.y, tr.z = (float(v) for v in t[:3, 3])
        q.x, q.y, q.z, q.w = (float(v) for v in Rotation.from_matrix(t[:3, :3]).as_quat())
        self.broadcaster.sendTransform(msg)


def main(args=None):
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
