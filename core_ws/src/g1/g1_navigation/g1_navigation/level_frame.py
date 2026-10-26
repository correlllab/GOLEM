'''Publishes pelvis -> pelvis_level: the pelvis's origin and heading, level with odom;
and pelvis_level -> base_footprint: the same, dropped to odom's floor (z = 0).

    ros2 run g1_navigation level_frame

pointcloud_to_laserscan cuts its height band in its target frame, and the G1's
pelvis tilts: under HOMIE in MuJoCo it leans back 1.8 deg standing, and about
5 deg back with up to 6 deg of roll walking and turning. A band in the pelvis
frame would then take in the floor from about 1.5 m behind the walking robot.
pelvis_level keeps the pelvis's origin and heading but has odom's z axis, so the
band stays horizontal.

base_footprint is slam_toolbox's base_frame and Nav2's robot_base_frame.
slam_toolbox puts its base_frame at z = 0 and level in map at each scan it adds.
With the pelvis there, map -> odom carried the pelvis's height and tilt at the
last scan added, and the scan lay tilted about z = 0 in map, where the global
costmap's obstacle layer dropped the returns below the floor. base_footprint is
level and on the floor in odom, so map -> odom is the identity (README
"slam_toolbox"). Nav2 on it leaves the robot's tilt out of the controller's
geometry (README "Nav2").

Each new odom -> pelvis transform gives one pelvis -> pelvis_level and one
pelvis_level -> base_footprint with the same stamp. odom must be level, with the
floor at z = 0: the sim's world in sim_gt mode, and in fastlio mode the frame
frames.py builds from the robot's standing start pose.
'''

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


def level_rotation(quat_xyzw):
    '''Rotation (x, y, z, w) from a frame to its level twin, given the frame in a level parent'''
    rotation = Rotation.from_quat(quat_xyzw)
    x_axis = rotation.as_matrix()[:, 0]
    heading = Rotation.from_euler('z', np.arctan2(x_axis[1], x_axis[0]))
    return (rotation.inv() * heading).as_quat()


class LevelFrame(Node):

    def __init__(self):
        super().__init__('level_frame')
        self.fixed_frame = self.declare_parameter('fixed_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'pelvis').value
        self.level_frame = self.declare_parameter('level_frame', 'pelvis_level').value
        self.floor_frame = self.declare_parameter('floor_frame', 'base_footprint').value
        rate = self.declare_parameter('rate', 100.0).value
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = TransformBroadcaster(self)
        self.last_stamp = None
        self.create_timer(1.0 / rate, self._tick)

    def _tick(self):
        try:
            base = self.buffer.lookup_transform(self.fixed_frame, self.base_frame, Time())
        except TransformException:
            return  # no odometry yet
        if base.header.stamp == self.last_stamp:
            return
        self.last_stamp = base.header.stamp
        q = base.transform.rotation
        level = TransformStamped()
        level.header.stamp = base.header.stamp
        level.header.frame_id = self.base_frame
        level.child_frame_id = self.level_frame
        rotation = level.transform.rotation
        rotation.x, rotation.y, rotation.z, rotation.w = (
            float(v) for v in level_rotation([q.x, q.y, q.z, q.w]))
        # pelvis_level's z axis is odom's: the floor is the pelvis's height below it
        floor = TransformStamped()
        floor.header.stamp = base.header.stamp
        floor.header.frame_id = self.level_frame
        floor.child_frame_id = self.floor_frame
        floor.transform.translation.z = -base.transform.translation.z
        floor.transform.rotation.w = 1.0
        self.broadcaster.sendTransform([level, floor])


def main(args=None):
    rclpy.init(args=args)
    node = LevelFrame()
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
