'''A kinematic G1 for the navigation tests: no physics, and nothing to or from Unitree.

    python3 fake_g1.py --odom --sway          # what test_nav2_smoke.py runs

It turns /cmd_vel (vx, wz) into the pelvis's pose, through a 0.2 s lag and
acceleration limits, and publishes what the G1 sim will:

- /livox/lidar (livox_ros_driver2/CustomMsg, 10 Hz) and, with --imu,
  /livox/imu (200 Hz): the MID-360 upside down on the torso, in livox_frame,
  ray-cast in ROOM. Every point of a scan is taken at its stamp (offset_time 0).
- /joint_states, all body joints at 0 (the arms in 'home') except the waist
  pitch (--waist), for robot_state_publisher.
- with --odom, the sim's ground truth: /odom and odom -> pelvis.
- the true pelvis pose on /fake_g1/pose (geometry_msgs/PoseStamped, in odom),
  for the tests only.

With --sway the pelvis leans like HOMIE's in MuJoCo: 1.8 deg back and 0.744 m
high standing; walking at 0.4 m/s, 5.3 deg back, rolling +-4.8 deg at 1.75 Hz,
and 0.724 m high. Without it the pelvis is at 0.744 m and pitched by --pitch.
--waist pitches the torso on the pelvis (HOMIE stands with 0.0256 rad); the
lidar and the IMU move with the torso.
The IMU does not model the sway, so --imu needs a constant tilt (no --sway).
'''

import argparse
import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, JointState
from tf2_ros import TransformBroadcaster

from g1_navigation import frames
from golem_ros2_models.paths import urdf_path

GRAVITY = 9.81

# the G1's body joints, all published on /joint_states (rt/lowstate slot order)
BODY_JOINTS = (
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
)

# (min corner, max corner) of each box, m, in odom: the robot starts at the
# origin facing +x. A 7 x 4 m room, one obstacle on the straight line to the
# goal at (4, 0), and a cabinet behind the start that breaks the symmetry.
ROOM = (
    ((-1.6, -2.1, 0.0), (-1.5, 2.1, 2.5)),   # walls
    ((5.5, -2.1, 0.0), (5.6, 2.1, 2.5)),
    ((-1.6, 2.0, 0.0), (5.6, 2.1, 2.5)),
    ((-1.6, -2.1, 0.0), (5.6, -2.0, 2.5)),
    ((1.7, -0.3, 0.0), (2.3, 0.3, 0.9)),     # the obstacle
    ((-1.0, 1.2, 0.0), (-0.4, 2.0, 1.0)),    # a cabinet
)
OBSTACLE = ROOM[4]

# MID-360: 360 deg around, -7..52 deg in elevation in its own frame (upside
# down on the G1: 7 deg above the horizon to 52 deg below)
AZIMUTHS = np.radians(np.arange(0.0, 360.0, 1.0))
ELEVATIONS = np.radians(np.linspace(-7.0, 52.0, 16))
MAX_RANGE = 40.0
MIN_RANGE = 0.1

# pelvis pose while standing / walking (HOMIE in MuJoCo, 2026-09-30)
STAND = {'z': 0.744, 'pitch': math.radians(-1.8)}
WALK = {'z': 0.724, 'pitch': math.radians(-5.3), 'roll': math.radians(4.8), 'hz': 1.75}

WAIST_PITCH = 'waist_pitch_joint'
HOMIE_WAIST = 0.0256  # rad, HOMIE standing in MuJoCo

LAG = 0.2  # s, first order, from command to velocity
MAX_ACCEL = (1.0, 2.0)  # vx m/s^2, wz rad/s^2
CMD_TIMEOUT = 0.5  # s without /cmd_vel: stop


def ray_directions():
    '''Unit rays in the MID-360's frame, elevation-major'''
    el, az = np.meshgrid(ELEVATIONS, AZIMUTHS, indexing='ij')
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
                    axis=-1).reshape(-1, 3)


def cast(origin, directions, boxes=ROOM):
    '''Distance along each ray to the floor (z = 0) or the nearest box; inf for a miss'''
    with np.errstate(divide='ignore', invalid='ignore'):
        dz = directions[:, 2]
        hit = np.where(dz < 0.0, -origin[2] / dz, np.inf)
        for low, high in boxes:
            t1 = (np.asarray(low) - origin) / directions
            t2 = (np.asarray(high) - origin) / directions
            near = np.nanmax(np.minimum(t1, t2), axis=1)
            far = np.nanmin(np.maximum(t1, t2), axis=1)
            box = np.where((far >= near) & (near > 0.0), near, np.inf)
            hit = np.minimum(hit, box)
    hit[(hit < MIN_RANGE) | (hit > MAX_RANGE)] = np.inf
    return hit


def room_distance(xy, boxes=ROOM):
    '''Distance in the plane from each point to the nearest box'''
    dist = np.full(len(xy), np.inf)
    for low, high in boxes:
        dx = np.maximum(np.maximum(low[0] - xy[:, 0], xy[:, 0] - high[0]), 0.0)
        dy = np.maximum(np.maximum(low[1] - xy[:, 1], xy[:, 1] - high[1]), 0.0)
        dist = np.minimum(dist, np.hypot(dx, dy))
    return dist


def occupied_cells(grid):
    '''xy of the centres of an OccupancyGrid's occupied cells, in its frame'''
    info = grid.info
    cells = np.argwhere(np.array(grid.data).reshape(info.height, info.width) > 50)
    return (cells[:, ::-1] + 0.5) * info.resolution + [info.origin.position.x,
                                                       info.origin.position.y]


def footprint_hits_box(x, y, yaw, footprint, box):
    '''True if the footprint polygon (pelvis frame) at (x, y, yaw) overlaps the box's xy'''
    c, s = math.cos(yaw), math.sin(yaw)
    corners = np.array([(x + c * px - s * py, y + s * px + c * py) for px, py in footprint])
    (x0, y0, _), (x1, y1, _) = box
    square = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    # separating axis test: the box's axes, then the footprint's edge normals
    axes = [(1.0, 0.0), (0.0, 1.0)]
    for i in range(len(corners)):
        ex, ey = corners[(i + 1) % len(corners)] - corners[i]
        axes.append((-ey, ex))
    for axis in axes:
        a, b = corners @ axis, square @ axis
        if a.max() < b.min() or b.max() < a.min():
            return False
    return True


class FakeG1(Node):

    def __init__(self, odom, sway, imu, cmd_topic, waist=0.0, pitch=0.0):
        super().__init__('fake_g1')
        self.sway = sway
        self.waist = waist
        self.pitch = pitch
        self.x = self.y = self.yaw = 0.0
        self.v = self.w = 0.0  # vx, wz
        self.accel = (0.0, 0.0)
        self.cmd = (0.0, 0.0)
        self.cmd_time = None
        self.gait = 0.0  # 0 standing, 1 walking
        self.t0 = self.last = self.get_clock().now()
        self.pelvis_lidar = frames.link_pose(urdf_path('g1'), frames.BASE_LINK,
                                             frames.LIDAR_LINK, {WAIST_PITCH: waist})
        sim_config = os.path.join(get_package_share_directory('g1_navigation'), 'config',
                                  'g1_fastlio_sim.yaml')
        self.lidar_imu = frames.invert(frames.fastlio_extrinsic(sim_config))
        self.rays = ray_directions()
        self.lines = (np.arange(len(self.rays)) // len(AZIMUTHS)) % 4

        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pub_lidar = self.create_publisher(CustomMsg, '/livox/lidar', reliable)
        self.pub_imu = self.create_publisher(Imu, '/livox/imu', reliable) if imu else None
        self.pub_joints = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_truth = self.create_publisher(PoseStamped, '/fake_g1/pose', 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10) if odom else None
        self.tf = TransformBroadcaster(self) if odom else None
        self.create_subscription(Twist, cmd_topic, self._on_cmd, 10)
        self.ticks = 0
        self.create_timer(0.005 if imu else 0.02, self._tick)

    def _on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z)
        self.cmd_time = self.get_clock().now()

    def pelvis(self, t):
        '''T_odom_pelvis at time t (s since start)'''
        if not self.sway:
            rotation = Rotation.from_euler('xyz', (0.0, self.pitch, self.yaw)).as_matrix()
            return frames.transform((self.x, self.y, STAND['z']), rotation)
        g = self.gait
        roll = g * WALK['roll'] * math.sin(2 * math.pi * WALK['hz'] * t)
        pitch = STAND['pitch'] + g * (WALK['pitch'] - STAND['pitch'])
        z = STAND['z'] + g * (WALK['z'] - STAND['z'])
        rotation = Rotation.from_euler('xyz', (roll, pitch, self.yaw)).as_matrix()
        return frames.transform((self.x, self.y, z), rotation)

    def _step(self, dt):
        now = self.get_clock().now()
        if self.cmd_time is None or (now - self.cmd_time).nanoseconds * 1e-9 > CMD_TIMEOUT:
            self.cmd = (0.0, 0.0)
        accel = []
        for i, (value, target) in enumerate(((self.v, self.cmd[0]), (self.w, self.cmd[1]))):
            limit = MAX_ACCEL[i]
            accel.append(max(-limit, min(limit, (target - value) / LAG)))
        self.accel = tuple(accel)
        self.v += self.accel[0] * dt
        self.w += self.accel[1] * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.w * dt
        moving = abs(self.v) > 0.05 or abs(self.w) > 0.05
        self.gait = max(0.0, min(1.0, self.gait + (dt if moving else -dt) / 0.5))

    def _tick(self):
        now = self.get_clock().now()
        dt = min((now - self.last).nanoseconds * 1e-9, 0.05)
        self.last = now
        self._step(dt)
        t = (now - self.t0).nanoseconds * 1e-9
        stamp = now.to_msg()
        pelvis = self.pelvis(t)
        self.ticks += 1
        if self.pub_imu is not None:
            self._publish_imu(stamp, pelvis)
        if self.pub_imu is None or self.ticks % 4 == 0:
            self._publish_state(stamp, pelvis)
        if self.ticks % (20 if self.pub_imu is not None else 5) == 0:
            self._publish_scan(stamp, pelvis)

    def _publish_state(self, stamp, pelvis):
        x, y, z = pelvis[:3, 3]
        qx, qy, qz, qw = Rotation.from_matrix(pelvis[:3, :3]).as_quat()
        truth = PoseStamped()
        truth.header.stamp = stamp
        truth.header.frame_id = 'odom'
        truth.pose.position.x, truth.pose.position.y, truth.pose.position.z = x, y, z
        o = truth.pose.orientation
        o.x, o.y, o.z, o.w = qx, qy, qz, qw
        self.pub_truth.publish(truth)
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = list(BODY_JOINTS)
        joints.position = [self.waist if name == WAIST_PITCH else 0.0 for name in BODY_JOINTS]
        self.pub_joints.publish(joints)
        if self.pub_odom is None:
            return
        odom = Odometry()  # like the sim's: pose only, the twist left at 0
        odom.header = truth.header
        odom.child_frame_id = frames.BASE_LINK
        odom.pose.pose = truth.pose
        self.pub_odom.publish(odom)
        tf = TransformStamped()
        tf.header = truth.header
        tf.child_frame_id = frames.BASE_LINK
        tf.transform.translation.x, tf.transform.translation.y = x, y
        tf.transform.translation.z = z
        tf.transform.rotation = truth.pose.orientation
        self.tf.sendTransform(tf)

    def _publish_scan(self, stamp, pelvis):
        lidar = pelvis @ self.pelvis_lidar
        dist = cast(lidar[:3, 3], self.rays @ lidar[:3, :3].T)
        valid = np.isfinite(dist)
        points = self.rays[valid] * dist[valid, None]
        msg = CustomMsg()
        msg.header.stamp = stamp
        msg.header.frame_id = frames.LIVOX_FRAME
        msg.timebase = stamp.sec * 1_000_000_000 + stamp.nanosec
        msg.point_num = int(len(points))
        msg.points = [CustomPoint(offset_time=0, x=float(p[0]), y=float(p[1]), z=float(p[2]),
                                  reflectivity=100, tag=0, line=int(line))
                      for p, line in zip(points, self.lines[valid])]
        self.pub_lidar.publish(msg)

    def _publish_imu(self, stamp, pelvis):
        # the IMU on a pelvis with a constant tilt, moving in the plane (no sway)
        imu = pelvis @ self.pelvis_lidar @ self.lidar_imu
        r = imu[:3, 3] - pelvis[:3, 3]  # lever arm, odom axes
        omega = np.array([0.0, 0.0, self.w])
        alpha = np.array([0.0, 0.0, self.accel[1]])
        heading = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        left = np.array([-math.sin(self.yaw), math.cos(self.yaw), 0.0])
        accel = (self.accel[0] * heading + self.v * self.w * left
                 + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r)))
        specific_force = imu[:3, :3].T @ (accel + np.array([0.0, 0.0, GRAVITY]))
        gyro = imu[:3, :3].T @ omega
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = frames.LIVOX_FRAME
        msg.orientation_covariance[0] = -1.0  # no orientation, like the MID-360
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gyro
        a = msg.linear_acceleration
        a.x, a.y, a.z = specific_force
        self.pub_imu.publish(msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--odom', action='store_true', help='publish ground-truth /odom and TF')
    parser.add_argument('--sway', action='store_true', help="lean the pelvis like HOMIE's")
    parser.add_argument('--imu', action='store_true', help='publish /livox/imu (needs no --sway)')
    parser.add_argument('--waist', type=float, default=0.0, help='waist pitch, rad')
    parser.add_argument('--pitch', type=float, default=0.0,
                        help='pelvis pitch without --sway, rad (< 0 leans back)')
    parser.add_argument('--cmd-topic', default='/cmd_vel')
    args = parser.parse_args()
    if args.imu and args.sway:
        parser.error('--imu does not model the sway')
    if args.pitch and args.sway:
        parser.error('--sway sets the pitch')
    rclpy.init()
    node = FakeG1(args.odom, args.sway, args.imu, args.cmd_topic, args.waist, args.pitch)
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
