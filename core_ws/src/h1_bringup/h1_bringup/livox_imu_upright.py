"""Rotate the MID360 IMU into the frame livox_ros_driver2 publishes points in.

livox_ros_driver2 applies the MID360_config.json ``extrinsic_parameter``
rotation to the point cloud but publishes the IMU in the raw sensor frame. On
the H1-2 the lidar is mounted inverted (roll 180), so the two streams disagree
by 180 deg of roll. FAST-LIO (mid360.yaml, identity extrinsic_R) needs
them in one frame, and seeds camera_init from the IMU attitude at startup, so
the IMU must arrive upright or the whole map comes out upside down.

The rotation is read from the same JSON file the driver loads and built with
the driver's convention, R = Rz(yaw) * Ry(pitch) * Rx(roll)
(livox_ros_driver2 src/comm/pub_handler.cpp, SetLidarsExtParam), so points and
IMU stay consistent if the extrinsic is ever changed.

Subscribes ``imu_in`` (sensor_msgs/Imu), publishes ``imu_out``.
Parameter ``livox_config_path``: path to the driver's MID360_config.json.
"""

import json
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

GRAVITY_CHECK_SAMPLES = 200  # ~1 s at the MID360's 200 Hz IMU rate
GRAVITY_MIN_Z = 0.5  # g; upright with the torso's ~14 deg lidar pitch reads ~0.98


def driver_rotation(roll_deg, pitch_deg, yaw_deg):
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return (
        (cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy),
        (cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy),
        (-sp, sr * cp, cr * cp),
    )


def load_rotation(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    configs = cfg['lidar_configs']
    if len(configs) != 1:
        raise ValueError(
            f'{config_path}: expected exactly one lidar_configs entry, '
            f'got {len(configs)}')
    ext = configs[0]['extrinsic_parameter']
    return driver_rotation(ext['roll'], ext['pitch'], ext['yaw'])


def rotate(r, v):
    x, y, z = v.x, v.y, v.z
    v.x = r[0][0] * x + r[0][1] * y + r[0][2] * z
    v.y = r[1][0] * x + r[1][1] * y + r[1][2] * z
    v.z = r[2][0] * x + r[2][1] * y + r[2][2] * z


class LivoxImuUpright(Node):

    def __init__(self):
        super().__init__('livox_imu_upright')
        config_path = self.declare_parameter('livox_config_path', '').value
        if not config_path:
            raise ValueError('livox_config_path parameter is required')
        self._r = load_rotation(config_path)
        self.get_logger().info(
            f'rotating IMU by driver extrinsic from {config_path}: {self._r}')
        # Depth matches FAST-LIO's IMU subscription so bursts are buffered,
        # not dropped, on the way through.
        self._pub = self.create_publisher(Imu, 'imu_out', 200000)
        self.create_subscription(Imu, 'imu_in', self._on_imu, 200000)
        self._check_acc_z = []

    def _on_imu(self, msg):
        rotate(self._r, msg.angular_velocity)
        rotate(self._r, msg.linear_acceleration)
        if self._check_acc_z is not None:
            self._check_gravity(msg.linear_acceleration)
        # The driver never fills orientation (it stays the all-zero default),
        # so there is nothing to rotate there.
        self._pub.publish(msg)

    def _check_gravity(self, acc):
        # FAST-LIO initialises from the first ~1 s of IMU with the robot
        # standing, so gravity should read along +z (units of g) once rotated.
        # A negative mean means the IMU is still inverted relative to the
        # points and FAST-LIO will build an upside-down map.
        self._check_acc_z.append(acc.z)
        if len(self._check_acc_z) < GRAVITY_CHECK_SAMPLES:
            return
        mean_z = sum(self._check_acc_z) / len(self._check_acc_z)
        self._check_acc_z = None
        if mean_z < GRAVITY_MIN_Z:
            self.get_logger().error(
                f'IMU z acceleration averages {mean_z:.2f} g after rotation; '
                f'expected about +1 g with the robot upright. The IMU and '
                f'point frames disagree, check MID360_config.json '
                f'extrinsic_parameter and mid360.yaml extrinsic_R.')
        else:
            self.get_logger().info(
                f'IMU gravity check passed: z averages {mean_z:.2f} g')


def main():
    rclpy.init()
    node = LivoxImuUpright()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
