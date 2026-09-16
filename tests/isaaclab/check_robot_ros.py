"""Native ROS motor/sensor interoperability check against a live Isaac simulation.

Invoked by run_robot_interop.sh on an isolated nonzero DDS domain. No DDS
participant is created by the SDK: it only supplies the command CRC encoding.
"""
import math
import os
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu, PointCloud2
from rosgraph_msgs.msg import Clock
from livox_ros_driver2.msg import CustomMsg
from unitree_hg.msg import LowCmd, LowState
from unitree_go.msg import MotorCmd, MotorCmds, MotorStates
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC


def main():
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    if not 1 <= domain <= 232:
        raise RuntimeError('ROS_DOMAIN_ID must be nonzero for simulation testing')
    rclpy.init()
    node = rclpy.create_node('isaac_robot_interop_check')
    publisher = node.create_publisher(LowCmd, '/lowcmd', 10)
    hand_publisher = node.create_publisher(MotorCmds, '/inspire/cmd', 10)
    crc = CRC()
    latest, seen, subscriptions = {}, set(), []
    samples = []

    def lowstate(msg):
        values = [v for motor in msg.motor_state[:27] for v in (motor.q, motor.dq, motor.tau_est)]
        quat = list(msg.imu_state.quaternion)
        assert all(math.isfinite(x) for x in values + quat), 'nonfinite lowstate'
        assert abs(sum(x*x for x in quat) - 1) < .05, 'invalid quaternion norm'
        # The base task spawns fixed upright at yaw -90 degrees in WXYZ.
        expected = (math.sqrt(.5), 0., 0., -math.sqrt(.5))
        assert abs(sum(a*b for a, b in zip(quat, expected))) > .99, f'expected fixture WXYZ yaw -90 quaternion, got {quat}'
        latest['state'] = msg
        samples.append((time.monotonic(), float(msg.motor_state[19].q)))

    subscriptions.append(node.create_subscription(LowState, '/lowstate', lowstate, qos_profile_sensor_data))

    def handstate(msg):
        assert len(msg.states) == 12, 'Inspire state must contain 12 motors'
        assert all(math.isfinite(m.q) for m in msg.states), 'nonfinite Inspire state'
        latest['hand'] = msg

    subscriptions.append(node.create_subscription(MotorStates, '/inspire/state', handstate, qos_profile_sensor_data))

    def send_hand(positions):
        if positions is None:
            return
        msg = MotorCmds()
        for position in positions:
            motor = MotorCmd()
            motor.mode = 1
            motor.q = float(position)
            msg.cmds.append(motor)
        hand_publisher.publish(msg)

    def sensor(topic, msg):
        if isinstance(msg, Image):
            assert msg.width > 0 and msg.height > 0 and len(msg.data) > 0, topic
        elif isinstance(msg, CompressedImage):
            assert msg.format and len(msg.data) > 0, topic
        elif isinstance(msg, CameraInfo):
            assert msg.width > 0 and msg.k[0] > 0, topic
        elif isinstance(msg, PointCloud2):
            assert msg.width > 0 and len(msg.data) > 0, topic
        elif isinstance(msg, CustomMsg):
            assert msg.point_num > 0 and msg.point_num == len(msg.points), topic
        elif isinstance(msg, Imu):
            assert all(math.isfinite(v) for v in (msg.linear_acceleration.x, msg.linear_acceleration.y,
                msg.linear_acceleration.z, msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)), topic
        elif isinstance(msg, Clock):
            latest['sim_time'] = msg.clock.sec + msg.clock.nanosec * 1e-9
        seen.add(topic)

    required = {'/clock': Clock, '/livox/lidar': CustomMsg, '/livox/pointcloud': PointCloud2, '/livox/imu': Imu}
    for camera in ('head', 'left_hand', 'right_hand'):
        root = f'/realsense/{camera}'
        required.update({root + '/color/image_raw': Image,
                         root + '/color/image_raw/compressed': CompressedImage,
                         root + '/aligned_depth_to_color/image_raw': Image,
                         root + '/aligned_depth_to_color/image_raw/compressedDepth': CompressedImage,
                         root + '/color/camera_info': CameraInfo})
    for topic, kind in required.items():
        subscriptions.append(node.create_subscription(kind, topic, lambda msg, t=topic: sensor(t, msg), qos_profile_sensor_data))

    def send(targets=None):
        wire, native = unitree_hg_msg_dds__LowCmd_(), LowCmd()
        wire.mode_pr = native.mode_pr = 0
        wire.mode_machine = native.mode_machine = int(latest['state'].mode_machine) if 'state' in latest else 0
        for i, (sdk_motor, ros_motor) in enumerate(zip(wire.motor_cmd, native.motor_cmd)):
            if targets is not None and i < 27:
                sdk_motor.mode = ros_motor.mode = 1
                sdk_motor.q = ros_motor.q = float(targets[i])
                sdk_motor.kp = ros_motor.kp = 20.
                sdk_motor.kd = ros_motor.kd = 2.
        native.crc = crc.Crc(wire)
        publisher.publish(native)

    deadline = time.monotonic() + float(os.environ.get('ROBOT_INTEROP_TIMEOUT', '300'))
    initial = None
    initial_hand = None
    hand_targets = None
    first_hand_actual = None
    stage = 0
    start_sim = None
    targets = None
    next_publish = 0.
    next_trace = 0.
    try:
        print('SUBSCRIBER_READY', flush=True)
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.005)
            if 'state' not in latest or 'sim_time' not in latest or 'hand' not in latest:
                continue
            if initial is None:
                initial = [float(m.q) for m in latest['state'].motor_state[:27]]
                targets = initial.copy()
                targets[19] += .1
                initial_hand = [float(m.q) for m in latest['hand'].states]
                hand_targets = initial_hand.copy()
                hand_targets[3] = .95
                start_sim = latest['sim_time']
                print(f'Initial wrist q={initial[19]:.6f}; target={targets[19]:.6f}; right index normalized q={initial_hand[3]:.6f}, target={hand_targets[3]:.6f}', flush=True)
            if latest['sim_time'] >= next_trace:
                motor = latest['state'].motor_state[19]
                print(f"TRACE sim={latest['sim_time']:.3f} tick={latest['state'].tick} wrist19 q={motor.q:.6f} dq={motor.dq:.6f} tau_est={motor.tau_est:.6f} command(mode=1 q={targets[19]:.6f} dq=0 tau=0 kp=20 kd=2) right_index3 q={latest['hand'].states[3].q:.6f} target={hand_targets[3]:.6f}", flush=True)
                next_trace = latest['sim_time'] + .1
            if time.monotonic() >= next_publish:
                send(targets)
                send_hand(hand_targets)
                next_publish = time.monotonic() + .01
            if latest['sim_time'] - start_sim >= 1.0:
                actual = float(latest['state'].motor_state[19].q)
                assert abs(actual - targets[19]) < .06, f'wrist did not converge: target={targets[19]}, actual={actual}'
                hand_actual = float(latest['hand'].states[3].q)
                assert abs(hand_actual - hand_targets[3]) < .03, f'right index did not converge: target={hand_targets[3]}, actual={hand_actual}'
                if stage == 0:
                    assert initial_hand[3] - hand_actual > .015, 'Inspire flex command did not produce finger motion'
                    first_hand_actual = hand_actual
                    hand_targets[3] = 1.0
                    assert actual - initial[19] > .04, 'positive target did not produce positive physical motion'
                    stage = 1
                    targets[19] = initial[19]
                    start_sim = latest['sim_time']
                    print(f'Wrist moved to {actual:.6f}; reversing to {targets[19]:.6f}', flush=True)
                else:
                    assert hand_actual - first_hand_actual > .015, 'Inspire reverse command did not produce finger motion'
                    missing = set(required) - seen
                    if not missing:
                        print(f'PASS: ROS LowCmd and Inspire commands drove wrist and right index in both directions; {len(samples)} LowState samples; all {len(required)} sensor topics received', flush=True)
                        return
        raise AssertionError(f'timeout: stage={stage}, lowstate_samples={len(samples)}, missing_topics={sorted(set(required)-seen)}')
    finally:
        # Release motors even when an assertion or transport error aborts the test.
        for _ in range(10):
            send()
            send_hand(initial_hand)
            time.sleep(.01)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
