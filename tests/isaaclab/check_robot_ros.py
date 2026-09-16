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
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC


class InspireCheck:
    def __init__(self, node):
        from unitree_go.msg import MotorCmd, MotorCmds, MotorStates
        self.MotorCmd, self.MotorCmds = MotorCmd, MotorCmds
        self.publisher = node.create_publisher(MotorCmds, '/inspire/cmd', 10)
        self.state = None
        self.initial = None
        self.subscription = node.create_subscription(MotorStates, '/inspire/state', self.receive, qos_profile_sensor_data)

    def receive(self, msg):
        assert len(msg.states) == 12 and all(math.isfinite(m.q) for m in msg.states), 'invalid Inspire state'
        self.state = msg

    def ready(self):
        return self.state is not None

    @property
    def actual(self):
        return float(self.state.states[3].q)

    def start(self):
        self.initial = [float(m.q) for m in self.state.states]
        self.positions = self.initial.copy()
        self.positions[3] = self.target = .95

    def tick(self):
        self.send(self.positions)

    def send(self, positions):
        msg = self.MotorCmds()
        for position in positions:
            motor = self.MotorCmd()
            motor.mode, motor.q = 1, float(position)
            msg.cmds.append(motor)
        self.publisher.publish(msg)

    def verify(self, stage):
        assert abs(self.actual - self.target) < .03, f'Inspire target={self.target}, actual={self.actual}'
        if stage == 0:
            assert self.initial[3] - self.actual > .015, 'Inspire flex did not move'
            self.first_actual = self.actual
        else:
            assert self.actual - self.first_actual > .015, 'Inspire reverse did not move'

    def reverse(self):
        self.positions[3] = self.target = 1.

    def action_check(self):
        return True

    def restore(self):
        if self.initial is not None:
            self.send(self.initial)


class MagpieCheck:
    def __init__(self, node):
        from rclpy.action import ActionClient
        from magpie_msgs.msg import GripperState
        from magpie_msgs.srv import SetGripperPosition, SetGripperForce
        from magpie_msgs.action import DeliGrasp
        from std_srvs.srv import Trigger
        self.Position, self.DeliGrasp = SetGripperPosition, DeliGrasp
        self.node = node
        self.state, self.initial = {}, None
        self.subscriptions, self.clients = [], {}
        self.pending = []
        self.goal_future = self.result_future = None
        self.state_sequence = 0
        self.result_state_sequence = None
        for side in ('left', 'right'):
            root = f'/{side}/gripper'
            self.subscriptions.append(node.create_subscription(GripperState, root + '/state', lambda msg, s=side: self.receive(s, msg), qos_profile_sensor_data))
            for name, kind in (('set_position', SetGripperPosition), ('set_force', SetGripperForce), ('open', Trigger), ('close', Trigger), ('calibrate', Trigger), ('reset_parameters', Trigger)):
                self.clients[(side, name)] = node.create_client(kind, root + '/' + name)
        self.action = ActionClient(node, DeliGrasp, '/right/gripper/deligrasp')
        self.left_action = ActionClient(node, DeliGrasp, '/left/gripper/deligrasp')

    def receive(self, side, msg):
        assert all(math.isfinite(x) for x in [msg.position, msg.force] + list(msg.finger_positions)), 'invalid Magpie state'
        self.state[side] = msg
        if side == 'right':
            self.state_sequence += 1

    def ready(self):
        return {'isaac_magpie_left', 'isaac_magpie_right'} <= set(self.node.get_node_names()) and len(self.state) == 2 and all(c.service_is_ready() for c in self.clients.values()) and self.action.server_is_ready() and self.left_action.server_is_ready()

    @property
    def actual(self):
        return float(self.state['right'].position)

    def request_position(self, position):
        request = self.Position.Request()
        request.position, request.speed = float(position), .3
        self.pending.append(self.clients[('right', 'set_position')].call_async(request))

    def start(self):
        self.initial = self.actual
        self.target = self.initial - 10. if self.initial > 15. else self.initial + 10.
        self.direction = 1. if self.target > self.initial else -1.
        self.request_position(self.target)

    def tick(self):
        for future in self.pending:
            if future.done():
                response = future.result()
                assert response.success, f'Magpie set_position rejected: {response.message}'

    def verify(self, stage):
        assert self.pending[-1].done(), 'Magpie service response missing'
        self.tick()
        assert abs(self.actual - self.target) < 3., f'Magpie target={self.target}mm actual={self.actual}mm'
        if stage == 0:
            assert self.direction * (self.actual - self.initial) > 3., 'Magpie service did not move physical fingers'
            self.first_actual = self.actual
        else:
            assert -self.direction * (self.actual - self.first_actual) > 3., 'Magpie reverse did not move physical fingers'

    def reverse(self):
        self.target = self.initial
        self.request_position(self.target)

    def action_check(self):
        if self.goal_future is None:
            goal = self.DeliGrasp.Goal()
            self.action_target = max(0., self.initial - 5.)
            goal.params.goal_aperture = self.action_target
            goal.params.initial_force = 1.
            goal.params.additional_closure = 0.
            goal.params.additional_force = 0.
            goal.params.complete_grasp = False
            self.goal_future = self.action.send_goal_async(goal)
            return False
        if not self.goal_future.done():
            return False
        handle = self.goal_future.result()
        assert handle.accepted, 'DeliGrasp goal rejected'
        if self.result_future is None:
            self.result_future = handle.get_result_async()
        if not self.result_future.done():
            return False
        result = self.result_future.result()
        assert result.status == 4 and result.result.success, f'DeliGrasp failed: {result.result.message}'
        assert abs(result.result.final_aperture - self.action_target) < 3., 'DeliGrasp result aperture missed target'
        # Action results and state topics arrive independently. Require a new
        # measured state after the result, bounded by the checker's global deadline.
        if self.result_state_sequence is None:
            self.result_state_sequence = self.state_sequence
        return self.state_sequence > self.result_state_sequence and abs(self.actual - self.action_target) < 3.

    def restore(self):
        if self.initial is not None:
            self.request_position(self.initial)


def main():
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    if not 1 <= domain <= 232:
        raise RuntimeError('ROS_DOMAIN_ID must be nonzero for simulation testing')
    rclpy.init()
    node = rclpy.create_node('isaac_robot_interop_check')
    publisher = node.create_publisher(LowCmd, '/lowcmd', 10)
    hand_type = os.environ.get('ISAAC_HAND_TYPE', 'magpie')
    assert hand_type in ('magpie', 'inspire'), hand_type
    hand = MagpieCheck(node) if hand_type == 'magpie' else InspireCheck(node)
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
    stage = 0
    start_sim = None
    targets = None
    next_publish = 0.
    next_trace = 0.
    try:
        print('SUBSCRIBER_READY', flush=True)
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.005)
            if 'state' not in latest or 'sim_time' not in latest or not hand.ready():
                continue
            if initial is None:
                initial = [float(m.q) for m in latest['state'].motor_state[:27]]
                targets = initial.copy()
                targets[19] += .1
                hand.start()
                start_sim = latest['sim_time']
                print(f'Initial wrist q={initial[19]:.6f}; target={targets[19]:.6f}; {hand_type} actual={hand.actual:.6f}, target={hand.target:.6f}', flush=True)
            if latest['sim_time'] >= next_trace:
                motor = latest['state'].motor_state[19]
                print(f"TRACE sim={latest['sim_time']:.3f} tick={latest['state'].tick} wrist19 q={motor.q:.6f} dq={motor.dq:.6f} tau_est={motor.tau_est:.6f} command(mode=1 q={targets[19]:.6f} dq=0 tau=0 kp=20 kd=2) {hand_type} actual={hand.actual:.6f} target={hand.target:.6f}", flush=True)
                next_trace = latest['sim_time'] + .1
            if time.monotonic() >= next_publish:
                send(targets)
                hand.tick()
                next_publish = time.monotonic() + .01
            if latest['sim_time'] - start_sim >= 1.0:
                actual = float(latest['state'].motor_state[19].q)
                assert abs(actual - targets[19]) < .06, f'wrist did not converge: target={targets[19]}, actual={actual}'
                if stage < 2:
                    hand.verify(stage)
                if stage == 0:
                    hand.reverse()
                    assert actual - initial[19] > .04, 'positive target did not produce positive physical motion'
                    stage = 1
                    targets[19] = initial[19]
                    start_sim = latest['sim_time']
                    print(f'Wrist moved to {actual:.6f}; reversing to {targets[19]:.6f}', flush=True)
                else:
                    if stage == 1:
                        stage = 2
                    missing = set(required) - seen
                    if not missing and hand.action_check():
                        print(f'PASS: ROS LowCmd and {hand_type} commands drove wrist and gripper in both directions; services/actions verified; {len(samples)} LowState samples; all {len(required)} sensor topics received', flush=True)
                        return
        raise AssertionError(f'timeout: stage={stage}, lowstate_samples={len(samples)}, missing_topics={sorted(set(required)-seen)}')
    finally:
        # Release motors even when an assertion or transport error aborts the test.
        for _ in range(10):
            send()
            hand.restore()
            time.sleep(.01)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
