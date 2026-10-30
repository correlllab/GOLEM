'''ROS 2 node driving both Dex3-1 hands over rt/dex3/*'''

import os
import signal
import threading
from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from custom_ros_messages.srv import SetHandPose

from dex_3.driver import Dex3Hand
from dex_3.dex3 import (
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_MAX_SPEED,
    JOINT_NAMES,
    SHUTDOWN_ACTIONS,
    SIDES,
    dds_domain,
    named_pose,
)


class Dex3HandNode(Node):
    '''
    Per hand <side> in {left, right}:
    - services /<side>/hand/set_pose (SetHandPose), /<side>/hand/open,
      /<side>/hand/close and /<side>/hand/release (std_srvs/Trigger)
    - topic /<side>/hand/pressure (Float32MultiArray, sensors x 12)
    and /joint_states with the finger joints, for robot_state_publisher.
    A hand is only commanded after its first service call. On shutdown a hand
    this node was commanding is released (on_shutdown 'release', the fingers go
    limp) or simply left with its last command ('silent')
    '''

    def __init__(self):
        super().__init__('dex3_hands')
        self.declare_parameter('sides', list(SIDES))
        self.declare_parameter('kp', DEFAULT_KP)
        self.declare_parameter('kd', DEFAULT_KD)
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('pose_duration', 1.0)  # s, for open and close
        self.declare_parameter('joint_state_hz', 50.0)
        self.declare_parameter('network_interface', '')
        self.declare_parameter('domain_id', -1)  # used when ROS_DOMAIN_ID is unset; < 0: unset
        self.declare_parameter('on_shutdown', 'release')  # release | silent
        self.declare_parameter('max_speed', DEFAULT_MAX_SPEED)  # rad/s, any finger joint

        sides = [s for s in self.get_parameter('sides').value]
        bad = [s for s in sides if s not in SIDES]
        if bad:
            raise ValueError(f'unknown hand sides {bad}; use {list(SIDES)}')
        self._pose_duration = float(self.get_parameter('pose_duration').value)
        self._on_shutdown = str(self.get_parameter('on_shutdown').value)
        if self._on_shutdown not in SHUTDOWN_ACTIONS:
            raise ValueError(f'on_shutdown must be one of {list(SHUTDOWN_ACTIONS)}, '
                             f'got {self._on_shutdown!r}')

        # Unitree DDS on the same domain as ROS (the real robot is domain 0),
        # never a default one: refused unless it is also the domain this node's
        # ROS side runs on. The SDK ignores CYCLONEDDS_URI, so the interface
        # comes from the network_interface parameter or G1_DDS_INTERFACE
        # (e.g. enp130s0)
        domain = dds_domain(os.environ.get('ROS_DOMAIN_ID'),
                            self.get_parameter('domain_id').value,
                            ros_domain=self.context.get_domain_id())
        interface = (self.get_parameter('network_interface').value
                     or os.environ.get('G1_DDS_INTERFACE', ''))
        if interface:
            ChannelFactoryInitialize(domain, interface)
        else:
            ChannelFactoryInitialize(domain)

        kp = float(self.get_parameter('kp').value)
        kd = float(self.get_parameter('kd').value)
        rate_hz = float(self.get_parameter('rate_hz').value)
        max_speed = float(self.get_parameter('max_speed').value)
        self.hands = {side: Dex3Hand(side, kp=kp, kd=kd, rate_hz=rate_hz, max_speed=max_speed)
                      for side in sides}

        self._pressure_pubs = {}
        for side in sides:
            prefix = f'/{side}/hand'
            self.create_service(SetHandPose, f'{prefix}/set_pose', partial(self._set_pose, side))
            for name in ('open', 'close'):
                self.create_service(Trigger, f'{prefix}/{name}', partial(self._named_pose, side, name))
            self.create_service(Trigger, f'{prefix}/release', partial(self._release, side))
            self._pressure_pubs[side] = self.create_publisher(
                Float32MultiArray, f'{prefix}/pressure', 10)
        self._joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_timer(1.0 / float(self.get_parameter('joint_state_hz').value),
                          self._publish_state)
        self.get_logger().info(
            f'Dex3 hands {sides} ready on DDS domain {domain}, interface '
            f'{interface or "auto"} (kp={kp}, kd={kd}, max_speed={max_speed} rad/s); '
            f'silent until the first command; on_shutdown={self._on_shutdown}')

    def _set_pose(self, side, request, response):
        try:
            goal = self.hands[side].set_pose(request.positions, request.duration)
        except (ValueError, RuntimeError) as err:
            response.success = False
            response.message = str(err)
            return response
        response.success = True
        response.message = f'{side} hand moving to {goal.round(3).tolist()}'
        return response

    def _named_pose(self, side, name, request, response):
        try:
            self.hands[side].set_pose(named_pose(side, name), self._pose_duration)
        except (ValueError, RuntimeError) as err:
            response.success = False
            response.message = str(err)
            return response
        response.success = True
        response.message = f'{side} hand: {name}'
        return response

    def _release(self, side, request, response):
        released = self.hands[side].release()
        response.success = True
        response.message = (f'{side} hand released' if released
                            else f'{side} hand was not being commanded')
        return response

    def _publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for side, hand in self.hands.items():
            state = hand.state
            if state is None:
                continue
            msg.name.extend(JOINT_NAMES[side])
            msg.position.extend(state['q'].tolist())
            msg.velocity.extend(state['dq'].tolist())
            msg.effort.extend(state['tau'].tolist())
            pressure = Float32MultiArray()
            rows, cols = state['pressure'].shape
            pressure.layout.dim = [
                MultiArrayDimension(label='sensor', size=rows, stride=rows * cols),
                MultiArrayDimension(label='reading', size=cols, stride=cols),
            ]
            pressure.data = state['pressure'].reshape(-1).tolist()
            self._pressure_pubs[side].publish(pressure)
        if msg.name:
            self._joint_state_pub.publish(msg)

    def destroy_node(self):
        for hand in self.hands.values():
            hand.shutdown(release=self._on_shutdown == 'release')
        return super().destroy_node()


# signals that end the node through destroy_node(), so on_shutdown applies:
# Ctrl-C, ros2 launch shutting down (SIGINT, then SIGTERM) and a closed terminal
# (SIGHUP, whose default would end the process at once). SIGKILL cannot be caught
EXIT_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def main(args=None):
    # an exit signal only requests the exit, however many arrive. rclpy's own
    # handlers stay off: Ctrl-C in a terminal reaches the node twice (from the
    # terminal, and from ros2 launch passing it on), and a second
    # KeyboardInterrupt would cut the release short
    exit_requested = threading.Event()
    for signum in EXIT_SIGNALS:
        signal.signal(signum, lambda *_: exit_requested.set())
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = Dex3HandNode()
    try:
        while not exit_requested.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
