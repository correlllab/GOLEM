"""Native ROS client assertions against a synthetic-physics Magpie sidecar."""
import math
import time
import rclpy
from rclpy.action import ActionClient
from magpie_msgs.action import DeliGrasp
from magpie_msgs.msg import GripperState
from magpie_msgs.srv import SetGripperPosition, SetGripperForce

rclpy.init()
node = rclpy.create_node('test_magpie_services')
states = {}
subscriptions = [node.create_subscription(GripperState, f'/{side}/gripper/state', lambda msg, s=side: states.__setitem__(s, msg), 10) for side in ('left', 'right')]

def complete(future, timeout=15):
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    assert future.done(), 'ROS request timed out'
    return future.result()

def wait_state(target):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.1)
        if 'left' in states and abs(states['left'].position - target) < .1:
            return
    raise AssertionError(f'measured state did not reach {target}')

try:
    position = node.create_client(SetGripperPosition, '/left/gripper/set_position')
    force = node.create_client(SetGripperForce, '/left/gripper/set_force')
    assert position.wait_for_service(timeout_sec=5)
    wait_state(110.)
    assert complete(position.call_async(SetGripperPosition.Request(position=40., speed=.5))).success
    wait_state(40.)
    assert not complete(position.call_async(SetGripperPosition.Request(position=float('nan'), speed=.5))).success
    assert not complete(force.call_async(SetGripperForce.Request(max_force=-1.))).success
    wait_state(40.)
    action = ActionClient(node, DeliGrasp, '/left/gripper/deligrasp')
    assert action.wait_for_server(timeout_sec=5)
    goal = DeliGrasp.Goal()
    goal.params.goal_aperture = 35.
    goal.params.initial_force = 1.
    goal.params.complete_grasp = False
    handle = complete(action.send_goal_async(goal))
    assert handle.accepted
    result = complete(handle.get_result_async())
    assert result.status == 4 and result.result.success and abs(result.result.final_aperture - 35.) < .1
    goal.params.complete_grasp = True
    goal.params.additional_closure = 1.
    handle = complete(action.send_goal_async(goal))
    assert handle.accepted
    result = complete(handle.get_result_async())
    assert result.status == 6 and not result.result.success, 'no-contact grasp must abort'
    assert result.result.final_force == 0.
    print('PASS: native Magpie service ack, invalid commands, approach action, and no-contact abort')
finally:
    node.destroy_node()
    rclpy.shutdown()
