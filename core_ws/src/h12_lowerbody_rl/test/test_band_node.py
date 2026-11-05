"""
The H1-2 leg node's elastic-band release, in one process on domain 97.

A fake /elastic_band/toggle (remapped to a name of the test's own) stands in
for the sim's. The node is past its start-up: the policy is engaged and
/left_ee_pose has come, so the release is due.
"""

import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import Trigger

from h12_lowerbody_rl.scripts.lowerbody_controller_node import LowerBodyControllerNode

TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)
# other packages' tests on domain 97 serve /elastic_band/toggle too
TOGGLE_REMAP = '/elastic_band/toggle:=/test_h12_band_node/toggle'


@pytest.fixture
def rig():
    rclpy.init(args=['--ros-args', '-r', TOGGLE_REMAP], domain_id=TEST_DOMAIN)
    node = LowerBodyControllerNode()
    helper = rclpy.create_node('test_h12_band_helper')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    log = []
    node._log = lambda level, message: log.append((level, message))
    # the policy is engaged, frame_task_server is up: the release is due
    node._band_due = True
    node._frame_task_ready = True
    try:
        yield node, helper, executor, log
    finally:
        executor.shutdown()
        helper.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


def _spin(node, executor, seconds, done=lambda: False):
    """Spin, asking the node for its band release every turn, until done() or `seconds`."""
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        node._update_band(time.monotonic())
        executor.spin_once(timeout_sec=0.02)
    return done()


def _toggle(calls, success, message):
    def serve(request, response):
        calls.append(time.monotonic())
        response.success, response.message = success, message
        return response
    return serve


def test_one_call_once_the_service_is_there(rig):
    node, helper, executor, log = rig
    calls = []
    # due, but no service: no call, one warning
    _spin(node, executor, 1.0)
    assert not node._band_called
    assert len([m for lvl, m in log if lvl == 'warn' and 'not there yet' in m]) == 1, log
    helper.create_service(Trigger, '/elastic_band/toggle', _toggle(calls, True, 'disabled'))
    assert _spin(node, executor, 5.0, lambda: node._band_released)
    _spin(node, executor, 1.0)
    assert len(calls) == 1, calls
    # the policy's history restarts on the next tick, not when the call was sent
    assert node._reset_due


def test_a_failed_release_is_not_retried(rig):
    node, helper, executor, log = rig
    calls = []
    helper.create_service(Trigger, '/elastic_band/toggle', _toggle(calls, False, 'nope'))
    assert _spin(node, executor, 5.0, lambda: any(lvl == 'error' for lvl, _ in log))
    _spin(node, executor, 1.0)
    assert len(calls) == 1, calls
    assert not node._band_released and not node._reset_due
    errors = [m for lvl, m in log if lvl == 'error']
    assert len(errors) == 1 and errors[0].startswith('elastic band release failed: ')
    assert '"nope"' in errors[0]


def test_a_toggle_that_turned_the_band_on_is_not_a_release(rig):
    node, helper, executor, log = rig
    calls = []
    helper.create_service(Trigger, '/elastic_band/toggle', _toggle(calls, True, 'enabled'))
    assert _spin(node, executor, 5.0, lambda: any(lvl == 'warn' and 'ON' in m for lvl, m in log))
    _spin(node, executor, 1.0)
    assert len(calls) == 1, calls
    assert not node._band_released and not node._reset_due
    assert not [m for lvl, m in log if lvl == 'error']
