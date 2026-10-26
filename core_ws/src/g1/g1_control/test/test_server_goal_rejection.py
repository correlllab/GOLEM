'''
The action servers abort goals that set fields the G1 controller does not
implement, before the controller or a feedback publisher is touched

The callbacks run as plain functions on a stand-in server, so no ROS graph or
robot model is needed.
'''

import threading

import pytest

from custom_ros_messages.action import FrameTask, NamedConfig

from g1_control.ros2.dual_arm_server import DualArmServer
from g1_control.ros2.frame_task_server import FrameTaskServer
from g1_control.utility.named_config import NAMED_CONFIGS

VALID_CONFIG_NAME = next(iter(NAMED_CONFIGS))


class ControllerReached(Exception):
    pass


class UntouchableController:
    '''Any attribute access means the callback got past the goal check'''

    def __getattr__(self, name):
        raise ControllerReached(name)


class FakeLogger:

    def __init__(self):
        self.records = []

    def info(self, message):
        self.records.append(('info', message))

    def warn(self, message):
        self.records.append(('warn', message))

    def warns(self):
        return [message for level, message in self.records if level == 'warn']


class FakeServer:
    '''The attributes the callbacks read before they use the controller'''

    def __init__(self):
        self.controller = UntouchableController()
        self.logger = FakeLogger()
        self.timeout = 1.0
        self.frame_names = []
        self.frame_targets = []
        self._controller_lock = threading.Lock()

    def get_logger(self):
        return self.logger


class FakeGoalHandle:

    def __init__(self, request):
        self.request = request
        self.aborts = 0

    def abort(self):
        self.aborts += 1

    def publish_feedback(self, feedback):
        raise AssertionError('feedback published')

    def succeed(self):
        raise AssertionError('goal succeeded')

    def canceled(self):
        raise AssertionError('goal cancelled')


def frame_task_goal(**fields):
    return FrameTask.Goal(**fields)


def named_config_goal(**fields):
    return NamedConfig.Goal(config_name=VALID_CONFIG_NAME, **fields)


REJECTED_GOALS = [
    pytest.param(FrameTaskServer.frame_task_callback, FrameTask,
                 frame_task_goal(relative=True), 'relative',
                 id='frame_task-relative'),
    pytest.param(FrameTaskServer.frame_task_callback, FrameTask,
                 frame_task_goal(slow_mode=True), 'slow_mode',
                 id='frame_task-slow_mode'),
    pytest.param(FrameTaskServer.frame_task_callback, FrameTask,
                 frame_task_goal(relative=True, slow_mode=True),
                 'relative, slow_mode', id='frame_task-both'),
    pytest.param(FrameTaskServer.named_config_callback, NamedConfig,
                 named_config_goal(slow_mode=True), 'slow_mode',
                 id='frame_task_server-named_config-slow_mode'),
    pytest.param(DualArmServer.named_config_callback, NamedConfig,
                 named_config_goal(slow_mode=True), 'slow_mode',
                 id='dual_arm_server-named_config-slow_mode'),
]

SUPPORTED_GOALS = [
    pytest.param(FrameTaskServer.frame_task_callback, frame_task_goal(),
                 id='frame_task'),
    pytest.param(FrameTaskServer.named_config_callback, named_config_goal(),
                 id='frame_task_server-named_config'),
    pytest.param(DualArmServer.named_config_callback, named_config_goal(),
                 id='dual_arm_server-named_config'),
]


@pytest.mark.parametrize('callback, action, goal, fields', REJECTED_GOALS)
def test_unsupported_field_aborts_before_the_controller(
        callback, action, goal, fields):
    server = FakeServer()
    goal_handle = FakeGoalHandle(goal)

    result = callback(server, goal_handle)

    assert isinstance(result, action.Result)
    assert result.success is False
    assert goal_handle.aborts == 1
    assert server.logger.warns() == [f'{fields} not supported on the G1']
    assert server.frame_names == []
    assert server.frame_targets == []


@pytest.mark.parametrize('callback, goal', SUPPORTED_GOALS)
def test_supported_goal_reaches_the_controller(callback, goal):
    server = FakeServer()
    goal_handle = FakeGoalHandle(goal)

    with pytest.raises(ControllerReached):
        callback(server, goal_handle)

    assert goal_handle.aborts == 0
    assert server.logger.warns() == []
