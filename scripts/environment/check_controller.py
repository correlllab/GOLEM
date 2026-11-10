"""Qualify imports and small math operations only, never controller construction."""

import importlib
from pathlib import Path
import site
import sys

import numpy as np
import pinocchio as pin
import qpsolvers

ROOT = Path(__file__).resolve().parents[2]
assert Path(sys.prefix) == ROOT / 'environments/controller/.venv'
assert sys.version_info[:2] == (3, 10) and sys.base_prefix == '/usr'
assert site.ENABLE_USER_SITE is False
assert np.__version__ == '2.2.6'
assert pin.__version__ == '3.9.0'
for name in ('pink', 'ompl.base', 'ompl.geometric', 'crocoddyl', 'rclpy',
             'h12_ros2_controller.core.controller.counter_frame_controller',
             'h12_ros2_controller.ros2.frame_task_server',
             'h12_ros2_controller.ros2.joint_state_publisher'):
    module = importlib.import_module(name)
    print(f'PASS controller import: {name} ({module.__file__})')
import unitree_sdk2py
assert Path(unitree_sdk2py.__file__).resolve().is_relative_to(ROOT / 'unitree_sdk2_python')
assert 'cv_bridge' not in sys.modules and 'cv2' not in sys.modules
np.testing.assert_allclose(pin.exp3(np.zeros(3)), np.eye(3))
solution = qpsolvers.solve_qp(np.eye(2), np.array([-1., -2.]), solver='quadprog')
np.testing.assert_allclose(solution, [1., 2.])
from h12_ros2_controller.utility.path_definition_ros import URDF_MAGPIE_PIN_PATH
model = pin.buildModelFromUrdf(URDF_MAGPIE_PIN_PATH)
data = model.createData()
pin.framesForwardKinematics(model, data, pin.neutral(model))
assert all(np.isfinite(frame.translation).all() for frame in data.oMf)
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import JointState
message = JointState(name=['offline'], position=[0.25])
np.testing.assert_array_equal(deserialize_message(serialize_message(message), JointState).position, [0.25])
print(f'PASS installed URDF offline FK (nq={model.nq}) and JointState CDR in NumPy 2')
print('PASS frame/counter native imports, Pinocchio rotation and QP; no ROS init or controller construction')
print('NOT checked by this command: live planning, actuation or balance performance')
