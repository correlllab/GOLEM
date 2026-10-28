'''
What the nodes share: pose conversions, their robot and config parameters, and run(), which
spins a node until SIGINT or SIGTERM and shuts it down while its context still works.
'''
import os
import signal
import sys
import threading
import warnings

from geometry_msgs.msg import Pose
import numpy as np
from pyquaternion import Quaternion
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.logging import get_logger
from rclpy.signals import SignalHandlerOptions

from golem_upper_ik.robots import ROBOTS, robot_module

__all__ = ['matrix_to_pose', 'node_parameters', 'pose_to_matrix', 'run']

# s to wait at exit for the callbacks still running (a goal sees the shut-down controller
# refuse its next command and ends within a control step), then for the executor's thread.
# The zero-gain command goes out first; after it the controller's planner may take up to 3 s
# (PlannerClient's 2 s grace for a plan in progress, then 1 s to terminate), so the whole
# shutdown fits within launch's default sigterm_timeout of 5 s
EXIT_CALLBACK_TIMEOUT = 1.0
EXIT_JOIN_TIMEOUT = 0.5


def _project_to_so3(rotation):
    u, _, vh = np.linalg.svd(rotation)
    projected = u @ vh
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1
        projected = u @ vh
    return projected


def pose_to_matrix(pose):
    '''A geometry_msgs Pose as a 4x4 transformation'''
    position = pose.position
    orientation = pose.orientation
    rotation = Quaternion(
        [orientation.w, orientation.x, orientation.y, orientation.z]).rotation_matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [position.x, position.y, position.z]
    return matrix


def matrix_to_pose(matrix):
    '''
    A 4x4 transformation as a geometry_msgs Pose. A non-finite translation becomes 0, a
    non-finite rotation the identity (with a warning), and a rotation that is not quite
    orthonormal is projected onto SO(3)
    '''
    pose = Pose()
    translation = np.asarray(matrix[0:3, 3], dtype=float)
    if not np.isfinite(translation).all():
        translation = np.zeros(3)
    pose.position.x = translation[0]
    pose.position.y = translation[1]
    pose.position.z = translation[2]
    rotation_matrix = np.asarray(matrix[:3, :3], dtype=float)

    if not np.isfinite(rotation_matrix).all():
        warnings.warn(
            'matrix_to_pose: rotation matrix contains non-finite values; '
            'falling back to identity rotation. This may indicate an upstream issue.',
            RuntimeWarning,
            stacklevel=2,
        )
        rotation_matrix = np.eye(3)
    else:
        orthogonality_error = np.linalg.norm(rotation_matrix.T @ rotation_matrix - np.eye(3),
                                             ord='fro')
        determinant = np.linalg.det(rotation_matrix)
        if orthogonality_error > 1e-6 or not np.isfinite(determinant) or determinant <= 0:
            rotation_matrix = _project_to_so3(rotation_matrix)

    rotation = Quaternion(matrix=rotation_matrix)
    pose.orientation.w = rotation.w
    pose.orientation.x = rotation.x
    pose.orientation.y = rotation.y
    pose.orientation.z = rotation.z
    return pose


def node_parameters(node):
    '''
    (robot, config) from node's parameters: robot is required; config is a controller config
    name or path, by default the robot's DEFAULT_CONFIG. ValueError for a robot missing or
    unknown
    '''
    robot = node.declare_parameter('robot', '').value
    if not robot:
        raise ValueError('the robot parameter is required: '
                         + ' or '.join(f'robot:={name}' for name in ROBOTS))
    module = robot_module(robot)
    config = node.declare_parameter('config', '').value or module.DEFAULT_CONFIG
    return robot, config


def _ignored(signum, frame):
    pass


def _ignore_signals():
    '''
    Ignore SIGINT and SIGTERM from now on, through a handler that does nothing: with SIG_IGN,
    a signal already pending when the handler changes raises OSError ("ignored due to race
    condition") in the main thread
    '''
    signal.signal(signal.SIGINT, _ignored)
    signal.signal(signal.SIGTERM, _ignored)


def _interrupt(signum, frame):
    '''The first SIGINT or SIGTERM: stop the node; any later one is ignored'''
    _ignore_signals()
    raise KeyboardInterrupt


def _spin(executor, errors):
    try:
        executor.spin()
    except Exception as error:  # run() reports it and exits 1
        errors.append(error)


def run(name, make_node, args=None):
    '''
    Run the node make_node() builds until SIGINT or SIGTERM, then end the process with the
    exit code.

    make_node() reads the node's parameters and starts nothing. An executor then spins the
    node in a thread, node.start() starts it in this one, and on the signal node.shutdown()
    runs while the executor still spins. A node that fails to build or start is logged
    (naming its robot and config once it knows them) and exits 1, as does one whose executor
    stops.

    rclpy's own signal handlers stay off: on SIGINT they shut the context down, after which
    the controller's last command, which turns the upper body's gains to zero, could not be
    published. SIGTERM, which launch sends after SIGINT, interrupts the same way. Once the
    shutdown has begun, both are ignored, so a second signal cannot cut it short.

    Nothing is destroyed at the end. The multi-threaded executor can start a callback it
    handed to its thread pool just before its shutdown() returned; destroying the node under
    that callback, or the interpreter's exit while a pool thread is inside rclpy, aborts the
    process. So once the executor has stopped and the context is shut down (which also
    flushes the ROS log files), the process ends with os._exit.
    '''
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    logger = get_logger(name)
    executor = MultiThreadedExecutor()
    spin_errors = []
    spinner = threading.Thread(target=_spin, args=(executor, spin_errors), name='executor',
                               daemon=True)
    node = None
    code = 0
    try:
        node = make_node()
        executor.add_node(node)
        spinner.start()
        node.start()
        while spinner.is_alive():
            spinner.join(0.5)
        logger.error(f'{name}: the executor stopped: {spin_errors!r}')
        code = 1
    except KeyboardInterrupt:
        pass
    except Exception as error:
        where = '' if node is None else f' (robot {node.robot}, config {node.config_name})'
        logger.error(f'{name}{where} did not start: {error}')
        code = 1
    finally:
        _ignore_signals()
        if node is not None:
            try:
                node.shutdown()
            except Exception as error:
                logger.error(f'{name}: shutdown failed: {error!r}')
                code = 1
        executor.shutdown(timeout_sec=EXIT_CALLBACK_TIMEOUT)
        if spinner.is_alive():
            spinner.join(EXIT_JOIN_TIMEOUT)
        rclpy.try_shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
