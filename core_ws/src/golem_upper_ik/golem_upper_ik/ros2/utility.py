"""Share what the nodes need: pose conversions, their parameters, and run().

run() spins a node until SIGINT or SIGTERM and shuts it down while its context still works.
"""
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

from golem_upper_ik.profile import known_robots

__all__ = ['matrix_to_pose', 'pose_to_matrix', 'robot_parameter', 'run']

# s to wait at exit for the callbacks still running (a goal sees the shut-down controller
# refuse its next command and ends within a control step), then for the executor's thread.
# The controller's shutdown command goes out first; after it the planner may take
# planner_process.SHUTDOWN_GRACE for a plan in progress and its terminate wait
# (PlannerClient.shutdown). All of it must fit within launch's default sigterm_timeout of 5 s
EXIT_CALLBACK_TIMEOUT = 1.0
EXIT_JOIN_TIMEOUT = 0.5


def _project_to_so3(rotation):
    """Return the rotation matrix nearest a 3x3 matrix (SVD, determinant +1)."""
    u, _, vh = np.linalg.svd(rotation)
    projected = u @ vh
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1
        projected = u @ vh
    return projected


def pose_to_matrix(pose):
    """Return a geometry_msgs Pose as a 4x4 transformation."""
    position = pose.position
    orientation = pose.orientation
    rotation = Quaternion(
        [orientation.w, orientation.x, orientation.y, orientation.z]).rotation_matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [position.x, position.y, position.z]
    return matrix


def matrix_to_pose(matrix):
    """Return a 4x4 transformation as a geometry_msgs Pose.

    A non-finite translation becomes 0, a non-finite rotation the identity (with a
    RuntimeWarning), and a rotation that is not quite orthonormal is projected onto SO(3).
    """
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


def robot_parameter(node):
    """Declare and return node's robot parameter; ValueError when it is missing.

    An unknown robot is refused where its profile is loaded (profile.load_profile).
    """
    robot = node.declare_parameter('robot', '').value
    if not robot:
        raise ValueError('the robot parameter is required: '
                         + ' or '.join(f'robot:={name}' for name in known_robots()))
    return robot


def _ignored(signum, frame):
    """Ignore a signal."""


def _ignore_signals():
    """Ignore SIGINT and SIGTERM from now on, through a handler that does nothing.

    With SIG_IGN instead, a signal that arrives while the handler changes is reported through
    sys.unraisablehook as OSError("Signal 2 ignored due to race condition"), a stray
    traceback on stderr.
    """
    signal.signal(signal.SIGINT, _ignored)
    signal.signal(signal.SIGTERM, _ignored)


def _interrupt(signum, frame):
    """Stop the node on the first SIGINT or SIGTERM; any later one is ignored."""
    _ignore_signals()
    raise KeyboardInterrupt


def _spin(executor, errors):
    """Spin executor until it stops; an error it raises is appended to errors."""
    try:
        executor.spin()
    except Exception as error:  # run() reports it and exits 1
        errors.append(error)


def run(name, make_node, args=None):
    """Run the node make_node() builds until SIGINT or SIGTERM, then exit the process.

    make_node() reads the node's parameters and starts nothing. An executor then spins the
    node in a thread, node.start() starts it in this one, and on the signal node.shutdown()
    runs while the executor still spins. A node that fails to build or start is logged
    (with its node.summary, its robot and settings, once built) and exits 1, as does one whose
    executor stops; a clean shutdown exits 0.

    SAFETY: rclpy's own signal handlers stay off: on SIGINT they shut the context down, after
    which the controller's last command, which turns the upper body's gains to zero, could
    not be published. SIGTERM, which launch sends after SIGINT, interrupts the same way. Once
    the shutdown has begun, both are ignored, so a second signal cannot cut it short.

    Nothing is destroyed at the end. The multi-threaded executor can start a callback it
    handed to its thread pool just before its shutdown() returned; destroying the node under
    that callback, or the interpreter's exit while a pool thread is inside rclpy, aborts the
    process. So once the executor has stopped and the context is shut down (which also
    flushes the ROS log files), the process ends with os._exit.
    """
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
        where = '' if node is None else f' ({node.summary})'
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
