"""Set up the tests: BLAS threads, the ROS domain, and the shared fixtures.

- Robot models: building one loads its URDF meshes (seconds each), so the session shares
  one per robot. A test that changes a shared model's state or re-locks its reduced model
  restores it before it ends.
- ROS: every test that uses ROS runs on domain 92 (never 0, the real robots' command bus),
  through the `ros` fixture (ros_harness.Ros).
- BLAS: the ROS tests publish commands at 500 Hz in this process. OpenBLAS starts one
  busy-waiting worker per core by default, whose spin stalls a 500 Hz publisher thread by
  tens of ms, so its thread counts are pinned here, before any test module imports numpy.
"""
import os

for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_var, '1')
os.environ['ROS_DOMAIN_ID'] = '92'

import pytest  # noqa: E402


@pytest.fixture(scope='session')
def models():
    """Return model(robot): the session's RobotModel of robot, built on first use."""
    from golem_upper_ik.core.robot_model import RobotModel
    from ros_harness import profile_of
    built = {}

    def model(robot):
        """Return robot's shared RobotModel, building it the first time."""
        if robot not in built:
            built[robot] = RobotModel(profile_of(robot))
        return built[robot]
    return model


@pytest.fixture
def ros():
    """Yield a ROS context on domain 92 with an executor spinning its nodes; close it after."""
    from ros_harness import Ros
    harness = Ros()
    try:
        yield harness
    finally:
        harness.close()
