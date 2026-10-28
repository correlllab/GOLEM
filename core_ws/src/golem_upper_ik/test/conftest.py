'''
Robot models shared by the tests: building one loads its URDF meshes (seconds each). A test
that changes a shared model's state or re-locks its reduced model restores it before it
ends.
'''
import pytest


def profile_of(robot):
    if robot == 'g1':
        pytest.importorskip('g1_description')
    from golem_upper_ik.profile import load_profile
    return load_profile(robot)


@pytest.fixture(scope='session')
def models():
    '''robot -> RobotModel, built on first use'''
    from golem_upper_ik.core.robot_model import RobotModel
    built = {}

    def model(robot):
        if robot not in built:
            built[robot] = RobotModel(profile_of(robot))
        return built[robot]
    return model
