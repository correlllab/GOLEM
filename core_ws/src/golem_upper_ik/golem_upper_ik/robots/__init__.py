'''
The controller's data for each robot, one module per robot (h12.py, g1.py): IK joints, held
joints, frames, model files, named configurations and the controller-config defaults where
the robots differ. The joint table itself is golem_safety's robots/<robot>.yaml.

A module imports nothing of another robot's packages at module level: the H1-2 builds and
runs without the g1_* packages.
'''
import importlib

ROBOTS = ('h12', 'g1')


def robot_module(robot: str):
    '''The data module of robot; ValueError for a robot without one'''
    if robot not in ROBOTS:
        raise ValueError(f"unknown robot '{robot}': one of {', '.join(ROBOTS)}")
    return importlib.import_module(f'{__name__}.{robot}')
