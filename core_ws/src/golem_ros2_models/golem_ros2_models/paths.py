'''
Installed locations of each robot's model files (needs the package built and sourced).

Each robot's directory, share/golem_ros2_models/<robot>/, holds:
- the Pinocchio URDF, whose relative mesh paths resolve against that directory;
- its ROS copy (*_ros.urdf), with package://golem_ros2_models/<robot>/meshes/ URIs, for
  robot_state_publisher and RViz;
- the self-collision model: a sphere-packed URDF and the SRDF of the pairs it leaves out.
'''
import os

ROBOTS = ('h12', 'g1')

# robot -> (Pinocchio URDF, sphere URDF, SRDF); the ROS copy is the URDF's name + _ros
_FILES = {
    # the H1-2 with both Magpie grippers
    'h12': ('h1_2_magpie.urdf', 'h1_2_magpie_sphere.urdf',
            'h1_2_magpie_sphere_collision.srdf'),
    # the G1 (29 DoF) with both Dex3-1 hands. Its sphere model was packed on the rubber-hand
    # G1: two spheres per hand at the Dex3-1 palm, none on the fingers
    'g1': ('g1_29dof_with_hand_rev_1_0.urdf', 'g1_sphere.urdf', 'g1_sphere_collision.srdf'),
}


def _files(robot):
    if robot not in _FILES:
        raise ValueError(f'unknown robot {robot!r}; the robots are {", ".join(ROBOTS)}')
    return _FILES[robot]


def model_dir(robot):
    '''share/golem_ros2_models/<robot>, the robot's installed model files'''
    _files(robot)
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('golem_ros2_models'), robot)


def urdf_path(robot, ros=False):
    '''The robot's URDF: for Pinocchio, or with ros=True its package:// copy'''
    name = _files(robot)[0]
    if ros:
        name = name[:-len('.urdf')] + '_ros.urdf'
    return os.path.join(model_dir(robot), name)


def collision_paths(robot):
    '''(sphere URDF, SRDF): the robot's self-collision model'''
    _, sphere, srdf = _files(robot)
    return (os.path.join(model_dir(robot), sphere), os.path.join(model_dir(robot), srdf))


def model_paths(robot):
    '''
    {'urdf_path', 'urdf_ros_path', 'collision'}: the Pinocchio URDF, its ROS copy and
    (sphere URDF, SRDF), as golem_upper_ik's robot profiles name them
    '''
    return {
        'urdf_path': urdf_path(robot),
        'urdf_ros_path': urdf_path(robot, ros=True),
        'collision': collision_paths(robot),
    }
