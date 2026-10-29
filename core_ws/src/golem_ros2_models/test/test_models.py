'''
The installed models of both robots, read through golem_ros2_models.paths (needs the package
built and sourced).
'''
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from golem_ros2_models import paths

PACKAGE_URI = 'package://golem_ros2_models/'
LFS_POINTER = b'version https://git-lfs'
MESH_REF = re.compile(r'<mesh\b[^>]*?\bfilename="([^"]*)"')
# The joints each robot's Pinocchio URDF has beyond its body joints: the Magpie grippers'
# finger hinges and rockers on the H1-2, the Dex3-1 fingers on the G1
NON_BODY_JOINT = {
    'h12': re.compile(r'^(lg|rg)_'),
    'g1': re.compile(r'^(left|right)_hand_'),
}
# the Magpie finger hinges the H1-2's ROS copy fixes
H12_HINGES = {f'{g}_{s}_hinge_{i}' for g in ('lg', 'rg') for s in ('left', 'right')
              for i in (1, 2, 3)}


def share_dir():
    from ament_index_python.packages import get_package_share_directory
    return get_package_share_directory('golem_ros2_models')


def parse(path):
    '''
    The URDF's root element. The H1-2's sphere URDF uses drake: elements without declaring
    the namespace (urdfdom does not need it), so it is declared here when missing.
    '''
    with open(path) as f:
        text = f.read()
    if 'drake:' in text and 'xmlns:drake' not in text:
        text = re.sub(r'<robot\b', '<robot xmlns:drake="http://drake.mit.edu"', text, count=1)
    return ET.fromstring(text)


def installed_urdfs(robot):
    '''(path, is the ROS copy) of every URDF installed for the robot'''
    found = []
    for root, _, files in os.walk(paths.model_dir(robot)):
        found += [(os.path.join(root, name), name.endswith('_ros.urdf'))
                  for name in files if name.endswith('.urdf')]
    return sorted(found)


def assert_mesh_file(path, ref):
    assert os.path.isfile(path), f'{ref} does not resolve ({path})'
    with open(path, 'rb') as f:
        assert not f.read(len(LFS_POINTER)) == LFS_POINTER, f'{path} is a Git-LFS pointer'


def joints(root):
    return {joint.get('name'): joint for joint in root.iter('joint')}


def actuated(root):
    return [joint for joint in root.iter('joint') if joint.get('type') != 'fixed']


def floats(element, tag, attribute, default):
    child = element.find(tag)
    value = default if child is None else child.get(attribute, default)
    return np.array([float(x) for x in value.split()])


def test_robots():
    assert paths.ROBOTS == ('h12', 'g1')


@pytest.mark.parametrize('call', [
    lambda: paths.model_dir('h1'),
    lambda: paths.urdf_path('go2'),
    lambda: paths.collision_paths('G1'),
    lambda: paths.model_paths(''),
])
def test_unknown_robot_is_refused(call):
    with pytest.raises(ValueError, match="h12.*g1"):
        call()


@pytest.mark.parametrize('robot', paths.ROBOTS)
def test_model_paths_exist(robot):
    model = paths.model_paths(robot)
    assert set(model) == {'urdf_path', 'urdf_ros_path', 'collision'}
    assert model['urdf_path'] == paths.urdf_path(robot)
    assert model['urdf_ros_path'] == paths.urdf_path(robot, ros=True)
    assert model['collision'] == paths.collision_paths(robot)
    assert len(model['collision']) == 2
    for path in (model['urdf_path'], model['urdf_ros_path'], *model['collision']):
        assert os.path.isfile(path), path
        assert os.path.dirname(path) == paths.model_dir(robot)
    assert paths.model_dir(robot) == os.path.join(share_dir(), robot)


def test_installed_files():
    '''The installed layout: each robot's models, meshes and licence, the viewer'''
    share = share_dir()
    expected = {
        'h12': {'h1_2_magpie.urdf', 'h1_2_magpie_ros.urdf', 'h1_2_magpie_sphere.urdf',
                'h1_2_magpie_sphere_collision.srdf', 'LICENSE.unitree_ros', 'meshes'},
        'g1': {'g1_29dof_with_hand_rev_1_0.urdf', 'g1_29dof_with_hand_rev_1_0_ros.urdf',
               'g1_sphere.urdf', 'g1_sphere_collision.srdf', 'LICENSE.unitree_ros', 'meshes'},
    }
    for robot, names in expected.items():
        assert set(os.listdir(os.path.join(share, robot))) == names
    assert os.listdir(os.path.join(share, 'h12', 'meshes', 'magpie'))
    assert os.path.isfile(os.path.join(share, 'launch', 'display.launch.py'))
    assert [n for n in os.listdir(os.path.join(share, 'rviz')) if n.endswith('.rviz')]


@pytest.mark.parametrize('robot', paths.ROBOTS)
def test_mesh_references_resolve(robot):
    urdfs = installed_urdfs(robot)
    assert len(urdfs) == 3
    for path, ros in urdfs:
        with open(path) as f:
            refs = MESH_REF.findall(f.read())
        if os.path.basename(path) == os.path.basename(paths.collision_paths(robot)[0]):
            if robot == 'g1':
                assert refs == [], f'{path}: the G1 sphere model has no meshes'
        else:
            assert refs, f'{path} has no mesh references'
        for ref in refs:
            if ros:
                assert ref.startswith(PACKAGE_URI + robot + '/meshes/'), f'{path}: {ref}'
                mesh = os.path.join(share_dir(), ref[len(PACKAGE_URI):])
            else:
                assert '://' not in ref and not os.path.isabs(ref), f'{path}: {ref}'
                mesh = os.path.normpath(os.path.join(os.path.dirname(path), ref))
                assert mesh.startswith(paths.model_dir(robot) + os.sep), f'{path}: {ref}'
            assert_mesh_file(mesh, ref)


@pytest.mark.parametrize('robot', paths.ROBOTS)
def test_ros_copy_matches_pinocchio_copy(robot):
    from xml.parsers import expat
    with open(paths.urdf_path(robot, ros=True)) as f:
        expat.ParserCreate().Parse(f.read(), True)
    plain = parse(paths.urdf_path(robot))
    ros = parse(paths.urdf_path(robot, ros=True))
    assert ros.get('name') == plain.get('name')
    assert ([link.get('name') for link in ros.iter('link')]
            == [link.get('name') for link in plain.iter('link')])
    plain_joints, ros_joints = joints(plain), joints(ros)
    assert list(ros_joints) == list(plain_joints)
    changed = {name for name in plain_joints
               if plain_joints[name].get('type') != ros_joints[name].get('type')}
    if robot == 'h12':
        assert changed == H12_HINGES
        for name in H12_HINGES:
            assert ros_joints[name].get('type') == 'fixed'
            assert plain_joints[name].get('type') != 'fixed'
    else:
        assert changed == set()


@pytest.mark.parametrize('robot', paths.ROBOTS)
def test_pinocchio_loads_the_model_with_meshes(robot):
    pin = pytest.importorskip('pinocchio')
    path = paths.urdf_path(robot)
    model, collision, visual = pin.buildModelsFromUrdf(path, package_dirs=os.path.dirname(path))
    assert model.njoints > 1
    # Pinocchio names a primitive's kind (SPHERE, CYLINDER) where a mesh's path would be
    mesh_paths = [geom.meshPath for geom in visual.geometryObjects
                  if geom.meshPath.lower().endswith('.stl')]
    assert mesh_paths and collision.ngeoms > 0
    for path in mesh_paths:
        assert os.path.isfile(path), path


@pytest.mark.parametrize('robot', paths.ROBOTS)
def test_pinocchio_loads_the_sphere_model_with_its_srdf(robot):
    pin = pytest.importorskip('pinocchio')
    urdf, srdf = paths.collision_paths(robot)
    model, collision, _ = pin.buildModelsFromUrdf(urdf, package_dirs=os.path.dirname(urdf))
    collision.addAllCollisionPairs()
    every_pair = len(collision.collisionPairs)
    pin.removeCollisionPairs(model, collision, srdf)
    assert 0 < len(collision.collisionPairs) < every_pair


@pytest.mark.parametrize('robot', paths.ROBOTS)
def test_sphere_body_joints_match_the_model(robot):
    '''The sphere model's actuated joints are the model's body joints, in order, alike'''
    plain = actuated(parse(paths.urdf_path(robot)))
    sphere = actuated(parse(paths.collision_paths(robot)[0]))
    body = [j for j in plain if not NON_BODY_JOINT[robot].match(j.get('name'))]
    assert [j.get('name') for j in sphere] == [j.get('name') for j in body]
    assert len(sphere) == {'h12': 27, 'g1': 29}[robot]
    for ours, theirs in zip(sphere, body):
        name = ours.get('name')
        assert ours.get('type') == theirs.get('type'), name
        for tag, attribute, default in (('origin', 'xyz', '0 0 0'), ('origin', 'rpy', '0 0 0'),
                                        ('axis', 'xyz', '1 0 0')):
            np.testing.assert_allclose(floats(ours, tag, attribute, default),
                                       floats(theirs, tag, attribute, default),
                                       atol=1e-12, err_msg=f'{name} {tag} {attribute}')
