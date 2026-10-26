import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'g1_description'

# The G1 URDFs and meshes come from the leprechaun_g1 submodule in the repo's
# third_party/, through the symlink leprechaun_g1 in this package
# (../../../../third_party/leprechaun_g1), so this package must be built from
# <repo>/core_ws/src/g1/g1_description (in docker, the ros service mounts the
# submodule at /home/code/third_party/leprechaun_g1). The data files must stay
# inside the package: with --symlink-install colcon links each one into the
# build directory at the same relative path, and the build directory,
# core_ws/build/g1_description, is one level shallower than this one.
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_REL = os.path.join('leprechaun_g1', 'assets', 'g1_description')
ASSETS_ABS = os.path.normpath(os.path.join(HERE, ASSETS_REL))
URDFS = ['g1_29dof_rev_1_0.urdf', 'g1_29dof_with_hand_rev_1_0.urdf']

if not os.path.isdir(ASSETS_ABS):
    raise RuntimeError(
        f'g1_description expected the leprechaun_g1 submodule at {ASSETS_ABS}; '
        'run `git submodule update --init third_party/leprechaun_g1` at the repo root.')


def _rel(abs_path):
    # colcon's ament_python builder wants data_files sources relative to setup.py.
    return os.path.relpath(abs_path, HERE)


meshes = [_rel(p) for p in sorted(glob(os.path.join(ASSETS_ABS, 'meshes', '*.STL')))]
if not meshes:
    raise RuntimeError(f'no G1 meshes found under {ASSETS_ABS}/meshes/')

# Upstream URDFs reference meshes as 'meshes/<name>.STL'. Installed beside
# share/g1_description/assets/meshes/ they resolve unchanged, so they are
# installed byte-for-byte. robot_state_publisher and RViz need package:// URIs,
# so a *_ros.urdf copy with rewritten mesh paths is staged next to them.
STAGE_DIR = os.path.join(HERE, '_install_staging')
os.makedirs(STAGE_DIR, exist_ok=True)
for stale in os.listdir(STAGE_DIR):
    os.remove(os.path.join(STAGE_DIR, stale))

urdfs = []
for name in URDFS:
    src = os.path.join(ASSETS_ABS, name)
    urdfs.append(_rel(src))
    with open(src) as f:
        text = f.read()
    n_meshes = text.count('filename="meshes/')
    text = text.replace('filename="meshes/', f'filename="package://{package_name}/assets/meshes/')
    if n_meshes == 0 or text.count(f'package://{package_name}/') != n_meshes:
        raise RuntimeError(f'unexpected mesh paths in {src}')
    dst = os.path.join(STAGE_DIR, name.replace('.urdf', '_ros.urdf'))
    with open(dst, 'w') as f:
        f.write(text)
    urdfs.append(_rel(dst))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test', 'tools']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/assets',
         urdfs + [_rel(os.path.join(ASSETS_ABS, '..', 'LICENSE.unitree_ros'))]),
        ('share/' + package_name + '/assets/meshes', meshes),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Niraj Pudasaini',
    maintainer_email='npudasai@nd.edu',
    description='Unitree G1 29-DoF + Dex3-1 URDFs, meshes, joint table and DDS motor layout',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
)
