'''
setup.py's refusals, run on a copy of the package beside a copy of CL_Assets at the same
relative path (../../../CL_Assets), whose files link to the real ones unless a test breaks one.
'''
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
ASSETS = PACKAGE.parents[2] / 'CL_Assets'
# the CL_Assets directories setup.py reads
ASSET_DIRS = ('ros_assets', 'meshes/h1_2', 'meshes/magpie', 'g1')


@pytest.fixture
def tree(tmp_path):
    '''(package copy, CL_Assets copy)'''
    if not ASSETS.is_dir():
        pytest.skip(f'no CL_Assets at {ASSETS}')
    for directory in ASSET_DIRS:
        for root, _, files in os.walk(ASSETS / directory):
            target = tmp_path / 'CL_Assets' / Path(root).relative_to(ASSETS)
            target.mkdir(parents=True, exist_ok=True)
            for name in files:
                (target / name).symlink_to(Path(root) / name)
    package = tmp_path / 'core_ws' / 'src' / 'golem_ros2_models'
    package.mkdir(parents=True)
    for name in ('setup.py', 'setup.cfg', 'package.xml'):
        shutil.copy(PACKAGE / name, package / name)
    for name in ('resource', 'launch', 'rviz', 'licenses', 'golem_ros2_models'):
        shutil.copytree(PACKAGE / name, package / name,
                        ignore=shutil.ignore_patterns('__pycache__'))
    return package, tmp_path / 'CL_Assets'


def run_setup(package):
    return subprocess.run([sys.executable, 'setup.py', '--name'], cwd=package,
                          capture_output=True, text=True)


def replace(path, text):
    '''Replace a linked file by a real one holding text'''
    path.unlink()
    path.write_text(text)


def edit(path, old, new, count=1):
    text = path.read_text()
    assert old in text, f'{old!r} not in {path}'
    replace(path, text.replace(old, new, count))


def assert_refused(result, *fragments):
    assert result.returncode != 0, result.stdout + result.stderr
    for fragment in fragments:
        assert fragment in result.stderr, result.stderr


def test_intact_tree_builds(tree):
    package, _ = tree
    result = run_setup(package)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == 'golem_ros2_models'
    staged = package / '_install_staging'
    assert sorted(p.relative_to(staged).as_posix() for p in staged.rglob('*.urdf')) == [
        'g1/g1_29dof_with_hand_rev_1_0_ros.urdf',
        'h12/h1_2_magpie.urdf', 'h12/h1_2_magpie_ros.urdf', 'h12/h1_2_magpie_sphere.urdf',
    ]


def test_staging_is_cleared_on_each_build(tree):
    package, _ = tree
    stale = package / '_install_staging' / 'h12' / 'stale.urdf'
    stale.parent.mkdir(parents=True)
    stale.write_text('<robot name="stale"/>')
    assert run_setup(package).returncode == 0
    assert not stale.exists()


def test_missing_cl_assets_is_refused(tree):
    package, assets = tree
    shutil.rmtree(assets)
    assert_refused(run_setup(package), 'CL_Assets', str(assets))


@pytest.mark.parametrize('missing', [
    'ros_assets/h1_2_magpie_sphere_collision.srdf',
    'ros_assets/gen_ros_urdf.py',
    'g1/g1_sphere.urdf',
    'g1/LICENSE.unitree_ros',
    'g1/g1_description/g1_29dof_with_hand_rev_1_0.urdf',
])
def test_missing_file_is_refused(tree, missing):
    package, assets = tree
    (assets / missing).unlink()
    assert_refused(run_setup(package), missing)


@pytest.mark.parametrize('mesh', [
    'meshes/h1_2/pelvis.STL',
    'meshes/magpie/base_bot.stl',
    'g1/g1_description/meshes/pelvis.STL',
])
def test_missing_mesh_is_refused(tree, mesh):
    package, assets = tree
    (assets / mesh).unlink()
    assert_refused(run_setup(package), os.path.basename(mesh))


@pytest.mark.parametrize('pointer', [
    'meshes/h1_2/pelvis.STL',
    'meshes/magpie/base_bot.stl',
    'g1/g1_description/meshes/pelvis.STL',
    'g1/g1_sphere.urdf',
])
def test_lfs_pointer_is_refused(tree, pointer):
    package, assets = tree
    replace(assets / pointer, 'version https://git-lfs.github.com/spec/v1\n'
            'oid sha256:0000\nsize 1\n')
    assert_refused(run_setup(package), os.path.basename(pointer), 'git lfs pull')


def test_unexpected_h12_mesh_path_is_refused(tree):
    package, assets = tree
    edit(assets / 'ros_assets/h1_2_magpie.urdf', 'filename="../meshes/h1_2/pelvis.STL"',
         'filename="../meshes/pelvis.STL"')
    assert_refused(run_setup(package), 'h1_2_magpie.urdf', 'mesh')


def test_unexpected_g1_mesh_path_is_refused(tree):
    package, assets = tree
    edit(assets / 'g1/g1_description/g1_29dof_with_hand_rev_1_0.urdf',
         'filename="meshes/pelvis.STL"', 'filename="package://g1_description/pelvis.STL"')
    assert_refused(run_setup(package), 'g1_29dof_with_hand_rev_1_0.urdf', 'mesh')


def test_mesh_in_the_g1_sphere_model_is_refused(tree):
    package, assets = tree
    edit(assets / 'g1/g1_sphere.urdf', '<sphere radius=',
         '<mesh filename="meshes/pelvis.STL"/><sphere radius=')
    assert_refused(run_setup(package), 'g1_sphere.urdf', 'mesh')
